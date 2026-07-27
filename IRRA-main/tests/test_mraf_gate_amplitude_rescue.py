import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from model.mraf import ModalityReliabilityAdaptiveFusion
from processor.processor import (
    CHECKPOINT_SELECTION_FIELDS,
    MRAF_EPOCH_METRIC_FIELDS,
    _DetachedPairedStats,
    _DetachedRunningStats,
    _append_tsv_row,
    _mraf_initial_parameter_snapshot,
    _mraf_scorer_parameter_stats,
    _save_last_checkpoint,
    _validate_tsv_schema,
)
from solver.build import (
    OPTIMIZER_PARAM_GROUP_FIELDS,
    build_lr_scheduler,
    build_optimizer,
    get_mraf_scorer_named_parameters,
    optimizer_param_group_rows,
)
from tools.revision import diagnose_mraf_mechanism as diagnosis
from tools.revision import summarize_mraf_gate_amplitude_rescue as rescue
from utils.options import validate_mraf_options
from utils.options import get_args


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "mraf" / "run_mraf_gate_amplitude_rescue_pipeline.sh"


def _load_test_module():
    spec = importlib.util.spec_from_file_location("irra_test_entry", ROOT / "test.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ToyModel(nn.Module):
    def __init__(self, mraf=True):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.mraf_enabled = mraf
        if mraf:
            self.mraf_fusion = SimpleNamespace(
                scorer=nn.Sequential(
                    nn.Linear(4, 3),
                    nn.GELU(),
                    nn.Linear(3, 1),
                )
            )
            self.add_module("mraf_scorer", self.mraf_fusion.scorer)


def _optimizer_args(tmp_path, multiplier=1.0):
    return SimpleNamespace(
        lr=1e-5,
        lr_factor=5.0,
        bias_lr_factor=2.0,
        weight_decay=4e-5,
        weight_decay_bias=0.0,
        optimizer="Adam",
        alpha=0.9,
        beta=0.999,
        mraf_scorer_lr_multiplier=multiplier,
        output_dir=str(tmp_path),
        milestones=[20, 50],
        gamma=0.1,
        warmup_factor=1.0 / 3.0,
        warmup_epochs=5,
        warmup_method="linear",
        num_epoch=60,
        lrscheduler="cosine",
        target_lr=0.0,
        power=0.9,
    )


def _validation_args(**updates):
    values = {
        "mraf_hidden_dim": 8,
        "mraf_dropout": 0.0,
        "mraf_gate_temperature": 1.0,
        "mraf_gate_logit_scale": 1.0,
        "mraf_scorer_lr_multiplier": 1.0,
        "mraf_target_temperature": 0.1,
        "mraf_rel_lambda": 0.01,
        "mraf_rel_warmup_epoch": 3,
        "mraf_balance_lambda": 0.0,
        "mraf_gate_base_grad_scale": 1.0,
        "mraf_enabled": True,
        "use_global_uncertainty": False,
        "use_mum": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _passing_unit():
    metrics = {
        scenario: {"R1": 49.9 if scenario == "clean" else 42.0}
        for scenario in rescue.SCREEN_SCENARIOS
    }
    return {
        "candidate": "A50",
        "checkpoint": "last",
        "complete": True,
        "has_nan": False,
        "has_inf": False,
        "has_oom": False,
        "metrics": metrics,
        "mechanism": {
            "mraf_scorer_lr_multiplier": 50.0,
            "mraf_gate_logit_scale": 1.0,
            "alpha_std": 0.006,
            "alpha_target_pearson": 0.5,
            "rgb_missing_alpha_rgb": 0.3,
            "ir_missing_alpha_ir": 0.3,
            "rgb_noise_alpha_drop": 0.03,
            "ir_noise_alpha_drop": 0.03,
            "rgb_blur_alpha_drop": 0.03,
            "ir_blur_alpha_drop": 0.03,
            "weighted_rel_base_ratio": 0.01,
            "scorer_parameter_updated": True,
            "fusion_path_sensitive": True,
        },
    }


def _reference_tsv(
    root,
    relative="screen/rescue_screen_summary.tsv",
    *,
    clean_field="clean_R1",
    degraded_field="key_degraded_mean_R1",
    clean="49.8795",
    degraded="41.7958",
    bom=False,
    candidate="S4",
):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\ufeff" if bom else ""
    path.write_text(
        f"{prefix}candidate\t{clean_field}\t{degraded_field}\n"
        f"{candidate}\t{clean}\t{degraded}\n",
        encoding="utf-8",
    )
    return path


def _legacy_epoch_text():
    original = []
    for field in MRAF_EPOCH_METRIC_FIELDS:
        if field == "target_rgb_mean":
            original.append("target_rgb_meantarget_rgb_std")
        elif field == "target_rgb_std":
            continue
        elif field == "scorer_first_weight_norm":
            original.append(
                "scorer_first_weight_normscorer_last_weight_norm"
            )
        elif field == "scorer_last_weight_norm":
            continue
        else:
            original.append(field)
    values = [str(index) for index in range(len(MRAF_EPOCH_METRIC_FIELDS))]
    return "\t".join(original) + "\n" + "\t".join(values) + "\n"


def _write_resume_fixture(tmp_path, *, manifest_path_style="native"):
    run_root = tmp_path / "existing_run"
    reference_root = tmp_path / "reference"
    data_root = tmp_path / "data"
    base_config = tmp_path / "base_configs.yaml"
    run_root.mkdir()
    data_root.mkdir()
    _reference_tsv(reference_root)
    base_config.write_text("placeholder: true\n", encoding="utf-8")

    def manifest_path(path):
        if manifest_path_style == "native":
            return str(path.resolve())
        drive = path.drive.rstrip(":").lower()
        suffix = path.resolve().as_posix().split(":/", 1)[-1]
        return f"/mnt/{drive}/{suffix}"

    manifest = {
        "PIPELINE_ID": "mraf_gate_amplitude_rescue_v3",
        "git_branch": "experiment/mraf-gate-amplitude-rescue-v3",
        "git_head": rescue.EXPECTED_ORIGINAL_COMMIT,
        "seed": "1",
        "degrade_seed": "2026",
        "device": "0",
        "data_root": manifest_path(data_root),
        "base_config": manifest_path(base_config),
        "reference_screen_root": manifest_path(reference_root),
        "reference_run_root": manifest_path(tmp_path / "old_run"),
        "auto_continue": "true",
        "shutdown_on_success": "false",
        "mock": "false",
    }
    (tmp_path / "old_run").mkdir()
    (run_root / "experiment_manifest.txt").write_bytes(
        "".join(f"{key}={value}\n" for key, value in manifest.items()).encode(
            "utf-8"
        )
    )
    (run_root / "status.txt").write_text(
        "STATUS=FAILED\nFAILED_STAGE=amplitude_screen\nEXIT_CODE=1\n",
        encoding="utf-8",
    )
    status_rows = ["candidate\tcheckpoint\tstatus\texit_code\tfailed_stage"]
    for candidate, matrix in rescue.STAGE1_MATRIX.items():
        train_dir = run_root / "stage1" / "train" / candidate
        train_dir.mkdir(parents=True)
        for name in ("configs.yaml", "best.pth", "last.pth"):
            (train_dir / name).write_text("complete\n", encoding="utf-8")
        for checkpoint in rescue.CHECKPOINT_KINDS:
            status_rows.append(f"{candidate}\t{checkpoint}\tCOMPLETED\t0\tNONE")
            unit = run_root / "stage1" / "units" / f"{candidate}_{checkpoint}"
            diagnostic = unit / "diagnostics"
            diagnostic.mkdir(parents=True)
            config = {
                **rescue.RESUME_PROTOCOL,
                **matrix,
            }
            (unit / "configs.yaml").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (unit / "checkpoint.pth").write_text(
                "checkpoint\n", encoding="utf-8"
            )
            (unit / "train.log").write_text("complete\n", encoding="utf-8")
            (unit / "mraf_epoch_metrics.tsv").write_text(
                _legacy_epoch_text(), encoding="utf-8"
            )
            (unit / "checkpoint_selection.tsv").write_text(
                "checkpoint\tepoch\tclean_R1\talpha_std\tPearson\t"
                "weighted_rel_base_ratio\n"
                f"{checkpoint}.pth\t1\t49.9\t0.006\t0.5\t0.01\n",
                encoding="utf-8",
            )
            (unit / "unit.json").write_text(
                json.dumps(
                    {
                        "candidate": candidate,
                        "checkpoint": checkpoint,
                        **matrix,
                        "has_nan": False,
                        "has_inf": False,
                        "has_oom": False,
                    }
                ),
                encoding="utf-8",
            )
            for scenario in rescue.SCREEN_SCENARIOS:
                score = unit / "eval" / scenario / "scores.json"
                score.parent.mkdir(parents=True)
                score.write_text(
                    json.dumps(
                        {
                            "metrics": {
                                "R1": 49.9 if scenario == "clean" else 42.0,
                                "R5": 70.0,
                                "R10": 80.0,
                                "mAP": 45.0,
                                "mINP": 30.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            diagnosis_payload = {
                "cases": {
                    candidate: {
                        "alpha_rgb_std": 0.006,
                        "alpha_target_pearson_degraded_scenarios": 0.5,
                        "weighted_rel_base_ratio_median": 0.01,
                        "fusion_path_sensitive": True,
                        "alarms": {"PARAMETER_NOT_UPDATED": False},
                    }
                }
            }
            (diagnostic / "diagnosis.json").write_text(
                json.dumps(diagnosis_payload), encoding="utf-8"
            )
            with (diagnostic / "scenario_gate_statistics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "scenario",
                        "alpha_rgb_mean",
                        "alpha_ir_mean",
                        "logit_gap_std",
                        "scaled_logit_gap_std",
                        "raw_logits_dtype",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "scenario": "clean",
                            "alpha_rgb_mean": 0.5,
                            "alpha_ir_mean": 0.5,
                            "logit_gap_std": 0.02,
                            "scaled_logit_gap_std": 0.02,
                            "raw_logits_dtype": "torch.float32",
                        },
                        {
                            "scenario": "rgb_missing",
                            "alpha_rgb_mean": 0.3,
                            "alpha_ir_mean": 0.7,
                            "logit_gap_std": 0.02,
                            "scaled_logit_gap_std": 0.02,
                            "raw_logits_dtype": "torch.float32",
                        },
                        {
                            "scenario": "ir_missing",
                            "alpha_rgb_mean": 0.7,
                            "alpha_ir_mean": 0.3,
                            "logit_gap_std": 0.02,
                            "scaled_logit_gap_std": 0.02,
                            "raw_logits_dtype": "torch.float32",
                        },
                    ]
                )
            with (diagnostic / "gate_direction_statistics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("scenario", "degraded_alpha_delta_mean"),
                )
                writer.writeheader()
                for scenario in (
                    "rgb_noise_0.10",
                    "ir_noise_0.10",
                    "rgb_blur_k7",
                    "ir_blur_k7",
                ):
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "degraded_alpha_delta_mean": -0.03,
                        }
                    )
            (diagnostic / "gradient_diagnostics.csv").write_text(
                "weighted_mraf_rel_over_loss_base\n0.01\n", encoding="utf-8"
            )
            (diagnostic / "fusion_path_check.json").write_text(
                '{"sensitive": true}\n', encoding="utf-8"
            )
    (run_root / "candidate_status.tsv").write_text(
        "\n".join(status_rows) + "\n", encoding="utf-8"
    )
    return run_root, reference_root


def test_stage1_matrix_is_exact():
    assert list(rescue.STAGE1_MATRIX) == ["A10", "A50", "A100", "B50"]
    assert [rescue.STAGE1_MATRIX[key]["mraf_scorer_lr_multiplier"] for key in rescue.STAGE1_MATRIX] == [
        10.0,
        50.0,
        100.0,
        50.0,
    ]
    assert rescue.STAGE1_MATRIX["B50"]["mraf_rel_lambda"] == 1.0


def test_training_config_file_supplies_defaults_and_cli_wins(monkeypatch, tmp_path):
    config = tmp_path / "configs.yaml"
    config.write_text(
        "lr: 0.000123\nbatch_size: 17\nhistorical_unknown: kept_out\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config_file",
            str(config),
            "--batch_size",
            "64",
        ],
    )
    args = get_args()
    assert args.lr == pytest.approx(0.000123)
    assert args.batch_size == 64
    assert not hasattr(args, "historical_unknown")


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field", ["mraf_scorer_lr_multiplier", "mraf_gate_logit_scale"]
)
def test_new_positive_parameters_reject_nonpositive_or_nonfinite(field, value):
    with pytest.raises(ValueError):
        validate_mraf_options(_validation_args(**{field: value}))


def test_multiplier_one_preserves_legacy_one_parameter_groups(tmp_path):
    model = ToyModel()
    optimizer = build_optimizer(_optimizer_args(tmp_path, 1.0), model)
    assert len(optimizer.param_groups) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )
    assert all(len(group["params"]) == 1 for group in optimizer.param_groups)


def test_nondefault_multiplier_partitions_every_parameter_once(tmp_path):
    model = ToyModel()
    optimizer = build_optimizer(_optimizer_args(tmp_path, 50.0), model)
    grouped = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    scorer_ids = {
        id(parameter) for _, parameter in get_mraf_scorer_named_parameters(model)
    }
    scorer_groups = [
        group
        for group in optimizer.param_groups
        if str(group.get("group_name", "")).startswith("mraf_scorer")
    ]
    assert scorer_groups
    assert {
        id(parameter)
        for group in scorer_groups
        for parameter in group["params"]
    } == scorer_ids
    assert all(group["lr"] == pytest.approx(5e-4) for group in scorer_groups)


def test_scheduler_and_warmup_preserve_multiplier(tmp_path):
    model = ToyModel()
    args = _optimizer_args(tmp_path, 50.0)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    def ratio():
        scorer = next(
            group["lr"]
            for group in optimizer.param_groups
            if str(group.get("group_name", "")).startswith("mraf_scorer")
        )
        base = next(
            group["lr"]
            for group in optimizer.param_groups
            if not str(group.get("group_name", "")).startswith("mraf_scorer")
            and math.isclose(group["initial_lr"], args.lr)
        )
        return scorer / base

    assert ratio() == pytest.approx(50.0)
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        assert ratio() == pytest.approx(50.0)


def test_optimizer_group_schema_stable(tmp_path):
    model = ToyModel()
    optimizer = build_optimizer(_optimizer_args(tmp_path, 10.0), model)
    rows = optimizer_param_group_rows(optimizer, model)
    assert tuple(rows[0]) == OPTIMIZER_PARAM_GROUP_FIELDS
    written = tmp_path / "optimizer_param_groups.tsv"
    assert written.is_file()
    with written.open(encoding="utf-8", newline="") as handle:
        assert tuple(next(csv.reader(handle, delimiter="\t"))) == OPTIMIZER_PARAM_GROUP_FIELDS


def test_mraf_disabled_default_ok_and_nondefault_fails(tmp_path):
    build_optimizer(_optimizer_args(tmp_path, 1.0), ToyModel(mraf=False))
    with pytest.raises(ValueError, match="no enabled MRAF scorer"):
        build_optimizer(_optimizer_args(tmp_path, 10.0), ToyModel(mraf=False))


def test_logit_scale_raw_scaled_alpha_and_default_compatibility():
    torch.manual_seed(7)
    base = ModalityReliabilityAdaptiveFusion(
        4, hidden_dim=4, dropout=0.0, gate_logit_scale=1.0, gate_fp32=True
    )
    scaled = ModalityReliabilityAdaptiveFusion(
        4, hidden_dim=4, dropout=0.0, gate_logit_scale=5.0, gate_fp32=True
    )
    with torch.no_grad():
        base.scorer[-1].weight.fill_(0.2)
        base.scorer[-1].bias.fill_(0.1)
    scaled.load_state_dict(base.state_dict(), strict=True)
    rgb = torch.randn(8, 4)
    ir = torch.randn(8, 4)
    _, base_aux = base(rgb, ir, return_aux=True)
    _, scaled_aux = scaled(rgb, ir, return_aux=True)
    assert torch.equal(base_aux["logits"], base_aux["raw_logits"])
    assert torch.allclose(scaled_aux["raw_logits"], base_aux["raw_logits"])
    assert torch.allclose(
        scaled_aux["scaled_logits"], scaled_aux["raw_logits"] * 5.0
    )
    assert torch.allclose(
        scaled_aux["alpha"],
        torch.softmax(scaled_aux["scaled_logits"], dim=-1),
    )
    assert base_aux["raw_logits"].dtype == torch.float32
    assert base_aux["alpha"].dtype == torch.float32
    assert not torch.allclose(base_aux["alpha"], scaled_aux["alpha"])


def test_old_checkpoint_strict_loads_with_scale_config_only():
    old = ModalityReliabilityAdaptiveFusion(4, hidden_dim=4, dropout=0.0)
    new = ModalityReliabilityAdaptiveFusion(
        4, hidden_dim=4, dropout=0.0, gate_logit_scale=20.0
    )
    new.load_state_dict(old.state_dict(), strict=True)


def test_test_override_is_read_only():
    module = _load_test_module()
    saved = SimpleNamespace(mraf_gate_logit_scale=1.0)
    runtime = module._apply_mraf_test_overrides(
        saved, SimpleNamespace(mraf_gate_logit_scale=10.0)
    )
    assert saved.mraf_gate_logit_scale == 1.0
    assert runtime.mraf_gate_logit_scale == 10.0


def test_diagnosis_override_is_read_only(monkeypatch, tmp_path):
    saved = SimpleNamespace(mraf_enabled=True, mraf_gate_logit_scale=1.0)
    monkeypatch.setattr(diagnosis, "load_train_configs", lambda _: saved)
    artifact = diagnosis.RunArtifacts(
        case="mraf_only",
        run_dir=tmp_path,
        config_path=tmp_path / "configs.yaml",
        checkpoint_path=tmp_path / "best.pth",
        train_log_path=tmp_path / "train.log",
    )
    runtime = diagnosis._load_config_args(
        artifact,
        tmp_path,
        training=False,
        mraf_gate_logit_scale=20.0,
    )
    assert saved.mraf_gate_logit_scale == 1.0
    assert runtime.mraf_gate_logit_scale == 20.0


def test_diagnosis_records_raw_scaled_logits_and_actual_scale():
    sample_ids = torch.tensor([1, 2])
    clean_rgb = torch.randn(2, 4)
    clean_ir = torch.randn(2, 4)
    raw = torch.tensor([[0.1, -0.1], [0.2, 0.0]])
    scaled = raw * 10.0
    outputs = {
        "global_feat": clean_rgb + clean_ir,
        "rgb_global_feat": clean_rgb,
        "thermal_global_feat": clean_ir,
        "mraf_alpha": torch.softmax(scaled, dim=-1),
        "mraf_logits": raw,
        "raw_mraf_logits": raw,
        "scaled_mraf_logits": scaled,
    }
    record = diagnosis._record_from_outputs(
        sample_ids,
        clean_rgb,
        clean_ir,
        outputs,
        target_temperature=0.1,
    )
    assert torch.allclose(record["logit_gap"], torch.tensor([0.2, 0.2]))
    assert torch.allclose(record["scaled_logit_gap"], torch.tensor([2.0, 2.0]))
    row = diagnosis._gate_statistics_row(
        "case",
        "clean",
        record,
        raw_logits_dtype="torch.float32",
        scaled_logits_dtype="torch.float32",
        raw_alpha_dtype="torch.float32",
        mraf_gate_logit_scale=10.0,
    )
    assert row["mraf_gate_logit_scale"] == 10.0
    assert row["scaled_logit_gap_mean"] == pytest.approx(2.0)


def test_epoch_statistics_detach_and_parameter_delta():
    vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    stats = _DetachedRunningStats()
    stats.update(vector)
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(math.sqrt(2.0 / 3.0))
    paired = _DetachedPairedStats()
    paired.update(vector, vector)
    assert paired.pearson() == pytest.approx(1.0)
    model = ToyModel()
    snapshot = _mraf_initial_parameter_snapshot(model)
    assert _mraf_scorer_parameter_stats(model, snapshot)[
        "scorer_parameter_delta_from_initial"
    ] == 0.0
    with torch.no_grad():
        model.mraf_fusion.scorer[-1].weight.add_(1.0)
    assert _mraf_scorer_parameter_stats(model, snapshot)[
        "scorer_parameter_delta_from_initial"
    ] > 0.0


def test_checkpoint_defaults_and_schemas():
    parser_source = (ROOT / "utils" / "options.py").read_text(encoding="utf-8")
    assert '--save_last_checkpoint", default=False' in parser_source
    assert tuple(CHECKPOINT_SELECTION_FIELDS) == (
        "checkpoint",
        "epoch",
        "clean_R1",
        "alpha_std",
        "Pearson",
        "weighted_rel_base_ratio",
    )
    assert len(MRAF_EPOCH_METRIC_FIELDS) == 30


def test_save_last_flag_and_no_overwrite(tmp_path):
    class FakeCheckpointer:
        def __init__(self):
            self.calls = []

        def save(self, name, **kwargs):
            self.calls.append((name, kwargs))
            (tmp_path / f"{name}.pth").write_text("checkpoint", encoding="utf-8")

    checkpointer = FakeCheckpointer()
    disabled = SimpleNamespace(output_dir=str(tmp_path), save_last_checkpoint=False)
    assert _save_last_checkpoint(checkpointer, disabled, {}, 4) is None
    enabled = SimpleNamespace(output_dir=str(tmp_path), save_last_checkpoint=True)
    assert _save_last_checkpoint(checkpointer, enabled, {"epoch": 2}, 4).endswith(
        "last.pth"
    )
    assert checkpointer.calls[0][1]["epoch"] == 4
    with pytest.raises(FileExistsError):
        _save_last_checkpoint(checkpointer, enabled, {}, 4)


def test_strong_and_viable_thresholds_are_distinct():
    reference = {"clean_R1": 49.88, "degraded_mean_R1": 41.796}
    strong = rescue.classify_candidate(_passing_unit(), reference)
    assert strong["classification"] == "STRONG_PASS"
    viable_unit = _passing_unit()
    viable_unit["mechanism"].update(
        alpha_std=0.0015,
        rgb_missing_alpha_rgb=0.4,
        ir_missing_alpha_ir=0.4,
        rgb_noise_alpha_drop=0.003,
        ir_noise_alpha_drop=0.003,
        rgb_blur_alpha_drop=0.003,
        ir_blur_alpha_drop=0.0,
    )
    viable = rescue.classify_candidate(viable_unit, reference)
    assert viable["classification"] == "VIABLE_PASS"
    viable_unit["mechanism"]["alpha_std"] = 0.0009
    assert rescue.classify_candidate(viable_unit, reference)["classification"] == "FAIL"


def test_ranking_is_deterministic_and_best_last_both_present():
    rows = []
    for candidate in rescue.STAGE1_MATRIX:
        for checkpoint in rescue.CHECKPOINT_KINDS:
            unit = _passing_unit()
            unit.update(candidate=candidate, checkpoint=checkpoint)
            rows.append(rescue.classify_candidate(unit, {"clean_R1": 49.88, "degraded_mean_R1": 41.796}))
    first = rescue.rank_candidates(rows)
    second = rescue.rank_candidates(list(reversed(rows)))
    assert [(row["candidate"], row["checkpoint"]) for row in first] == [
        (row["candidate"], row["checkpoint"]) for row in second
    ]
    assert len(first) == 8


def test_mock_summarizer_writes_all_required_outputs(tmp_path):
    output = tmp_path / "mock"
    rescue.create_mock_outputs(output)
    required = {
        "amplitude_screen_metrics.tsv",
        "amplitude_screen_mechanism.tsv",
        "checkpoint_comparison.tsv",
        "amplitude_candidate_ranking.tsv",
        "calibration_screen.tsv",
        "selected_amplitude_candidate.json",
        "confirmation_summary.tsv",
        "final_metrics.tsv",
        "final_robustness_summary.tsv",
        "final_mechanism_summary.tsv",
        "comparison_with_previous_screen.tsv",
        "comparison_with_old_seed1.tsv",
        "FINAL_REPORT.txt",
    }
    assert required <= {path.name for path in output.iterdir()}
    selected = json.loads((output / "selected_amplitude_candidate.json").read_text())
    assert selected["status"] == "STRONG_SELECTED"


def test_pipeline_contains_safety_gates_and_read_only_calibration():
    script = PIPELINE.read_text(encoding="utf-8")
    assert "for candidate in A10 A50 A100 B50" in script
    assert "for checkpoint_kind in best last" in script
    assert "for scale in 2 5 10 20 50" in script
    assert '--config_file "$base_config"' in script
    assert "--require-gate-rescue-protocol" in script
    assert "--mraf-gate-logit-scale" in script
    assert "60 3" in script
    assert "CONFIRMATION_FAILED" in script
    assert 'if [[ "$shutdown_on_success" == "true"' in script
    assert "git add ." not in script and "git add -A" not in script
    assert 'mktemp "${status_file}.tmp.XXXXXX"' in script
    assert 'mv -f "$temporary_status" "$status_file"' in script


def test_pipeline_mock_never_calls_training_or_shutdown():
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh",
            "--mock",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert "STATUS=MOCK_COMPLETED" in completed.stdout
    assert "shutdown_executed=false" in completed.stdout
    assert "train.py" not in completed.stdout
    assert "test.py" not in completed.stdout


def test_pipeline_dry_run_writes_nothing():
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh",
            "--mock",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert "No directory, train.py, test.py, GPU" in completed.stdout
    assert "mraf_gate_amplitude_mock_DRY_RUN" in completed.stdout


@pytest.mark.parametrize(
    ("relative", "clean_field", "degraded_field"),
    [
        ("screen/rescue_screen_summary.tsv", "clean_R1", "key_degraded_mean_R1"),
        ("rescue_screen_summary.tsv", "clean_r1", "degraded_mean_R1"),
        ("screen/rescue_screen_metrics.tsv", "clean_rank1", "mean_degraded_R1"),
        ("rescue_screen_metrics.tsv", "R1_clean", "key_degraded_R1_mean"),
    ],
)
def test_reference_s4_real_paths_and_aliases(
    tmp_path, relative, clean_field, degraded_field
):
    source = _reference_tsv(
        tmp_path,
        relative,
        clean_field=clean_field,
        degraded_field=degraded_field,
    )
    result = rescue.load_reference_s4(tmp_path)
    assert result["clean_R1"] == pytest.approx(49.8795)
    assert result["key_degraded_mean_R1"] == pytest.approx(41.7958)
    assert result["degraded_mean_R1"] == pytest.approx(41.7958)
    assert result["source"] == str(source.resolve())


def test_reference_s4_bom_whitespace_and_exact_candidate(tmp_path):
    path = _reference_tsv(tmp_path, bom=True, candidate=" S4 ")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("S40\t99\t99\n")
    result = rescue.load_reference_s4(tmp_path)
    assert result["clean_R1"] == pytest.approx(49.8795)


@pytest.mark.parametrize(
    ("clean", "degraded", "message"),
    [
        ("NaN", "41.7", "must be finite"),
        ("49.8", "Inf", "must be finite"),
        ("bad", "41.7", "cannot be converted"),
    ],
)
def test_reference_s4_rejects_nonfinite_and_invalid_values(
    tmp_path, clean, degraded, message
):
    _reference_tsv(tmp_path, clean=clean, degraded=degraded)
    with pytest.raises(ValueError, match=message):
        rescue.load_reference_s4(tmp_path)


@pytest.mark.parametrize(
    ("header", "row", "expected"),
    [
        ("candidate\tkey_degraded_mean_R1", "S4\t41.7", "clean field missing"),
        ("candidate\tclean_R1", "S4\t49.8", "degraded field missing"),
        (
            "candidate\tclean_R1\tkey_degraded_mean_R1",
            "S40\t49.8\t41.7",
            "exact candidate 'S4' was not found",
        ),
    ],
)
def test_reference_s4_errors_are_actionable(tmp_path, header, row, expected):
    path = tmp_path / "screen" / "rescue_screen_summary.tsv"
    path.parent.mkdir()
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError) as captured:
        rescue.load_reference_s4(tmp_path)
    message = str(captured.value)
    assert expected in message
    assert "reference_root=" in message
    assert "searched_paths=" in message
    assert "headers=" in message
    assert path.name in message


def test_reference_summary_preferred_over_metrics(tmp_path):
    _reference_tsv(
        tmp_path,
        "screen/rescue_screen_metrics.tsv",
        clean="1",
        degraded="2",
    )
    _reference_tsv(
        tmp_path,
        "screen/rescue_screen_summary.tsv",
        clean="49.8795",
        degraded="41.7958",
    )
    assert rescue.load_reference_s4(tmp_path)["clean_R1"] == pytest.approx(
        49.8795
    )


def test_mraf_epoch_metric_schema_and_atomic_standard_write(tmp_path):
    required = {
        "target_rgb_mean",
        "target_rgb_std",
        "scorer_first_weight_norm",
        "scorer_last_weight_norm",
        "scorer_parameter_delta_from_initial",
    }
    assert len(MRAF_EPOCH_METRIC_FIELDS) == len(set(MRAF_EPOCH_METRIC_FIELDS))
    assert required <= set(MRAF_EPOCH_METRIC_FIELDS)
    assert _validate_tsv_schema(
        MRAF_EPOCH_METRIC_FIELDS, require_mraf_epoch_fields=True
    ) == MRAF_EPOCH_METRIC_FIELDS
    path = tmp_path / "mraf_epoch_metrics.tsv"
    row = {field: index for index, field in enumerate(MRAF_EPOCH_METRIC_FIELDS)}
    _append_tsv_row(str(path), MRAF_EPOCH_METRIC_FIELDS, row)
    _append_tsv_row(str(path), MRAF_EPOCH_METRIC_FIELDS, row)
    raw = path.read_bytes()
    assert b"\t" in raw and b"\r\n" not in raw
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert tuple(rows[0]) == MRAF_EPOCH_METRIC_FIELDS
    assert all(len(written) == len(MRAF_EPOCH_METRIC_FIELDS) for written in rows)
    assert not list(tmp_path.glob(".mraf_epoch_metrics.tsv.*.tmp"))


def test_mraf_epoch_metric_writer_rejects_bad_schema_and_extra_field(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _validate_tsv_schema(("epoch", "epoch"))
    with pytest.raises(ValueError, match="empty"):
        _validate_tsv_schema(("epoch", ""))
    with pytest.raises(ValueError, match="missing required"):
        _validate_tsv_schema(("epoch",), require_mraf_epoch_fields=True)
    row = {field: 0 for field in MRAF_EPOCH_METRIC_FIELDS}
    row["unexpected"] = 1
    with pytest.raises(ValueError, match="outside schema"):
        _append_tsv_row(
            str(tmp_path / "metrics.tsv"), MRAF_EPOCH_METRIC_FIELDS, row
        )


def test_legacy_tsv_repairs_two_known_merges_read_only(tmp_path):
    path = tmp_path / "mraf_epoch_metrics.tsv"
    path.write_text(_legacy_epoch_text(), encoding="utf-8")
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    result = rescue.read_tsv_compatible(path)
    assert result.legacy_header_repaired is True
    assert "target_rgb_mean" in result.fields
    assert "target_rgb_std" in result.fields
    assert "scorer_first_weight_norm" in result.fields
    assert "scorer_last_weight_norm" in result.fields
    assert len(result.fields) == len(MRAF_EPOCH_METRIC_FIELDS)
    assert len(result.rows[0]) == len(result.fields)
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime


def test_standard_tsv_does_not_trigger_legacy_repair(tmp_path):
    path = tmp_path / "new.tsv"
    rescue._write_tsv(
        path,
        MRAF_EPOCH_METRIC_FIELDS,
        [{field: index for index, field in enumerate(MRAF_EPOCH_METRIC_FIELDS)}],
    )
    result = rescue.read_tsv_compatible(path)
    assert result.legacy_header_repaired is False
    assert result.fields == MRAF_EPOCH_METRIC_FIELDS


def test_screen_reference_failure_does_not_publish_summary_outputs(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        rescue.summarize_screen(tmp_path / "units", reference, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "text",
    [
        "unknownmerged\tother\n1\t2\t3\n",
        "a\tb\n1\t2\n3\n",
    ],
)
def test_legacy_tsv_rejects_unknown_or_inconsistent_schema(tmp_path, text):
    path = tmp_path / "bad.tsv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        rescue.read_tsv_compatible(path)


def test_resume_plan_accepts_complete_failed_amplitude_fixture(tmp_path):
    run_root, _ = _write_resume_fixture(tmp_path)
    plan = rescue.validate_resume_plan(
        run_root, ROOT.parent, rescue.EXPECTED_ORIGINAL_COMMIT
    )
    assert plan.valid, plan.reasons
    assert plan.next_stage == "amplitude_screen"
    assert plan.training_candidates_skipped == 4
    assert plan.checkpoint_units_skipped == 8
    assert plan.evaluations_skipped == 8
    assert plan.diagnostics_skipped == 8
    assert "STAGE1_CHECKPOINT_UNITS_SKIPPED=8" in plan.machine_lines()


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (
            "stage1/units/A10_best/checkpoint.pth",
            "missing or empty",
        ),
        (
            "stage1/units/A10_best/eval/clean/scores.json",
            "missing non-empty evaluation",
        ),
        (
            "stage1/units/A10_best/diagnostics/diagnosis.json",
            "missing or empty diagnostic",
        ),
    ],
)
def test_resume_plan_rejects_missing_artifacts(tmp_path, relative, expected):
    run_root, _ = _write_resume_fixture(tmp_path)
    target = run_root / relative
    target.unlink()
    plan = rescue.validate_resume_plan(
        run_root, ROOT.parent, rescue.EXPECTED_ORIGINAL_COMMIT
    )
    assert not plan.valid
    assert expected in "\n".join(plan.reasons)


def test_resume_plan_rejects_seed_matrix_and_arbitrary_cross_commit(tmp_path):
    run_root, _ = _write_resume_fixture(tmp_path)
    manifest = run_root / "experiment_manifest.txt"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("seed=1", "seed=2"),
        encoding="utf-8",
    )
    metadata = run_root / "stage1/units/A10_best/unit.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["mraf_scorer_lr_multiplier"] = 999
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    plan = rescue.validate_resume_plan(
        run_root,
        ROOT.parent,
        "descendant",
        ancestry_checker=lambda *_: True,
        changed_paths_reader=lambda *_: ("IRRA-main/model/build.py",),
    )
    errors = "\n".join(plan.reasons)
    assert not plan.valid
    assert "seed must remain 1" in errors
    assert "fixed Stage-1 matrix" in errors
    assert "non-hotfix paths" in errors


def test_resume_plan_allows_strict_hotfix_descendant_and_records_history(tmp_path):
    run_root, _ = _write_resume_fixture(tmp_path)
    plan = rescue.validate_resume_plan(
        run_root,
        ROOT.parent,
        "hotfix-descendant",
        ancestry_checker=lambda *_: True,
        changed_paths_reader=lambda *_: tuple(rescue.ALLOWED_HOTFIX_PATHS),
    )
    assert plan.valid, plan.reasons
    manifest_before = (run_root / "experiment_manifest.txt").read_bytes()
    rescue.record_resume_history(run_root, plan)
    assert (run_root / "experiment_manifest.txt").read_bytes() == manifest_before
    rows = rescue.read_tsv_compatible(run_root / "resume_history.tsv").rows
    assert rows[-1]["original_commit"] == rescue.EXPECTED_ORIGINAL_COMMIT
    assert rows[-1]["resume_commit"] == "hotfix-descendant"
    assert rows[-1]["old_status"] == "FAILED"
    assert rows[-1]["old_failed_stage"] == "amplitude_screen"


def _as_wsl_path(path):
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = resolved.as_posix().split(":/", 1)[-1]
    return f"/mnt/{drive}/{suffix}"


def test_pipeline_resume_dry_run_is_read_only_and_machine_readable(tmp_path):
    run_root, _ = _write_resume_fixture(tmp_path, manifest_path_style="wsl")
    status_before = (run_root / "status.txt").read_bytes()
    manifest_before = (run_root / "experiment_manifest.txt").read_bytes()
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh",
            "--resume-run-root",
            _as_wsl_path(run_root),
            "--device",
            "0",
            "--auto-continue",
            "--mock",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert "RESUME_PLAN_VALID=true" in completed.stdout
    assert "STAGE1_TRAIN_CANDIDATES_SKIPPED=4" in completed.stdout
    assert "STAGE1_CHECKPOINT_UNITS_SKIPPED=8" in completed.stdout
    assert "NEXT_STAGE=amplitude_screen" in completed.stdout
    assert (run_root / "status.txt").read_bytes() == status_before
    assert (run_root / "experiment_manifest.txt").read_bytes() == manifest_before
    assert not (run_root / "resume_history.tsv").exists()


def test_pipeline_mock_resume_skips_stage1_and_runs_summarizer(tmp_path):
    run_root, _ = _write_resume_fixture(tmp_path, manifest_path_style="wsl")
    candidate_status_before = (run_root / "candidate_status.tsv").read_bytes()
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh",
            "--resume-run-root",
            _as_wsl_path(run_root),
            "--device",
            "0",
            "--auto-continue",
            "--mock",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert "STATUS=MOCK_RESUME_COMPLETED" in completed.stdout
    assert "STAGE1_TRAIN_CALLS=0" in completed.stdout
    assert "STAGE1_TEST_CALLS=0" in completed.stdout
    assert "STAGE1_DIAGNOSIS_CALLS=0" in completed.stdout
    assert "SUMMARIZER_CALLED=true" in completed.stdout
    assert "shutdown_executed=false" in completed.stdout
    assert (run_root / "candidate_status.tsv").read_bytes() == candidate_status_before
    assert (run_root / "resume_history.tsv").is_file()
    assert (run_root / "selected_amplitude_candidate.json").is_file()
    mechanism_rows = rescue.read_tsv_compatible(
        run_root / "amplitude_screen_mechanism.tsv"
    ).rows
    assert mechanism_rows
    assert {row["legacy_header_repaired"] for row in mechanism_rows} == {"True"}
    assert all(row["legacy_header_path"] for row in mechanism_rows)
    assert "STATUS=MOCK_RESUME_COMPLETED" in (
        run_root / "status.txt"
    ).read_text(encoding="utf-8")
