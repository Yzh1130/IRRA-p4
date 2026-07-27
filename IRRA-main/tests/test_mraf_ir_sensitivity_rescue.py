import csv
import json
import math
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tools.revision import summarize_mraf_gate_amplitude_rescue as v3
from tools.revision import summarize_mraf_ir_sensitivity_rescue as rescue


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "mraf" / "run_mraf_ir_sensitivity_rescue_pipeline.sh"


def _as_shell_path(path):
    raw = str(path).replace("\\", "/")
    if (
        len(raw) >= 3
        and raw[0].isalpha()
        and raw[1] == ":"
        and raw[2] == "/"
    ):
        return f"/mnt/{raw[0].lower()}/{raw[3:].lstrip('/')}"
    return raw


def _source_fixture(tmp_path):
    root = tmp_path / "v3_source"
    root.mkdir()
    (root / "status.txt").write_text(
        "STATUS=CONFIRMATION_FAILED\n"
        "FAILED_STAGE=confirmation\n"
        "EXIT_CODE=5\n",
        encoding="utf-8",
    )
    (root / "experiment_manifest.txt").write_text(
        "PIPELINE_ID=mraf_gate_amplitude_rescue_v3\n"
        f"git_head={rescue.V3_TRAINING_COMMIT}\n"
        "seed=1\n",
        encoding="utf-8",
    )
    selection = {
        "status": "STRONG_SELECTED",
        "selection_source": "read_only_calibration",
        "selected_candidate": "B50",
        "selected_checkpoint": "best",
        "mraf_scorer_lr_multiplier": 50,
        "mraf_gate_logit_scale": 2,
        "mraf_rel_lambda": 1,
        "mraf_gate_base_grad_scale": 1,
    }
    (root / "selected_amplitude_candidate.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    for checkpoint in rescue.CHECKPOINTS:
        unit = root / "confirmation" / "units" / checkpoint
        unit.mkdir(parents=True)
        for name in (
            "checkpoint.pth",
            "configs.yaml",
            "train.log",
            "mraf_epoch_metrics.tsv",
            "checkpoint_selection.tsv",
        ):
            if name == "configs.yaml":
                (unit / name).write_text(
                    "mraf_gate_fp32: true\n"
                    "mraf_scorer_lr_multiplier: 50\n"
                    "mraf_gate_logit_scale: 2\n"
                    "mraf_rel_lambda: 1\n"
                    "mraf_gate_base_grad_scale: 1\n"
                    "num_epoch: 12\n"
                    "warmup_epochs: 5\n"
                    "mraf_rel_warmup_epoch: 1\n",
                    encoding="utf-8",
                )
            else:
                (unit / name).write_bytes(f"{checkpoint}:{name}\n".encode())
    with (root / "confirmation_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "checkpoint",
                "mraf_rel_lambdamraf_gate_base_grad_scale",
                "scorer_parameter_deltaraw_logit_gap_std",
            ]
        )
        writer.writerow(["best", "1", "1", "0.2", "0.1"])
    return root


def _snapshot_tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_protocol_constants_are_fixed():
    assert rescue.READONLY_SCALES == (2.5, 3.0, 4.0, 5.0)
    assert rescue.FORMAL_SEEDS == (1, 2, 3, 4, 5)
    assert len(rescue.SCREEN_SCENARIOS) == 8
    assert len(rescue.FORMAL_SCENARIOS) == 11
    assert "both_noise_0.10" in rescue.FORMAL_SCENARIOS
    assert "both_noise_0.05" not in rescue.FORMAL_SCENARIOS


@pytest.mark.parametrize(
    "status",
    [
        "COMPLETED",
        "IR_SENSITIVITY_RESCUE_FAILED",
        "IR_SENSITIVITY_CONFIRMATION_FAILED",
        "FORMAL_SEED1_FAILED",
        "MULTISEED_FAILED",
        "FAILED",
    ],
)
def test_all_required_terminal_statuses_exist(status):
    assert status in rescue.TERMINAL_STATUSES


def test_source_plan_accepts_exact_v3_confirmation_failure(tmp_path):
    source = _source_fixture(tmp_path)
    plan = rescue.validate_source_run(source)
    assert plan.valid
    assert plan.confirmation_status == "CONFIRMATION_FAILED"
    assert plan.best_found and plan.last_found
    assert plan.legacy_header_repaired
    assert "PLAN_VALID=true" in plan.machine_lines()


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("STATUS", "RUNNING"),
        ("FAILED_STAGE", "formal"),
        ("EXIT_CODE", "4"),
    ],
)
def test_source_plan_rejects_wrong_status_tuple(tmp_path, key, bad):
    source = _source_fixture(tmp_path)
    path = source / "status.txt"
    values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    values[key] = bad
    path.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )
    plan = rescue.validate_source_run(source)
    assert not plan.valid
    assert key in "\n".join(plan.reasons)


@pytest.mark.parametrize("checkpoint", ["best", "last"])
def test_source_plan_requires_both_checkpoints(tmp_path, checkpoint):
    source = _source_fixture(tmp_path)
    (source / "confirmation" / "units" / checkpoint / "checkpoint.pth").unlink()
    plan = rescue.validate_source_run(source)
    assert not plan.valid
    assert checkpoint in "\n".join(plan.reasons)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("selected_candidate", "A50"),
        ("selected_checkpoint", "last"),
        ("mraf_scorer_lr_multiplier", 100),
        ("mraf_gate_logit_scale", 5),
        ("mraf_rel_lambda", 0.5),
        ("mraf_gate_base_grad_scale", 0.1),
    ],
)
def test_source_plan_rejects_changed_v3_selection(tmp_path, field, bad):
    source = _source_fixture(tmp_path)
    path = source / "selected_amplitude_candidate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = bad
    path.write_text(json.dumps(payload), encoding="utf-8")
    plan = rescue.validate_source_run(source)
    assert not plan.valid
    assert field in "\n".join(plan.reasons)


def test_source_validation_is_strictly_read_only(tmp_path):
    source = _source_fixture(tmp_path)
    before = _snapshot_tree(source)
    assert rescue.validate_source_run(source).valid
    assert _snapshot_tree(source) == before


@pytest.mark.parametrize(
    ("merged", "expected"),
    [
        (
            "mraf_rel_lambdamraf_gate_base_grad_scale",
            ("mraf_rel_lambda", "mraf_gate_base_grad_scale"),
        ),
        (
            "scorer_parameter_deltaraw_logit_gap_std",
            ("scorer_parameter_delta", "raw_logit_gap_std"),
        ),
        (
            "target_rgb_meantarget_rgb_std",
            ("target_rgb_mean", "target_rgb_std"),
        ),
        (
            "scorer_first_weight_normscorer_last_weight_norm",
            ("scorer_first_weight_norm", "scorer_last_weight_norm"),
        ),
    ],
)
def test_legacy_header_repairs_only_known_fields(tmp_path, merged, expected):
    path = tmp_path / "legacy.tsv"
    path.write_text(f"id\t{merged}\nrow\t1\t2\n", encoding="utf-8")
    result = rescue.read_tsv_compatible(path)
    assert result.legacy_header_repaired
    assert result.fields == ("id", *expected)
    assert result.rows == [{"id": "row", expected[0]: "1", expected[1]: "2"}]


def test_unknown_merged_header_is_rejected(tmp_path):
    path = tmp_path / "unknown.tsv"
    path.write_text("id\tunknownmerged\nrow\t1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only whitelisted"):
        rescue.read_tsv_compatible(path)


def test_inconsistent_legacy_row_width_is_rejected(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text(
        "id\tmraf_rel_lambdamraf_gate_base_grad_scale\n"
        "a\t1\t2\n"
        "b\t1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Inconsistent"):
        rescue.read_tsv_compatible(path)


def test_atomic_tsv_has_fixed_unique_lf_schema(tmp_path):
    path = tmp_path / "fixed.tsv"
    rescue.write_tsv_atomic(path, ("a", "b"), [{"a": 1, "b": 2}])
    assert path.read_bytes() == b"a\tb\n1\t2\n"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert all(len(row) == 2 for row in rows)


def test_atomic_tsv_rejects_duplicate_fields(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        rescue.write_tsv_atomic(tmp_path / "bad.tsv", ("a", "a"), [])


def test_atomic_tsv_extrasaction_raise_preserves_old_file(tmp_path):
    path = tmp_path / "atomic.tsv"
    path.write_bytes(b"old\n")
    with pytest.raises(ValueError):
        rescue.write_tsv_atomic(
            path,
            ("a",),
            [{"a": 1, "unexpected": 2}],
        )
    assert path.read_bytes() == b"old\n"
    assert not list(tmp_path.glob(".atomic.tsv.*.tmp"))


def _row(scale, checkpoint, classification, degraded, clean):
    return {
        "scale": scale,
        "checkpoint": checkpoint,
        "classification": classification,
        "degraded_mean_R1": degraded,
        "clean_R1": clean,
    }


def test_scale_selection_never_selects_viable():
    rows = [_row(2.5, "best", "VIABLE_PASS", 99, 99)]
    assert rescue.select_readonly_scale(rows) is None


def test_scale_selection_uses_smallest_strong_scale_first():
    rows = [
        _row(3.0, "best", "STRONG_PASS", 99, 99),
        _row(2.5, "last", "STRONG_PASS", 1, 1),
    ]
    assert rescue.select_readonly_scale(rows)["scale"] == 2.5


def test_same_scale_prefers_higher_degraded_mean():
    rows = [
        _row(2.5, "best", "STRONG_PASS", 40, 55),
        _row(2.5, "last", "STRONG_PASS", 41, 50),
    ]
    assert rescue.select_readonly_scale(rows)["checkpoint"] == "last"


def test_same_scale_then_prefers_higher_clean():
    rows = [
        _row(2.5, "best", "STRONG_PASS", 41, 50),
        _row(2.5, "last", "STRONG_PASS", 41, 51),
    ]
    assert rescue.select_readonly_scale(rows)["checkpoint"] == "last"


def test_exact_scale_tie_prefers_best():
    rows = [
        _row(2.5, "last", "STRONG_PASS", 41, 51),
        _row(2.5, "best", "STRONG_PASS", 41, 51),
    ]
    assert rescue.select_readonly_scale(rows)["checkpoint"] == "best"


def test_checkpoint_ranking_prefers_strong_over_better_viable_metrics():
    rows = [
        _row(2, "last", "VIABLE_PASS", 99, 99),
        _row(2, "best", "STRONG_PASS", 40, 50),
    ]
    assert rescue.rank_checkpoint_rows(rows)[0]["checkpoint"] == "best"


def test_classifier_is_reused_without_threshold_fork():
    assert rescue.v3.classify_candidate is v3.classify_candidate
    source = Path(rescue.__file__).read_text(encoding="utf-8")
    assert "alpha_std >= 0.005" not in source
    assert "pearson >= 0.30" not in source


def test_mock_summarizer_creates_all_paper_outputs(tmp_path):
    output = tmp_path / "mock"
    rescue.create_mock_outputs(output)
    required = {
        "multiseed_retrieval_summary.tsv",
        "multiseed_robustness_summary.tsv",
        "multiseed_mechanism_summary.tsv",
        "multiseed_classification_summary.tsv",
        "PAPER_READY_RESULTS_ZH.md",
        "PAPER_READY_RESULTS_EN.md",
        "FINAL_REPORT.txt",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert {
        "main_retrieval_results.tsv",
        "robustness_results.tsv",
        "mechanism_results.tsv",
        "ablation_scale_results.tsv",
        "seed_level_results.tsv",
    } <= {path.name for path in (output / "paper_tables").iterdir()}


def test_multiseed_statistics_use_five_seeds(tmp_path):
    output = tmp_path / "mock"
    rescue.create_mock_outputs(output)
    rows = rescue.read_tsv_compatible(
        output / "multiseed_retrieval_summary.tsv"
    ).rows
    assert rows
    assert {row["n"] for row in rows} == {"5"}
    assert all(math.isclose(float(row["std"]), 0.0) for row in rows)
    robustness = rescue.read_tsv_compatible(
        output / "multiseed_robustness_summary.tsv"
    ).rows
    assert len(robustness) == len(rescue.FORMAL_SCENARIOS) * 5 + 1
    assert {row["n"] for row in robustness} == {"5"}
    assert set(rescue.FORMAL_SCENARIOS) <= {
        row["scenario"] for row in robustness
    }
    main = rescue.read_tsv_compatible(
        output / "paper_tables" / "main_retrieval_results.tsv"
    ).rows
    assert {row["scope"] for row in main} >= {
        "five_seed_mean",
        "five_seed_std",
    }


@pytest.mark.parametrize(
    "title",
    [
        "最终方法配置",
        "24-batch复诊结论",
        "五种子均值与标准差",
        "不能写入论文的过度结论",
        "单seed与五seed结果边界",
    ],
)
def test_chinese_paper_report_has_required_sections(tmp_path, title):
    output = tmp_path / "mock"
    rescue.create_mock_outputs(output)
    text = (output / "PAPER_READY_RESULTS_ZH.md").read_text(encoding="utf-8")
    assert title in text


@pytest.mark.parametrize(
    "title",
    [
        "Final Method Configuration",
        "24-Batch Rediagnosis",
        "Five-Seed Mean and Standard Deviation",
        "Claims That Must Not Be Made",
        "Single-Seed Versus Five-Seed Boundary",
    ],
)
def test_english_paper_report_has_required_sections(tmp_path, title):
    output = tmp_path / "mock"
    rescue.create_mock_outputs(output)
    text = (output / "PAPER_READY_RESULTS_EN.md").read_text(encoding="utf-8")
    assert title in text


def test_cli_help_lists_all_modes_and_arguments():
    completed = subprocess.run(
        [
            os.sys.executable,
            "-B",
            "tools/revision/summarize_mraf_ir_sensitivity_rescue.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for value in (
        "source-plan",
        "full-rediagnosis",
        "readonly-scale",
        "confirmation",
        "formal-seed",
        "multiseed",
    ):
        assert value in completed.stdout


def test_pipeline_dry_run_is_machine_readable_and_read_only(tmp_path):
    output = tmp_path / "dry run output"
    output.mkdir()
    before = _snapshot_tree(output)
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh",
            "--mock",
            "--dry-run",
            "--output-root",
            _as_shell_path(output),
            "--shutdown-on-terminal",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    required = (
        "PLAN_VALID=true",
        "SOURCE_CONFIRMATION_STATUS=CONFIRMATION_FAILED",
        "SOURCE_BEST_FOUND=true",
        "SOURCE_LAST_FOUND=true",
        "FULL_REDIAGNOSIS_BATCHES=24",
        "READONLY_SCALES=2.5,3,4,5",
        "NEXT_STAGE=full_rediagnosis",
        "FORMAL_SEEDS=1,2,3,4,5",
        "SHUTDOWN_ON_TERMINAL=true",
        "DRY_RUN_READ_ONLY=true",
        "TRAINING_EXECUTED=false",
        "GPU_ACCESSED=false",
        "SHUTDOWN_EXECUTED=false",
    )
    assert all(value in completed.stdout for value in required)
    assert _snapshot_tree(output) == before


def test_pipeline_mock_never_trains_uses_gpu_or_shutdown(tmp_path):
    output = tmp_path / "mock output"
    output.mkdir()
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh",
            "--mock",
            "--output-root",
            _as_shell_path(output),
            "--shutdown-on-terminal",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert "STATUS=MOCK_COMPLETED" in completed.stdout
    assert "TRAINING_EXECUTED=false" in completed.stdout
    assert "GPU_ACCESSED=false" in completed.stdout
    assert "SHUTDOWN_EXECUTED=false" in completed.stdout
    assert "train.py" not in completed.stdout
    run_root = next(path for path in output.iterdir() if path.is_dir())
    assert (run_root / "mraf_ir_sensitivity_rescue_reports.tar.gz").is_file()


def test_mock_archive_excludes_checkpoints(tmp_path):
    output = tmp_path / "archive"
    output.mkdir()
    completed = subprocess.run(
        [
            "bash",
            "scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh",
            "--mock",
            "--output-root",
            _as_shell_path(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    run_root = next(path for path in output.iterdir() if path.is_dir())
    archive = _as_shell_path(
        run_root / "mraf_ir_sensitivity_rescue_reports.tar.gz"
    )
    listing = subprocess.run(
        ["bash", "-lc", f"tar -tzf {shlex.quote(archive)}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    ).stdout
    assert ".pth" not in listing
    assert "scores.pt" not in listing
    assert "paper_tables/main_retrieval_results.tsv" in listing


def test_pipeline_static_training_and_gate_protocol():
    script = PIPELINE.read_text(encoding="utf-8")
    assert "for scale in 2.5 3 4 5" in script
    assert 'evaluate_unit "$unit" source_confirmation "$screen_scenarios" 24 2' in script
    assert '"mraf_ir_sensitivity_confirmation_seed1" 12 1 1' in script
    assert '"mraf_ir_sensitivity_formal_seed${formal_seed}"' in script
    assert '60 3 "$formal_seed" "$selected_scale"' in script
    assert "both_noise_0.10" in script
    assert "both_noise_0.05" not in script
    assert "FORMAL_SEED1_FAILED" in script
    assert 'for formal_seed in 1 2 3 4 5' in script


def test_training_commands_never_resume_from_source_checkpoint():
    script = PIPELINE.read_text(encoding="utf-8")
    training = script.split("run_training() {", 1)[1].split(
        "scenario_values() {", 1
    )[0]
    assert "--resume" not in training
    assert "--test_ckpt_file" not in training
    assert "source_run_root" not in training
    assert "checkpoint.pth" not in training


def test_source_checkpoint_is_used_only_for_readonly_units():
    script = PIPELINE.read_text(encoding="utf-8")
    source_function = script.split("prepare_source_unit() {", 1)[1].split(
        "prepare_trained_unit() {", 1
    )[0]
    assert "source_run_root" in source_function
    assert "ln -s" in source_function
    assert "train.py" not in source_function


def test_resume_requires_artifact_validation_before_skip():
    script = PIPELINE.read_text(encoding="utf-8")
    assert "unit_complete" in script
    assert "scores.pt" in script
    assert "diagnosis.json" in script
    assert 'if seed_complete "$formal_seed"' in script
    assert "Skipping complete formal seed" in script


def test_shutdown_is_terminal_only_and_mock_bypasses_it():
    script = PIPELINE.read_text(encoding="utf-8")
    assert "--shutdown-on-terminal" in script
    assert 'if [[ "$mock" == "true" ]]' in script
    terminal = script.split("terminal_exit() {", 1)[1].split(
        "on_exit() {", 1
    )[0]
    assert "write_status" in terminal
    assert "verify_source_unchanged" in terminal
    assert '"$shutdown_command" -h now' in terminal


def test_existing_v3_pipeline_is_not_referenced_as_an_output():
    script = PIPELINE.read_text(encoding="utf-8")
    assert "mraf_ir_sensitivity_rescue_v4" in script
    assert "MRAF_IR_SENSITIVITY" not in script or "output_root" in script
    assert "git add ." not in script
    assert "git add -A" not in script
