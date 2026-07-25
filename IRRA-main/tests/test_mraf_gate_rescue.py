import csv
import inspect
import json
import re
from pathlib import Path

import pytest

from tools.revision import summarize_mraf_gate_rescue as rescue
from tools.revision import validate_mraf_rescue_config as validator
from utils.config import ConfigLoadError, load_config_mapping


EXPECTED_MATRIX = {
    "S0": (False, 0.01, 1.00),
    "S1": (True, 0.01, 1.00),
    "S2": (True, 0.25, 1.00),
    "S3": (True, 1.00, 1.00),
    "S4": (True, 0.25, 0.10),
    "S5": (True, 0.25, 0.00),
}


def _write_protocol_config(path, *, label="S4", tuple_tag=True):
    gate_fp32, rel_lambda, grad_scale = EXPECTED_MATRIX[label]
    tuple_value = (
        "img_size: !!python/tuple\n"
        "  - 384\n"
        "  - 128\n"
        if tuple_tag
        else "img_size: [384, 128]\n"
    )
    path.write_text(
        "dataset_name: RGBT-PEDES\n"
        f"{tuple_value}"
        "num_epoch: 8\n"
        "batch_size: 64\n"
        "loss_names: sdm+mlm+id\n"
        "MLM: true\n"
        "warmup_epochs: 5\n"
        "eval_use_fused_mu: true\n"
        "mdrc_loss_scale: 100\n"
        "mdrc_lambda: 0.05\n"
        "mdrc_prob: 0.5\n"
        "mdrc_warmup_epoch: 3\n"
        f"mraf_rel_lambda: {rel_lambda}\n"
        f"mraf_gate_fp32: {str(gate_fp32).lower()}\n"
        f"mraf_gate_base_grad_scale: {grad_scale}\n",
        encoding="utf-8",
    )


def _metrics(clean=70.0, degraded=60.0):
    return {
        scenario: {
            "R1": clean if scenario == "clean" else degraded,
            "R5": 80.0,
            "R10": 85.0,
            "mAP": 65.0,
            "mINP": 55.0,
        }
        for scenario in rescue.SCREEN_SCENARIOS
    }


def _candidate(label, *, passing=True, clean=70.0, degraded=60.0):
    mechanism = {
        "candidate": label,
        "alpha_rgb_std": 0.006 if passing else 0.004,
        "alpha_target_pearson": 0.31,
        "rgb_missing_alpha_rgb_mean": 0.30,
        "ir_missing_alpha_ir_mean": 0.30,
        "rgb_noise_alpha_drop": 0.02,
        "ir_noise_alpha_drop": 0.02,
        "rgb_blur_alpha_drop": 0.02,
        "ir_blur_alpha_drop": 0.02,
        "direction_correct_fraction": 0.75,
        "weighted_rel_base_median": 0.001,
        "scorer_rel_only_gradient": 0.02,
        "scorer_parameter_updated": True,
        "fusion_path_sensitive": True,
        "gate_logits_fp32": label != "S0",
    }
    return {
        "candidate": label,
        **rescue.CANDIDATE_MATRIX[label],
        "complete": True,
        "has_nan": False,
        "has_inf": False,
        "has_oom": False,
        "metrics": _metrics(clean, degraded),
        "mechanism": mechanism,
    }


def test_candidate_matrix_is_exact_and_stable():
    actual = {
        label: (
            row["mraf_gate_fp32"],
            row["mraf_rel_lambda"],
            row["mraf_gate_base_grad_scale"],
        )
        for label, row in rescue.CANDIDATE_MATRIX.items()
    }
    assert actual == EXPECTED_MATRIX


def test_all_hard_filter_boundaries_pass_inclusively():
    rows = rescue.apply_hard_filters(
        [_candidate("S0"), _candidate("S4", clean=69.5, degraded=59.8)]
    )
    by_label = {row["candidate"]: row for row in rows}
    assert by_label["S0"]["hard_filter_pass"] is True
    assert by_label["S4"]["hard_filter_pass"] is True
    assert by_label["S4"]["failed_criteria"] == ""


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("alpha_rgb_std", 0.0049, "ALPHA_STD"),
        ("alpha_target_pearson", 0.299, "PEARSON"),
        ("rgb_missing_alpha_rgb_mean", 0.351, "RGB_MISSING_WEIGHT"),
        ("ir_missing_alpha_ir_mean", 0.351, "IR_MISSING_WEIGHT"),
        ("rgb_noise_alpha_drop", 0.019, "RGB_NOISE_DIRECTION"),
        ("ir_noise_alpha_drop", 0.019, "IR_NOISE_DIRECTION"),
        ("rgb_blur_alpha_drop", 0.019, "RGB_BLUR_DIRECTION"),
        ("ir_blur_alpha_drop", 0.019, "IR_BLUR_DIRECTION"),
        ("weighted_rel_base_median", 0.0009, "LOSS_RATIO"),
        ("weighted_rel_base_median", 0.051, "LOSS_RATIO"),
        ("scorer_parameter_updated", False, "PARAMETER_NOT_UPDATED"),
        ("fusion_path_sensitive", False, "FUSION_PATH_NOT_SENSITIVE"),
    ],
)
def test_each_mechanism_hard_filter_rejects(field, value, failure):
    baseline = _candidate("S0")
    candidate = _candidate("S4")
    candidate["mechanism"][field] = value
    row = rescue.apply_hard_filters([baseline, candidate])[1]
    assert row["hard_filter_pass"] is False
    assert failure in row["failed_criteria"].split(",")


def test_training_failure_flags_are_hard_failures():
    baseline = _candidate("S0")
    candidate = _candidate("S4")
    candidate.update(
        {"complete": False, "has_nan": True, "has_inf": True, "has_oom": True}
    )
    failures = rescue.apply_hard_filters([baseline, candidate])[1][
        "failed_criteria"
    ].split(",")
    assert {"INCOMPLETE", "NAN", "INF", "OOM"}.issubset(failures)


def test_ranking_order_and_tie_breaks_are_deterministic():
    baseline, s1, s4, s5 = (
        _candidate("S0"),
        _candidate("S1"),
        _candidate("S4"),
        _candidate("S5"),
    )
    summary = rescue.apply_hard_filters([baseline, s1, s4, s5])
    # Equal metrics: smaller lambda first, then larger base-grad scale.
    ranking = rescue.rank_candidates(summary)
    assert [row["candidate"] for row in ranking] == ["S0", "S1", "S4", "S5"]
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4]


def test_no_passing_candidate_reports_screen_failed(tmp_path):
    root = tmp_path / "candidates"
    root.mkdir()
    for label in rescue.CANDIDATE_MATRIX:
        (root / label).mkdir()
    paths = rescue.summarize(root, tmp_path / "summary")
    selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
    assert selected["status"] == "SCREEN_FAILED"
    assert selected["selected_candidate"] is None


def test_tsv_schema_and_non_overwrite_are_stable(tmp_path):
    root = tmp_path / "candidates"
    root.mkdir()
    for label in rescue.CANDIDATE_MATRIX:
        (root / label).mkdir()
    output = tmp_path / "summary"
    paths = rescue.summarize(root, output)
    with paths["summary"].open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    assert tuple(header) == rescue.SUMMARY_FIELDS
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        rescue.summarize(root, output)


def test_pipeline_contains_failure_continuation_and_strict_guards():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mraf"
        / "run_mraf_gate_rescue_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in script
    assert "if run_candidate" in script
    assert 'selection_status" != "SELECTED"' in script
    assert 'auto_continue" != "true"' in script
    assert 'shutdown_on_success" == "true"' in script
    assert 'formal_completed" == "true"' in script
    assert '"$shutdown_command" -h now' in script
    assert "MRAF_SHUTDOWN_COMMAND" in script
    assert "mraf_gate_rescue_reports.tar.gz" in script
    assert "--dry-run" in script and "--mock" in script
    assert not re.search(r"(^|[;&|]\s*)screen(?:\s|$)", script, re.MULTILINE)


def test_mock_branch_precedes_real_training_preflight():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mraf"
        / "run_mraf_gate_rescue_pipeline.sh"
    ).read_text(encoding="utf-8")
    mock_branch = script.index('if [[ "$mock" == "true" ]]')
    cuda_preflight = script.index("command -v nvidia-smi")
    candidate_loop = script.index("for label in S0 S1 S2 S3 S4 S5")
    assert mock_branch < cuda_preflight
    assert candidate_loop < cuda_preflight


def test_cli_help_exposes_screen_and_final_modes():
    help_text = rescue.build_parser().format_help()
    assert "--candidate-root" in help_text
    assert "--final-run-dir" in help_text
    assert "--old-run-root" in help_text
    assert "--output-dir" in help_text


def test_pipeline_exposes_reference_and_resume_protocol():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mraf"
        / "run_mraf_gate_rescue_pipeline.sh"
    ).read_text(encoding="utf-8")
    for flag in (
        "--base-config",
        "--reference-run-root",
        "--resume-run-root",
    ):
        assert flag in script
    assert '--old-run-root "$reference_run_root"' in script
    assert 'STATUS=COMPLETED runs cannot be resumed' in script
    assert "RESUME_SKIPPED_COMPLETE" in script
    assert "Resume commit differs" in script
    assert "Resume seed differs" in script
    assert "Resume candidate matrix differs" in script


def test_summarizer_source_does_not_run_training_or_optimizer_steps():
    source = inspect.getsource(rescue)
    assert "optimizer.step" not in source
    assert "train.py" not in source


def test_plain_yaml_and_scalar_semantics_are_preserved(tmp_path):
    config_path = tmp_path / "plain.yaml"
    config_path.write_text(
        "enabled: true\n"
        "disabled: false\n"
        "nothing: null\n"
        "name: RGBT-PEDES\n"
        "count: 64\n"
        "ratio: 0.25\n"
        "items:\n"
        "  - one\n"
        "  - two\n",
        encoding="utf-8",
    )
    config = load_config_mapping(config_path)
    assert config == {
        "enabled": True,
        "disabled": False,
        "nothing": None,
        "name": "RGBT-PEDES",
        "count": 64,
        "ratio": 0.25,
        "items": ["one", "two"],
    }


def test_historical_tuple_supports_nested_and_mixed_sequences(tmp_path):
    config_path = tmp_path / "tuple.yaml"
    config_path.write_text(
        "img_size: !!python/tuple\n"
        "  - 384\n"
        "  - 128\n"
        "nested:\n"
        "  value: !!python/tuple\n"
        "    - !!python/tuple\n"
        "      - 1\n"
        "      - 2\n"
        "    - 3\n"
        "tuple_with_list: !!python/tuple\n"
        "  - [4, 5]\n"
        "list_with_tuple:\n"
        "  - !!python/tuple\n"
        "    - 6\n"
        "    - 7\n",
        encoding="utf-8",
    )
    config = load_config_mapping(config_path)
    assert config["img_size"] == (384, 128)
    assert config["nested"]["value"] == ((1, 2), 3)
    assert config["tuple_with_list"] == ([4, 5],)
    assert config["list_with_tuple"] == [(6, 7)]


@pytest.mark.parametrize(
    "payload",
    [
        "value: !!python/object:builtins.dict {}\n",
        "value: !!python/object/apply:builtins.str [unsafe]\n",
    ],
)
def test_unknown_python_tags_are_rejected_with_path_and_tag(tmp_path, payload):
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ConfigLoadError) as raised:
        load_config_mapping(config_path)
    message = str(raised.value)
    assert str(config_path.resolve()) in message
    assert "python/object" in message


def test_arbitrary_python_constructor_is_not_executed(tmp_path):
    marker = tmp_path / "must_not_exist.txt"
    expression = (
        f"__import__('pathlib').Path({str(marker)!r}).write_text('owned')"
    )
    config_path = tmp_path / "unsafe_apply.yaml"
    config_path.write_text(
        "value: !!python/object/apply:builtins.eval\n"
        f"  - {json.dumps(expression)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError, match="python/object/apply"):
        load_config_mapping(config_path)
    assert not marker.exists()


def test_empty_config_is_rejected(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="empty"):
        load_config_mapping(config_path)


def test_non_mapping_top_level_is_rejected(tmp_path):
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="top level must be a mapping"):
        load_config_mapping(config_path)


def test_missing_config_is_rejected_with_absolute_path(tmp_path):
    config_path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigLoadError) as raised:
        load_config_mapping(config_path)
    assert "Config file not found" in str(raised.value)
    assert str(config_path.resolve()) in str(raised.value)


def test_validator_reports_required_fields_for_tuple_config(tmp_path, capsys):
    config_path = tmp_path / "configs.yaml"
    _write_protocol_config(config_path)
    exit_code = validator.main(
        [
            "--config",
            str(config_path),
            "--require-gate-rescue-protocol",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    for expected in (
        "config_valid=true",
        "top_level_type=dict",
        "tuple_tag_supported=true",
        "dataset_name=RGBT-PEDES",
        "num_epoch=8",
        "batch_size=64",
        "mraf_rel_lambda=0.25",
        "mraf_gate_fp32=true",
        "mraf_gate_base_grad_scale=0.1",
    ):
        assert expected in output


def test_validator_returns_nonzero_without_traceback(tmp_path, capsys):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "value: !!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )
    exit_code = validator.main(["--config", str(config_path)])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert "config_valid=false" in captured.err
    assert "config_error=" in captured.err
    assert "Traceback" not in captured.err


def test_summarizer_reads_tuple_tagged_candidate_config(tmp_path):
    candidate_root = tmp_path / "candidates"
    candidate_dir = candidate_root / "S4"
    candidate_dir.mkdir(parents=True)
    _write_protocol_config(candidate_dir / "configs.yaml", label="S4")
    candidate = rescue.collect_candidate(candidate_root, "S4")
    assert candidate["config_valid"] is True
    assert candidate["complete"] is False


def test_summarizer_rejects_saved_candidate_matrix_mismatch(tmp_path):
    candidate_root = tmp_path / "candidates"
    candidate_dir = candidate_root / "S4"
    candidate_dir.mkdir(parents=True)
    config_path = candidate_dir / "configs.yaml"
    _write_protocol_config(config_path, label="S4")
    text = config_path.read_text(encoding="utf-8").replace(
        "mraf_rel_lambda: 0.25", "mraf_rel_lambda: 1.0"
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="saved config mismatch"):
        rescue.collect_candidate(candidate_root, "S4")


def test_final_summarizer_reads_formal_and_base_tuple_configs(
    tmp_path, monkeypatch
):
    final_run = tmp_path / "formal"
    diagnostics = final_run / "diagnostics"
    diagnostics.mkdir(parents=True)
    _write_protocol_config(final_run / "configs.yaml", label="S4")
    base_config = tmp_path / "historical_configs.yaml"
    _write_protocol_config(base_config, label="S0")
    (diagnostics / "diagnosis.json").write_text(
        json.dumps({"cases": {"S4_formal": {}}}),
        encoding="utf-8",
    )
    aggregate = {
        "clean_R1": 70.0,
        "clean_mAP": 65.0,
        "clean_mINP": 55.0,
        "degraded_mean_R1": 60.0,
        "relative_clean_mean_drop_R1": 10.0,
        "worst_scenario": "rgb_missing",
        "worst_R1": 50.0,
    }
    monkeypatch.setattr(rescue, "_final_metrics", lambda _: ([], aggregate))
    paths = rescue.summarize_final(
        final_run,
        tmp_path / "summary",
        base_config=base_config,
    )
    report = paths["report"].read_text(encoding="utf-8")
    assert "formal_config_valid=True" in report
    assert "base_config_valid=True" in report


def test_historical_comparison_does_not_hide_unsafe_config_tags(tmp_path):
    old_run = tmp_path / "20260724_mraf_only_seed1"
    old_run.mkdir()
    (old_run / "configs.yaml").write_text(
        "value: !!python/object/apply:builtins.str [unsafe]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError, match="python/object/apply"):
        rescue._old_comparison_row(tmp_path, "mraf_only")


def test_pipeline_uses_central_validator_for_all_config_stages():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mraf"
        / "run_mraf_gate_rescue_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "validate_mraf_rescue_config.py" in script
    assert "yaml.safe_load" not in script
    assert "config_is_valid" in script
    assert 'config_is_valid "$config_path"' in script
    assert 'config_is_valid "${path}/configs.yaml"' in script
    assert 'config_is_valid "${formal_run_path}/configs.yaml"' in script
    assert '--base-config "$base_config"' in script


def test_pipeline_config_failure_schema_is_stable():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "mraf"
        / "run_mraf_gate_rescue_pipeline.sh"
    ).read_text(encoding="utf-8")
    for field in (
        "STATUS=FAILED",
        "FAILED_STAGE=PREFLIGHT_CONFIG",
        "EXIT_CODE=",
        "CONFIG_PATH=",
        "CONFIG_ERROR=",
    ):
        assert field in script
    assert "Traceback" not in script


def test_validator_help_exposes_read_only_config_interface():
    help_text = validator.build_parser().format_help()
    assert "--config" in help_text
    assert "--require-gate-rescue-protocol" in help_text
    source = inspect.getsource(validator)
    assert "torch" not in source
    assert "build_model" not in source
    assert "optimizer" not in source
