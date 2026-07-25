import csv
import inspect
import json
import re
from pathlib import Path

import pytest

from tools.revision import summarize_mraf_gate_rescue as rescue


EXPECTED_MATRIX = {
    "S0": (False, 0.01, 1.00),
    "S1": (True, 0.01, 1.00),
    "S2": (True, 0.25, 1.00),
    "S3": (True, 1.00, 1.00),
    "S4": (True, 0.25, 0.10),
    "S5": (True, 0.25, 0.00),
}


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
