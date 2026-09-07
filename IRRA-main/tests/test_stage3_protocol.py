import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import processor.processor as processor
from tools.protocol import freeze_hparams
from utils.options import get_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _RecordingCheckpointer:
    def __init__(self, events):
        self.events = events

    def save(self, name, **kwargs):
        self.events.append(("checkpoint", name, dict(kwargs)))


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def test_save_metric_best_checkpoint_default_remains_false(monkeypatch):
    monkeypatch.setattr("sys.argv", ["test_stage3_protocol"])
    assert get_args().save_metric_best_ckpt is False


def _save_best(monkeypatch, tmp_path, setting, *, save_last_checkpoint=True):
    events = []
    checkpointer = _RecordingCheckpointer(events)
    args_values = {
        "output_dir": str(tmp_path),
        "save_last_checkpoint": save_last_checkpoint,
    }
    if setting != "missing":
        args_values["save_metric_best_ckpt"] = setting
    args = SimpleNamespace(**args_values)

    def record_selection(path, fields, row):
        events.append(("selection", path, tuple(fields), dict(row)))

    def record_metric_checkpoint(**kwargs):
        events.append(("metric", dict(kwargs)))

    monkeypatch.setattr(processor, "_append_tsv_row", record_selection)
    monkeypatch.setattr(
        processor, "_save_metric_checkpoint", record_metric_checkpoint
    )
    saved = processor._save_best_checkpoint_artifacts(
        checkpointer=checkpointer,
        args=args,
        epoch=4,
        metric_r1=51.25,
        eval_metrics={"R1": 51.25, "mAP": 46.5},
        arguments={"epoch": 4, "iteration": 20},
        last_mraf_epoch_row={
            "alpha_rgb_std": 0.12,
            "alpha_target_pearson": 0.34,
            "weighted_rel_base_ratio": 0.56,
        },
        checkpoint_selection_path=str(tmp_path / "checkpoint_selection.tsv"),
        logger=object(),
    )
    return saved, events, args


def test_false_or_missing_setting_suppresses_all_best_artifacts(
    monkeypatch, tmp_path
):
    for setting in (False, "missing"):
        saved, events, args = _save_best(monkeypatch, tmp_path, setting)
        assert processor._metric_best_checkpoint_enabled(args) is False
        assert saved is False
        assert events == []


def test_true_setting_preserves_best_artifact_content_and_order(
    monkeypatch, tmp_path
):
    saved, events, args = _save_best(monkeypatch, tmp_path, True)

    assert processor._metric_best_checkpoint_enabled(args) is True
    assert saved is True
    assert [event[0] for event in events] == ["checkpoint", "selection", "metric"]
    assert events[0] == (
        "checkpoint",
        "best",
        {"epoch": 4, "iteration": 20},
    )
    assert events[1][2] == processor.CHECKPOINT_SELECTION_FIELDS
    assert events[1][3] == {
        "checkpoint": "best.pth",
        "epoch": 4,
        "clean_R1": 51.25,
        "alpha_std": 0.12,
        "Pearson": 0.34,
        "weighted_rel_base_ratio": 0.56,
    }
    assert events[2][1]["reason"] == "best"
    assert events[2][1]["epoch"] == 4
    assert events[2][1]["r1"] == 51.25


def test_true_without_last_checkpoint_keeps_old_selection_behavior(
    monkeypatch, tmp_path
):
    saved, events, _ = _save_best(
        monkeypatch, tmp_path, True, save_last_checkpoint=False
    )

    assert saved is True
    assert [event[0] for event in events] == ["checkpoint", "metric"]


def test_best_gate_does_not_change_last_checkpoint_call_or_payload(tmp_path):
    calls_by_setting = {}
    for setting in (False, True):
        events = []
        output_dir = tmp_path / str(setting).lower()
        output_dir.mkdir()
        args = SimpleNamespace(
            output_dir=str(output_dir),
            save_last_checkpoint=True,
            save_metric_best_ckpt=setting,
        )

        saved_path = processor._save_last_checkpoint(
            _RecordingCheckpointer(events), args, {"epoch": 4}, 60
        )

        assert saved_path == str(output_dir / "last.pth")
        calls_by_setting[setting] = events

    expected = [("checkpoint", "last", {"epoch": 60})]
    assert calls_by_setting[False] == expected
    assert calls_by_setting[True] == expected


def test_best_helper_does_not_mutate_training_state_and_disabled_is_silent(
    monkeypatch, tmp_path
):
    events = []
    logger = _RecordingLogger()
    eval_metrics = {"R1": 51.25, "mAP": 46.5, "sentinel": (1, 2, 3)}
    arguments = {"epoch": 4, "iteration": 20, "sentinel": "unchanged"}
    mraf_row = {
        "alpha_rgb_std": 0.12,
        "alpha_target_pearson": 0.34,
        "weighted_rel_base_ratio": 0.56,
    }
    original_eval_metrics = dict(eval_metrics)
    original_arguments = dict(arguments)
    original_mraf_row = dict(mraf_row)

    monkeypatch.setattr(
        processor,
        "_append_tsv_row",
        lambda *args, **kwargs: events.append(("selection", args, kwargs)),
    )
    monkeypatch.setattr(
        processor,
        "_save_metric_checkpoint",
        lambda **kwargs: events.append(("metric", kwargs)),
    )

    saved = processor._save_best_checkpoint_artifacts(
        checkpointer=_RecordingCheckpointer(events),
        args=SimpleNamespace(
            output_dir=str(tmp_path),
            save_last_checkpoint=True,
            save_metric_best_ckpt=False,
        ),
        epoch=4,
        metric_r1=51.25,
        eval_metrics=eval_metrics,
        arguments=arguments,
        last_mraf_epoch_row=mraf_row,
        checkpoint_selection_path=str(tmp_path / "checkpoint_selection.tsv"),
        logger=logger,
    )

    assert saved is False
    assert events == []
    assert logger.calls == []
    assert eval_metrics == original_eval_metrics
    assert arguments == original_arguments
    assert mraf_row == original_mraf_row

    enabled_saved = processor._save_best_checkpoint_artifacts(
        checkpointer=_RecordingCheckpointer(events),
        args=SimpleNamespace(
            output_dir=str(tmp_path),
            save_last_checkpoint=True,
            save_metric_best_ckpt=True,
        ),
        epoch=4,
        metric_r1=51.25,
        eval_metrics=eval_metrics,
        arguments=arguments,
        last_mraf_epoch_row=mraf_row,
        checkpoint_selection_path=str(tmp_path / "checkpoint_selection.tsv"),
        logger=logger,
    )

    assert enabled_saved is True
    assert [event[0] for event in events] == ["checkpoint", "selection", "metric"]
    assert logger.calls == []
    assert eval_metrics == original_eval_metrics
    assert arguments == original_arguments
    assert mraf_row == original_mraf_row


def test_checkpoint_selection_rows_share_the_best_checkpoint_gate(tmp_path):
    selection_path = tmp_path / "checkpoint_selection.tsv"
    row = {
        "checkpoint": "last.pth",
        "epoch": 60,
        "clean_R1": 50.0,
        "alpha_std": None,
        "Pearson": None,
        "weighted_rel_base_ratio": None,
    }

    disabled = SimpleNamespace(save_metric_best_ckpt=False)
    assert (
        processor._append_checkpoint_selection_row(
            disabled, str(selection_path), row
        )
        is False
    )
    assert not selection_path.exists()

    enabled = SimpleNamespace(save_metric_best_ckpt=True)
    assert (
        processor._append_checkpoint_selection_row(
            enabled, str(selection_path), row
        )
        is True
    )
    assert selection_path.read_text(encoding="utf-8").splitlines() == [
        "checkpoint\tepoch\tclean_R1\talpha_std\tPearson\tweighted_rel_base_ratio",
        "last.pth\t60\t50.0\t\t\t",
    ]


def test_clean_r1_column_is_reproducible_without_changing_frozen_selection():
    metrics_path = PROJECT_ROOT / "frozen" / "dev_grid_metrics.tsv"
    split_path = PROJECT_ROOT / "splits" / "dev_190.json"
    frozen_path = PROJECT_ROOT / "frozen" / "FROZEN_HPARAMS.json"
    existing = json.loads(frozen_path.read_text(encoding="utf-8"))

    rows, _ = freeze_hparams.load_dev_metrics(metrics_path)
    expected_clean = {
        Decimal("0.1"): (Decimal("69.2339"), Decimal("0.7369")),
        Decimal("0.2"): (Decimal("69.1524"), Decimal("0.8321")),
        Decimal("0.3"): (Decimal("68.5412"), Decimal("1.1359")),
        Decimal("0.5"): (Decimal("67.9707"), Decimal("1.5608")),
    }
    for temperature, expected in expected_clean.items():
        summary = freeze_hparams._temperature_summary(rows, temperature)
        assert (summary["clean_mean"], summary["clean_std"]) == expected

    payload, markdown = freeze_hparams.build_frozen_artifacts(
        metrics_path,
        split_path,
        frozen_at=existing["selection_evidence"]["frozen_at"],
    )
    assert payload == existing
    assert payload["E"]["mraf_target_temperature"] == 0.2
    assert "clean R1 mean ± SD" in markdown
    assert "| 0.2 | candidate |" in markdown
    assert "| 69.1524 ± 0.8321 | yes |" in markdown
