import json
import logging
import os.path as op
from pathlib import Path
from types import SimpleNamespace

import pytest

from datasets.build import __factory, build_dataloader
from datasets.rgbtpedes import RGBNT201PEDES, RGBTPEDES
from tools.audit.probe_dataset import (
    _baseline_measurements,
    _loader_dynamic_probe,
    _loader_static_audit,
)


RGBNT_ANNOTATION = "captionform0_8B.json"
EVAL_KEYS = {
    "image_pids",
    "img_paths",
    "img_paths_t",
    "caption_pids",
    "captions",
}


def _records():
    return [
        {
            "split": "train",
            "captions": ["train ten", "", " \t"],
            "file_path": "RGBT/train_10.jpg",
            "id": 10,
        },
        {
            "split": "train",
            "captions": ["train thirty"],
            "file_path": "RGBT/train_30.jpg",
            "id": 30,
        },
        {
            "split": "test",
            "captions": ["test twenty", "   "],
            "file_path": "RGBT/test_20.jpg",
            "id": 20,
        },
        {
            "split": "test",
            "captions": ["test forty"],
            "file_path": "RGBT/test_40.jpg",
            "id": 40,
        },
    ]


def _write_dataset(root: Path, dataset_name: str, annotation_name: str):
    data_dir = root / dataset_name
    rgb_dir = data_dir / "RGB"
    thermal_dir = data_dir / "T"
    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir()
    records = _records()
    for record in records:
        name = Path(record["file_path"]).name
        (rgb_dir / name).touch()
        (thermal_dir / name).touch()
    (data_dir / annotation_name).write_text(
        json.dumps(records), encoding="utf-8"
    )
    return data_dir


def test_rgbnt_annotation_candidate_is_exclusive_and_missing_error_is_clear(tmp_path):
    data_dir = _write_dataset(tmp_path, "RGBNT201-PEDES", "Caption.json")
    (data_dir / "captionform0_merged.json").write_text("[]", encoding="utf-8")

    assert RGBNT201PEDES.annotation_filenames == (RGBNT_ANNOTATION,)
    with pytest.raises(RuntimeError, match="captionform0_8B[.]json"):
        RGBNT201PEDES(root=str(tmp_path), verbose=False)


def test_rgbnt_filters_empty_captions_and_preserves_parent_shapes(tmp_path, caplog):
    _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)

    with caplog.at_level(logging.INFO, logger="IRRA.dataset"):
        dataset = RGBNT201PEDES(root=str(tmp_path), verbose=True)

    assert dataset.filtered_empty_captions == 3
    assert dataset.effective_train_captions == 2
    assert [item[-1] for item in dataset.train] == ["train ten", "train thirty"]
    assert all(len(item) == 5 for item in dataset.train)
    assert set(dataset.test) == EVAL_KEYS
    assert set(dataset.val) == EVAL_KEYS
    assert dataset.test["captions"] == ["test twenty", "test forty"]
    assert dataset.val["captions"] == dataset.test["captions"]
    assert len(dataset.test["image_pids"]) == 2
    assert (
        "[RGBNT201-PEDES] filtered_empty_captions=3 "
        "effective_train_captions=2" in caplog.text
    )


def test_rgbnt_relabels_each_split_independently(tmp_path):
    _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)

    dataset = RGBNT201PEDES(root=str(tmp_path), verbose=False)

    assert dataset.train_id_container == {0, 1}
    assert dataset.test_id_container == {0, 1}
    assert dataset.val_id_container == {0, 1}
    assert [item[0] for item in dataset.train] == [0, 1]
    assert dataset.test["image_pids"] == [0, 1]
    assert dataset.test["caption_pids"] == [0, 1]


def test_rgbnt_keeps_gallery_record_when_all_of_its_captions_are_empty(tmp_path):
    data_dir = _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)
    records = _records()
    records.append(
        {
            "split": "test",
            "captions": [" \n\t"],
            "file_path": "RGBT/test_60.jpg",
            "id": 60,
        }
    )
    (data_dir / "RGB" / "test_60.jpg").touch()
    (data_dir / "T" / "test_60.jpg").touch()
    (data_dir / RGBNT_ANNOTATION).write_text(
        json.dumps(records), encoding="utf-8"
    )

    dataset = RGBNT201PEDES(root=str(tmp_path), verbose=False)

    assert dataset.test["image_pids"] == [0, 1, 2]
    assert len(dataset.test["img_paths"]) == 3
    assert len(dataset.test["img_paths_t"]) == 3
    assert dataset.test["caption_pids"] == [0, 1]
    assert dataset.test["captions"] == ["test twenty", "test forty"]


def test_rgbnt_rejects_dev_split_in_direct_and_factory_paths(tmp_path):
    _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)
    message = "RGBNT201-PEDES does not support --dev_split_file in Stage 5"

    with pytest.raises(ValueError, match=message):
        RGBNT201PEDES(
            root=str(tmp_path), verbose=False, dev_split_file="splits/dev.json"
        )

    args = SimpleNamespace(
        dataset_name="RGBNT201-PEDES",
        root_dir=str(tmp_path),
        dev_split_file="splits/dev.json",
        num_workers=0,
    )
    with pytest.raises(ValueError, match=message):
        build_dataloader(args)


def test_rgbnt_is_registered_and_dynamic_audit_uses_loader(tmp_path):
    data_dir = _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)
    repo = Path(__file__).resolve().parents[1]

    assert __factory["RGBNT201-PEDES"] is RGBNT201PEDES
    probe = _loader_dynamic_probe(repo, data_dir, "RGBNT201-PEDES")
    assert probe["status"] == "OK"
    assert "RGBNT201PEDES" in probe["source"]
    assert probe["train_length"] == 2
    assert probe["test_gallery_length"] == 2
    assert probe["test_query_length"] == 2
    assert probe["train_unique_labels"] == 2
    assert probe["test_unique_labels"] == 2
    assert probe["train_labels_contiguous"] is True
    assert probe["test_labels_contiguous"] is True
    assert probe["filtered_empty_captions"] == 3
    assert probe["effective_train_captions"] == 2


def test_rgbnt_confirmed_baseline_includes_effective_training_records():
    stats = {
        "splits": {
            "train": {"captions": 3951, "empty_captions": 3},
            "test": {},
        }
    }

    assert _baseline_measurements(stats)["effective_train_records"] == 3948


def test_static_audit_scopes_filter_detection_to_the_selected_class():
    repo = Path(__file__).resolve().parents[1]

    rgbt = _loader_static_audit(repo, inferred_train_ids=None, dataset_name="RGBT-PEDES")
    rgbnt = _loader_static_audit(
        repo, inferred_train_ids=None, dataset_name="RGBNT201-PEDES"
    )

    assert rgbt["facts"][-1]["status"] == "OK"
    assert rgbt["facts"][-1]["excerpt"] == []
    assert rgbnt["facts"][-1]["status"] == "OK"
    assert rgbnt["facts"][-1]["excerpt"]
    assert "captionform0_8B.json" in rgbnt["facts"][1]["text"]


def test_rgbtpedes_regression_preserves_empty_captions_and_exact_outputs(tmp_path):
    _write_dataset(tmp_path, "RGBT-PEDES", "Caption.json")

    dataset = RGBTPEDES(root=str(tmp_path), verbose=False)
    rgb_dir = op.join(str(tmp_path), "RGBT-PEDES", "RGB")
    thermal_dir = op.join(str(tmp_path), "RGBT-PEDES", "T")
    expected_train = [
        (0, 0, op.join(rgb_dir, "train_10.jpg"), op.join(thermal_dir, "train_10.jpg"), "train ten"),
        (0, 0, op.join(rgb_dir, "train_10.jpg"), op.join(thermal_dir, "train_10.jpg"), ""),
        (0, 0, op.join(rgb_dir, "train_10.jpg"), op.join(thermal_dir, "train_10.jpg"), " \t"),
        (1, 1, op.join(rgb_dir, "train_30.jpg"), op.join(thermal_dir, "train_30.jpg"), "train thirty"),
    ]
    expected_test = {
        "image_pids": [0, 1],
        "img_paths": [
            op.join(rgb_dir, "test_20.jpg"),
            op.join(rgb_dir, "test_40.jpg"),
        ],
        "img_paths_t": [
            op.join(thermal_dir, "test_20.jpg"),
            op.join(thermal_dir, "test_40.jpg"),
        ],
        "caption_pids": [0, 0, 1],
        "captions": ["test twenty", "   ", "test forty"],
    }

    assert dataset.train == expected_train
    assert dataset.test == expected_test
    assert dataset.val == expected_test
    assert dataset.val_annos is dataset.test_annos
    assert dataset.train_id_container == {0, 1}
    assert dataset.test_id_container == {0, 1}
    assert dataset.val_id_container == {0, 1}
