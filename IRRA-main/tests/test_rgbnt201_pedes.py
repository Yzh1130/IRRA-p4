import csv
import hashlib
import json
import logging
import os.path as op
from pathlib import Path
from types import SimpleNamespace

import pytest

from datasets.build import __factory, build_dataloader
from datasets.rgbtpedes import RGBNT201PEDES, RGBTPEDES
from tools.protocol import make_dev2_split, measure_precondition
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


def _write_official_scale_rgbnt_annotation(path: Path):
    """Write the confirmed 4787-row schema without using the private dataset."""
    rows = []
    empty_caption_ids = {31, 82, 108}
    for pid in range(1, 172):
        record_count = 24 if pid <= 18 else 23
        for image_index in range(record_count):
            caption = f"train identity {pid} image {image_index}"
            if pid in empty_caption_ids and image_index == 0:
                caption = " \t "
            rows.append(
                {
                    "split": "train",
                    "captions": [caption],
                    "file_path": f"RGBT/train_{pid:03d}_{image_index:02d}.jpg",
                    "id": pid,
                    "processed_tokens": [],
                    "rgb_captions": [],
                    "tir_captions": [],
                }
            )
    for offset, pid in enumerate(range(172, 202), start=1):
        record_count = 28 if offset <= 26 else 27
        for image_index in range(record_count):
            rows.append(
                {
                    "split": "test",
                    "captions": [f"test identity {pid} image {image_index}"],
                    "file_path": f"RGBT/test_{pid:03d}_{image_index:02d}.jpg",
                    "id": pid,
                    "processed_tokens": [],
                    "rgb_captions": [],
                    "tir_captions": [],
                }
            )
    assert len(rows) == 4787
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _write_precondition_metrics(path: Path, *, group="A"):
    scenario_r1 = {
        "clean": (60, 60, 60),
        "rgb_missing": (20, 21, 22),
        "ir_missing": (40, 42, 44),
        "rgb_noise_005": (57, 57, 57),
        "rgb_noise_010": (54, 54, 54),
        "ir_noise_005": (54, 54, 54),
        "ir_noise_010": (48, 48, 48),
        "rgb_blur_k7": (50, 50, 50),
        "ir_blur_k7": (49, 49, 49),
        "both_noise_005": (47, 47, 47),
        "both_blur_k7": (46, 46, 46),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(measure_precondition.METRICS_HEADER)
        for seed_index, seed in enumerate(measure_precondition.SEEDS):
            for scenario, spec in measure_precondition.SCENARIO_SPECS.items():
                r1 = scenario_r1[scenario][seed_index]
                writer.writerow(
                    [
                        group,
                        seed,
                        scenario,
                        spec.mode,
                        spec.strength,
                        spec.kernel,
                        spec.degradation_seed,
                        "val",
                        r1,
                        min(r1 + 5, 100),
                        min(r1 + 10, 100),
                        r1 - 2,
                        r1 - 3,
                    ]
                )


def test_rgbnt_annotation_candidate_is_exclusive_and_missing_error_is_clear(tmp_path):
    data_dir = _write_dataset(tmp_path, "RGBNT201-PEDES", "Caption.json")
    (data_dir / "captionform0_merged.json").write_text("[]", encoding="utf-8")

    assert RGBNT201PEDES.annotation_filenames == (RGBNT_ANNOTATION,)
    with pytest.raises(RuntimeError, match="captionform0_8B[.]json"):
        RGBNT201PEDES(root=str(tmp_path), verbose=False)


def test_make_dev2_split_official_scale_invariants_and_reproducibility(tmp_path):
    annotation_path = tmp_path / RGBNT_ANNOTATION
    rows = _write_official_scale_rgbnt_annotation(annotation_path)
    output_path = tmp_path / "rgbnt201_dev2_30.json"

    first = make_dev2_split.make_dev2_split(
        anno_path=annotation_path,
        out_path=output_path,
        n_dev=30,
        seed=20260907,
    )
    first_md5 = hashlib.md5(output_path.read_bytes()).hexdigest()
    second = make_dev2_split.make_dev2_split(
        anno_path=annotation_path,
        out_path=output_path,
        n_dev=30,
        seed=20260907,
    )
    second_md5 = hashlib.md5(output_path.read_bytes()).hexdigest()

    test_ids = {
        int(row["id"]) for row in rows if row["split"] == "test"
    }
    assert first == second
    assert first_md5 == second_md5
    assert first["anno_md5"] == hashlib.md5(annotation_path.read_bytes()).hexdigest()
    assert first["n_dev"] == len(first["dev_ids"]) == 30
    assert first["n_train_reduced"] == len(first["train_reduced_ids"]) == 141
    assert first["n_dev"] + first["n_train_reduced"] == 171
    assert set(first["dev_ids"]).isdisjoint(test_ids)
    assert set(first["dev_ids"]).isdisjoint(first["train_reduced_ids"])
    assert first["dev_records"] + first["train_reduced_records"] == 3951
    assert (
        first["dev_effective_captions"]
        + first["train_reduced_effective_captions"]
        == 3948
    )
    assert "141/30/30" in make_dev2_split.__doc__


def test_make_dev2_split_rejects_any_train_test_identity_overlap(tmp_path):
    annotation_path = tmp_path / RGBNT_ANNOTATION
    rows = _write_official_scale_rgbnt_annotation(annotation_path)
    first_test_row = next(row for row in rows if row["split"] == "test")
    first_test_row["id"] = 1
    annotation_path.write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(
        make_dev2_split.SplitValidationError,
        match="train/test identity sets overlap",
    ):
        make_dev2_split.make_dev2_split(
            anno_path=annotation_path,
            out_path=tmp_path / "should_not_exist.json",
            n_dev=30,
            seed=20260907,
        )


def test_measure_precondition_validates_locked_grid_and_writes_auditable_report(
    tmp_path,
):
    metrics_path = tmp_path / "metrics.tsv"
    output_path = tmp_path / "EXTERNAL_PRECONDITION.md"
    _write_precondition_metrics(metrics_path)

    assert measure_precondition.main(
        ["--metrics", str(metrics_path), "--out", str(output_path)]
    ) == 0

    report_bytes = output_path.read_bytes()
    report = report_bytes.decode("utf-8")
    expected_md5 = hashlib.md5(metrics_path.read_bytes()).hexdigest()
    assert b"\r" not in report_bytes
    assert "Measured utility ratio `û_rgb / û_ir`: 2.0000" in report
    assert "Sensitivity and utility orderings are opposite: **yes**" in report
    assert "| RGBT-PEDES | test | 3.80 |" in report
    assert "| RGBT-PEDES | development | 2.68 |" in report
    assert f"Metrics MD5: `{expected_md5}`" in report

    invalid_path = tmp_path / "wrong_group.tsv"
    _write_precondition_metrics(invalid_path, group="D")
    with pytest.raises(
        measure_precondition.PreconditionValidationError,
        match="group must be 'A'",
    ):
        measure_precondition.load_precondition_metrics(invalid_path)


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


def test_rgbnt_supports_dev_split_and_preserves_test_in_direct_and_factory_paths(
    tmp_path,
):
    _write_dataset(tmp_path, "RGBNT201-PEDES", RGBNT_ANNOTATION)
    split_path = tmp_path / "rgbnt_dev.json"
    split_path.write_text(
        json.dumps({"dev_ids": [10], "n_dev": 1}), encoding="utf-8"
    )

    baseline = RGBNT201PEDES(root=str(tmp_path), verbose=False)
    empty_string_baseline = RGBNT201PEDES(
        root=str(tmp_path), verbose=False, dev_split_file=""
    )
    with_dev = RGBNT201PEDES(
        root=str(tmp_path),
        verbose=False,
        dev_split_file=str(split_path),
    )

    assert baseline.val_annos is baseline.test_annos
    assert empty_string_baseline.train == baseline.train
    assert empty_string_baseline.test == baseline.test
    assert empty_string_baseline.val == baseline.val
    assert empty_string_baseline.val_annos is empty_string_baseline.test_annos

    assert with_dev.val_annos is not with_dev.test_annos
    assert {int(row["id"]) for row in with_dev.val_annos} == {10}
    assert {int(row["id"]) for row in with_dev.train_annos} == {30}
    assert with_dev.test_annos == baseline.test_annos
    assert with_dev.test == baseline.test
    assert with_dev.val["captions"] == ["train ten"]
    assert [row[-1] for row in with_dev.train] == ["train thirty"]
    assert all(len(row) == 5 for row in with_dev.train)

    args = SimpleNamespace(
        dataset_name="RGBNT201-PEDES",
        root_dir=str(tmp_path),
        dev_split_file=str(split_path),
        num_workers=0,
        training=False,
        img_size=(32, 16),
        img_aug=False,
        test_batch_size=2,
        text_length=77,
        eval_dataset="val",
    )
    val_img_loader, val_txt_loader, _ = build_dataloader(args)
    assert len(val_img_loader.dataset) == 1
    assert len(val_txt_loader.dataset) == 1
    assert val_img_loader.dataset.img_paths_rgb == with_dev.val["img_paths"]
    assert val_txt_loader.dataset.captions == ["train ten"]


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

    assert __factory["RGBT-PEDES"] is RGBTPEDES
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
