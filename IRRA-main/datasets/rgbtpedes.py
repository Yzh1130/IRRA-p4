import os.path as op
from typing import List, Tuple

from utils.iotools import read_json
from .bases import BaseDataset


class RGBTPEDES(BaseDataset):
    """
    RGBT-PEDES

    The annotation file stores one relative path per sample (e.g. "RGBT/1_1.jpg").
    We map it to a registered RGB-T pair with the same basename:
      RGB/<name>.jpg and T/<name>.jpg.
    """

    dataset_dir = "RGBT-PEDES"

    def __init__(self, root="", verbose=True, dev_split_file=None):
        super(RGBTPEDES, self).__init__()

        # dataset root, e.g. <root>/RGBT-PEDES
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.data_dir = self._resolve_data_dir(self.dataset_dir)
        self.rgb_dir = op.join(self.data_dir, "RGB")
        self.t_dir = op.join(self.data_dir, "T")
        self.anno_path = self._resolve_anno_path(self.data_dir)
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(
            self.anno_path, dev_split_file=dev_split_file
        )
        self.train, self.train_id_container = self._process_anno(
            self.train_annos, training=True
        )
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RGBT-PEDES Images and Captions are loaded")
            self.show_dataset_info()

    def _resolve_data_dir(self, dataset_dir: str) -> str:
        # Support both layouts:
        #   <root>/RGBT-PEDES/RGBT-PEDES/...
        #   <root>/RGBT-PEDES/RGBT-PEDES-Final/...
        # and the flat one:
        #   <root>/RGBT-PEDES/...
        candidates = [
            op.join(dataset_dir, "RGBT-PEDES-Final"),
            op.join(dataset_dir, "RGBT-PEDES"),
            dataset_dir,
        ]
        for cand in candidates:
            if op.isdir(op.join(cand, "RGB")) and op.isdir(op.join(cand, "T")):
                return cand
        return dataset_dir

    def _resolve_anno_path(self, data_dir: str) -> str:
        for fname in ("Caption.json", "caption.json"):
            path = op.join(data_dir, fname)
            if op.exists(path):
                return path
        # Return default path for clearer downstream error message.
        return op.join(data_dir, "Caption.json")

    def _split_anno(self, anno_path: str, dev_split_file=None):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        dev_ids = None
        if dev_split_file is not None:
            split_data = read_json(dev_split_file)
            if not isinstance(split_data, dict) or "dev_ids" not in split_data:
                raise ValueError(
                    "dev split file must be a JSON object containing 'dev_ids'."
                )
            dev_ids = {int(pid) for pid in split_data["dev_ids"]}
            official_train_ids = {
                int(anno["id"])
                for anno in annos
                if anno.get("split", "train") == "train"
            }
            official_test_ids = {
                int(anno["id"])
                for anno in annos
                if anno.get("split", "train") == "test"
            }
            unknown_ids = sorted(dev_ids - official_train_ids)
            test_overlap = sorted(dev_ids & official_test_ids)
            if unknown_ids or test_overlap:
                raise ValueError(
                    "dev_ids must be drawn only from the official train split; "
                    f"unknown_train_ids={unknown_ids}, test_overlap={test_overlap}."
                )
        for anno in annos:
            split = anno.get("split", "train")
            if split == "train":
                if dev_ids is not None and int(anno["id"]) in dev_ids:
                    val_annos.append(anno)
                else:
                    train_annos.append(anno)
            elif split == "test":
                test_annos.append(anno)
            else:
                val_annos.append(anno)

        # This dataset usually has train/test only.
        if dev_ids is None and len(val_annos) == 0:
            val_annos = test_annos
        return train_annos, test_annos, val_annos

    def _pair_paths(self, rel_path: str) -> Tuple[str, str]:
        name = op.basename(rel_path)
        rgb_path = op.join(self.rgb_dir, name)
        t_path = op.join(self.t_dir, name)
        return rgb_path, t_path

    def _process_anno(self, annos: List[dict], training=False):
        # Re-index pids to contiguous labels inside each split.
        raw_pids = sorted({int(anno["id"]) for anno in annos})
        pid2label = {pid: idx for idx, pid in enumerate(raw_pids)}

        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = pid2label[int(anno["id"])]
                pid_container.add(pid)
                rgb_path, t_path = self._pair_paths(anno["file_path"])

                for caption in anno["captions"]:
                    # (pid, image_id, rgb_path, thermal_path, caption)
                    dataset.append((pid, image_id, rgb_path, t_path, caption))
                image_id += 1

            return dataset, pid_container

        dataset = {}
        img_paths = []
        img_paths_t = []
        captions = []
        image_pids = []
        caption_pids = []
        for anno in annos:
            pid = pid2label[int(anno["id"])]
            pid_container.add(pid)
            rgb_path, t_path = self._pair_paths(anno["file_path"])

            image_pids.append(pid)
            img_paths.append(rgb_path)
            img_paths_t.append(t_path)
            for caption in anno["captions"]:
                captions.append(caption)
                caption_pids.append(pid)

        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "img_paths_t": img_paths_t,
            "caption_pids": caption_pids,
            "captions": captions,
        }
        return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not op.exists(self.rgb_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_dir))
        if not op.exists(self.t_dir):
            raise RuntimeError("'{}' is not available".format(self.t_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
