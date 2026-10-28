"""RGBNT201 adapter; the RGBT-PEDES base processing and splitting stay inherited."""
from pathlib import Path
from .rgbtpedes import RGBTPEDES

class RGBNT201PEDES(RGBTPEDES):
    dataset_dir = "RGBNT201"

    def __init__(self, root="", verbose=True, dev_split_file=None):
        import json
        directory = Path(root) / self.dataset_dir
        if not (directory / "RGB").is_dir() or not (directory / "T").is_dir():
            raise ValueError("RGBNT201 requires RGB/ and T/; NI is never read")
        annotations = json.loads((directory / "captionform0_8B.json").read_text(encoding="utf-8"))
        if {a.get("split") for a in annotations} - {"train", "test"}:
            raise ValueError("RGBNT201 split must be train or test")
        if any(not isinstance(a.get("captions"), list) or len(a["captions"]) != 1 or not isinstance(a["captions"][0], str) for a in annotations):
            raise ValueError("Each RGBNT201 annotation must have exactly one existing caption string")
        self.blank_caption_files = [Path(a["file_path"]).name for a in annotations if not a["captions"][0].strip()]
        super().__init__(root=root, verbose=verbose, dev_split_file=dev_split_file)
        print(f"[RGBNT201_BLANK] filtered={len(self.blank_caption_files)} files={self.blank_caption_files}")

    def _resolve_data_dir(self, dataset_dir):
        return dataset_dir

    def _resolve_anno_path(self, data_dir):
        return str(Path(data_dir) / "captionform0_8B.json")

    def _process_anno(self, annos, training=False):
        from tools.p4.r18.common import nonblank_annotations
        # Empty captions produce no train/query sample; every image remains.
        return super()._process_anno(nonblank_annotations(annos), training=training)
