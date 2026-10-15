"""Synthetic text-only export checks; no dataset or checkpoint is needed."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tools.p4.export_text_bank import encode_text_batches, main, training_caption_records


class TextBankTests(unittest.TestCase):
    def test_training_annotation_caption_order_and_raw_ids(self):
        annotations = [{"id": 9, "captions": ["a", "b"], "split": "train"},
                       {"id": 12, "captions": ["dev"], "split": "train"},
                       {"id": 7, "captions": ["c"], "split": "train"},
                       {"id": 20, "captions": ["test"], "split": "test"}]
        rows = training_caption_records(annotations, [7, 9])
        self.assertEqual([row["caption"] for row in rows], ["a", "b", "c"])
        self.assertEqual([row["pid"] for row in rows], [1, 1, 0])
        self.assertEqual([row["raw_pid"] for row in rows], [9, 9, 7])
        self.assertEqual([row["image_id"] for row in rows], [0, 0, 1])
        self.assertEqual([row["annotation_index"] for row in rows], [0, 0, 2])

    def test_encoding_is_no_grad_ordered_and_normalized(self):
        class Encoder(torch.nn.Module):
            def encode_text_eval(self, value):
                self_grad_flags.append(torch.is_grad_enabled())
                return value.float() * 2
        self_grad_flags = []
        loader = [(torch.tensor([2, 2]), torch.tensor([[3., 4.], [0., 2.]])),
                  (torch.tensor([0]), torch.tensor([[2., 0.]]))]
        raw, normalized, pids = encode_text_batches(Encoder(), loader, "cpu")
        self.assertEqual(pids, [2, 2, 0])
        self.assertEqual(self_grad_flags, [False, False])
        np.testing.assert_array_equal(raw, [[6, 8], [0, 4], [4, 0]])
        np.testing.assert_allclose(normalized, [[.6, .8], [0, 1], [1, 0]])

    def test_existing_output_rejected_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bank.npz"
            output.write_bytes(b"previous output")
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", "missing", "--config_file", "missing", "--root_dir", "missing",
                      "--dev_split_file", "missing", "--output", str(output)])
            self.assertEqual(output.read_bytes(), b"previous output")
