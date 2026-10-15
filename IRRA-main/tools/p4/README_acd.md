# Round-nine protocol tools and ACD

Work on `p4/acd-s101`. The existing `train.py`, `processor/processor.py` and
`model/build.py` are unchanged. New training is a separate entry point; exporting
without the new flags keeps the existing base/C0 computation and output bytes.

## Immutable offline protocol

`analysis/calib_protocol.py`, `pack_feats.py` and `analysis/verify_joint_loss.py`
are the supplied files with their original SHA256. Local `.gitattributes` rules
disable line-ending conversion for exactly these three files. Do not edit them.
`branch_mass_check.py` and the supplied outputs are not included.

On Windows the supplied calibration script requires NumPy 2's default int64
indices for `one_hot`; install `requirements_calib.txt` in a compatible dedicated
environment. Linux NumPy 1.x already defaults to int64. Anaconda distributions
with separate NumPy/PyTorch Intel OpenMP runtimes may need the supported
`MKL_THREADING_LAYER=SEQUENTIAL` in the calibration child process only. Do not
use `KMP_DUPLICATE_LIB_OK` or change system-wide runtime settings.

The immutable scripts do not contain complete runtime provenance. Use the
separate wrapper to preserve their source while recording the git commit,
effective argparse defaults/options, seeds, elapsed time, CPU/GPU-memory scope,
stdout and stderr. Its record parent is created only after the script finishes,
so calibration's new-output-directory requirement is retained:

```bash
python -B tools/p4/run_recorded.py --script calib_protocol \
  --record_file "$P4_OUT/calib/acd_s101/tool_run_record.json" -- \
  --feats_dir "$P4_OUT/p4_feats" --models base_clean_s101 \
  --assignments "$B12_ASSIGNMENTS" --skip_rank \
  --out_dir "$P4_OUT/calib/acd_s101"

python -B tools/p4/run_recorded.py --script pack_feats \
  --record_file "$P4_OUT/p4_feats/pack_run_record.json" -- \
  --export_dirs "$P4_OUT/dev_exports/base_clean_s101_feat" \
  --names base_clean_s101 --out_dir "$P4_OUT/p4_feats"

python -B tools/p4/run_recorded.py --script verify_joint_loss \
  --record_file "$P4_OUT/calib/verify/run_record.json"
```

The wrapper only runs those three CPU scripts and hides CUDA in its child;
recorded peak CUDA memory is therefore zero. Calling their original file paths
directly remains supported, with their original (less complete) metadata.

## ACD training

Set `BASE_RUN`, `DATA_ROOT`, `DEV_SPLIT` and `P4_OUT` to actual paths. The base
directory must contain `last.pth` and `configs.yaml`. Its recorded split and the
requested split must have identical identity hashes: 1076 training identities,
190 development identities. A different split with the same counts is rejected.
The historical random sampler, batch 64 and image augmentations are inherited.
Only the loader's MLM masking is disabled; loss module construction is retained
so every base weight loads strictly. No degradation branch is run.

```bash
for V in A0 A0bal A1; do
  python -B tools/p4/train_acd.py \
    --base_run "$BASE_RUN" --root_dir "$DATA_ROOT" \
    --dev_split_file "$DEV_SPLIT" --variant "$V" \
    --epochs 20 --lr 1e-4 --lora_rank 8 --lora_alpha 16 \
    --lora_init_seed 20261015 --data_seed 101 --a1_impl joint \
    --output_dir "$P4_OUT/acd/${V}_s101"
done
```

For a separate smoke output add `--smoke_steps 5`; the production command above
runs all 20 epochs. There is no validation selection: save the last adapter.
All outputs go directly into the supplied new/empty directory.

The frozen base is in eval mode (including dropout). All twelve visual attention
layers receive FP32 q/v LoRA, initialized with a private CPU generator; B is
zero. The explicit context manager is OFF by default and calls the original
attention forward directly. ON computes the effective q/v weights in FP32,
then converts them to the original attention dtype. The remaining visual
operators retain their original precision; base weights are never modified.
This avoids the zero-adapter baseline shift of converting the whole visual
tower to FP32. Both CPU-half and CUDA-half synthetic tests verify bitwise zero
initialization and a learned feature change after five steps at the default LR.

Per step there are four encodings: one frozen text, one frozen RGB, one frozen
IR teacher, and one IR student with LoRA ON. All variants compute the teacher
fusion under `no_grad`; only A1 uses it in the objective. Features/logits and
loss calculations are FP32, with fixed scale 50. Losses use PID-based uniform
multi-positive targets and row means. A1 supports joint and algebraically
decomposed forms, with a detached teacher.

The repository `train.set_seed(data_seed)` initializes the ordinary pipeline.
The reconstructed loader retains the original dataset, transforms and collate,
with its own seeded generator for random sampling/worker seeds. Each worker
seeds Python, NumPy and Torch. LoRA initialization consumes no data RNG. The
three variants consequently have equal step counts and matching first-20 input
hashes, also checked by tests. AdamW uses zero weight decay, one epoch of linear
warmup, cosine decay, and gradient clipping at norm 1.

Outputs: `adapter.pt` (LoRA/config/base-file SHA256 only), `run_record.json`,
`step_log.jsonl`, `step_hashes.txt`, and `train_log.txt`. Runtime records include
all resolved arguments, source hashes, seeds, encoding counts, epoch timings,
elapsed time and peak allocated CUDA memory. An early failed training attempt
also records its error; never reuse that output directory.

## Adapter evaluation and invariance

First export the unadapted checkpoint into a fresh reference directory with
`--save_modality_features`. Use identical checkpoint/config, data, split,
degradation seed and loader batch size for the candidate:

```bash
python -B tools/p4/export_dev_records.py \
  --checkpoint "$BASE_RUN/last.pth" --config_file "$BASE_RUN/configs.yaml" \
  --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
  --eval_degrade_seed 2026 --device cuda --save_modality_features \
  --output_dir "$P4_OUT/dev_exports/acd_base_reference"

python -B tools/p4/export_dev_records.py \
  --checkpoint "$BASE_RUN/last.pth" --config_file "$BASE_RUN/configs.yaml" \
  --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
  --eval_degrade_seed 2026 --device cuda \
  --acd_adapter "$P4_OUT/acd/A1_s101/adapter.pt" \
  --compare_export "$P4_OUT/dev_exports/acd_base_reference" \
  --output_dir "$P4_OUT/dev_exports/acd_A1_s101"
```

Adapter loading verifies the actual base file's SHA256 before injection. Only
`rgb_missing` thermal token encoding enters the ON context; RGB, text, complete
IR, other core scenarios and all six training scenarios remain OFF.
`--acd_adapter` and `--compare_export` implicitly enable feature saving.

`metrics.tsv` retains encoded-blank additive fusion. The extra
`metrics_dropblank.tsv` uses only normalized `f_ir` for `rgb_missing`, only
normalized `f_rgb` for `ir_missing`, and copies the original other scenario
metrics. It is written only with effective feature saving. The opt-in
`export_run_record.json` carries additional provenance without changing old
metrics, feature keys or summary fields.

`invariance.json` compares every feature key recursively, including ids,
strings, container structure, dtype and raw tensor bytes (signed zeros/NaNs
included). Floating values also have maximum absolute differences. Only
`rgb_missing`'s `f_ir`, `f_fused`, reconstruction-error metadata and original
metrics row may change. Any other difference is recorded and raises an error.
The original 11-scenario or complete 17-scenario sets must agree.

## Tests and optional text bank

```bash
python -B -m unittest discover -s tests/p4 -v
python -B tools/p4/train_acd.py --help
python -B tools/p4/export_dev_records.py --help
python -B tools/p4/analysis/calib_protocol.py --help
python -B tools/p4/pack_feats.py --help
```

The six real-checkpoint ACD tests require CUDA plus valid `BASE_RUN`,
`DATA_ROOT`, `DEV_SPLIT`; otherwise each gives an explicit skip reason. CPU
algebra, adapter routing/freezing, input hashes and synthetic CUDA-half tests
do not need real weights. Synthetic successes do not establish real-checkpoint
retrieval performance.

T6 is available without changing training: export all training captions in
original annotation/caption order, retaining contiguous/raw PIDs and indices.
No training image is read. It uses the same strictly loaded frozen text tower
and verifies the recorded/requested development split before exporting:

```bash
python -B tools/p4/export_text_bank.py \
  --checkpoint "$BASE_RUN/last.pth" --config_file "$BASE_RUN/configs.yaml" \
  --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
  --output "$P4_OUT/text_banks/base_clean_s101.npz"
```

The NPZ contains raw/normalized text features, PID/raw PID, image id, annotation
index, caption index and caption. A companion run record stores provenance,
seed, runtime and peak memory. Neither output is overwritten. Optional T5
`--full_row_weight` is not implemented in this change.
