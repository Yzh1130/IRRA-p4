# R10: A1 repulsion-rule controls

Branch `p4/r10-mech`, based on
`ee70023798e68a082f571775a5f23a4f25684d97`. No changes to LoRA, base freezing,
image/text encoders, data augmentation, sampler, optimizer, schedule, clipping,
or evaluation/packing/calibration paths.

## Objectives

For each text row, let `q` be the uniform same-PID target, `b=LSE(z_I)`,
`a=LSE(stopgrad(z_F))`, and `p_I=softmax(z_I)`. Row objectives are averaged.
Every variant retains the same `uniform_ce(z_I.T, pids)` i2t term.

| Variant | t2i positive coefficient | t2i repulsion |
|---|---:|---|
| A0 (unchanged) | 1 | `1 * p_I` |
| A0bal (unchanged) | 0.5 | `0.5 * p_I` |
| A1 (unchanged) | 0.5 | `sigmoid(b-a) * p_I` |
| Pull | 0.5 | None |
| A1const | 0.5 | `sigmoid(b-ref_mean_b64) * p_I` |
| A1shuf | 0.5 | `sigmoid(b-a[permutation]) * p_I` |
| DCLbal | 0.5 | `0.5 * softmax(z_I restricted to different-PID columns)` |

Pull is `-0.5 * sum(q*z_I)`. A1const/A1shuf add `softplus(b-reference)`;
the implementation uses stable `logaddexp(0, b-reference)` so FP64 gradients
retain the exact sigmoid even above PyTorch softplus's default threshold.
DCLbal adds half the different-PID logsumexp. All positive columns are excluded,
including the diagonal, other images with the same identity, and repeated-image
columns. A row with no negatives retains only its pull term; no all-`-inf`
logsumexp is evaluated.

Teacher logits are detached before forming row references and before row
permutation. The constant is a Python float converted to the student's dtype,
without autograd. Teacher features are already encoded under `no_grad` in the
unchanged training pipeline. Pull/DCLbal do not use teacher values in their
objective; the existing frozen teacher is still computed for diagnostics.

New variants log `t2i_repulsion_coeff_mean` on the original logging schedule.
DCLbal also logs integer `t2i_no_negative_rows`; its coefficient is the nominal
0.5 applying only to existing negatives. Other diagnostics retain their
original meanings, including `pi_I=sigmoid(b-a)`; use the new coefficient field
for the modified rule.

Original A0/A0bal/A1 loss operations and diagnostics dictionaries are unchanged.
Their default step-log fields are also unchanged. Add the default-OFF
`--mechanism_diagnostics` flag to log the new coefficient for those variants.
This flag does not change the objective, gradients, adapters or input stream.
For a legacy run with all new options left at defaults, even the argument
record keeps its original fields. New variants and explicit mechanism options
record all new parameters; git/time metadata naturally belongs to the current run.

## Fixed teacher reference (separate process)

Use the same `BASE_RUN`, `DATA_ROOT`, `DEV_SPLIT` and data seed as training.
The tool reuses `prepare_training`'s strict checkpoint, original loader,
1076/190 identity-split and batch-64 checks. LoRA is injected by that shared
preparation but remains OFF; all model parameters are frozen. There is no
optimizer, student encoding, backward pass or development evaluation.

```bash
python -B tools/p4/teacher_ref_stats.py \
  --base_run "$BASE_RUN" --root_dir "$DATA_ROOT" \
  --dev_split_file "$DEV_SPLIT" --data_seed 101 \
  --output_json "$P4_OUT/ref_stats/base_clean_s101_d101.json"
```

The default traverses one full training epoch. Logits and per-row logsumexp
follow the existing FP32 teacher computation exactly; aggregation is CPU FP64.
`ref_mean_b64` averages only rows from batches of exactly 64. `ref_mean_all`,
median, population standard deviation and 5/25/75/95 percentiles use all rows,
including any tail batch. Median uses linear quantile interpolation. The
positive-logit statistic first takes each row's uniform-positive mean and then
averages across rows. Row counts, batch counts, batch sizes and full-epoch
status make the scope explicit. An epoch without any 64-row batch has
`ref_mean_b64=null` and cannot supply A1const.

For a diagnostic smoke output use `--num_batches 3` and a different filename.
It is marked incomplete if the loader has more than three batches. A1const
rejects JSON explicitly marked `complete_epoch=false` or
`valid_as_full_epoch_reference=false`, as well as missing/nonfinite references.
Never use smoke statistics as the production constant.

The JSON records git, all CLI/resolved preparation parameters, seed, base and
split provenance, elapsed time and peak allocated CUDA memory. It never
overwrites an existing output. Run it in its own process so its one-epoch
loader traversal cannot advance a training run's data stream. Global RNG state
is additionally saved/restored by the collector and CLI.

## Training

Only `--variant` changes the t2i objective. All variants retain A1's four
encodings per step: frozen text/RGB/IR teacher and routed IR student. Keep the
same base, data seed, LoRA initialization seed, epochs and LR settings.

```bash
python -B tools/p4/train_acd.py \
  --base_run "$BASE_RUN" --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
  --variant A1const --ref_stats_json "$P4_OUT/ref_stats/base_clean_s101_d101.json" \
  --epochs 20 --lr 1e-4 --lora_rank 8 --lora_alpha 16 \
  --lora_init_seed 20261015 --data_seed 101 \
  --output_dir "$P4_OUT/acd_r10/A1const_s101"

for V in Pull A1shuf DCLbal; do
  python -B tools/p4/train_acd.py \
    --base_run "$BASE_RUN" --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
    --variant "$V" --shuffle_seed 20261017 \
    --epochs 20 --lr 1e-4 --lora_rank 8 --lora_alpha 16 \
    --lora_init_seed 20261015 --data_seed 101 \
    --output_dir "$P4_OUT/acd_r10/${V}_s101"
done
```

A1shuf creates one independent CPU `torch.Generator(shuffle_seed)` for the run.
It advances once per step via `torch.randperm(B, generator=...)`, across epoch
boundaries, without using global Torch/NumPy/Python RNG. The teacher row
reference is detached before indexing. The first 20 permutations' SHA1 hashes
are stored as ordered `{step, sha1}` entries in `teacher_permutation_sha1`;
the hash input is the contiguous CPU int64 permutation's bytes.

A1const records the loaded float plus the full JSON, absolute source path and
SHA256. Every variant keeps the existing `step_hashes.txt` format and logging
schedule. The original default `--a1_impl joint` remains; the new objectives
are defined directly and do not alter the old joint/decomposed paths.

## Verification

```bash
python -B -m unittest discover -s tests/p4 -v
python -B tools/p4/train_acd.py --help
python -B tools/p4/teacher_ref_stats.py --help
```

`test_acd_mech.py` checks FP64 closed-form gradients, identical i2t gradients,
A1's invariance to within-row teacher permutations, its sensitivity to teacher
row offsets, private RNG, all-positive DCL rows, reference statistics, seven
variants' first-20 input hashes, and trusted `git show ee700237` parity for the
three legacy loss/gradient/diagnostic paths and fixed-clock training artifacts.

Three new real-checkpoint tests require CUDA and valid `BASE_RUN`, `DATA_ROOT`,
`DEV_SPLIT`; otherwise they explicitly skip. They cover 20-step hashes,
constant coefficients loaded from JSON, and a three-batch teacher smoke run.
These tests are available for the server; this implementation task runs only
synthetic models locally and does not start real training.
