# MGA-RDE Training Usage

This document describes the train-time MGA + RDE/DURA-style robust training
switches added for RGBT-PEDES text-to-RGBT person retrieval. All switches are
default-off. Evaluation/test code does not consume pseudo captions.

## Default-Off Guarantees

Without MGA/RDE/CONF-HN switches:

- dataloader does not load `--mga_pseudo_json`;
- model does not encode `mga_*_ids`;
- processor returns no MGA/RDE auxiliary loss terms;
- original human-caption training loss is unchanged;
- `test.py`, QALR v3, and QALR v4 gate logic are not used by these train-time
  switches.

Switch independence:

- `--use_mga_pseudo` loads and batches pseudo fields, but pseudo loss is added
  only when `--lambda_mga_pseudo_sdm > 0`.
- `--use_mga_attr_align` does not load pseudo JSON by itself. Use it together
  with `--use_mga_pseudo` and a JSONL file when attribute text supervision is
  desired.
- `--use_rde_robust` does not enable MGA by itself and only adjusts
  pseudo/attribute auxiliary weights when those losses are active.
- `--use_confidence_hard_negative` is independent. Current hard-negative losses
  use original caption features, so this switch logs pseudo confidence
  eligibility and safely skips masking when no confidence is available.

## Main Switches

`--use_mga_pseudo`

Enable structured pseudo caption auxiliary supervision during training. The
original human caption remains the main caption and is not replaced.

`--mga_pseudo_json PATH`

Path to the structured pseudo JSONL file. If empty or missing, training logs a
warning and continues without pseudo auxiliary fields.

`--mga_min_weight`

Minimum pseudo text weight used when sampling pseudo/attribute text candidates.

`--mga_pseudo_sample_prob`

Probability of sampling a pseudo full/short/attribute text for each train
sample. This only creates auxiliary batch fields.

`--mga_attr_sample_prob`

Probability of sampling RGB color / thermal shape attribute texts for each train
sample.

`--use_mga_attr_align`

Enable decoupled RGB/T attribute alignment. RGB visual features align to
`rgb_color_texts`; thermal visual features align to `thermal_shape_texts`.

`--use_mga_negative_loss`

Enable reserved negative/absence text margin loss.

`--lambda_mga_pseudo_sdm`

Weight for fused visual feature to pseudo text alignment.

`--lambda_mga_rgb_color`

Weight for RGB visual feature to RGB color text alignment.

`--lambda_mga_t_shape`

Weight for thermal visual feature to thermal shape text alignment.

`--lambda_mga_negative`

Weight for negative/absence margin loss.

`--mga_neg_margin`

Margin for the negative/absence text loss.

`--use_rde_robust`

Enable RDE/DURA-style robust weighting for MGA pseudo/attribute auxiliary
supervision. It does not down-weight the original human-caption base loss.

`--rde_warmup_epochs`

Number of 1-based epochs before robust pseudo/attribute auxiliary supervision
becomes active.

`--rde_enable_pseudo_during_warmup`

Keep pseudo/attribute auxiliary losses active during RDE warmup. Default is off,
so RDE warmup uses zero effective weight for pseudo/attribute auxiliary losses.

`--rde_conf_source {mga_weight,online_sim,hybrid}`

Confidence source for robust weighting.

`--lambda_rde_consensus`

Weight for consistency between original-caption visual similarity and
high-confidence pseudo-caption visual similarity.

`--use_confidence_hard_negative`

Enable confidence-aware hard-negative bookkeeping. Current hard-negative
branches use original caption features, so this switch logs pseudo confidence
eligibility (`conf_hn_active_ratio`) and does not alter existing hard-negative
losses.

## Pseudo JSONL Format

Each line is one JSON object. Keys are matched by `image_id` first, then RGB path
basename, then `caption_hash`.

```json
{
  "pid": 1,
  "image_id": 123,
  "rgb_path": "RGB/000123.jpg",
  "thermal_path": "T/000123.jpg",
  "caption": "a person wearing a dark jacket",
  "caption_hash": "optional_hash",
  "pseudo_texts": [
    {
      "text": "a person in a dark jacket and light pants",
      "kind": "full",
      "clip_score": 0.82,
      "caption_consensus": 0.76,
      "attr_confidence": 0.90,
      "weight": 0.84,
      "source": "external"
    }
  ],
  "rgb_color_texts": [
    {"text": "dark upper clothing and light lower clothing", "weight": 0.90}
  ],
  "thermal_shape_texts": [
    {"text": "upright pedestrian silhouette with backpack shape", "weight": 0.82}
  ],
  "negative_texts": [
    {"text": "not wearing a red coat", "weight": 0.70}
  ]
}
```

Supported `kind` values: `full`, `short`, `long`, `attribute`, `rgb_color`,
`thermal_shape`, `object`, `negative`.

Validate a file with:

```bash
python tools/validate_mga_pseudo_json.py --json path/to/pseudo.jsonl --show_examples 5
```

## Four-Experiment Screening

Run:

```bash
ROOT_DIR=/path/to/dataset_parent_or_RGBT-PEDES \
PSEUDO_JSON=/path/to/mga_pseudo.jsonl \
OUT_TAG=my_screening \
bash run_mga_rde_screening_4exp_20ep.sh
```

`ROOT_DIR` may point either to the parent containing `RGBT-PEDES/` or directly to
the `RGBT-PEDES` directory. The script normalizes it for the current dataset
loader.

The script writes outputs to:

```text
pack/mga_rde_screening_${OUT_TAG}/
```

Experiments:

`E0_control_current_best_20ep`

Control run. No MGA and no RDE. The script default baseline follows the recent
RGBT-PEDES GU screening command style. If your current best has extra prototype,
USPC, or DCAlign arguments, override via `BASE_CONFIG`.

`E1_mga_pseudo_only_20ep`

Adds pseudo-caption alignment only:

```text
--use_mga_pseudo --lambda_mga_pseudo_sdm 0.30
```

`E2_mga_attr_align_20ep`

Adds RGB/T attribute alignment:

```text
--use_mga_attr_align
--lambda_mga_rgb_color 0.10
--lambda_mga_t_shape 0.10
```

`E3_mga_attr_rde_20ep`

Adds robust weighting and consensus:

```text
--use_rde_robust
--rde_warmup_epochs 10
--rde_conf_source hybrid
--lambda_rde_consensus 0.20
--use_confidence_hard_negative
--conf_hn_min_weight 0.50
```

## Recommended Interpretation

`E1 > E0`: pseudo captions are useful.

`E2 > E1`: RGB/T attribute disentangled alignment is useful.

`E3 > E2`: robust weighting is useful.

If `E3` reaches 58+ R1 at 20 epochs, run a 60-epoch version.

## Recommended Parameters

First 20-epoch screening:

```bash
ROOT_DIR=/path/to/dataset_parent_or_RGBT-PEDES \
PSEUDO_JSON=/path/to/mga_pseudo.jsonl \
OUT_TAG=mga_rde_20ep_screening \
bash run_mga_rde_screening_4exp_20ep.sh
```

If `E3` is promising, run the 60-epoch robust setting:

```bash
python train.py \
  --name E3_mga_attr_rde_60ep \
  --dataset_name RGBT-PEDES \
  --root_dir /path/to/dataset_parent \
  --img_aug \
  --MLM \
  --loss_names "sdm+mlm" \
  --batch_size 16 \
  --test_batch_size 512 \
  --num_workers 8 \
  --num_epoch 60 \
  --warmup_epochs 1 \
  --log_period 100 \
  --eval_period 1 \
  --use_rgbt_add_fusion \
  --use_global_uncertainty \
  --uncertainty_dist_type diag_gaussian \
  --use_precision_fusion \
  --uncertainty_loss_type w2 \
  --lambda_unc 0.09 \
  --lambda_tr 1.10 \
  --lambda_rt 0.90 \
  --lambda_tt 0.0 \
  --lambda_var_reg 1e-4 \
  --eval_use_fused_mu \
  --use_mga_pseudo \
  --mga_pseudo_json /path/to/mga_pseudo.jsonl \
  --use_mga_attr_align \
  --use_rde_robust \
  --rde_warmup_epochs 10 \
  --rde_conf_source hybrid \
  --lambda_mga_pseudo_sdm 0.30 \
  --lambda_mga_rgb_color 0.10 \
  --lambda_mga_t_shape 0.10 \
  --lambda_rde_consensus 0.20 \
  --use_confidence_hard_negative \
  --conf_hn_min_weight 0.50 \
  --save_metric_best_ckpt \
  --save_full_eval_state_ckpt
```

## BASE_CONFIG Override

`BASE_CONFIG` is an optional bash fragment sourced by the run script. Use it to
override `COMMON_ARGS` or append arrays such as:

```bash
EXTRA_COMMON_ARGS+=(--use_proto_calib --proto_version v2)
EXTRA_E0_ARGS+=(--some_control_only_arg value)
```

This keeps the training code unchanged while still allowing local best-command
reuse.
