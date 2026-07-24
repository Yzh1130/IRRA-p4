# MRAF code audit

## Repository state before implementation

- Worktree: `C:\Users\12823\Desktop\IRRA\IRRA-main-mraf`
- Project directory: `IRRA-main/`
- Branch: `feature/mraf-reliability-fusion`
- HEAD: `5f787e67ca32404fee0256b78829456aed5a9484`
- Initial status: clean
- Original worktree `C:\Users\12823\Desktop\IRRA\IRRA-main` was not read,
  modified, cleaned, or otherwise processed.

## Existing model and fusion path

- Model build entry: `model.build.build_model`
- Main class: `model.build.IRRA`
- Shared RGB/T CLIP encoding:
  - `IRRA._encode_visual_tokens`
  - `IRRA.encode_rgbt_global`
- Existing deterministic addition:
  - `IRRA.encode_rgbt_global` creates the legacy intermediate
    `v_rgb_c + v_t_c`.
  - The effective train/eval fused representation is selected centrally by
    `IRRA._fuse_rgbt_global`.
- Existing uncertainty precision fusion:
  - `IRRA._fuse_rgbt_global`
  - `IRRA._build_eval_precision_weights`
- Retrieval evaluation:
  - `IRRA.encode_rgbt_deterministic`
  - `IRRA.encode_rgbt_eval`
  - `utils.metrics.Evaluator._encode_gallery_feature`
  - `utils.metrics.Evaluator.eval` normalizes text/image features and computes
    `qfeats @ gfeats.t()`.

## Existing MDRC data flow

1. `processor.processor.do_train` runs the normal clean model forward.
2. `_resolve_mdrc_lambda` applies the existing MDRC delayed warmup/ramp.
3. `_compute_mdrc_terms` samples one degraded view with
   `utils.mdrc.degrade_rgbt_batch`.
4. The degraded view calls `model(..., mdrc_feature_only=True)`.
5. `mdrc_feature_consistency` aligns the degraded fused feature to a detached
   clean fused feature.
6. `_add_mdrc_to_total_loss` applies
   `mdrc_lambda * mdrc_loss_scale * loss_mdrc`.
7. The feature-only path does not encode text or compute SDM/MLM/ID.

The supported training degradation modes are `rgb_drop`, `ir_drop`,
`rgb_noise`, `ir_noise`, `rgb_attenuate`, and `ir_attenuate`.

## Configuration, logging, and checkpoint

- CLI/config definitions: `utils/options.py`
- Saved training configuration: `utils.iotools.save_train_configs`
- TensorBoard and epoch/iteration logging: `processor.processor.do_train`
- Checkpoint save/load: `utils.checkpoint.Checkpointer` and
  `utils.checkpoint.load_state_dict`
- Existing checkpoint loading aligns historical keys and then performs a
  strict model load.

## Existing tests and revision tools

- Tests:
  - `tests/test_evaluation_degradation.py`
  - `tests/test_revision_analysis_tools.py`
- Revision tools:
  - `tools/revision/`
- Evaluation degradation implementation:
  - `utils.metrics.Evaluator._apply_eval_degradation`
- Existing degradation runner:
  - `scripts/revision/run_old_degradation_regression.sh`
  - `scripts/revision/run_new_degradation_matrix.sh`

## Implementation scope selected after audit

Modified:

- `model/build.py`
- `processor/processor.py`
- `utils/options.py`
- `utils/mdrc.py`
- `utils/checkpoint.py`

Added:

- `model/mraf.py`
- `tests/test_mraf.py`
- `tools/revision/analyze_mraf_weights.py`
- `tools/revision/summarize_mraf_experiment.py`
- `scripts/mraf/run_mraf_experiments.sh`
- MRAF revision documents

Not modified:

- CLIP backbone and tokenizer
- dataset contents or split definitions
- SDM, MLM, and ID objectives
- existing checkpoint, logs, and experiment results
- paper DOCX/PDF files

## MDRC scale finding

The effective formal MDRC scale could not be established from available local
evidence. See `docs/revision/mraf_mdrc_scale_audit.md`. The code default remains
`1.0`; formal MDRC-related scripts require an explicit verified environment
variable.
