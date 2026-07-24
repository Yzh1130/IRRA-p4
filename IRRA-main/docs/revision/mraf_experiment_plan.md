# MRAF experiment plan

## Evidence gate for MDRC scale

The current audit conclusion is C: the effective formal MDRC
`mdrc_loss_scale` is unconfirmed. The CLI default remains `1.0`, but this
default is not treated as formal-experiment evidence.

Before any MDRC+MRAF run, an operator must inspect the historical formal MDRC
training directory, launch command, or saved `configs.yaml`, then export:

```bash
export MDRC_LOSS_SCALE=<verified effective value>
```

The experiment script exits before creating a run when this variable is
missing for an MDRC case.

## Stage 1

Run only:

1. `mraf_only`, seed 1
2. `mdrc_mraf`, seed 1

Prepared but not part of the first comparison:

- `mdrc_mraf_norel`, seed 1

Launcher:

```bash
bash scripts/mraf/run_mraf_experiments.sh --case mraf_only
MDRC_LOSS_SCALE=<verified> \
  bash scripts/mraf/run_mraf_experiments.sh --case mdrc_mraf
```

The launcher records Git HEAD, complete command, fixed seed, configuration,
and effective MDRC scale in a timestamped independent directory. It runs in the
foreground and can be wrapped with `nohup`; it does not use `screen`.

## Evaluation outputs

For the best checkpoint, record:

- best epoch and checkpoint path;
- clean R1/R5/R10, mAP, and mINP;
- R1/mAP for clean plus ten degraded scenarios;
- mean absolute R1 change and mean degraded R1;
- RGB missing, IR missing, RGB noise 0.10, IR noise 0.10, and both blur k7;
- parameter count, per-batch inference time, and total evaluation time;
- per-scenario MRAF gate statistics.

Gate statistics are generated with:

```bash
python tools/revision/analyze_mraf_weights.py \
  --config-file <configs.yaml> \
  --checkpoint <best.pth> \
  --output-prefix <result-dir>/mraf_weights
```

Experiment summaries are generated from an explicit manifest:

```bash
python tools/revision/summarize_mraf_experiment.py \
  --manifest <completed-runs.json> \
  --output-dir <result-dir>
```

Missing metrics remain null/NA and are never synthesized.

## Gate for extending to five formal seeds

Proceed only if stage 1 shows:

1. no clear clean R1 regression;
2. mean degraded R1 not below the verified MDRC baseline;
3. improvement in at least one RGB-missing or RGB-noise condition;
4. weights respond in the expected direction to the damaged modality;
5. no persistent single-modality collapse;
6. stable training and inference without NaN/Inf.

No formal training was run during implementation.
