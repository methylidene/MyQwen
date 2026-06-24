# Experiment Protocol

## One YAML, One Run

Run every experiment through `python -m src.experiments.run --config ...`.
The runner creates a fresh `outputs/experiments/<name>/<run-id>/` directory and
writes `resolved_config.yaml`, manifest, command, Git state, environment,
dataset fingerprint, metrics, summary, predictions and `checkpoints/` before
or during execution. Existing run directories are rejected unless an explicit
non-default recovery workflow is implemented.

Formal configurations must set `seed`, `data_seed`, and `generation.seed`.
Smoke templates live under `configs/experiments/smoke/`; they are separate from
formal configurations and must not be promoted by editing only `max_steps`.

## Fair G4 vs G8 Comparisons

A fixed number of optimizer steps is not a fair GRPO budget by itself. At each
prompt, G4 produces four completions while G8 produces eight; therefore G8
normally consumes roughly twice the rollout generations and completion tokens.

Report and compare all of the following:

- number of prompts;
- number of generations;
- generated completion tokens;
- optimization tokens (completion tokens used in policy loss);
- wall-clock time;
- peak VRAM;
- group size, generation temperature/top-p/max tokens, beta KL and clipping.

Use either an equal-rollout budget (same prompts times group size), an
equal-generated-token budget, or explicitly label results as equal-step only.
Do not claim a group-size improvement from equal-step runs without reporting
the additional rollout budget.

## Resume

`logging.resume_from` points to a prior runner directory. Before loading a
rotating checkpoint, the runner compares semantic resolved configuration:
model, dataset, generation, trainer and budget settings must match. Logging
location/run id/resume metadata are excluded. A mismatch fails before training.

## Matrix Runs

Use serial matrix mode for seed, beta and group-size sweeps:

```bash
python -m src.experiments.matrix \
  --config configs/experiments/qwen25_3b_gsm8k_grpo_g4.yaml \
  --matrix seed=101,102,103 \
  --matrix grpo.beta_kl=0.0,0.005,0.02 \
  --matrix grpo.group_size=4,8 \
  --dry-run
```

`--dry-run` prints the resolved execution variants only. Without it, variants
run serially. `--continue-on-error` records failures and continues remaining
variants.
