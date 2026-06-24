# Experiment Protocol

## English

### One YAML, One Run

Run every experiment through `python -m src.experiments.run --config ...`. The runner creates a fresh run directory and writes `README.md`, `resolved_config.yaml`, manifest, command, Git state, environment, dataset fingerprint, metrics, summary, predictions and `checkpoints/` before or during execution.

For grouped experiments, set `logging.output_root`, `logging.experiment_dir`, and `logging.stage_name`. The output path is:

```text
<output_root>/<experiment_dir>/<stage_name>/<run-id>/
```

Use one `experiment_dir` for a coherent batch: SFT-only, SFT-continued, GRPO variants and evaluation runs that share the same SFT data. If the SFT data changes, create the next numbered experiment directory and include the data identity in the name.

Formal configurations must set `seed`, `data_seed`, and `generation.seed`. Smoke templates live under `configs/experiments/smoke/`; they are separate from formal configurations and must not be promoted by editing only `max_steps`.

### Fair G4 vs G8 Comparisons

A fixed number of optimizer steps is not a fair GRPO budget by itself. At each prompt, G4 produces four completions while G8 produces eight; therefore G8 normally consumes roughly twice the rollout generations and completion tokens.

Report and compare all of the following:

- number of prompts;
- number of generations;
- generated completion tokens;
- optimization tokens;
- wall-clock time;
- peak VRAM;
- group size, generation temperature/top-p/max tokens, beta KL and clipping.

Use either an equal-rollout budget, an equal-generated-token budget, or explicitly label results as equal-step only. Do not claim a group-size improvement from equal-step runs without reporting the additional rollout budget.

### Resume

`logging.resume_from` points to a prior runner directory. Before loading a rotating checkpoint, the runner compares semantic resolved configuration: model, dataset, generation, trainer and budget settings must match. Logging location/run id/resume metadata are excluded. A mismatch fails before training.

### Matrix Runs

Use serial matrix mode for seed, beta and group-size sweeps:

```bash
python -m src.experiments.matrix \
  --config configs/experiments/qwen25_3b_gsm8k_grpo_g4.yaml \
  --matrix seed=101,102,103 \
  --matrix grpo.beta_kl=0.0,0.005,0.02 \
  --matrix grpo.group_size=4,8 \
  --dry-run
```

`--dry-run` prints the resolved execution variants only. Without it, variants run serially. `--continue-on-error` records failures and continues remaining variants.

## 中文

### 一个 YAML，对应一次运行

所有实验都通过 `python -m src.experiments.run --config ...` 运行。runner 会创建新的运行目录，并在运行前或运行中写入 `README.md`、`resolved_config.yaml`、manifest、命令、Git 状态、环境信息、数据指纹、指标、summary、预测结果和 `checkpoints/`。

对于分组实验，请设置 `logging.output_root`、`logging.experiment_dir` 和 `logging.stage_name`。输出路径为：

```text
<output_root>/<experiment_dir>/<stage_name>/<run-id>/
```

同一批实验共用一个 `experiment_dir`：使用同一份 SFT 数据的 SFT-only、SFT-continued、GRPO 变体和评测运行都放在这里。如果更换 SFT 数据，请创建下一个编号实验目录，并在目录名中写出数据身份。

正式实验必须设置 `seed`、`data_seed` 和 `generation.seed`。Smoke 模板位于 `configs/experiments/smoke/`，它们与正式配置分开，不能只改 `max_steps` 就提升为正式实验。

### 公平比较 G4 与 G8

固定优化步数本身并不是公平的 GRPO 预算。每个 prompt 下，G4 生成四个 completion，G8 生成八个 completion；因此 G8 通常会消耗约两倍的 rollout generation 和 completion token。

请报告并比较以下信息：

- prompt 数量；
- generation 数量；
- 生成的 completion token 数；
- 用于 policy loss 的 optimization token 数；
- wall-clock time；
- peak VRAM；
- group size、temperature、top-p、max tokens、beta KL 和 clipping。

可以采用等 rollout 预算、等生成 token 预算，或明确标注为 equal-step only。若没有报告额外 rollout 预算，不要声称 group size 带来了提升。

### 断点恢复

`logging.resume_from` 指向一个旧 runner 目录。加载轮转 checkpoint 前，runner 会比较语义配置：model、dataset、generation、trainer 和 budget 必须一致。日志位置、run id 和 resume 元数据不参与比较。不匹配会在训练前失败。

### 矩阵运行

seed、beta 和 group-size sweep 使用串行 matrix 模式：

```bash
python -m src.experiments.matrix \
  --config configs/experiments/qwen25_3b_gsm8k_grpo_g4.yaml \
  --matrix seed=101,102,103 \
  --matrix grpo.beta_kl=0.0,0.005,0.02 \
  --matrix grpo.group_size=4,8 \
  --dry-run
```

`--dry-run` 只打印解析后的运行变体。去掉它后会串行运行所有变体。`--continue-on-error` 会记录失败并继续剩余变体。
