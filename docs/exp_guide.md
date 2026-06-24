# Qwen2.5 Inference And Alignment Experiments

## English

This repository contains a compact framework for controlled Qwen2.5 inference ablations and GRPO-style arithmetic alignment.

### Synthetic Arithmetic Data

```bash
python scripts/make_synthetic_math.py \
  --output_dir data/synthetic_arithmetic/v01_original \
  --num_train 10000 \
  --num_val 1000 \
  --num_test 1000 \
  --seed 42
```

Smoke:

```bash
python scripts/make_synthetic_math.py --output_dir data/synthetic_arithmetic/v00_smoke --num_train 32 --num_val 8 --num_test 8 --seed 42
```

### KV Cache Ablation

```bash
python scripts/run_kv_cache_ablation.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --prompt_file data/prompts.jsonl \
  --output_dir outputs/inference_ablation \
  --max_new_tokens 128 \
  --cache_windows -1 128 256 512 \
  --dtype bf16
```

Smoke:

```bash
python scripts/run_kv_cache_ablation.py --model_name_or_path Qwen/Qwen2.5-0.5B --output_dir outputs/v00_smoke/exp01_kv_cache/kv_cache --max_new_tokens 16 --cache_windows -1 32 --dtype bf16
```

Outputs include `config.json`, `results.jsonl`, `summary.csv`, `metrics.json`, and `report.md`.

### Manual SFT/GRPO Batch Layout

Keep one coherent SFT/GRPO/evaluation batch under one numbered experiment directory. For GSM8K main SFT data, use:

```text
outputs/v02_qwen25_3b/exp01_gsm8k_main/
```

If the SFT data changes, start a new directory and name the data, for example:

```text
outputs/v02_qwen25_3b/exp02_synthetic_arithmetic_v01_original/
```

### SFT

```bash
python scripts/run_sft.py \
  --config configs/sft_only.yaml \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only
```

```bash
python scripts/run_sft.py \
  --config configs/sft_continued.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-continued
```

### GRPO

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g4_1000.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G4-1000
```

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g8_250.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G8-250
```

GRPO samples `group_size` responses per prompt, computes rule rewards, normalizes group-relative advantages, and uses a frozen reference model for approximate token-level KL.

### Unified Evaluation

```bash
python scripts/run_alignment_eval.py \
  --test_file data/gsm8k/main/test.jsonl \
  --checkpoint_dirs \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-continued \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G4-1000 \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G8-250 \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/evaluation \
  --max_new_tokens 128
```

The evaluator writes per-experiment predictions, metrics by difficulty, a combined `summary.csv`, and `report.md`. Reports remain English-only by design.

### YAML Runner

For YAML-driven runs, set `logging.output_root`, `logging.experiment_dir`, and `logging.stage_name`. Example output path:

```text
outputs/v02_qwen25_3b/exp01_gsm8k_main/SFT-only/<run-id>/
```

Run with:

```bash
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_sft.yaml
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_grpo_g4.yaml
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_grpo_g8.yaml
```

## 中文

这个仓库提供一个紧凑框架，用于可控的 Qwen2.5 推理消融和 GRPO 风格算术对齐实验。

### 合成算术数据

```bash
python scripts/make_synthetic_math.py \
  --output_dir data/synthetic_arithmetic/v01_original \
  --num_train 10000 \
  --num_val 1000 \
  --num_test 1000 \
  --seed 42
```

Smoke：

```bash
python scripts/make_synthetic_math.py --output_dir data/synthetic_arithmetic/v00_smoke --num_train 32 --num_val 8 --num_test 8 --seed 42
```

### KV Cache 消融

```bash
python scripts/run_kv_cache_ablation.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --prompt_file data/prompts.jsonl \
  --output_dir outputs/inference_ablation \
  --max_new_tokens 128 \
  --cache_windows -1 128 256 512 \
  --dtype bf16
```

Smoke：

```bash
python scripts/run_kv_cache_ablation.py --model_name_or_path Qwen/Qwen2.5-0.5B --output_dir outputs/v00_smoke/exp01_kv_cache/kv_cache --max_new_tokens 16 --cache_windows -1 32 --dtype bf16
```

输出包括 `config.json`、`results.jsonl`、`summary.csv`、`metrics.json` 和 `report.md`。

### 手工 SFT/GRPO 批次布局

同一批 SFT/GRPO/评测产物应放在同一个编号实验目录下。GSM8K main SFT 数据使用：

```text
outputs/v02_qwen25_3b/exp01_gsm8k_main/
```

如果更换 SFT 数据，请创建新目录并在目录名中标注数据，例如：

```text
outputs/v02_qwen25_3b/exp02_synthetic_arithmetic_v01_original/
```

### SFT

```bash
python scripts/run_sft.py \
  --config configs/sft_only.yaml \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only
```

```bash
python scripts/run_sft.py \
  --config configs/sft_continued.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-continued
```

### GRPO

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g4_1000.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G4-1000
```

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g8_250.yaml \
  --model_name_or_path outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
  --train_file data/gsm8k/main/train.jsonl \
  --val_file data/gsm8k/main/test.jsonl \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G8-250
```

GRPO 会对每个 prompt 采样 `group_size` 个回答，计算规则奖励，进行组内 advantage 归一化，并使用冻结 reference model 估计 token 级 KL。

### 统一评测

```bash
python scripts/run_alignment_eval.py \
  --test_file data/gsm8k/main/test.jsonl \
  --checkpoint_dirs \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-only \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/SFT-continued \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G4-1000 \
    outputs/v02_qwen25_3b/exp01_gsm8k_main/checkpoints/GRPO-G8-250 \
  --output_dir outputs/v02_qwen25_3b/exp01_gsm8k_main/evaluation \
  --max_new_tokens 128
```

评测脚本会写出每个实验的预测、按 difficulty 分组的指标、合并的 `summary.csv` 和 `report.md`。`report.md` 按设计保持英文，不做双语输出。

### YAML Runner

YAML 驱动的运行请设置 `logging.output_root`、`logging.experiment_dir` 和 `logging.stage_name`。示例输出路径：

```text
outputs/v02_qwen25_3b/exp01_gsm8k_main/SFT-only/<run-id>/
```

运行命令：

```bash
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_sft.yaml
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_grpo_g4.yaml
python -m src.experiments.run --config configs/experiments/qwen25_3b_gsm8k_grpo_g8.yaml
```
