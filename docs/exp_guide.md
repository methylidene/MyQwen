# Qwen2.5-0.5B Inference And Alignment Experiments

This repository contains a compact framework for controlled Qwen2.5-0.5B inference ablations and GRPO-style arithmetic alignment.

## Synthetic Arithmetic Data

```bash
python scripts/make_synthetic_math.py \
  --output_dir data/synthetic_math \
  --num_train 10000 \
  --num_val 1000 \
  --num_test 1000 \
  --seed 42
```

Smoke:

```bash
python scripts/make_synthetic_math.py --output_dir data/smoke_math --num_train 32 --num_val 8 --num_test 8 --seed 42
```

## KV Cache Ablation

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
python scripts/run_kv_cache_ablation.py --model_name_or_path Qwen/Qwen2.5-0.5B --output_dir outputs/smoke_inference --max_new_tokens 16 --cache_windows -1 32 --dtype bf16
```

Outputs include `config.json`, `results.jsonl`, `summary.csv`, `metrics.json`, and `report.md`.

## SFT

```bash
python scripts/run_sft.py \
  --config configs/sft_only.yaml \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --train_file data/synthetic_math/train.jsonl \
  --val_file data/synthetic_math/val.jsonl \
  --output_dir outputs/checkpoints/SFT-only
```

```bash
python scripts/run_sft.py \
  --config configs/sft_continued.yaml \
  --model_name_or_path outputs/checkpoints/SFT-only \
  --train_file data/synthetic_math/train.jsonl \
  --val_file data/synthetic_math/val.jsonl \
  --output_dir outputs/checkpoints/SFT-continued
```

## GRPO

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g4_1000.yaml \
  --model_name_or_path outputs/checkpoints/SFT-only \
  --train_file data/synthetic_math/train.jsonl \
  --val_file data/synthetic_math/val.jsonl \
  --output_dir outputs/checkpoints/GRPO-G4-1000
```

```bash
python scripts/run_grpo.py \
  --config configs/grpo_g8_250.yaml \
  --model_name_or_path outputs/checkpoints/SFT-only \
  --train_file data/synthetic_math/train.jsonl \
  --val_file data/synthetic_math/val.jsonl \
  --output_dir outputs/checkpoints/GRPO-G8-250
```

GRPO samples `group_size` responses per prompt, computes rule rewards, normalizes group-relative advantages, and uses a frozen reference model for approximate token-level KL.

## Unified Evaluation

```bash
python scripts/run_alignment_eval.py \
  --test_file data/synthetic_math/test.jsonl \
  --checkpoint_dirs \
    outputs/checkpoints/SFT-only \
    outputs/checkpoints/SFT-continued \
    outputs/checkpoints/GRPO-G4-1000 \
    outputs/checkpoints/GRPO-G8-250 \
  --output_dir outputs/alignment_eval \
  --max_new_tokens 128
```

The evaluator writes per-experiment predictions, metrics by difficulty, a combined `summary.csv`, and `report.md`.
