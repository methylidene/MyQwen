# Data And Output Layout

## Data

- `data/synthetic_arithmetic/v00_smoke/`: small smoke dataset.
- `data/synthetic_arithmetic/v01_original/`: original formal synthetic arithmetic dataset.
- `data/synthetic_arithmetic/calibration/target_v*/`: calibration variants.
- `data/gsm8k/main/train.jsonl` and `test.jsonl`: canonical `ReasoningExample` JSONL materialized from GSM8K, with per-split manifests.
- `data/prompts/`: inference prompt files.

The Hugging Face cache remains under `data/.cache/hf/`; training and evaluation templates use visible canonical GSM8K JSONL paths, so ordinary runs need not discover hidden cache files.

## Outputs

`outputs/v00_smoke`, `outputs/v01_qwen25_0_5b`, and `outputs/v02_qwen25_3b` identify smoke, 0.5B and 3B result families. Each numbered `expNN_*` directory is one coherent experimental unit. Do not place new files directly beneath `outputs/`.
