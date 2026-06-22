# Qwen2.5-0.5B Experiment Framework

See [docs/exp_guide.md](docs/exp_guide.md) for commands covering KV-cache inference ablations, synthetic arithmetic SFT, GRPO, and evaluation.

Quick smoke test:

```bash
python scripts/make_synthetic_math.py --output_dir data/smoke_math --num_train 32 --num_val 8 --num_test 8 --seed 42
pytest -q
```
