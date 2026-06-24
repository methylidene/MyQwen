# Qwen2.5 Experiment Framework

## English

See [docs/exp_guide.md](docs/exp_guide.md) for commands covering KV-cache inference ablations, SFT, GRPO, and evaluation.

Quick smoke test:

```bash
python scripts/make_synthetic_math.py --output_dir data/smoke_math --num_train 32 --num_val 8 --num_test 8 --seed 42
pytest -q
```

## 中文

实验命令请参考 [docs/exp_guide.md](docs/exp_guide.md)，其中包含 KV-cache 推理消融、SFT、GRPO 和评测流程。

快速 smoke test：

```bash
python scripts/make_synthetic_math.py --output_dir data/smoke_math --num_train 32 --num_val 8 --num_test 8 --seed 42
pytest -q
```
