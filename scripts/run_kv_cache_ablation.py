#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import time
from pathlib import Path

from src.inference.generation_utils import add_generation_args, peak_gpu_memory_mb
from src.inference.kv_cache_generator import KVCacheGenerator
from src.inference.metrics import compare_token_outputs, summarize_ablation, write_ablation_report
from src.utils.io import read_jsonl, save_json, write_csv, write_jsonl
from src.utils.seed import set_seed


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return [
            "Solve 12 + 7. Use <reasoning>...</reasoning> and <answer>...</answer>.",
            "Problem: (23 + 18) * 2 - 7. Return reasoning and answer tags.",
        ]
    rows = read_jsonl(path)
    return [x.get("prompt", str(x)) for x in rows]


def main() -> None:
    p = add_generation_args(argparse.ArgumentParser())
    p.add_argument("--prompt_file")
    p.add_argument("--cache_windows", nargs="*", type=int, default=[-1, 128, 256, 512])
    args = p.parse_args()
    set_seed(args.seed)
    prompts = load_prompts(args.prompt_file)
    gen = KVCacheGenerator.from_pretrained(args.model_name_or_path, args.device, args.dtype, args.trust_remote_code)
    rows = []
    references = []
    for prompt in prompts:
        ref = gen.generate(prompt, args.max_new_tokens, use_cache=False, cache_window=None)[0]
        references.append(ref.generated_token_ids)
        rows.append(result_row(ref, "no-cache", True, args))
    for window in args.cache_windows:
        cache_window = None if window <= 0 else window
        for prompt, ref_tokens in zip(prompts, references):
            res = gen.generate(prompt, args.max_new_tokens, use_cache=True, cache_window=cache_window)[0]
            cmp = compare_token_outputs(ref_tokens, res.generated_token_ids)
            rows.append(result_row(res, res.mode, cmp["exact_match"], args, cmp))
    summary = summarize_ablation(rows)
    config = vars(args) | {"num_prompts": len(prompts), "created_at": time.time()}
    out = Path(args.output_dir)
    save_json(config, out / "config.json")
    write_jsonl(rows, out / "results.jsonl")
    write_csv(summary, out / "summary.csv")
    save_json({"summary": summary}, out / "metrics.json")
    write_ablation_report(config, summary, out / "report.md")


def result_row(res, mode: str, exact_match: bool, args, cmp: dict | None = None) -> dict:
    n = max(len(res.generated_token_ids), 1)
    return {
        "mode": mode,
        "prompt": res.prompt,
        "total_latency": res.total_latency,
        "prefill_latency": res.prefill_latency,
        "decode_latency": sum(res.step_latencies),
        "avg_latency_per_token": res.total_latency / n,
        "tokens_per_second": n / max(res.total_latency, 1e-9),
        "peak_gpu_memory_mb": peak_gpu_memory_mb(args.device),
        "generated_length": len(res.generated_token_ids),
        "output_text": res.generated_text,
        "generated_token_ids": res.generated_token_ids,
        "exact_output_match_with_no_cache": exact_match,
        "cache_rebuild_count": res.cache_rebuild_count,
        "avg_cache_seq_len": res.avg_cache_seq_len,
        "max_cache_seq_len": res.max_cache_seq_len,
        "diff": (cmp or {}).get("first_difference"),
    }


if __name__ == "__main__":
    main()
