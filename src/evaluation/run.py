"""CLI for unified evaluation from DatasetRegistry and ModelBackend generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import DatasetLoadConfig, DatasetRegistry, PromptFormatter
from src.inference.kv_cache_generator import KVCacheGenerator
from src.utils.seed import set_seed

from .evaluator import EvaluationGenerationConfig, Evaluator, KVGenerationBackend, write_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-window", type=int, default=-1)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--final-answer-format", default="<answer>{answer}</answer>")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--backend-name", default="huggingface")
    parser.add_argument("--custom-factory-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.split == "train":
        raise ValueError("Evaluator refuses train split; choose benchmark test or validation explicitly.")
    set_seed(args.generation_seed)
    examples, fingerprint, _ = DatasetRegistry.load(DatasetLoadConfig(
        dataset_name=args.dataset, split=args.split, config_name=args.dataset_config, revision=args.dataset_revision,
        source_path=args.source_path, cache_dir=args.cache_dir, max_samples=args.max_samples, seed=args.data_seed, purpose="eval",
    ))
    checkpoint = Path(args.checkpoint)
    candidates = (checkpoint / "checkpoints" / "checkpoint", checkpoint / "checkpoint", checkpoint / "model", checkpoint)
    model_path = next((item for item in candidates if item.exists()), checkpoint)
    generator = KVCacheGenerator.from_pretrained(str(model_path), args.device, args.dtype, args.trust_remote_code, args.backend_name, args.custom_factory_name)
    generation = EvaluationGenerationConfig(args.max_new_tokens, args.num_generations, args.use_cache, None if args.cache_window <= 0 else args.cache_window, args.generation_seed)
    evaluator = Evaluator(KVGenerationBackend(generator), PromptFormatter(args.system_prompt, args.final_answer_format), checkpoint_id=args.checkpoint_id or checkpoint.name, seed=args.generation_seed)
    predictions = evaluator.evaluate(examples, generation)
    write_evaluation_outputs(args.output_dir, predictions, checkpoint, {"arguments": vars(args), "dataset_fingerprint": fingerprint.to_dict()})


if __name__ == "__main__":
    main()
