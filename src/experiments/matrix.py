"""Serial matrix runner for reproducible experiment variants."""
from __future__ import annotations

import argparse
import itertools
import sys

from .config import load_experiment_config
from .runner import ExperimentRunner


def parse_axis(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise ValueError("Matrix axis must use dotted.path=v1,v2.")
    path, raw = value.split("=", 1)
    values = [item for item in raw.split(",") if item]
    if not values:
        raise ValueError("Matrix axis requires at least one value.")
    return path, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix", action="append", default=[], help="dotted.path=v1,v2")
    parser.add_argument("--checkpoint", action="append", default=[], help="Add evaluation.checkpoint_dirs variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    axes = [parse_axis(item) for item in args.matrix]
    if args.checkpoint:
        import json
        axes.append(("evaluation.checkpoint_dirs", [json.dumps([path]) for path in args.checkpoint]))
    combinations = itertools.product(*(values for _, values in axes)) if axes else [()]
    failures = 0
    for values in combinations:
        overrides = [f"{path}={value}" for (path, _), value in zip(axes, values)]
        config = load_experiment_config(args.config, overrides)
        runner = ExperimentRunner(config, [sys.executable, "-m", "src.experiments.matrix", "--config", args.config, *sum((["--matrix", f"{path}={','.join(items)}"] for path, items in axes), [])])
        print(f"{runner.command_preview()} {' '.join('--set ' + item for item in overrides)}")
        if args.dry_run:
            continue
        try:
            runner.run()
        except Exception:
            failures += 1
            if not args.continue_on_error:
                raise
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
