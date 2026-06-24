"""CLI for one YAML-defined reproducible experiment."""
from __future__ import annotations

import argparse
import sys

from .config import load_experiment_config
from .runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.path=value; written into resolved_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.overrides)
    runner = ExperimentRunner(config, [sys.executable, "-m", "src.experiments.run", "--config", args.config, *sum((["--set", value] for value in args.overrides), [])])
    if args.dry_run:
        print(runner.command_preview())
        print(f"run_dir={runner.run_dir}")
        return
    print(runner.run())


if __name__ == "__main__":
    main()
