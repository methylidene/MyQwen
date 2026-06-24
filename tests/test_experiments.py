from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.config import experiment_from_dict, load_experiment_config
from src.experiments.runner import ExperimentRunner


def mapping(tmp_path: Path) -> dict:
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"id": "train-0", "prompt": "1+1", "answer": "2", "difficulty": "easy"}) + "\n", encoding="utf-8")
    return {
        "name": "unit-experiment",
        "task": "sft",
        "smoke": False,
        "seed": 1,
        "data_seed": 2,
        "model": {"model_name_or_path": "tiny", "backend_name": "custom", "device": "cpu", "dtype": "fp32", "use_lora": False},
        "dataset": {"name": "synthetic_arithmetic", "split": "train", "source_path": str(source), "shuffle": False},
        "generation": {"seed": 3},
        "logging": {"output_root": str(tmp_path / "runs"), "run_id": "fixed"},
    }


def test_config_validation_requires_formal_seeds(tmp_path):
    value = mapping(tmp_path)
    value["generation"]["seed"] = None
    with pytest.raises(ValueError, match="Formal experiments require"):
        experiment_from_dict(value)


def test_config_overrides_and_unknown_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    import yaml
    yaml.safe_dump(mapping(tmp_path), config_path.open("w", encoding="utf-8"))
    config = load_experiment_config(config_path, ["sft.max_steps=3"])
    assert config.sft.max_steps == 3
    with pytest.raises(ValueError, match="Unknown override path"):
        load_experiment_config(config_path, ["sft.missing=1"])


def test_runner_prepare_writes_required_artifacts_and_refuses_overwrite(tmp_path):
    runner = ExperimentRunner(experiment_from_dict(mapping(tmp_path)), ["python", "-m", "test"])
    runner.prepare()
    expected = {
        "resolved_config.yaml", "run_manifest.json", "command.txt", "git_state.txt", "environment.txt",
        "dataset_fingerprint.json", "metrics.jsonl", "summary.json", "predictions.jsonl", "checkpoints",
    }
    # summary is completed later; create contract is checked by runner completion tests, all other artifacts exist now.
    assert expected - {"summary.json"} <= {item.name for item in runner.run_dir.iterdir()}
    with pytest.raises(FileExistsError):
        runner.prepare()


def test_resume_compatibility_rejects_semantic_change(tmp_path):
    import yaml
    initial = experiment_from_dict(mapping(tmp_path))
    previous = tmp_path / "previous"
    previous.mkdir()
    yaml.safe_dump(initial.to_dict(), (previous / "resolved_config.yaml").open("w", encoding="utf-8"))
    changed = mapping(tmp_path)
    changed["logging"] = {"output_root": str(tmp_path / "new"), "run_id": "new", "resume_from": str(previous)}
    changed["sft"] = {"learning_rate": 9e-5}
    runner = ExperimentRunner(experiment_from_dict(changed))
    with pytest.raises(ValueError, match="incompatible"):
        runner.prepare()


def test_evaluation_cannot_use_train_split(tmp_path):
    value = mapping(tmp_path)
    value["task"] = "evaluation"
    with pytest.raises(ValueError, match="cannot use the train split"):
        experiment_from_dict(value)
