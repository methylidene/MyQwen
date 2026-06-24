"""Experiment lifecycle: reproducible artifacts, safe outputs and self-hosted runners."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.data import DatasetLoadConfig, DatasetRegistry, PromptFormatter
from src.utils.io import append_jsonl, read_jsonl, save_json, write_jsonl
from src.utils.seed import set_seed

from .config import ExperimentConfig, _semantic_dict, dump_resolved_config


@dataclass
class RunManifest:
    experiment_name: str
    run_id: str
    status: str
    utc_start_time: str
    utc_end_time: str | None
    git_commit: str | None
    git_dirty: bool
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    transformers_version: str | None
    trl_version: str | None
    gpu_name: str | None
    model: dict[str, Any]
    dataset: dict[str, Any]
    seed: int | None
    data_seed: int | None
    generation_seed: int | None
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version(package: str) -> str | None:
    try:
        module = __import__(package)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.replace(".yaml", ""))
    return "_".join(part for part in sanitized.split("_") if part) or "experiment"


def _default_stage_name(config: ExperimentConfig) -> str:
    if config.task == "sft":
        model_path = Path(config.model.model_name_or_path)
        return "SFT-continued" if (model_path / "checkpoint").exists() or model_path.name.startswith("SFT-") else "SFT-only"
    if config.task == "grpo":
        return f"GRPO-G{config.grpo.group_size}-{config.grpo.max_steps}"
    if config.task in {"evaluation", "zero_shot"}:
        return "evaluation" if config.task == "evaluation" else "zero-shot"
    return _slug(config.task)


def _readme(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class ExperimentRunner:
    """Execute one validated experiment without silently overwriting artifacts."""

    def __init__(self, config: ExperimentConfig, command: list[str] | None = None) -> None:
        self.config = config
        self.command = command or list(sys.argv)
        self.run_id = config.logging.run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{config.config_hash()[:8]}"
        self.grouped_layout = bool(config.logging.experiment_dir or config.logging.stage_name)
        if self.grouped_layout:
            experiment_dir = config.logging.experiment_dir or _slug(config.name)
            stage_name = config.logging.stage_name or _default_stage_name(config)
            self.experiment_dir = Path(config.logging.output_root) / experiment_dir
            self.stage_dir = self.experiment_dir / stage_name
            self.run_dir = self.stage_dir / self.run_id
        else:
            self.experiment_dir = Path(config.logging.output_root) / config.name
            self.stage_dir = self.experiment_dir
            self.run_dir = self.experiment_dir / self.run_id
        self._started: float | None = None
        self._start_utc: str | None = None
        self._fingerprint: dict[str, Any] | None = None
        self._num_examples = 0

    def command_preview(self) -> str:
        return shlex.join([sys.executable, "-m", "src.experiments.run", "--config", "<config.yaml>"])

    def prepare(self) -> None:
        if self.run_dir.exists() and not self.config.logging.allow_existing:
            raise FileExistsError(f"Refusing to overwrite existing run directory: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "checkpoints").mkdir()
        self._write_readmes()
        write_jsonl([], self.run_dir / "metrics.jsonl")
        write_jsonl([], self.run_dir / "predictions.jsonl")
        dump_resolved_config(self.config, self.run_dir / "resolved_config.yaml")
        (self.run_dir / "command.txt").write_text(shlex.join(self.command) + "\n", encoding="utf-8")
        self._write_git_state()
        self._write_environment()
        self._check_resume_compatibility()
        self._fingerprint_dataset()
        manifest = self._manifest("running", None)
        save_json(manifest.to_dict(), self.run_dir / "run_manifest.json")

    def run(self, *, dry_run: bool = False) -> Path:
        if dry_run:
            return self.run_dir
        self._started = perf_counter()
        self._start_utc = _utc_now()
        try:
            self.prepare()
            set_seed(self.config.seed if self.config.seed is not None else 0)
            if self.config.task == "sft":
                self._run_sft()
            elif self.config.task == "grpo":
                self._run_grpo()
            else:
                self._run_evaluation()
            summary = self._summarize()
            save_json(summary, self.run_dir / "summary.json")
            manifest = self._manifest("completed", _utc_now())
            save_json(manifest.to_dict(), self.run_dir / "run_manifest.json")
            return self.run_dir
        except Exception as exc:
            failure = {"utc_time": _utc_now(), "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            save_json(failure, self.run_dir / "failure.json")
            save_json({"status": "failed", "wall_clock_seconds": (perf_counter() - self._started) if self._started is not None else 0.0, "failure": failure}, self.run_dir / "summary.json")
            save_json(self._manifest("failed", _utc_now()).to_dict(), self.run_dir / "run_manifest.json")
            raise

    def _fingerprint_dataset(self) -> None:
        purpose = "eval" if self.config.task in {"evaluation", "zero_shot"} else "train"
        examples, fingerprint, _ = DatasetRegistry.load(DatasetLoadConfig(
            dataset_name=self.config.dataset.name,
            split=self.config.dataset.split,
            config_name=self.config.dataset.config_name,
            revision=self.config.dataset.revision,
            source_path=self.config.dataset.source_path,
            cache_dir=self.config.dataset.cache_dir,
            max_samples=self.config.dataset.max_samples,
            shuffle=self.config.dataset.shuffle,
            seed=self.config.data_seed if self.config.data_seed is not None else 0,
            purpose=purpose,
        ))
        self._num_examples = len(examples)
        self._fingerprint = fingerprint.to_dict()
        save_json(self._fingerprint, self.run_dir / "dataset_fingerprint.json")

    def _run_sft(self) -> None:
        from src.alignment.sft_trainer import SFTConfig as TrainerConfig, train_sft
        c = self.config
        resume = self._trainer_resume_path()
        trainer = TrainerConfig(
            model_name_or_path=c.model.model_name_or_path,
            model_revision=c.model.revision,
            output_dir=str(self.run_dir / "checkpoints"),
            train_file=c.dataset.source_path,
            learning_rate=c.sft.learning_rate,
            weight_decay=c.sft.weight_decay,
            num_train_epochs=c.sft.num_train_epochs,
            max_steps=c.sft.max_steps,
            batch_size=c.sft.batch_size,
            gradient_accumulation_steps=c.sft.gradient_accumulation_steps,
            max_length=c.sft.max_length,
            use_lora=c.model.use_lora,
            backend_name=c.model.backend_name,
            custom_factory_name=c.model.custom_factory_name,
            device=c.model.device,
            dtype=c.model.dtype,
            trust_remote_code=c.model.trust_remote_code,
            lora_r=c.model.lora_r,
            lora_alpha=c.model.lora_alpha,
            lora_dropout=c.model.lora_dropout,
            lora_target_modules=c.model.lora_target_modules,
            gradient_checkpointing=c.sft.gradient_checkpointing,
            checkpoint_interval=c.sft.checkpoint_interval,
            checkpoint_keep=c.sft.checkpoint_keep,
            resume_from_checkpoint=resume,
            dataset_name=c.dataset.name,
            dataset_split=c.dataset.split,
            dataset_config_name=c.dataset.config_name,
            dataset_revision=c.dataset.revision,
            dataset_cache_dir=c.dataset.cache_dir,
            max_samples=c.dataset.max_samples,
            dataset_seed=c.data_seed if c.data_seed is not None else 0,
            dataset_shuffle=c.dataset.shuffle,
            system_prompt=c.generation.system_prompt,
            final_answer_format=c.generation.final_answer_format,
            use_chat_template=c.generation.use_chat_template,
            deterministic_smoke=c.smoke,
        )
        set_seed(c.generation.seed if c.generation.seed is not None else 0)
        train_sft(trainer)
        self._copy_trainer_metrics()

    def _run_grpo(self) -> None:
        from src.alignment.grpo_trainer import GRPOConfig as TrainerConfig, train_grpo
        c = self.config
        trainer = TrainerConfig(
            model_name_or_path=c.model.model_name_or_path,
            model_revision=c.model.revision,
            output_dir=str(self.run_dir / "checkpoints"),
            train_file=c.dataset.source_path,
            group_size=c.grpo.group_size,
            max_steps=c.grpo.max_steps,
            max_generated_completion_tokens=c.grpo.max_generated_completion_tokens,
            beta_kl=c.grpo.beta_kl,
            clip_eps=c.grpo.clip_eps,
            entropy_coef=c.grpo.entropy_coef,
            advantage_epsilon=c.grpo.advantage_epsilon,
            learning_rate=c.grpo.learning_rate,
            weight_decay=c.grpo.weight_decay,
            temperature=c.generation.temperature,
            top_p=c.generation.top_p,
            max_new_tokens=c.generation.max_new_tokens,
            max_prompt_length=c.generation.max_prompt_length,
            forward_micro_batch_size=c.grpo.forward_micro_batch_size,
            use_reference_policy=c.grpo.use_reference_policy,
            use_lora=c.model.use_lora,
            backend_name=c.model.backend_name,
            custom_factory_name=c.model.custom_factory_name,
            device=c.model.device,
            dtype=c.model.dtype,
            trust_remote_code=c.model.trust_remote_code,
            lora_r=c.model.lora_r,
            lora_alpha=c.model.lora_alpha,
            lora_dropout=c.model.lora_dropout,
            lora_target_modules=c.model.lora_target_modules,
            gradient_checkpointing=c.grpo.gradient_checkpointing,
            checkpoint_interval=c.grpo.checkpoint_interval,
            checkpoint_keep=c.grpo.checkpoint_keep,
            resume_from_checkpoint=self._trainer_resume_path(),
            dataset_name=c.dataset.name,
            dataset_split=c.dataset.split,
            dataset_config_name=c.dataset.config_name,
            dataset_revision=c.dataset.revision,
            dataset_cache_dir=c.dataset.cache_dir,
            max_samples=c.dataset.max_samples,
            dataset_seed=c.data_seed if c.data_seed is not None else 0,
            dataset_shuffle=c.dataset.shuffle,
            system_prompt=c.generation.system_prompt,
            final_answer_format=c.generation.final_answer_format,
            use_chat_template=c.generation.use_chat_template,
            deterministic_smoke=c.smoke,
        )
        set_seed(c.generation.seed if c.generation.seed is not None else 0)
        train_grpo(trainer)
        self._copy_trainer_metrics()
        source = self.run_dir / "checkpoints" / "sampled_responses.jsonl"
        if source.exists():
            write_jsonl(read_jsonl(source), self.run_dir / "predictions.jsonl")

    def _run_evaluation(self) -> None:
        from src.alignment.eval_math import evaluate_with_generator
        from src.inference.kv_cache_generator import KVCacheGenerator
        c = self.config
        checkpoints = c.evaluation.checkpoint_dirs or [c.model.model_name_or_path]
        examples, _, _ = DatasetRegistry.load(DatasetLoadConfig(
            dataset_name=c.dataset.name, split=c.dataset.split, config_name=c.dataset.config_name,
            revision=c.dataset.revision, source_path=c.dataset.source_path, cache_dir=c.dataset.cache_dir,
            max_samples=c.dataset.max_samples, shuffle=c.dataset.shuffle, seed=c.data_seed or 0, purpose="eval",
        ))
        rows: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            model_path = Path(checkpoint)
            candidates = (model_path / "checkpoints" / "checkpoint", model_path / "checkpoint", model_path / "model", model_path)
            resolved_model_path = next((candidate for candidate in candidates if candidate.exists()), model_path)
            generator = KVCacheGenerator.from_pretrained(
                str(resolved_model_path), c.model.device, c.model.dtype,
                c.model.trust_remote_code, c.model.backend_name, c.model.custom_factory_name,
            )
            predictions, metrics = evaluate_with_generator(generator, examples, c.evaluation.max_new_tokens, PromptFormatter(c.generation.system_prompt, c.generation.final_answer_format))
            rows.extend(predictions)
            append_jsonl([{"checkpoint": checkpoint, "metrics": metrics}], self.run_dir / "metrics.jsonl")
        write_jsonl(rows, self.run_dir / "predictions.jsonl")

    def _copy_trainer_metrics(self) -> None:
        source = self.run_dir / "checkpoints" / "train_metrics.jsonl"
        if source.exists():
            write_jsonl(read_jsonl(source), self.run_dir / "metrics.jsonl")

    def _summarize(self) -> dict[str, Any]:
        metrics = read_jsonl(self.run_dir / "metrics.jsonl")
        rollout_tokens = sum(int(row.get("rollout_tokens", 0) or 0) for row in metrics)
        optimization_tokens = sum(int(row.get("train_tokens", 0) or 0) for row in metrics)
        steps = len(metrics)
        number_generations = steps * self.config.grpo.group_size if self.config.task == "grpo" else 0
        peak_vram = 0.0
        if self.config.budget.record_peak_vram and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {
            "status": "completed",
            "number_of_prompts": steps if self.config.task == "grpo" else self._num_examples,
            "number_of_generations": number_generations,
            "generated_completion_tokens": rollout_tokens,
            "optimization_tokens": optimization_tokens,
            "wall_clock_seconds": (perf_counter() - self._started) if self._started is not None else 0.0,
            "peak_vram_mb": peak_vram,
            "dataset_fingerprint": self._fingerprint,
            "config_hash": self.config.config_hash(),
        }

    def _manifest(self, status: str, end: str | None) -> RunManifest:
        gpu = None
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        return RunManifest(
            experiment_name=self.config.name, run_id=self.run_id, status=status,
            utc_start_time=self._start_utc or _utc_now(), utc_end_time=end,
            git_commit=_git(["git", "rev-parse", "HEAD"]), git_dirty=bool(_git(["git", "status", "--porcelain"])),
            python_version=platform.python_version(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
            transformers_version=_version("transformers"), trl_version=_version("trl"), gpu_name=gpu,
            model={"id_or_path": self.config.model.model_name_or_path, "revision": self.config.model.revision, "backend": self.config.model.backend_name},
            dataset={"name": self.config.dataset.name, "config": self.config.dataset.config_name, "split": self.config.dataset.split, "revision": self.config.dataset.revision, "fingerprint": self._fingerprint},
            seed=self.config.seed, data_seed=self.config.data_seed, generation_seed=self.config.generation.seed, config_hash=self.config.config_hash(),
        )

    def _write_git_state(self) -> None:
        status = _git(["git", "status", "--short"]) or "clean"
        commit = _git(["git", "rev-parse", "HEAD"]) or "unavailable"
        (self.run_dir / "git_state.txt").write_text(f"commit: {commit}\n{status}\n", encoding="utf-8")

    def _write_environment(self) -> None:
        lines = [f"python={platform.python_version()}", f"torch={torch.__version__}", f"cuda={torch.version.cuda}"]
        for package in ("transformers", "trl", "datasets", "peft"):
            lines.append(f"{package}={_version(package) or 'not-installed'}")
        if torch.cuda.is_available():
            lines.append(f"gpu={torch.cuda.get_device_name(torch.cuda.current_device())}")
        (self.run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_readmes(self) -> None:
        if self.grouped_layout:
            _readme(self.experiment_dir / "README.md", [
                f"# {self.experiment_dir.name}",
                "",
                "## English",
                "This directory groups one coherent experiment batch. SFT-only, SFT-continued, GRPO variants and evaluation runs that use the same SFT data should stay under this directory.",
                "Use a new numbered experiment directory when the SFT training data changes, and include the data identity in the directory name.",
                "",
                "## 中文",
                "这个目录用于归档同一批完整实验。使用同一份 SFT 数据的 SFT-only、SFT-continued、GRPO 变体和评测结果应放在这里。",
                "如果更换了 SFT 训练数据，请创建新的编号实验目录，并在目录名称中标注数据身份。",
            ])
            _readme(self.stage_dir / "README.md", [
                f"# {self.stage_dir.name}",
                "",
                "## English",
                f"Stage directory for `{self.config.task}` runs in experiment `{self.experiment_dir.name}`.",
                "Each child directory is one immutable runner invocation with resolved config, metrics, predictions and checkpoints.",
                "",
                "## 中文",
                f"这是实验 `{self.experiment_dir.name}` 下的 `{self.config.task}` 阶段目录。",
                "每个子目录对应一次不可变的 runner 调用，包含解析后的配置、指标、预测和 checkpoint。",
            ])
        _readme(self.run_dir / "README.md", [
            f"# {self.config.name}",
            "",
            "## English",
            f"Task: `{self.config.task}`. Run id: `{self.run_id}`.",
            f"Dataset: `{self.config.dataset.name}` / `{self.config.dataset.split}` / `{self.config.dataset.source_path}`.",
            "Artifacts include resolved configuration, manifest, command, Git state, environment, dataset fingerprint, metrics, predictions and checkpoints when produced.",
            "Evaluation reports remain in `report.md` and are intentionally not bilingual.",
            "",
            "## 中文",
            f"任务：`{self.config.task}`。运行 ID：`{self.run_id}`。",
            f"数据集：`{self.config.dataset.name}` / `{self.config.dataset.split}` / `{self.config.dataset.source_path}`。",
            "产物包括解析后的配置、manifest、命令、Git 状态、环境信息、数据指纹、指标、预测，以及训练产生的 checkpoint。",
            "评测报告仍写入 `report.md`，并且有意不做双语输出。",
        ])

    def _check_resume_compatibility(self) -> None:
        resume = self.config.logging.resume_from
        if not resume:
            return
        previous = Path(resume) / "resolved_config.yaml"
        if not previous.exists():
            raise FileNotFoundError(f"resume_from must point to a previous run directory containing resolved_config.yaml: {resume}")
        import yaml
        old = yaml.safe_load(previous.read_text(encoding="utf-8"))
        if _semantic_dict(old) != _semantic_dict(self.config.to_dict()):
            raise ValueError("Resume config is incompatible with the previous run's semantic configuration.")

    def _trainer_resume_path(self) -> str | None:
        if not self.config.logging.resume_from:
            return None
        checkpoints = sorted((Path(self.config.logging.resume_from) / "checkpoints").glob("checkpoint-step-*"))
        if not checkpoints:
            raise FileNotFoundError("No rotating checkpoint found under resume_from/checkpoints.")
        return str(checkpoints[-1])
