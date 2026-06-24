from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from tqdm import tqdm

from src.alignment.kl_utils import (
    approximate_kl,
    clipped_policy_loss,
    normalize_group_advantages,
    sequence_logprobs,
    token_entropy,
)
from src.alignment.training_core import (
    AlignmentConfig,
    BaseAlignmentTrainer,
    GRPOBatch,
    ModelGenerationBackend,
    OptimizerConfig,
    RewardPipeline,
    RolloutBatch,
    RuleBasedRewardPipeline,
    TrainingMetrics,
)
from src.data import DatasetLoadConfig, DatasetRegistry, PromptFormatter, ReasoningExample
from src.models.backend import ModelBackend, ModelBackendRegistry, ModelInputs, ModelLoadConfig
from src.utils.io import append_jsonl, ensure_dir, save_json, write_bilingual_readme


@dataclass
class GRPOConfig:
    model_name_or_path: str
    output_dir: str
    train_file: str | None = None
    val_file: str | None = None
    group_size: int = 4
    max_steps: int = 1000
    beta_kl: float = 0.02
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    advantage_epsilon: float = 1e-6
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 128
    max_prompt_length: int = 384
    batch_size: int = 1
    forward_micro_batch_size: int = 4
    use_reference_policy: bool = True
    use_lora: bool = True
    backend_name: str = "huggingface"
    model_revision: str | None = None
    custom_factory_name: str | None = None
    device: str = "cuda"
    trust_remote_code: bool = True
    dtype: str = "bf16"
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    checkpoint_interval: int = 0
    checkpoint_keep: int = 2
    resume_from_checkpoint: str | None = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    dataset_name: str = "synthetic_arithmetic"
    dataset_split: str = "train"
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    dataset_cache_dir: str | None = None
    max_samples: int | None = None
    dataset_seed: int = 42
    dataset_shuffle: bool = False
    system_prompt: str | None = None
    final_answer_format: str = "<answer>{answer}</answer>"
    use_chat_template: bool = False
    deterministic_smoke: bool = False


class GRPOTrainerEngine(BaseAlignmentTrainer):
    """Self-hosted GRPO with explicit rollout, old-policy and update roles.

    ``policy`` receives gradients. ``reference_policy`` is optional and frozen.
    ``old_policy`` is represented by rollout log probabilities captured before
    any optimizer update; it is never recomputed after an update.
    """

    def __init__(
        self,
        policy: ModelBackend,
        config: GRPOConfig,
        output_dir: str | Path,
        reference_policy: ModelBackend | None = None,
        generation_backend: Any | None = None,
        reward_pipeline: RewardPipeline | None = None,
    ) -> None:
        alignment = AlignmentConfig(
            seed=config.dataset_seed,
            gradient_checkpointing=config.gradient_checkpointing,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_keep=config.checkpoint_keep,
            resume_from_checkpoint=config.resume_from_checkpoint,
            deterministic_smoke=config.deterministic_smoke,
        )
        super().__init__(policy, OptimizerConfig(config.learning_rate, config.weight_decay), alignment, output_dir)
        self.config = config
        self.policy = policy
        self.reference_policy = reference_policy
        self.generation_backend = generation_backend or ModelGenerationBackend(policy)
        self.reward_pipeline = reward_pipeline or RuleBasedRewardPipeline()
        self.output_dir = Path(output_dir)
        if reference_policy is not None:
            reference_policy.eval()
            for parameter in reference_policy.parameters():
                parameter.requires_grad_(False)

    def _completion_mask(self, generated: torch.Tensor, prompt_length: int, pad_token_id: int) -> torch.Tensor:
        labels = generated[:, 1:]
        positions = torch.arange(labels.shape[1], device=generated.device).unsqueeze(0)
        return ((labels != pad_token_id) & (positions >= max(prompt_length - 1, 0))).float()

    def rollout(self, example: ReasoningExample, formatter: PromptFormatter) -> GRPOBatch:
        started = perf_counter()
        prompt = formatter.grpo_prompt(example, chat=self.config.use_chat_template)
        tokenizer = self.policy.tokenizer
        encoded = tokenizer(
            [prompt] * self.config.group_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.policy.device)
        prompt_length = int(encoded["input_ids"].shape[-1])
        self.policy.eval()
        with torch.no_grad():
            generated = self.generation_backend.generate(
                **encoded,
                do_sample=not self.config.deterministic_smoke,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            attention = (generated != tokenizer.pad_token_id).long()
            labels = generated[:, 1:].contiguous()
            completion_mask = self._completion_mask(generated, prompt_length, tokenizer.pad_token_id)
            old_output = self.policy.forward(ModelInputs(input_ids=generated[:, :-1], attention_mask=attention[:, :-1]))
            old_logprobs = sequence_logprobs(old_output.logits, labels, completion_mask)
            reference_logprobs = None
            if self.reference_policy is not None:
                ref_output = self.reference_policy.forward(ModelInputs(input_ids=generated[:, :-1], attention_mask=attention[:, :-1]))
                reference_logprobs = sequence_logprobs(ref_output.logits, labels, completion_mask)
        self.policy.train()
        responses = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
        rewards = self.reward_pipeline.evaluate(responses, example.reference_answer)
        reward_values = torch.tensor([item["total_reward"] for item in rewards], dtype=torch.float32, device=self.policy.device)
        advantages, zero_variance = normalize_group_advantages(reward_values, self.config.advantage_epsilon)
        completion_lengths = completion_mask.sum(dim=-1)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        completion_ids = generated[:, prompt_length:]
        has_eos = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
        if eos_id is not None and completion_ids.numel():
            has_eos = (completion_ids == eos_id).any(dim=-1)
        truncated = (completion_lengths >= self.config.max_new_tokens) & ~has_eos
        self.state.rollout_tokens += int(completion_lengths.sum().item())
        batch = GRPOBatch(
            prompt=prompt,
            prompt_length=prompt_length,
            generated_ids=generated,
            attention_mask=attention,
            completion_mask=completion_mask,
            responses=responses,
            rewards=rewards,
            old_logprobs=old_logprobs.detach(),
            reference_logprobs=reference_logprobs.detach() if reference_logprobs is not None else None,
            truncated=truncated,
            rollout_seconds=perf_counter() - started,
            advantages=advantages.detach(),
        )
        batch._zero_variance = zero_variance  # Local rollout metadata, not serialized.
        return batch

    def optimize(self, batch: GRPOBatch) -> TrainingMetrics:
        started = perf_counter()
        generated = batch.generated_ids
        labels = generated[:, 1:].contiguous()
        output = self.policy.forward(ModelInputs(input_ids=generated[:, :-1], attention_mask=batch.attention_mask[:, :-1]))
        new_logprobs = sequence_logprobs(output.logits, labels, batch.completion_mask)
        policy_loss, ratios = clipped_policy_loss(
            new_logprobs,
            batch.old_logprobs,
            batch.advantages if batch.advantages is not None else torch.zeros(generated.shape[0], device=generated.device),
            batch.completion_mask,
            self.config.clip_eps,
        )
        kl = torch.zeros((), device=generated.device)
        if batch.reference_logprobs is not None:
            kl = approximate_kl(new_logprobs, batch.reference_logprobs, batch.completion_mask)
        entropy = token_entropy(output.logits, batch.completion_mask)
        kl_loss = self.config.beta_kl * kl
        loss = policy_loss + kl_loss - self.config.entropy_coef * entropy
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        numeric_components: dict[str, list[float]] = {}
        for reward in batch.rewards:
            for name, value in reward.items():
                if isinstance(value, (float, int)):
                    numeric_components.setdefault(name, []).append(float(value))
        reward_tensor = torch.tensor([item["total_reward"] for item in batch.rewards], device=generated.device)
        completion_mask = batch.completion_mask.bool()
        active_ratios = ratios[completion_mask]
        return TrainingMetrics(
            step=self.state.global_step,
            loss=float(loss.detach().cpu()),
            policy_loss=float(policy_loss.detach().cpu()),
            kl_loss=float(kl_loss.detach().cpu()),
            kl=float(kl.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            clip_fraction=float(((active_ratios - 1.0).abs() > self.config.clip_eps).float().mean().cpu()) if active_ratios.numel() else 0.0,
            reward_mean=float(reward_tensor.mean().cpu()),
            reward_std=float(reward_tensor.std(unbiased=False).cpu()),
            advantage_mean=float(batch.advantages.mean().cpu()) if batch.advantages is not None else 0.0,
            advantage_std=float(batch.advantages.std(unbiased=False).cpu()) if batch.advantages is not None else 0.0,
            zero_variance_groups=int(getattr(batch, "_zero_variance", False)),
            truncated_completions=int(batch.truncated.sum().item()),
            rollout_tokens=int(batch.completion_mask.sum().item()),
            rollout_seconds=batch.rollout_seconds,
            optimization_seconds=perf_counter() - started,
            reward_components={name: sum(values) / len(values) for name, values in numeric_components.items()},
        )

    def train(self, examples: list[ReasoningExample], formatter: PromptFormatter) -> None:
        self.maybe_resume()
        for index in tqdm(range(self.state.global_step, self.config.max_steps), desc="grpo"):
            example = examples[index % len(examples)]
            batch = self.rollout(example, formatter)
            metrics = self.optimize(batch)
            append_jsonl([metrics.to_dict()], self.output_dir / "train_metrics.jsonl")
            append_jsonl([
                {"step": metrics.step, "prompt": batch.prompt, "response": response, "gold_answer": example.reference_answer, "reward": reward}
                for response, reward in zip(batch.responses, batch.rewards)
            ], self.output_dir / "sampled_responses.jsonl")
            if metrics.step % max(self.config.logging_steps, 1) == 0:
                print(f"grpo step={metrics.step} loss={metrics.loss:.6f} reward={metrics.reward_mean:.4f} kl={metrics.kl:.6f}", flush=True)
            if self.config.checkpoint_interval and metrics.step % self.config.checkpoint_interval == 0:
                self.checkpoint(asdict(self.config))


def _load_backend(config: GRPOConfig, *, reference: bool = False) -> ModelBackend:
    if config.bf16:
        config.dtype = "bf16"
    if config.fp16:
        config.dtype = "fp16"
    model_path = str(Path(config.resume_from_checkpoint) / "model") if config.resume_from_checkpoint else config.model_name_or_path
    if not config.resume_from_checkpoint and (Path(model_path) / "checkpoint").exists():
        model_path = str(Path(model_path) / "checkpoint")
    return ModelBackendRegistry.from_config(ModelLoadConfig(
        model_name_or_path=model_path,
        revision=config.model_revision,
        backend_name=config.backend_name,
        custom_factory_name=config.custom_factory_name,
        device=config.device,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        use_lora=config.use_lora and not reference,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    ))


def train_grpo(config: GRPOConfig) -> None:
    """Backward-compatible GRPO entry point backed by :class:`GRPOTrainerEngine`."""
    ensure_dir(config.output_dir)
    policy = _load_backend(config)
    reference = _load_backend(config, reference=True) if config.use_reference_policy else None
    examples, fingerprint, _ = DatasetRegistry.load(DatasetLoadConfig(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        config_name=config.dataset_config_name,
        revision=config.dataset_revision,
        source_path=config.train_file,
        cache_dir=config.dataset_cache_dir,
        max_samples=config.max_samples,
        shuffle=config.dataset_shuffle,
        seed=config.dataset_seed,
        purpose="train",
    ))
    save_json(asdict(config), Path(config.output_dir) / "config.json")
    save_json(fingerprint.to_dict(), Path(config.output_dir) / "dataset_fingerprint.json")
    engine = GRPOTrainerEngine(policy, config, config.output_dir, reference)
    engine.train(examples, PromptFormatter(config.system_prompt, config.final_answer_format))
    final_checkpoint = Path(config.output_dir) / "checkpoint"
    policy.save_pretrained(final_checkpoint)  # Legacy path.
    write_bilingual_readme(
        final_checkpoint,
        title="GRPO Checkpoint",
        english="Final GRPO policy checkpoint for this run. Load this directory for evaluation or continued training.",
        chinese="这是本次运行的最终 GRPO policy checkpoint。评测或继续训练时请加载这个目录。",
        preserve_existing=True,
    )
    engine.checkpoint(asdict(config))
    save_json({"final_step": engine.state.global_step, "rollout_tokens": engine.state.rollout_tokens}, Path(config.output_dir) / "metrics.json")
