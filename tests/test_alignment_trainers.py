from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from src.alignment.grpo_trainer import GRPOConfig, GRPOTrainerEngine
from src.alignment.kl_utils import approximate_kl, clipped_policy_loss, normalize_group_advantages
from src.alignment.sft_trainer import MathSFTDataset, SFTConfig, SFTTrainerEngine, collate
from src.alignment.training_core import GRPOBatch, RuleBasedRewardPipeline
from src.data import PromptFormatter, ReasoningExample
from src.models.backend import CustomCausalLMBackend, ModelInputs


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 15

    def __call__(self, text, **kwargs):
        ids = [min(14, max(1, ord(char) % 15)) for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def backend():
    return CustomCausalLMBackend(TinyCausalLM(), TinyTokenizer(), device="cpu")


def example():
    return ReasoningExample(
        uid="tiny-1",
        dataset_name="tiny",
        split="train",
        question="2+2?",
        reference_answer="4",
        reference_solution="<reasoning>two plus two</reasoning>",
        difficulty="easy",
    )


def test_sft_prompt_loss_mask():
    tokenizer = TinyTokenizer()
    dataset = MathSFTDataset([example()], tokenizer, 128, PromptFormatter(), False)
    item = dataset[0]
    assert (item["labels"] == -100).any()
    assert (item["labels"] != -100).any()


def test_group_advantage_and_constant_reward_group():
    advantages, zero_variance = normalize_group_advantages(torch.tensor([1.0, 2.0, 3.0]))
    assert not zero_variance
    assert torch.isclose(advantages.mean(), torch.tensor(0.0))
    constant, zero_variance = normalize_group_advantages(torch.tensor([2.0, 2.0, 2.0]))
    assert zero_variance
    assert torch.equal(constant, torch.zeros_like(constant))


def test_completion_padding_mask_and_clipped_objective(tmp_path):
    engine = GRPOTrainerEngine(backend(), GRPOConfig(model_name_or_path="tiny", output_dir=str(tmp_path), group_size=2), tmp_path)
    generated = torch.tensor([[2, 3, 4, 0, 0], [2, 5, 6, 7, 0]])
    mask = engine._completion_mask(generated, prompt_length=2, pad_token_id=0)
    assert mask.tolist() == [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    new = torch.log(torch.tensor([[1.4, 1.4]]))
    old = torch.zeros_like(new)
    loss, _ = clipped_policy_loss(new, old, torch.tensor([1.0]), torch.ones_like(new), 0.2)
    assert torch.isclose(loss, torch.tensor(-1.2))


def test_kl_and_reward_pipeline_integration():
    policy = torch.tensor([[1.0, 2.0]])
    ref = torch.tensor([[0.5, 1.5]])
    mask = torch.ones_like(policy)
    assert approximate_kl(policy, ref, mask) >= 0
    rewards = RuleBasedRewardPipeline().evaluate(["<reasoning>x</reasoning><answer>4</answer>"], "4")
    assert rewards[0]["accuracy"] == 1.0
    assert "correctness_reward" in rewards[0]


def test_sft_gradient_accumulation_and_checkpoint_resume(tmp_path):
    model_backend = backend()
    config = SFTConfig(
        model_name_or_path="tiny",
        output_dir=str(tmp_path),
        gradient_accumulation_steps=2,
        max_steps=1,
        batch_size=1,
        deterministic_smoke=True,
        checkpoint_keep=1,
    )
    rows = [
        {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([-100, 2])},
        {"input_ids": torch.tensor([2, 3]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor([-100, 3])},
    ]
    loader = DataLoader(rows, batch_size=1, collate_fn=lambda batch: collate(batch, 0))
    engine = SFTTrainerEngine(model_backend, config, tmp_path)
    state = engine.train(loader)
    assert state.global_step == 1
    assert state.micro_step == 2
    checkpoint = engine.checkpoint({"test": True})
    resumed = SFTTrainerEngine(backend(), config, tmp_path)
    resumed.alignment_config = resumed.alignment_config.__class__(resume_from_checkpoint=str(checkpoint), checkpoint_keep=1)
    resumed.maybe_resume()
    assert resumed.state.global_step == 1


def test_one_step_deterministic_grpo_update(tmp_path):
    model_backend = backend()
    config = GRPOConfig(model_name_or_path="tiny", output_dir=str(tmp_path), group_size=2, max_steps=1, use_reference_policy=False)
    engine = GRPOTrainerEngine(model_backend, config, tmp_path)
    generated = torch.tensor([[1, 2, 3], [1, 4, 5]])
    attention = torch.ones_like(generated)
    completion_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    labels = generated[:, 1:]
    with torch.no_grad():
        old = engine.policy.forward(ModelInputs(input_ids=generated[:, :-1], attention_mask=attention[:, :-1]))
        old_logprobs = torch.log_softmax(old.logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * completion_mask
    batch = GRPOBatch(
        prompt="tiny",
        prompt_length=2,
        generated_ids=generated,
        attention_mask=attention,
        completion_mask=completion_mask,
        responses=["a", "b"],
        rewards=[{"total_reward": 0.0}, {"total_reward": 1.0}],
        old_logprobs=old_logprobs,
        reference_logprobs=None,
        truncated=torch.tensor([False, False]),
        rollout_seconds=0.0,
        advantages=torch.tensor([-1.0, 1.0]),
    )
    metrics = engine.optimize(batch)
    assert engine.state.global_step == 1
    assert metrics.rollout_tokens == 2
    assert torch.isfinite(torch.tensor(metrics.loss))
