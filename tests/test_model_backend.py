from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.backend import (
    CustomCausalLMBackend,
    ModelBackend,
    ModelBackendRegistry,
    ModelInputs,
    ModelLoadConfig,
    UnsupportedModelCapabilityError,
)


class TinyTokenizer:
    pad_token_id = 0

    def save_pretrained(self, path):
        return (str(path),)


class TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 6)
        self.head = torch.nn.Linear(6, 16)
        self.last_inputs = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=False,
    ):
        self.last_inputs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "labels": labels,
            "use_cache": use_cache,
        }
        logits = self.head(self.embedding(input_ids))
        past = past_key_values if past_key_values is not None else ((input_ids.detach().clone(),),)
        return SimpleNamespace(logits=logits, past_key_values=past)


class InputOnlyCausalLM(torch.nn.Module):
    def forward(self, input_ids):
        return torch.zeros((*input_ids.shape, 4))


class RegistryProbeBackend(ModelBackend):
    @classmethod
    def from_config(cls, config):
        return cls()

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def forward(self, inputs):
        raise AssertionError("not used")

    def save_pretrained(self, path):
        raise AssertionError("not used")

    def to(self, device):
        return self

    def train(self):
        return None

    def eval(self):
        return None

    def parameters(self):
        return iter(())

    def named_parameters(self):
        return iter(())


def test_custom_backend_forwards_position_ids_and_past_key_values():
    model = TinyCausalLM()
    backend = CustomCausalLMBackend(model, TinyTokenizer(), device="cpu")
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[4, 5, 6]])
    past = ((torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1)),)
    out = backend.forward(
        ModelInputs(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            past_key_values=past,
            labels=ids,
            use_cache=True,
        )
    )
    assert out.logits.shape == (1, 3, 16)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.past_key_values is past
    assert model.last_inputs["position_ids"] is positions
    assert model.last_inputs["past_key_values"] is past
    assert model.last_inputs["use_cache"] is True


def test_custom_backend_rejects_unsupported_input_explicitly():
    backend = CustomCausalLMBackend(InputOnlyCausalLM(), TinyTokenizer())
    with pytest.raises(UnsupportedModelCapabilityError, match="position_ids"):
        backend.forward(ModelInputs(input_ids=torch.tensor([[1]]), position_ids=torch.tensor([[0]])))
    with pytest.raises(UnsupportedModelCapabilityError, match="generate"):
        backend.generate(input_ids=torch.tensor([[1]]))


def test_custom_checkpoint_round_trip_via_registry(tmp_path):
    factory_name = "pytest-tiny-causal-lm"

    def factory(config):
        return TinyCausalLM(), TinyTokenizer()

    CustomCausalLMBackend.register_factory(factory_name, factory)
    source = CustomCausalLMBackend(TinyCausalLM(), TinyTokenizer())
    for parameter in source.parameters():
        torch.nn.init.constant_(parameter, 0.125)
    source.save_pretrained(tmp_path)

    restored = ModelBackendRegistry.from_config(
        ModelLoadConfig(
            backend_name="custom",
            model_name_or_path=str(tmp_path),
            custom_factory_name=factory_name,
            device="cpu",
        )
    )
    source_logits = source.forward(ModelInputs(input_ids=torch.tensor([[1, 2]]))).logits
    restored_logits = restored.forward(ModelInputs(input_ids=torch.tensor([[1, 2]]))).logits
    assert torch.equal(source_logits, restored_logits)


def test_registry_selects_registered_backend_and_rejects_unknown_name():
    ModelBackendRegistry.register_backend("pytest-probe", RegistryProbeBackend)
    assert isinstance(ModelBackendRegistry.from_config(ModelLoadConfig(backend_name="pytest-probe")), RegistryProbeBackend)
    assert ModelBackendRegistry.get_backend_class("huggingface").__name__ == "HuggingFaceCausalLMBackend"
    with pytest.raises(KeyError, match="Unknown backend"):
        ModelBackendRegistry.from_config(ModelLoadConfig(backend_name="missing-backend"))
