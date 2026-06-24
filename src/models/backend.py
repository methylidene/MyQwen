"""Unified model boundary for inference and alignment workflows.

Backends keep framework-specific model loading, tokenizers, checkpoints and
optional PEFT support out of trainers and evaluators. A custom causal model
only needs to be a ``torch.nn.Module``; it does not need to inherit from
``transformers.PreTrainedModel``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F


class UnsupportedModelCapabilityError(NotImplementedError):
    """Raised when a caller requests an input the selected backend cannot use."""


@dataclass(frozen=True)
class ModelLoadConfig:
    """Framework-neutral settings used to construct a :class:`ModelBackend`."""

    model_name_or_path: str | None = None
    revision: str | None = None
    backend_name: str = "huggingface"
    device: str = "cuda"
    dtype: str = "auto"
    trust_remote_code: bool = True
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    extra_load_kwargs: Mapping[str, Any] = field(default_factory=dict)
    custom_factory_name: str | None = None


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared backend features; callers receive explicit errors if absent."""

    supports_generate: bool = False
    supports_use_cache: bool = False
    supports_past_key_values: bool = False
    supports_position_ids: bool = False
    supports_attention_mask: bool = False
    supports_labels: bool = False
    supports_loss: bool = False
    supports_hidden_states: bool = False
    supports_lora: bool = False
    supports_gradient_checkpointing: bool = False
    supports_save_pretrained: bool = True


@dataclass(frozen=True)
class ModelInputs:
    """Inputs accepted by causal-LM forward passes across all backends."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    past_key_values: Any | None = None
    labels: torch.Tensor | None = None
    output_hidden_states: bool = False
    use_cache: bool = False


@dataclass(frozen=True)
class ModelForwardOutput:
    """Normalized result returned from a causal-LM forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    past_key_values: Any | None = None
    raw_model_output: Any | None = None


class ModelBackend(ABC):
    """Small common interface used by training, evaluation and generation."""

    tokenizer: Any
    capabilities: ModelCapabilities

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Current device of the wrapped model."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Requested or inferred floating-point dtype."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelLoadConfig) -> "ModelBackend":
        """Load a backend from a portable configuration."""

    @abstractmethod
    def forward(self, inputs: ModelInputs) -> ModelForwardOutput:
        """Run one causal-LM forward pass."""

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Generate token ids when the backend supports generation."""
        raise UnsupportedModelCapabilityError(f"{type(self).__name__} does not support generate().")

    @abstractmethod
    def save_pretrained(self, path: str | Path) -> None:
        """Save a reloadable checkpoint and tokenizer when available."""

    @abstractmethod
    def to(self, device: str | torch.device) -> "ModelBackend":
        """Move the wrapped model to ``device``."""

    @abstractmethod
    def train(self) -> None:
        """Switch the wrapped model to train mode."""

    @abstractmethod
    def eval(self) -> None:
        """Switch the wrapped model to eval mode."""

    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return wrapped-model parameters."""

    @abstractmethod
    def named_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        """Return named wrapped-model parameters."""

    def enable_gradient_checkpointing(self) -> None:
        """Enable checkpointing or raise a capability-specific error."""
        raise UnsupportedModelCapabilityError(f"{type(self).__name__} does not support gradient checkpointing.")


def _resolved_device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    return torch.device("cpu") if requested.type == "cuda" and not torch.cuda.is_available() else requested


def _torch_dtype(dtype: str) -> torch.dtype | None:
    choices = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "auto": None}
    try:
        return choices[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of: {', '.join(choices)}.") from exc


def _validate_inputs(inputs: ModelInputs, capabilities: ModelCapabilities) -> None:
    requested = (
        (inputs.attention_mask is not None, "attention_mask", capabilities.supports_attention_mask),
        (inputs.position_ids is not None, "position_ids", capabilities.supports_position_ids),
        (inputs.past_key_values is not None, "past_key_values", capabilities.supports_past_key_values),
        (inputs.labels is not None, "labels", capabilities.supports_labels),
        (inputs.output_hidden_states, "output_hidden_states", capabilities.supports_hidden_states),
        (inputs.use_cache, "use_cache", capabilities.supports_use_cache),
    )
    for is_requested, name, supported in requested:
        if is_requested and not supported:
            raise UnsupportedModelCapabilityError(f"{name} is not supported by this model backend.")


class ModelBackendRegistry:
    """Registry for built-in and application-provided backend constructors."""

    _backends: ClassVar[dict[str, type[ModelBackend]]] = {}

    @classmethod
    def register_backend(cls, name: str, backend_cls: type[ModelBackend]) -> None:
        if not name:
            raise ValueError("Backend name must not be empty.")
        cls._backends[name] = backend_cls

    @classmethod
    def get_backend_class(cls, name: str) -> type[ModelBackend]:
        try:
            return cls._backends[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._backends)) or "none"
            raise KeyError(f"Unknown backend '{name}'. Available backends: {available}.") from exc

    @classmethod
    def from_config(cls, config: ModelLoadConfig) -> ModelBackend:
        return cls.get_backend_class(config.backend_name).from_config(config)


class HuggingFaceCausalLMBackend(ModelBackend):
    """Adapter around ``AutoModelForCausalLM`` and its tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str | torch.device = "cuda", dtype: torch.dtype | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._device = _resolved_device(device)
        self.model.to(self._device)
        self._dtype = dtype or getattr(model, "dtype", torch.float32)
        self.capabilities = ModelCapabilities(
            supports_generate=hasattr(model, "generate"),
            supports_use_cache=True,
            supports_past_key_values=True,
            supports_position_ids=True,
            supports_attention_mask=True,
            supports_labels=True,
            supports_loss=True,
            supports_hidden_states=True,
            supports_lora=True,
            supports_gradient_checkpointing=hasattr(model, "gradient_checkpointing_enable"),
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def from_config(cls, config: ModelLoadConfig) -> "HuggingFaceCausalLMBackend":
        if not config.model_name_or_path:
            raise ValueError("HuggingFaceCausalLMBackend requires model_name_or_path.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code, revision=config.revision)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = _torch_dtype(config.dtype)
        load_kwargs = dict(config.extra_load_kwargs)
        load_kwargs.update({"torch_dtype": dtype, "trust_remote_code": config.trust_remote_code, "low_cpu_mem_usage": True, "revision": config.revision})
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
        backend = cls(model, tokenizer, device=config.device, dtype=dtype)
        if config.use_lora:
            backend._enable_lora(config)
        return backend

    def _enable_lora(self, config: ModelLoadConfig) -> None:
        if not self.capabilities.supports_lora:
            raise UnsupportedModelCapabilityError("This Hugging Face model does not support LoRA.")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA was requested but the 'peft' package is not installed.") from exc
        if hasattr(self.model, "peft_config"):
            if hasattr(self.model, "enable_adapter_layers"):
                self.model.enable_adapter_layers()
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_("lora_" in name or "modules_to_save" in name)
            return
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora)
        self.model.to(self._device)

    def forward(self, inputs: ModelInputs) -> ModelForwardOutput:
        _validate_inputs(inputs, self.capabilities)
        kwargs: dict[str, Any] = {"input_ids": inputs.input_ids}
        if inputs.use_cache:
            kwargs["use_cache"] = True
        for name in ("attention_mask", "position_ids", "past_key_values", "labels"):
            value = getattr(inputs, name)
            if value is not None:
                kwargs[name] = value
        if inputs.output_hidden_states:
            kwargs["output_hidden_states"] = True
        output = self.model(**kwargs)
        return ModelForwardOutput(
            logits=output.logits,
            loss=getattr(output, "loss", None),
            hidden_states=getattr(output, "hidden_states", None),
            past_key_values=getattr(output, "past_key_values", None),
            raw_model_output=output,
        )

    def generate(self, **kwargs: Any) -> torch.Tensor:
        if not self.capabilities.supports_generate:
            return super().generate(**kwargs)
        return self.model.generate(**kwargs)

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(destination)

    def to(self, device: str | torch.device) -> "HuggingFaceCausalLMBackend":
        self._device = _resolved_device(device)
        self.model.to(self._device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        if not self.capabilities.supports_gradient_checkpointing:
            return super().enable_gradient_checkpointing()
        self.model.gradient_checkpointing_enable()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        return self.model.named_parameters()


class CustomCausalLMBackend(ModelBackend):
    """Wrapper for a plain PyTorch causal LM plus a tokenizer or tokenizer adapter.

    ``register_factory`` makes a custom model reloadable from registry configs.
    The wrapped module may return a Tensor, mapping, object with ``logits`` or
    :class:`ModelForwardOutput`.
    """

    _factories: ClassVar[dict[str, Callable[[ModelLoadConfig], tuple[torch.nn.Module, Any]]]] = {}

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._device = _resolved_device(device)
        self.model.to(self._device)
        self._dtype = dtype or getattr(model, "dtype", torch.float32)
        self.capabilities = capabilities or self._capabilities_from_signature(model)

    @staticmethod
    def _capabilities_from_signature(model: torch.nn.Module) -> ModelCapabilities:
        parameters = inspect.signature(model.forward).parameters
        accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
        def accepts(name: str) -> bool:
            return accepts_kwargs or name in parameters
        return ModelCapabilities(
            supports_generate=hasattr(model, "generate"),
            supports_use_cache=accepts("use_cache"),
            supports_past_key_values=accepts("past_key_values"),
            supports_position_ids=accepts("position_ids"),
            supports_attention_mask=accepts("attention_mask"),
            supports_labels=accepts("labels"),
            supports_loss=accepts("labels"),
            supports_hidden_states=accepts("output_hidden_states"),
            supports_lora=False,
            supports_gradient_checkpointing=hasattr(model, "gradient_checkpointing_enable"),
        )

    @classmethod
    def register_factory(cls, name: str, factory: Callable[[ModelLoadConfig], tuple[torch.nn.Module, Any]]) -> None:
        if not name:
            raise ValueError("Custom model factory name must not be empty.")
        cls._factories[name] = factory

    @classmethod
    def from_config(cls, config: ModelLoadConfig) -> "CustomCausalLMBackend":
        factory_name = config.custom_factory_name or config.model_name_or_path
        if not factory_name:
            raise ValueError("CustomCausalLMBackend requires custom_factory_name or model_name_or_path to select a factory.")
        try:
            model, tokenizer = cls._factories[factory_name](config)
        except KeyError as exc:
            names = ", ".join(sorted(cls._factories)) or "none"
            raise ValueError(
                f"No custom model factory registered for '{factory_name}'. Registered factories: {names}."
            ) from exc
        backend = cls(model, tokenizer, device=config.device, dtype=_torch_dtype(config.dtype))
        checkpoint = Path(config.model_name_or_path or "") / "custom_model.pt"
        if checkpoint.is_file():
            try:
                state = torch.load(checkpoint, map_location=backend.device, weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location=backend.device)
            backend.model.load_state_dict(state)
        if config.use_lora:
            raise UnsupportedModelCapabilityError("CustomCausalLMBackend does not provide LoRA; implement it in the custom model factory.")
        return backend

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, inputs: ModelInputs) -> ModelForwardOutput:
        _validate_inputs(inputs, self.capabilities)
        kwargs: dict[str, Any] = {"input_ids": inputs.input_ids}
        if inputs.use_cache:
            kwargs["use_cache"] = True
        for name in ("attention_mask", "position_ids", "past_key_values", "labels"):
            value = getattr(inputs, name)
            if value is not None:
                kwargs[name] = value
        if inputs.output_hidden_states:
            kwargs["output_hidden_states"] = True
        output = self.model(**kwargs)
        if isinstance(output, ModelForwardOutput):
            return output
        if isinstance(output, Mapping):
            logits = output.get("logits")
            loss = output.get("loss")
            hidden_states = output.get("hidden_states")
            past_key_values = output.get("past_key_values")
        else:
            logits = getattr(output, "logits", output)
            loss = getattr(output, "loss", None)
            hidden_states = getattr(output, "hidden_states", None)
            past_key_values = getattr(output, "past_key_values", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Custom causal model output must provide a Tensor named 'logits' or return a Tensor.")
        if inputs.labels is not None and loss is None:
            if logits.shape[:-1] != inputs.labels.shape:
                raise ValueError("Cannot compute fallback loss: logits and labels have incompatible sequence shapes.")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), inputs.labels.reshape(-1), ignore_index=-100)
        return ModelForwardOutput(logits, loss, hidden_states, past_key_values, output)

    def generate(self, **kwargs: Any) -> torch.Tensor:
        if not self.capabilities.supports_generate:
            return super().generate(**kwargs)
        return self.model.generate(**kwargs)

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination / "custom_model.pt")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(destination)
        elif hasattr(self.tokenizer, "save_vocabulary"):
            self.tokenizer.save_vocabulary(destination)

    def to(self, device: str | torch.device) -> "CustomCausalLMBackend":
        self._device = _resolved_device(device)
        self.model.to(self._device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        if not self.capabilities.supports_gradient_checkpointing:
            return super().enable_gradient_checkpointing()
        self.model.gradient_checkpointing_enable()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.model.parameters()

    def named_parameters(self) -> Iterable[tuple[str, torch.nn.Parameter]]:
        return self.model.named_parameters()


ModelBackendRegistry.register_backend("huggingface", HuggingFaceCausalLMBackend)
ModelBackendRegistry.register_backend("custom", CustomCausalLMBackend)
