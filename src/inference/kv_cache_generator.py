from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .cache_utils import CacheStats, cache_seq_len, next_position_ids, position_ids_from_attention_mask, trim_past_key_values
from src.models.backend import ModelBackendRegistry, ModelLoadConfig, ModelInputs


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    token_ids: list[int]
    generated_token_ids: list[int]
    step_latencies: list[float]
    prefill_latency: float
    total_latency: float
    cache_rebuild_count: int
    avg_cache_seq_len: float
    max_cache_seq_len: int
    mode: str
    extra: dict[str, Any] = field(default_factory=dict)


class KVCacheGenerator:
    """Token-by-token generation with explicit no-cache and KV-cache paths."""

    def __init__(self, model, tokenizer, device: str = "cuda", eos_token_id: int | None = None):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        trust_remote_code: bool = True,
        backend_name: str = "huggingface",
        custom_factory_name: str | None = None,
    ):
        backend = ModelBackendRegistry.from_config(
            ModelLoadConfig(
                model_name_or_path=model_name_or_path,
                backend_name=backend_name,
                custom_factory_name=custom_factory_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                use_lora=False,
            )
        )
        return cls(model=backend, tokenizer=backend.tokenizer, device=device)

    def generate(
        self,
        prompts: str | list[str],
        max_new_tokens: int = 128,
        use_cache: bool = True,
        cache_window: int | None = None,
        batch_size: int = 1,
    ) -> list[GenerationResult]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        if batch_size > 1 and use_cache and cache_window is None and getattr(getattr(self.model, "capabilities", None), "supports_generate", False):
            return self._generate_hf_batches(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
        return [
            self._generate_one(prompt, max_new_tokens=max_new_tokens, use_cache=use_cache, cache_window=cache_window)
            for prompt in prompts
        ]

    def _encode(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt", padding=False).to(self.device)

    def _generate_hf_batches(self, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[GenerationResult]:
        import torch

        results: list[GenerationResult] = []
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                enc = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
                input_width = int(enc["input_ids"].shape[-1])
                started = time.perf_counter()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                        eos_token_id=self.eos_token_id,
                        use_cache=True,
                    )
                batch_latency = time.perf_counter() - started
                per_item_latency = batch_latency / max(len(batch_prompts), 1)
                for prompt, row in zip(batch_prompts, output_ids):
                    token_ids = row.tolist()
                    generated = token_ids[input_width:]
                    if self.eos_token_id is not None and self.eos_token_id in generated:
                        generated = generated[: generated.index(self.eos_token_id) + 1]
                    results.append(
                        GenerationResult(
                            prompt=prompt,
                            generated_text=self.tokenizer.decode(generated, skip_special_tokens=True),
                            token_ids=token_ids,
                            generated_token_ids=generated,
                            step_latencies=[],
                            prefill_latency=0.0,
                            total_latency=per_item_latency,
                            cache_rebuild_count=0,
                            avg_cache_seq_len=float(input_width + len(generated) / 2),
                            max_cache_seq_len=input_width + len(generated),
                            mode=f"hf-batch-generate-{len(batch_prompts)}",
                        )
                    )
        finally:
            self.tokenizer.padding_side = original_padding_side
        return results

    def _generate_one(self, prompt: str, max_new_tokens: int, use_cache: bool, cache_window: int | None) -> GenerationResult:
        if use_cache:
            return self._generate_cache(prompt, max_new_tokens, cache_window)
        return self._generate_no_cache(prompt, max_new_tokens)

    def _generate_no_cache(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        import torch

        enc = self._encode(prompt)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        prompt_len = int(input_ids.shape[-1])
        step_latencies: list[float] = []
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                position_ids = position_ids_from_attention_mask(attention_mask)
                t0 = time.perf_counter()
                out = self.model.forward(
                    ModelInputs(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                )
                step_latencies.append(time.perf_counter() - t0)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                if self.eos_token_id is not None and int(next_token[0, 0]) == int(self.eos_token_id):
                    break
        total = time.perf_counter() - started
        token_ids = input_ids[0].tolist()
        generated = token_ids[prompt_len:]
        return GenerationResult(
            prompt=prompt,
            generated_text=self.tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=token_ids,
            generated_token_ids=generated,
            step_latencies=step_latencies,
            prefill_latency=0.0,
            total_latency=total,
            cache_rebuild_count=0,
            avg_cache_seq_len=0.0,
            max_cache_seq_len=0,
            mode="no-cache",
        )

    def _generate_cache(self, prompt: str, max_new_tokens: int, cache_window: int | None) -> GenerationResult:
        import torch

        enc = self._encode(prompt)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        prompt_len = int(input_ids.shape[-1])
        context_ids = input_ids.clone()
        context_attention = attention_mask.clone()
        stats = CacheStats()
        step_latencies: list[float] = []
        started = time.perf_counter()

        with torch.no_grad():
            prefill_start = time.perf_counter()
            position_ids = position_ids_from_attention_mask(context_attention)
            out = self.model.forward(
                ModelInputs(
                    input_ids=context_ids,
                    attention_mask=context_attention,
                    position_ids=position_ids,
                    use_cache=True,
                )
            )
            prefill_latency = time.perf_counter() - prefill_start
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for step in range(max_new_tokens):
                context_ids = torch.cat([context_ids, next_token], dim=-1)
                context_attention = torch.cat([context_attention, torch.ones_like(next_token)], dim=-1)
                stats.add_length(cache_seq_len(past))
                if self.eos_token_id is not None and int(next_token[0, 0]) == int(self.eos_token_id):
                    break
                if step == max_new_tokens - 1:
                    break

                if cache_window is not None and cache_window > 0 and cache_seq_len(past) >= cache_window:
                    # Rebuild from recent window while preserving absolute position ids.
                    stats.rebuild_count += 1
                    start = max(0, context_ids.shape[-1] - cache_window)
                    window_ids = context_ids[:, start:]
                    window_mask = context_attention[:, start:]
                    absolute_positions = torch.arange(start, context_ids.shape[-1], device=self.device).unsqueeze(0)
                    t0 = time.perf_counter()
                    rebuild = self.model.forward(
                        ModelInputs(
                            input_ids=window_ids,
                            attention_mask=window_mask,
                            position_ids=absolute_positions,
                            use_cache=True,
                        )
                    )
                    step_latencies.append(time.perf_counter() - t0)
                    past = trim_past_key_values(rebuild.past_key_values, cache_window)
                    next_token = rebuild.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    continue

                past = trim_past_key_values(past, cache_window)
                cache_len = cache_seq_len(past)
                decode_attention = torch.ones((1, cache_len + 1), dtype=context_attention.dtype, device=self.device)
                pos = next_position_ids([context_ids.shape[-1] - 1], self.device)
                t0 = time.perf_counter()
                out = self.model.forward(
                    ModelInputs(
                        input_ids=next_token,
                        attention_mask=decode_attention,
                        position_ids=pos,
                        past_key_values=past,
                        use_cache=True,
                    )
                )
                step_latencies.append(time.perf_counter() - t0)
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        total = time.perf_counter() - started
        token_ids = context_ids[0].tolist()
        generated = token_ids[prompt_len:]
        return GenerationResult(
            prompt=prompt,
            generated_text=self.tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=token_ids,
            generated_token_ids=generated,
            step_latencies=step_latencies,
            prefill_latency=prefill_latency,
            total_latency=total,
            cache_rebuild_count=stats.rebuild_count,
            avg_cache_seq_len=stats.avg_cache_seq_len,
            max_cache_seq_len=stats.max_cache_seq_len,
            mode="full-cache" if not cache_window or cache_window <= 0 else f"window-cache-{cache_window}",
        )
