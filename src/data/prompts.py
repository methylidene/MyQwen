"""Prompt construction for completion, chat, SFT and GRPO workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import ReasoningExample


@dataclass(frozen=True)
class PromptFormatter:
    """Build prompts without placing a reference answer in GRPO inputs."""

    system_prompt: str | None = None
    final_answer_format: str = "<answer>{answer}</answer>"
    reasoning_open_tag: str = "<reasoning>"
    reasoning_close_tag: str = "</reasoning>"

    def _instruction(self) -> str:
        return (
            f"Show your reasoning between {self.reasoning_open_tag} and {self.reasoning_close_tag}, "
            f"keep it concise, then provide only the bare numeric final answer as "
            f"{self.final_answer_format.format(answer='...')}. Do not include units, currency symbols, "
            "or explanatory text inside the answer tags."
        )

    def plain_completion(self, example: ReasoningExample) -> str:
        return example.question.strip()

    def chat_template(self, example: ReasoningExample) -> str:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt.strip()}")
        parts.extend((f"User: {example.question.strip()}", "Assistant:"))
        return "\n".join(parts)

    def grpo_prompt(self, example: ReasoningExample, *, chat: bool = False) -> str:
        base = self.chat_template(example) if chat else self.plain_completion(example)
        return f"{base}\n{self._instruction()}"

    def sft_completion(self, example: ReasoningExample) -> str:
        solution = example.reference_solution.strip()
        answer = self.final_answer_format.format(answer=example.reference_answer)
        if self.reasoning_open_tag in solution and self.reasoning_close_tag in solution:
            reasoning = solution
        else:
            reasoning_lines = []
            for line in solution.splitlines():
                if line.strip().startswith("####"):
                    continue
                reasoning_lines.append(line)
            reasoning_body = "\n".join(reasoning_lines).strip()
            if not reasoning_body:
                return answer
            reasoning = f"{self.reasoning_open_tag}\n{reasoning_body}\n{self.reasoning_close_tag}"
        if answer in reasoning:
            return reasoning
        return f"{reasoning}\n{answer}"

    def sft_text(self, example: ReasoningExample, *, chat: bool = False) -> tuple[str, str]:
        prompt = self.grpo_prompt(example, chat=chat)
        return prompt, self.sft_completion(example)

    def format(self, example: ReasoningExample, mode: str, *, chat: bool = False) -> str:
        if mode == "plain":
            return self.plain_completion(example)
        if mode == "chat":
            return self.chat_template(example)
        if mode == "grpo":
            return self.grpo_prompt(example, chat=chat)
        if mode == "sft":
            prompt, completion = self.sft_text(example, chat=chat)
            return f"{prompt}\n{completion}"
        raise ValueError(f"Unsupported prompt mode '{mode}'.")
