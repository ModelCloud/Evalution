from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class GenerationRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    add_generation_prompt: bool = True
    stop: list[str] = field(default_factory=list)
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationOutput:
    prompt: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceSession(Protocol):
    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]: ...

    def close(self) -> None: ...
