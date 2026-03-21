from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class GenerationRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    rendered_prompt: str | None = None
    input_ids: list[int] | None = None
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


@dataclass(slots=True)
class LoglikelihoodRequest:
    # Score `continuation` conditioned on `context`, optionally reusing pretokenized ids.
    context: str = ""
    continuation: str = ""
    context_input_ids: list[int] | None = None
    continuation_input_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoglikelihoodOutput:
    # Report the summed continuation log-probability plus whether greedy decoding agrees.
    logprob: float
    is_greedy: bool
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RollingLoglikelihoodRequest:
    # Score a full text span token-by-token for perplexity-style evaluations.
    text: str
    input_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RollingLoglikelihoodOutput:
    # Report the total rolling log-probability across every scored token.
    logprob: float
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceSession(Protocol):
    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]: ...

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]: ...

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]: ...

    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]: ...

    def gc(self) -> None: ...

    def close(self) -> None: ...
