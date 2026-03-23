# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalution.config import Model
    from evalution.runtime import EvaluationRun


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


class BaseInferenceSession(ABC):
    # Define the reusable model-session contract that every engine runtime must satisfy.
    # Runtime code and test suites talk to this interface instead of branching on engine type.

    @abstractmethod
    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]: ...

    @abstractmethod
    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]: ...

    @abstractmethod
    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]: ...

    @abstractmethod
    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]: ...

    # Let engines release reusable caches between suites without fully unloading model state.
    @abstractmethod
    def gc(self) -> None: ...

    # Tear down the full session state and release heavyweight runtime objects.
    @abstractmethod
    def close(self) -> None: ...

    # Return engine-specific execution metadata for logs and result payloads.
    def describe_execution(self) -> dict[str, Any] | None:
        return None


class BaseEngine(ABC):
    # Define the engine-level contract shared by transformers, vLLM, SGLang, and vendor runtimes.
    # A BaseEngine owns configuration and builds a reusable BaseInferenceSession for one model.

    @abstractmethod
    def build(self, model: Model) -> BaseInferenceSession:
        # Construct a reusable inference session for one model configuration.
        # The returned object must inherit BaseInferenceSession because runtime orchestration,
        # suite execution, and result materialization depend on the common inference APIs.
        raise NotImplementedError

    def model(self, model: Model | dict, *, label: str | None = None) -> EvaluationRun:
        from evalution.config import coerce_model, model_with_label
        from evalution.runtime import EvaluationRun

        return EvaluationRun(
            _engine_impl=self,
            _model_config=model_with_label(coerce_model(model), label=label),
        )

    # Serialize engine controls into the public run result payload.
    def to_dict(self) -> dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)
        return {}


# Keep the older type name available inside suite modules while the concrete base class stays explicit.
InferenceSession = BaseInferenceSession
