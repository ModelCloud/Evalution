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
    num_beams: int = 1
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

    # RequestExecutor work must stay off the Python main thread by default.
    @property
    def request_executor_requires_non_main_thread(self) -> bool:
        return True

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

    # Stream scored continuations back to the caller while the session keeps refilling fixed-size
    # batches in the background.
    def loglikelihood_continuous(
        self,
        requests: Iterable[tuple[Any, LoglikelihoodRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, LoglikelihoodOutput]]:
        # Fall back to the batched log-likelihood API so simple engines and test doubles do not
        # need a dedicated continuous implementation when refill behavior is irrelevant.
        request_items = list(requests)
        outputs = self.loglikelihood(
            [request for _, request in request_items],
            batch_size=batch_size,
        )
        for (item_id, _request), output in zip(request_items, outputs, strict=True):
            yield item_id, output

    @abstractmethod
    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]: ...

    # Implementations should keep backend progress decoupled from caller iteration. In
    # particular, do not hold engine/session locks across user-visible yields; use a queued
    # request/result handoff when the backend must stay active and refill independently of the
    # caller.
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

    def model(
        self,
        *,
        path: str,
        tokenizer: Any | None = None,
        tokenizer_path: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> EvaluationRun:
        from evalution.config import Model, model_with_label
        from evalution.runtime import EvaluationRun

        model = Model(
            path=path,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs or {},
            tokenizer_kwargs=tokenizer_kwargs or {},
        )
        return EvaluationRun(
            _engine_impl=self,
            _model_config=model_with_label(model, label=label),
        )

    # Serialize engine controls into the public run result payload.
    def to_dict(self) -> dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)
        return {}


@dataclass(slots=True)
class SharedEngineConfig(BaseEngine):
    # Hold engine controls shared across the major runtime families.
    dtype: str | None = "auto"
    batch_size: int | str = "auto"
    max_new_tokens: int = 256
    trust_remote_code: bool | None = None
    seed: int | None = None
    padding_side: str = "left"
    resolved_engine: str | None = field(default=None, init=False)


@dataclass
class BaseEngineDeviceConfig:
    # Share the simple explicit runtime device override across engine families.
    device: str | None = None


@dataclass
class BaseEngineTokenizerModeConfig:
    # Share tokenizer runtime mode selection across backends that expose it.
    tokenizer_mode: str = "auto"


@dataclass
class BaseEngineQuantizationConfig:
    # Share runtime quantization mode selection across backends that expose it.
    quantization: str | None = None


@dataclass
class BaseEngineTransformersRuntimeConfig(BaseEngineDeviceConfig):
    # Group the Hugging Face style runtime controls shared by transformer-like backends.
    attn_implementation: str | None = None
    device_map: str | dict[str, Any] | None = None


@dataclass
class BaseEnginePagedBatchingConfig:
    # Group the paged-batching controls shared by engines that expose the same scheduler knobs.
    manual_eviction: bool = False
    allow_block_sharing: bool = True
    use_async_batching: bool | None = None
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0
    max_cached_graphs: int = 0


# Keep the older type name available inside suite modules while the concrete base class stays explicit.
InferenceSession = BaseInferenceSession
