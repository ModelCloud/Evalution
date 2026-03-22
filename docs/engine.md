# Engine Guide

Evalution runs every backend through two base contracts:

- `BaseEngine`
- `BaseInferenceSession`

If you want to add another backend such as vLLM, SGLang, TensorRT-LLM, ROCm-specific runtimes, or
oneDNN-backed Intel inference, implement those two layers and keep the rest of the runtime
unchanged.

## Overview

`BaseEngine` owns backend configuration.

- It receives engine-level controls such as dtype, device, parallelism, memory knobs, or backend
  toggles.
- It builds a reusable `BaseInferenceSession` for one `Model`.
- It serializes engine configuration into the public result payload with `to_dict()`.

`BaseInferenceSession` owns the live runtime state.

- It loads or attaches to the model runtime.
- It implements the common inference APIs used by every suite.
- It releases reusable caches on `gc()`.
- It tears down heavyweight state on `close()`.

The runtime only depends on these contracts. It does not need engine-specific branches if the
engine implements them correctly.

## Required Interfaces

An engine class must inherit `BaseEngine` and implement:

```python
from dataclasses import dataclass

from evalution.config import Model
from evalution.engines.base import BaseEngine, BaseInferenceSession


@dataclass(slots=True)
class MyEngine(BaseEngine):
    def build(self, model: Model) -> BaseInferenceSession:
        return MySession(...)
```

A session class must inherit `BaseInferenceSession` and implement:

- `generate(...)`
- `loglikelihood(...)`
- `loglikelihood_rolling(...)`
- `generate_continuous(...)`
- `gc()`
- `close()`

`describe_execution()` is optional but strongly recommended. Evalution logs its return value and
stores it under `result.engine["execution"]`.

## Request And Response Types

Evalution already defines the shared request/response dataclasses in
`evalution.engines.base`:

- `GenerationRequest`
- `GenerationOutput`
- `LoglikelihoodRequest`
- `LoglikelihoodOutput`
- `RollingLoglikelihoodRequest`
- `RollingLoglikelihoodOutput`

Do not introduce engine-specific request types into suite execution. Translate these shared objects
into backend-native calls inside your session implementation.

## Generation Requirements

`generate(requests, batch_size=...)` must:

- preserve request order
- return one `GenerationOutput` per input request
- respect `request.stop`
- respect `request.max_new_tokens`
- handle either `prompt` or `messages`
- propagate useful per-sample metadata when available

If your backend cannot natively consume chat messages, render them into plain text inside the
session before submission.

## Continuous Generation Requirements

`generate_continuous(requests, batch_size=...)` accepts an iterable of:

```python
(request_id, GenerationRequest)
```

It must yield:

```python
(request_id, GenerationOutput)
```

Requirements:

- preserve the caller-provided `request_id`
- yield one result for each submitted request
- support streaming completion order, not only input order
- keep behavior compatible with `generate(...)`

If the backend does not support real continuous batching, emulate it with fixed batches. That is
how `TransformerCompat` works today. The suite runtime depends on the method shape, not on a
specific batching implementation.

## Log-Likelihood Requirements

`loglikelihood(requests, batch_size=...)` must score the continuation conditioned on the context.

Each output must provide:

- `logprob`: summed continuation log-probability
- `is_greedy`: whether greedy decoding agrees with the continuation tokens
- `token_count`: scored continuation token count

This API powers multiple-choice suites such as `hellaswag`, `piqa`, `mmlu`, and GLUE-style
classification tasks.

`loglikelihood_rolling(requests, batch_size=...)` must score a full text span token by token. It is
used for perplexity-style tasks.

If the backend only exposes token logprobs on generation APIs, build an internal scorer that still
returns the shared output objects.

## Resource Management

`gc()` should release reusable intermediate state between suites without unloading the whole model.

Examples:

- KV cache blocks
- paged-attention managers
- compiled graph caches
- temporary scoring buffers
- allocator-cached memory the backend can safely return

`close()` should fully release the session.

Examples:

- unload model handles
- stop scheduler threads
- close RPC clients
- release tokenizer/runtime references

Keep comments precise. Only document thread-safety or memory behavior that the implementation
actually guarantees.

## Execution Metadata

If your session implements:

```python
def describe_execution(self) -> dict[str, Any] | None:
    ...
```

return stable, user-facing metadata such as:

- effective attention backend
- batching backend
- tensor parallel size
- KV cache mode
- device placement
- backend-specific fallback mode

Avoid returning transient counters or object ids that would make tests noisy.

## Recommended Structure

For non-trivial engines, keep shared helpers outside the concrete engine class.

Suggested layout:

- `my_engine.py`: public engine and session classes
- `my_engine_common.py`: tokenizer, request translation, scoring, and batching helpers
- `memory.py` or similar: backend-specific memory probing if needed

This keeps the engine class readable and makes it easier to split compat and modern variants without
mixing their control paths.

## Minimal Skeleton

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    BaseEngine,
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
)


@dataclass(slots=True)
class MyEngine(BaseEngine):
    device: str | None = None

    def build(self, model: Model) -> BaseInferenceSession:
        return MySession(model_path=model.path, device=self.device)


@dataclass(slots=True)
class MySession(BaseInferenceSession):
    model_path: str
    device: str | None = None

    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        raise NotImplementedError

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        raise NotImplementedError

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size: int | None = None):
        raise NotImplementedError

    def gc(self) -> None:
        return None

    def close(self) -> None:
        return None

    def describe_execution(self) -> dict[str, Any] | None:
        return {
            "backend": "my_engine",
            "device": self.device,
        }
```

## Validation Checklist

Before wiring a new engine into YAML or public exports, verify:

- it inherits `BaseEngine`
- `build()` returns a `BaseInferenceSession`
- `generate(...)` works for plain prompts and chat-style messages
- `generate_continuous(...)` returns `(request_id, output)` pairs correctly
- `loglikelihood(...)` handles empty and non-empty contexts correctly
- `gc()` is safe between suites
- `close()` is idempotent
- `describe_execution()` is stable enough for result snapshots and tests

## Integration Points

Once the engine is implemented:

1. Export it from `evalution/engines/__init__.py`.
2. Export it from `evalution/__init__.py` if it is part of the public API.
3. Add YAML factory support in `evalution/yaml.py` if you want declarative execution.
4. Add unit tests for the engine contract.
5. Add at least one runtime/API test that exercises it through `evalution.engine(...).model(...).run(...)`.

## Current Reference Implementations

Use these as concrete examples:

- `evalution/engines/transformer.py`: modern `transformers` backend with paged attention and
  continuous batching
- `evalution/engines/transformer_compat.py`: fixed-batch compatibility backend that still emulates
  the continuous generation API
- `evalution/engines/transformers_common.py`: shared request preparation, generation, and scoring
  helpers for the two transformer backends
