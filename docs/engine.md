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
how `TransformersCompat` works today. The suite runtime depends on the method shape, not on a
specific batching implementation.

## Built-in Engines

Evalution ships these built-in engines:

- `engines.Transformers()`: the modern backend
- `engines.TransformersCompat()`: the compatibility backend
- `engines.GPTQModel()`: the quantized GPTQModel backend
- `engines.OpenAICompatible()`: an OpenAI-style HTTP backend
- `engines.OpenVINO()`: the Optimum Intel OpenVINO backend
- `engines.SGLang()`: the in-process SGLang runtime backend
- `engines.TensorRTLLM()`: the TensorRT-LLM runtime backend
- `engines.VLLM()`: the vLLM runtime backend

The preferred import shape is:

```python
import evalution as eval
import evalution.engines as engines
```

`engines.Transformers()` defaults batching, paged attention, dtype resolution, and attention
selection to auto behavior. On compatible CUDA `transformers` setups it can switch to paged
continuous batching for `flash_attention_2`.

If the installed `transformers` build predates the first release that includes
`generation/continuous_batching` (`4.56.0`), `engines.Transformers()` falls back to
`engines.TransformersCompat()`. `engines.TransformersCompat()` also remains available as the
explicit fixed-batch execution path.

The modern `engines.Transformers(...)` engine also exposes the upstream continuous batching manager
knobs `manual_eviction`, `allow_block_sharing`, `max_blocks_per_request`, `use_async_batching`,
`use_cuda_graph`, `q_padding_interval_size`, `kv_padding_interval_size`, and `max_cached_graphs`.
When `attn_implementation` resolves to paged FlashAttention and `max_blocks_per_request` is left
unset in Evalution, the engine seeds the block-table decode fast path defaults it needs for that
runtime. Evalution also keeps a compatibility monkeypatch for `transformers` builds that still
need FA2 decode-fast-path enablement, and that fallback defaults `use_cuda_graph=False`. Evalution
keeps a session-owned manager alive while stop
strings and sampling settings stay compatible, then tears it down on `gc()` between suites or on
`close()`.

`engines.GPTQModel()` loads quantized checkpoints through GPTQModel's native loader, then reuses
the same shared generation, scoring, and paged continuous-batching path as the built-in
transformer engines when the loaded quantized model exposes the required HF hooks. It also
surfaces the resolved quantized runtime backend in execution metadata.

`engines.OpenAICompatible()` talks to an OpenAI-compatible HTTP endpoint for generation, while
using Evalution-specific `/v1/eval/loglikelihood` and `/v1/eval/loglikelihood/rolling` routes for
the scoring APIs that Evalution benchmarks require. Evalution also ships
`engines.build_openai_compatible_server(...)` to wrap a local engine session, such as
`engines.Transformers()`, in a queued microbatching HTTP server for local testing. The OpenAI
engine defaults to `batch_size=4`, treating that as the client-side in-flight queue depth: it
submits up to four requests at once and injects the next queued request whenever one result comes
back. For this engine, Evalution's required `.model(path=...)` call is translated into the remote
OpenAI-compatible HTTP `model` argument. Set `batch_size=0` to disable this emulated batching and
fall back to single requests.

`engines.OpenVINO()` loads decoder-only models through `optimum.intel.openvino.OVModelForCausalLM`
while reusing Evalution's shared transformer-style generation, log-likelihood, and rolling
log-likelihood session logic.

`engines.SGLang()` loads the runtime through `sglang.Engine(...)`, keeps generation and
log-likelihood execution fully in process, and normalizes SGLang's generation and prompt-logprob
responses into Evalution's shared output objects. 

`engines.TensorRTLLM()` loads the runtime through `tensorrt_llm.LLM(...)`, keeps tokenizer-based
request preparation inside Evalution, and implements generation plus log-likelihood scoring through
TensorRT-LLM request outputs and prompt log-probabilities when the installed runtime exposes them.
If the runtime does not expose request-level scheduling primitives, the engine falls back to
fixed-batch emulation for `generate_continuous(...)`.

`engines.VLLM()` loads the runtime through `vllm.LLM(...)`, keeps a tokenizer for prompt rendering
and scoring prep, and implements generation plus both log-likelihood APIs through vLLM-native
calls. When request-level engine methods are available, Evalution drives continuous batching by
submitting request ids into `llm_engine.add_request(...)`, reading finished outputs from
`llm_engine.step()`, and reconciling completions by request id instead of by positional order. If
the installed vLLM runtime does not expose that lower-level request API, the engine falls back to
fixed-batch emulation to preserve the `generate_continuous(...)` contract.

The built-in vLLM engine currently expects `num_beams=1`. Benchmarks or custom requests that
require beam search should use another engine until vLLM beam routing is added to this backend.


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
5. Add at least one runtime/API test that exercises it through `engine.model(...).run(...)`.

## Current Reference Implementations

Use these as concrete examples:

- `evalution/engines/transformers.py`: modern `transformers` backend with paged attention and
  continuous batching
- `evalution/engines/transformers_compat.py`: fixed-batch compatibility backend that still emulates
  the continuous generation API
- `evalution/engines/transformers_common.py`: shared request preparation, generation, and scoring
  helpers for the two transformer backends
