# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import gc
import importlib
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, islice
from pathlib import Path
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    BaseEngineDeviceConfig,
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
    SharedEngineConfig,
)
from evalution.engines.transformers_common import (
    _friendly_batch_size,
    _load_tokenizer_from_model,
    _normalize_batch_size,
    _resolve_tokenizer_source,
    _truncate_at_stop,
)

# Keep the engine defaults explicit at module scope.
_AUTO_BATCH_SIZE = "auto"
_A100_CLASS_MIN_VRAM_BYTES = 90 * 1024**3


@dataclass(frozen=True, slots=True)
class _TinygradRuntimeProfile:
    """Describe the JIT controls pinned for one tinygrad runtime profile."""

    name: str
    jit: int | None
    jitbeam: int | None


# Local RTX 4090 and A100 sweeps found that graph-free JIT is the only stable CUDA default for
# this LLM path. JIT=1 trips CUDA graph construction, and JITBEAM=2 failed to finish within the
# benchmark timeout on both cards, so keep the beam search off by default for runtime startup.
_DEFAULT_TINYGRAD_CUDA_PROFILE = _TinygradRuntimeProfile(name="cuda_default", jit=2, jitbeam=0)
_TINYGRAD_CUDA_PROFILES = {
    "rtx4090": _TinygradRuntimeProfile(name="rtx4090", jit=2, jitbeam=0),
    "a100": _TinygradRuntimeProfile(name="a100", jit=2, jitbeam=0),
    "cuda_default": _DEFAULT_TINYGRAD_CUDA_PROFILE,
}


@dataclass(frozen=True, slots=True)
class _TinygradModules:
    """Bundle the imported tinygrad modules so the session can use one stable runtime view."""

    tinygrad: Any
    Tensor: Any
    Device: Any
    helpers: Any
    nn_state: Any
    llm_model: Any
    llm_cli: Any
    dtypes: Any


@dataclass(frozen=True, slots=True)
class _LoadedTinygradRuntime:
    """Capture the live tinygrad runtime state produced by engine construction."""

    modules: _TinygradModules
    model: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    load_format: str
    model_type: str
    compute_device: str
    max_context: int
    runtime_profile: _TinygradRuntimeProfile


@dataclass(slots=True)
class Tinygrad(BaseEngineDeviceConfig, SharedEngineConfig):
    """Configure Evalution to run GGUF generation and scoring through tinygrad."""

    # Tinygrad support stays optional so the core install remains lean.
    max_context: int | None = None
    jit: int | None = None
    jitbeam: int | None = None

    def build(self, model: Model) -> BaseInferenceSession:
        """Construct a tinygrad-backed inference session for the requested model."""

        self.resolved_engine = "Tinygrad"
        return TinygradSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine configuration for reporting and YAML output."""

        return asdict(self)


@dataclass(slots=True)
class TinygradSession(BaseInferenceSession):
    """Own one live GGUF tinygrad LLM runtime plus Evalution request translation logic."""

    # Keep the runtime and tokenizer state on the session so one built engine can be reused
    # across multiple suites without rebuilding the model every time.
    config: Tinygrad
    model_config: Model
    modules: _TinygradModules
    model: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    load_format: str
    model_type: str
    compute_device: str
    max_context: int
    runtime_profile: _TinygradRuntimeProfile
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Tinygrad, model_config: Model) -> TinygradSession:
        """Load tinygrad, build the model runtime, and attach the tokenizer helpers."""

        runtime = _load_tinygrad_runtime(config, model_config)
        return cls(
            config=config,
            model_config=model_config,
            modules=runtime.modules,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            load_format=runtime.load_format,
            model_type=runtime.model_type,
            compute_device=runtime.compute_device,
            max_context=runtime.max_context,
            runtime_profile=runtime.runtime_profile,
        )

    @property
    def batch_size(self) -> int | str:
        """Expose the configured batch-size policy for compatibility with other engines."""

        return self.config.batch_size

    def describe_execution(self) -> dict[str, Any]:
        """Report the runtime settings that influence tinygrad execution behavior."""

        execution = {
            "generation_backend": "tinygrad_generate",
            "load_format": self.load_format,
            "model_type": self.model_type,
            "device": self.compute_device,
            "max_model_len": self.max_context,
            "jit_profile": self.runtime_profile.name,
        }
        if self.runtime_profile.jit is not None:
            execution["jit"] = self.runtime_profile.jit
        if self.runtime_profile.jitbeam is not None:
            execution["jitbeam"] = self.runtime_profile.jitbeam
        return execution

    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        """Resolve an explicit or heuristic batch size for the given request set."""

        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return int(configured_batch_size)
        return min(len(requests) or 1, _friendly_batch_size(len(requests) or 1))

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        """Normalize prompts and token ids before generation reaches the runtime."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        with self._tokenizer_lock:
            prepared: list[GenerationRequest] = []
            for request in requests:
                rendered_prompt = request.rendered_prompt
                input_ids: list[int] | None = None

                if request.input_ids is not None:
                    input_ids = list(request.input_ids)
                elif request.messages is not None and self.prepare_tokenizer is None and _is_tinygrad_simple_tokenizer(self.tokenizer):
                    input_ids = self._encode_chat_messages_with_tinygrad_tokenizer(request)
                elif request.messages is not None and self.prepare_tokenizer is not None:
                    input_ids = self._tokenize_chat_messages_with_tokenizer(tokenizer, request)
                    if rendered_prompt is None:
                        rendered_prompt = self._render_request_with_tokenizer(tokenizer, request)
                else:
                    if rendered_prompt is None:
                        rendered_prompt = self._render_request_with_tokenizer(tokenizer, request)
                    input_ids = self._tokenize_generation_text(rendered_prompt, request)

                prepared.append(
                    replace(
                        request,
                        rendered_prompt=rendered_prompt,
                        input_ids=input_ids,
                    )
                )
            return prepared

    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate one completion per request while preserving caller order."""

        if not requests:
            return []

        prepared_requests = self.prepare_requests(requests)
        effective_batch_size = (
            max(int(batch_size), 1)
            if batch_size is not None
            else self.resolve_batch_size(prepared_requests)
        )
        outputs: list[GenerationOutput | None] = [None] * len(prepared_requests)
        with self._generation_lock:
            for generation_batch in self._iter_generation_batches(
                prepared_requests,
                batch_size=effective_batch_size,
            ):
                batch_outputs = self._generate_batch(
                    [request for _, request in generation_batch]
                )
                for (request_index, _request), output in zip(
                    generation_batch,
                    batch_outputs,
                    strict=True,
                ):
                    outputs[request_index] = output
        return [output for output in outputs if output is not None]

    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Emulate continuous generation with same-thread fixed batches."""

        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            request_iter = iter(requests)
            if batch_size is not None:
                effective_batch_size = max(int(batch_size), 1)
                items = request_iter
            else:
                preview_items = list(islice(request_iter, 64))
                if not preview_items:
                    return
                effective_batch_size = self.resolve_batch_size([request for _, request in preview_items])
                items = chain(preview_items, request_iter)

            pending_batch: list[tuple[Any, GenerationRequest]] = []
            for request_key, request in items:
                pending_batch.append((request_key, request))
                if len(pending_batch) < effective_batch_size:
                    continue
                outputs = self.generate(
                    [batched_request for _, batched_request in pending_batch],
                    batch_size=effective_batch_size,
                )
                for (batched_key, _request), output in zip(pending_batch, outputs, strict=True):
                    yield batched_key, output
                pending_batch = []
            if pending_batch:
                outputs = self.generate(
                    [batched_request for _, batched_request in pending_batch],
                    batch_size=effective_batch_size,
                )
                for (batched_key, _request), output in zip(pending_batch, outputs, strict=True):
                    yield batched_key, output

        return iterator()

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score continuations token by token through tinygrad logits."""

        if not requests:
            return []

        del batch_size
        prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
        chunk_counts: list[int] = []
        chunk_outputs: list[list[LoglikelihoodOutput]] = []
        with self._generation_lock:
            for prefix_ids, target_ids, metadata in prepared_requests:
                request_chunks = self._build_loglikelihood_chunks(
                    prefix_ids=prefix_ids,
                    target_ids=target_ids,
                    metadata=metadata,
                )
                chunk_counts.append(len(request_chunks))
                for input_ids, score_start, score_count, chunk_metadata in request_chunks:
                    chunk_outputs.append(
                        [
                            self._score_loglikelihood_chunk(
                                input_ids=input_ids,
                                score_start=score_start,
                                score_count=score_count,
                                metadata=chunk_metadata,
                            )
                        ]
                    )

        outputs: list[LoglikelihoodOutput] = []
        cursor = 0
        for request, chunk_count in zip(requests, chunk_counts, strict=True):
            request_chunk_outputs = [
                chunk_outputs[cursor + index][0]
                for index in range(chunk_count)
            ]
            cursor += chunk_count
            outputs.append(
                LoglikelihoodOutput(
                    logprob=sum(output.logprob for output in request_chunk_outputs),
                    is_greedy=all(output.is_greedy for output in request_chunk_outputs),
                    token_count=sum(output.token_count for output in request_chunk_outputs),
                    metadata=dict(request.metadata),
                )
            )
        return outputs

    def loglikelihood_continuous(
        self,
        requests: Iterable[tuple[Any, LoglikelihoodRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, LoglikelihoodOutput]]:
        """Emulate continuous log-likelihood scoring with same-thread fixed batches."""

        def iterator() -> Iterator[tuple[Any, LoglikelihoodOutput]]:
            request_iter = iter(requests)
            if batch_size is not None:
                effective_batch_size = max(int(batch_size), 1)
                items = request_iter
            else:
                preview_items = list(islice(request_iter, 64))
                if not preview_items:
                    return
                effective_batch_size = min(len(preview_items), _friendly_batch_size(len(preview_items)))
                items = chain(preview_items, request_iter)

            pending_batch: list[tuple[Any, LoglikelihoodRequest]] = []
            for request_key, request in items:
                pending_batch.append((request_key, request))
                if len(pending_batch) < effective_batch_size:
                    continue
                outputs = self.loglikelihood(
                    [batched_request for _, batched_request in pending_batch],
                    batch_size=effective_batch_size,
                )
                for (batched_key, _request), output in zip(pending_batch, outputs, strict=True):
                    yield batched_key, output
                pending_batch = []
            if pending_batch:
                outputs = self.loglikelihood(
                    [batched_request for _, batched_request in pending_batch],
                    batch_size=effective_batch_size,
                )
                for (batched_key, _request), output in zip(pending_batch, outputs, strict=True):
                    yield batched_key, output

        return iterator()

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        """Score rolling log-likelihood requests token by token for perplexity-style suites."""

        del batch_size
        if not requests:
            return []

        scoring_requests: list[LoglikelihoodRequest] = []
        request_window_counts: list[int] = []
        with self._tokenizer_lock:
            for request in requests:
                token_list = (
                    list(request.input_ids)
                    if request.input_ids is not None
                    else self._tokenize_rolling_text(request.text)
                )
                windows = list(self._rolling_loglikelihood_windows(token_list))
                request_window_counts.append(len(windows))
                for context_ids, continuation_ids in windows:
                    scoring_requests.append(
                        LoglikelihoodRequest(
                            context_input_ids=context_ids,
                            continuation_input_ids=continuation_ids,
                            metadata=dict(request.metadata),
                        )
                    )

        scored = self.loglikelihood(scoring_requests)
        outputs: list[RollingLoglikelihoodOutput] = []
        cursor = 0
        for request, window_count in zip(requests, request_window_counts, strict=True):
            if window_count == 0:
                outputs.append(
                    RollingLoglikelihoodOutput(
                        logprob=0.0,
                        token_count=0,
                        metadata=dict(request.metadata),
                    )
                )
                continue

            request_outputs = scored[cursor : cursor + window_count]
            cursor += window_count
            outputs.append(
                RollingLoglikelihoodOutput(
                    logprob=sum(output.logprob for output in request_outputs),
                    token_count=sum(output.token_count for output in request_outputs),
                    metadata=dict(request.metadata),
                )
            )
        return outputs

    def gc(self) -> None:
        """Reset reusable tinygrad KV caches between suites."""

        with self._generation_lock:
            self._reset_model_cache()
        gc.collect()

    def close(self) -> None:
        """Release the resources owned by this session."""

        if self._closed:
            return
        self._closed = True
        with self._generation_lock:
            self._reset_model_cache()
            self.model = None
            self.tokenizer = None
            self.prepare_tokenizer = None
        gc.collect()

    def _generate_one(self, request: GenerationRequest) -> GenerationOutput:
        """Generate one completion by running tinygrad autoregressively with reusable KV cache."""

        return self._generate_batch([request])[0]

    def _iter_generation_batches(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> Iterator[list[tuple[int, GenerationRequest]]]:
        """Group prepared generation requests into safe static batches."""

        grouped: dict[tuple[bool, float, int], list[tuple[int, GenerationRequest]]] = {}
        for request_index, request in enumerate(requests):
            signature = self._generation_batch_signature(request)
            grouped.setdefault(signature, []).append((request_index, request))

        for grouped_requests in grouped.values():
            for start in range(0, len(grouped_requests), batch_size):
                yield grouped_requests[start : start + batch_size]

    def _generation_batch_signature(self, request: GenerationRequest) -> tuple[bool, float, int]:
        """Return the static-batching signature that must match before requests can share one lane group."""

        if request.num_beams != 1:
            raise ValueError("Tinygrad engine currently requires num_beams=1")
        if request.input_ids is None:
            raise ValueError("prepared generation requests must include input_ids")
        if len(request.input_ids) <= 0:
            raise ValueError("generation requests must provide at least one prompt token")
        max_new_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens
        )
        return (
            bool(request.do_sample),
            float(request.temperature),
            int(max_new_tokens),
        )

    def _generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        """Generate one static batch through tinygrad's batched forward path."""

        if not requests:
            return []

        batch_signature = self._generation_batch_signature(requests[0])
        do_sample, temperature, _max_new_tokens = batch_signature
        for request in requests[1:]:
            if self._generation_batch_signature(request) != batch_signature:
                raise ValueError("Tinygrad static batching requires matching sampling settings")

        prompt_token_ids = [
            [int(token_id) for token_id in request.input_ids or []]
            for request in requests
        ]
        prompt_lengths = [len(token_ids) for token_ids in prompt_token_ids]
        if len(set(prompt_lengths)) == 1:
            return self._generate_uniform_prompt_batch(
                requests=requests,
                prompt_token_ids=prompt_token_ids,
                prompt_lengths=prompt_lengths,
                do_sample=do_sample,
                temperature=temperature,
            )
        return self._generate_mixed_prompt_batch(
            requests=requests,
            prompt_token_ids=prompt_token_ids,
            prompt_lengths=prompt_lengths,
            do_sample=do_sample,
            temperature=temperature,
        )

    def _generate_uniform_prompt_batch(
        self,
        *,
        requests: list[GenerationRequest],
        prompt_token_ids: list[list[int]],
        prompt_lengths: list[int],
        do_sample: bool,
        temperature: float,
    ) -> list[GenerationOutput]:
        """Generate one static batch when every row shares the same prompt length."""

        prompt_length = prompt_lengths[0]
        generation_limits = [
            max(
                min(
                    int(request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens),
                    max(int(self.max_context) - prompt_length, 0),
                ),
                0,
            )
            for request in requests
        ]

        generated_token_ids = [[] for _ in requests]
        generated_texts = [""] * len(requests)
        finish_reasons = ["length"] * len(requests)
        finished = [limit <= 0 for limit in generation_limits]
        finished_row_input_id = self._rollout_filler_token_id()

        self._reset_model_cache(reset_jit=True)
        if not all(finished):
            with self._runtime_context():
                current_tokens, current_position = self._prefill_batch(
                    prompt_token_ids=prompt_token_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                while True:
                    self._consume_generation_step(
                        requests=requests,
                        current_tokens=current_tokens,
                        generated_token_ids=generated_token_ids,
                        generated_texts=generated_texts,
                        finish_reasons=finish_reasons,
                        finished=finished,
                        generation_limits=generation_limits,
                    )
                    if all(finished):
                        break
                    if current_position >= int(self.max_context):
                        break
                    current_tokens = self._batched_next_token_ids(
                        [
                            [finished_row_input_id if finished[row_index] else token_id]
                            for row_index, token_id in enumerate(current_tokens)
                        ],
                        start_pos=current_position,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                    current_position += 1

        return self._build_generation_outputs(
            requests=requests,
            prompt_lengths=prompt_lengths,
            generated_token_ids=generated_token_ids,
            generated_texts=generated_texts,
            finish_reasons=finish_reasons,
        )

    def _generate_mixed_prompt_batch(
        self,
        *,
        requests: list[GenerationRequest],
        prompt_token_ids: list[list[int]],
        prompt_lengths: list[int],
        do_sample: bool,
        temperature: float,
    ) -> list[GenerationOutput]:
        """Generate one static batch while letting shorter prompts enter decode before longer prompts finish prefill."""

        generation_limits = [
            max(
                min(
                    int(request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens),
                    max(int(self.max_context) - prompt_length, 0),
                ),
                0,
            )
            for request, prompt_length in zip(requests, prompt_lengths, strict=True)
        ]
        generated_token_ids = [[] for _ in requests]
        generated_texts = [""] * len(requests)
        finish_reasons = ["length"] * len(requests)
        finished = [limit <= 0 for limit in generation_limits]
        finished_row_input_id = self._rollout_filler_token_id()

        self._reset_model_cache(reset_jit=True)
        current_position = 0
        with self._runtime_context():
            while not all(finished) and current_position < int(self.max_context):
                step_token_rows: list[list[int]] = []
                for row_index, prompt_ids in enumerate(prompt_token_ids):
                    if finished[row_index]:
                        step_token_rows.append([finished_row_input_id])
                        continue
                    prompt_length = prompt_lengths[row_index]
                    if current_position < prompt_length:
                        step_token_rows.append([prompt_ids[current_position]])
                        continue
                    if not generated_token_ids[row_index]:
                        raise RuntimeError("tinygrad mixed-length batch decode requires a generated seed token")
                    step_token_rows.append([generated_token_ids[row_index][-1]])

                current_tokens = self._batched_next_token_ids(
                    step_token_rows,
                    start_pos=current_position,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                self._consume_generation_step(
                    requests=requests,
                    current_tokens=current_tokens,
                    generated_token_ids=generated_token_ids,
                    generated_texts=generated_texts,
                    finish_reasons=finish_reasons,
                    finished=finished,
                    generation_limits=generation_limits,
                    current_position=current_position,
                    prompt_lengths=prompt_lengths,
                )
                current_position += 1

        return self._build_generation_outputs(
            requests=requests,
            prompt_lengths=prompt_lengths,
            generated_token_ids=generated_token_ids,
            generated_texts=generated_texts,
            finish_reasons=finish_reasons,
        )

    def _build_generation_outputs(
        self,
        *,
        requests: list[GenerationRequest],
        prompt_lengths: list[int],
        generated_token_ids: list[list[int]],
        generated_texts: list[str],
        finish_reasons: list[str],
    ) -> list[GenerationOutput]:
        """Materialize per-request GenerationOutput objects after one batched decode pass completes."""

        outputs: list[GenerationOutput] = []
        for request, prompt_length, output_token_ids, generated_text, finish_reason in zip(
            requests,
            prompt_lengths,
            generated_token_ids,
            generated_texts,
            finish_reasons,
            strict=True,
        ):
            outputs.append(
                GenerationOutput(
                    prompt=request.rendered_prompt or request.prompt or str(request.messages),
                    text=generated_text,
                    metadata={
                        **dict(request.metadata),
                        "finish_reason": finish_reason,
                        "prompt_token_count": prompt_length,
                        "completion_token_count": len(output_token_ids),
                    },
                )
            )
        return outputs

    def _consume_generation_step(
        self,
        *,
        requests: list[GenerationRequest],
        current_tokens: list[int],
        generated_token_ids: list[list[int]],
        generated_texts: list[str],
        finish_reasons: list[str],
        finished: list[bool],
        generation_limits: list[int],
        current_position: int | None = None,
        prompt_lengths: list[int] | None = None,
    ) -> None:
        """Apply one batched decode step to the per-request generation state."""

        for row_index, token_id in enumerate(current_tokens):
            if finished[row_index]:
                continue
            if prompt_lengths is not None and current_position is not None:
                # Prompt-only positions should update KV cache but must not count as generated text yet.
                if current_position < prompt_lengths[row_index] - 1:
                    continue
            if self._is_end_token(token_id):
                finish_reasons[row_index] = "stop"
                finished[row_index] = True
                generated_texts[row_index] = _truncate_at_stop(
                    self._decode_tokens(generated_token_ids[row_index]),
                    requests[row_index].stop,
                )
                continue

            generated_token_ids[row_index].append(int(token_id))
            generated_text = self._decode_tokens(generated_token_ids[row_index])
            truncated_text = _truncate_at_stop(generated_text, requests[row_index].stop)
            generated_texts[row_index] = truncated_text
            if truncated_text != generated_text:
                finish_reasons[row_index] = "stop"
                finished[row_index] = True
                continue

            if len(generated_token_ids[row_index]) >= generation_limits[row_index]:
                finish_reasons[row_index] = "length"
                finished[row_index] = True

    def _uses_packaged_llm_runtime(self) -> bool:
        """Detect the packaged tinygrad LLM runtime used for GGUF and generic loader paths."""

        return all(
            hasattr(self.model, attr_name)
            for attr_name in ("blk", "token_embd", "output_norm")
        )

    def _batched_next_token_ids(
        self,
        token_ids: list[list[int]],
        *,
        start_pos: int,
        do_sample: bool,
        temperature: float,
    ) -> list[int]:
        """Run one batched tinygrad forward pass and return one sampled token id per row."""

        token_tensor = self.modules.Tensor(
            [list(row) for row in token_ids],
            dtype=self.modules.dtypes.int32,
            device=self.compute_device,
        )

        temperature_value = temperature if do_sample else 0.0
        temperature_tensor = self.modules.Tensor(float(temperature_value), device=self.compute_device).contiguous()
        start_pos_uop = self.modules.tinygrad.UOp.variable(
            "start_pos",
            0,
            int(self.max_context) - 1,
        ).bind(int(start_pos))
        next_token_tensor = self.model(token_tensor, start_pos_uop, temperature_tensor).realize()
        token_rows = next_token_tensor.tolist()
        return [int(row[0] if isinstance(row, list) else row) for row in token_rows]

    def _prefill_batch(
        self,
        *,
        prompt_token_ids: list[list[int]],
        do_sample: bool,
        temperature: float,
    ) -> tuple[list[int], int]:
        """Mirror tinygrad's chunked prompt prefill before the autoregressive rollout phase."""

        chunk_size = 1 if bool(getattr(self.model, "has_recurrent_block", False)) else 32
        prompt_length = len(prompt_token_ids[0])
        current_position = 0
        current_tokens: list[int] = []
        previous_chunk_length: int | None = None
        while current_position < prompt_length:
            chunk_end = min(current_position + chunk_size, prompt_length)
            current_chunk_length = chunk_end - current_position
            if previous_chunk_length is not None and current_chunk_length != previous_chunk_length:
                prefill_jit = getattr(self.model, "prefill_jit", None)
                if prefill_jit is not None and hasattr(prefill_jit, "reset"):
                    prefill_jit.reset()
            current_tokens = self._batched_next_token_ids(
                [row[current_position:chunk_end] for row in prompt_token_ids],
                start_pos=current_position,
                do_sample=do_sample,
                temperature=temperature,
            )
            previous_chunk_length = current_chunk_length
            current_position = chunk_end
        return current_tokens, current_position

    def _forward_batch_logits(
        self,
        token_ids: list[list[int]],
        *,
        start_pos: int,
    ) -> Any:
        """Run one batched forward pass and return full logits for the provided token rows."""

        token_tensor = self.modules.Tensor(
            [list(row) for row in token_ids],
            dtype=self.modules.dtypes.int32,
            device=self.compute_device,
        )
        if self._uses_packaged_llm_runtime():
            x = self.model.token_embd(token_tensor).float()
            for block in self.model.blk:
                x = block(x, start_pos)
            return self.model.output(self.model.output_norm(x)).realize()
        raise ValueError("unsupported tinygrad model runtime for batched logits")

    def _sample_next_token_ids_from_logits(
        self,
        logits: Any,
        *,
        do_sample: bool,
        temperature: float,
    ) -> list[int]:
        """Convert one logits matrix into sampled or greedy next-token ids row by row."""

        if do_sample:
            temperature_value = max(float(temperature), 1e-12)
            gumbel_noise = (self.modules.Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()
            next_token_tensor = (logits / temperature_value - gumbel_noise).argmax(axis=-1).realize()
        else:
            next_token_tensor = logits.argmax(axis=-1).realize()
        token_list = next_token_tensor.tolist()
        return [int(token_id) for token_id in token_list]

    def _rollout_filler_token_id(self) -> int:
        """Return a stable token id for finished rows that remain in a static batch."""

        tokenizer = self.tokenizer
        if _is_tinygrad_simple_tokenizer(tokenizer):
            for attr_name in ("eos_id", "bos_id"):
                token_id = getattr(tokenizer, attr_name, None)
                if isinstance(token_id, int):
                    return int(token_id)
            with suppress(Exception):
                prefix = list(tokenizer.prefix())
                if prefix:
                    return int(prefix[-1])
            return 0

        for attr_name in ("eos_token_id", "pad_token_id", "bos_token_id"):
            token_id = getattr(tokenizer, attr_name, None)
            if isinstance(token_id, int):
                return int(token_id)
            if isinstance(token_id, list) and token_id:
                return int(token_id[0])
        return 0

    def _render_request_with_tokenizer(self, tokenizer: Any, request: GenerationRequest) -> str:
        """Render either a plain prompt or a chat-formatted prompt string."""

        if request.messages is not None:
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise ValueError("generation requests with messages require tokenizer.apply_chat_template")
            return apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        if request.prompt is None:
            raise ValueError("generation requests must define either `prompt` or `messages`")
        return request.prompt

    def _tokenize_chat_messages_with_tokenizer(self, tokenizer: Any, request: GenerationRequest) -> list[int]:
        """Prefer tokenizer-native chat tokenization so Llama special tokens are not double-applied."""

        if request.messages is None:
            raise ValueError("chat tokenization requires request.messages")
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            with suppress(TypeError, ValueError):
                tokenized = apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=request.add_generation_prompt,
                )
                if isinstance(tokenized, Mapping):
                    tokenized = tokenized.get("input_ids")
                elif hasattr(tokenized, "input_ids"):
                    tokenized = tokenized.input_ids
                if tokenized and isinstance(tokenized[0], list):
                    tokenized = tokenized[0]
                if isinstance(tokenized, list):
                    return [int(token_id) for token_id in tokenized]

        rendered_prompt = request.rendered_prompt
        if rendered_prompt is None:
            rendered_prompt = self._render_request_with_tokenizer(tokenizer, request)
        return self._tokenize_generation_text(rendered_prompt, request)

    def _encode_chat_messages_with_tinygrad_tokenizer(self, request: GenerationRequest) -> list[int]:
        """Encode chat messages directly through tinygrad's built-in GGUF tokenizer helpers."""

        if request.messages is None:
            raise ValueError("chat encoding requires request.messages")
        tokenizer = self.tokenizer
        token_ids = list(tokenizer.prefix())
        for index, message in enumerate(request.messages):
            role = message["role"]
            content = message["content"]
            token_ids.extend(tokenizer.role(role))
            token_ids.extend(tokenizer.encode(content))
            is_last_message = index == len(request.messages) - 1
            if role == "assistant" and is_last_message and not request.add_generation_prompt:
                break
            token_ids.extend(tokenizer.end_turn())
        else:
            if request.add_generation_prompt:
                token_ids.extend(tokenizer.role("assistant"))
        return [int(token_id) for token_id in token_ids]

    def _tokenize_generation_text(self, text: str, request: GenerationRequest) -> list[int]:
        """Tokenize one generation prompt according to the active tokenizer family."""

        if _is_tinygrad_simple_tokenizer(self.tokenizer) and request.rendered_prompt is None:
            return self._tokenize_text(text, add_prefix=True)
        if self.prepare_tokenizer is not None and request.messages is not None:
            return self._tokenize_text(text, add_special_tokens=False)
        if request.rendered_prompt is not None:
            return self._tokenize_text(text, add_special_tokens=False)
        return self._tokenize_text(text, add_special_tokens=True)

    def _prepare_loglikelihood_request(
        self,
        request: LoglikelihoodRequest,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        """Convert one scoring request into prefix ids, continuation ids, and copied metadata."""

        with self._tokenizer_lock:
            if request.context_input_ids is not None:
                prefix_ids = list(request.context_input_ids)
            elif request.context:
                prefix_ids = self._tokenize_loglikelihood_context(request.context)
            else:
                prefix_ids = []

            if request.continuation_input_ids is not None:
                target_ids = list(request.continuation_input_ids)
            elif request.continuation:
                if _is_tinygrad_simple_tokenizer(self.tokenizer):
                    target_ids = self._tokenize_text(request.continuation, add_prefix=False)
                else:
                    target_ids = self._tokenize_text(request.continuation, add_special_tokens=False)
            else:
                target_ids = []

        if not target_ids:
            raise ValueError("loglikelihood requests must provide a non-empty continuation")
        return prefix_ids, target_ids, dict(request.metadata)

    def _tokenize_loglikelihood_context(self, text: str) -> list[int]:
        """Tokenize scored context text while avoiding a doubled synthetic prefix token."""

        if _is_tinygrad_simple_tokenizer(self.tokenizer):
            return self._tokenize_text(text, add_prefix=True)
        return self._tokenize_text(text, add_special_tokens=True)

    def _tokenize_rolling_text(self, text: str) -> list[int]:
        """Tokenize rolling-perplexity text without prepending an extra synthetic prefix token."""

        if _is_tinygrad_simple_tokenizer(self.tokenizer):
            return self._tokenize_text(text, add_prefix=False)
        return self._tokenize_text(text, add_special_tokens=False)

    def _build_loglikelihood_chunks(
        self,
        *,
        prefix_ids: list[int],
        target_ids: list[int],
        metadata: dict[str, Any],
    ) -> list[tuple[list[int], int, int, dict[str, Any]]]:
        """Split a scored continuation into disjoint windows that fit the active context length."""

        max_input_length = self._max_scoring_input_length()
        max_scored_window = max_input_length + 1
        history_ids = list(prefix_ids)
        if not history_ids:
            history_ids = self._prefix_token_ids()

        chunks: list[tuple[list[int], int, int, dict[str, Any]]] = []
        cursor = 0
        while cursor < len(target_ids):
            remaining = len(target_ids) - cursor
            target_count = min(remaining, max_input_length)
            context_count = min(len(history_ids), max_scored_window - target_count)
            if context_count <= 0:
                raise ValueError("model context window is too small to score continuation tokens")

            context_ids = history_ids[-context_count:]
            continuation_slice = target_ids[cursor : cursor + target_count]
            chunks.append(
                (
                    context_ids + continuation_slice,
                    len(context_ids),
                    len(continuation_slice),
                    dict(metadata),
                )
            )
            history_ids.extend(continuation_slice)
            cursor += target_count
        return chunks

    def _score_loglikelihood_chunk(
        self,
        *,
        input_ids: list[int],
        score_start: int,
        score_count: int,
        metadata: dict[str, Any],
    ) -> LoglikelihoodOutput:
        """Score one continuation chunk by gathering target-token log-probabilities from logits."""

        if score_count <= 0:
            return LoglikelihoodOutput(logprob=0.0, is_greedy=True, token_count=0, metadata=metadata)

        self._reset_model_cache()
        logits = self._forward_logits_for_ids(input_ids[:-1], start_pos=0)
        row_start = max(score_start - 1, 0)
        row_end = row_start + score_count
        target_ids = input_ids[score_start : score_start + score_count]

        scored_logits = logits[0, row_start:row_end, :]
        log_probs = scored_logits.log_softmax(axis=-1)
        target_tensor = self.modules.Tensor(
            target_ids,
            dtype=self.modules.dtypes.int32,
            device=self.compute_device,
        ).reshape(score_count, 1)
        selected = log_probs.gather(-1, target_tensor).reshape(score_count).realize()
        greedy = scored_logits.argmax(axis=-1).realize()

        token_logprobs = [float(value) for value in selected.tolist()]
        greedy_ids = [int(value) for value in greedy.tolist()]
        return LoglikelihoodOutput(
            logprob=sum(token_logprobs),
            is_greedy=greedy_ids == [int(token_id) for token_id in target_ids],
            token_count=score_count,
            metadata=dict(metadata),
        )

    def _rolling_loglikelihood_windows(
        self,
        token_list: list[int],
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Split one token list into the disjoint scoring windows used for rolling perplexity."""

        if not token_list:
            return

        max_seq_len = self._max_scoring_input_length()
        prefix_tokens = self._prefix_token_ids()
        pred_len = max_seq_len
        predicted = 0

        first_seq_len = min(max_seq_len, len(token_list))
        first_window = (
            prefix_tokens + token_list[: max(first_seq_len - len(prefix_tokens), 0)],
            token_list[:first_seq_len],
        )
        yield self._make_disjoint_window(first_window)
        predicted += first_seq_len

        while predicted < len(token_list):
            window_pred_len = min(len(token_list) - predicted, pred_len)
            window_end = predicted + window_pred_len
            window = (
                token_list[window_end - max_seq_len - 1 : window_end - 1],
                token_list[window_end - window_pred_len : window_end],
            )
            yield self._make_disjoint_window(window)
            predicted += window_pred_len

    def _make_disjoint_window(
        self,
        pair: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Trim overlapping context tokens because the scorer already conditions on their prefix."""

        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    def _prefix_token_ids(self) -> list[int]:
        """Return the synthetic prefix token sequence used when a scored request has no context."""

        tokenizer = self.tokenizer
        if _is_tinygrad_simple_tokenizer(tokenizer):
            prefix = list(tokenizer.prefix())
            if prefix:
                return [int(token_id) for token_id in prefix]
            if getattr(tokenizer, "bos_id", None) is not None:
                return [int(tokenizer.bos_id)]
            return [int(getattr(tokenizer, "eos_id", 0))]

        for attr_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(tokenizer, attr_name, None)
            if token_id is None:
                continue
            if isinstance(token_id, int):
                return [int(token_id)]
            if isinstance(token_id, list) and token_id:
                return [int(token_id[0])]
        raise ValueError(
            "Tinygrad token-level scoring requires tokenizer.bos_token_id, eos_token_id, or pad_token_id"
        )

    def _max_scoring_input_length(self) -> int:
        """Return the runtime context window used for prompt evaluation."""

        return max(int(self.max_context) - 1, 1)

    def _tokenize_text(
        self,
        text: str,
        *,
        add_special_tokens: bool | None = None,
        add_prefix: bool = False,
    ) -> list[int]:
        """Tokenize text through either the tinygrad or Hugging Face tokenizer interface."""

        tokenizer = self.tokenizer if self.prepare_tokenizer is None else self.prepare_tokenizer
        if _is_tinygrad_simple_tokenizer(tokenizer):
            token_ids: list[int] = []
            if add_prefix:
                token_ids.extend(int(token_id) for token_id in tokenizer.prefix())
            token_ids.extend(int(token_id) for token_id in tokenizer.encode(text))
            return token_ids

        kwargs: dict[str, Any] = {}
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens

        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                encoded = encode(text, **kwargs)
            except TypeError:
                encoded = encode(text)
            if isinstance(encoded, list):
                return [int(token_id) for token_id in encoded]

        tokenized = tokenizer(text, **kwargs)
        input_ids = tokenized["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token_id) for token_id in input_ids]

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token ids back into text through the active tokenizer family."""

        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            raise ValueError("Tinygrad engine requires tokenizer.decode(...) for generation output")
        return str(decode([int(token_id) for token_id in token_ids]))

    def _is_end_token(self, token_id: int) -> bool:
        """Check whether one token id should terminate generation."""

        if _is_tinygrad_simple_tokenizer(self.tokenizer):
            return bool(self.tokenizer.is_end(int(token_id)))

        stop_ids: set[int] = set()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            stop_ids.add(int(eos_token_id))
        elif isinstance(eos_token_id, list):
            stop_ids.update(int(value) for value in eos_token_id)

        convert_tokens_to_ids = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            with suppress(Exception):
                eot_id = convert_tokens_to_ids("<|eot_id|>")
                unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
                if isinstance(eot_id, int) and eot_id != unk_token_id and eot_id >= 0:
                    stop_ids.add(int(eot_id))
        return int(token_id) in stop_ids

    def _forward_logits_for_ids(self, token_ids: list[int], *, start_pos: int) -> Any:
        """Run the tinygrad transformer forward path and return full logits for the provided ids."""

        if not token_ids:
            raise ValueError("tinygrad forward requires at least one input token")
        with self._runtime_context():
            return self._forward_batch_logits([list(token_ids)], start_pos=start_pos)

    def _reset_model_cache(self, *, reset_jit: bool = False) -> None:
        """Drop cached KV and recurrent state so the next request starts from a clean prompt."""

        if self.model is None:
            return
        for block in getattr(self.model, "blk", []):
            for attr_name in ("cache_kv", "cache_k", "cache_v", "conv_state", "recurrent_state", "freqs_cis"):
                if hasattr(block, attr_name):
                    delattr(block, attr_name)
        if hasattr(self.model, "_cached_tokens"):
            self.model._cached_tokens = []
        if reset_jit:
            for attr_name in ("prefill_jit", "rollout_jit"):
                jit_cache = getattr(self.model, attr_name, None)
                if jit_cache is not None and hasattr(jit_cache, "reset"):
                    jit_cache.reset()
            forward_jit = getattr(self.model, "forward_jit", None)
            if forward_jit is not None and hasattr(forward_jit, "reset"):
                forward_jit.reset()

    def _runtime_context(self):
        """Bind tinygrad's default device while model work is in flight."""

        context_kwargs: dict[str, Any] = {"DEV": self.compute_device}
        if self.runtime_profile.jit is not None:
            context_kwargs["JIT"] = self.runtime_profile.jit

        env_updates = {}
        if self.runtime_profile.jitbeam is not None:
            env_updates["JITBEAM"] = str(self.runtime_profile.jitbeam)

        @contextmanager
        def bound_context() -> Iterator[None]:
            with _temporary_environment(env_updates):
                with self.modules.helpers.Context(**context_kwargs):
                    yield

        return bound_context()


def _load_tinygrad_runtime(config: Tinygrad, model_config: Model) -> _LoadedTinygradRuntime:
    """Load the requested tinygrad runtime and return the reusable session state."""

    # Keep the public contract explicit: the Tinygrad engine is GGUF-only until tinygrad's native
    # dense Hugging Face path is correct on the shared Llama 3.2 benchmark matrix again.
    if not _is_gguf_path(model_config.path):
        raise ValueError(
            "Tinygrad engine currently supports local GGUF checkpoints only; "
            "native dense Hugging Face loading was removed after incorrect Llama 3.2 outputs"
        )

    modules = _import_tinygrad_modules()
    requested_device = _resolve_tinygrad_device(config.device)
    with _tinygrad_context(modules, requested_device):
        with suppress(Exception):
            if config.seed is not None:
                modules.Tensor.manual_seed(config.seed)

        model, tokenizer, prepare_tokenizer, model_type = _build_gguf_runtime(
            modules=modules,
            config=config,
            model_config=model_config,
        )
        load_format = "gguf"

        compute_device = str(modules.Device.DEFAULT if requested_device is None else modules.Device.canonicalize(requested_device))
        max_context = int(getattr(model, "max_context", config.max_context or 2048))
        runtime_profile = _resolve_tinygrad_runtime_profile(config, compute_device)
        return _LoadedTinygradRuntime(
            modules=modules,
            model=model,
            tokenizer=tokenizer,
            prepare_tokenizer=prepare_tokenizer,
            load_format=load_format,
            model_type=model_type,
            compute_device=compute_device,
            max_context=max_context,
            runtime_profile=runtime_profile,
        )


def _build_gguf_runtime(
    *,
    modules: _TinygradModules,
    config: Tinygrad,
    model_config: Model,
) -> tuple[Any, Any, Any | None, str]:
    """Build the general GGUF runtime through tinygrad's packaged LLM loader."""

    model_path = _require_local_model_path(model_config.path)
    raw_model = modules.Tensor.empty(
        model_path.stat().st_size,
        dtype=modules.dtypes.uint8,
        device=f"disk:{model_path}",
    ).to(modules.Device.DEFAULT)
    model, kv = modules.llm_model.Transformer.from_gguf(
        raw_model,
        max_context=config.max_context,
    )
    del raw_model
    gc.collect()

    tokenizer = modules.llm_cli.SimpleTokenizer.from_gguf_kv(kv)
    prepare_tokenizer = None
    if model_config.tokenizer is not None or model_config.tokenizer_path is not None:
        prepare_tokenizer = _load_prepare_tokenizer(config, model_config)
    return model, tokenizer, prepare_tokenizer, str(kv.get("general.architecture", "gguf"))


def _load_prepare_tokenizer(config: Tinygrad, model_config: Model) -> Any:
    """Load the optional Hugging Face tokenizer used for GGUF chat-template rendering."""

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = _load_tokenizer_from_model(
        _resolve_tokenizer_source(model_config),
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = config.padding_side
    return tokenizer


def _import_tinygrad_modules() -> _TinygradModules:
    """Import tinygrad from the active Python environment."""
    try:
        return _load_tinygrad_modules()
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name.startswith("tinygrad.llm"):
            raise ModuleNotFoundError(
                "tinygrad is installed without the packaged `tinygrad.llm` runtime; "
                "install a tinygrad build that ships the LLM modules required by Evalution"
            ) from exc
        raise ModuleNotFoundError(
            "tinygrad is not importable; install the optional `tinygrad` dependency"
        ) from exc


def _load_tinygrad_modules() -> _TinygradModules:
    """Import the tinygrad modules required by this backend."""

    tinygrad = importlib.import_module("tinygrad")
    helpers = importlib.import_module("tinygrad.helpers")
    nn_state = importlib.import_module("tinygrad.nn.state")
    llm_model = importlib.import_module("tinygrad.llm.model")
    llm_cli = importlib.import_module("tinygrad.llm.cli")
    dtypes = importlib.import_module("tinygrad.dtype").dtypes
    return _TinygradModules(
        tinygrad=tinygrad,
        Tensor=tinygrad.Tensor,
        Device=tinygrad.Device,
        helpers=helpers,
        nn_state=nn_state,
        llm_model=llm_model,
        llm_cli=llm_cli,
        dtypes=dtypes,
    )


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    """Apply temporary environment variable overrides around one tinygrad runtime call."""

    if not updates:
        yield
        return

    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _resolve_tinygrad_runtime_profile(
    config: Tinygrad,
    compute_device: str,
) -> _TinygradRuntimeProfile:
    """Resolve the JIT controls used for one built tinygrad session."""

    default_profile = _default_tinygrad_runtime_profile(compute_device)
    if default_profile is None:
        return _TinygradRuntimeProfile(name="tinygrad_default", jit=config.jit, jitbeam=config.jitbeam)

    return _TinygradRuntimeProfile(
        name=default_profile.name,
        jit=default_profile.jit if config.jit is None else int(config.jit),
        jitbeam=default_profile.jitbeam if config.jitbeam is None else int(config.jitbeam),
    )


def _default_tinygrad_runtime_profile(compute_device: str) -> _TinygradRuntimeProfile | None:
    """Return the profiled default runtime controls for the active tinygrad device family."""

    if not compute_device.startswith(("CUDA", "NV")):
        return None
    bucket = _visible_cuda_profile_bucket()
    return _TINYGRAD_CUDA_PROFILES.get(bucket, _DEFAULT_TINYGRAD_CUDA_PROFILE)


def _visible_cuda_profile_bucket() -> str:
    """Bucket the one visible CUDA device into the profiled tinygrad runtime families."""

    with suppress(Exception):
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name = props.name.lower()
            if "rtx 4090" in name:
                return "rtx4090"
            if props.major == 8 and props.total_memory >= _A100_CLASS_MIN_VRAM_BYTES:
                return "a100"
    return "cuda_default"


def _tinygrad_context(modules: _TinygradModules, device: str | None):
    """Bind tinygrad's default device while runtime construction is in flight."""

    if device is None:
        return nullcontext()
    return modules.helpers.Context(DEV=device)


def _resolve_tinygrad_device(device: str | None) -> str | None:
    """Normalize Evalution-style device strings into tinygrad runtime device names."""

    if device is None:
        return None
    normalized = str(device).strip()
    if not normalized or normalized.lower() == "auto":
        return None

    lowered = normalized.lower()
    if lowered.startswith("cuda"):
        suffix = normalized.split(":", maxsplit=1)[1] if ":" in normalized else ""
        return "CUDA" if suffix in {"", "0"} else f"CUDA:{suffix}"
    if lowered.startswith("nv"):
        suffix = normalized.split(":", maxsplit=1)[1] if ":" in normalized else ""
        return "NV" if suffix in {"", "0"} else f"NV:{suffix}"
    if lowered.startswith("cpu"):
        return "CPU"
    if lowered.startswith("metal"):
        return "METAL"
    if lowered.startswith("amd"):
        return "AMD"
    if lowered.startswith("qcom"):
        return "QCOM"
    if lowered.startswith("cl"):
        return "CL"
    if lowered.startswith("webgpu"):
        return "WEBGPU"
    raise ValueError(f"unsupported tinygrad device override: {device!r}")


def _is_gguf_path(path_or_name: str) -> bool:
    """Detect whether the provided model path points at one local GGUF file."""

    return Path(path_or_name).suffix.lower() == ".gguf"


def _require_local_model_path(path_or_name: str) -> Path:
    """Resolve one model path and fail with a targeted local-only error when it is missing."""

    path = Path(path_or_name).expanduser()
    if path.exists():
        return path
    raise ValueError(
        "Tinygrad engine currently expects a local model path; remote Hub ids should be downloaded first"
    )


def _is_tinygrad_simple_tokenizer(tokenizer: Any) -> bool:
    """Detect tinygrad's packaged GGUF tokenizer without importing its concrete class in tests."""

    return all(hasattr(tokenizer, attr_name) for attr_name in ("prefix", "role", "end_turn", "is_end"))
