# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import inspect
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from functools import wraps
from itertools import chain, islice
from typing import Any

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformers_common import (
    BaseTransformerSession,
    _TransformersCommonConfig,
    _base_attn_implementation,
    _fallback_batch_size,
    _friendly_batch_size,
    _requests_paged_attention,
    _truncate_at_stop,
    load_transformer_runtime,
    transformers_continuous_batching_support,
)
from evalution.logbar import get_logger

_PENDING_NOGIL_TRANSFORMERS_PR_URL = "https://github.com/huggingface/transformers/pull/44924"
_PENDING_NOGIL_TRANSFORMERS_PR_WARNED = False
_PENDING_NOGIL_TRANSFORMERS_PR_WARN_LOCK = threading.Lock()
# Serialize the one-time Evalution-side stop-gap patch for transformers continuous batching.
_CONTINUOUS_BATCHING_CUDA_CONTEXT_PATCH_LOCK = threading.Lock()
_FLASH_ATTN_VARLEN_FWD_CUDA_CONTEXT_PATCH_LOCK = threading.Lock()


@dataclass(slots=True)
class Transformers(_TransformersCommonConfig):
    # Use the modern transformers engine path that can enable paged attention and continuous batching.
    manual_eviction: bool = False
    allow_block_sharing: bool = True
    use_async_batching: bool | None = None
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0
    max_cached_graphs: int = 0

    # Build the modern session, or fall back to the compat engine when the installed package is too old.
    def build(self, model: Model) -> BaseTransformerSession:
        supports_continuous_batching, reason = transformers_continuous_batching_support()
        if not supports_continuous_batching:
            if _requests_paged_attention(self.attn_implementation):
                raise ValueError(
                    "paged attn_implementation requires a transformers build with continuous batching support"
                )
            self.resolved_engine = "TransformersCompat"
            get_logger().warning(
                "transformers continuous batching is unavailable: %s; falling back to TransformersCompat",
                reason,
            )
            from evalution.engines.transformers_compat import TransformersCompat

            return TransformersCompat.from_transformers(self).build(model)

        _warn_pending_nogil_transformers_pr_once()
        _patch_flash_attn_varlen_fwd_cuda_context_once()
        self.resolved_engine = "Transformers"
        return TransformersSession.from_config(self, model)


def _warn_pending_nogil_transformers_pr_once() -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if not callable(is_gil_enabled) or is_gil_enabled():
        return

    global _PENDING_NOGIL_TRANSFORMERS_PR_WARNED
    with _PENDING_NOGIL_TRANSFORMERS_PR_WARN_LOCK:
        if _PENDING_NOGIL_TRANSFORMERS_PR_WARNED:
            return
        _PENDING_NOGIL_TRANSFORMERS_PR_WARNED = True

    get_logger().warning(
        "Python free-threading is enabled; upstream transformers PR #44924 is still pending for no-GIL "
        "continuous batching fixes. Released transformers builds may affect paged attention, compare, "
        "and concurrent model loading until that PR ships: %s",
        _PENDING_NOGIL_TRANSFORMERS_PR_URL,
    )


def _patch_continuous_batching_manager_cuda_context_once(ContinuousBatchingManager: Any) -> None:
    """Apply Evalution's local continuous-batching CUDA-device stop-gap exactly once.

    The upstream manager performs its real paged-attention work on a background thread, not on
    the caller thread that constructs or starts the manager. CUDA's "current device" is
    thread-local, so wrapping Evalution's outer generate call would not affect the manager's
    worker thread. We therefore patch the manager entrypoint that actually runs on that worker.

    This is intentionally local to Evalution instead of /root/transformers because it is only a
    stop-gap while upstream no-GIL / multi-manager fixes are still settling. The wrapper makes
    device-global CUDA APIs such as graph creation, stream synchronization, and graph pool setup
    observe the manager model's device on that specific worker thread.

    Important limitation: this assumes one effective CUDA device per manager thread. It is a good
    fit for the current Evalution compare case where each lane owns one model/device. It is not a
    general fix for a single process driving one model across several CUDA devices directly.
    """
    # Older or alternate transformers builds may not expose the internal manager thread entrypoint
    # that this stop-gap relies on. In that case we leave the class untouched instead of trying to
    # patch a different API surface heuristically.
    run_generation_loop = getattr(ContinuousBatchingManager, "_run_generation_loop", None)
    if not callable(run_generation_loop):
        return

    with _CONTINUOUS_BATCHING_CUDA_CONTEXT_PATCH_LOCK:
        current = getattr(ContinuousBatchingManager, "_run_generation_loop", None)
        # Another session may have patched the class first; avoid wrapping the same method twice.
        if not callable(current) or getattr(current, "__evalution_cuda_context_patch__", False):
            return

        @wraps(current)
        def _wrapped_run_generation_loop(self: Any, *args: Any, **kwargs: Any) -> Any:
            import torch

            model_device = getattr(getattr(self, "model", None), "device", None)
            # The manager loop is the first code that runs on the background generation thread.
            # Enter the model's CUDA device here so any current-device CUDA calls made deeper in
            # transformers resolve against the manager's model instead of whatever device happened
            # to be current on that thread previously.
            maybe_device = (
                torch.cuda.device(model_device)
                if getattr(model_device, "type", None) == "cuda"
                else nullcontext()
            )
            with maybe_device:
                return current(self, *args, **kwargs)

        # Mark the wrapper so repeated manager construction in the same process stays idempotent.
        _wrapped_run_generation_loop.__evalution_cuda_context_patch__ = True
        ContinuousBatchingManager._run_generation_loop = _wrapped_run_generation_loop


def _patch_flash_attn_varlen_fwd_cuda_context_once() -> None:
    """Set the correct CUDA context before launching the FlashAttention kernel."""
    try:
        import flash_attn.flash_attn_interface as flash_attn_interface
    except Exception:
        return

    flash_attn_gpu = getattr(flash_attn_interface, "flash_attn_gpu", None)
    current = getattr(flash_attn_gpu, "varlen_fwd", None)
    if not callable(current):
        return

    with _FLASH_ATTN_VARLEN_FWD_CUDA_CONTEXT_PATCH_LOCK:
        current = getattr(flash_attn_gpu, "varlen_fwd", None)
        if not callable(current) or getattr(current, "__evalution_cuda_context_patch__", False):
            return

        def _wrapped_varlen_fwd(*args: Any, **kwargs: Any) -> Any:
            import torch

            query = args[0] if args else None
            with torch.cuda.device(query.device):
                return current(*args, **kwargs)

        _wrapped_varlen_fwd.__evalution_cuda_context_patch__ = True
        flash_attn_gpu.varlen_fwd = _wrapped_varlen_fwd


@dataclass(slots=True)
class TransformersSession(BaseTransformerSession):
    # Keep the session-owned continuous batching manager alive while request settings stay compatible.
    continuous_batching_manager: Any | None = field(default=None, repr=False)
    continuous_batching_signature: tuple[Any, ...] | None = field(default=None, repr=False)
    # Track completed request ids whose cache blocks remain resident until explicit manual eviction.
    continuous_batching_completed_request_ids: set[str] = field(default_factory=set, repr=False)
    continuous_batching_request_counter: int = field(default=0, repr=False)

    @classmethod
    def from_config(cls, config: Transformers, model_config: Model) -> TransformersSession:
        runtime = load_transformer_runtime(config, model_config)
        raw_attn_implementation = config.attn_implementation or runtime.requested_attn_implementation
        paged_attention_enabled = _resolve_paged_attention(
            attn_implementation=raw_attn_implementation,
            model=runtime.model,
            input_device=runtime.input_device,
        )
        effective_attn_implementation = _effective_attn_implementation(
            raw_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
        )
        generation_backend = "continuous_batching" if paged_attention_enabled else "generate"
        get_logger().info(
            "transformers attention requested=%s effective=%s backend=%s paged_attention=%s",
            raw_attn_implementation,
            effective_attn_implementation,
            generation_backend,
            paged_attention_enabled,
        )

        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            requested_attn_implementation=raw_attn_implementation,
            effective_attn_implementation=effective_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
            generation_backend=generation_backend,
        )

    # Prefer the live paged-attention path and only fall back to fixed batches after a real runtime failure.
    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        if not requests:
            return []

        with self._generation_lock:
            effective_batch_size = batch_size or self.resolve_batch_size(requests)
            with self._state_lock:
                paged_attention_enabled = self.paged_attention_enabled
                standard_batch_size_cap = self.standard_batch_size_cap
            if not paged_attention_enabled and standard_batch_size_cap is not None:
                effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
            self._log_generation_execution()
            if paged_attention_enabled:
                try:
                    return self._generate_paged(requests, batch_size=effective_batch_size)
                except Exception as exc:
                    fallback_batch_size = _fallback_batch_size(effective_batch_size)
                    self._disable_paged_attention(exc, fallback_batch_size=fallback_batch_size)
                    return self._generate_standard(requests, batch_size=fallback_batch_size)
            return self._generate_standard(requests, batch_size=effective_batch_size)

    # Keep the suite-level continuous request stream on the paged manager while the backend remains healthy.
    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            request_iter = iter(requests)
            preview_items = list(islice(request_iter, 64))
            if not preview_items:
                return

            effective_batch_size = batch_size or self.resolve_batch_size(
                [request for _, request in preview_items]
            )
            items = chain(preview_items, request_iter)

            with self._generation_lock:
                with self._state_lock:
                    paged_attention_enabled = self.paged_attention_enabled
                    standard_batch_size_cap = self.standard_batch_size_cap
                if not paged_attention_enabled and standard_batch_size_cap is not None:
                    effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
                self._log_generation_execution()
                if paged_attention_enabled:
                    try:
                        yield from self._generate_paged_continuous(items, batch_size=effective_batch_size)
                        return
                    except Exception as exc:
                        fallback_batch_size = _fallback_batch_size(effective_batch_size)
                        self._disable_paged_attention(exc, fallback_batch_size=fallback_batch_size)
                        effective_batch_size = fallback_batch_size
                yield from self._generate_standard_continuous(items, batch_size=effective_batch_size)

        return iterator()

    # Stop the paged manager before clearing shared caches and allocator state.
    def gc(self) -> None:
        with self._generation_lock:
            self._stop_continuous_batching_manager()
        super(TransformersSession, self).gc()

    # Stop paged generation state before tearing down the model and tokenizer.
    def close(self) -> None:
        with self._generation_lock:
            self._stop_continuous_batching_manager()
            with self._prepare_tokenizer_lock:
                with self._tokenizer_lock:
                    with self._state_lock:
                        self.stop_criteria_cache.clear()
                        self.auto_batch_size_cache.clear()
                        with suppress(Exception):
                            del self.model
                        with suppress(Exception):
                            del self.tokenizer
                        with suppress(Exception):
                            del self.prepare_tokenizer
        self.gc()

    # Collect paged generation results into positional order for the non-streaming API.
    def _generate_paged(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        outputs_by_position: list[GenerationOutput | None] = [None] * len(requests)
        for position, output in self._generate_paged_continuous(
            enumerate(requests),
            batch_size=batch_size,
        ):
            outputs_by_position[int(position)] = output
        if any(output is None for output in outputs_by_position):
            raise RuntimeError("continuous batching returned incomplete results")
        return [output for output in outputs_by_position if output is not None]

    # Submit requests lazily to the long-lived paged manager and yield each finished result as it completes.
    def _generate_paged_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        request_iter = iter(requests)
        inflight_requests: dict[str, tuple[Any, str, list[str], dict[str, Any]]] = {}
        source_exhausted = False
        expected_signature: tuple[Any, ...] | None = None
        manager: Any | None = None

        def submit_one() -> bool:
            nonlocal source_exhausted
            nonlocal expected_signature
            if source_exhausted:
                return False
            try:
                request_key, request = next(request_iter)
            except StopIteration:
                source_exhausted = True
                return False

            request_signature = _continuous_request_signature(request)
            if expected_signature is None:
                expected_signature = request_signature
            elif request_signature != expected_signature:
                raise ValueError(
                    "continuous batching requires shared stop strings and sampling settings "
                    "within a generation stream"
                )

            if manager is None:
                raise RuntimeError("continuous batching manager was not initialized")

            rendered_prompt, input_ids = self._prepare_request_for_generation(request)
            request_id = self._next_continuous_batching_request_id()
            manager.add_request(
                input_ids,
                request_id=request_id,
                max_new_tokens=request.max_new_tokens,
                streaming=False,
            )
            inflight_requests[request_id] = (
                request_key,
                rendered_prompt,
                list(request.stop),
                dict(request.metadata),
            )
            return True

        preview_items = list(islice(request_iter, max(1, batch_size)))
        if not preview_items:
            return
        request_iter = iter(chain(preview_items, request_iter))

        first_request = preview_items[0][1]
        expected_signature = _continuous_request_signature(first_request)
        manager = self._ensure_continuous_batching_manager(
            request_signature=expected_signature,
            request=first_request,
        )

        while len(inflight_requests) < batch_size and submit_one():
            continue

        while inflight_requests:
            request_output = manager.get_result(timeout=0.1)
            if request_output is None:
                if not manager.is_running():
                    raise RuntimeError(
                        "continuous batching manager stopped before all requests completed"
                    )
                continue

            request_id = request_output.request_id
            request_state = inflight_requests.get(request_id)
            if request_state is None:
                raise RuntimeError(
                    f"continuous batching returned unknown request_id={request_id!r}"
                )
            if request_output.error is not None:
                raise RuntimeError(
                    f"continuous batching request {request_id!r} failed: {request_output.error}"
                )
            if not request_output.is_finished():
                continue

            request_key, rendered_prompt, stop_strings, metadata = inflight_requests.pop(request_id)
            if self.config.manual_eviction:
                with self._state_lock:
                    self.continuous_batching_completed_request_ids.add(request_id)
            with self._tokenizer_lock:
                text = self.tokenizer.decode(
                    request_output.generated_tokens,
                    skip_special_tokens=False,
                )
            text = _truncate_at_stop(text, stop_strings).strip()
            yield request_key, GenerationOutput(
                prompt=rendered_prompt,
                text=text,
                metadata=metadata,
            )

            while len(inflight_requests) < batch_size and submit_one():
                continue

    # Reuse the existing manager while request settings match; otherwise rebuild it around the new signature.
    def _ensure_continuous_batching_manager(
        self,
        *,
        request_signature: tuple[Any, ...],
        request: GenerationRequest,
    ) -> Any:
        from transformers import ContinuousBatchingManager

        _patch_continuous_batching_manager_cuda_context_once(ContinuousBatchingManager)
        generation_config = self._build_generation_config([request])

        with self._state_lock:
            manager = self.continuous_batching_manager
            manager_signature = self.continuous_batching_signature
            if (
                manager is not None
                and manager_signature == request_signature
                and manager.is_running()
            ):
                return manager

        self._stop_continuous_batching_manager()

        manager = self._build_continuous_batching_manager(
            ContinuousBatchingManager=ContinuousBatchingManager,
            generation_config=generation_config,
        )
        manager.start()

        with self._state_lock:
            self.continuous_batching_manager = manager
            self.continuous_batching_signature = request_signature
            return manager

    # Support both legacy kwarg-style and latest config-object continuous batching constructors.
    def _build_continuous_batching_manager(
        self,
        *,
        ContinuousBatchingManager: Any,
        generation_config: Any,
    ) -> Any:
        manager_init = inspect.signature(ContinuousBatchingManager.__init__)
        if "continuous_batching_config" in manager_init.parameters:
            from transformers import ContinuousBatchingConfig

            continuous_batching_config = ContinuousBatchingConfig(
                allow_block_sharing=self.config.allow_block_sharing,
                use_async_batching=self.config.use_async_batching,
                q_padding_interval_size=self.config.q_padding_interval_size,
                kv_padding_interval_size=self.config.kv_padding_interval_size,
                max_cached_graphs=self.config.max_cached_graphs,
            )
            return ContinuousBatchingManager(
                self.model,
                generation_config=generation_config,
                continuous_batching_config=continuous_batching_config,
            )

        return ContinuousBatchingManager(
            self.model,
            generation_config=generation_config,
            manual_eviction=self.config.manual_eviction,
            q_padding_interval_size=self.config.q_padding_interval_size,
            kv_padding_interval_size=self.config.kv_padding_interval_size,
            max_cached_graphs=self.config.max_cached_graphs,
            allow_block_sharing=self.config.allow_block_sharing,
            use_async_batching=self.config.use_async_batching,
        )

    # Allocate stable monotonic request ids for the paged manager.
    def _next_continuous_batching_request_id(self) -> str:
        with self._state_lock:
            request_id = f"req_{self.continuous_batching_request_counter}"
            self.continuous_batching_request_counter += 1
            return request_id

    # Stop the session-owned paged manager and evict retained requests when manual eviction is enabled.
    def _stop_continuous_batching_manager(self) -> None:
        with self._state_lock:
            manager = self.continuous_batching_manager
            retained_request_ids = set(self.continuous_batching_completed_request_ids)
            self.continuous_batching_manager = None
            self.continuous_batching_signature = None
            self.continuous_batching_completed_request_ids.clear()

        if manager is None:
            return

        if self.config.manual_eviction:
            for request_id in sorted(retained_request_ids):
                with suppress(Exception):
                    manager.evict_request_from_cache(request_id)

        with suppress(Exception):
            manager.stop(block=True)

    # Restore the base attention kernel while token scoring runs on a paged-attention session.
    @contextmanager
    def _scoring_attention_context(self) -> Iterator[None]:
        active_attention = self.effective_attn_implementation or self.requested_attn_implementation
        if not isinstance(active_attention, str) or not active_attention.startswith("paged|"):
            yield
            return

        setter = getattr(self.model, "set_attn_implementation", None)
        base_attention = _base_attn_implementation(active_attention)
        if not callable(setter) or base_attention is None:
            yield
            return

        setter(base_attention)
        try:
            yield
        finally:
            setter(active_attention)

    # Disable paged attention after a real failure and pin a safer fixed-batch fallback for the rest of the suite.
    def _disable_paged_attention(self, exc: Exception, *, fallback_batch_size: int) -> None:
        logger = get_logger()
        self._stop_continuous_batching_manager()
        with self._state_lock:
            previous_attention = self.effective_attn_implementation or self.requested_attn_implementation
            requested_attention = self.requested_attn_implementation
        base_attention = _base_attn_implementation(requested_attention)
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter) and base_attention is not None:
            with suppress(Exception):
                setter(base_attention)

        with self._state_lock:
            self.paged_attention_enabled = False
            self.generation_backend = "generate"
            self.effective_attn_implementation = base_attention or requested_attention
            self.standard_batch_size_cap = fallback_batch_size
            self.execution_logged = False
            effective_attention = self.effective_attn_implementation
            generation_backend = self.generation_backend
        logger.warning(
            "paged attention failed for attention=%s: %s; falling back to backend=%s attention=%s batch_size_cap=%d",
            previous_attention,
            exc,
            generation_backend,
            effective_attention,
            fallback_batch_size,
        )


def _continuous_request_signature(request: GenerationRequest) -> tuple[Any, ...]:
    return (
        tuple(request.stop),
        request.num_beams,
        request.do_sample,
        request.temperature if request.do_sample else None,
    )


def _effective_attn_implementation(
    attn_implementation: str | None,
    *,
    paged_attention_enabled: bool,
) -> str | None:
    if attn_implementation is None:
        return None
    if paged_attention_enabled and not attn_implementation.startswith("paged|"):
        return f"paged|{attn_implementation}"
    return attn_implementation


def _resolve_paged_attention(
    *,
    attn_implementation: str | None,
    model: Any,
    input_device: Any,
) -> bool:
    if not _requests_paged_attention(attn_implementation):
        return False

    can_use_paged_attention = (
        getattr(input_device, "type", None) == "cuda"
        and callable(getattr(model, "generate_batch", None))
        and callable(getattr(model, "set_attn_implementation", None))
    )
    if not can_use_paged_attention:
        raise ValueError(
            "paged attn_implementation requires a CUDA model with generate_batch() and "
            "set_attn_implementation() support"
        )
    return True
