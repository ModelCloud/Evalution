# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import islice
from queue import Full, Queue
from threading import Event, Lock
from typing import Any

from evalution.engines.base import GenerationRequest, InferenceSession

# Number of rows to prepare up front when a session needs a sample window to infer
# a workable batch size without scanning the full dataset.
AUTO_BATCH_PREVIEW_ROWS = 256
# Keep the pretokenized queue larger than a single generation batch so background
# preparation can stay ahead of consumption without growing unbounded.
_PRETOKENIZED_POOL_MULTIPLIER = 2
# Bound queue.put() blocking so the prefetch worker can re-check cancellation and
# exit promptly instead of hanging on a full queue.
_BATCH_PREFETCH_PUT_TIMEOUT_S = 0.1
# Briefly wait for more free slots before preparing a partial refill; this helps
# the worker produce fuller chunks and reduces tiny pretokenization bursts.
_PRETOKENIZED_REFILL_COALESCE_S = 0.01
# Protect shared executor state while threads initialize or clear the singleton.
_PREFETCH_EXECUTOR_LOCK = Lock()
# Process-wide single-thread executor used for sample prefetch so suites reuse one
# background worker instead of repeatedly creating short-lived executors.
_PREFETCH_EXECUTOR: ThreadPoolExecutor | None = None


@dataclass(frozen=True, slots=True)
class PreparedSample:
    index: int
    doc: dict[str, Any]
    target: str
    request: GenerationRequest


@dataclass(frozen=True, slots=True)
class PrefetchFailure:
    error: BaseException


# Prefer an explicit session resolver, then fall back to configured batch size hints.
def session_batch_size(
    session: InferenceSession,
    requests: list[GenerationRequest],
) -> int | None:
    resolver = getattr(session, "resolve_batch_size", None)
    if callable(resolver):
        resolved_batch_size = resolver(requests)
        if resolved_batch_size is not None:
            return int(resolved_batch_size)

    batch_size = getattr(session, "batch_size", None)
    if batch_size is not None:
        return int(batch_size)

    config = getattr(session, "config", None)
    config_batch_size = getattr(config, "batch_size", None)
    if config_batch_size is not None:
        return int(config_batch_size)
    return None


# Decide whether the suite needs a preview window to auto-resolve batch size.
def needs_batch_size_preview(
    suite_batch_size: int | None,
    session: InferenceSession,
) -> bool:
    if suite_batch_size is not None:
        return False

    batch_size = getattr(session, "batch_size", None)
    if isinstance(batch_size, int):
        return False

    config = getattr(session, "config", None)
    config_batch_size = getattr(config, "batch_size", None)
    if isinstance(config_batch_size, int):
        return False

    return True


# Pull a bounded preview window from the prepared iterator for batch-size probing.
def collect_preview_samples(
    prepared_iter: Any,
    *,
    preview_size: int,
    prepare_bar: Any,
) -> list[PreparedSample]:
    preview_samples: list[PreparedSample] = []
    for sample in islice(prepared_iter, preview_size):
        preview_samples.append(sample)
        prepare_bar.next().draw()
    return preview_samples


# Let the session pre-tokenize or otherwise transform a batch before generation.
def prepare_batch_for_session(
    session: InferenceSession,
    batch: list[PreparedSample],
) -> list[PreparedSample]:
    if not batch:
        return batch

    prepare_requests = getattr(session, "prepare_requests", None)
    if not callable(prepare_requests):
        return batch

    prepared_requests = prepare_requests([sample.request for sample in batch])
    return [
        replace(sample, request=prepared_request)
        for sample, prepared_request in zip(batch, prepared_requests, strict=True)
    ]


# Size the pretokenized queue so the worker can stay ahead of generation.
def _pretokenized_pool_size(batch_size: int) -> int:
    return max(batch_size, batch_size * _PRETOKENIZED_POOL_MULTIPLIER)


# Reuse a single prefetch thread across suites to avoid executor churn.
def _prefetch_executor() -> ThreadPoolExecutor:
    global _PREFETCH_EXECUTOR
    with _PREFETCH_EXECUTOR_LOCK:
        if _PREFETCH_EXECUTOR is None:
            _PREFETCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="evalution-prefetch",
            )
    return _PREFETCH_EXECUTOR


# Tear down the shared prefetch executor when the process exits.
def _shutdown_prefetch_executor() -> None:
    global _PREFETCH_EXECUTOR
    with _PREFETCH_EXECUTOR_LOCK:
        executor = _PREFETCH_EXECUTOR
        _PREFETCH_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=False)


atexit.register(_shutdown_prefetch_executor)


# Stream prepared samples while a background worker keeps the queue filled.
def iter_prefetched_samples(
    session: InferenceSession,
    preview_samples: list[PreparedSample],
    prepared_iter: Any,
    *,
    batch_size: int,
    prepare_bar: Any,
    pool_size: int | None = None,
) -> Any:
    sentinel = object()
    queue_maxsize = pool_size or _pretokenized_pool_size(batch_size)
    queue: Queue[Any] = Queue(maxsize=queue_maxsize)
    cancelled = Event()

    # Retry bounded queue puts so the worker can stop promptly on cancellation.
    def put_prefetched(item: Any) -> bool:
        while not cancelled.is_set():
            try:
                queue.put(item, timeout=_BATCH_PREFETCH_PUT_TIMEOUT_S)
                return True
            except Full:
                continue
        return False

    # Tokenize future work in chunks sized to the available queue capacity.
    def worker() -> None:
        try:
            while not cancelled.is_set():
                available_slots = max(0, queue_maxsize - queue.qsize())
                if available_slots <= 0:
                    cancelled.wait(_BATCH_PREFETCH_PUT_TIMEOUT_S)
                    continue
                if 0 < available_slots < batch_size:
                    cancelled.wait(_PRETOKENIZED_REFILL_COALESCE_S)
                    available_slots = max(0, queue_maxsize - queue.qsize())
                    if available_slots <= 0:
                        continue
                chunk_size = min(batch_size, available_slots)
                chunk = list(islice(prepared_iter, chunk_size))
                if not chunk:
                    break
                prepared_chunk = prepare_batch_for_session(session, chunk)
                for sample in prepared_chunk:
                    if not put_prefetched(sample):
                        return
        except BaseException as exc:
            put_prefetched(PrefetchFailure(exc))
        finally:
            put_prefetched(sentinel)

    future = _prefetch_executor().submit(worker)
    try:
        for sample in preview_samples:
            yield sample

        while True:
            prefetched = queue.get()
            if prefetched is sentinel:
                break
            if isinstance(prefetched, PrefetchFailure):
                raise prefetched.error
            yield prefetched
            prepare_bar.next().draw()
    finally:
        cancelled.set()
        close_prepared_iter = getattr(prepared_iter, "close", None)
        if callable(close_prepared_iter):
            close_prepared_iter()
        with suppress(Exception):
            future.result(timeout=1.0)


# Group prefetched samples into fixed-size batches for standard `generate()` sessions.
def iter_prefetched_batches(
    session: InferenceSession,
    preview_samples: list[PreparedSample],
    prepared_iter: Any,
    *,
    batch_size: int,
    prepare_bar: Any,
    pool_size: int | None = None,
) -> Any:
    batch: list[PreparedSample] = []
    for sample in iter_prefetched_samples(
        session,
        preview_samples,
        prepared_iter,
        batch_size=batch_size,
        prepare_bar=prepare_bar,
        pool_size=pool_size,
    ):
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
