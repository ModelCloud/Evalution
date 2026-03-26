# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared queued request/result runtime for continuous generation.

Terminology
+------------------+-----------------------------------------------------------+
| Term             | Meaning                                                   |
+------------------+-----------------------------------------------------------+
| RequestProducer  | End user or API caller yielding GenerationRequest items   |
| RequestQueue     | Thread-safe queue carrying Request items to the session   |
| RequestConsumer  | Session-side loop draining RequestQueue and deciding refill|
| RequestExecutor  | Backend work executed by the RequestConsumer              |
| ResultQueue      | Thread-safe queue carrying Result or Error back to caller |
| ResultConsumer   | Caller thread iterating over generate_continuous outputs  |
+------------------+-----------------------------------------------------------+

Relationship graph

    RequestProducer
           |
           | put Request
           v
    +---------------+     drain / refill     +-------------------------------+
    | RequestQueue  | ---------------------> | RequestConsumer               |
    +---------------+                        | + RequestExecutor             |
                                             +-------------------------------+
                                                           |
                                                           | put Result / Error
                                                           v
                                                    +---------------+
                                                    | ResultQueue   |
                                                    +---------------+
                                                           |
                                                           v
                                                     ResultConsumer

The producer thread only moves submitted requests into RequestQueue.
The consumer thread owns backend progress and refill policy.
The caller thread only yields items drained from ResultQueue, so session locks
do not stay held across user-visible yields.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from evalution.engines.base import GenerationOutput, GenerationRequest


@dataclass(slots=True)
class Request:
    # Carry one caller-submitted request across the producer-to-consumer queue boundary.
    request_key: Any
    request: GenerationRequest


@dataclass(slots=True)
class Result:
    # Carry one engine-produced output from the consumer thread back to the caller thread.
    request_key: Any
    output: GenerationOutput


@dataclass(slots=True)
class Error:
    # Preserve asynchronous producer or consumer failures so the public iterator can raise them.
    exc: Exception


# Mark that the producer exhausted its request source and no more Request items will arrive.
_REQUESTS_DONE = object()
# Mark that the consumer finished producing results and the public iterator can stop.
_RESULTS_DONE = object()


class RequestQueue:
    # Expose queued request pulls to session-side consumers without coupling them to the caller
    # iterable. The queue transports Request items plus one terminal done sentinel or Error.
    def __init__(self) -> None:
        # Keep the transport unbounded so the producer thread never blocks the caller on a small
        # queue size; the session-side consumer owns refill and backpressure policy instead.
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False

    @property
    def closed(self) -> bool:
        # Report whether the producer already emitted the terminal done marker or a fatal error.
        return self._closed

    def put_request(self, request_key: Any, request: GenerationRequest) -> None:
        # Enqueue one caller-submitted generation request for the consumer thread.
        self._queue.put(Request(request_key=request_key, request=request))

    def put_error(self, exc: Exception) -> None:
        # Forward a producer-side failure through the same queue so the caller raises it in order.
        self._queue.put(Error(exc=exc))

    def close(self) -> None:
        # Mark that the producer exhausted its source iterable and no more requests will arrive.
        self._queue.put(_REQUESTS_DONE)

    def get(self, *, timeout_s: float | None = None) -> tuple[Any, GenerationRequest] | None:
        # Poll for the next request item, returning None only when the timeout expires or the
        # queue delivers the terminal done marker.
        try:
            item = self._queue.get(timeout=timeout_s) if timeout_s is not None else self._queue.get()
        except queue.Empty:
            return None
        return self._decode_item(item)

    def get_nowait(self) -> tuple[Any, GenerationRequest] | None:
        # Fast-path non-blocking poll used by consumers that already know work should be ready.
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            return None
        return self._decode_item(item)

    def iter_requests(
        self,
        *,
        stop_event: threading.Event | None = None,
        poll_timeout_s: float = 0.05,
    ) -> Iterator[tuple[Any, GenerationRequest]]:
        # Yield requests until the producer closes the queue or the caller asks the consumer to
        # stop. Timeout polling keeps the stop_event responsive even when the queue is idle.
        while stop_event is None or not stop_event.is_set():
            item = self.get(timeout_s=poll_timeout_s)
            if item is not None:
                yield item
                continue
            if self.closed:
                return

    def _decode_item(self, item: object) -> tuple[Any, GenerationRequest] | None:
        # Normalize sentinels and failures into either a usable request tuple, None for end of
        # stream, or a raised exception for fatal producer-side errors.
        if item is _REQUESTS_DONE:
            self._closed = True
            return None
        if isinstance(item, Error):
            self._closed = True
            raise item.exc
        if not isinstance(item, Request):
            raise RuntimeError("request queue received an unknown item")
        return item.request_key, item.request


def assert_non_main_thread() -> None:
    # RequestExecutor work must not run on the Python main thread.
    assert threading.current_thread() is not threading.main_thread(), (
        "RequestExecutor must run on a non-main thread"
    )


# Type alias for the session-side request consumer callback. The callback may also perform the
# concrete backend execution work directly when the engine has not yet split those roles apart.
ProcessRequests = Callable[
    [threading.Event, RequestQueue, Callable[[Any, GenerationOutput], None]],
    None,
]


def stream_request_results(
    requests: Iterable[tuple[Any, GenerationRequest]],
    *,
    producer_name: str,
    consumer_name: str,
    process_requests: ProcessRequests,
    result_poll_timeout_s: float = 0.05,
    require_non_main_thread: bool = True,
) -> Iterator[tuple[Any, GenerationOutput]]:
    # Bridge a RequestProducer iterable to a RequestConsumer callback by running each side on its
    # own thread and moving Request / Result / Error items across explicit queues.
    def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
        # Carry caller-submitted requests into the session-side consumer thread.
        request_queue = RequestQueue()
        # Carry finished outputs or fatal consumer failures back to the caller thread.
        result_queue: queue.Queue[object] = queue.Queue()
        # Let iterator close() or caller teardown stop both background threads promptly.
        stop_event = threading.Event()

        def put_result(request_key: Any, output: GenerationOutput) -> None:
            # Send one completed backend output back to the caller thread in result order.
            result_queue.put(Result(request_key=request_key, output=output))

        def run_producer() -> None:
            # Drain the caller iterable into RequestQueue without blocking backend progress on the
            # caller thread's pacing between next() calls.
            try:
                for request_key, request in requests:
                    if stop_event.is_set():
                        break
                    request_queue.put_request(request_key, request)
            except Exception as exc:
                request_queue.put_error(exc)
            finally:
                request_queue.close()

        def run_consumer() -> None:
            # Let the engine/session own refill policy and backend execution on a dedicated thread.
            try:
                if require_non_main_thread:
                    assert_non_main_thread()
                process_requests(stop_event, request_queue, put_result)
            except Exception as exc:
                result_queue.put(Error(exc=exc))
            finally:
                result_queue.put(_RESULTS_DONE)

        # Name the threads after their role so debugger output makes producer/consumer ownership clear.
        producer_thread = threading.Thread(target=run_producer, name=producer_name, daemon=True)
        consumer_thread = threading.Thread(target=run_consumer, name=consumer_name, daemon=True)
        producer_thread.start()
        consumer_thread.start()
        try:
            while True:
                try:
                    item = result_queue.get(timeout=result_poll_timeout_s)
                except queue.Empty:
                    # Timeout polling keeps iterator shutdown responsive even when no results are ready.
                    continue
                if item is _RESULTS_DONE:
                    return
                if isinstance(item, Error):
                    raise item.exc
                if not isinstance(item, Result):
                    raise RuntimeError("result queue received an unknown item")
                yield item.request_key, item.output
        finally:
            # Stop the background threads before generator teardown so a paused caller cannot leak
            # a live backend worker past iterator close() or exception unwinding.
            stop_event.set()
            consumer_thread.join()
            producer_thread.join(timeout=result_poll_timeout_s)

    return iterator()
