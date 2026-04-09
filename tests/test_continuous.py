# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import threading

from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.continuous import assert_non_main_thread, stream_request_results


def test_stream_request_results_keeps_request_consumer_active_while_caller_is_paused() -> None:
    """Verify that the consumer thread keeps draining queued work while iteration is paused."""

    # What this test is actually verifying:
    # The producer/consumer bridge must continue feeding the session-side consumer
    # even when the caller pauses after reading only the first yielded result.
    third_request_seen = threading.Event()
    stop_seen = threading.Event()

    def consume_requests(stop_event, request_queue, put_result) -> None:
        """Simulate a consumer that proves queued requests keep flowing while paused."""

        first = request_queue.get(timeout_s=1.0)
        second = request_queue.get(timeout_s=1.0)

        assert first is not None
        assert second is not None

        first_key, first_request = first
        put_result(
            first_key,
            GenerationOutput(
                prompt=first_request.prompt or "",
                text=f"out::{first_request.prompt}",
                metadata={},
            ),
        )

        third = request_queue.get(timeout_s=1.0)
        assert third is not None
        third_request_seen.set()

        while not stop_event.is_set():
            threading.Event().wait(0.01)
        stop_seen.set()

    iterator = stream_request_results(
        [
            (10, GenerationRequest(prompt="alpha")),
            (11, GenerationRequest(prompt="beta")),
            (12, GenerationRequest(prompt="gamma")),
        ],
        producer_name="test.request_producer",
        consumer_name="test.request_consumer",
        process_requests=consume_requests,
    )

    assert next(iterator) == (
        10,
        GenerationOutput(prompt="alpha", text="out::alpha", metadata={}),
    )
    assert third_request_seen.wait(timeout=1.0)

    iterator.close()

    assert stop_seen.wait(timeout=1.0)


def test_stream_request_results_applies_backpressure_to_request_producer() -> None:
    fifth_request_seen = threading.Event()
    stop_seen = threading.Event()
    produced_count = {"value": 0}

    def iter_requests():
        for index in range(100):
            produced_count["value"] += 1
            if produced_count["value"] >= 5:
                fifth_request_seen.set()
            yield index, GenerationRequest(prompt=f"prompt::{index}")

    def consume_requests(stop_event, request_queue, put_result) -> None:
        first = request_queue.get(timeout_s=1.0)
        assert first is not None
        threading.Event().wait(0.1)
        first_key, first_request = first
        put_result(
            first_key,
            GenerationOutput(
                prompt=first_request.prompt or "",
                text=f"out::{first_request.prompt}",
                metadata={},
            ),
        )

        while not stop_event.is_set():
            threading.Event().wait(0.01)
        stop_seen.set()

    iterator = stream_request_results(
        iter_requests(),
        producer_name="test.backpressure_request_producer",
        consumer_name="test.backpressure_request_consumer",
        process_requests=consume_requests,
        request_queue_max_size=2,
    )

    assert next(iterator) == (
        0,
        GenerationOutput(prompt="prompt::0", text="out::prompt::0", metadata={}),
    )
    threading.Event().wait(0.1)
    assert produced_count["value"] <= 4
    assert not fifth_request_seen.is_set()

    iterator.close()

    assert stop_seen.wait(timeout=1.0)


def test_assert_non_main_thread_rejects_main_thread() -> None:
    """Verify that the runtime guard rejects executor work on the main thread."""

    # What this test is actually verifying:
    # The shared executor guard should fail fast on the main thread so request
    # execution cannot silently run in the caller thread.
    try:
        assert_non_main_thread()
    except AssertionError as exc:
        assert "non-main thread" in str(exc)
    else:  # pragma: no cover - the assertion must fire on the main thread
        raise AssertionError("expected RequestExecutor main-thread assertion")


def test_stream_request_results_runs_executor_on_non_main_thread() -> None:
    """Verify that stream_request_results runs process_requests on the consumer thread."""

    # What this test is actually verifying:
    # The process_requests callback should execute on the named consumer thread,
    # not on the caller thread that iterates over results.
    executor_thread_name: list[str] = []

    def consume_requests(stop_event, request_queue, put_result) -> None:
        """Record the thread name used to execute the consumer callback."""

        assert_non_main_thread()
        executor_thread_name.append(threading.current_thread().name)
        item = request_queue.get(timeout_s=1.0)
        assert item is not None
        request_key, request = item
        put_result(
            request_key,
            GenerationOutput(
                prompt=request.prompt or "",
                text="ok",
                metadata={},
            ),
        )

    iterator = stream_request_results(
        [(1, GenerationRequest(prompt="alpha"))],
        producer_name="test.request_producer",
        consumer_name="test.request_consumer",
        process_requests=consume_requests,
    )

    assert next(iterator) == (
        1,
        GenerationOutput(prompt="alpha", text="ok", metadata={}),
    )
    iterator.close()

    assert executor_thread_name == ["test.request_consumer"]
