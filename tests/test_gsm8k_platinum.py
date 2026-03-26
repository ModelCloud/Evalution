# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import replace
from queue import Queue

import pcre
import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.scorers.gsm8k import INVALID_ANSWER
from evalution.scorers.gsm8k import extract_format_insensitive_numeric_answer
from evalution.scorers.gsm8k import extract_gsm8k_platinum_reference_answer
from evalution.benchmarks.execution import AUTO_BATCH_PREVIEW_ROWS
from evalution.benchmarks.execution import PreparedSample
from evalution.benchmarks.execution import _prefetch_executor
from evalution.benchmarks.execution import iter_prefetched_batches
from evalution.benchmarks.execution import iter_prefetched_samples

gsm8k_platinum_module = importlib.import_module("evalution.benchmarks.gsm8k_platinum")

_BOXED_RE = pcre.compile(r"\\boxed\{(?:\\text\{)?([^\\{}]+)\}")
_PLATINUM_REFERENCE_NUMBER_RE = pcre.compile(r"-?[0-9.]*[0-9]")
_TRAILING_ZERO_RE = pcre.compile(r"\.0+$")


def _official_gsm8k_platinum_reference_answer(output: str) -> str:
    text = output
    lowered_without_stars = text.lower().replace("*", "")
    if "answer:" in lowered_without_stars:
        answer_section = text.lower().split("answer: ")[-1]
        boxed_match = _BOXED_RE.search(answer_section)
        if boxed_match is not None:
            text = f"Answer: {boxed_match.group(1)}"
    else:
        boxed_match = _BOXED_RE.search(text)
        if boxed_match is not None:
            text = f"Answer: {boxed_match.group(1)}"
        else:
            last_line = text.strip("\n").split("\n")[-1].lower()
            text = f"Answer: {last_line}"

    match = _PLATINUM_REFERENCE_NUMBER_RE.search(
        text.replace("*", "").replace("#", "").lower().split("answer: ")[-1].replace(",", "")
    )
    if match is None:
        return INVALID_ANSWER
    return _TRAILING_ZERO_RE.sub("", match.group())


class FakeSession:
    def __init__(
        self,
        responses: list[str],
        *,
        batch_size: int | None = None,
        resolved_batch_size: int | None = None,
    ) -> None:
        self.responses = responses
        self.requests = []
        self.batch_size = batch_size
        self.resolved_batch_size = resolved_batch_size
        self.resolve_request_counts: list[int] = []
        self.generate_batch_sizes: list[int | None] = []
        self._response_index = 0
        self.resolve_calls = 0

    def resolve_batch_size(self, requests) -> int:
        self.resolve_calls += 1
        self.resolve_request_counts.append(len(requests))
        return self.resolved_batch_size

    def generate(self, requests, *, batch_size=None):
        self.generate_batch_sizes.append(batch_size)
        self.requests.extend(requests)
        batch_responses = self.responses[self._response_index : self._response_index + len(requests)]
        self._response_index += len(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, batch_responses, strict=True)
        ]

    def close(self) -> None:
        return None


class PreparingFakeSession(FakeSession):
    def __init__(
        self,
        responses: list[str],
        *,
        batch_size: int | None = None,
        resolved_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            responses,
            batch_size=batch_size,
            resolved_batch_size=resolved_batch_size,
        )
        self.prepare_thread_names: list[str] = []
        self.prepare_batch_sizes: list[int] = []

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        self.prepare_thread_names.append(threading.current_thread().name)
        self.prepare_batch_sizes.append(len(requests))
        prepared: list[GenerationRequest] = []
        for index, request in enumerate(requests):
            prompt = request.prompt or str(request.messages)
            prepared.append(
                GenerationRequest(
                    prompt=request.prompt,
                    messages=request.messages,
                    rendered_prompt=prompt,
                    input_ids=[index + 1, index + 2, index + 3],
                    add_generation_prompt=request.add_generation_prompt,
                    stop=list(request.stop),
                    max_new_tokens=request.max_new_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    metadata=dict(request.metadata),
                )
            )
        return prepared


class ContinuousPreparingFakeSession(PreparingFakeSession):
    def __init__(
        self,
        responses: list[str],
        *,
        batch_size: int | None = None,
        resolved_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            responses,
            batch_size=batch_size,
            resolved_batch_size=resolved_batch_size,
        )
        self.continuous_batch_sizes: list[int | None] = []
        self.max_inflight = 0
        self.completion_order: list[int] = []

    def generate(self, requests, *, batch_size=None):
        del requests, batch_size
        raise AssertionError("continuous sessions should not fall back to generate()")

    def generate_continuous(self, requests, *, batch_size=None):
        self.continuous_batch_sizes.append(batch_size)
        request_iter = iter(requests)
        active: list[tuple[int, GenerationRequest, str]] = []

        def submit_one() -> bool:
            try:
                request_key, request = next(request_iter)
            except StopIteration:
                return False
            response = self.responses[self._response_index]
            self._response_index += 1
            active.append((request_key, request, response))
            self.max_inflight = max(self.max_inflight, len(active))
            return True

        while batch_size is not None and len(active) < batch_size and submit_one():
            continue
        if batch_size is None and not active:
            submit_one()

        while active:
            active_index = 1 if len(active) > 1 else 0
            request_key, request, response = active.pop(active_index)
            self.completion_order.append(request_key)
            yield request_key, GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            while batch_size is not None and len(active) < batch_size and submit_one():
                continue
            if batch_size is None and not active:
                submit_one()


def test_gsm8k_platinum_cot_llama_uses_multiturn_chat_by_default(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "How many apples remain if you start with 20 and eat 2?",
                "answer": "20 - 2 = 18\n#### 18",
                "cleaning_status": "consensus",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot_llama",
        max_rows=1,
        apply_chat_template=True,
    )
    session = FakeSession(["Reasoning. The final answer is 18"])
    result = suite.evaluate(session)

    assert result.name == "gsm8k_platinum_cot_llama"
    assert set(result.metrics) == {"acc,num"}
    assert result.metrics["acc,num"] == 1.0
    assert len(session.requests) == 1
    assert session.requests[0].messages is not None
    assert len(session.requests[0].messages) == 17
    assert result.metadata["fewshot_as_multiturn"] is True


def test_gsm8k_platinum_scores_numeric_primary(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 7 plus 5?",
                "answer": "7 + 5 = 12\n#### 12",
                "cleaning_status": "consensus",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        max_rows=1,
        apply_chat_template=False,
    )
    session = FakeSession(["I think it comes out to 12 in total."])
    result = suite.evaluate(session)

    assert set(result.metrics) == {"acc,num"}
    assert result.metrics["acc,num"] == 1.0
    assert result.samples[0].extracted["numeric-extract"] == "12"
    assert set(result.samples[0].extracted) == {"numeric-extract"}


def test_gsm8k_platinum_reference_parser_matches_madrylab_release_cases() -> None:
    cases = [
        ("Answer: 42", "42"),
        ("Reasoning\n\\boxed{18}", "18"),
        ("The answer is 7.", "7"),
        ("No numeric answer", INVALID_ANSWER),
    ]

    for completion, expected in cases:
        assert _official_gsm8k_platinum_reference_answer(completion) == expected
        assert extract_gsm8k_platinum_reference_answer(completion) == expected

    assert extract_format_insensitive_numeric_answer("The answer is 7.") == "7"


def test_gsm8k_platinum_uses_engine_batch_size_by_default(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": f"What is 40 plus {offset}?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for offset in range(5)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=4)
    result = suite.evaluate(session)

    assert result.metrics["acc,num"] == 1.0
    assert session.generate_batch_sizes == [4, 1]


def test_gsm8k_platinum_suite_batch_size_overrides_engine_default(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": f"What is 40 plus {offset}?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for offset in range(5)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        batch_size=2,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=4)
    result = suite.evaluate(session)

    assert result.metrics["acc,num"] == 1.0
    assert session.generate_batch_sizes == [2, 2, 1]


def test_gsm8k_platinum_uses_session_batch_size_resolver(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": f"What is 40 plus {offset}?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for offset in range(5)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=99, resolved_batch_size=3)
    result = suite.evaluate(session)

    assert result.metrics["acc,num"] == 1.0
    assert session.resolve_calls == 1
    assert session.generate_batch_sizes == [3, 2]


def test_gsm8k_platinum_uses_bounded_preview_for_auto_batch_size_resolution(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": f"What is 40 plus {offset}?",
                "answer": f"40 + {offset} = {40 + offset}\n#### {40 + offset}",
                "cleaning_status": "consensus",
            }
            for offset in range(300)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 300, resolved_batch_size=32)
    suite.evaluate(session)

    assert session.resolve_calls == 1
    assert session.resolve_request_counts == [AUTO_BATCH_PREVIEW_ROWS]


def test_gsm8k_platinum_passes_streaming_flag_to_load_dataset(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args, **kwargs):
        del args
        if "stream" in kwargs:
            raise TypeError("unexpected keyword argument 'stream'")
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", fake_load_dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
    )
    session = FakeSession(["The answer is 42."])
    result = suite.evaluate(session)

    assert result.metadata["stream"] is True
    assert calls
    assert calls[0]["streaming"] is True


def test_gsm8k_platinum_stream_rejects_non_native_order(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
        order="length|desc",
    )

    with pytest.raises(ValueError, match="order='native'"):
        suite.evaluate(FakeSession(["The answer is 42."]))


def test_gsm8k_platinum_prefetches_remaining_streaming_batches_on_background_thread(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": f"What is 40 plus {offset}?",
                "answer": f"40 + {offset} = {40 + offset}\n#### {40 + offset}",
                "cleaning_status": "consensus",
            }
            for offset in range(260)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
    )
    session = PreparingFakeSession(
        ["The answer is 42."] * 260,
        resolved_batch_size=32,
    )

    suite.evaluate(session)

    assert session.resolve_request_counts == [AUTO_BATCH_PREVIEW_ROWS]
    assert session.prepare_batch_sizes[0] == AUTO_BATCH_PREVIEW_ROWS
    assert session.prepare_thread_names[0] == "MainThread"
    assert any(
        name.startswith("evalution-prefetch")
        for name in session.prepare_thread_names[1:]
    )
    assert sum(session.prepare_batch_sizes) == 260


def test_gsm8k_platinum_streaming_prefetch_and_generation_respect_resolved_batch_size(
    monkeypatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for _ in range(300)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
    )
    max_batch_size = 32
    session = PreparingFakeSession(
        ["The answer is 42."] * 300,
        resolved_batch_size=max_batch_size,
    )

    result = suite.evaluate(session)

    assert len(result.samples) == 300
    assert session.generate_batch_sizes == [32, 32, 32, 32, 32, 32, 32, 32, 32, 12]
    assert all(size is not None and size <= max_batch_size for size in session.generate_batch_sizes)
    assert session.prepare_batch_sizes[0] == AUTO_BATCH_PREVIEW_ROWS
    assert session.prepare_batch_sizes[1:] == [32, 12]
    assert all(size <= max_batch_size for size in session.prepare_batch_sizes[1:])
    assert result.metadata["generation_submission_mode"] == "fixed_batches"


def test_gsm8k_platinum_streaming_uses_continuous_generation_to_refill_slots(
    monkeypatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for _ in range(300)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
    )
    max_batch_size = 32
    session = ContinuousPreparingFakeSession(
        ["The answer is 42."] * 300,
        resolved_batch_size=max_batch_size,
    )

    result = suite.evaluate(session)

    assert len(result.samples) == 300
    assert [sample.index for sample in result.samples] == list(range(300))
    assert session.generate_batch_sizes == []
    assert session.continuous_batch_sizes == [32]
    assert session.max_inflight == 32
    assert session.completion_order[0] == 1
    assert session.completion_order.index(0) > 0
    assert session.prepare_batch_sizes[0] == AUTO_BATCH_PREVIEW_ROWS
    assert session.prepare_batch_sizes[1:] == [32, 12]
    assert result.metadata["generation_submission_mode"] == "continuous_refill"


def test_gsm8k_platinum_skips_auto_batch_preview_when_suite_batch_size_is_fixed(
    monkeypatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
            for _ in range(300)
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        stream=True,
        batch_size=24,
    )
    session = ContinuousPreparingFakeSession(["The answer is 42."] * 300)

    result = suite.evaluate(session)

    assert len(result.samples) == 300
    assert session.continuous_batch_sizes == [24]
    assert session.prepare_batch_sizes[0] == 24
    assert AUTO_BATCH_PREVIEW_ROWS not in session.prepare_batch_sizes


def test_iter_prefetched_batches_closes_promptly_when_consumer_stops_early() -> None:
    class FakePrepareBar:
        def next(self) -> FakePrepareBar:
            return self

        def draw(self) -> FakePrepareBar:
            return self

    class TrackingSession:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.third_prepare_started = threading.Event()

        def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
            self.prepare_calls += 1
            if self.prepare_calls >= 3:
                self.third_prepare_started.set()
            return requests

    def make_sample(index: int) -> PreparedSample:
        prompt = f"Q: {index}\nA:"
        return PreparedSample(
            index=index,
            doc={"question": str(index), "answer": "42\n#### 42", "cleaning_status": "consensus"},
            target="42",
            request=GenerationRequest(
                prompt=prompt,
                rendered_prompt=prompt,
                input_ids=[1, 2, 3],
            ),
        )

    session = TrackingSession()
    iterator = iter_prefetched_batches(
        session,
        [make_sample(0)],
        iter([make_sample(1), make_sample(2), make_sample(3), make_sample(4)]),
        batch_size=1,
        prepare_bar=FakePrepareBar(),
    )

    first_batch = next(iterator)

    assert len(first_batch) == 1
    assert first_batch[0].index == 0

    close_errors: list[BaseException] = []

    def close_iterator() -> None:
        try:
            iterator.close()
        except BaseException as exc:  # pragma: no cover - asserted below
            close_errors.append(exc)

    close_thread = threading.Thread(target=close_iterator)
    close_thread.start()
    close_thread.join(timeout=1.0)

    assert not close_errors
    assert not close_thread.is_alive()
    assert session.prepare_calls >= 1


@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="requires Python free-threading with GIL disabled",
)
def test_iter_prefetched_batches_allows_nogil_inflight_work_while_next_batch_prepares() -> None:
    batch_size = 32
    total_rows = 64

    class FakePrepareBar:
        def next(self) -> FakePrepareBar:
            return self

        def draw(self) -> FakePrepareBar:
            return self

    class BlockingPrepareSession:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.next_batch_prepare_started = threading.Event()
            self.allow_next_batch_prepare_finish = threading.Event()

        def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                self.next_batch_prepare_started.set()
                assert self.allow_next_batch_prepare_finish.wait(timeout=1.0)
            return [
                replace(
                    request,
                    rendered_prompt=request.prompt,
                    input_ids=[index + 1, index + 2, index + 3],
                )
                for index, request in enumerate(requests)
            ]

    def make_sample(index: int, *, prepared: bool) -> PreparedSample:
        prompt = f"Q: {index}\nA:"
        return PreparedSample(
            index=index,
            doc={"question": str(index), "answer": "42\n#### 42", "cleaning_status": "consensus"},
            target="42",
            request=GenerationRequest(
                prompt=prompt,
                rendered_prompt=prompt if prepared else None,
                input_ids=[index + 1, index + 2, index + 3] if prepared else None,
            ),
        )

    session = BlockingPrepareSession()
    iterator = iter_prefetched_batches(
        session,
        [make_sample(index, prepared=True) for index in range(batch_size)],
        iter(make_sample(index, prepared=False) for index in range(batch_size, total_rows)),
        batch_size=batch_size,
        prepare_bar=FakePrepareBar(),
    )

    ready_samples: Queue[PreparedSample | None] = Queue()
    feeder_errors: list[BaseException] = []
    worker_errors: list[BaseException] = []
    processed_indexes: list[int] = []
    initial_progress_reached = threading.Event()
    state_lock = threading.Lock()
    completed_initial = 0

    def feed_ready_samples() -> None:
        try:
            for prefetched_batch in iterator:
                for sample in prefetched_batch:
                    ready_samples.put(sample)
        except BaseException as exc:  # pragma: no cover - asserted below
            feeder_errors.append(exc)
        finally:
            for _ in range(batch_size):
                ready_samples.put(None)

    def run_slot_worker() -> None:
        nonlocal completed_initial
        try:
            while True:
                sample = ready_samples.get()
                if sample is None:
                    return

                if sample.index < 8:
                    duration_s = 0.003
                elif sample.index < batch_size:
                    duration_s = 0.03 + ((sample.index - 8) % 4) * 0.003
                else:
                    duration_s = 0.004 + ((sample.index - batch_size) % 4) * 0.001

                deadline = time.perf_counter() + duration_s
                spin = 0
                while time.perf_counter() < deadline:
                    spin += 1

                with state_lock:
                    processed_indexes.append(sample.index)
                    if sample.index < batch_size:
                        completed_initial += 1
                        if completed_initial >= 8:
                            initial_progress_reached.set()
                del spin
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    feeder_thread = threading.Thread(target=feed_ready_samples)
    workers = [threading.Thread(target=run_slot_worker) for _ in range(batch_size)]

    feeder_thread.start()
    for worker in workers:
        worker.start()

    assert session.next_batch_prepare_started.wait(timeout=1.0)
    assert initial_progress_reached.wait(timeout=1.0)

    session.allow_next_batch_prepare_finish.set()

    feeder_thread.join(timeout=2.0)
    for worker in workers:
        worker.join(timeout=2.0)

    assert not feeder_errors
    assert not worker_errors
    assert not feeder_thread.is_alive()
    assert all(not worker.is_alive() for worker in workers)
    assert session.prepare_calls == 1
    assert sorted(processed_indexes) == list(range(total_rows))
    assert any(index >= batch_size for index in processed_indexes)


def test_iter_prefetched_samples_refills_partial_free_capacity_with_partial_tokenization() -> None:
    class FakePrepareBar:
        def next(self) -> FakePrepareBar:
            return self

        def draw(self) -> FakePrepareBar:
            return self

    class TrackingSession:
        def __init__(self) -> None:
            self.prepare_batch_sizes: list[int] = []
            self.second_prepare_started = threading.Event()

        def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
            self.prepare_batch_sizes.append(len(requests))
            if len(self.prepare_batch_sizes) == 2:
                self.second_prepare_started.set()
            return [
                replace(
                    request,
                    rendered_prompt=request.prompt,
                    input_ids=[index + 1, index + 2, index + 3],
                )
                for index, request in enumerate(requests)
            ]

    def make_sample(index: int) -> PreparedSample:
        prompt = f"Q: {index}\nA:"
        return PreparedSample(
            index=index,
            doc={"question": str(index), "answer": "42\n#### 42", "cleaning_status": "consensus"},
            target="42",
            request=GenerationRequest(prompt=prompt),
        )

    session = TrackingSession()
    iterator = iter_prefetched_samples(
        session,
        [],
        iter(make_sample(index) for index in range(36)),
        batch_size=32,
        prepare_bar=FakePrepareBar(),
        pool_size=32,
    )

    consumed = [next(iterator) for _ in range(4)]

    assert [sample.index for sample in consumed] == [0, 1, 2, 3]
    assert session.second_prepare_started.wait(timeout=1.0)
    assert session.prepare_batch_sizes[0] == 32
    assert 0 < session.prepare_batch_sizes[1] < 32

    remainder = list(iterator)
    assert [sample.index for sample in remainder] == list(range(4, 36))


def test_prefetch_executor_is_reused() -> None:
    first = _prefetch_executor()
    second = _prefetch_executor()

    assert first is second
