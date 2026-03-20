from __future__ import annotations

import importlib
import threading

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, GenerationRequest

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")


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

    suite = evalution.gsm8k_platinum(
        variant="cot_llama",
        limit=1,
        apply_chat_template=True,
    )
    session = FakeSession(["Reasoning. The final answer is 18"])
    result = suite.evaluate(session)

    assert result.name == "gsm8k_platinum_cot_llama"
    assert result.metrics["exact_match,strict-match"] == 1.0
    assert result.metrics["exact_match,flexible-extract"] == 1.0
    assert len(session.requests) == 1
    assert session.requests[0].messages is not None
    assert len(session.requests[0].messages) == 17
    assert result.metadata["fewshot_as_multiturn"] is True


def test_gsm8k_platinum_scores_strict_and_flexible_extract_separately(monkeypatch) -> None:
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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        limit=1,
        apply_chat_template=False,
    )
    session = FakeSession(["I think it comes out to 12 in total."])
    result = suite.evaluate(session)

    assert result.metrics["exact_match,strict-match"] == 0.0
    assert result.metrics["exact_match,flexible-extract"] == 1.0


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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=4)
    result = suite.evaluate(session)

    assert result.metrics["exact_match,strict-match"] == 1.0
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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        batch_size=2,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=4)
    result = suite.evaluate(session)

    assert result.metrics["exact_match,strict-match"] == 1.0
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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 5, batch_size=99, resolved_batch_size=3)
    result = suite.evaluate(session)

    assert result.metrics["exact_match,strict-match"] == 1.0
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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
    )
    session = FakeSession(["The answer is 42."] * 300, resolved_batch_size=32)
    suite.evaluate(session)

    assert session.resolve_calls == 1
    assert session.resolve_request_counts == [gsm8k_platinum_module._AUTO_BATCH_PREVIEW_ROWS]


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
        calls.append(kwargs)
        return dataset

    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", fake_load_dataset)

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        streaming=True,
    )
    session = FakeSession(["The answer is 42."])
    result = suite.evaluate(session)

    assert result.metadata["streaming"] is True
    assert calls
    assert calls[0]["streaming"] is True


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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        streaming=True,
    )
    session = PreparingFakeSession(
        ["The answer is 42."] * 260,
        resolved_batch_size=32,
    )

    suite.evaluate(session)

    assert session.resolve_request_counts == [gsm8k_platinum_module._AUTO_BATCH_PREVIEW_ROWS]
    assert session.prepare_batch_sizes[0] == gsm8k_platinum_module._AUTO_BATCH_PREVIEW_ROWS
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

    suite = evalution.gsm8k_platinum(
        variant="cot",
        apply_chat_template=False,
        streaming=True,
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
    assert session.prepare_batch_sizes[0] == gsm8k_platinum_module._AUTO_BATCH_PREVIEW_ROWS
    assert session.prepare_batch_sizes[1:] == [32, 12]
    assert all(size <= max_batch_size for size in session.prepare_batch_sizes[1:])
