from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput
from evalution.suites import base as base_suite_module

gsm8k_module = importlib.import_module("evalution.suites.gsm8k")


class FakeSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        del batch_size
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        return None


def test_gsm8k_suite_uses_shared_pipeline_and_default_task_names(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.gsm8k(max_rows=1)
    session = FakeSession(["The answer is 42."])
    result = suite.evaluate(session)

    assert result.name == "gsm8k_cot"
    assert result.metrics["exact_match,strict-match"] == 1.0
    assert result.metadata["dataset_path"] == "openai/gsm8k"
    assert result.metadata["variant"] == "cot"


def test_gsm8k_base_variant_omits_platinum_specific_cleaning_metadata(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.gsm8k(
        variant="base",
        max_rows=1,
    )
    session = FakeSession(["40 + 2 = 42\n#### 42"])
    result = suite.evaluate(session)

    assert result.name == "gsm8k"
    assert result.samples[0].metadata == {}


def test_gsm8k_logs_executed_rows_and_warns_on_short_generation(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            },
            {
                "question": "What is 40 plus 3?",
                "answer": "40 + 3 = 43\n#### 43",
            },
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeLogger:
        def __init__(self) -> None:
            self.info_messages: list[str] = []
            self.warning_messages: list[str] = []

        def info(self, message: str, *args) -> None:
            self.info_messages.append(message % args if args else message)

        def warning(self, message: str, *args) -> None:
            self.warning_messages.append(message % args if args else message)

    class ShortOutputSession:
        def generate(self, requests, *, batch_size=None):
            del batch_size
            request = requests[0]
            return [
                GenerationOutput(
                    prompt=request.prompt if request.prompt is not None else str(request.messages),
                    text="The answer is 42.",
                )
            ]

        def close(self) -> None:
            return None

    fake_logger = FakeLogger()
    monkeypatch.setattr(base_suite_module, "get_logger", lambda: fake_logger)

    suite = evalution.gsm8k(max_rows=2, batch_size=2)
    result = suite.evaluate(ShortOutputSession())

    assert len(result.samples) == 1
    assert any("gsm8k_cot: executed 1/2 sample(s)" in message for message in fake_logger.info_messages)
    assert any(
        "gsm8k_cot: only executed 1/2 sample(s); generation returned fewer outputs than expected"
        in message
        for message in fake_logger.warning_messages
    )
