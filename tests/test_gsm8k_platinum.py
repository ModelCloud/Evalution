from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")


class FakeSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        del batch_size
        self.requests = requests
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        return None


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
