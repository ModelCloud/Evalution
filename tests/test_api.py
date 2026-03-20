from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")


class FakeEngine:
    def __init__(self) -> None:
        self.session = FakeSession()

    def build(self, model):
        self.model = model
        return self.session

    def to_dict(self):
        return {"name": "fake"}


class FakeSession:
    def generate(self, requests, *, batch_size=None):
        del batch_size
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text="The answer is 42.",
            )
            for request in requests
        ]

    def close(self) -> None:
        return None


def test_run_accepts_dict_model_and_returns_structured_results(monkeypatch) -> None:
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

    result = evalution.run(
        model={"path": "/tmp/model"},
        engine=FakeEngine(),
        tests=[evalution.gsm8k_platinum(limit=1)],
    )

    assert result.model["path"] == "/tmp/model"
    assert result.engine["name"] == "fake"
    assert len(result.tests) == 1
    assert result.tests[0].name == "gsm8k_platinum_cot"
    assert result.tests[0].metrics["exact_match,strict-match"] == 1.0
