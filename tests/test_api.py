from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")
arc_challenge_module = importlib.import_module("evalution.suites.arc_challenge")


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

    def describe_execution(self):
        return {
            "requested_attn_implementation": "flash_attention_2",
            "effective_attn_implementation": "paged|flash_attention_2",
            "paged_attention": True,
            "generation_backend": "continuous_batching",
            "standard_batch_size_cap": None,
        }

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
        tests=[evalution.gsm8k_platinum(max_rows=1)],
    )

    assert result.model["path"] == "/tmp/model"
    assert result.engine["name"] == "fake"
    assert result.engine["execution"]["generation_backend"] == "continuous_batching"
    assert len(result.tests) == 1
    assert result.tests[0].name == "gsm8k_platinum_cot"
    assert result.tests[0].metrics["exact_match,strict-match"] == 1.0


def test_run_accepts_arc_challenge_suite(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "Mercury_7175875",
                "question": (
                    "An astronomer observes that a planet rotates faster after a meteorite "
                    "impact. Which is the most likely effect of this increase in rotation?"
                ),
                "choices": {
                    "text": [
                        "Planetary density will decrease.",
                        "Planetary years will become longer.",
                        "Planetary days will become shorter.",
                        "Planetary gravity will become stronger.",
                    ],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            }
        ]
    )
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: dataset)
    engine = FakeEngine()
    engine.session.generate = lambda requests, *, batch_size=None: [
        GenerationOutput(
            prompt=request.prompt if request.prompt is not None else str(request.messages),
            text="The answer is C.",
        )
        for request in requests
    ]

    result = evalution.run(
        model={"path": "/tmp/model"},
        engine=engine,
        tests=[evalution.arc_challenge(max_rows=1)],
    )

    assert len(result.tests) == 1
    assert result.tests[0].name == "arc_challenge"
    assert result.tests[0].metrics["exact_match,choice-label"] == 1.0
