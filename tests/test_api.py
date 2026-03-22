# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import BaseEngine, BaseInferenceSession, GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")
arc_challenge_module = importlib.import_module("evalution.suites.arc_challenge")


class FakeEngine(BaseEngine):
    def __init__(self) -> None:
        self.session = FakeSession()

    def build(self, model):
        self.model = model
        return self.session

    def to_dict(self):
        return {"name": "fake"}


class FakeSession(BaseInferenceSession):
    def __init__(self) -> None:
        self.gc_calls = 0
        self.close_calls = 0

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

    def loglikelihood(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size=None):
        request_items = list(requests)
        outputs = self.generate([request for _, request in request_items], batch_size=batch_size)
        for (item_id, _request), output in zip(request_items, outputs, strict=True):
            yield item_id, output

    def gc(self) -> None:
        self.gc_calls += 1

    def close(self) -> None:
        self.close_calls += 1
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


def test_engine_runner_chains_model_and_test_runs(monkeypatch) -> None:
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

    engine = FakeEngine()
    result = evalution.engine(engine).model({"path": "/tmp/model"}).run(
        evalution.gsm8k_platinum(max_rows=1)
    )

    assert result.model["path"] == "/tmp/model"
    assert result.engine["name"] == "fake"
    assert len(result.tests) == 1
    assert result.tests[0].name == "gsm8k_platinum_cot"
    assert engine.session.close_calls == 1


def test_engine_builder_requires_model_before_run() -> None:
    builder = evalution.engine(FakeEngine())

    assert not hasattr(builder, "run")


def test_engine_rejects_objects_that_do_not_inherit_base_engine() -> None:
    class InvalidEngine:
        def build(self, model):
            del model
            return FakeSession()

    try:
        evalution.engine(InvalidEngine())
    except TypeError as exc:
        assert str(exc) == "engine must inherit BaseEngine"
    else:
        raise AssertionError("expected invalid engine to raise TypeError")


def test_run_rejects_engines_that_return_non_session_objects(monkeypatch) -> None:
    class InvalidEngine(BaseEngine):
        def build(self, model):
            del model
            return object()

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

    try:
        evalution.run(
            model={"path": "/tmp/model"},
            engine=InvalidEngine(),
            tests=[evalution.gsm8k_platinum(max_rows=1)],
        )
    except TypeError as exc:
        assert str(exc) == "engine.build(model) must return a BaseInferenceSession"
    else:
        raise AssertionError("expected invalid session to raise TypeError")


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


def test_run_calls_session_gc_between_test_suites(monkeypatch) -> None:
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

    engine = FakeEngine()
    result = evalution.run(
        model={"path": "/tmp/model"},
        engine=engine,
        tests=[
            evalution.gsm8k_platinum(max_rows=1),
            evalution.gsm8k_platinum(max_rows=1),
        ],
    )

    assert len(result.tests) == 2
    assert engine.session.gc_calls == 1
    assert engine.session.close_calls == 1


def test_engine_runner_calls_session_gc_between_test_runs(monkeypatch) -> None:
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

    engine = FakeEngine()
    result = (
        evalution.engine(engine)
        .model({"path": "/tmp/model"})
        .run(evalution.gsm8k_platinum(max_rows=1))
        .run(evalution.gsm8k_platinum(max_rows=1))
    )

    assert len(result.tests) == 2
    assert engine.session.gc_calls == 1
    assert engine.session.close_calls == 1
