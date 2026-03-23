# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import BaseEngine, BaseInferenceSession, GenerationOutput

arc_challenge_module = importlib.import_module("evalution.suites.arc_challenge")


class FakeEngine(BaseEngine):
    def __init__(self, *, text: str, name: str) -> None:
        self.session = FakeSession(text=text)
        self.name = name

    def build(self, model):
        self.model = model
        return self.session

    def to_dict(self):
        return {"name": self.name}


class FakeSession(BaseInferenceSession):
    def __init__(self, *, text: str) -> None:
        self.text = text
        self.gc_calls = 0
        self.close_calls = 0

    def generate(self, requests, *, batch_size=None):
        del batch_size
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=self.text,
            )
            for request in requests
        ]

    def describe_execution(self):
        return {"generation_backend": "fake"}

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


def _dataset() -> Dataset:
    return Dataset.from_list(
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


def test_compare_runs_same_suite_on_both_lanes_and_computes_delta(monkeypatch) -> None:
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(text="The answer is C.", name="left-engine")
    right_engine = FakeEngine(text="The answer is A.", name="right-engine")
    left_lane = evalution.engine(left_engine).model({"path": "/tmp/left-model"}, label="model_a")
    right_lane = evalution.engine(right_engine).model({"path": "/tmp/right-model"}, label="model_b")

    result = (
        evalution.compare(left_lane, right_lane)
        .run(evalution.arc_challenge(max_rows=1))
        .result()
    )

    assert result.left.model["path"] == "/tmp/left-model"
    assert result.right.model["path"] == "/tmp/right-model"
    assert result.left.engine["name"] == "left-engine"
    assert result.right.engine["name"] == "right-engine"
    assert len(result.tests) == 1
    assert result.tests[0].name == "arc_challenge"
    assert result.tests[0].left.metrics["exact_match,choice-label"] == 1.0
    assert result.tests[0].right.metrics["exact_match,choice-label"] == 0.0
    metric = result.tests[0].metrics["exact_match,choice-label"]
    assert metric.left_value == 1.0
    assert metric.right_value == 0.0
    assert metric.delta == 1.0
    assert metric.winner == "model_a"
    assert result.left_name == "model_a"
    assert result.right_name == "model_b"
    assert left_engine.session.close_calls == 1
    assert right_engine.session.close_calls == 1


def test_run_compare_calls_gc_between_shared_suite_list(monkeypatch) -> None:
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(text="The answer is C.", name="left-engine")
    right_engine = FakeEngine(text="The answer is C.", name="right-engine")
    left_lane = evalution.engine(left_engine).model({"path": "/tmp/left-model"}, label="model_a")
    right_lane = evalution.engine(right_engine).model({"path": "/tmp/right-model"}, label="model_b")

    result = evalution.run_compare(
        left_lane,
        right_lane,
        tests=[
            evalution.arc_challenge(max_rows=1),
            evalution.arc_challenge(max_rows=1),
        ],
    )

    assert len(result.tests) == 2
    assert left_engine.session.gc_calls == 1
    assert right_engine.session.gc_calls == 1
    assert left_engine.session.close_calls == 1
    assert right_engine.session.close_calls == 1


def test_compare_defaults_lane_names_to_model_paths_when_labels_are_omitted(monkeypatch) -> None:
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(text="The answer is C.", name="left-engine")
    right_engine = FakeEngine(text="The answer is C.", name="right-engine")

    result = (
        evalution.compare(
            evalution.engine(left_engine).model({"path": "/tmp/left-model"}),
            evalution.engine(right_engine).model({"path": "/tmp/right-model"}),
        )
        .run(evalution.arc_challenge(max_rows=1))
        .result()
    )

    assert result.left_name == "/tmp/left-model"
    assert result.right_name == "/tmp/right-model"
