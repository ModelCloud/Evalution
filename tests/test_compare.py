# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import BaseEngine, BaseInferenceSession, LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
arc_challenge_module = importlib.import_module("evalution.benchmarks.arc_challenge")


class FakeEngine(BaseEngine):
    """Provide the fake engine helper used by the surrounding tests."""
    def __init__(self, *, choice_index: int, name: str) -> None:
        """Initialize this object."""
        self.session = FakeSession(choice_index=choice_index)
        self.name = name
        self.model_config = None

    def build(self, model):
        """Build build."""
        self.model_config = model
        return self.session

    def to_dict(self):
        """Implement to dict for fake engine."""
        return {"name": self.name}


class FakeSession(BaseInferenceSession):
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, *, choice_index: int) -> None:
        """Initialize this object."""
        self.choice_index = choice_index
        self.gc_calls = 0
        self.close_calls = 0

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        del requests, batch_size
        raise NotImplementedError

    def describe_execution(self):
        """Implement describe execution for fake session."""
        return {"loglikelihood_backend": "fake"}

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        del batch_size
        outputs = []
        for request_index, _request in enumerate(requests):
            outputs.append(
                LoglikelihoodOutput(
                    logprob=-0.1 if request_index == self.choice_index else -1.0,
                    is_greedy=request_index == self.choice_index,
                    token_count=1,
                )
            )
        return outputs

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        """Implement loglikelihood rolling for fake session."""
        del requests, batch_size
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size=None):
        """Generate continuous."""
        del requests, batch_size
        raise NotImplementedError

    def gc(self) -> None:
        """Release reusable intermediate state for this object."""
        self.gc_calls += 1

    def close(self) -> None:
        """Release the resources owned by this object."""
        self.close_calls += 1


def _dataset() -> Dataset:
    """Support the surrounding tests with dataset."""
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
    """Verify compare runs same suite on both lanes and computes delta."""
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(choice_index=2, name="left-engine")
    right_engine = FakeEngine(choice_index=0, name="right-engine")
    left_lane = left_engine.model(path="/tmp/left-model", label="model_a")
    right_lane = right_engine.model(path="/tmp/right-model", label="model_b")

    result = (
        evalution.compare(left_lane, right_lane)
        .run(evalution.benchmarks.arc_challenge(max_rows=1))
        .result()
    )

    assert result.left.model["path"] == "/tmp/left-model"
    assert result.right.model["path"] == "/tmp/right-model"
    assert result.left.engine["name"] == "left-engine"
    assert result.right.engine["name"] == "right-engine"
    assert len(result.tests) == 1
    assert result.tests[0].name == "arc_challenge"
    assert result.tests[0].left.metrics["acc,exam"] == 1.0
    assert result.tests[0].right.metrics["acc,exam"] == 0.0
    metric = result.tests[0].metrics["acc,exam"]
    assert metric.left_value == 1.0
    assert metric.right_value == 0.0
    assert metric.delta == 1.0
    assert metric.winner == "model_a"
    assert result.left_name == "model_a"
    assert result.right_name == "model_b"
    assert left_engine.session.close_calls == 1
    assert right_engine.session.close_calls == 1


def test_run_compare_calls_gc_between_shared_suite_list(monkeypatch) -> None:
    """Verify run compare calls gc between shared suite list."""
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(choice_index=2, name="left-engine")
    right_engine = FakeEngine(choice_index=2, name="right-engine")
    left_lane = left_engine.model(path="/tmp/left-model", label="model_a")
    right_lane = right_engine.model(path="/tmp/right-model", label="model_b")

    result = evalution.run_compare(
        left_lane,
        right_lane,
        tests=[
            evalution.benchmarks.arc_challenge(max_rows=1),
            evalution.benchmarks.arc_challenge(max_rows=1),
        ],
    )

    assert len(result.tests) == 2
    assert left_engine.session.gc_calls == 1
    assert right_engine.session.gc_calls == 1
    assert left_engine.session.close_calls == 1
    assert right_engine.session.close_calls == 1


def test_compare_defaults_lane_names_to_model_paths_when_labels_are_omitted(monkeypatch) -> None:
    """Verify compare defaults lane names to model paths when labels are omitted."""
    monkeypatch.setattr(arc_challenge_module, "load_dataset", lambda *args, **kwargs: _dataset())
    left_engine = FakeEngine(choice_index=2, name="left-engine")
    right_engine = FakeEngine(choice_index=2, name="right-engine")

    result = (
        evalution.compare(
            left_engine.model(path="/tmp/left-model"),
            right_engine.model(path="/tmp/right-model"),
        )
        .run(evalution.benchmarks.arc_challenge(max_rows=1))
        .result()
    )

    assert result.left_name == "/tmp/left-model"
    assert result.right_name == "/tmp/right-model"
