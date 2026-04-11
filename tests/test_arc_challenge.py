# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
arc_challenge_module = importlib.import_module("evalution.benchmarks.arc_challenge")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, outputs: list[LoglikelihoodOutput]) -> None:
        """Initialize this object."""
        self.outputs = outputs
        self.requests = []

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 7
        assert len(requests) == 4
        self.requests.extend(requests)
        return self.outputs


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


def test_arc_challenge_scores_original_style_exam_score(monkeypatch) -> None:
    """Verify ARC challenge scores original style exam score. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.benchmarks.arc_challenge(max_rows=1, batch_size=7)
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.name == "arc_challenge"
    assert result.metrics == {"acc,exam": 1.0}
    assert result.metadata == {
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Challenge",
        "split": "test",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_exam_score",
        "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
    }

    sample = result.samples[0]
    assert sample.prompt == (
        "Question: An astronomer observes that a planet rotates faster after a meteorite "
        "impact. Which is the most likely effect of this increase in rotation?\nAnswer:"
    )
    assert sample.target == "Planetary days will become shorter."
    assert sample.prediction == "Planetary days will become shorter."
    assert sample.extracted == {
        "gold_index": "2",
        "selected_indices": "2",
        "selected_labels": "C",
    }
    assert sample.metadata["id"] == "Mercury_7175875"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_logprobs"] == [-1.3, -1.1, -0.2, -1.0]
    assert sample.metadata["selected_count"] == 1
    assert session.requests[0].context.endswith("\nAnswer:")
    assert session.requests[0].continuation == " Planetary density will decrease."
    assert session.requests[2].continuation == " Planetary days will become shorter."


def test_arc_challenge_awards_partial_credit_for_tied_top_choices(monkeypatch) -> None:
    """Verify ARC challenge awards partial credit for tied top choices."""
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.benchmarks.arc_challenge(max_rows=1, batch_size=7)
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-0.4, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.4, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.metrics == {"acc,exam": 0.5}
    assert result.samples[0].prediction == (
        "Planetary density will decrease. | Planetary days will become shorter."
    )
    assert result.samples[0].extracted["selected_indices"] == "0,2"
    assert result.samples[0].extracted["selected_labels"] == "A,C"
    assert result.samples[0].metadata["selected_count"] == 2


def test_arc_challenge_passes_streaming_flag_to_load_dataset(monkeypatch) -> None:
    """Verify ARC challenge passes streaming flag to load dataset."""
    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args, **kwargs):
        """Support the surrounding tests with fake load dataset."""
        del args
        if "stream" in kwargs:
            raise TypeError("unexpected keyword argument 'stream'")
        calls.append(kwargs)
        return _dataset()

    monkeypatch.setattr(arc_challenge_module, "load_dataset", fake_load_dataset)

    suite = evalution.benchmarks.arc_challenge(
        max_rows=1,
        batch_size=7,
        stream=True,
    )
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.metadata["stream"] is True
    assert calls
    assert calls[0]["streaming"] is True


def test_arc_challenge_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify ARC challenge can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    class LabelPermutationSession:
        """Define the label permutation session helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for label permutation session."""
            assert batch_size == 7
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=5),
                    LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
                    LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
                ]

            assert len(requests) == 24
            gold_text = "Planetary days will become shorter."
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. {gold_text}" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.5,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.arc_challenge(
        max_rows=1,
        batch_size=7,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,exam": 1.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "2"
    assert result.samples[0].metadata["label_permutation_count"] == 6
