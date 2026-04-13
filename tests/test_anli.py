# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
anli_module = importlib.import_module("evalution.benchmarks.anli")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 5
        assert len(requests) == 3
        assert requests[0].continuation == " True"
        assert requests[1].continuation == " Neither"
        assert requests[2].continuation == " False"
        assert "True, False, or Neither?\nAnswer:" in requests[0].context
        return [
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


@pytest.mark.parametrize(
    ("factory_name", "expected_name", "expected_split"),
    [
        ("anli_r1", "anli_r1", "test_r1"),
        ("anli_r2", "anli_r2", "test_r2"),
        ("anli_r3", "anli_r3", "test_r3"),
    ],
)
def test_anli_round_scores_three_way_nli(monkeypatch, factory_name, expected_name, expected_split) -> None:
    """Verify anli round scores three way nli. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "uid": "anli-1",
                "premise": "Ernest Jones is a British jeweller and watchmaker.",
                "hypothesis": "The company is based in Europe.",
                "label": 2,
                "reason": "The premise does not say where the company is based today.",
            }
        ]
    )
    monkeypatch.setattr(anli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=5).evaluate(FakeSession())

    assert result.name == expected_name
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "facebook/anli"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == expected_split
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == (
        "Ernest Jones is a British jeweller and watchmaker.\n"
        "Question: The company is based in Europe. True, False, or Neither?\n"
        "Answer:"
    )
    assert sample.target == "False"
    assert sample.prediction == "False"
    assert sample.extracted == {
        "gold_index": "2",
        "predicted_index": "2",
        "predicted_index_norm": "2",
    }
    assert sample.metadata["uid"] == "anli-1"
    assert sample.metadata["round"] == expected_name.removeprefix("anli_")
    assert sample.metadata["choice_labels"] == ["A", "B", "C"]
    assert sample.metadata["choice_texts"] == ["True", "Neither", "False"]


def test_anli_round_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify anli round can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "uid": "anli-2",
                "premise": "A restaurant is open late into the evening.",
                "hypothesis": "The restaurant is closed right now.",
                "label": 1,
                "reason": "The premise does not establish the current state at this exact moment.",
            }
        ]
    )
    monkeypatch.setattr(anli_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        """Define the label permutation session helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for label permutation session."""
            assert batch_size == 5
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 3
                return [
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 9
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. Neither" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.5,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.anli_r1(
        max_rows=1,
        batch_size=5,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "1"
    assert result.samples[0].metadata["label_permutation_count"] == 3
