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
copa_module = importlib.import_module("evalution.benchmarks.copa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "The man turned on the faucet therefore"
        assert requests[0].continuation == " the toilet filled with water."
        assert requests[1].continuation == " water flowed from the spout."
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=6),
        ]


def test_copa_scores_causal_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify COPA scores causal multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "premise": "The man turned on the faucet.",
                "choice1": "The toilet filled with water.",
                "choice2": "Water flowed from the spout.",
                "question": "effect",
                "idx": 0,
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(copa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.copa(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "copa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "copa"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "The man turned on the faucet therefore"
    assert sample.target == "water flowed from the spout."
    assert sample.prediction == "water flowed from the spout."
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["idx"] == 0
    assert sample.metadata["question"] == "effect"


def test_copa_helpers_format_connectors_and_choices() -> None:
    """Verify COPA helpers format connectors and choices."""
    assert copa_module._copa_connector("cause") == "because"
    assert copa_module._copa_choice_text("Water flowed.") == "water flowed."
