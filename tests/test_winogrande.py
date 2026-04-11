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
winogrande_module = importlib.import_module("evalution.benchmarks.winogrande")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 5
        assert len(requests) == 2
        assert requests[0].context == "Sarah was a much better surgeon than Maria so"
        assert requests[0].continuation == " Sarah always got the easier cases."
        assert requests[1].continuation == " Maria always got the easier cases."
        return [
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=6),
        ]


def test_winogrande_scores_cloze_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify winogrande scores cloze multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "sentence": "Sarah was a much better surgeon than Maria so _ always got the easier cases.",
                "option1": "Sarah",
                "option2": "Maria",
                "answer": "2",
            }
        ]
    )
    monkeypatch.setattr(winogrande_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.winogrande(max_rows=1, batch_size=5).evaluate(FakeSession())

    assert result.name == "winogrande"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "winogrande"
    assert result.metadata["dataset_name"] == "winogrande_xl"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Sarah was a much better surgeon than Maria so"
    assert sample.target == "Maria always got the easier cases."
    assert sample.prediction == "Maria always got the easier cases."
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["sentence"] == "Sarah was a much better surgeon than Maria so _ always got the easier cases."


def test_winogrande_sentence_splitter_returns_prefix_and_suffix() -> None:
    """Verify winogrande sentence splitter returns prefix and suffix."""
    prefix, suffix = winogrande_module._split_winogrande_sentence("The trophy did not fit because _ was too large.")

    assert prefix == "The trophy did not fit because"
    assert suffix == "was too large."
