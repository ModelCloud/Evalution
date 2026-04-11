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
boolq_module = importlib.import_module("evalution.benchmarks.boolq")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "Water freezes at 0 C.\nQuestion: Does water freeze at 0 C?\nAnswer:"
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_boolq_scores_boolean_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify boolq scores boolean multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "Does water freeze at 0 C",
                "passage": "Water freezes at 0 C.",
                "idx": 7,
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(boolq_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.boolq(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "boolq"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "boolq"
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Water freezes at 0 C.\nQuestion: Does water freeze at 0 C?\nAnswer:"
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["idx"] == 7


def test_boolq_question_formatter_adds_question_mark_once() -> None:
    """Verify boolq question formatter adds question mark once."""
    assert boolq_module._format_boolq_question("Is snow cold") == "Is snow cold?"
    assert boolq_module._format_boolq_question("Is snow cold?") == "Is snow cold?"
