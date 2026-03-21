from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

piqa_module = importlib.import_module("evalution.suites.piqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 2
        assert requests[0].context == "Question: Chill a drink quickly\nAnswer:"
        assert requests[0].continuation == " Put the bottle in the freezer for a short time."
        assert requests[1].continuation == " Leave the bottle near a warm oven."
        return [
            LoglikelihoodOutput(logprob=-0.4, is_greedy=True, token_count=11),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=8),
        ]


def test_piqa_scores_binary_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "goal": "Chill a drink quickly",
                "sol1": "Put the bottle in the freezer for a short time.",
                "sol2": "Leave the bottle near a warm oven.",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(piqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.piqa(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "piqa"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert result.metadata["dataset_path"] == "baber/piqa"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Question: Chill a drink quickly\nAnswer:"
    assert sample.target == "Put the bottle in the freezer for a short time."
    assert sample.prediction == "Put the bottle in the freezer for a short time."
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.scores == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert sample.metadata["choice_logprobs"] == [-0.4, -1.4]
    assert len(sample.metadata["choice_logprobs_norm"]) == 2
