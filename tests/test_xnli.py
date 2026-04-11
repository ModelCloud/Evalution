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
xnli_module = importlib.import_module("evalution.benchmarks.xnli")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 4
        assert len(requests) == 6
        assert requests[0].context == (
            "Premise: And he said, Mama, I'm home.\n"
            "Hypothesis: He called his mom as soon as the school bus dropped him off.\n"
            "Question: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\n"
            "Answer:"
        )
        assert [request.continuation for request in requests[:3]] == [
            " entailment",
            " neutral",
            " contradiction",
        ]
        return [
            LoglikelihoodOutput(logprob=-2.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
        ]


def test_xnli_scores_three_way_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify XNLI scores three way multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "premise": "And he said, Mama, I'm home.",
                "hypothesis": "He called his mom as soon as the school bus dropped him off.",
                "label": 1,
            },
            {
                "premise": "A black race car starts up in front of a crowd of people.",
                "hypothesis": "A man is driving down a lonely road.",
                "label": 0,
            },
        ]
    )
    monkeypatch.setattr(xnli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xnli(language="en", max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "xnli_en"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "facebook/xnli"
    assert result.metadata["dataset_name"] == "en"
    assert result.metadata["split"] == "validation"

    first_sample = result.samples[0]
    assert first_sample.target == "neutral"
    assert first_sample.prediction == "neutral"
    assert first_sample.metadata["language"] == "en"
    assert first_sample.metadata["choice_texts"] == ["entailment", "neutral", "contradiction"]

    second_sample = result.samples[1]
    assert second_sample.target == "entailment"
    assert second_sample.prediction == "entailment"


def test_xnli_prompt_helper_formats_nli_prompt() -> None:
    """Verify XNLI prompt helper formats nli prompt."""
    assert (
        xnli_module._xnli_prompt("Premise text", "Hypothesis text")
        == "Premise: Premise text\nHypothesis: Hypothesis text\nQuestion: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\nAnswer:"
    )


def test_xnli_rejects_unknown_language() -> None:
    """Verify XNLI rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported xnli language"):
        evalution.benchmarks.xnli(language="zzz")


def test_xnli_rejects_dataset_name_mismatch() -> None:
    """Verify XNLI rejects dataset name mismatch."""
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.xnli(language="en", dataset_name="fr")
