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
xcopa_module = importlib.import_module("evalution.benchmarks.xcopa")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Premise: L'uomo apri il rubinetto.\n"
            "Question: Which option is the more likely effect?\n"
            "A. Il bagno si riempi d'acqua.\n"
            "B. L'acqua usci dal beccuccio.\n"
            "Answer:"
        )
        assert requests[0].continuation == " A"
        assert requests[1].continuation == " B"
        assert requests[2].context.startswith("Premise: La ragazza trovo un insetto")
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
        ]


def test_xcopa_scores_label_choices(monkeypatch) -> None:
    """Verify XCOPA scores label choices. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "premise": "L'uomo apri il rubinetto.",
                "choice1": "Il bagno si riempi d'acqua.",
                "choice2": "L'acqua usci dal beccuccio.",
                "question": "effect",
                "label": 1,
                "idx": 0,
                "changed": False,
            },
            {
                "premise": "La ragazza trovo un insetto nei cereali.",
                "choice1": "Verso il latte nella ciotola.",
                "choice2": "Perse l'appetito.",
                "question": "effect",
                "label": 1,
                "idx": 1,
                "changed": False,
            },
        ]
    )
    monkeypatch.setattr(xcopa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xcopa(language="it", max_rows=2, batch_size=6).evaluate(FakeSession())

    assert result.name == "xcopa_it"
    assert result.metrics == {
        "acc,ll": 0.5,
        "acc,ll_avg": 0.5,
    }
    assert result.metadata["dataset_path"] == "xcopa"
    assert result.metadata["dataset_name"] == "it"
    assert len(result.samples) == 2

    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["idx"] == 0
    assert sample.metadata["language"] == "it"
    assert sample.metadata["question"] == "effect"
    assert sample.metadata["raw_choices"] == [
        "Il bagno si riempi d'acqua.",
        "L'acqua usci dal beccuccio.",
    ]


def test_xcopa_helpers_format_prompt_and_relation() -> None:
    """Verify XCOPA helpers format prompt and relation."""
    assert xcopa_module._xcopa_relation_text("cause") == "cause"
    assert xcopa_module._xcopa_relation_text("effect") == "effect"
    assert xcopa_module._xcopa_prompt(
        "Premessa.",
        "cause",
        "Scelta uno.",
        "Scelta due.",
    ) == (
        "Premise: Premessa.\n"
        "Question: Which option is the more likely cause?\n"
        "A. Scelta uno.\n"
        "B. Scelta due.\n"
        "Answer:"
    )


def test_xcopa_rejects_unknown_language() -> None:
    """Verify XCOPA rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported xcopa language"):
        evalution.benchmarks.xcopa(language="en")
