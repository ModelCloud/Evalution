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
xwinograd_module = importlib.import_module("evalution.benchmarks.xwinograd")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "The city councilmen refused the demonstrators a permit because the demonstrators"
        assert requests[0].continuation == " feared violence."
        assert requests[1].context == "The city councilmen refused the demonstrators a permit because The city councilmen"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=3),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=4),
        ]


def test_xwinograd_scores_partial_evaluation_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify xwinograd scores partial evaluation multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "sentence": "The city councilmen refused the demonstrators a permit because _ feared violence.",
                "option1": "the demonstrators",
                "option2": "The city councilmen",
                "answer": "2",
            }
        ]
    )
    monkeypatch.setattr(xwinograd_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xwinograd_en(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "xwinograd_en"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "Muennighoff/xwinograd"
    assert result.metadata["dataset_name"] == "en"
    assert result.metadata["split"] == "test"
    assert result.metadata["prompt_variant"] == "partial_evaluation_blank_replacement"

    sample = result.samples[0]
    assert sample.prompt == "The city councilmen refused the demonstrators a permit because _ feared violence."
    assert sample.target == "The city councilmen refused the demonstrators a permit because The city councilmen feared violence."
    assert sample.prediction == sample.target
    assert sample.metadata["language"] == "en"
    assert sample.metadata["choice_labels"] == ["A", "B"]
    assert sample.metadata["choice_texts"] == ["the demonstrators", "The city councilmen"]
    assert sample.metadata["answer_label"] == "2"


def test_xwinograd_helper_handles_spaced_and_unspaced_blanks() -> None:
    """Verify xwinograd helper handles spaced and unspaced blanks."""
    assert xwinograd_module._xwinograd_choice_contexts_and_suffix(
        "The city councilmen refused the demonstrators a permit because _ feared violence.",
        "the demonstrators",
        "The city councilmen",
    ) == (
        [
            "The city councilmen refused the demonstrators a permit because the demonstrators",
            "The city councilmen refused the demonstrators a permit because The city councilmen",
        ],
        " feared violence.",
    )
    assert xwinograd_module._xwinograd_choice_contexts_and_suffix(
        "市政府拒绝给示威者颁发游行许可证，因_担心暴力事件。",
        "市政府",
        "示威者",
    ) == (
        [
            "市政府拒绝给示威者颁发游行许可证，因市政府",
            "市政府拒绝给示威者颁发游行许可证，因示威者",
        ],
        "担心暴力事件。",
    )


def test_xwinograd_rejects_unknown_language() -> None:
    """Verify xwinograd rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported xwinograd language"):
        xwinograd_module.XWinograd(language="de")
