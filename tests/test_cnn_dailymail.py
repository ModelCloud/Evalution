# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput
from evalution.scorers.summary_rouge import summary_rouge_scores

cnn_dailymail_module = importlib.import_module("evalution.benchmarks.cnn_dailymail")


class FakeSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt == (
            "Summarize the following news article.\n\n"
            "Article:\nA cat rescued a child from a tree in Seattle.\n\n"
            "Summary:"
        )
        assert requests[0].max_new_tokens == 128
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="A cat rescued a child from a tree in Seattle.",
            )
        ]


def test_cnn_dailymail_scores_generated_summary_rouge(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "article": "A cat rescued a child from a tree in Seattle.",
                "highlights": "A cat rescued a child.\nThe rescue happened in Seattle.",
                "id": "row-1",
            }
        ]
    )
    monkeypatch.setattr(cnn_dailymail_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.cnn_dailymail(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "cnn_dailymail"
    assert result.metrics == summary_rouge_scores(
        "A cat rescued a child from a tree in Seattle.",
        "A cat rescued a child.\nThe rescue happened in Seattle.",
    )
    assert result.metadata == {
        "dataset_path": "cnn_dailymail",
        "dataset_name": "3.0.0",
        "split": "validation",
        "stream": True,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_summary_rouge",
        "primary_metric": "rougeLsum",
    }

    sample = result.samples[0]
    assert sample.target == "A cat rescued a child.\nThe rescue happened in Seattle."
    assert sample.prediction == "A cat rescued a child from a tree in Seattle."
    assert sample.metadata["id"] == "row-1"
    assert sample.metadata["article_chars"] > 0
    assert sample.metadata["reference_lines"] == 2


def test_cnn_dailymail_prompt_formats_article() -> None:
    assert cnn_dailymail_module._cnn_dailymail_prompt("  News text.  ") == (
        "Summarize the following news article.\n\n"
        "Article:\nNews text.\n\n"
        "Summary:"
    )
