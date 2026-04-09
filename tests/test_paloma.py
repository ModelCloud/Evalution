# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import math

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import RollingLoglikelihoodOutput

paloma_module = importlib.import_module("evalution.benchmarks.paloma")


class FakeSession:
    def __init__(self, outputs: list[RollingLoglikelihoodOutput]) -> None:
        self.outputs = outputs

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        assert batch_size == 3
        request_items = list(requests)
        assert len(request_items) == 1
        assert request_items[0].text == "hello world"
        return list(self.outputs)


def test_paloma_scores_rolling_perplexity_for_one_subset(monkeypatch) -> None:
    dataset = Dataset.from_list([{"text": "hello world"}])
    monkeypatch.setattr(paloma_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.paloma_c4_en(max_rows=1, batch_size=3).evaluate(
        FakeSession([RollingLoglikelihoodOutput(logprob=-4.0, token_count=2)])
    )

    assert result.name == "paloma_c4_en"
    assert result.metadata == {
        "dataset_path": "allenai/paloma",
        "dataset_name": "c4_en",
        "split": "test",
        "stream": True,
        "scoring_mode": "rolling_loglikelihood_perplexity",
        "primary_metric": "word_perplexity",
        "subset": "c4_en",
    }
    assert result.metrics == {
        "word_perplexity": math.exp(2.0),
        "byte_perplexity": math.exp(4.0 / len("hello world".encode("utf-8"))),
        "bits_per_byte": 4.0 / len("hello world".encode("utf-8")) / math.log(2),
    }
    sample = result.samples[0]
    assert sample.metadata["subset"] == "c4_en"
    assert sample.metadata["text_preview"] == "hello world"
    assert sample.metadata["text_char_count"] == len("hello world")


def test_paloma_normalizes_subset_tokens_and_rejects_unknown_subset() -> None:
    suite = evalution.benchmarks.paloma(subset="dolma_v1_5")
    assert suite.dataset_name == "dolma-v1_5"
    assert suite.task_name() == "paloma_dolma_v1_5"

    with pytest.raises(ValueError, match="unsupported paloma subset"):
        evalution.benchmarks.paloma(subset="unknown_subset")
