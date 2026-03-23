# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import math

from datasets import Dataset

import evalution
from evalution.engines.base import RollingLoglikelihoodOutput

c4_module = importlib.import_module("evalution.benchmarks.c4")


class FakeSession:
    def __init__(self, outputs: list[RollingLoglikelihoodOutput]) -> None:
        self.outputs = outputs
        self.requests = []

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        assert batch_size == 3
        self.requests.extend(requests)
        return list(self.outputs)


def test_c4_scores_weighted_perplexity_metrics(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "Hello world from C4",
                "url": "https://example.com/doc",
                "timestamp": "2024-01-01 00:00:00",
            }
        ]
    )
    monkeypatch.setattr(c4_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.c4(max_rows=1, batch_size=3).evaluate(
        FakeSession([RollingLoglikelihoodOutput(logprob=-4.0, token_count=3)])
    )

    assert result.name == "c4"
    assert result.metadata == {
        "dataset_path": "allenai/c4",
        "dataset_name": "en",
        "split": "validation",
        "streaming": True,
        "scoring_mode": "rolling_loglikelihood_perplexity",
        "primary_metric": "word_perplexity",
    }
    assert result.metrics == {
        "word_perplexity": math.exp(4.0 / 4.0),
        "byte_perplexity": math.exp(4.0 / len("Hello world from C4".encode("utf-8"))),
        "bits_per_byte": 4.0 / len("Hello world from C4".encode("utf-8")) / math.log(2),
    }

    sample = result.samples[0]
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert sample.extracted == {
        "token_count": "3",
        "word_count": "4",
        "byte_count": str(len("Hello world from C4".encode("utf-8"))),
    }
    assert sample.metadata["text_preview"] == "Hello world from C4"
    assert sample.metadata["text_char_count"] == len("Hello world from C4")
    assert sample.metadata["url"] == "https://example.com/doc"
    assert sample.metadata["timestamp"] == "2024-01-01 00:00:00"
    assert sample.metadata["logprob"] == -4.0
