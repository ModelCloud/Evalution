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

pile_10k_module = importlib.import_module("evalution.benchmarks.pile_10k")


class FakeSession:
    def __init__(self, outputs: list[RollingLoglikelihoodOutput]) -> None:
        self.outputs = outputs
        self.requests = []

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        assert batch_size == 3
        self.requests.extend(requests)
        return list(self.outputs)


def test_pile_10k_scores_weighted_perplexity_metrics(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "Hello world from The Pile",
                "meta": {"pile_set_name": "Pile-CC"},
            }
        ]
    )
    monkeypatch.setattr(pile_10k_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.pile_10k(max_rows=1, batch_size=3).evaluate(
        FakeSession([RollingLoglikelihoodOutput(logprob=-5.0, token_count=4)])
    )

    assert result.name == "pile_10k"
    assert result.metadata == {
        "dataset_path": "monology/pile-uncopyrighted",
        "dataset_name": None,
        "split": "train",
        "streaming": True,
        "scoring_mode": "rolling_loglikelihood_perplexity",
        "primary_metric": "word_perplexity",
    }
    assert result.metrics == {
        "word_perplexity": math.exp(5.0 / 5.0),
        "byte_perplexity": math.exp(5.0 / len("Hello world from The Pile".encode("utf-8"))),
        "bits_per_byte": 5.0 / len("Hello world from The Pile".encode("utf-8")) / math.log(2),
    }

    sample = result.samples[0]
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert sample.extracted == {
        "token_count": "4",
        "word_count": "5",
        "byte_count": str(len("Hello world from The Pile".encode("utf-8"))),
    }
    assert sample.metadata["text_preview"] == "Hello world from The Pile"
    assert sample.metadata["text_char_count"] == len("Hello world from The Pile")
    assert sample.metadata["pile_set_name"] == "Pile-CC"
    assert sample.metadata["logprob"] == -5.0
