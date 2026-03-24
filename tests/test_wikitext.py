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

wikitext_module = importlib.import_module("evalution.benchmarks.wikitext")


class FakeSession:
    def __init__(self, outputs: list[RollingLoglikelihoodOutput]) -> None:
        self.outputs = outputs
        self.requests = []

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        assert batch_size == 3
        self.requests.extend(requests)
        return list(self.outputs)


def test_wikitext_scores_weighted_perplexity_metrics(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "page": "Hello , world !",
            }
        ]
    )
    monkeypatch.setattr(wikitext_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.wikitext(max_rows=1, batch_size=3).evaluate(
        FakeSession([RollingLoglikelihoodOutput(logprob=-6.0, token_count=4)])
    )

    assert result.name == "wikitext"
    assert result.metadata == {
        "dataset_path": "EleutherAI/wikitext_document_level",
        "dataset_name": "wikitext-2-raw-v1",
        "split": "test",
        "stream": False,
        "scoring_mode": "rolling_loglikelihood_perplexity",
        "primary_metric": "word_perplexity",
    }
    assert result.metrics == {
        "word_perplexity": math.exp(6.0 / 4.0),
        "byte_perplexity": math.exp(6.0 / len("Hello , world !".encode("utf-8"))),
        "bits_per_byte": 6.0 / len("Hello , world !".encode("utf-8")) / math.log(2),
    }

    sample = result.samples[0]
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert sample.extracted == {
        "token_count": "4",
        "word_count": "4",
        "byte_count": str(len("Hello , world !".encode("utf-8"))),
    }
    assert sample.metadata["page_preview"] == "Hello , world !"
    assert sample.metadata["detokenized_preview"] == "Hello, world !"
    assert sample.metadata["page_char_count"] == len("Hello , world !")
    assert sample.metadata["logprob"] == -6.0


def test_wikitext_detokenizer_matches_upstream_rules() -> None:
    doc = {
        "page": " = = Heading = = \n Foo @-@ bar , baz . \" qux \" ( quux )",
    }

    assert wikitext_module._wikitext_detokenizer(doc) == " == Heading ==\nFoo-bar, baz. \"qux\" (quux)"
