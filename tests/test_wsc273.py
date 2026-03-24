# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

wsc273_module = importlib.import_module("evalution.benchmarks.wsc273")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 2
        assert requests[0].context == "The city councilmen refused the demonstrators a permit because the city councilmen"
        assert requests[0].continuation == " feared violence."
        assert requests[1].context == "The city councilmen refused the demonstrators a permit because the demonstrators"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=3),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=3),
        ]


def test_wsc273_scores_partial_evaluation_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
                {
                    "text": "The city councilmen refused the demonstrators a permit because they feared violence.",
                    "pronoun": "they",
                    "pronoun_loc": 63,
                    "options": ["the city councilmen", "the demonstrators"],
                    "label": 0,
                    "source": "Levesque",
                }
        ]
    )
    monkeypatch.setattr(wsc273_module, "_load_wsc273_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.wsc273(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "wsc273"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "winograd_wsc"
    assert result.metadata["dataset_name"] == "wsc273"
    assert result.metadata["split"] == "test"
    assert result.metadata["prompt_variant"] == "partial_evaluation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "The city councilmen refused the demonstrators a permit because they feared violence."
    assert sample.target == "The city councilmen refused the demonstrators a permit because the city councilmen feared violence."
    assert sample.prediction == sample.target
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["choice_labels"] == ["A", "B"]
    assert sample.metadata["choice_texts"] == ["the city councilmen", "the demonstrators"]


def test_wsc273_normalizes_possessive_options_like_upstream() -> None:
    doc = {
        "text": "Anna did not pass the ball to Maria although her hands were free.",
        "pronoun": "her",
        "pronoun_loc": 38,
    }

    assert wsc273_module._normalize_wsc273_option(doc, "Anna") == "Anna's"


def test_wsc273_loader_reads_xml_subset(tmp_path, monkeypatch) -> None:
    xml_path = tmp_path / "WSCollection.xml"
    xml_path.write_text(
        """<collection>
<schema>
  <text>
    <txt1>The city councilmen refused the demonstrators a permit because</txt1>
    <pron>they</pron>
    <txt2>feared violence.</txt2>
  </text>
  <answers>
    <answer>the city councilmen</answer>
    <answer>the demonstrators</answer>
  </answers>
  <correctAnswer>A</correctAnswer>
  <source>Levesque</source>
</schema>
</collection>""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        wsc273_module,
        "_ensure_wsc273_xml",
        lambda *, cache_dir: xml_path,
    )

    dataset = wsc273_module._load_wsc273_dataset(
        "winograd_wsc",
        "wsc273",
        split="test",
        cache_dir=None,
    )

    assert len(dataset) == 1
    assert dataset[0]["text"] == "The city councilmen refused the demonstrators a permit because they feared violence."
    assert dataset[0]["options"] == ["the city councilmen", "the demonstrators"]
    assert dataset[0]["label"] == 0
