# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

darijahellaswag_module = importlib.import_module("evalution.benchmarks.darijahellaswag")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == "قلع قرميد السطح: راجل گالس فوق السطح. هو"
        assert requests[0].continuation == " كايستعمل البلاستيك باش يلف جوج ديال الزلاجات."
        assert requests[3].continuation == " بدا كايقلع السقف ديال الدار."
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.6, is_greedy=True, token_count=7),
        ]


def test_darijahellaswag_scores_raw_and_normalized_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ind": 24,
                "activity_label": "قلع قرميد السطح",
                "ctx": "راجل گالس فوق السطح. هو",
                "endings": [
                    "كايستعمل البلاستيك باش يلف جوج ديال الزلاجات.",
                    "كايقلع القرميد من السطح.",
                    "كايشد مكعب روبيك فيدو.",
                    "بدا كايقلع السقف ديال الدار.",
                ],
                "source_id": "activitynet~v_-JhWjGDPHMY",
                "split": "val",
                "split_type": "indomain",
                "label": "3",
            }
        ]
    )
    monkeypatch.setattr(darijahellaswag_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.darijahellaswag(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "darijahellaswag"
    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
    }
    assert result.metadata["dataset_path"] == "MBZUAI-Paris/DarijaHellaSwag"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    sample = result.samples[0]
    assert sample.prompt == "قلع قرميد السطح: راجل گالس فوق السطح. هو"
    assert sample.target == "بدا كايقلع السقف ديال الدار."
    assert sample.prediction == "كايقلع القرميد من السطح."
    assert sample.metadata["activity_label"] == "قلع قرميد السطح"
    assert sample.metadata["split_type"] == "indomain"


def test_darijahellaswag_label_prompt_lists_explicit_choices() -> None:
    suite = evalution.benchmarks.darijahellaswag()
    sample = evalution.benchmarks.MultipleChoiceSample(
        index=0,
        prompt="قلع قرميد السطح: راجل گالس فوق السطح. هو",
        choices=[
            "كايستعمل البلاستيك باش يلف جوج ديال الزلاجات.",
            "كايقلع القرميد من السطح.",
            "كايشد مكعب روبيك فيدو.",
            "بدا كايقلع السقف ديال الدار.",
        ],
        gold_index=3,
        metadata={},
    )

    prompt = suite.label_prompt(
        sample,
        choice_order=(0, 1, 2, 3),
        labels=("A", "B", "C", "D"),
    )

    assert prompt.startswith("Context: قلع قرميد السطح: راجل گالس فوق السطح. هو\n")
    assert "\nA. كايستعمل البلاستيك باش يلف جوج ديال الزلاجات." in prompt
    assert prompt.endswith("\nAnswer:")
