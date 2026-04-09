# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

egyhellaswag_module = importlib.import_module("evalution.benchmarks.egyhellaswag")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == "شيل قرميد السقف: راجل قاعد على سقف. هو"
        assert requests[0].continuation == " بيستخدم شريط لاصق عشان يلف تلتين من التزلج."
        assert requests[3].continuation == " يبدأ يرفّع في السقف."
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.6, is_greedy=True, token_count=7),
        ]


def test_egyhellaswag_scores_raw_and_normalized_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ind": 24,
                "activity_label": "شيل قرميد السقف",
                "ctx": "راجل قاعد على سقف. هو",
                "endings": [
                    "بيستخدم شريط لاصق عشان يلف تلتين من التزلج.",
                    "بيشيل بلاط مكسّر من فوق.",
                    "بيمسك مكعب روبيك.",
                    "يبدأ يرفّع في السقف.",
                ],
                "source_id": "activitynet~v_-JhWjGDPHMY",
                "split": "val",
                "split_type": "indomain",
                "label": 3,
            }
        ]
    )
    monkeypatch.setattr(egyhellaswag_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.egyhellaswag(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "egyhellaswag"
    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
    }
    assert result.metadata["dataset_path"] == "UBC-NLP/EgyHellaSwag"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    sample = result.samples[0]
    assert sample.prompt == "شيل قرميد السقف: راجل قاعد على سقف. هو"
    assert sample.target == "يبدأ يرفّع في السقف."
    assert sample.prediction == "بيشيل بلاط مكسّر من فوق."
    assert sample.metadata["activity_label"] == "شيل قرميد السقف"
    assert sample.metadata["split_type"] == "indomain"


def test_egyhellaswag_label_prompt_lists_explicit_choices() -> None:
    suite = evalution.benchmarks.egyhellaswag()
    sample = evalution.benchmarks.MultipleChoiceSample(
        index=0,
        prompt="شيل قرميد السقف: راجل قاعد على سقف. هو",
        choices=[
            "بيستخدم شريط لاصق عشان يلف تلتين من التزلج.",
            "بيشيل بلاط مكسّر من فوق.",
            "بيمسك مكعب روبيك.",
            "يبدأ يرفّع في السقف.",
        ],
        gold_index=3,
        metadata={},
    )

    prompt = suite.label_prompt(
        sample,
        choice_order=(0, 1, 2, 3),
        labels=("A", "B", "C", "D"),
    )

    assert prompt.startswith("Context: شيل قرميد السقف: راجل قاعد على سقف. هو\n")
    assert "\nA. بيستخدم شريط لاصق عشان يلف تلتين من التزلج." in prompt
    assert prompt.endswith("\nAnswer:")
