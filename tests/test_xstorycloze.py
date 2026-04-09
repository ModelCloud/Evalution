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

xstorycloze_module = importlib.import_module("evalution.benchmarks.xstorycloze")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == (
            "I became a Law and Order fan in 2011. "
            "I was recovering from a stroke. "
            "When I got home I tried to watch every episode. "
            "It was hard trying to binge watch 20 Year's of a show."
        )
        assert requests[0].continuation == " I think Law and Order is one of the worst shows ever made."
        assert requests[1].continuation == " Eventually I watched them all."
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=11),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=5),
        ]


def test_xstorycloze_scores_multilingual_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "story_id": "story-1",
                "input_sentence_1": "I became a Law and Order fan in 2011.",
                "input_sentence_2": "I was recovering from a stroke.",
                "input_sentence_3": "When I got home I tried to watch every episode.",
                "input_sentence_4": "It was hard trying to binge watch 20 Year's of a show.",
                "sentence_quiz1": "I think Law and Order is one of the worst shows ever made.",
                "sentence_quiz2": "Eventually I watched them all.",
                "answer_right_ending": 2,
            }
        ]
    )
    monkeypatch.setattr(xstorycloze_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xstorycloze_en(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "xstorycloze_en"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "juletxara/xstory_cloze"
    assert result.metadata["dataset_name"] == "en"
    assert result.metadata["split"] == "eval"
    assert result.samples[0].metadata["language"] == "en"
    assert len(result.samples[0].metadata["input_sentences"]) == 4
    assert result.samples[0].metadata["choice_texts"][1] == "Eventually I watched them all."


def test_xstorycloze_prompt_builder_and_language_validation() -> None:
    prompt = xstorycloze_module._xstorycloze_prompt(
        {
            "input_sentence_1": "One.",
            "input_sentence_2": "Two.",
            "input_sentence_3": "Three.",
            "input_sentence_4": "Four.",
        }
    )
    assert prompt == "One. Two. Three. Four."

    with pytest.raises(ValueError, match="unsupported xstorycloze language"):
        xstorycloze_module.XStoryCloze(language="de")
