# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
storycloze_module = importlib.import_module("evalution.benchmarks.storycloze")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == "One. Two. Three. Four."
        assert requests[0].continuation == " Bad ending."
        assert requests[1].continuation == " Good ending."
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=2),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=2),
        ]


def test_storycloze_scores_year_specific_multiple_choice_rows(monkeypatch) -> None:
    """Verify storycloze scores year specific multiple choice rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "story_id": "story-1",
                "input_sentence_1": "One.",
                "input_sentence_2": "Two.",
                "input_sentence_3": "Three.",
                "input_sentence_4": "Four.",
                "sentence_quiz1": "Bad ending.",
                "sentence_quiz2": "Good ending.",
                "answer_right_ending": 2,
            }
        ]
    )
    monkeypatch.setattr(storycloze_module, "_load_storycloze_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.storycloze_2016(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "storycloze_2016"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "LSDSem/story_cloze",
        "dataset_name": "2016",
        "split": "validation",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "Good ending."
    assert sample.prediction == "Good ending."
    assert sample.metadata["year"] == "2016"
    assert sample.metadata["story_id"] == "story-1"
    assert len(sample.metadata["input_sentences"]) == 4
    assert sample.metadata["choice_texts"] == ["Bad ending.", "Good ending."]


def test_storycloze_prompt_builder_and_year_validation() -> None:
    """Verify storycloze prompt builder and year validation."""
    assert storycloze_module._storycloze_prompt(
        {
            "input_sentence_1": "One.",
            "input_sentence_2": "Two.",
            "input_sentence_3": "Three.",
            "input_sentence_4": "Four.",
        }
    ) == "One. Two. Three. Four."

    with pytest.raises(ValueError, match="unsupported storycloze year"):
        evalution.benchmarks.storycloze(year="2014")

    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.storycloze(year="2016", dataset_name="2018")


def test_storycloze_loader_reads_csv_without_dataset_script(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "cloze_test_val__spring2016 - cloze_test_ALL_val.csv"
    csv_path.write_text(
        "\n".join(
            [
                "InputStoryid,InputSentence1,InputSentence2,InputSentence3,InputSentence4,RandomFifthSentenceQuiz1,RandomFifthSentenceQuiz2,AnswerRightEnding",
                '"story-1","One.","Two.","Three.","Four.","Bad ending.","Good ending.","2"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        storycloze_module,
        "_storycloze_csv_path",
        lambda dataset_name, *, split, cache_dir=None: str(csv_path),
    )

    dataset = storycloze_module._load_storycloze_dataset(
        "LSDSem/story_cloze",
        "2016",
        split="validation",
    )

    assert list(dataset) == [
        {
            "story_id": "story-1",
            "input_sentence_1": "One.",
            "input_sentence_2": "Two.",
            "input_sentence_3": "Three.",
            "input_sentence_4": "Four.",
            "sentence_quiz1": "Bad ending.",
            "sentence_quiz2": "Good ending.",
            "answer_right_ending": 2,
        }
    ]
