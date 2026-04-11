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

# Keep shared test fixtures and expectations explicit at module scope.
belebele_module = importlib.import_module("evalution.benchmarks.belebele")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "P: Make sure your hand is as relaxed as possible while still hitting all the notes correctly.\n"
            "Q: According to the passage, what would not be considered an accurate tip for successfully playing the accordion?\n"
            "A: For additional volume, increase the force with which you hit the keys\n"
            "B: Keep unnecessary movement to a minimum in order to preserve your stamina\n"
            "C: Be mindful of hitting the notes while maintaining a relaxed hand\n"
            "D: Increase the speed with which you operate the bellows to achieve extra volume\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
        ]


def test_belebele_scores_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify belebele scores multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "link": "https://example.com/passage",
                "question_number": 1,
                "flores_passage": "Make sure your hand is as relaxed as possible while still hitting all the notes correctly.",
                "question": "According to the passage, what would not be considered an accurate tip for successfully playing the accordion?",
                "mc_answer1": "For additional volume, increase the force with which you hit the keys",
                "mc_answer2": "Keep unnecessary movement to a minimum in order to preserve your stamina",
                "mc_answer3": "Be mindful of hitting the notes while maintaining a relaxed hand",
                "mc_answer4": "Increase the speed with which you operate the bellows to achieve extra volume",
                "correct_answer_num": "1",
                "dialect": "eng_Latn",
            }
        ]
    )
    monkeypatch.setattr(belebele_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.belebele(language="eng_Latn", max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "belebele_eng_Latn"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "facebook/belebele"
    assert result.metadata["dataset_name"] == "eng_Latn"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["dialect"] == "eng_Latn"
    assert sample.metadata["correct_answer_num"] == "1"
    assert len(sample.metadata["raw_choices"]) == 4


def test_belebele_prompt_helper_formats_prompt() -> None:
    """Verify belebele prompt helper formats prompt."""
    assert (
        belebele_module._belebele_prompt(
            {
                "flores_passage": "Passage text",
                "question": "Question text",
                "mc_answer1": "Choice 1",
                "mc_answer2": "Choice 2",
                "mc_answer3": "Choice 3",
                "mc_answer4": "Choice 4",
            }
        )
        == "P: Passage text\nQ: Question text\nA: Choice 1\nB: Choice 2\nC: Choice 3\nD: Choice 4\nAnswer:"
    )


def test_belebele_rejects_empty_language() -> None:
    """Verify belebele rejects empty language."""
    with pytest.raises(ValueError, match="non-empty dataset config"):
        evalution.benchmarks.belebele(language="")


def test_belebele_rejects_dataset_name_mismatch() -> None:
    """Verify belebele rejects dataset name mismatch."""
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.belebele(language="eng_Latn", dataset_name="fra_Latn")


def test_belebele_language_aliases_bind_expected_dataset_configs() -> None:
    """Verify belebele language aliases bind expected dataset configs."""
    portuguese = evalution.benchmarks.belebele_por_Latn(max_rows=1)
    spanish = evalution.benchmarks.belebele_spa_Latn(max_rows=1)

    assert portuguese.task_name() == "belebele_por_Latn"
    assert portuguese.language == "por_Latn"
    assert portuguese.dataset_name == "por_Latn"
    assert spanish.task_name() == "belebele_spa_Latn"
    assert spanish.language == "spa_Latn"
    assert spanish.dataset_name == "spa_Latn"
