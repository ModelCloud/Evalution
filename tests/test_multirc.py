# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import pcre

from evalution.benchmarks.multirc import (
    MultiRC,
    _extract_indices,
    _group_questions,
    _precision_recall_f1,
)
from evalution.engines.base import GenerationOutput


def test_grouping_builds_questions():
    raw = [
        {
            "paragraph": "para",
            "question": "q?",
            "answer": "a0",
            "idx": {"paragraph": 0, "question": 0, "answer": 0},
            "label": 1,
        },
        {
            "paragraph": "para",
            "question": "q?",
            "answer": "a1",
            "idx": {"paragraph": 0, "question": 0, "answer": 1},
            "label": 0,
        },
    ]
    grouped = _group_questions(raw)
    assert len(grouped) == 1
    q = grouped[0]
    assert q["paragraph_idx"] == 0
    assert q["question_idx"] == 0
    assert len(q["answers"]) == 2


def test_extract_indices_handles_none_and_bounds():
    assert _extract_indices("none", 3) == set()
    assert _extract_indices("0,2", 3) == {0, 2}
    assert _extract_indices("5,1", 2) == {1}  # drops out-of-range


def test_precision_recall_f1_basic():
    p, r, f1 = _precision_recall_f1({0, 1}, {0, 2})
    assert round(p, 3) == 0.5
    assert round(r, 3) == 0.5
    assert round(f1, 3) == 0.5


def test_score_sample_exact_match_and_f1():
    suite = MultiRC()
    raw_rows = [
        {
            "paragraph": "p",
            "question": "q",
            "answer": "a0",
            "idx": {"paragraph": 0, "question": 0, "answer": 0},
            "label": 1,
        },
        {
            "paragraph": "p",
            "question": "q",
            "answer": "a1",
            "idx": {"paragraph": 0, "question": 0, "answer": 1},
            "label": 0,
        },
        {
            "paragraph": "p",
            "question": "q",
            "answer": "a2",
            "idx": {"paragraph": 0, "question": 0, "answer": 2},
            "label": 1,
        },
    ]
    prepared = next(suite.iter_prepared_samples(raw_rows))
    output = GenerationOutput(text="0,2", prompt="p", metadata={})
    scored = suite.score_sample(prepared, output)
    assert scored.scores["em"] == 1.0
    assert scored.scores["f1a"] == 1.0
