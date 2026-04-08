# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from evalution.scorers.choice_label import choice_label_exact_match
from evalution.scorers.math_exact_match import (
    extract_math_answer,
    math_exact_match,
    normalize_math_string,
)
from evalution.scorers.multiple_choice import (
    build_choice_scores,
    exam_score_outcome,
    format_label_permutation_fraction,
    label_permutation_metric_name,
    label_permutations_for_mode,
    multiple_choice_outcome,
)


def test_multiple_choice_outcome_tracks_raw_and_normalized_winners() -> None:
    choice_scores = build_choice_scores(
        [
            (0, -1.0, 1),
            (1, -2.0, 1),
            (2, -3.0, 4),
        ]
    )

    outcome = multiple_choice_outcome(choice_scores, gold_index=2)

    assert outcome.raw_best_index == 0
    assert outcome.normalized_best_index == 2
    assert outcome.raw_accuracy == 0.0
    assert outcome.normalized_accuracy == 1.0


def test_exam_score_outcome_awards_partial_credit_on_tie() -> None:
    choice_scores = build_choice_scores(
        [
            (0, -0.5, 1),
            (1, -0.5, 1),
            (2, -1.0, 1),
        ]
    )

    outcome = exam_score_outcome(choice_scores, gold_index=1)

    assert outcome.selected_indices == (0, 1)
    assert outcome.exam_score == 0.5


def test_choice_label_exact_match_is_strict() -> None:
    assert choice_label_exact_match("B", "B") == 1.0
    assert choice_label_exact_match("b", "B") == 0.0


def test_label_permutation_fraction_uses_balanced_minimum() -> None:
    assert format_label_permutation_fraction(0.25) == "0.25"
    assert format_label_permutation_fraction(1.0) == "1.0"
    assert label_permutation_metric_name(0.25) == "acc,label_perm:0.25"
    assert label_permutation_metric_name(0.75) == "acc,label_perm:0.75"
    assert len(label_permutations_for_mode(2, 0.25)) == 2
    assert len(label_permutations_for_mode(4, 0.25)) == 6
    assert len(label_permutations_for_mode(4, 0.5)) == 12
    assert len(label_permutations_for_mode(4, 0.75)) == 18
    assert len(label_permutations_for_mode(4, 1.0)) == 24


def test_math_exact_match_handles_boxed_and_fraction_normalization() -> None:
    assert extract_math_answer("Reasoning... \\boxed{033}") == "033"
    assert normalize_math_string("0.5") == "\\frac{1}{2}"
    assert math_exact_match("Reasoning... \\boxed{033}", "33") == 0.0
    assert math_exact_match("Reasoning... \\boxed{\\frac12}", "\\frac{1}{2}") == 1.0


def test_math_exact_match_extracts_simple_dollar_delimited_answer() -> None:
    assert extract_math_answer("Final answer is $42$") == "42"


def test_math_normalizer_strips_text_units_without_asserting() -> None:
    assert normalize_math_string("12\\text{ cm}\\text{ squared}") == "12"
