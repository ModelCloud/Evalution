# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
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
    """Verify multiple choice outcome tracks raw and normalized winners."""
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
    """Verify exam score outcome awards partial credit on tie. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
    """Verify choice label exact match is strict. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert choice_label_exact_match("B", "B") == 1.0
    assert choice_label_exact_match("b", "B") == 0.0


def test_label_permutation_fraction_uses_balanced_minimum() -> None:
    """Verify label permutation fraction uses balanced minimum."""
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
    """Verify math exact match handles boxed and fraction normalization. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert extract_math_answer("Reasoning... \\boxed{033}") == "033"
    assert normalize_math_string("0.5") == "\\frac{1}{2}"
    assert math_exact_match("Reasoning... \\boxed{033}", "33") == 0.0
    assert math_exact_match("Reasoning... \\boxed{\\frac12}", "\\frac{1}{2}") == 1.0


def test_math_exact_match_extracts_simple_dollar_delimited_answer() -> None:
    """Verify math exact match extracts simple dollar delimited answer. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert extract_math_answer("Final answer is $42$") == "42"


def test_math_exact_match_extracts_explicit_unboxed_final_answer() -> None:
    """Verify math exact match extracts explicit unboxed answers and normalizes gold formatting."""
    prediction = "Work...\nFinal answer: \\frac{1}{576}."
    target = "$\\frac{1}{576}$."

    assert extract_math_answer(prediction) == "\\frac{1}{576}"
    assert math_exact_match(prediction, target) == 1.0


def test_math_exact_match_uses_the_last_inline_math_span() -> None:
    """Verify math exact match prefers the final inline math span instead of spanning the whole response."""
    assert extract_math_answer("First $x$ then final $42$") == "42"


def test_math_normalizer_strips_text_units_without_asserting() -> None:
    """Verify math normalizer strips text units without asserting."""
    assert normalize_math_string("12\\text{ cm}\\text{ squared}") == "12"


def test_math_normalizer_strips_latex_wrappers_and_terminal_punctuation() -> None:
    """Verify math normalizer removes formatting-only LaTeX delimiters and sentence punctuation."""
    assert normalize_math_string("$\\left\\lfloor x \\right\\rfloor + 1$.") == "\\lfloorx\\rfloor+1"
    assert normalize_math_string("\\(-((d - 2k)^2) + d\\)") == "-((d-2k)^2)+d"
