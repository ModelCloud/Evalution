# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import permutations
from math import ceil
from typing import Iterable, Sequence

# Keep scorer defaults and parser helpers explicit at module scope.
_CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True, slots=True)
class ChoiceScore:
    """Define the choice score helper class."""
    # Keep the class-level state explicit for this helper.
    index: int
    logprob: float
    logprob_norm: float


@dataclass(frozen=True, slots=True)
class MultipleChoiceOutcome:
    """Define the multiple choice outcome helper class."""
    # Keep the class-level state explicit for this helper.
    raw_best_index: int
    normalized_best_index: int
    raw_accuracy: float
    normalized_accuracy: float


@dataclass(frozen=True, slots=True)
class ExamScoreOutcome:
    """Define the exam score outcome helper class."""
    # Keep the class-level state explicit for this helper.
    selected_indices: tuple[int, ...]
    exam_score: float


@dataclass(frozen=True, slots=True)
class LabelPermutationOutcome:
    """Define the label permutation outcome helper class."""
    # Keep the class-level state explicit for this helper.
    predicted_index: int
    accuracy: float
    averaged_choice_logprobs: tuple[float, ...]
    permutation_count: int


def normalized_logprob(logprob: float, token_count: int) -> float:
    """Implement normalized logprob for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return logprob / max(token_count, 1)


def build_choice_score(
    *,
    choice_index: int,
    logprob: float,
    token_count: int,
) -> ChoiceScore:
    """Build choice score. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return ChoiceScore(
        index=choice_index,
        logprob=logprob,
        logprob_norm=normalized_logprob(logprob, token_count),
    )


def build_choice_scores(
    rows: Iterable[tuple[int, float, int]],
) -> list[ChoiceScore]:
    """Build choice scores. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return sorted(
        (
            build_choice_score(
                choice_index=choice_index,
                logprob=logprob,
                token_count=token_count,
            )
            for choice_index, logprob, token_count in rows
        ),
        key=lambda item: item.index,
    )


def choice_logprobs(choice_scores: Sequence[ChoiceScore]) -> list[float]:
    """Implement choice logprobs for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return [score.logprob for score in _sorted_choice_scores(choice_scores)]


def choice_logprobs_norm(choice_scores: Sequence[ChoiceScore]) -> list[float]:
    """Implement choice logprobs norm for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return [score.logprob_norm for score in _sorted_choice_scores(choice_scores)]


def multiple_choice_outcome(
    choice_scores: Sequence[ChoiceScore],
    gold_index: int,
) -> MultipleChoiceOutcome:
    """Implement multiple choice outcome for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    ordered_choice_scores = _sorted_choice_scores(choice_scores)
    raw_best_index = max(ordered_choice_scores, key=lambda item: item.logprob).index
    normalized_best_index = max(
        ordered_choice_scores,
        key=lambda item: item.logprob_norm,
    ).index
    return MultipleChoiceOutcome(
        raw_best_index=raw_best_index,
        normalized_best_index=normalized_best_index,
        raw_accuracy=float(raw_best_index == gold_index),
        normalized_accuracy=float(normalized_best_index == gold_index),
    )


def exam_score_outcome(
    choice_scores: Sequence[ChoiceScore],
    gold_index: int,
) -> ExamScoreOutcome:
    """Implement exam score outcome for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    ordered_choice_scores = _sorted_choice_scores(choice_scores)
    max_choice_score = max(score.logprob for score in ordered_choice_scores)
    selected_indices = tuple(
        score.index
        for score in ordered_choice_scores
        if score.logprob == max_choice_score
    )
    exam_score = 1.0 / len(selected_indices) if gold_index in selected_indices else 0.0
    return ExamScoreOutcome(
        selected_indices=selected_indices,
        exam_score=exam_score,
    )


def choice_labels(choice_count: int) -> tuple[str, ...]:
    """Implement choice labels for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if choice_count < 1:
        raise ValueError("at least one choice is required")
    if choice_count > len(_CHOICE_LABELS):
        raise ValueError(f"choice label scoring supports at most {len(_CHOICE_LABELS)} choices")
    return _CHOICE_LABELS[:choice_count]


def normalize_label_permutation_fraction(fraction: float | int | str | None) -> float:
    """Normalize label permutation fraction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if fraction is None:
        return 0.0

    if isinstance(fraction, str):
        stripped = fraction.strip()
        if not stripped:
            return 0.0
        fraction = stripped

    try:
        canonical_fraction = float(Decimal(str(fraction)))
    except (InvalidOperation, ValueError):
        raise ValueError("label_permutations must be a float in [0.0, 1.0]") from None

    if canonical_fraction < 0.0 or canonical_fraction > 1.0:
        raise ValueError("label_permutations must be a float in [0.0, 1.0]")
    return canonical_fraction


def format_label_permutation_fraction(fraction: float | int | str | None) -> str:
    """Format label permutation fraction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    canonical_fraction = normalize_label_permutation_fraction(fraction)
    decimal_fraction = Decimal(str(canonical_fraction)).normalize()
    fraction_text = format(decimal_fraction, "f")
    if "." in fraction_text:
        fraction_text = fraction_text.rstrip("0").rstrip(".")
    if "." not in fraction_text:
        fraction_text = f"{fraction_text}.0"
    return fraction_text


def label_permutation_metric_name(fraction: float | int | str | None) -> str:
    """Implement label permutation metric name for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    canonical_fraction = normalize_label_permutation_fraction(fraction)
    if canonical_fraction == 0.0:
        raise ValueError("label_permutations=0.0 disables the extra label-permutation metric")
    return f"acc,label_perm:{format_label_permutation_fraction(canonical_fraction)}"


def label_permutations_for_mode(
    choice_count: int,
    fraction: float | int | str | None,
) -> list[tuple[int, ...]]:
    """Implement label permutations for mode for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    canonical_fraction = normalize_label_permutation_fraction(fraction)
    if canonical_fraction == 0.0:
        return []

    all_permutations = [tuple(permutation) for permutation in permutations(range(choice_count))]
    if canonical_fraction == 1.0:
        return all_permutations

    # Use at least one full cycle over label positions so nonzero fractions still average label priors
    # instead of collapsing to a single fixed labeling on small-choice tasks like binary MC.
    target_count = min(
        len(all_permutations),
        max(choice_count, ceil(len(all_permutations) * canonical_fraction)),
    )
    return _balanced_permutation_subset(
        all_permutations,
        choice_count=choice_count,
        target_count=target_count,
    )


def label_permutation_outcome(
    *,
    permutations: Sequence[tuple[int, ...]],
    permutation_label_logprobs: Sequence[Sequence[float]],
    gold_index: int,
) -> LabelPermutationOutcome:
    """Implement label permutation outcome for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    if not permutations:
        raise ValueError("at least one permutation is required")
    if len(permutations) != len(permutation_label_logprobs):
        raise ValueError("permutation count and score count must match")

    choice_count = len(permutations[0])
    aggregate_choice_scores = [0.0] * choice_count
    for permutation, label_logprobs in zip(permutations, permutation_label_logprobs, strict=True):
        if len(permutation) != choice_count:
            raise ValueError("all permutations must have the same length")
        if len(label_logprobs) != choice_count:
            raise ValueError("each permutation must provide one score per label")
        for label_index, original_choice_index in enumerate(permutation):
            aggregate_choice_scores[original_choice_index] += label_logprobs[label_index]

    averaged_choice_scores = tuple(
        total_score / len(permutations)
        for total_score in aggregate_choice_scores
    )
    predicted_index = max(range(choice_count), key=averaged_choice_scores.__getitem__)
    return LabelPermutationOutcome(
        predicted_index=predicted_index,
        accuracy=float(predicted_index == gold_index),
        averaged_choice_logprobs=averaged_choice_scores,
        permutation_count=len(permutations),
    )


def _sorted_choice_scores(choice_scores: Sequence[ChoiceScore]) -> list[ChoiceScore]:
    """Implement sorted choice scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    ordered_choice_scores = sorted(choice_scores, key=lambda item: item.index)
    if not ordered_choice_scores:
        raise ValueError("at least one choice score is required")
    return ordered_choice_scores


def _balanced_permutation_subset(
    all_permutations: list[tuple[int, ...]],
    *,
    choice_count: int,
    target_count: int,
) -> list[tuple[int, ...]]:
    """Implement balanced permutation subset for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    if target_count >= len(all_permutations):
        return all_permutations

    selected: list[tuple[int, ...]] = []
    remaining = list(all_permutations)
    pair_counts = [[0] * choice_count for _ in range(choice_count)]
    while len(selected) < target_count:
        next_target = (len(selected) + 1) / choice_count
        best_permutation = min(
            remaining,
            key=lambda permutation: _permutation_penalty(
                permutation,
                pair_counts=pair_counts,
                target=next_target,
            ),
        )
        selected.append(best_permutation)
        remaining.remove(best_permutation)
        for label_index, original_choice_index in enumerate(best_permutation):
            pair_counts[original_choice_index][label_index] += 1
    return selected


def _permutation_penalty(
    permutation: tuple[int, ...],
    *,
    pair_counts: list[list[int]],
    target: float,
) -> float:
    """Implement permutation penalty for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    penalty = 0.0
    for original_choice_index, label_counts in enumerate(pair_counts):
        for label_index, count in enumerate(label_counts):
            new_count = count + int(permutation[label_index] == original_choice_index)
            penalty += (new_count - target) ** 2
    return penalty
