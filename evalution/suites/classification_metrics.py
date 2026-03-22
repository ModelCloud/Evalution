# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from math import sqrt


def f1_for_label(
    gold_labels: list[int],
    predicted_labels: list[int],
    *,
    label: int,
) -> float:
    # Compute the one-vs-rest F1 score for a single label so binary and multiclass suites can share the same math.
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for gold_label, predicted_label in zip(gold_labels, predicted_labels, strict=True):
        if gold_label == label and predicted_label == label:
            true_positive += 1
        elif gold_label != label and predicted_label == label:
            false_positive += 1
        elif gold_label == label and predicted_label != label:
            false_negative += 1

    if true_positive == 0:
        return 0.0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return (2.0 * precision * recall) / (precision + recall)


def macro_f1(
    gold_labels: list[int],
    predicted_labels: list[int],
    *,
    labels: list[int],
) -> float:
    # Average the per-label F1 scores so tasks like CB can report the same macro-style metric as upstream.
    if not labels:
        return 0.0
    total = 0.0
    for label in labels:
        total += f1_for_label(gold_labels, predicted_labels, label=label)
    return total / len(labels)


def matthews_corrcoef(
    gold_labels: list[int],
    predicted_labels: list[int],
    *,
    positive_label: int = 1,
) -> float:
    # Compute binary Matthews correlation so GLUE tasks like CoLA can report the benchmark's primary metric.
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for gold_label, predicted_label in zip(gold_labels, predicted_labels, strict=True):
        gold_is_positive = gold_label == positive_label
        predicted_is_positive = predicted_label == positive_label
        if gold_is_positive and predicted_is_positive:
            true_positive += 1
        elif not gold_is_positive and not predicted_is_positive:
            true_negative += 1
        elif not gold_is_positive and predicted_is_positive:
            false_positive += 1
        else:
            false_negative += 1

    denominator = sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0.0:
        return 0.0

    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    return numerator / denominator
