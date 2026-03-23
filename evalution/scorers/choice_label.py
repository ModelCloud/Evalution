# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations


def exact_match(prediction: str, target: str) -> float:
    return float(prediction == target)


def choice_label_exact_match(predicted_label: str, gold_label: str) -> float:
    return exact_match(predicted_label, gold_label)
