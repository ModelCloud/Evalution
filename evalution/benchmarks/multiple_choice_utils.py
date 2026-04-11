# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations


def choice_index_from_labels(labels: list[str], answer_key: str) -> int:
    # Resolve a dataset-provided answer label into the zero-based index used by the multiple-choice helper.
    """Implement choice index from labels for this module."""
    return labels.index(answer_key.strip())


def question_answer_prompt(question: str) -> str:
    # Build the standard question-and-answer prompt header used by several benchmark families.
    """Implement question answer prompt for this module."""
    return f"Question: {question.strip()}\nAnswer:"
