# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations


CHOICE_LABELS = ("A", "B", "C", "D")


def slugify_subset_name(subset: str) -> str:
    slug = subset.lower()
    for old, new in (
        ("(", ""),
        (")", ""),
        ("/", " "),
        ("-", " "),
        ("&", " and "),
    ):
        slug = slug.replace(old, new)
    return "_".join(part for part in slug.replace(",", " ").split())


def subject_mmlu_prompt(
    *,
    benchmark_name: str,
    subject_native: str,
    question: str,
    choices: list[str],
    context: str | None = None,
) -> str:
    prompt_question = question.strip()
    if context and context.strip():
        prompt_question = f"{context.strip()}\n\n{prompt_question}"
    lines = [
        f"This is a {benchmark_name} multiple-choice question about {subject_native.strip()}.",
        "",
        f"Question: {prompt_question}",
    ]
    labels = CHOICE_LABELS[: len(choices)]
    lines.extend(
        f"{label}. {choice.strip()}"
        for label, choice in zip(labels, choices, strict=True)
    )
    lines.append("Answer:")
    return "\n".join(lines)
