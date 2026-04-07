# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pcre


CHOICE_LABELS = ("A", "B", "C")
_CAMEL_BOUNDARY_RE = pcre.compile(r"(?<!^)(?=[A-Z])")
_NON_ALNUM_RE = pcre.compile(r"[^a-z0-9]+")


def slugify_config_name(name: str) -> str:
    slug = _CAMEL_BOUNDARY_RE.sub("_", name).lower()
    return _NON_ALNUM_RE.sub("_", slug).strip("_")


def bbq_prompt(context: str, question: str, choices: list[str]) -> str:
    lines = [f"Context: {context.strip()}", f"Question: {question.strip()}"]
    lines.extend(
        f"{label}. {choice.strip()}"
        for label, choice in zip(CHOICE_LABELS, choices, strict=True)
    )
    lines.append("Answer:")
    return "\n".join(lines)
