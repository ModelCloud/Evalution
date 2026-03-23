# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import pcre

INVALID_ANSWER = "[invalid]"

# OpenAI's released GSM8K scorer in grade_school_math/dataset.py only accepts
# answers emitted in the `#### number` format. We keep a faithful copy here for
# regression tests against the benchmark-owner parser, but the live suite score
# uses the format-insensitive numeric matcher below.
_GSM8K_REFERENCE_ANSWER_RE = pcre.compile(r"#### (\-?[0-9\.\,]+)")

# MadryLab's GSM8K-Platinum scorer in src/utils.py recovers boxed answers and
# then parses a numeric answer out of the answer section or last line. This is
# also retained for regression tests rather than the live suite score.
_GSM8K_PLATINUM_BOXED_RE = pcre.compile(r"\\boxed\{(?:\\text\{)?([^\\{}]+)\}")

# Format-insensitive numeric extraction used by the live scorer. This keeps
# chat-template wrappers like `The answer is 42.` from zeroing out a correct
# answer purely because it was not rendered as `#### 42`.
_HASH_ANSWER_RE = pcre.compile(r"####\s*(-?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+))")
_ANSWER_LINE_RE = pcre.compile(
    r"(?:the\s+final\s+answer\s+is|final\s+answer\s*:|the\s+answer\s+is|answer\s*:)\s*(.+)",
    pcre.IGNORECASE,
)
_NUMERIC_TOKEN_RE = pcre.compile(r"-?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)")
_PLATINUM_REFERENCE_NUMBER_RE = pcre.compile(r"-?[0-9.]*[0-9]")
_TRAILING_ZERO_RE = pcre.compile(r"\.0+$")


def extract_gsm8k_reference_answer(text: str) -> str:
    match = _GSM8K_REFERENCE_ANSWER_RE.search(text or "")
    if match is None:
        return INVALID_ANSWER
    return match.group(1).strip().replace(",", "")


def extract_gsm8k_platinum_reference_answer(text: str) -> str:
    output = text or ""
    lowered_without_stars = output.lower().replace("*", "")
    boxed_match = None

    if "answer:" in lowered_without_stars:
        answer_section = output.lower().split("answer: ")[-1]
        boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(answer_section)
        if boxed_match is not None:
            output = f"Answer: {boxed_match.group(1)}"
    else:
        boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(output)
        if boxed_match is not None:
            output = f"Answer: {boxed_match.group(1)}"
        else:
            last_line = output.strip("\n").split("\n")[-1].lower()
            output = f"Answer: {last_line}"

    cleaned = output.replace("*", "").replace("#", "").lower()
    answer_section = cleaned.split("answer: ")[-1].replace(",", "")
    match = _PLATINUM_REFERENCE_NUMBER_RE.search(answer_section)
    if match is None:
        return INVALID_ANSWER
    return _TRAILING_ZERO_RE.sub("", match.group())


def extract_format_insensitive_numeric_answer(text: str) -> str:
    output = text or ""

    hash_match = _HASH_ANSWER_RE.search(output)
    if hash_match is not None:
        return canonicalize_numeric_token(hash_match.group(1))

    answer_matches = list(_ANSWER_LINE_RE.finditer(output))
    if answer_matches:
        extracted = _extract_first_numeric_token(answer_matches[-1].group(1))
        if extracted != INVALID_ANSWER:
            return extracted

    boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(output)
    if boxed_match is not None:
        extracted = _extract_first_numeric_token(boxed_match.group(1))
        if extracted != INVALID_ANSWER:
            return extracted

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        extracted = _extract_last_numeric_token(lines[-1])
        if extracted != INVALID_ANSWER:
            return extracted

    return _extract_last_numeric_token(output)


def canonicalize_numeric_token(token: str) -> str:
    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return INVALID_ANSWER
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return INVALID_ANSWER
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def numbers_equal(left: str, right: str) -> bool:
    if left == INVALID_ANSWER or right == INVALID_ANSWER:
        return False
    return canonicalize_numeric_token(left) == canonicalize_numeric_token(right)


def _extract_first_numeric_token(text: str) -> str:
    match = _NUMERIC_TOKEN_RE.search(text)
    if match is None:
        return INVALID_ANSWER
    return canonicalize_numeric_token(match.group())


def _extract_last_numeric_token(text: str) -> str:
    matches = list(_NUMERIC_TOKEN_RE.finditer(text))
    if not matches:
        return INVALID_ANSWER
    return canonicalize_numeric_token(matches[-1].group())


def gsm8k_reference_target(doc: dict[str, Any]) -> str:
    target = extract_gsm8k_reference_answer(str(doc["answer"]))
    if target == INVALID_ANSWER:
        raise ValueError("GSM8K ground-truth answer is missing the official `#### number` marker")
    return target


def gsm8k_numeric_target(doc: dict[str, Any]) -> str:
    return canonicalize_numeric_token(gsm8k_reference_target(doc))


def gsm8k_platinum_reference_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"]).split("\n#### ")[-1].replace(",", "").strip()


def gsm8k_platinum_numeric_target(doc: dict[str, Any]) -> str:
    target = canonicalize_numeric_token(gsm8k_platinum_reference_target(doc))
    if target == INVALID_ANSWER:
        raise ValueError("GSM8K-Platinum ground-truth answer could not be parsed into a numeric target")
    return target
