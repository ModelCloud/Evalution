# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations


def extract_math_answer(text: str) -> str:
    """Extract math answer. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    indices = [index for index, char in enumerate(text) if char == "$"]
    if len(indices) > 1:
        answer = text[indices[0] + 1 : indices[-1]]
    else:
        answer = text

    boxed_answer = last_boxed_only_string(text)
    if boxed_answer is not None:
        try:
            boxed_content = remove_boxed(boxed_answer)
            if boxed_content is not None:
                answer = boxed_content
        except (AssertionError, IndexError):
            pass
    return answer


def math_exact_match(prediction: str, target: str) -> float:
    """Implement math exact match for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return float(math_strings_equivalent(extract_math_answer(prediction), target))


def math_strings_equivalent(left: str | None, right: str | None) -> bool:
    """Implement math strings equivalent for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    try:
        return normalize_math_string(left) == normalize_math_string(right)
    except Exception:
        return left == right


def remove_boxed(text: str) -> str:
    """Implement remove boxed for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if "\\boxed " in text:
        prefix = "\\boxed "
        assert text[: len(prefix)] == prefix
        return text[len(prefix) :]

    prefix = "\\boxed{"
    assert text[: len(prefix)] == prefix
    assert text[-1] == "}"
    return text[len(prefix) : -1]


def last_boxed_only_string(text: str) -> str | None:
    """Implement last boxed only string for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    index = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = text.rfind("\\fbox")
        if index < 0:
            return None

    cursor = index
    right_brace_index: int | None = None
    open_brace_count = 0
    while cursor < len(text):
        if text[cursor] == "{":
            open_brace_count += 1
        if text[cursor] == "}":
            open_brace_count -= 1
            if open_brace_count == 0:
                right_brace_index = cursor
                break
        cursor += 1

    if right_brace_index is None:
        return None
    return text[index : right_brace_index + 1]


def fix_fracs(text: str) -> str:
    """Implement fix fracs for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    parts = text.split("\\frac")
    rebuilt = parts[0]
    if len(parts) == 1:
        return text
    for part in parts[1:]:
        rebuilt += "\\frac"
        if part[0] == "{":
            rebuilt += part
            continue
        if len(part) < 2:
            return text
        numerator = part[0]
        denominator = part[1]
        if denominator != "{":
            suffix = part[2:] if len(part) > 2 else ""
            rebuilt += "{" + numerator + "}{" + denominator + "}" + suffix
        else:
            suffix = part[2:] if len(part) > 2 else ""
            rebuilt += "{" + numerator + "}" + denominator + suffix
    return rebuilt


def fix_a_slash_b(text: str) -> str:
    """Implement fix a slash b for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if len(text.split("/")) != 2:
        return text
    numerator, denominator = text.split("/")
    try:
        numerator_int = int(numerator)
        denominator_int = int(denominator)
        assert text == f"{numerator_int}/{denominator_int}"
        return f"\\frac{{{numerator_int}}}{{{denominator_int}}}"
    except (AssertionError, ValueError):
        return text


def remove_right_units(text: str) -> str:
    # Trim trailing unit annotations without assuming the marker appears only once.
    """Implement remove right units for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if "\\text{ " in text:
        return text.split("\\text{ ", 1)[0]
    return text


def fix_sqrt(text: str) -> str:
    """Implement fix sqrt for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if "\\sqrt" not in text:
        return text
    parts = text.split("\\sqrt")
    rebuilt = parts[0]
    for part in parts[1:]:
        if part[0] != "{":
            rebuilt += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            rebuilt += "\\sqrt" + part
    return rebuilt


def normalize_math_string(text: str) -> str:
    """Normalize math string. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    normalized = text.replace("\n", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\\\", "\\")
    normalized = normalized.replace("tfrac", "frac")
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("^{\\circ}", "")
    normalized = normalized.replace("^\\circ", "")
    normalized = normalized.replace("\\$", "")
    normalized = remove_right_units(normalized)
    normalized = normalized.replace("\\%", "")
    normalized = normalized.replace("\\%", "")
    normalized = normalized.replace(" .", " 0.")
    normalized = normalized.replace("{.", "{0.")
    if not normalized:
        return normalized
    if normalized[0] == ".":
        normalized = "0" + normalized
    if len(normalized.split("=")) == 2 and len(normalized.split("=")[0]) <= 2:
        normalized = normalized.split("=")[1]
    normalized = fix_sqrt(normalized)
    normalized = normalized.replace(" ", "")
    normalized = fix_fracs(normalized)
    if normalized == "0.5":
        normalized = "\\frac{1}{2}"
    normalized = fix_a_slash_b(normalized)
    return normalized
