# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
import re
import string
import unicodedata

from rouge_score import rouge_scorer

_EN_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_DIGIT_RE = re.compile(r"\d+")
_PARAGRAPH_RE = re.compile(r"Paragraph (\d+)")
_PARAGRAPH_ZH_RE = re.compile(r"段落(\d+)")


def _is_punctuation(character: str) -> bool:
    return character in string.punctuation or unicodedata.category(character).startswith("P")


def _normalize_english_qa_text(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(character for character in lowered if not _is_punctuation(character))
    no_articles = _EN_ARTICLES_RE.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def _tokenize_zh_text(text: str) -> list[str]:
    # Keep Chinese scoring dependency-free by splitting CJK code points as standalone tokens
    # while still preserving contiguous Latin or digit sequences.
    tokens: list[str] = []
    buffered: list[str] = []
    for character in text.lower():
        if character.isspace() or _is_punctuation(character):
            if buffered:
                tokens.append("".join(buffered))
                buffered.clear()
            continue
        if "\u4e00" <= character <= "\u9fff":
            if buffered:
                tokens.append("".join(buffered))
                buffered.clear()
            tokens.append(character)
            continue
        buffered.append(character)
    if buffered:
        tokens.append("".join(buffered))
    return tokens


def _token_f1(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    remaining: dict[str, int] = {}
    for token in reference_tokens:
        remaining[token] = remaining.get(token, 0) + 1

    overlap = 0
    for token in prediction_tokens:
        count = remaining.get(token, 0)
        if count <= 0:
            continue
        remaining[token] = count - 1
        overlap += 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def longbench_qa_f1_score(prediction: str, reference: str) -> float:
    return _token_f1(
        _normalize_english_qa_text(prediction).split(),
        _normalize_english_qa_text(reference).split(),
    )


def longbench_qa_f1_zh_score(prediction: str, reference: str) -> float:
    return _token_f1(
        _tokenize_zh_text(prediction),
        _tokenize_zh_text(reference),
    )


@lru_cache(maxsize=1)
def _rouge_l_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def longbench_rouge_score(prediction: str, reference: str) -> float:
    return _rouge_l_scorer().score(reference, prediction)["rougeL"].fmeasure


def longbench_rouge_zh_score(prediction: str, reference: str) -> float:
    return longbench_rouge_score(
        " ".join(_tokenize_zh_text(prediction)),
        " ".join(_tokenize_zh_text(reference)),
    )


def longbench_classification_score(
    prediction: str,
    reference: str,
    *,
    all_classes: list[str],
) -> float:
    matches = [class_name for class_name in all_classes if class_name and class_name in prediction]
    filtered_matches = [
        match
        for match in matches
        if not (match != reference and match in reference)
    ]
    if reference not in filtered_matches:
        return 0.0
    return 1.0 / len(filtered_matches)


def longbench_count_score(prediction: str, reference: str) -> float:
    numbers = _DIGIT_RE.findall(prediction)
    if not numbers:
        return 0.0
    reference_text = str(reference).strip()
    correct = sum(1 for number in numbers if number == reference_text)
    return correct / len(numbers)


def _retrieval_score(
    prediction: str,
    reference: str,
    *,
    pattern: re.Pattern[str],
) -> float:
    match = pattern.search(reference)
    if match is None:
        return 0.0
    target_number = match.group(1)
    numbers = _DIGIT_RE.findall(prediction)
    if not numbers:
        return 0.0
    correct = sum(1 for number in numbers if number == target_number)
    return correct / len(numbers)


def longbench_retrieval_score(prediction: str, reference: str) -> float:
    return _retrieval_score(prediction, reference, pattern=_PARAGRAPH_RE)


def longbench_retrieval_zh_score(prediction: str, reference: str) -> float:
    return _retrieval_score(prediction, reference, pattern=_PARAGRAPH_ZH_RE)


def _first_code_candidate_line(prediction: str) -> str:
    for line in prediction.lstrip("\n").splitlines():
        if "`" in line or "#" in line or "//" in line:
            continue
        if line:
            return line
    return ""


def longbench_code_sim_score(prediction: str, reference: str) -> float:
    candidate_line = _first_code_candidate_line(prediction)
    if not candidate_line and not reference:
        return 1.0
    return SequenceMatcher(None, candidate_line, reference).ratio()


__all__ = [
    "longbench_classification_score",
    "longbench_code_sim_score",
    "longbench_count_score",
    "longbench_qa_f1_score",
    "longbench_qa_f1_zh_score",
    "longbench_retrieval_score",
    "longbench_retrieval_zh_score",
    "longbench_rouge_score",
    "longbench_rouge_zh_score",
]
