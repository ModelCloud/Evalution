# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import string
from collections import Counter
import pcre

# Keep scorer defaults and parser helpers explicit at module scope.
_ARTICLES_RE = pcre.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = pcre.compile(r"\s+")
_NO_ANSWER_CANONICAL = "unanswerable"
_NO_ANSWER_ALIASES = {
    "",
    "unanswerable",
    "no answer",
    "cannot be answered",
    "can't be answered",
    "not answerable",
}


def normalize_qa_text(text: str) -> str:
    """Normalize QA text. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def canonicalize_no_answer(text: str) -> str:
    """Implement canonicalize no answer for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    normalized = normalize_qa_text(text)
    if normalized in _NO_ANSWER_ALIASES:
        return _NO_ANSWER_CANONICAL
    return normalized


def qa_exact_match(prediction: str, target: str) -> float:
    """Implement QA exact match for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return float(canonicalize_no_answer(prediction) == canonicalize_no_answer(target))


def qa_f1(prediction: str, target: str) -> float:
    """Implement QA F1 for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    prediction_tokens = canonicalize_no_answer(prediction).split()
    target_tokens = canonicalize_no_answer(target).split()
    if not prediction_tokens or not target_tokens:
        return float(prediction_tokens == target_tokens)

    overlap = Counter(prediction_tokens) & Counter(target_tokens)
    match_count = sum(overlap.values())
    if match_count == 0:
        return 0.0

    precision = match_count / len(prediction_tokens)
    recall = match_count / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def best_qa_scores(prediction: str, targets: list[str]) -> tuple[float, float, int]:
    """Implement best QA scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if not targets:
        raise ValueError("qa targets must contain at least one reference answer")

    best_index = 0
    best_exact = -1.0
    best_f1_score = -1.0
    for index, target in enumerate(targets):
        exact = qa_exact_match(prediction, target)
        f1_score = qa_f1(prediction, target)
        if (exact, f1_score) > (best_exact, best_f1_score):
            best_index = index
            best_exact = exact
            best_f1_score = f1_score
    return best_exact, best_f1_score, best_index
