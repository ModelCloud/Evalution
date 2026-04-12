# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Sequence
import pcre

# Keep scorer defaults and parser helpers explicit at module scope.
_SPLIT_PUNCTS_PATTERN = pcre.compile(r"[\w]+|[^\s\w]")


def _split_punct_tokens(text: str) -> list[str]:
    """Split punct tokens. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return _SPLIT_PUNCTS_PATTERN.findall(text.strip().lower())


def _as_reference_texts(reference: str | Sequence[str]) -> list[str]:
    """Implement as reference texts for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if isinstance(reference, str):
        return [reference]
    return [str(item) for item in reference]


def _count_ngrams(tokens: Sequence[str], *, max_order: int) -> Counter[tuple[str, ...]]:
    """Implement count ngrams for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    counts: Counter[tuple[str, ...]] = Counter()
    for order in range(1, max_order + 1):
        for index in range(len(tokens) - order + 1):
            counts[tuple(tokens[index : index + order])] += 1
    return counts


def smoothed_corpus_bleu_4(
    references: Sequence[str | Sequence[str]],
    predictions: Sequence[str],
) -> float:
    """Implement smoothed corpus bleu 4 for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")

    max_order = 4
    total_reference_length = 0
    total_prediction_length = 0
    total_guess = [0] * max_order
    total_correct = [0] * max_order

    for reference, prediction in zip(references, predictions, strict=True):
        reference_token_lists = [
            _split_punct_tokens(reference_text)
            for reference_text in _as_reference_texts(reference)
        ]
        prediction_tokens = _split_punct_tokens(str(prediction))

        total_reference_length += min(len(tokens) for tokens in reference_token_lists)
        total_prediction_length += len(prediction_tokens)

        prediction_counts = _count_ngrams(prediction_tokens, max_order=max_order)
        reference_max_counts: Counter[tuple[str, ...]] = Counter()
        for reference_tokens in reference_token_lists:
            for ngram, count in _count_ngrams(reference_tokens, max_order=max_order).items():
                reference_max_counts[ngram] = max(reference_max_counts.get(ngram, 0), count)

        for order in range(1, max_order + 1):
            total_guess[order - 1] += max(len(prediction_tokens) - order + 1, 0)

        for ngram, count in prediction_counts.items():
            total_correct[len(ngram) - 1] += min(reference_max_counts.get(ngram, 0), count)

    log_bleu = 0.0
    for order in range(max_order):
        smooth = 1 if order > 0 else 0
        log_bleu += math.log(total_correct[order] + smooth + sys.float_info.min) - math.log(
            total_guess[order] + smooth + sys.float_info.min
        )
    log_bleu /= max_order

    brevity_penalty = min(
        0.0,
        1.0 - float(total_reference_length + 1) / float(total_prediction_length + 1),
    )
    return math.exp(log_bleu + brevity_penalty) * 100.0
