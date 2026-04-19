# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from functools import lru_cache

from evalution.scorers.rouge import RougeScorer


@lru_cache(maxsize=1)
def _summary_rouge_scorer() -> RougeScorer:
    """Implement summary ROUGE scorer for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"],
        use_stemmer=True,
    )


def summary_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """Implement summary ROUGE scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    scores = _summary_rouge_scorer().score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeLsum": scores["rougeLsum"].fmeasure,
    }
