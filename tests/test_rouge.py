# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest

from evalution.scorers.rouge import RougeScorer


# Frozen from the Google Research ROUGE reference behavior on 2026-04-19.
# Reference repo: https://github.com/google-research/google-research/tree/master/rouge
_REFERENCE_SCORE_CASES = (
    {
        "name": "basic_ngram_and_lcs",
        "rouge_types": ("rouge1", "rouge2", "rougeL"),
        "use_stemmer": False,
        "split_summaries": False,
        "target": "The quick brown fox jumps over the lazy dog",
        "prediction": "The quick brown dog jumps on the log.",
        "expected": {
            "rouge1": (0.75, 0.6666666666666666, 0.7058823529411765),
            "rouge2": (0.2857142857142857, 0.25, 0.26666666666666666),
            "rougeL": (0.625, 0.5555555555555556, 0.5882352941176471),
        },
    },
    {
        "name": "stemming_and_rougelsum",
        "rouge_types": ("rouge1", "rouge2", "rougeLsum"),
        "use_stemmer": True,
        "split_summaries": False,
        "target": "Cats are running faster\nthan the other animals.",
        "prediction": "A cat runs fast\nthan other animal.",
        "expected": {
            "rouge1": (0.7142857142857143, 0.625, 0.6666666666666666),
            "rouge2": (0.16666666666666666, 0.14285714285714285, 0.15384615384615383),
            "rougeLsum": (0.7142857142857143, 0.625, 0.6666666666666666),
        },
    },
    {
        "name": "duplicate_tokens_rougelsum",
        "rouge_types": ("rougeLsum",),
        "use_stemmer": False,
        "split_summaries": False,
        "target": "alpha beta alpha\nbeta gamma alpha",
        "prediction": "alpha beta beta\nalpha gamma",
        "expected": {
            "rougeLsum": (0.8, 0.6666666666666666, 0.7272727272727272),
        },
    },
    {
        "name": "empty_prediction",
        "rouge_types": ("rouge1", "rouge2", "rougeL", "rougeLsum"),
        "use_stemmer": True,
        "split_summaries": False,
        "target": "alpha beta\ngamma",
        "prediction": "",
        "expected": {
            "rouge1": (0.0, 0.0, 0.0),
            "rouge2": (0.0, 0.0, 0.0),
            "rougeL": (0.0, 0.0, 0.0),
            "rougeLsum": (0.0, 0.0, 0.0),
        },
    },
    {
        "name": "empty_target",
        "rouge_types": ("rouge1", "rouge2", "rougeL", "rougeLsum"),
        "use_stemmer": True,
        "split_summaries": False,
        "target": "",
        "prediction": "alpha beta\ngamma",
        "expected": {
            "rouge1": (0.0, 0.0, 0.0),
            "rouge2": (0.0, 0.0, 0.0),
            "rougeL": (0.0, 0.0, 0.0),
            "rougeLsum": (0.0, 0.0, 0.0),
        },
    },
    {
        "name": "sentence_split_mode",
        "rouge_types": ("rougeLsum",),
        "use_stemmer": True,
        "split_summaries": True,
        "target": "First sentence here. Second sentence is longer.",
        "prediction": "First sentence here. Second sentence changed.",
        "expected": {
            "rougeLsum": (0.8333333333333334, 0.7142857142857143, 0.7692307692307692),
        },
    },
)


@pytest.mark.parametrize("case", _REFERENCE_SCORE_CASES, ids=[case["name"] for case in _REFERENCE_SCORE_CASES])
def test_rouge_scorer_matches_frozen_reference_vectors(case: dict[str, object]) -> None:
    """Verify the local ROUGE scorer matches frozen reference vectors from the Google Research behavior."""

    scorer = RougeScorer(
        case["rouge_types"],
        use_stemmer=bool(case["use_stemmer"]),
        split_summaries=bool(case["split_summaries"]),
    )
    scores = scorer.score(case["target"], case["prediction"])

    expected = case["expected"]
    for rouge_type, expected_values in expected.items():
        assert tuple(scores[rouge_type]) == pytest.approx(expected_values)


def test_rouge_scorer_score_multi_matches_frozen_reference_vector() -> None:
    """Verify score_multi matches the frozen multi-reference reference behavior."""

    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score_multi(
        ["alpha beta gamma", "alpha zeta", "gamma zeta"],
        "alpha gamma",
    )

    assert tuple(scores["rouge1"]) == pytest.approx((1.0, 0.6666666666666666, 0.8))
    assert tuple(scores["rougeL"]) == pytest.approx((1.0, 0.6666666666666666, 0.8))
