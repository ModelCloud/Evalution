# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections.abc import Sequence

import sacrebleu


def corpus_translation_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, float]:
    """Implement corpus translation metrics for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not references:
        return {
            "bleu": 0.0,
            "chrf": 0.0,
            "ter": 0.0,
        }

    reference_sets = [list(references)]
    return {
        "bleu": float(sacrebleu.corpus_bleu(list(predictions), reference_sets).score),
        "chrf": float(sacrebleu.corpus_chrf(list(predictions), reference_sets).score),
        "ter": float(sacrebleu.corpus_ter(list(predictions), reference_sets).score),
    }
