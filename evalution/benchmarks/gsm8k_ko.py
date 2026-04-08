# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.benchmarks.gsm8k_common import build_variant_specs as _build_variant_specs
from evalution.scorers.gsm8k import gsm8k_numeric_target

# Keep translated GSM8K task names aligned with the shared variant registry.
_VARIANTS = _build_variant_specs("gsm8k_ko")


@dataclass(slots=True)
class GSM8KKO(BaseGSM8KSuite):
    # Score the Korean GSM8K translation with the shared numeric-answer evaluator.
    VARIANTS = _VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    variant: GSM8KVariant = "base"
    dataset_path: str = "kuotient/gsm8k-ko"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = True

    # Reuse the standard Hugging Face datasets loader for the translated corpus.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Normalize the Korean reference solution to the shared GSM8K numeric target.
    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        return gsm8k_numeric_target(doc)


# Expose the translated suite through the public benchmark namespace.
def gsm8k_ko(**kwargs: Any) -> GSM8KKO:
    return GSM8KKO(**kwargs)
