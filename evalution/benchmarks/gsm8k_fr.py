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
_VARIANTS = _build_variant_specs("gsm8k_fr")


@dataclass(slots=True)
class GSM8KFR(BaseGSM8KSuite):
    # Score the French GSM8K translation with the shared numeric-answer evaluator.
    """Define the gsm8 kfr helper class."""
    VARIANTS = _VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    variant: GSM8KVariant = "base"
    dataset_path: str = "cmh/gsm8k_fr"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = True

    # Reuse the standard Hugging Face datasets loader for the translated corpus.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Normalize the French reference solution to the shared GSM8K numeric target.
    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        """Implement numeric target from doc for gsm8 kfr."""
        return gsm8k_numeric_target(doc)


# Expose the translated suite through the public benchmark namespace.
def gsm8k_fr(**kwargs: Any) -> GSM8KFR:
    """Implement GSM8K fr for this module."""
    return GSM8KFR(**kwargs)
