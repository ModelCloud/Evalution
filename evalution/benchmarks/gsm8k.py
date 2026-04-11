# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.scorers.gsm8k import gsm8k_numeric_target
from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.benchmarks.gsm8k_common import build_variant_specs as _build_variant_specs

# Keep benchmark defaults and public task ids explicit at module scope.
_VARIANTS = _build_variant_specs("gsm8k")


@dataclass(slots=True)
class GSM8K(BaseGSM8KSuite):
    """Define the gsm8 k helper class."""
    # Keep the class-level state explicit for this helper.
    VARIANTS = _VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    dataset_path: str = "openai/gsm8k"
    dataset_name: str | None = "main"

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        """Implement numeric target from doc for gsm8 k."""
        return gsm8k_numeric_target(doc)

    # Use the Hugging Face datasets loader for the standard GSM8K benchmark.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset


# Convenience constructor mirroring the public suite factory style.
def gsm8k(**kwargs: Any) -> GSM8K:
    """Implement GSM8K for this module."""
    return GSM8K(**kwargs)
