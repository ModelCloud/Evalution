# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.suites.gsm8k_common import build_variant_specs as _build_variant_specs
from evalution.suites.gsm8k_common import gsm8k_platinum_numeric_target

GSM8KPlatinumVariant = GSM8KVariant
_VARIANTS = _build_variant_specs("gsm8k_platinum")


@dataclass(slots=True)
class GSM8KPlatinum(BaseGSM8KSuite):
    VARIANTS = _VARIANTS
    INCLUDE_CLEANING_STATUS = True
    SCORING_MODE = "numeric_format_insensitive"
    dataset_path: str = "madrylab/gsm8k-platinum"
    dataset_name: str | None = "main"

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        return gsm8k_platinum_numeric_target(doc)

    # Use the Hugging Face datasets loader for the GSM8K-Platinum benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset


# Convenience constructor mirroring the public suite factory style.
def gsm8k_platinum(**kwargs: Any) -> GSM8KPlatinum:
    return GSM8KPlatinum(**kwargs)
