from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.suites.gsm8k_common import build_variant_specs as _build_variant_specs

GSM8KPlatinumVariant = GSM8KVariant
_VARIANTS = _build_variant_specs("gsm8k_platinum")


@dataclass(slots=True)
class GSM8KPlatinum(BaseGSM8KSuite):
    VARIANTS = _VARIANTS
    INCLUDE_CLEANING_STATUS = True
    dataset_path: str = "madrylab/gsm8k-platinum"
    dataset_name: str | None = "main"

    # Use the Hugging Face datasets loader for the GSM8K-Platinum benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset


# Convenience constructor mirroring the public suite factory style.
def gsm8k_platinum(**kwargs: Any) -> GSM8KPlatinum:
    return GSM8KPlatinum(**kwargs)
