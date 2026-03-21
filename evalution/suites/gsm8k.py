from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.suites.gsm8k_common import build_variant_specs as _build_variant_specs

_VARIANTS = _build_variant_specs("gsm8k")


@dataclass(slots=True)
class GSM8K(BaseGSM8KSuite):
    VARIANTS = _VARIANTS
    dataset_path: str = "openai/gsm8k"
    dataset_name: str | None = "main"

    def dataset_loader(self) -> Any:
        return load_dataset


def gsm8k(**kwargs: Any) -> GSM8K:
    return GSM8K(**kwargs)
