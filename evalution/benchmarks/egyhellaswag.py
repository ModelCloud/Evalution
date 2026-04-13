# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.translated_hellaswag import BaseTranslatedHellaSwagSuite


@dataclass(slots=True)
class EgyHellaSwag(BaseTranslatedHellaSwagSuite):
    """Define the egy hella swag helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str = "UBC-NLP/EgyHellaSwag"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "egyhellaswag"


def egyhellaswag(**kwargs: Any) -> EgyHellaSwag:
    """Implement egyhellaswag for this module."""
    return EgyHellaSwag(**kwargs)
