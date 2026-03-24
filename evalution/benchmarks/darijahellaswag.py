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
class DarijaHellaSwag(BaseTranslatedHellaSwagSuite):
    dataset_path: str = "MBZUAI-Paris/DarijaHellaSwag"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "darijahellaswag"


def darijahellaswag(**kwargs: Any) -> DarijaHellaSwag:
    return DarijaHellaSwag(**kwargs)
