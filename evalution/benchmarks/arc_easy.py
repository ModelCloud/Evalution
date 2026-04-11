# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.arc_exam import BaseARCExamSuite


@dataclass(slots=True)
class ARCEasy(BaseARCExamSuite):
    # ARC-Easy uses the same original ARC exam-score rule as ARC-Challenge.
    """Define the arceasy helper class."""
    dataset_name: str | None = "ARC-Easy"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "arc_easy"


def arc_easy(**kwargs: Any) -> ARCEasy:
    """Implement ARC easy for this module."""
    return ARCEasy(**kwargs)
