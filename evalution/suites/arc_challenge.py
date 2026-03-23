# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.arc_exam import BaseARCExamSuite


@dataclass(slots=True)
class ARCChallenge(BaseARCExamSuite):
    # ARC-Challenge uses the original ARC exam-score rule via BaseARCExamSuite.
    dataset_name: str | None = "ARC-Challenge"
    split: str = "test"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "arc_challenge"


def arc_challenge(**kwargs: Any) -> ARCChallenge:
    return ARCChallenge(**kwargs)
