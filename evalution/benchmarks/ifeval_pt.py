# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.benchmarks.ifeval import IFEval

# Point the Portuguese IFEval variant at the translated corpus from the recent harness PR.
IFEVAL_PT_DATASET_PATH = "Polygl0t/IFEval-PT"
IFEVAL_PT_SPLIT = "train"


@dataclass(slots=True)
class IFEvalPT(IFEval):
    # Reuse the clean-room IFEval scorer against the Portuguese translation.
    """Define the IFEval pt helper class."""
    dataset_path: str = IFEVAL_PT_DATASET_PATH
    split: str = IFEVAL_PT_SPLIT

    # Keep the registry-visible task name distinct from the English suite.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "ifeval_pt"


# Expose the Portuguese translated suite through the public benchmark namespace.
def ifeval_pt(**kwargs: Any) -> IFEvalPT:
    """Implement IFEval pt for this module."""
    return IFEvalPT(**kwargs)
