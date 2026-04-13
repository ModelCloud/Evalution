# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.mmlu_pro import MMLUPro


@dataclass(slots=True)
class MMLUProPlus(MMLUPro):
    # Reuse the MMLU-Pro prompting and scoring pipeline against the expanded MMLU-Pro-Plus dataset release.
    """Define the mmlupro plus helper class."""
    dataset_path: str = "saeidasgari/mmlu-pro-plus"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return "mmlu_pro_plus"
        suffix = "__".join(canonical.replace(".", "_") for canonical in resolved_subsets.canonicals)
        return f"mmlu_pro_plus_{suffix}"


def mmlu_pro_plus(**kwargs: Any) -> MMLUProPlus:
    # Keep the public constructor symmetric with mmlu_pro so YAML and Python stay parallel.
    """Implement MMLU pro plus for this module."""
    return MMLUProPlus(**kwargs)
