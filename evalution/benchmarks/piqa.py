# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


@dataclass(slots=True)
class PIQA(BaseMultipleChoiceSuite):
    # Evaluate physical commonsense by ranking two candidate solutions with token log-likelihood.
    """Implement the PIQA benchmark suite."""
    dataset_path: str = "baber/piqa"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the maintained PIQA mirror that works with current datasets.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "piqa"

    # Convert one PIQA row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=f"Question: {doc['goal']}\nAnswer:",
            choices=[doc["sol1"], doc["sol2"]],
            gold_index=int(doc["label"]),
        )


# Mirror the public suite factory style used by the rest of the package.
def piqa(**kwargs: Any) -> PIQA:
    """Implement PIQA for this module."""
    return PIQA(**kwargs)
