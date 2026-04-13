# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _histoires_morales_query(doc: dict[str, Any]) -> str:
    """Implement histoires morales query for this module."""
    return " ".join(
        str(doc[field]).strip()
        for field in ("norm", "situation", "intention")
    )


@dataclass(slots=True)
class HistoiresMorales(BaseMultipleChoiceSuite):
    """Implement the histoires morales benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "LabHC/histoires_morales"
    split: str = "train"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "histoires_morales"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        moral_action = str(doc["moral_action"]).strip()
        immoral_action = str(doc["immoral_action"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_histoires_morales_query(doc),
            choices=[moral_action, immoral_action],
            gold_index=0,
            metadata={
                "guid": str(doc["guid"]),
                "norm": str(doc["norm"]).strip(),
                "situation": str(doc["situation"]).strip(),
                "intention": str(doc["intention"]).strip(),
                "moral_action": moral_action,
                "immoral_action": immoral_action,
                "moral_consequence": str(doc["moral_consequence"]).strip(),
                "immoral_consequence": str(doc["immoral_consequence"]).strip(),
            },
        )


def histoires_morales(**kwargs: Any) -> HistoiresMorales:
    """Implement histoires morales for this module."""
    return HistoiresMorales(**kwargs)
