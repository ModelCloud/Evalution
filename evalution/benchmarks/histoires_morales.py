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
    return " ".join(
        str(doc[field]).strip()
        for field in ("norm", "situation", "intention")
    )


@dataclass(slots=True)
class HistoiresMorales(BaseMultipleChoiceSuite):
    dataset_path: str = "LabHC/histoires_morales"
    split: str = "train"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "histoires_morales"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
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
    return HistoiresMorales(**kwargs)
