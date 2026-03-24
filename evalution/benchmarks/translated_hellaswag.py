# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.benchmarks.hellaswag import _clean_hellaswag_text
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


@dataclass(slots=True)
class BaseTranslatedHellaSwagSuite(BaseMultipleChoiceSuite):
    split: str = "validation"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        prompt = _clean_hellaswag_text(f"{doc['activity_label']}: {doc['ctx']}")
        choices = [_clean_hellaswag_text(choice) for choice in doc["endings"]]
        return MultipleChoiceSample(
            index=index,
            prompt=prompt,
            choices=choices,
            gold_index=int(doc["label"]),
            metadata={
                "activity_label": str(doc["activity_label"]).strip(),
                "source_id": str(doc["source_id"]).strip(),
                "split_type": str(doc["split_type"]).strip(),
            },
        )

    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        lines = [
            f"Context: {sample.prompt}",
            "Question: Which ending best continues the context?",
        ]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.choices[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)
