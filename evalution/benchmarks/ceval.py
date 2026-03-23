# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

CEVAL_SUBSETS = (
    "accountant",
    "advanced_mathematics",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "chinese_language_and_literature",
    "civil_servant",
    "clinical_medicine",
    "college_chemistry",
    "college_economics",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "ideological_and_moral_cultivation",
    "law",
    "legal_professional",
    "logic",
    "mao_zedong_thought",
    "marxism",
    "metrology_engineer",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics",
    "modern_chinese_history",
    "operating_system",
    "physician",
    "plant_protection",
    "probability_and_statistics",
    "professional_tour_guide",
    "sports_science",
    "tax_accountant",
    "teacher_qualification",
    "urban_and_rural_planner",
    "veterinary_medicine",
)

_CEVAL_LABELS = ("A", "B", "C", "D")


def _ceval_prompt(doc: dict[str, Any]) -> str:
    lines = [str(doc["question"]).strip()]
    for label in _CEVAL_LABELS:
        lines.append(f"{label}. {str(doc[label]).strip()}")
    lines.append("答案：")
    return "\n".join(lines)


@dataclass(slots=True)
class CEval(BaseMultipleChoiceSuite):
    dataset_path: str = "ceval/ceval-exam"
    dataset_name: str | None = None
    split: str = "val"
    subset: str = ""

    def __post_init__(self) -> None:
        if self.subset not in CEVAL_SUBSETS:
            raise ValueError(f"unsupported ceval subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("ceval dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"ceval_{self.subset}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raw_choices = [str(doc[label]).strip() for label in _CEVAL_LABELS]
        answer_label = str(doc["answer"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_ceval_prompt(doc),
            choices=list(_CEVAL_LABELS),
            gold_index=_CEVAL_LABELS.index(answer_label),
            metadata={
                "id": int(doc["id"]),
                "subset": self.subset,
                "question": str(doc["question"]).strip(),
                "raw_choices": raw_choices,
                "answer_label": answer_label,
            },
        )


def ceval(*, subset: str, **kwargs: Any) -> CEval:
    return CEval(subset=subset, dataset_name=subset, **kwargs)
