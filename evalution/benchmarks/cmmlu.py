# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from zipfile import ZipFile

from datasets import Dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.fewshot_multiple_choice import BaseFewshotMultipleChoiceSuite

# Freeze the upstream subject registry so imports stay deterministic and offline-safe.
CMMLU_SUBSETS = (
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions",
)
CMMLU_TASKS = tuple(f"cmmlu_{subset}" for subset in CMMLU_SUBSETS)
# Keep the CMMLU answer-label convention explicit for prompt rendering and scoring.
_CMMLU_LABELS = ("A", "B", "C", "D")
# Mirror the short Chinese instruction used by the upstream benchmark prompts.
_CMMLU_DESCRIPTION = "以下是单项选择题，请直接给出正确答案的选项。"


@lru_cache(maxsize=1)
def _cmmlu_archive_path() -> str:
    # Reuse one cached archive path so repeated subset loads do not re-resolve the Hub file.
    """Implement cmmlu archive path for this module."""
    return hf_hub_download(
        repo_id="haonan-li/cmmlu",
        filename="cmmlu_v1_0_1.zip",
        repo_type="dataset",
    )


@lru_cache(maxsize=None)
def _cmmlu_rows(dataset_name: str, split: str) -> tuple[dict[str, Any], ...]:
    # Materialize one subject split from the published CMMLU zip archive into plain row dicts.
    """Implement cmmlu rows for this module."""
    member_name = f"{split}/{dataset_name}.csv"
    rows: list[dict[str, Any]] = []
    with ZipFile(_cmmlu_archive_path()) as archive:
        with archive.open(member_name) as rows_file:
            text_file = io.TextIOWrapper(rows_file, encoding="utf-8")
            reader = csv.DictReader(text_file)
            for row in reader:
                cleaned = {key: value for key, value in row.items() if key}
                rows.append(cleaned)
    return tuple(rows)


def _load_cmmlu_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Dataset:
    # Adapt the raw CMMLU archive to the standard Evalution dataset-loader contract.
    """Load cmmlu dataset. Preserve the fallback order expected by the surrounding caller."""
    del cache_dir
    if dataset_path != "haonan-li/cmmlu":
        raise ValueError(f"unsupported CMMLU dataset path: {dataset_path!r}")
    if dataset_name not in CMMLU_SUBSETS:
        raise ValueError(f"unsupported CMMLU subset: {dataset_name!r}")
    if split not in {"dev", "test"}:
        raise ValueError(f"unsupported CMMLU split: {split!r}")
    if stream:
        raise ValueError("CMMLU raw zip loader requires non-stream dataset materialization")
    return Dataset.from_list(list(_cmmlu_rows(dataset_name, split)))


@dataclass(slots=True)
class CMMLU(BaseFewshotMultipleChoiceSuite):
    # Keep CMMLU close to the upstream lm-eval setup by using per-subject dev examples as few-shot context.
    """Define the cmmlu helper class."""
    dataset_path: str = "haonan-li/cmmlu"
    dataset_name: str | None = None
    split: str = "test"
    fewshot_split: str = "dev"
    subset: str = ""

    def __post_init__(self) -> None:
        # Keep the factory subset and the dataset name locked together to avoid silent mismatches.
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset not in CMMLU_SUBSETS:
            raise ValueError(f"unsupported CMMLU subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("cmmlu dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        # Route CMMLU through the archive-backed loader above.
        """Return the dataset loader bound to this suite."""
        return _load_cmmlu_dataset

    def task_name(self) -> str:
        # Expose one stable task name per CMMLU subject factory.
        """Return the exported task name for this suite."""
        return f"cmmlu_{self.subset}"

    def prompt_description(self) -> str:
        # Prefix each evaluation prompt with the benchmark's short Chinese instruction.
        """Implement prompt description for cmmlu."""
        return _CMMLU_DESCRIPTION

    def format_question(self, doc: dict[str, Any], *, include_answer: bool) -> str:
        # Render one CMMLU question block in the same A/B/C/D layout used for few-shot examples.
        """Format question."""
        answer_text = str(doc["Answer"]).strip() if include_answer else ""
        lines = [str(doc["Question"]).strip()]
        lines.extend(
            f"{label}. {str(doc[label]).strip()}"
            for label in _CMMLU_LABELS
        )
        lines.append(f"答案：{answer_text}".rstrip())
        return "\n".join(lines)

    def gold_label(self, doc: dict[str, Any]) -> str:
        # Normalize the gold answer key to one uppercase label token.
        """Implement gold label for cmmlu."""
        return str(doc["Answer"]).strip().upper()

    def sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        # Preserve the evaluated CMMLU subject and source question text for debugging.
        """Implement sample metadata for cmmlu."""
        return {
            "subset": self.subset,
            "question": str(doc["Question"]).strip(),
        }


def cmmlu(*, subset: str, **kwargs: Any) -> CMMLU:
    # Build the generic CMMLU suite while pinning the requested subject as the dataset name.
    """Implement cmmlu for this module."""
    kwargs.setdefault("dataset_name", subset)
    return CMMLU(subset=subset, **kwargs)


def _make_cmmlu_factory(subset: str) -> Any:
    # Emit one import-stable zero-argument factory per CMMLU subject.
    """Make cmmlu factory."""
    def factory(**kwargs: Any) -> CMMLU:
        """Implement factory for this module."""
        return cmmlu(subset=subset, **kwargs)

    factory.__name__ = f"cmmlu_{subset}"
    return factory


# Register all subject-specific CMMLU factories eagerly for import-time discovery.
for _subset in CMMLU_SUBSETS:
    globals()[f"cmmlu_{_subset}"] = _make_cmmlu_factory(_subset)

del _subset
