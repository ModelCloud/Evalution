# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.fewshot_multiple_choice import BaseFewshotMultipleChoiceSuite

# Freeze the KMMLU subset registry so dynamic task factories do not depend on live Hub metadata.
KMMLU_SUBSETS = (
    "accounting",
    "agricultural_sciences",
    "aviation_engineering_and_maintenance",
    "biology",
    "chemical_engineering",
    "chemistry",
    "civil_engineering",
    "computer_science",
    "construction",
    "criminal_law",
    "ecology",
    "economics",
    "education",
    "electrical_engineering",
    "electronics_engineering",
    "energy_management",
    "environmental_science",
    "fashion",
    "food_processing",
    "gas_technology_and_engineering",
    "geomatics",
    "health",
    "industrial_engineer",
    "information_technology",
    "interior_architecture_and_design",
    "korean_history",
    "law",
    "machine_design_and_manufacturing",
    "management",
    "maritime_engineering",
    "marketing",
    "materials_engineering",
    "math",
    "mechanical_engineering",
    "nondestructive_testing",
    "patent",
    "political_science_and_sociology",
    "psychology",
    "public_safety",
    "railway_and_automotive_engineering",
    "real_estate",
    "refrigerating_machinery",
    "social_welfare",
    "taxation",
    "telecommunications_and_wireless_technology",
)
KMMLU_TASKS = tuple(f"kmmlu_{subset}" for subset in KMMLU_SUBSETS)
# Keep the task-name to upstream CSV-file mapping explicit because the dataset uses title-cased filenames.
KMMLU_DATASET_NAMES = {
    "accounting": "Accounting",
    "agricultural_sciences": "Agricultural-Sciences",
    "aviation_engineering_and_maintenance": "Aviation-Engineering-and-Maintenance",
    "biology": "Biology",
    "chemical_engineering": "Chemical-Engineering",
    "chemistry": "Chemistry",
    "civil_engineering": "Civil-Engineering",
    "computer_science": "Computer-Science",
    "construction": "Construction",
    "criminal_law": "Criminal-Law",
    "ecology": "Ecology",
    "economics": "Economics",
    "education": "Education",
    "electrical_engineering": "Electrical-Engineering",
    "electronics_engineering": "Electronics-Engineering",
    "energy_management": "Energy-Management",
    "environmental_science": "Environmental-Science",
    "fashion": "Fashion",
    "food_processing": "Food-Processing",
    "gas_technology_and_engineering": "Gas-Technology-and-Engineering",
    "geomatics": "Geomatics",
    "health": "Health",
    "industrial_engineer": "Industrial-Engineer",
    "information_technology": "Information-Technology",
    "interior_architecture_and_design": "Interior-Architecture-and-Design",
    "korean_history": "Korean-History",
    "law": "Law",
    "machine_design_and_manufacturing": "Machine-Design-and-Manufacturing",
    "management": "Management",
    "maritime_engineering": "Maritime-Engineering",
    "marketing": "Marketing",
    "materials_engineering": "Materials-Engineering",
    "math": "Math",
    "mechanical_engineering": "Mechanical-Engineering",
    "nondestructive_testing": "Nondestructive-Testing",
    "patent": "Patent",
    "political_science_and_sociology": "Political-Science-and-Sociology",
    "psychology": "Psychology",
    "public_safety": "Public-Safety",
    "railway_and_automotive_engineering": "Railway-and-Automotive-Engineering",
    "real_estate": "Real-Estate",
    "refrigerating_machinery": "Refrigerating-Machinery",
    "social_welfare": "Social-Welfare",
    "taxation": "Taxation",
    "telecommunications_and_wireless_technology": "Telecommunications-and-Wireless-Technology",
}
# Keep the KMMLU answer-label convention explicit for prompt rendering and scoring.
_KMMLU_LABELS = ("A", "B", "C", "D")


def _load_kmmlu_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Any:
    # Resolve the subject CSV from Hugging Face and hand it to the standard datasets CSV loader.
    if dataset_path != "HAERAE-HUB/KMMLU":
        raise ValueError(f"unsupported KMMLU dataset path: {dataset_path!r}")
    if dataset_name not in KMMLU_DATASET_NAMES.values():
        raise ValueError(f"unsupported KMMLU dataset_name: {dataset_name!r}")
    if split not in {"dev", "test"}:
        raise ValueError(f"unsupported KMMLU split: {split!r}")
    file_path = hf_hub_download(
        repo_id=dataset_path,
        filename=f"data/{dataset_name}-{split}.csv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return load_dataset(
        "csv",
        data_files={split: file_path},
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
    )


@dataclass(slots=True)
class KMMLU(BaseFewshotMultipleChoiceSuite):
    # Use the upstream dev split as few-shot context so KMMLU matches the benchmark's default setup.
    dataset_path: str = "HAERAE-HUB/KMMLU"
    dataset_name: str | None = None
    split: str = "test"
    fewshot_split: str = "dev"
    subset: str = ""

    def __post_init__(self) -> None:
        # Keep the public subset slug and the upstream title-cased filename mapping in sync.
        if self.subset not in KMMLU_SUBSETS:
            raise ValueError(f"unsupported KMMLU subset: {self.subset!r}")
        expected_dataset_name = KMMLU_DATASET_NAMES[self.subset]
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("kmmlu dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        # Route KMMLU through the subject-CSV loader above.
        return _load_kmmlu_dataset

    def task_name(self) -> str:
        # Expose one stable task name per KMMLU subject factory.
        return f"kmmlu_{self.subset}"

    def format_question(self, doc: dict[str, Any], *, include_answer: bool) -> str:
        # Render one KMMLU question block in the benchmark's standard Korean answer layout.
        answer_index = int(doc["answer"]) - 1
        answer_text = _KMMLU_LABELS[answer_index] if include_answer else ""
        lines = [str(doc["question"]).strip()]
        lines.extend(
            f"{label}. {str(doc[label]).strip()}"
            for label in _KMMLU_LABELS
        )
        lines.append(f"정답：{answer_text}".rstrip())
        return "\n".join(lines)

    def gold_label(self, doc: dict[str, Any]) -> str:
        # Convert the dataset's one-based answer index to the corresponding label token.
        return _KMMLU_LABELS[int(doc["answer"]) - 1]

    def sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        # Preserve the evaluated KMMLU subject metadata that helps explain failures later.
        return {
            "subset": self.subset,
            "category": str(doc["Category"]).strip(),
            "question": str(doc["question"]).strip(),
            "human_accuracy": float(doc["Human Accuracy"]),
        }


def kmmlu(*, subset: str, **kwargs: Any) -> KMMLU:
    # Build the generic KMMLU suite while pinning the mapped upstream dataset name.
    if subset not in KMMLU_DATASET_NAMES:
        raise ValueError(f"unsupported KMMLU subset: {subset!r}")
    kwargs.setdefault("dataset_name", KMMLU_DATASET_NAMES[subset])
    return KMMLU(subset=subset, **kwargs)


def _make_kmmlu_factory(subset: str) -> Any:
    # Emit one import-stable zero-argument factory per KMMLU subject.
    def factory(**kwargs: Any) -> KMMLU:
        return kmmlu(subset=subset, **kwargs)

    factory.__name__ = f"kmmlu_{subset}"
    return factory


# Register all subject-specific KMMLU factories eagerly for import-time discovery.
for _subset in KMMLU_SUBSETS:
    globals()[f"kmmlu_{_subset}"] = _make_kmmlu_factory(_subset)

del _subset
