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

# Freeze the contamination-free subject list so the new PR-backed suite stays import-stable.
MMLU_CF_SUBJECTS = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "miscellaneous",
    "philosophy",
    "physics",
    "psychology",
)
MMLU_CF_TASKS = tuple(f"mmlu_cf_{subject}" for subject in MMLU_CF_SUBJECTS)
# Keep the subject-to-file-prefix mapping explicit because the dataset stores each split as a separate parquet file.
MMLU_CF_FILE_PREFIXES = {
    "biology": "Biology",
    "business": "Business",
    "chemistry": "Chemistry",
    "computer_science": "Computer_Science",
    "economics": "Economics",
    "engineering": "Engineering",
    "health": "Health",
    "history": "History",
    "law": "Law",
    "math": "Math",
    "miscellaneous": "Other",
    "philosophy": "Philosophy",
    "physics": "Physics",
    "psychology": "Psychology",
}
# Keep the MMLU-CF answer-label convention explicit for prompt rendering and scoring.
_MMLU_CF_LABELS = ("A", "B", "C", "D")
# Mirror the prompt instruction described in the open MMLU-CF task PR.
_MMLU_CF_DESCRIPTION = (
    "There is a single choice question (with answers). "
    "Answer the question by replying A, B, C or D."
)


def _load_mmlu_cf_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool | None = None,
) -> Any:
    # Resolve the published parquet shard for one MMLU-CF subject split.
    """Load MMLU cf dataset."""
    if dataset_path != "microsoft/MMLU-CF":
        raise ValueError(f"unsupported MMLU-CF dataset path: {dataset_path!r}")
    if dataset_name not in MMLU_CF_FILE_PREFIXES:
        raise ValueError(f"unsupported MMLU-CF subject: {dataset_name!r}")
    if split not in {"dev", "val"}:
        raise ValueError(f"unsupported MMLU-CF split: {split!r}")
    effective_stream = False if stream is None else stream
    file_prefix = MMLU_CF_FILE_PREFIXES[dataset_name]
    file_path = hf_hub_download(
        repo_id=dataset_path,
        filename=f"{split}/{file_prefix}_{split}.parquet",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return load_dataset(
        "parquet",
        data_files={split: file_path},
        split=split,
        cache_dir=cache_dir,
        streaming=effective_stream,
    )


@dataclass(slots=True)
class MMLUCF(BaseFewshotMultipleChoiceSuite):
    # Match the open PR's task shape by scoring the public validation rows with dev few-shots.
    """Define the mmlucf helper class."""
    dataset_path: str = "microsoft/MMLU-CF"
    dataset_name: str | None = None
    split: str = "val"
    fewshot_split: str = "dev"
    subject: str = ""

    def __post_init__(self) -> None:
        # Keep the public subject slug and the dataset name locked together for safety.
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subject not in MMLU_CF_SUBJECTS:
            raise ValueError(f"unsupported MMLU-CF subject: {self.subject!r}")
        if self.dataset_name in {None, self.subject}:
            self.dataset_name = self.subject
            return
        raise ValueError("mmlu_cf dataset_name must match the configured subject")

    def dataset_loader(self) -> Any:
        # Route MMLU-CF through the parquet loader above.
        """Return the dataset loader bound to this suite."""
        return _load_mmlu_cf_dataset

    def task_name(self) -> str:
        # Expose one stable task name per MMLU-CF subject factory.
        """Return the exported task name for this suite."""
        return f"mmlu_cf_{self.subject}"

    def prompt_description(self) -> str:
        # Prefix each evaluation prompt with the benchmark's contamination-free instruction.
        """Implement prompt description for mmlucf."""
        return _MMLU_CF_DESCRIPTION

    def format_question(self, doc: dict[str, Any], *, include_answer: bool) -> str:
        # Render one MMLU-CF question block in the benchmark's A/B/C/D layout.
        """Format question."""
        answer_text = str(doc["Answer"]).strip() if include_answer else ""
        lines = [str(doc["Question"]).strip()]
        lines.extend(
            f"{label}. {str(doc[label]).strip()}"
            for label in _MMLU_CF_LABELS
        )
        lines.append(f"Answer: {answer_text}".rstrip())
        return "\n".join(lines)

    def gold_label(self, doc: dict[str, Any]) -> str:
        # Normalize the gold answer key to one uppercase label token.
        """Implement gold label for mmlucf."""
        return str(doc["Answer"]).strip().upper()

    def sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        # Preserve the evaluated subject and source question text for debugging.
        """Implement sample metadata for mmlucf."""
        return {
            "subject": self.subject,
            "question": str(doc["Question"]).strip(),
        }


def mmlu_cf(*, subject: str, **kwargs: Any) -> MMLUCF:
    # Build the generic MMLU-CF suite while pinning the requested subject as the dataset name.
    """Implement MMLU cf for this module."""
    kwargs.setdefault("dataset_name", subject)
    return MMLUCF(subject=subject, **kwargs)


def _make_mmlu_cf_factory(subject: str) -> Any:
    # Emit one import-stable zero-argument factory per MMLU-CF subject.
    """Make MMLU cf factory."""
    def factory(**kwargs: Any) -> MMLUCF:
        """Implement factory for this module."""
        return mmlu_cf(subject=subject, **kwargs)

    factory.__name__ = f"mmlu_cf_{subject}"
    return factory


# Register all subject-specific MMLU-CF factories eagerly for import-time discovery.
for _subject in MMLU_CF_SUBJECTS:
    globals()[f"mmlu_cf_{_subject}"] = _make_mmlu_cf_factory(_subject)

del _subject
