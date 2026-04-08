# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, GSM8KVariant, VariantSpec

# Freeze the supported MGSM language list so imports do not depend on live task discovery.
MGSM_LANGUAGES = (
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "ja",
    "ru",
    "sw",
    "te",
    "th",
    "zh",
)
MGSM_TASKS = tuple(f"mgsm_direct_{language}" for language in MGSM_LANGUAGES)


def _load_mgsm_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Any:
    # Resolve the language-specific MGSM TSV and adapt it to the datasets CSV reader.
    if dataset_path != "juletxara/mgsm":
        raise ValueError(f"unsupported MGSM dataset path: {dataset_path!r}")
    if dataset_name not in MGSM_LANGUAGES:
        raise ValueError(f"unsupported MGSM language: {dataset_name!r}")
    if split != "test":
        raise ValueError(f"unsupported MGSM split: {split!r}")
    file_path = hf_hub_download(
        repo_id=dataset_path,
        filename=f"mgsm_{dataset_name}.tsv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return load_dataset(
        "csv",
        data_files={split: file_path},
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
        delimiter="\t",
        column_names=["question", "answer_number"],
    )


def _mgsm_prompt(doc: dict[str, Any]) -> str:
    # Match the direct-answer MGSM prompt shape used by the lm-eval task configuration.
    return f"Question: {str(doc['question']).strip()}\nAnswer:"


def _mgsm_target(doc: dict[str, Any]) -> str:
    # Normalize numeric answers to the compact string form used by the GSM-style scorer.
    value = doc["answer_number"]
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_DIRECT_VARIANTS = {
    # Keep MGSM aligned with the zero-shot direct-answer variant rather than the CoT GSM8K defaults.
    "base": VariantSpec(
        task_name="mgsm",
        stop_strings=("\n\n", "\n"),
        prompt_builder=_mgsm_prompt,
        target_builder=_mgsm_target,
        num_fewshot=0,
        fewshots=(),
    )
}


@dataclass(slots=True)
class MGSM(BaseGSM8KSuite):
    # Keep MGSM aligned with the direct numeric-extraction task that ships in the lm-eval package.
    VARIANTS = _DIRECT_VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    dataset_path: str = "juletxara/mgsm"
    dataset_name: str | None = "en"
    split: str = "test"
    language: str = "en"
    variant: GSM8KVariant = "default"

    def __post_init__(self) -> None:
        # Restrict MGSM to the published direct-answer language set and one matching dataset name.
        if self.language not in MGSM_LANGUAGES:
            raise ValueError(f"unsupported MGSM language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
        else:
            raise ValueError("mgsm dataset_name must match the configured language")
        if self.variant not in {"default", "base"}:
            raise ValueError("mgsm only supports the direct base variant")

    def dataset_loader(self) -> Any:
        # Route MGSM through the language-specific TSV loader above.
        return _load_mgsm_dataset

    def task_name(self) -> str:
        # Expose one stable task name per direct MGSM language factory.
        return f"mgsm_direct_{self.language}"

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        # Reuse the shared numeric normalization helper for benchmark targets.
        return _mgsm_target(doc)

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        # Add the resolved MGSM language to the shared GSM-style result metadata.
        metadata = super().result_metadata(generation_submission_mode=generation_submission_mode)
        metadata["language"] = self.language
        return metadata

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        # Preserve the source question and normalized numeric answer for debugging.
        return {
            "language": self.language,
            "question": str(doc["question"]).strip(),
            "answer_number": _mgsm_target(doc),
        }


def mgsm(*, language: str, **kwargs: Any) -> MGSM:
    # Build the generic MGSM suite while pinning the requested language as the dataset name.
    kwargs.setdefault("dataset_name", language)
    return MGSM(language=language, **kwargs)


def _make_mgsm_factory(language: str) -> Any:
    # Emit one import-stable zero-argument factory per MGSM language.
    def factory(**kwargs: Any) -> MGSM:
        return mgsm(language=language, **kwargs)

    factory.__name__ = f"mgsm_direct_{language}"
    return factory


# Register all language-specific MGSM factories eagerly for import-time discovery.
for _language in MGSM_LANGUAGES:
    globals()[f"mgsm_direct_{_language}"] = _make_mgsm_factory(_language)

del _language
