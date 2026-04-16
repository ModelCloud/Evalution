# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.math_exact_match import math_strings_equivalent, remove_boxed, last_boxed_only_string

# Keep benchmark defaults and public task ids explicit at module scope.
# Evalution targets the public text-only mirror because the official HLE release is gated on the Hub.
HLE_DATASET_PATH = "macabdul9/hle_text_only"
_HLE_STOP_STRINGS = ("\n\nQuestion:", "</s>", "<|im_end|>", "<|eot_id|>")
_CHOICE_LABELS = tuple(chr(ord("A") + index) for index in range(10))
_CHOICE_PATTERN = pcre.compile(r"\b([A-J])\b")
_EXPLICIT_CHOICE_PATTERNS = (
    pcre.compile(r"(?i)\bthe correct answer is\s*\(?([A-J])\)?"),
    pcre.compile(r"(?i)\bthe answer is\s*\(?([A-J])\)?"),
    pcre.compile(r"(?i)\bfinal answer\s*[:\-]\s*\(?([A-J])\)?"),
    pcre.compile(r"(?i)\banswer\s*[:\-]\s*\(?([A-J])\)?"),
)
_EXACT_PREFIX_PATTERNS = (
    pcre.compile(r"(?is)\bfinal answer\s*[:\-]\s*(.+)$"),
    pcre.compile(r"(?is)\bthe answer is\s*(.+)$"),
    pcre.compile(r"(?is)\banswer\s*[:\-]\s*(.+)$"),
)
_CODE_FENCE_PATTERN = pcre.compile(r"(?is)```(?:text|markdown)?\s*(.*?)\s*```")
_WHITESPACE_PATTERN = pcre.compile(r"\s+")


def _hle_dataset_loader() -> Callable[..., Any]:
    # Filter the official HLE release down to the text-only rows because Evalution currently exposes text-only runtimes.
    """Implement hle dataset loader for this module."""

    def _loader(
        dataset_path: str,
        *,
        split: str,
        cache_dir: str | None = None,
        streaming: bool = False,
    ) -> Any:
        """Implement loader for this module."""
        docs = load_dataset(
            dataset_path,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        return [
            dict(doc)
            for doc in docs
            if not str(doc.get("image") or "").strip()
        ]

    return _loader


def _hle_prompt(question: str) -> str:
    """Implement hle prompt for this module."""
    return f"{question.strip()}\n\nAnswer:"


def _extract_choice_label(text: str) -> str:
    """Extract hle choice label from a model response."""
    response = text or ""
    for pattern in _EXPLICIT_CHOICE_PATTERNS:
        matches = list(pattern.findall(response))
        if not matches:
            continue
        candidate = str(matches[-1]).strip().upper()
        if candidate in _CHOICE_LABELS:
            return candidate
    matches = list(_CHOICE_PATTERN.findall(response))
    if matches:
        return str(matches[-1]).strip().upper()
    return ""


def _extract_short_answer(text: str) -> str:
    """Extract the short-answer payload from a model response."""
    response = (text or "").strip()
    fenced_match = _CODE_FENCE_PATTERN.search(response)
    if fenced_match:
        response = str(fenced_match.group(1)).strip()
    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            return remove_boxed(boxed_answer).strip()
        except (AssertionError, IndexError):
            pass
    for pattern in _EXACT_PREFIX_PATTERNS:
        match = pattern.search(response)
        if match:
            response = str(match.group(1)).strip()
            break
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    candidate = lines[-1] if lines else response
    return candidate.strip().strip("`\"'")


def _normalize_hle_text(text: str) -> str:
    """Normalize HLE text-only answers with conservative whitespace folding."""
    return _WHITESPACE_PATTERN.sub(" ", text.strip().strip("`\"'").lower()).strip()


def _hle_exact_match(prediction: str, target: str) -> float:
    """Score one HLE short answer prediction against the reference answer."""
    extracted_prediction = _extract_short_answer(prediction)
    if math_strings_equivalent(extracted_prediction, target):
        return 1.0
    return float(_normalize_hle_text(extracted_prediction) == _normalize_hle_text(target))


@dataclass(slots=True)
class HLE(BaseTestSuite):
    """Implement the text-only Humanity's Last Exam benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = HLE_DATASET_PATH
    dataset_name: str | None = None
    split: str = "test"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _hle_dataset_loader()

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "hle"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "modality_subset": "text_only",
            "source_benchmark": "cais/hle",
            "scoring_mode": "generated_hle_answer_accuracy",
            "primary_metric": "acc",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["answer"]).strip(),
                request=GenerationRequest(
                    prompt=_hle_prompt(str(doc["question"])),
                    stop=list(_HLE_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        answer_type = str(prepared_sample.doc["answer_type"]).strip()
        extracted: dict[str, str] = {}
        if answer_type == "multipleChoice":
            extracted_label = _extract_choice_label(output.text)
            extracted["choice-label"] = extracted_label
            extracted["answer-extract"] = extracted_label
            score = float(extracted_label == prepared_sample.target)
        else:
            extracted_answer = _extract_short_answer(output.text)
            extracted["answer-extract"] = extracted_answer
            score = _hle_exact_match(output.text, prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted=extracted,
            scores={"acc": score},
            metadata={
                "id": str(prepared_sample.doc["id"]),
                "answer_type": answer_type,
                "raw_subject": str(prepared_sample.doc["raw_subject"]).strip(),
                "category": str(prepared_sample.doc["category"]).strip(),
                "author_name": str(prepared_sample.doc["author_name"]).strip(),
                "text_only": True,
            },
        )


def hle(**kwargs: Any) -> HLE:
    """Implement hle for this module."""
    return HLE(**kwargs)
