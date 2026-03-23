# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, GSM8KVariant
from evalution.benchmarks.gsm8k_common import build_variant_specs as _build_variant_specs
from evalution.benchmarks.single_continuation import (
    BaseSingleContinuationSuite,
    SingleContinuationSample,
)


def _asdiv_numeric_target(doc: dict[str, Any]) -> str:
    answer = str(doc["answer"])
    return answer.split(" (", 1)[0].strip()


def _asdiv_prompt(doc: dict[str, Any]) -> str:
    return f"{str(doc['body'])}\nQuestion:{str(doc['question'])}\nAnswer:"


def _asdiv_llama_prompt(doc: dict[str, Any]) -> str:
    body = str(doc.get("body", "")).strip()
    question = str(doc["question"]).strip()
    problem = f"{body} {question}".strip()
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {problem}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


_ASDIV_VARIANTS = _build_variant_specs("asdiv")
_ASDIV_COT_LLAMA_VARIANTS = {
    "cot_llama": replace(
        _ASDIV_VARIANTS["cot_llama"],
        prompt_builder=_asdiv_llama_prompt,
        target_builder=_asdiv_numeric_target,
    )
}


@dataclass(slots=True)
class ASDiv(BaseSingleContinuationSuite):
    dataset_path: str = "EleutherAI/asdiv"
    dataset_name: str | None = None
    split: str = "validation"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "asdiv"

    def continuation_for_target(self, target: str) -> str:
        return target

    def include_perplexity(self) -> bool:
        return False

    def build_sample(self, doc: dict[str, Any], *, index: int) -> SingleContinuationSample:
        return SingleContinuationSample(
            index=index,
            prompt=_asdiv_prompt(doc),
            target=_asdiv_numeric_target(doc),
            metadata={
                "body": str(doc["body"]),
                "question": str(doc["question"]),
                "answer": str(doc["answer"]),
                "solution_type": str(doc["solution_type"]),
                "formula": str(doc["formula"]),
            },
        )


@dataclass(slots=True)
class ASDivCoTLlama(BaseGSM8KSuite):
    VARIANTS = _ASDIV_COT_LLAMA_VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    dataset_path: str = "EleutherAI/asdiv"
    dataset_name: str | None = None
    split: str = "validation"
    variant: GSM8KVariant = "cot_llama"

    def dataset_loader(self) -> Any:
        return load_dataset

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        return _asdiv_numeric_target(doc)

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "solution_type": str(doc["solution_type"]),
            "formula": str(doc["formula"]),
        }


def asdiv(**kwargs: Any) -> ASDiv:
    return ASDiv(**kwargs)


def asdiv_cot_llama(**kwargs: Any) -> ASDivCoTLlama:
    return ASDivCoTLlama(**kwargs)
