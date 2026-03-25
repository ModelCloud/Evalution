# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
import pcre

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.choice_label import exact_match

# FLD predicts one of three world-assumption verdict labels from serialized deduction prompts.
FLD_LABELS = ("PROVED", "DISPROVED", "UNKNOWN")
# Match the lm-eval stopping behavior so generation can continue past single newlines.
_STOP_STRINGS = ("\n\n",)
_WHITESPACE_RE = pcre.compile(r"\s+")


def _fld_prompt(doc: dict[str, Any]) -> str:
    return (
        "Based on the provided facts ($context$), either prove or disprove the hypothesis "
        f"or state that it is unknown. {str(doc['prompt_serial']).strip()}"
    )


def _normalize_label(text: str) -> str:
    return _WHITESPACE_RE.sub("", text).upper()


@dataclass(slots=True)
class FLD(BaseTestSuite):
    # FLD uses direct generation with exact-match label scoring on the world assumption verdict.
    dataset_path: str = "hitachi-nlp/FLD.v2"
    dataset_name: str | None = "default"
    split: str = "test"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

    def task_name(self) -> str:
        return "fld"

    def dataset_loader(self) -> Any:
        return load_dataset

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            target = str(doc["world_assump_label"]).strip().upper()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_fld_prompt(doc),
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        normalized_prediction = _normalize_label(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": normalized_prediction,
                "target-stripped": _normalize_label(prepared_sample.target),
            },
            scores={"em": exact_match(normalized_prediction, _normalize_label(prepared_sample.target))},
            metadata={
                "proof_label": str(prepared_sample.doc["proof_label"]),
                "world_assump_label": prepared_sample.target,
                "negative_world_assump_label": str(prepared_sample.doc["negative_world_assump_label"]),
                "num_formula_distractors": int(prepared_sample.doc["num_formula_distractors"]),
                "num_translation_distractors": int(prepared_sample.doc["num_translation_distractors"]),
                "num_all_distractors": int(prepared_sample.doc["num_all_distractors"]),
            },
        )


def fld(**kwargs: Any) -> FLD:
    return FLD(**kwargs)
