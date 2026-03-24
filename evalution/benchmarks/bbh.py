# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.choice_label import exact_match

BBH_SUBSETS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)
BBH_TASKS = tuple(f"bbh_{subset}" for subset in BBH_SUBSETS)
_STOP_STRINGS = ("\n",)
_ANSWER_PREFIX_RE = re.compile(r"^(?:A:|Answer:|The answer is)\s*", re.IGNORECASE)
_LETTER_RE = re.compile(r"\(([A-Z])\)")


def _bbh_prompt(input_text: str) -> str:
    return f"Q: {input_text.strip()}\nA:"


def _normalize_bbh_prediction(prediction: str, *, target: str) -> str:
    candidate = prediction.strip()
    if "\n" in candidate:
        candidate = candidate.splitlines()[0].strip()
    candidate = _ANSWER_PREFIX_RE.sub("", candidate).strip()
    if target in {"True", "False", "Yes", "No"}:
        match = re.search(r"\b(True|False|Yes|No)\b", candidate)
        if match is not None:
            return match.group(1)
    if re.fullmatch(r"\([A-Z]\)", target):
        match = _LETTER_RE.search(candidate)
        if match is not None:
            return f"({match.group(1)})"
        if re.fullmatch(r"[A-Z]", candidate):
            return f"({candidate})"
    if candidate.endswith(".") and not target.endswith("."):
        candidate = candidate[:-1].rstrip()
    return candidate


@dataclass(slots=True)
class BBH(BaseTestSuite):
    dataset_path: str = "lukaemon/bbh"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = ""
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.subset not in BBH_SUBSETS:
            raise ValueError(f"unsupported bbh subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("bbh dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"bbh_{self.subset}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            target = str(doc["target"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_bbh_prompt(str(doc["input"])),
                    stop=list(_STOP_STRINGS),
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
        normalized_prediction = _normalize_bbh_prediction(output.text, target=prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": normalized_prediction,
                "target-stripped": prepared_sample.target,
            },
            scores={"em": exact_match(normalized_prediction, prepared_sample.target)},
            metadata={
                "subset": self.subset,
                "input": str(prepared_sample.doc["input"]).strip(),
                "target_text": prepared_sample.target,
            },
        )


def bbh(*, subset: str, **kwargs: Any) -> BBH:
    return BBH(subset=subset, dataset_name=subset, **kwargs)


def _make_bbh_subset_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> BBH:
        return bbh(subset=subset, **kwargs)

    factory.__name__ = f"bbh_{subset}"
    return factory


for _subset in BBH_SUBSETS:
    globals()[f"bbh_{_subset}"] = _make_bbh_subset_factory(_subset)

del _subset
