# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.choice_label import exact_match

POLEMO2_VARIANTS = ("polemo2_in", "polemo2_out")
_POLEMO2_DATASET_PATHS = {
    "polemo2_in": "allegro/klej-polemo2-in",
    "polemo2_out": "allegro/klej-polemo2-out",
}
_POLEMO2_LABELS = {
    "__label__meta_zero": "A",
    "__label__meta_minus_m": "B",
    "__label__meta_plus_m": "C",
    "__label__meta_amb": "D",
}
_POLEMO2_STOP = (".", ",")
_POLEMO2_LABEL_RE = re.compile(r"\b([ABCD])\b")


def _polemo2_prompt(sentence: str) -> str:
    return (
        f'Opinia: "{sentence}"\n'
        "Określ sentyment podanej opinii. Możliwe odpowiedzi:\n"
        "A - Neutralny\n"
        "B - Negatywny\n"
        "C - Pozytywny\n"
        "D - Niejednoznaczny\n"
        "Prawidłowa odpowiedź:"
    )


def _normalize_polemo2_prediction(prediction: str) -> str:
    candidate = prediction.strip()
    match = _POLEMO2_LABEL_RE.search(candidate)
    if match is not None:
        return match.group(1)
    return ""


@dataclass(slots=True)
class Polemo2(BaseTestSuite):
    # POLEMO2 is evaluated by generating one label token for a four-way sentiment class.
    dataset_path: str = "allegro/klej-polemo2-in"
    dataset_name: str | None = None
    split: str = "test"
    variant: str = "polemo2_in"
    stream: bool = False
    max_new_tokens: int = 50
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in POLEMO2_VARIANTS:
            raise ValueError(f"unsupported polemo2 variant: {self.variant!r}")
        expected_path = _POLEMO2_DATASET_PATHS[self.variant]
        if self.dataset_path != expected_path:
            raise ValueError("polemo2 dataset_path must match the configured variant")
        if self.dataset_name is not None:
            raise ValueError("polemo2 does not use a dataset_name")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.variant

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_choice_label_micro_f1",
            "primary_metric": "f1",
            "variant": self.variant,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            target = _POLEMO2_LABELS[str(doc["target"])]
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_polemo2_prompt(str(doc["sentence"])),
                    stop=list(_POLEMO2_STOP),
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
        normalized_prediction = _normalize_polemo2_prediction(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": normalized_prediction,
                "target-stripped": prepared_sample.target,
            },
            scores={"f1": exact_match(normalized_prediction, prepared_sample.target)},
            metadata={
                "variant": self.variant,
                "sentence": str(prepared_sample.doc["sentence"]),
                "target_label": str(prepared_sample.doc["target"]),
            },
        )


def polemo2(*, variant: str, **kwargs: Any) -> Polemo2:
    if variant not in POLEMO2_VARIANTS:
        raise ValueError(f"unsupported polemo2 variant: {variant!r}")
    return Polemo2(variant=variant, dataset_path=_POLEMO2_DATASET_PATHS[variant], **kwargs)


def polemo2_in(**kwargs: Any) -> Polemo2:
    return polemo2(variant="polemo2_in", **kwargs)


def polemo2_out(**kwargs: Any) -> Polemo2:
    return polemo2(variant="polemo2_out", **kwargs)
