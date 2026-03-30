# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.execution import PreparedSample
from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, VariantSpec
from evalution.engines.base import GenerationOutput
from evalution.results import SampleResult
from evalution.scorers.choice_label import exact_match
from evalution.scorers.gsm8k import INVALID_ANSWER

GSM_PLUS_TASKS = ("gsm_plus", "gsm_plus_mini")
_STRICT_MATCH_RE = pcre.compile(r"#### (\-?[0-9\.,]+)")
_FLEXIBLE_EXTRACT_RE = pcre.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_TARGET_PREFIX_RE = pcre.compile(r"(?s).*#### ")
_TRAILING_PERIOD_RE = pcre.compile(r"\.$")


def _gsm_plus_prompt(doc: dict[str, Any]) -> str:
    return f"Question: {doc['question']}\nAnswer:"


def _gsm_plus_target(doc: dict[str, Any]) -> str:
    return str(doc["solution"])


@dataclass(slots=True)
class _BaseGSMPlusSuite(BaseGSM8KSuite):
    # GSM-Plus reuses GSM8K prompting, but matches lm-eval's strict/flexible answer extraction.
    variant: str = "base"
    stream: bool = True
    SCORING_MODE = "generated_regex_extract_exact_match"
    PRIMARY_METRIC = "em,strict"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return load_dataset

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        _, spec = self._resolved_variant()
        fewshot_docs = list(docs) if self.requires_full_doc_materialization() else list(spec.fewshots)
        fewshot_as_multiturn = self._resolved_fewshot_as_multiturn()
        rng = random.Random(self.fewshot_seed)
        for index, doc in enumerate(docs):
            fewshots = self._select_fewshots_like_lm_eval(
                spec=spec,
                docs=fewshot_docs,
                doc=doc,
                rng=rng,
            )
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["solution"]),
                request=self._build_request(
                    spec=spec,
                    doc=doc,
                    fewshots=fewshots,
                    fewshot_as_multiturn=fewshot_as_multiturn,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        strict_prediction = _extract_strict_match(output.text)
        flexible_prediction = _extract_flexible_match(output.text)
        normalized_target = _normalize_gsm_plus_exact_match(prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "strict-match": strict_prediction,
                "flexible-extract": flexible_prediction,
                "target-stripped": normalized_target,
            },
            scores={
                "em,strict": exact_match(
                    _normalize_gsm_plus_exact_match(strict_prediction),
                    normalized_target,
                ),
                "em,flex": exact_match(
                    _normalize_gsm_plus_exact_match(flexible_prediction),
                    normalized_target,
                ),
            },
            metadata=self._sample_metadata(prepared_sample.doc),
        )

    def invalid_prediction_count(self, sample: SampleResult) -> int:
        return int(sample.extracted["flexible-extract"] == INVALID_ANSWER)

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "perturbation_type": str(doc.get("perturbation_type", "")),
            "seed_question": str(doc.get("seed_question", "")),
            "seed_solution": str(doc.get("seed_solution", "")),
            "seed_answer": str(doc.get("seed_answer", "")),
        }

    def _select_fewshots_like_lm_eval(
        self,
        *,
        spec: VariantSpec,
        docs: list[dict[str, Any]],
        doc: dict[str, Any],
        rng: random.Random,
    ) -> list[dict[str, str]]:
        if spec.fewshots:
            return list(spec.fewshots[: spec.num_fewshot])
        if spec.num_fewshot == 0:
            return []

        sampled = rng.sample(docs, k=min(spec.num_fewshot + 1, len(docs)))
        sampled = [candidate for candidate in sampled if candidate != doc][: spec.num_fewshot]
        return [
            {
                "question": str(candidate["question"]),
                "target": spec.target_builder(candidate),
            }
            for candidate in sampled
        ]


def _normalize_gsm_plus_exact_match(text: str) -> str:
    normalized = text.strip().lower().replace(",", "").replace("$", "")
    normalized = _TARGET_PREFIX_RE.sub("", normalized)
    normalized = _TRAILING_PERIOD_RE.sub("", normalized)
    return normalized.strip()


def _extract_strict_match(text: str) -> str:
    match = _STRICT_MATCH_RE.search(text)
    if match is None:
        return INVALID_ANSWER
    return match.group(1)


def _extract_flexible_match(text: str) -> str:
    matches = _FLEXIBLE_EXTRACT_RE.findall(text)
    if not matches:
        return INVALID_ANSWER
    match = matches[-1]
    if isinstance(match, tuple):
        for candidate in match:
            if candidate:
                return candidate
        return INVALID_ANSWER
    return match


@dataclass(slots=True)
class GSMPlus(_BaseGSMPlusSuite):
    VARIANTS = {
        "base": VariantSpec(
            task_name="gsm_plus",
            stop_strings=("Question:", "</s>", "<|im_end|>"),
            prompt_builder=_gsm_plus_prompt,
            target_builder=_gsm_plus_target,
            num_fewshot=5,
            fewshots=(),
        )
    }
    dataset_path: str = "qintongli/GSM-Plus"
    dataset_name: str | None = None
    split: str = "test"


@dataclass(slots=True)
class GSMPlusMini(_BaseGSMPlusSuite):
    VARIANTS = {
        "base": VariantSpec(
            task_name="gsm_plus_mini",
            stop_strings=("Question:", "</s>", "<|im_end|>"),
            prompt_builder=_gsm_plus_prompt,
            target_builder=_gsm_plus_target,
            num_fewshot=5,
            fewshots=(),
        )
    }
    dataset_path: str = "qintongli/GSM-Plus"
    dataset_name: str | None = None
    split: str = "testmini"


def gsm_plus(**kwargs: Any) -> GSMPlus:
    return GSMPlus(**kwargs)


def gsm_plus_mini(**kwargs: Any) -> GSMPlusMini:
    return GSMPlusMini(**kwargs)
