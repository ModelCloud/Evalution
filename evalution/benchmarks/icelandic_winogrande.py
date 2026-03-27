# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    ChoiceScore,
    build_choice_score,
    choice_logprobs,
    choice_logprobs_norm,
    multiple_choice_outcome,
)

_ANSWER_TO_INDEX = {"1": 0, "2": 1}


def _blank_choice_contexts_and_suffix(
    sentence: str,
    option1: str,
    option2: str,
) -> tuple[list[str], str]:
    blank_index = sentence.index("_")
    suffix = sentence[blank_index + 1 :]
    prefix = sentence[:blank_index]
    return [f"{prefix}{option1}", f"{prefix}{option2}"], suffix


@dataclass(slots=True)
class IcelandicWinoGrande:
    dataset_path: str = "mideind/icelandic-winogrande"
    split: str = "train"
    stream: bool = True
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "icelandic_winogrande"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": None,
            "split": self.split,
            "stream": self.stream,
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        }

    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=None,
            split=self.split,
            cache_dir=self.cache_dir,
            stream=self.stream,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        requests: list[LoglikelihoodRequest] = []
        request_to_choice: list[tuple[int, int]] = []
        sample_payloads: list[dict[str, Any]] = []
        for index, doc in enumerate(docs):
            sentence = str(doc["sentence"])
            option1 = str(doc["option1"])
            option2 = str(doc["option2"])
            answer_label = str(doc["answer"])
            choice_contexts, suffix = _blank_choice_contexts_and_suffix(
                sentence,
                option1,
                option2,
            )
            completed_choices = [f"{context}{suffix}" for context in choice_contexts]
            sample_payloads.append(
                {
                    "index": index,
                    "prompt": sentence,
                    "gold_index": _ANSWER_TO_INDEX[answer_label],
                    "completed_choices": completed_choices,
                    "metadata": {
                        "qID": str(doc["qID"]),
                        "sentence": sentence,
                        "answer_label": answer_label,
                        "choice_labels": ["A", "B"],
                        "choice_texts": [option1, option2],
                        "target_suffix": suffix,
                        "blank_index": sentence.index("_"),
                    },
                }
            )
            for choice_index, choice_context in enumerate(choice_contexts):
                requests.append(
                    LoglikelihoodRequest(
                        context=choice_context,
                        continuation=suffix,
                    )
                )
                request_to_choice.append((index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_payloads), total)

        sample_choice_scores: dict[int, list[ChoiceScore]] = defaultdict(list)
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            sample_choice_scores[sample_index].append(
                build_choice_score(
                    choice_index=choice_index,
                    logprob=output.logprob,
                    token_count=output.token_count,
                )
            )

        sample_results: list[SampleResult] = []
        raw_total = 0.0
        norm_total = 0.0
        for sample_payload in sample_payloads:
            choice_scores = sorted(
                sample_choice_scores[sample_payload["index"]],
                key=lambda item: item.index,
            )
            outcome = multiple_choice_outcome(choice_scores, sample_payload["gold_index"])
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            sample_results.append(
                SampleResult(
                    index=sample_payload["index"],
                    prompt=sample_payload["prompt"],
                    target=sample_payload["completed_choices"][sample_payload["gold_index"]],
                    prediction=sample_payload["completed_choices"][outcome.normalized_best_index],
                    extracted={
                        "gold_index": str(sample_payload["gold_index"]),
                        "predicted_index": str(outcome.raw_best_index),
                        "predicted_index_norm": str(outcome.normalized_best_index),
                    },
                    scores={
                        "acc,ll": outcome.raw_accuracy,
                        "acc,ll_avg": outcome.normalized_accuracy,
                    },
                    metadata={
                        **sample_payload["metadata"],
                        "choice_logprobs": choice_logprobs(choice_scores),
                        "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        return TestResult(
            name=task_name,
            metrics={
                "acc,ll": raw_total / denominator,
                "acc,ll_avg": norm_total / denominator,
            },
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def icelandic_winogrande(**kwargs: Any) -> IcelandicWinoGrande:
    return IcelandicWinoGrande(**kwargs)
