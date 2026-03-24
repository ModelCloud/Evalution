# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalution.benchmarks.arabic_subject_mmlu import CHOICE_LABELS
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    build_choice_score,
    choice_logprobs,
    choice_logprobs_norm,
    multiple_choice_outcome,
)
from evalution.engines.base import LoglikelihoodRequest

EUS_EXAMS_SUBSETS = tuple(get_dataset_config_names("HiTZ/EusExams"))
EUS_EXAMS_TASKS = tuple(f"eus_exams_{subset}" for subset in EUS_EXAMS_SUBSETS)
_SUBSET_TO_TASK = dict(zip(EUS_EXAMS_SUBSETS, EUS_EXAMS_TASKS, strict=True))


def _eus_exams_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines.extend(
        f"{label}. {choice.strip()}"
        for label, choice in zip(CHOICE_LABELS[: len(choices)], choices, strict=True)
    )
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class EusExams(BaseMultipleChoiceSuite):
    dataset_path: str = "HiTZ/EusExams"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = "eu_opeosakiadmineu"

    def __post_init__(self) -> None:
        if self.subset not in EUS_EXAMS_SUBSETS:
            raise ValueError(f"unsupported eus_exams subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("eus_exams dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(choice).strip() for choice in doc["candidates"]]
        answer_index = int(doc["answer"])
        return MultipleChoiceSample(
            index=index,
            prompt=_eus_exams_prompt(str(doc["question"]), choices),
            choices=list(CHOICE_LABELS[: len(choices)]),
            gold_index=answer_index,
            metadata={
                "subset": self.subset,
                "language": self.subset.split("_", 1)[0],
                "question_id": str(doc["id"]),
                "link": str(doc["link"]),
                "raw_choices": choices,
            },
        )

    def evaluate(self, session: Any) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)
        docs = [doc for doc in docs if doc.get("answer") is not None]

        total = doc_count(
            docs,
            loaded_docs=docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        samples = [self.build_sample(doc, index=index) for index, doc in enumerate(docs)]
        requests: list[LoglikelihoodRequest] = []
        request_to_choice: list[tuple[int, int]] = []
        for sample in samples:
            for choice_index, choice in enumerate(sample.choices):
                requests.append(
                    LoglikelihoodRequest(
                        context=sample.prompt,
                        continuation=self.continuation_for_choice(choice),
                    )
                )
                request_to_choice.append((sample.index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        sample_choice_scores: dict[int, list[Any]] = {}
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            sample_choice_scores.setdefault(sample_index, []).append(
                build_choice_score(
                    choice_index=choice_index,
                    logprob=output.logprob,
                    token_count=output.token_count,
                )
            )

        sample_results: list[SampleResult] = []
        raw_total = 0.0
        norm_total = 0.0
        for sample in samples:
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item.index)
            outcome = multiple_choice_outcome(choice_scores, sample.gold_index)
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=sample.choices[outcome.raw_best_index],
                    extracted={
                        "gold_index": sample.gold_index,
                        "predicted_index": outcome.raw_best_index,
                        "predicted_index_norm": outcome.normalized_best_index,
                    },
                    scores={
                        "acc,ll": outcome.raw_accuracy,
                        "acc,ll_avg": outcome.normalized_accuracy,
                    },
                    metadata={
                        **sample.metadata,
                        "choice_logprobs": choice_logprobs(choice_scores),
                        "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
                    },
                )
            )

        sample_count = len(samples)
        metrics = {
            "acc,ll": raw_total / sample_count,
            "acc,ll_avg": norm_total / sample_count,
        }
        return TestResult(
            name=task_name,
            metrics=metrics,
            metadata=self.result_metadata(),
            samples=sample_results,
        )


def eus_exams(*, subset: str, **kwargs: Any) -> EusExams:
    return EusExams(subset=subset, dataset_name=subset, **kwargs)


def _make_eus_exams_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> EusExams:
        return eus_exams(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in EUS_EXAMS_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_eus_exams_factory(_subset)

del _subset
