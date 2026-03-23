# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.suites.data import doc_count, limit_docs, load_suite_dataset
from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.suites.multiple_choice_utils import choice_index_from_labels, question_answer_prompt


@dataclass(slots=True)
class BaseARCExamSuite(BaseMultipleChoiceSuite):
    # Reference scoring rule: Clark et al. (2018), "Think you have Solved Question Answering?
    # Try ARC, the AI2 Reasoning Challenge", plus the released ARC solver scorer
    # `arc_solvers/processing/calculate_scores.py`.
    #
    # The original ARC exam score treats each question as a multiple-choice item: select all
    # top-scoring answer options and award 1/k credit when the gold answer appears in a k-way tie.
    # We apply that same tie-aware rule to the model's per-choice log-likelihood scores here.
    dataset_path: str = "allenai/ai2_arc"

    def dataset_loader(self) -> Any:
        return load_dataset

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        labels = list(doc["choices"]["label"])
        texts = list(doc["choices"]["text"])
        return MultipleChoiceSample(
            index=index,
            prompt=question_answer_prompt(doc["question"]),
            choices=texts,
            gold_index=choice_index_from_labels(labels, doc["answerKey"]),
            metadata={"id": doc["id"], "choice_labels": labels},
        )

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
        }

    def evaluate(self, session: InferenceSession) -> TestResult:
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

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
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

        sample_choice_scores: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            sample_choice_scores[sample_index].append((output.logprob, choice_index))

        sample_results: list[SampleResult] = []
        total_score = 0.0
        for sample in samples:
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item[1])
            max_choice_score = max(score for score, _choice_index in choice_scores)
            selected_indices = [
                choice_index
                for score, choice_index in choice_scores
                if score == max_choice_score
            ]
            question_score = (
                1.0 / len(selected_indices)
                if sample.gold_index in selected_indices
                else 0.0
            )
            total_score += question_score
            choice_labels = sample.metadata["choice_labels"]
            selected_labels = [choice_labels[index] for index in selected_indices]
            selected_texts = [sample.choices[index] for index in selected_indices]
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=" | ".join(selected_texts),
                    extracted={
                        "gold_index": str(sample.gold_index),
                        "selected_indices": ",".join(str(index) for index in selected_indices),
                        "selected_labels": ",".join(selected_labels),
                    },
                    scores={"accuracy,exam_score": question_score},
                    metadata={
                        **sample.metadata,
                        "choice_logprobs": [score for score, _choice_index in choice_scores],
                        "selected_count": len(selected_indices),
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {"accuracy,exam_score": total_score / denominator}
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
