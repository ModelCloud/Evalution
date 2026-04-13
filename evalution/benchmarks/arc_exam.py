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
from evalution.logbar import get_logger, loglikelihood_progress_metadata
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    ChoiceScore,
    build_choice_score,
    choice_logprobs,
    exam_score_outcome,
)
from evalution.benchmarks.data import apply_order, doc_count, limit_docs, load_suite_dataset, normalize_order
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels, question_answer_prompt


@dataclass(slots=True)
class BaseARCExamSuite(BaseMultipleChoiceSuite):
    # Reference scoring rule: Clark et al. (2018), "Think you have Solved Question Answering?
    # Try ARC, the AI2 Reasoning Challenge", plus the released ARC solver scorer
    # `arc_solvers/processing/calculate_scores.py`.
    #
    # The original ARC exam score treats each question as a multiple-choice item: select all
    # top-scoring answer options and award 1/k credit when the gold answer appears in a k-way tie.
    # We apply that same tie-aware rule to the model's per-choice log-likelihood scores here.
    """Implement the base arcexam suite benchmark suite."""
    dataset_path: str = "allenai/ai2_arc"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
        """Return the result metadata emitted for this suite."""
        metadata = {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "order": normalize_order(self.order),
            "stream": self.stream,
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
        }
        metadata.update(self._label_permutation_metadata())
        return metadata

    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate. Keep the nested traversal explicit so ordering and metadata stay aligned."""
        task_name = self.task_name()
        resolved_order = normalize_order(self.order)
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            stream=self.stream,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if resolved_order != "native" and self.stream:
            raise ValueError("benchmark `stream=True` requires `order='native'`")
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
        samples = apply_order(
            samples,
            order=resolved_order,
            length_key=self.order_length,
        )
        requests: list[LoglikelihoodRequest] = []
        request_to_choice: list[tuple[int, int]] = []
        request_progress_metadata = loglikelihood_progress_metadata(
            title=f"{task_name}: scoring answer choices",
        )
        for sample in samples:
            for choice_index, choice in enumerate(sample.choices):
                requests.append(
                    LoglikelihoodRequest(
                        context=sample.prompt,
                        continuation=self.continuation_for_choice(choice),
                        metadata=dict(request_progress_metadata),
                    )
                )
                request_to_choice.append((sample.index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        label_metric_name, label_outcomes = self._label_permutation_scores(session, samples=samples)

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
        total_score = 0.0
        label_total = 0.0
        for sample in samples:
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item.index)
            outcome = exam_score_outcome(choice_scores, sample.gold_index)
            selected_indices = list(outcome.selected_indices)
            total_score += outcome.exam_score
            choice_labels = sample.metadata["choice_labels"]
            selected_labels = [choice_labels[index] for index in selected_indices]
            selected_texts = [sample.choices[index] for index in selected_indices]
            sample_scores = {"acc,exam": outcome.exam_score}
            sample_extracted = {
                "gold_index": str(sample.gold_index),
                "selected_indices": ",".join(str(index) for index in selected_indices),
                "selected_labels": ",".join(selected_labels),
            }
            sample_metadata = {
                **sample.metadata,
                "choice_logprobs": choice_logprobs(choice_scores),
                "selected_count": len(selected_indices),
            }
            if label_metric_name is not None:
                label_outcome = label_outcomes[sample.index]
                label_total += label_outcome.accuracy
                sample_scores[label_metric_name] = label_outcome.accuracy
                sample_extracted[f"predicted_index_{label_metric_name.split(',', maxsplit=1)[1]}"] = str(
                    label_outcome.predicted_index
                )
                sample_metadata[f"choice_logprobs_{label_metric_name.split(',', maxsplit=1)[1]}"] = list(
                    label_outcome.averaged_choice_logprobs
                )
                sample_metadata["label_permutation_count"] = label_outcome.permutation_count
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=" | ".join(selected_texts),
                    extracted=sample_extracted,
                    scores=sample_scores,
                    metadata=sample_metadata,
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {"acc,exam": total_score / denominator}
        if label_metric_name is not None:
            metrics[label_metric_name] = label_total / denominator
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
