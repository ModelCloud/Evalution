# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger, loglikelihood_progress_metadata
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    build_choice_scores,
    choice_logprobs,
    choice_logprobs_norm,
    multiple_choice_outcome,
)


@dataclass(slots=True)
class BaseFewshotMultipleChoiceSuite(TestSuite, ABC):
    # Share one clean-room evaluator for MMLU-style suites that prepend fixed few-shot exemplars.
    """Define the base fewshot multiple choice suite helper class."""
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    fewshot_split: str = "dev"
    num_fewshot: int = 5
    stream: bool = False
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None

    @abstractmethod
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        raise NotImplementedError

    @abstractmethod
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        raise NotImplementedError

    @abstractmethod
    def format_question(self, doc: dict[str, Any], *, include_answer: bool) -> str:
        """Format question."""
        raise NotImplementedError

    @abstractmethod
    def gold_label(self, doc: dict[str, Any]) -> str:
        """Implement gold label for base fewshot multiple choice suite."""
        raise NotImplementedError

    def prompt_description(self) -> str:
        """Implement prompt description for base fewshot multiple choice suite."""
        return ""

    def choice_labels(self) -> tuple[str, ...]:
        """Implement choice labels for base fewshot multiple choice suite."""
        return ("A", "B", "C", "D")

    def choice_texts(self, doc: dict[str, Any]) -> list[str]:
        """Implement choice texts for base fewshot multiple choice suite."""
        return [str(doc[label]).strip() for label in self.choice_labels()]

    def sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Implement sample metadata for base fewshot multiple choice suite."""
        del doc
        return {}

    # Keep the few-shot prefix construction centralized so concrete suites only define row formatting.
    def _fewshot_prompt(self, fewshot_docs: list[dict[str, Any]]) -> str:
        """Implement fewshot prompt for base fewshot multiple choice suite."""
        sections: list[str] = []
        description = self.prompt_description().strip()
        if description:
            sections.append(description)
        sections.extend(
            self.format_question(doc, include_answer=True)
            for doc in fewshot_docs
        )
        if not sections:
            return ""
        return "\n\n".join(sections) + "\n\n"

    # Return one stable metadata payload for every derived few-shot multiple-choice result.
    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "fewshot_split": self.fewshot_split,
            "num_fewshot": self.num_fewshot,
            "stream": self.stream,
            "scoring_mode": "multiple_choice_loglikelihood",
        }

    # Materialize the few-shot prefix once, then score each answer label with log-likelihood.
    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate. Keep the nested traversal explicit so ordering and metadata stay aligned."""
        task_name = self.task_name()
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
        if not isinstance(docs, list):
            docs = list(docs)
        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        fewshot_loaded_docs, _fewshot_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.fewshot_split,
            cache_dir=self.cache_dir,
            stream=self.stream,
            purpose="few-shot",
        )
        fewshot_docs = limit_docs(fewshot_loaded_docs, self.num_fewshot)
        if not isinstance(fewshot_docs, list):
            fewshot_docs = list(fewshot_docs)
        prompt_prefix = self._fewshot_prompt(fewshot_docs)

        labels = self.choice_labels()
        requests: list[LoglikelihoodRequest] = []
        prompts: list[str] = []
        sample_docs: list[dict[str, Any]] = []
        request_progress_metadata = loglikelihood_progress_metadata(
            title=f"{task_name}: scoring answer choices",
        )
        for doc in docs:
            prompt = prompt_prefix + self.format_question(doc, include_answer=False)
            prompts.append(prompt)
            sample_docs.append(doc)
            for label in labels:
                requests.append(
                    LoglikelihoodRequest(
                        context=prompt,
                        continuation=f" {label}",
                        metadata=dict(request_progress_metadata),
                    )
                )

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_docs), total)

        raw_total = 0.0
        norm_total = 0.0
        sample_results: list[SampleResult] = []
        for index, doc in enumerate(sample_docs):
            choice_count = len(labels)
            start = index * choice_count
            choice_outputs = outputs[start : start + choice_count]
            choice_scores = build_choice_scores(
                (
                    choice_index,
                    output.logprob,
                    output.token_count,
                )
                for choice_index, output in enumerate(choice_outputs)
            )
            gold_label = self.gold_label(doc)
            gold_index = labels.index(gold_label)
            outcome = multiple_choice_outcome(choice_scores, gold_index)
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            sample_results.append(
                SampleResult(
                    index=index,
                    prompt=prompts[index],
                    target=gold_label,
                    prediction=labels[outcome.normalized_best_index],
                    extracted={
                        "gold_index": str(gold_index),
                        "predicted_index": str(outcome.raw_best_index),
                        "predicted_index_norm": str(outcome.normalized_best_index),
                    },
                    scores={
                        "acc,ll": outcome.raw_accuracy,
                        "acc,ll_avg": outcome.normalized_accuracy,
                    },
                    metadata={
                        "choice_labels": list(labels),
                        "choice_texts": self.choice_texts(doc),
                        "choice_logprobs": choice_logprobs(choice_scores),
                        "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
                        "fewshot_count": len(fewshot_docs),
                        **self.sample_metadata(doc),
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
