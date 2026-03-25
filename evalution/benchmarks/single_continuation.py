# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult


@dataclass(slots=True)
class SingleContinuationSample:
    # Represent one prompt plus exactly one gold continuation scored by log-likelihood.
    index: int
    prompt: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaseSingleContinuationSuite(TestSuite, ABC):
    # Dataset-backed suites that score one fixed continuation per prompt.
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = False
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None

    @abstractmethod
    def dataset_loader(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_sample(self, doc: dict[str, Any], *, index: int) -> SingleContinuationSample:
        raise NotImplementedError

    def continuation_for_target(self, target: str) -> str:
        return target if target[:1].isspace() else f" {target}"

    def include_perplexity(self) -> bool:
        return True

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "scoring_mode": "single_continuation_loglikelihood",
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

        samples = [self.build_sample(doc, index=index) for index, doc in enumerate(docs)]
        requests = [
            LoglikelihoodRequest(
                context=sample.prompt,
                continuation=self.continuation_for_target(sample.target),
            )
            for sample in samples
        ]

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        sample_results: list[SampleResult] = []
        accuracy_total = 0.0
        logprob_total = 0.0
        include_perplexity = self.include_perplexity()
        for sample, output in zip(samples, outputs, strict=True):
            greedy_match = 1.0 if output.is_greedy else 0.0
            accuracy_total += greedy_match
            logprob_total += output.logprob
            scores = {"acc,ll": greedy_match}
            if include_perplexity:
                scores["ppl,ll"] = math.exp(-output.logprob)
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.target,
                    prediction=sample.target if output.is_greedy else "[not-greedy]",
                    extracted={
                        "greedy_match": str(int(greedy_match)),
                        "token_count": str(output.token_count),
                    },
                    scores=scores,
                    metadata={
                        **sample.metadata,
                        "logprob": output.logprob,
                        "token_count": output.token_count,
                        "is_greedy": output.is_greedy,
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {"acc,ll": accuracy_total / denominator}
        if include_perplexity:
            # Aggregate perplexity from the mean document log-probability.
            metrics["ppl,ll"] = math.exp(-(logprob_total / denominator))
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
