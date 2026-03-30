# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evalution.benchmarks.base import BaseTestSuite, TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, RollingLoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult


@dataclass(slots=True)
class RollingPerplexitySample:
    index: int
    source_text: str
    scored_text: str
    word_count: int
    byte_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def preview_text(text: str, *, limit: int = 160) -> str:
    preview = text.replace("\n", "\\n")
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


@dataclass(slots=True)
class BaseRollingPerplexitySuite(TestSuite, ABC):
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    # Materialize datasets by default so document-level perplexity runs keep stable metadata and ordering.
    stream: bool = (False)
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
    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        raise NotImplementedError

    def primary_metric(self) -> str:
        return "word_perplexity"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "scoring_mode": "rolling_loglikelihood_perplexity",
            "primary_metric": self.primary_metric(),
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
        requests = [RollingLoglikelihoodRequest(text=sample.scored_text) for sample in samples]
        outputs = session.loglikelihood_rolling(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        sample_results: list[SampleResult] = []
        total_logprob = 0.0
        total_words = 0
        total_bytes = 0
        for sample, output in zip(samples, outputs, strict=True):
            if sample.word_count <= 0:
                raise ValueError(
                    f"{task_name} sample {sample.index} produced a non-positive word count"
                )
            if sample.byte_count <= 0:
                raise ValueError(
                    f"{task_name} sample {sample.index} produced a non-positive byte count"
                )

            total_logprob += output.logprob
            total_words += sample.word_count
            total_bytes += sample.byte_count
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt="",
                    target="[document]",
                    prediction="[rolling-loglikelihood]",
                    extracted={
                        "token_count": str(output.token_count),
                        "word_count": str(sample.word_count),
                        "byte_count": str(sample.byte_count),
                    },
                    scores={
                        "word_perplexity": math.exp(-(output.logprob / sample.word_count)),
                        "byte_perplexity": math.exp(-(output.logprob / sample.byte_count)),
                        "bits_per_byte": -(output.logprob / sample.byte_count) / math.log(2),
                    },
                    metadata={
                        **sample.metadata,
                        "logprob": output.logprob,
                        "token_count": output.token_count,
                    },
                )
            )

        metrics = {
            "word_perplexity": math.exp(-(total_logprob / total_words)),
            "byte_perplexity": math.exp(-(total_logprob / total_bytes)),
            "bits_per_byte": -(total_logprob / total_bytes) / math.log(2),
        }
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
