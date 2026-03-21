from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.suites.base import TestSuite
from evalution.suites.data import doc_count, limit_docs, load_suite_dataset


@dataclass(slots=True)
class MultipleChoiceSample:
    # Represent one multiple-choice question after prompt formatting and choice extraction.
    index: int
    prompt: str
    choices: list[str]
    gold_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaseMultipleChoiceSuite(TestSuite, ABC):
    # Dataset location and execution options shared by multiple-choice benchmark families.
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "validation"
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    # Return the callable used to fetch the underlying dataset rows.
    @abstractmethod
    def dataset_loader(self) -> Any:
        raise NotImplementedError

    # Return the stable result name for the concrete suite instance.
    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError

    # Convert one dataset row into the normalized prompt/choices form scored by the helper.
    @abstractmethod
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raise NotImplementedError

    # Control how each choice is appended to the shared prompt during scoring.
    def continuation_for_choice(self, choice: str) -> str:
        return choice if choice[:1].isspace() else f" {choice}"

    # Report suite-level metadata that is stable across all samples in the run.
    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "scoring_mode": "multiple_choice_loglikelihood",
        }

    # Allow concrete suites to publish extra aggregate metrics without reimplementing the shared scoring loop.
    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        del samples, raw_predictions, normalized_predictions
        return {}

    # Execute the shared dataset loading, flattened choice scoring, and accuracy aggregation flow.
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

        sample_choice_scores: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            # Track both raw and length-normalized scores so suites can expose `acc` and `acc_norm`.
            sample_choice_scores[sample_index].append(
                (
                    output.logprob,
                    output.logprob / max(output.token_count, 1),
                    choice_index,
                )
            )

        sample_results: list[SampleResult] = []
        raw_total = 0.0
        norm_total = 0.0
        raw_predictions: list[int] = []
        normalized_predictions: list[int] = []
        for sample in samples:
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item[2])
            raw_best = max(choice_scores, key=lambda item: item[0])[2]
            norm_best = max(choice_scores, key=lambda item: item[1])[2]
            raw_score = 1.0 if raw_best == sample.gold_index else 0.0
            norm_score = 1.0 if norm_best == sample.gold_index else 0.0
            raw_total += raw_score
            norm_total += norm_score
            raw_predictions.append(raw_best)
            normalized_predictions.append(norm_best)
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=sample.choices[norm_best],
                    extracted={
                        "gold_index": str(sample.gold_index),
                        "predicted_index": str(raw_best),
                        "predicted_index_norm": str(norm_best),
                    },
                    scores={
                        "accuracy,loglikelihood": raw_score,
                        "accuracy,loglikelihood_norm": norm_score,
                    },
                    metadata={
                        **sample.metadata,
                        "choice_logprobs": [score for score, _norm, _index in choice_scores],
                        "choice_logprobs_norm": [norm for _score, norm, _index in choice_scores],
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {
            "accuracy,loglikelihood": raw_total / denominator,
            "accuracy,loglikelihood_norm": norm_total / denominator,
        }
        metrics.update(
            self.extra_metrics(
                samples=samples,
                raw_predictions=raw_predictions,
                normalized_predictions=normalized_predictions,
            )
        )
        logger.info("%s: metrics=%s", task_name, metrics)
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
