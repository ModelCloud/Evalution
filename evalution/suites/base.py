from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from evalution.engines.base import GenerationOutput, InferenceSession
from evalution.logbar import get_logger, manual_progress
from evalution.results import SampleResult, TestResult
from evalution.suites.data import doc_count, limit_docs, load_suite_dataset
from evalution.suites.execution import (
    AUTO_BATCH_PREVIEW_ROWS,
    PreparedSample,
    collect_preview_samples,
    iter_prefetched_batches,
    iter_prefetched_samples,
    needs_batch_size_preview,
    prepare_batch_for_session,
    session_batch_size,
)


class TestSuite(ABC):
    @abstractmethod
    def evaluate(self, session: InferenceSession) -> TestResult:
        raise NotImplementedError


@dataclass(slots=True)
class BaseTestSuite(TestSuite):
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    limit: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    @abstractmethod
    def dataset_loader(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError

    def requires_full_doc_materialization(self) -> bool:
        return False

    @abstractmethod
    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        raise NotImplementedError

    def invalid_prediction_count(self, sample: SampleResult) -> int:
        del sample
        return 0

    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        del processed, aggregate_scores, invalid_predictions
        return f"{self.task_name()}: scoring"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return self.base_result_metadata(
            generation_submission_mode=generation_submission_mode,
        )

    def base_result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "generation_submission_mode": generation_submission_mode,
        }

    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )

        docs = limit_docs(loaded_docs, self.limit)
        if self.requires_full_doc_materialization():
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            limit=self.limit,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        prepare_bar = manual_progress(
            total,
            title=f"{task_name}: preparing requests",
        )
        prepared_iter = self.iter_prepared_samples(docs)
        preview_size = (
            min(total, AUTO_BATCH_PREVIEW_ROWS)
            if needs_batch_size_preview(self.batch_size, session)
            else 0
        )
        preview_samples = collect_preview_samples(
            prepared_iter,
            preview_size=preview_size,
            prepare_bar=prepare_bar,
        )
        preview_samples = prepare_batch_for_session(session, preview_samples)

        aggregate_scores: defaultdict[str, float] = defaultdict(float)
        samples_by_index: list[SampleResult | None] = [None] * total
        effective_batch_size = (
            self.batch_size
            or session_batch_size(session, [sample.request for sample in preview_samples])
            or 1
        )
        logger.info("%s: using batch_size=%d", task_name, effective_batch_size)
        logger.info(
            "%s: request preparation mode=%s",
            task_name,
            "threaded_prefetch"
            if callable(getattr(session, "prepare_requests", None))
            else "inline",
        )
        generate_continuous = getattr(session, "generate_continuous", None)
        use_continuous_generation = callable(generate_continuous)
        generation_submission_mode = (
            "continuous_refill" if use_continuous_generation else "fixed_batches"
        )
        logger.info(
            "%s: generation submission mode=%s",
            task_name,
            generation_submission_mode,
        )

        invalid_predictions = 0
        generation_wall_s = 0.0
        scoring_wall_s = 0.0
        processed_count = 0
        score_bar = manual_progress(
            total,
            title=self.score_progress_title(
                processed=0,
                aggregate_scores={},
                invalid_predictions=0,
            ),
            subtitle=f"batch_size={effective_batch_size}",
        )

        def score_output(prepared_sample: PreparedSample, output: GenerationOutput) -> None:
            nonlocal invalid_predictions
            nonlocal processed_count
            nonlocal scoring_wall_s

            scoring_started = perf_counter()
            sample = self.score_sample(prepared_sample, output)
            for metric_name, score in sample.scores.items():
                aggregate_scores[metric_name] += score
            invalid_predictions += self.invalid_prediction_count(sample)
            samples_by_index[sample.index] = sample
            processed_count += 1
            score_bar.title(
                self.score_progress_title(
                    processed=processed_count,
                    aggregate_scores=dict(aggregate_scores),
                    invalid_predictions=invalid_predictions,
                )
            )
            score_bar.next().draw()
            scoring_wall_s += perf_counter() - scoring_started

        try:
            if use_continuous_generation:
                sample_by_request_key: dict[int, PreparedSample] = {}

                def iter_request_stream() -> Any:
                    request_key = 0
                    prefetched_samples = iter_prefetched_samples(
                        session,
                        preview_samples,
                        prepared_iter,
                        batch_size=effective_batch_size,
                        prepare_bar=prepare_bar,
                    )
                    try:
                        for prepared_sample in prefetched_samples:
                            sample_by_request_key[request_key] = prepared_sample
                            yield request_key, prepared_sample.request
                            request_key += 1
                    finally:
                        close_prefetched_samples = getattr(prefetched_samples, "close", None)
                        if callable(close_prefetched_samples):
                            close_prefetched_samples()

                generation_started = perf_counter()
                for request_key, output in generate_continuous(
                    iter_request_stream(),
                    batch_size=effective_batch_size,
                ):
                    prepared_sample = sample_by_request_key.pop(request_key)
                    score_output(prepared_sample, output)
                generation_wall_s += perf_counter() - generation_started
            else:
                batch_index = 0
                for prepared_batch in iter_prefetched_batches(
                    session,
                    preview_samples,
                    prepared_iter,
                    batch_size=effective_batch_size,
                    prepare_bar=prepare_bar,
                ):
                    batch_index += 1
                    batch_requests = [sample.request for sample in prepared_batch]
                    generation_started = perf_counter()
                    batch_outputs = session.generate(
                        batch_requests,
                        batch_size=len(batch_requests),
                    )
                    generation_wall_s += perf_counter() - generation_started
                    score_bar.subtitle(
                        f"batch={batch_index}/{(total + effective_batch_size - 1) // effective_batch_size}"
                    )
                    for batch_offset, output in enumerate(batch_outputs):
                        score_output(prepared_batch[batch_offset], output)
        finally:
            close_prepared_iter = getattr(prepared_iter, "close", None)
            if callable(close_prepared_iter):
                close_prepared_iter()
            prepare_bar.title(f"{task_name}: prepared requests")
            prepare_bar.draw()
            prepare_bar.close()
            score_bar.close()

        samples = [sample for sample in samples_by_index if sample is not None]
        denominator = len(samples) or 1
        metrics = {
            metric_name: total_score / denominator
            for metric_name, total_score in aggregate_scores.items()
        }
        logger.info(
            "%s: wall_times dataset_load=%.3fs generation=%.3fs scoring=%.3fs",
            task_name,
            dataset_load_wall_s,
            generation_wall_s,
            scoring_wall_s,
        )
        logger.info("%s: metrics=%s", task_name, metrics)
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=samples,
            metadata=self.result_metadata(
                generation_submission_mode=generation_submission_mode,
            ),
        )
