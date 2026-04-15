# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from time import perf_counter
from typing import Any

from evalution.engines.base import GenerationOutput, InferenceSession
from evalution.logbar import get_logger, manual_progress
from evalution.results import SampleResult, TestResult
from evalution.benchmarks.data import (
    apply_order,
    doc_count,
    load_suite_dataset,
    normalize_order,
    select_docs,
)
from evalution.benchmarks.execution import (
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
    # Run the suite against an initialized inference session.
    """Define the test suite helper class."""
    @abstractmethod
    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate."""
        raise NotImplementedError


# Keep benchmark defaults and public task ids explicit at module scope.
DEFAULT_STREAM = False


@dataclass(slots=True)
class BaseTestSuite(TestSuite):
    """Define the base test suite helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    order: str = "native"
    # Materialize datasets by default; benchmarks that rely on streaming opt in explicitly.
    stream: bool = DEFAULT_STREAM
    max_rows: int | None = None
    # Allow deterministic test-only subsets when a benchmark's first rows are too large to run.
    row_indices: tuple[int, ...] | None = None
    batch_size: int | None = None
    cache_dir: str | None = None

    # Return the callable used to fetch the underlying dataset rows.
    @abstractmethod
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        raise NotImplementedError

    # Return the stable result name for the concrete suite instance.
    @abstractmethod
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        raise NotImplementedError

    # Tell the pipeline whether the suite needs all rows materialized up front.
    def requires_full_doc_materialization(self) -> bool:
        """Implement requires full doc materialization for base test suite."""
        return False

    # Let suites override how generation requests are length-ranked for benchmark-level ordering.
    def order_length(self, prepared_sample: PreparedSample) -> int:
        """Implement order length for base test suite. Preserve the fallback order expected by the surrounding caller."""
        request = prepared_sample.request
        if request.input_ids is not None:
            return len(request.input_ids)
        if request.rendered_prompt is not None:
            return len(request.rendered_prompt)
        if request.prompt is not None:
            return len(request.prompt)
        if request.messages is not None:
            return sum(
                len(str(message.get("role", ""))) + len(str(message.get("content", "")))
                for message in request.messages
            )
        return 0

    # Convert dataset rows into generation requests and scoring targets.
    @abstractmethod
    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        raise NotImplementedError

    # Turn a model output into the per-sample result object.
    @abstractmethod
    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        raise NotImplementedError

    # Report how many invalid predictions a sample contributes to progress stats.
    def invalid_prediction_count(self, sample: SampleResult) -> int:
        """Implement invalid prediction count for base test suite."""
        del sample
        return 0

    # Format the live scoring progress title shown during evaluation.
    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        """Score progress title. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        del processed, aggregate_scores, invalid_predictions
        return f"{self.task_name()}: scoring"

    # Allow suites to append suite-specific metadata to the final result.
    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return self.base_result_metadata(
            generation_submission_mode=generation_submission_mode,
        )

    # Populate dataset-level metadata that every dataset-backed suite exposes.
    def base_result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Implement base result metadata for base test suite."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "order": normalize_order(self.order),
            "stream": self.stream,
            "generation_submission_mode": generation_submission_mode,
        }
        if self.row_indices is not None:
            metadata["row_indices"] = list(self.row_indices)
        return metadata

    # Execute the shared dataset, batching, generation, and scoring pipeline.
    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate. Preserve the fallback order expected by the surrounding caller."""
        task_name = self.task_name()
        resolved_order = normalize_order(self.order)
        logger = get_logger()
        loaded_docs, dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            stream=self.stream,
        )

        docs = select_docs(
            loaded_docs,
            row_indices=self.row_indices,
            max_rows=self.max_rows,
        )
        if resolved_order != "native" and self.stream:
            raise ValueError("benchmark `stream=True` requires `order='native'`")
        if self.requires_full_doc_materialization() or resolved_order != "native":
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        prepare_bar = manual_progress(
            total,
            title=f"{task_name}: preparing requests",
        )
        prepared_iter = self.iter_prepared_samples(docs)
        ordered_sample_indices: list[int] | None = None
        ordered_prepared_samples: list[PreparedSample] | None = None
        prepared_samples_by_index: dict[int, PreparedSample] = {}
        if resolved_order == "native":
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
        else:
            ordered_prepared_samples = list(prepared_iter)
            for _ in ordered_prepared_samples:
                prepare_bar.next().draw()
            ordered_prepared_samples = apply_order(
                ordered_prepared_samples,
                order=resolved_order,
                length_key=self.order_length,
            )
            ordered_sample_indices = [sample.index for sample in ordered_prepared_samples]
            ordered_prepared_samples = prepare_batch_for_session(session, ordered_prepared_samples)
            preview_samples = []
            prepared_samples_by_index = {
                sample.index: sample
                for sample in ordered_prepared_samples
            }

        aggregate_scores: defaultdict[str, float] = defaultdict(float)
        samples_by_index: list[SampleResult | None] = [None] * total
        effective_batch_size = (
            self.batch_size
            or session_batch_size(
                session,
                [
                    sample.request
                    for sample in (
                        preview_samples
                        if resolved_order == "native"
                        else list(islice(ordered_prepared_samples or [], AUTO_BATCH_PREVIEW_ROWS))
                    )
                ],
            )
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

        # Keep scoring side effects centralized so both generation modes share them.
        def score_output(prepared_sample: PreparedSample, output: GenerationOutput) -> None:
            """Score output. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
            if use_continuous_generation and resolved_order == "native":
                sample_by_request_key: dict[int, PreparedSample] = {}

                # Feed requests lazily so continuous generation can refill slots as outputs finish.
                def iter_request_stream() -> Any:
                    """Yield request items for the continuous generation loop."""
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
            elif resolved_order == "native":
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
            elif use_continuous_generation:
                generation_started = perf_counter()
                for request_key, output in generate_continuous(
                    (
                        (sample.index, sample.request)
                        for sample in (ordered_prepared_samples or [])
                    ),
                    batch_size=effective_batch_size,
                ):
                    prepared_sample = prepared_samples_by_index[int(request_key)]
                    score_output(prepared_sample, output)
                generation_wall_s += perf_counter() - generation_started
            else:
                batch_index = 0
                ordered_batches = ordered_prepared_samples or []
                for batch_start in range(0, len(ordered_batches), effective_batch_size):
                    batch_index += 1
                    prepared_batch = ordered_batches[batch_start : batch_start + effective_batch_size]
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

        if ordered_sample_indices is None:
            samples = [sample for sample in samples_by_index if sample is not None]
        else:
            samples = [
                samples_by_index[sample_index]
                for sample_index in ordered_sample_indices
                if samples_by_index[sample_index] is not None
            ]
        logger.info("%s: executed %d/%d sample(s)", task_name, processed_count, total)
        if processed_count != total:
            logger.warning(
                "%s: only executed %d/%d sample(s); generation returned fewer outputs than expected",
                task_name,
                processed_count,
                total,
            )
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
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=samples,
            metadata=self.result_metadata(
                generation_submission_mode=generation_submission_mode,
            ),
        )
