# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

import sacrebleu
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
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
from evalution.engines.base import GenerationOutput, GenerationRequest, InferenceSession
from evalution.logbar import get_logger, manual_progress
from evalution.results import SampleResult, TestResult
from evalution.scorers.summary_rouge import summary_rouge_scores


def _cocoteros_prompt(keywords: str, context: str) -> str:
    """Implement cocoteros prompt for this module."""
    return (
        "Genera una frase corta con estas palabras: "
        f"{str(keywords).strip()}. El contexto es: {str(context).strip()} \n\n"
        "Respuesta:"
    )


def _mean_rouge1(predictions: list[str], references: list[str]) -> float:
    """Implement mean rouge1 for this module."""
    if not predictions:
        return 0.0
    return mean(
        summary_rouge_scores(prediction, reference)["rouge1"]
        for prediction, reference in zip(predictions, references, strict=True)
    )


@dataclass(slots=True)
class CocoterosES(BaseTestSuite):
    # Evaluate the SpanishBench constrained sentence-generation task with corpus BLEU and mean ROUGE-1.
    """Implement the cocoteros es benchmark suite."""
    dataset_path: str = "gplsi/cocoteros"
    dataset_name: str | None = None
    split: str = "test"
    max_new_tokens: int = 40
    stop: tuple[str, ...] = ("\n",)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "cocoteros_es"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_corpus_bleu_mean_rouge1",
            "primary_metric": "bleu",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["text"]).strip(),
                request=GenerationRequest(
                    prompt=_cocoteros_prompt(doc["keywords"], doc["context"]),
                    stop=list(self.stop),
                    max_new_tokens=self.max_new_tokens,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        prediction = output.text.strip()
        reference = prepared_sample.target.strip()
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": prediction,
                "reference-stripped": reference,
            },
            scores={},
            metadata={
                "keywords": str(prepared_sample.doc["keywords"]).strip(),
                "context": str(prepared_sample.doc["context"]).strip(),
            },
        )

    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate."""
        task_name = self.task_name()
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

        docs = limit_docs(loaded_docs, self.max_rows)
        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        prepare_bar = manual_progress(total, title=f"{task_name}: preparing requests")
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

        samples_by_index: list[SampleResult | None] = [None] * total
        generation_wall_s = 0.0
        scoring_wall_s = 0.0
        processed_count = 0
        score_bar = manual_progress(
            total,
            title=f"{task_name}: collecting generations",
            subtitle=f"batch_size={effective_batch_size}",
        )

        def record_output(prepared_sample: PreparedSample, output: GenerationOutput) -> None:
            """Implement record output for cocoteros es."""
            nonlocal processed_count
            nonlocal scoring_wall_s

            scoring_started = perf_counter()
            sample = self.score_sample(prepared_sample, output)
            samples_by_index[sample.index] = sample
            processed_count += 1
            score_bar.next().draw()
            scoring_wall_s += perf_counter() - scoring_started

        try:
            if use_continuous_generation:
                sample_by_request_key: dict[int, PreparedSample] = {}

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
                    record_output(prepared_sample, output)
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
                        record_output(prepared_batch[batch_offset], output)
        finally:
            close_prepared_iter = getattr(prepared_iter, "close", None)
            if callable(close_prepared_iter):
                close_prepared_iter()
            prepare_bar.title(f"{task_name}: prepared requests")
            prepare_bar.draw()
            prepare_bar.close()
            score_bar.close()

        samples = [sample for sample in samples_by_index if sample is not None]
        predictions = [sample.prediction.strip() for sample in samples]
        references = [sample.target.strip() for sample in samples]
        metrics = {
            "bleu": float(sacrebleu.corpus_bleu(predictions, [references]).score)
            if predictions
            else 0.0,
            "rouge1": _mean_rouge1(predictions, references),
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


def cocoteros_es(**kwargs: Any) -> CocoterosES:
    """Implement cocoteros es for this module."""
    return CocoterosES(**kwargs)
