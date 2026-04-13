# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

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
from evalution.datasets.flores200 import FLORES200_ARCHIVE_SHA256, FLORES200_ARCHIVE_URL, load_flores200_pair
from evalution.engines.base import GenerationOutput, GenerationRequest, InferenceSession
from evalution.logbar import get_logger, manual_progress
from evalution.results import SampleResult, TestResult
from evalution.scorers.translation import corpus_translation_metrics

# Keep benchmark defaults and public task ids explicit at module scope.
_LANGUAGE_CODE_BY_TOKEN = {
    "ca": "cat_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "eu": "eus_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
}
_LANGUAGE_NAME_BY_TOKEN = {
    "ca": "Catalan",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "eu": "Basque",
    "fr": "French",
    "gl": "Galician",
    "it": "Italian",
    "pt": "Portuguese",
}
FLORES_PT_DIRECTIONS = (
    "ca-pt",
    "de-pt",
    "en-pt",
    "es-pt",
    "eu-pt",
    "fr-pt",
    "gl-pt",
    "it-pt",
    "pt-ca",
    "pt-de",
    "pt-en",
    "pt-es",
    "pt-eu",
    "pt-fr",
    "pt-gl",
    "pt-it",
)
FLORES_PT_TASKS = tuple(f"flores_pt_{direction.replace('-', '_')}" for direction in FLORES_PT_DIRECTIONS)
_FLORES_PT_TASK_BY_DIRECTION = {
    direction: f"flores_pt_{direction.replace('-', '_')}"
    for direction in FLORES_PT_DIRECTIONS
}
_FLORES_PT_DIRECTION_BY_TASK = {
    task_name: direction
    for direction, task_name in _FLORES_PT_TASK_BY_DIRECTION.items()
}


def _normalize_direction(direction: str) -> str:
    """Normalize direction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    normalized = direction.strip().lower()
    if normalized in FLORES_PT_DIRECTIONS:
        return normalized
    if normalized in _FLORES_PT_DIRECTION_BY_TASK:
        return _FLORES_PT_DIRECTION_BY_TASK[normalized]
    raise ValueError(f"unsupported flores_pt direction: {direction!r}")


def _translation_prompt(source_language: str, target_language: str, source_sentence: str) -> str:
    """Implement translation prompt for this module."""
    return f"{source_language} sentence: {source_sentence.strip()}\n{target_language} sentence:"


@dataclass(slots=True)
class FloresPT(BaseTestSuite):
    # Evaluate PortugueseBench FLORES directions with a local audited FLORES-200 archive loader.
    """Implement the flores pt benchmark suite."""
    dataset_path: str = "facebook/flores"
    dataset_name: str | None = "all"
    split: str = "devtest"
    direction: str = "en-pt"
    max_new_tokens: int = 256
    stop: tuple[str, ...] = ("\n",)

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        self.direction = _normalize_direction(self.direction)
        if self.dataset_path != "facebook/flores":
            raise ValueError("flores_pt dataset_path must be 'facebook/flores'")
        if self.dataset_name not in {None, "all"}:
            raise ValueError("flores_pt dataset_name must be None or 'all'")
        if self.dataset_name is None:
            self.dataset_name = "all"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        source_language, target_language = self.language_pair_tokens()

        def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            """Implement loader for flores pt."""
            return load_flores200_pair(
                dataset_path,
                dataset_name,
                source_language=_LANGUAGE_CODE_BY_TOKEN[source_language],
                target_language=_LANGUAGE_CODE_BY_TOKEN[target_language],
                **kwargs,
            )

        return loader

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _FLORES_PT_TASK_BY_DIRECTION[self.direction]

    def language_pair_tokens(self) -> tuple[str, str]:
        """Implement language pair tokens for flores pt."""
        return tuple(self.direction.split("-", maxsplit=1))  # type: ignore[return-value]

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        source_language, target_language = self.language_pair_tokens()
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": self.direction,
            "source_language": source_language,
            "target_language": target_language,
            "upstream_task": f"portuguese_bench_flores_{self.direction}",
            "archive_url": FLORES200_ARCHIVE_URL,
            "archive_sha256": FLORES200_ARCHIVE_SHA256,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        source_token, target_token = self.language_pair_tokens()
        source_field = f"sentence_{_LANGUAGE_CODE_BY_TOKEN[source_token]}"
        target_field = f"sentence_{_LANGUAGE_CODE_BY_TOKEN[target_token]}"
        source_name = _LANGUAGE_NAME_BY_TOKEN[source_token]
        target_name = _LANGUAGE_NAME_BY_TOKEN[target_token]
        for index, doc in enumerate(docs):
            prompt = _translation_prompt(source_name, target_name, str(doc[source_field]))
            target = str(doc[target_field]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=prompt,
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
        source_token, target_token = self.language_pair_tokens()
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
                "id": int(prepared_sample.doc["id"]),
                "URL": str(prepared_sample.doc["URL"]),
                "domain": str(prepared_sample.doc["domain"]),
                "topic": str(prepared_sample.doc["topic"]),
                "has_image": bool(prepared_sample.doc["has_image"]),
                "has_hyperlink": bool(prepared_sample.doc["has_hyperlink"]),
                "direction": self.direction,
                "source_language": source_token,
                "target_language": target_token,
            },
        )

    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate. Preserve the fallback order expected by the surrounding caller."""
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
            """Implement record output for flores pt."""
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
        logger.info("%s: executed %d/%d sample(s)", task_name, processed_count, total)
        if processed_count != total:
            logger.warning(
                "%s: only executed %d/%d sample(s); generation returned fewer outputs than expected",
                task_name,
                processed_count,
                total,
            )

        references = [sample.extracted["reference-stripped"] for sample in samples]
        predictions = [sample.extracted["prediction-stripped"] for sample in samples]
        metrics = corpus_translation_metrics(references, predictions)
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


def flores_pt(*, direction: str, **kwargs: Any) -> FloresPT:
    """Implement flores pt for this module."""
    return FloresPT(direction=direction, **kwargs)


def _make_flores_pt_factory(direction: str) -> Any:
    """Make flores pt factory."""
    def factory(**kwargs: Any) -> FloresPT:
        """Implement factory for this module."""
        return flores_pt(direction=direction, **kwargs)

    factory.__name__ = _FLORES_PT_TASK_BY_DIRECTION[direction]
    return factory


for _direction in FLORES_PT_DIRECTIONS:
    globals()[_FLORES_PT_TASK_BY_DIRECTION[_direction]] = _make_flores_pt_factory(_direction)

del _direction
