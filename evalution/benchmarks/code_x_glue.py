# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

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
from evalution.scorers.bleu import smoothed_corpus_bleu_4

# Keep benchmark defaults and public task ids explicit at module scope.
CODE_X_GLUE_LANGUAGES = ("go", "java", "javascript", "php", "python", "ruby")


def _normalized_code_text(code_tokens: list[str]) -> str:
    """Implement normalized code text for this module."""
    text = " ".join(str(token) for token in code_tokens).replace("\n", " ")
    return " ".join(text.strip().split())


def _normalized_docstring_text(docstring_tokens: list[str]) -> str:
    """Implement normalized docstring text for this module."""
    text = " ".join(str(token) for token in docstring_tokens).replace("\n", "")
    return " ".join(text.strip().split())


@dataclass(slots=True)
class CodeXGLUECodeToText(BaseTestSuite):
    """Implement the code xgluecode to text benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    language: str = "go"
    dataset_path: str = ""
    split: str = "test"
    max_new_tokens: int = 128
    num_beams: int = 10
    stop: tuple[str, ...] = ("</s>",)

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.language not in CODE_X_GLUE_LANGUAGES:
            raise ValueError(
                f"language must be one of {CODE_X_GLUE_LANGUAGES!r}, got {self.language!r}"
            )
        if not self.dataset_path:
            self.dataset_path = f"CM/codexglue_code2text_{self.language}"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"code2text_{self.language}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": self.language,
            "num_beams": self.num_beams,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            prompt = _normalized_code_text(list(doc["code_tokens"]))
            target = _normalized_docstring_text(list(doc["docstring_tokens"]))
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=prompt,
                    stop=list(self.stop),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
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
                "id": int(prepared_sample.doc["id"]),
                "language": str(prepared_sample.doc["language"]),
                "repo": str(prepared_sample.doc["repo"]),
                "path": str(prepared_sample.doc["path"]),
                "func_name": str(prepared_sample.doc["func_name"]),
                "sha": str(prepared_sample.doc["sha"]),
                "url": str(prepared_sample.doc["url"]),
                "code_token_count": len(prepared_sample.doc["code_tokens"]),
                "docstring_token_count": len(prepared_sample.doc["docstring_tokens"]),
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
            """Implement record output for code xgluecode to text."""
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
        metrics = {
            "bleu4": smoothed_corpus_bleu_4(references, predictions),
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


def code_x_glue(language: str = "go", **kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code x glue for this module."""
    return CodeXGLUECodeToText(language=language, **kwargs)


def code2text_go(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text go for this module."""
    return code_x_glue("go", **kwargs)


def code2text_java(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text java for this module."""
    return code_x_glue("java", **kwargs)


def code2text_javascript(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text javascript for this module."""
    return code_x_glue("javascript", **kwargs)


def code2text_php(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text php for this module."""
    return code_x_glue("php", **kwargs)


def code2text_python(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text python for this module."""
    return code_x_glue("python", **kwargs)


def code2text_ruby(**kwargs: Any) -> CodeXGLUECodeToText:
    """Implement code2text ruby for this module."""
    return code_x_glue("ruby", **kwargs)
