# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import pcre
from datasets import load_dataset

from evalution.engines.base import GenerationOutput, GenerationRequest, InferenceSession
from evalution.logbar import get_logger, manual_progress
from evalution.results import SampleResult, TestResult
from evalution.scorers.choice_label import choice_label_exact_match
from evalution.suites.base import TestSuite
from evalution.suites.data import limit_docs, load_suite_dataset
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
from evalution.suites.subsets import ResolvedSubsets, SubsetTree, normalize_subset_token

_INVALID_CHOICE = "[invalid]"
_OPTION_LABELS = tuple("ABCDEFGHIJKLMNOP")
_STOP_STRINGS = ("Question:", "</s>", "<|im_end|>", "<|eot_id|>")
_NON_ALNUM_PATTERN = pcre.compile(r"[^a-z0-9]+")
_MMLU_PRO_SUBSET_TREE = {
    "stem": {
        "biology": "biology",
        "chemistry": "chemistry",
        "computer_science": "computer science",
        "engineering": "engineering",
        "math": "math",
        "physics": "physics",
    },
    "humanities": {
        "history": "history",
        "law": "law",
        "philosophy": "philosophy",
    },
    "social_sciences": {
        "business": "business",
        "economics": "economics",
        "psychology": "psychology",
    },
    "other": {
        "health": "health",
        "other": "other",
    },
}
_MMLU_PRO_SUBSETS = SubsetTree(_MMLU_PRO_SUBSET_TREE)
_EXPLICIT_ANSWER_PATTERNS = (
    pcre.compile(r"(?i)\bthe answer is\s*\(?([A-Z])\)?"),
    pcre.compile(r"(?i)\banswer is\s*\(?([A-Z])\)?"),
    pcre.compile(r"(?i)\banswer\s*[:\-]\s*\(?([A-Z])\)?"),
)
_CHOICE_TOKEN_PATTERN = pcre.compile(r"\b([A-Z])\b")


def _choice_labels(option_count: int) -> list[str]:
    if option_count > len(_OPTION_LABELS):
        raise ValueError(
            f"MMLU-Pro question has {option_count} options but only {len(_OPTION_LABELS)} labels are supported"
        )
    return list(_OPTION_LABELS[:option_count])


def _choice_options(doc: dict[str, Any]) -> list[str]:
    return [
        str(option).strip()
        for option in doc.get("options", [])
        if str(option).strip() and str(option).strip() != "N/A"
    ]


def _preprocess_doc(doc: dict[str, Any]) -> dict[str, Any]:
    processed = dict(doc)
    processed["options"] = _choice_options(doc)
    return processed


def _format_cot_example(doc: dict[str, Any], *, include_answer: bool) -> str:
    options = _choice_options(doc)
    labels = _choice_labels(len(options))
    prompt = "Question:\n"
    prompt += f"{str(doc['question']).strip()}\n"
    prompt += "Options:\n"
    prompt += "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True))
    prompt += "\n"

    if include_answer:
        cot_content = str(doc.get("cot_content") or "").strip()
        cot_content = cot_content.replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
            1,
        )
        if cot_content and not cot_content.startswith("Answer:"):
            cot_content = f"Answer: {cot_content}"
        prompt += cot_content + "\n\n"
        return prompt

    prompt += "Answer: Let's think step by step."
    return prompt


def _build_prompt(
    *,
    subset_value: str,
    fewshot_docs: list[dict[str, Any]],
    doc: dict[str, Any],
) -> str:
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{subset_value}. Think step by step and then finish your answer with "
        "'the answer is (X)' where X is the correct letter choice."
    )
    sections = [header, *(_format_cot_example(example, include_answer=True) for example in fewshot_docs)]
    return "\n\n\n".join(sections) + "\n" + _format_cot_example(doc, include_answer=False)


def _normalize_choice_text(text: Any) -> str:
    normalized = _NON_ALNUM_PATTERN.sub(" ", str(text).lower())
    return normalized.strip()


def _extract_choice_label(text: str, valid_labels: set[str]) -> str:
    response = text or ""
    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        for match in pattern.findall(response):
            candidate = str(match).strip().upper()
            if candidate in valid_labels:
                return candidate

    matches = list(_CHOICE_TOKEN_PATTERN.findall(response))
    for match in reversed(matches):
        candidate = str(match).strip().upper()
        if candidate in valid_labels:
            return candidate
    return _INVALID_CHOICE


def _extract_choice_label_from_text(text: str, options: list[str]) -> str:
    normalized_response = _normalize_choice_text(text)
    if not normalized_response:
        return _INVALID_CHOICE

    labels = _choice_labels(len(options))
    exact_matches = [
        label
        for label, option in zip(labels, options, strict=True)
        if _normalize_choice_text(option) == normalized_response
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contained_matches = [
        (label, len(_normalize_choice_text(option)))
        for label, option in zip(labels, options, strict=True)
        if _normalize_choice_text(option) and _normalize_choice_text(option) in normalized_response
    ]
    if not contained_matches:
        return _INVALID_CHOICE

    contained_matches.sort(key=lambda item: item[1], reverse=True)
    if len(contained_matches) > 1 and contained_matches[0][1] == contained_matches[1][1]:
        return _INVALID_CHOICE
    return contained_matches[0][0]


def _session_tokenizer(session: InferenceSession) -> Any | None:
    return getattr(session, "prepare_tokenizer", None) or getattr(session, "tokenizer", None)


def _session_context_limit(session: InferenceSession) -> int | None:
    tokenizer = _session_tokenizer(session)
    model_length = getattr(getattr(getattr(session, "model", None), "config", None), "max_position_embeddings", None)
    tokenizer_length = getattr(tokenizer, "model_max_length", None)
    candidate_lengths = [
        int(length)
        for length in (
            model_length,
            tokenizer_length,
            getattr(session, "max_input_length", None),
            getattr(session, "context_window", None),
            getattr(session, "max_model_length", None),
            getattr(session, "max_sequence_length", None),
            getattr(session, "max_seq_len", None),
        )
        if isinstance(length, int) and 1 < length < 1_000_000
    ]
    if candidate_lengths:
        return min(candidate_lengths)
    return None


def _prepare_request_for_context_fit(
    session: InferenceSession,
    request: GenerationRequest,
) -> GenerationRequest:
    prepare_requests = getattr(session, "prepare_requests", None)
    if callable(prepare_requests):
        return prepare_requests([request])[0]

    tokenizer = _session_tokenizer(session)
    if tokenizer is None:
        return request

    rendered_prompt = request.rendered_prompt
    if rendered_prompt is None:
        if request.messages is not None:
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                return request
            rendered_prompt = apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        elif request.prompt is not None:
            rendered_prompt = request.prompt
        else:
            return request

    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        padding=False,
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return replace(request, rendered_prompt=rendered_prompt)
    return replace(
        request,
        rendered_prompt=rendered_prompt,
        input_ids=list(input_ids),
    )


def _request_fits_context(
    session: InferenceSession,
    request: GenerationRequest,
) -> tuple[bool, GenerationRequest]:
    prepared_request = _prepare_request_for_context_fit(session, request)
    context_limit = _session_context_limit(session)
    if context_limit is None or prepared_request.input_ids is None:
        return True, prepared_request

    prompt_token_count = len(prepared_request.input_ids)
    return (
        prompt_token_count + prepared_request.max_new_tokens < context_limit,
        prepared_request,
    )


@dataclass(slots=True)
class MMLUPro(TestSuite):
    dataset_path: str = "TIGER-Lab/MMLU-Pro"
    split: str = "test"
    fewshot_split: str = "validation"
    subsets: str | list[str] = "all"
    num_fewshot: int = 5
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False
    apply_chat_template: bool = False
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.0
    _fewshot_by_subset_value: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return "mmlu_pro"
        suffix = "__".join(canonical.replace(".", "_") for canonical in resolved_subsets.canonicals)
        return f"mmlu_pro_{suffix}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        resolved_subsets = self._resolved_subsets()
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": None,
            "split": self.split,
            "fewshot_split": self.fewshot_split,
            "subsets": list(resolved_subsets.canonicals),
            "subset_paths": [list(path) for path in resolved_subsets.paths],
            "subset_kinds": list(resolved_subsets.kinds),
            "selection_mode": resolved_subsets.selection_mode,
            "num_fewshot": self.num_fewshot,
            "streaming": self.streaming,
            "apply_chat_template": self.apply_chat_template,
            "generation_submission_mode": generation_submission_mode,
            "scoring_mode": "generated_choice_label_exact_match",
        }

    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        accuracy = (
            aggregate_scores.get("em,choice_label", 0.0) / processed
            if processed
            else 0.0
        )
        return (
            f"{self.task_name()}: scoring "
            f"accuracy={accuracy:.4f} "
            f"invalid={invalid_predictions}"
        )

    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()

        loaded_docs, dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=None,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )
        docs = self._select_docs([_preprocess_doc(doc) for doc in loaded_docs])
        docs = limit_docs(docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)
        total = len(docs)
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        fewshot_loaded_docs, _fewshot_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=None,
            split=self.fewshot_split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )
        self._fewshot_by_subset_value = defaultdict(list)
        for doc in self._select_docs([_preprocess_doc(doc) for doc in fewshot_loaded_docs]):
            self._fewshot_by_subset_value[normalize_subset_token(doc["category"])].append(doc)

        prepare_bar = manual_progress(
            total,
            title=f"{task_name}: preparing requests",
        )
        prepared_iter = self._iter_prepared_samples(session, docs)
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
            sample = self._score_sample(prepared_sample, output)
            for metric_name, score in sample.scores.items():
                aggregate_scores[metric_name] += score
            invalid_predictions += self._invalid_prediction_count(sample)
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
            "em,choice_label": aggregate_scores.get("em,choice_label", 0.0) / denominator,
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

    def _select_docs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return docs
        selected_subset_values = {
            normalize_subset_token(subset_value)
            for subset_value in resolved_subsets.leaf_values
        }
        selected_docs = [
            doc
            for doc in docs
            if normalize_subset_token(doc.get("category")) in selected_subset_values
        ]
        if selected_docs:
            return selected_docs
        raise ValueError(f"MMLU-Pro subsets {self.subsets!r} are not present in the dataset")

    def _iter_prepared_samples(
        self,
        session: InferenceSession,
        docs: list[dict[str, Any]],
    ) -> Any:
        for index, doc in enumerate(docs):
            subset_value = str(doc["category"]).strip()
            request, fewshot_count = self._build_request_with_backoff(
                session=session,
                subset_value=subset_value,
                doc=doc,
            )
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["answer"]).strip().upper(),
                request=request,
            )

    def _build_request_with_backoff(
        self,
        *,
        session: InferenceSession,
        subset_value: str,
        doc: dict[str, Any],
    ) -> tuple[GenerationRequest, int]:
        subset_token = normalize_subset_token(subset_value)
        subset_fewshots = self._fewshot_by_subset_value.get(subset_token, [])
        max_fewshots = min(self.num_fewshot, len(subset_fewshots))
        fallback_request: GenerationRequest | None = None

        for fewshot_count in range(max_fewshots, -1, -1):
            request = self._build_request(
                subset_value=subset_value,
                fewshot_docs=subset_fewshots[:fewshot_count],
                doc=doc,
            )
            fits_context, prepared_request = _request_fits_context(session, request)
            fallback_request = prepared_request
            if fits_context:
                return prepared_request, fewshot_count

        if fallback_request is None:
            fallback_request = self._build_request(
                subset_value=subset_value,
                fewshot_docs=[],
                doc=doc,
            )
        return fallback_request, 0

    def _build_request(
        self,
        *,
        subset_value: str,
        fewshot_docs: list[dict[str, Any]],
        doc: dict[str, Any],
    ) -> GenerationRequest:
        prompt = _build_prompt(
            subset_value=subset_value,
            fewshot_docs=fewshot_docs,
            doc=doc,
        )
        metadata = {"fewshot_count": len(fewshot_docs)}
        if self.apply_chat_template:
            return GenerationRequest(
                messages=[{"role": "user", "content": prompt}],
                stop=list(_STOP_STRINGS),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                metadata=metadata,
            )
        return GenerationRequest(
            prompt=prompt,
            stop=list(_STOP_STRINGS),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            metadata=metadata,
        )

    def _score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        options = _choice_options(prepared_sample.doc)
        labels = _choice_labels(len(options))
        valid_labels = set(labels)
        predicted_label = _extract_choice_label(output.text, valid_labels)
        if predicted_label == _INVALID_CHOICE:
            predicted_label = _extract_choice_label_from_text(output.text, options)

        gold_label = str(prepared_sample.doc["answer"]).strip().upper()
        predicted_text = (
            options[labels.index(predicted_label)]
            if predicted_label in valid_labels
            else _INVALID_CHOICE
        )
        leaf_subset = _MMLU_PRO_SUBSETS.leaf_subset(prepared_sample.doc.get("category"))
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "choice-label": predicted_label,
                "choice-text": predicted_text,
            },
            scores={
                "em,choice_label": choice_label_exact_match(predicted_label, gold_label),
            },
            metadata={
                "question_id": prepared_sample.doc.get("question_id"),
                "subset": leaf_subset,
                "subset_path": leaf_subset.split("."),
                "subset_kind": "leaf",
                "subset_value": str(prepared_sample.doc.get("category")).strip(),
                "src": prepared_sample.doc.get("src"),
                "answer_index": prepared_sample.doc.get("answer_index"),
                "choice_texts": options,
                "fewshot_count": prepared_sample.request.metadata.get("fewshot_count", 0),
            },
        )

    def _invalid_prediction_count(self, sample: SampleResult) -> int:
        return int(sample.extracted["choice-label"] == _INVALID_CHOICE)

    def _resolved_subsets(self) -> ResolvedSubsets:
        return _MMLU_PRO_SUBSETS.resolve_many(self.subsets)


def mmlu_pro(**kwargs: Any) -> MMLUPro:
    return MMLUPro(**kwargs)
