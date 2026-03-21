from __future__ import annotations

import atexit
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import islice
from queue import Full, Queue
from threading import Event, Lock
from time import perf_counter
from typing import Any, Literal

from datasets import load_dataset
import pcre

from evalution.engines.base import GenerationRequest, InferenceSession
from evalution.logbar import get_logger, manual_progress, spinner
from evalution.results import SampleResult, TestResult
from evalution.suites.base import TestSuite

GSM8KPlatinumVariant = Literal["base", "cot", "cot_llama", "cot_zeroshot", "default"]

_AUTO_BATCH_PREVIEW_ROWS = 256
_PRETOKENIZED_POOL_MULTIPLIER = 2
_BATCH_PREFETCH_PUT_TIMEOUT_S = 0.1
_PRETOKENIZED_REFILL_COALESCE_S = 0.01
_PREFETCH_EXECUTOR_LOCK = Lock()
_PREFETCH_EXECUTOR: ThreadPoolExecutor | None = None


def _compile_regex(pattern: str) -> Any:
    return pcre.compile(pattern)


_REGEXES_TO_IGNORE = tuple(
    _compile_regex(pattern)
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$")
)
_FLEXIBLE_EXTRACT_REGEX = _compile_regex(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

_COT_FEWSHOTS = (
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
)

_LLAMA_FEWSHOTS = (
    {
        "question": _COT_FEWSHOTS[0]["question"],
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
    },
    {
        "question": _COT_FEWSHOTS[1]["question"],
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
    },
    {
        "question": _COT_FEWSHOTS[2]["question"],
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
    },
    {
        "question": _COT_FEWSHOTS[3]["question"],
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
    },
    {
        "question": _COT_FEWSHOTS[4]["question"],
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
    },
    {
        "question": _COT_FEWSHOTS[5]["question"],
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
    },
    {
        "question": _COT_FEWSHOTS[6]["question"],
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
    },
    {
        "question": _COT_FEWSHOTS[7]["question"],
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
    },
)


@dataclass(frozen=True, slots=True)
class _VariantSpec:
    task_name: str
    strict_regex: Any
    strict_group_select: int
    stop_strings: tuple[str, ...]
    prompt_builder: Any
    target_builder: Any
    num_fewshot: int
    fewshots: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    index: int
    doc: dict[str, Any]
    target: str
    request: GenerationRequest


@dataclass(frozen=True, slots=True)
class _PrefetchFailure:
    error: BaseException


def _base_prompt(doc: dict[str, Any]) -> str:
    return f"Question: {doc['question']}\nAnswer:"


def _cot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA:"


def _cot_zeroshot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA: Let's think step by step."


def _llama_prompt(doc: dict[str, Any]) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {doc['question']}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


def _full_answer(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


def _numeric_answer(doc: dict[str, Any]) -> str:
    return str(doc["answer"]).split("####")[-1].strip()


_VARIANTS: dict[str, _VariantSpec] = {
    "base": _VariantSpec(
        task_name="gsm8k_platinum",
        strict_regex=_compile_regex(r"#### (\-?[0-9\.\,]+)"),
        strict_group_select=0,
        stop_strings=("Question:", "</s>", "<|im_end|>"),
        prompt_builder=_base_prompt,
        target_builder=_full_answer,
        num_fewshot=5,
        fewshots=(),
    ),
    "cot": _VariantSpec(
        task_name="gsm8k_platinum_cot",
        strict_regex=_compile_regex(r"The answer is (\-?[0-9\.\,]+)."),
        strict_group_select=0,
        stop_strings=("Q:", "</s>", "<|im_end|>"),
        prompt_builder=_cot_prompt,
        target_builder=_numeric_answer,
        num_fewshot=8,
        fewshots=_COT_FEWSHOTS,
    ),
    "cot_zeroshot": _VariantSpec(
        task_name="gsm8k_platinum_cot_zeroshot",
        strict_regex=_compile_regex(r"The answer is (\-?[0-9\.\,]+)."),
        strict_group_select=0,
        stop_strings=("Q:", "</s>", "<|im_end|>"),
        prompt_builder=_cot_zeroshot_prompt,
        target_builder=_full_answer,
        num_fewshot=0,
        fewshots=(),
    ),
    "cot_llama": _VariantSpec(
        task_name="gsm8k_platinum_cot_llama",
        strict_regex=_compile_regex(r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"),
        strict_group_select=-1,
        stop_strings=("<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"),
        prompt_builder=_llama_prompt,
        target_builder=_numeric_answer,
        num_fewshot=8,
        fewshots=_LLAMA_FEWSHOTS,
    ),
}


@dataclass(slots=True)
class GSM8KPlatinum(TestSuite):
    variant: GSM8KPlatinumVariant = "cot"
    dataset_path: str = "madrylab/gsm8k-platinum"
    dataset_name: str = "main"
    split: str = "test"
    limit: int | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool | None = None
    batch_size: int | None = None
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    fewshot_seed: int = 0
    cache_dir: str | None = None
    streaming: bool = False

    def evaluate(self, session: InferenceSession) -> TestResult:
        variant_name = "base" if self.variant == "default" else self.variant
        spec = _VARIANTS[variant_name]
        logger = get_logger()
        fewshot_as_multiturn = (
            self.fewshot_as_multiturn
            if self.fewshot_as_multiturn is not None
            else self.apply_chat_template
        )

        logger.info(
            "loading dataset %s/%s split=%s for %s",
            self.dataset_path,
            self.dataset_name,
            self.split,
            spec.task_name,
        )
        dataset_load_started = perf_counter()
        with spinner(f"{spec.task_name}: loading dataset"):
            loaded_docs = load_dataset(
                self.dataset_path,
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=self.streaming,
            )
        dataset_load_wall_s = perf_counter() - dataset_load_started
        logger.info("%s: dataset load wall_time=%.3fs", spec.task_name, dataset_load_wall_s)
        docs = _limit_docs(loaded_docs, self.limit)
        if _requires_full_doc_materialization(spec):
            docs = list(docs)
            fewshot_docs = docs
        else:
            fewshot_docs = list(spec.fewshots)

        total = _doc_count(docs, loaded_docs=loaded_docs, limit=self.limit, split=self.split)
        logger.info("%s: evaluating %d sample(s)", spec.task_name, total)

        prepare_bar = manual_progress(
            total,
            title=f"{spec.task_name}: preparing requests",
        )
        prepared_iter = self._iter_prepared_samples(
            spec=spec,
            docs=docs,
            fewshot_docs=fewshot_docs,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )
        preview_size = (
            min(total, _AUTO_BATCH_PREVIEW_ROWS)
            if _needs_batch_size_preview(self.batch_size, session)
            else 0
        )
        preview_samples = _collect_preview_samples(
            prepared_iter,
            preview_size=preview_size,
            prepare_bar=prepare_bar,
        )
        preview_samples = _prepare_batch_for_session(session, preview_samples)

        aggregate_scores: defaultdict[str, float] = defaultdict(float)
        samples_by_index: list[SampleResult | None] = [None] * total
        effective_batch_size = (
            self.batch_size
            or _session_batch_size(session, [sample.request for sample in preview_samples])
            or 1
        )
        logger.info("%s: using batch_size=%d", spec.task_name, effective_batch_size)
        logger.info(
            "%s: request preparation mode=%s",
            spec.task_name,
            "threaded_prefetch"
            if callable(getattr(session, "prepare_requests", None))
            else "inline",
        )
        generate_continuous = getattr(session, "generate_continuous", None)
        use_continuous_generation = callable(generate_continuous)
        logger.info(
            "%s: generation submission mode=%s",
            spec.task_name,
            "continuous_refill" if use_continuous_generation else "fixed_batches",
        )
        invalid_predictions = 0
        generation_wall_s = 0.0
        scoring_wall_s = 0.0
        processed_count = 0
        score_bar = manual_progress(
            total,
            title=self._score_progress_title(
                task_name=spec.task_name,
                processed=0,
                strict_total=0.0,
                flexible_total=0.0,
                invalid_predictions=0,
            ),
            subtitle=f"batch_size={effective_batch_size}",
        )

        def score_output(prepared_sample: _PreparedSample, output: GenerationOutput) -> None:
            nonlocal invalid_predictions
            nonlocal scoring_wall_s
            nonlocal processed_count
            scoring_started = perf_counter()
            index = prepared_sample.index
            doc = prepared_sample.doc
            target = prepared_sample.target
            strict_prediction = _extract_match(
                output.text,
                spec.strict_regex,
                group_select=spec.strict_group_select,
            )
            flexible_prediction = _extract_match(
                output.text,
                _FLEXIBLE_EXTRACT_REGEX,
                group_select=-1,
            )
            scores = {
                "exact_match,strict-match": float(_exact_match(strict_prediction, target)),
                "exact_match,flexible-extract": float(_exact_match(flexible_prediction, target)),
            }
            for metric_name, score in scores.items():
                aggregate_scores[metric_name] += score
            if flexible_prediction == "[invalid]":
                invalid_predictions += 1

            samples_by_index[index] = SampleResult(
                index=index,
                prompt=output.prompt,
                target=target,
                prediction=output.text,
                extracted={
                    "strict-match": strict_prediction,
                    "flexible-extract": flexible_prediction,
                },
                scores=scores,
                metadata={
                    "cleaning_status": doc.get("cleaning_status"),
                },
            )

            processed_count += 1
            score_bar.title(
                self._score_progress_title(
                    task_name=spec.task_name,
                    processed=processed_count,
                    strict_total=aggregate_scores["exact_match,strict-match"],
                    flexible_total=aggregate_scores["exact_match,flexible-extract"],
                    invalid_predictions=invalid_predictions,
                )
            )
            score_bar.next().draw()
            scoring_wall_s += perf_counter() - scoring_started

        try:
            if use_continuous_generation:
                sample_by_request_key: dict[int, _PreparedSample] = {}

                def iter_request_stream() -> Any:
                    request_key = 0
                    prefetched_samples = _iter_prefetched_samples(
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
                for prepared_batch in _iter_prefetched_batches(
                    session,
                    preview_samples,
                    prepared_iter,
                    batch_size=effective_batch_size,
                    prepare_bar=prepare_bar,
                ):
                    batch_index += 1
                    batch_requests = [sample.request for sample in prepared_batch]
                    generation_started = perf_counter()
                    batch_outputs = session.generate(batch_requests, batch_size=len(batch_requests))
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
            prepare_bar.title(f"{spec.task_name}: prepared requests")
            prepare_bar.draw()
            prepare_bar.close()
            score_bar.close()

        samples = [sample for sample in samples_by_index if sample is not None]
        denominator = len(samples) or 1
        metrics = {
            metric_name: total / denominator
            for metric_name, total in aggregate_scores.items()
        }
        logger.info(
            "%s: wall_times dataset_load=%.3fs generation=%.3fs scoring=%.3fs",
            spec.task_name,
            dataset_load_wall_s,
            generation_wall_s,
            scoring_wall_s,
        )
        logger.info("%s: metrics=%s", spec.task_name, metrics)
        return TestResult(
            name=spec.task_name,
            metrics=metrics,
            samples=samples,
            metadata={
                "dataset_path": self.dataset_path,
                "dataset_name": self.dataset_name,
                "split": self.split,
                "variant": variant_name,
                "num_fewshot": spec.num_fewshot,
                "apply_chat_template": self.apply_chat_template,
                "fewshot_as_multiturn": fewshot_as_multiturn,
                "streaming": self.streaming,
                "generation_submission_mode": (
                    "continuous_refill" if use_continuous_generation else "fixed_batches"
                ),
            },
        )

    @staticmethod
    def _score_progress_title(
        *,
        task_name: str,
        processed: int,
        strict_total: float,
        flexible_total: float,
        invalid_predictions: int,
    ) -> str:
        if processed == 0:
            strict_score = 0.0
            flexible_score = 0.0
        else:
            strict_score = strict_total / processed
            flexible_score = flexible_total / processed
        return (
            f"{task_name}: scoring "
            f"strict={strict_score:.4f} "
            f"flex={flexible_score:.4f} "
            f"invalid={invalid_predictions}"
        )

    def _build_request(
        self,
        *,
        spec: _VariantSpec,
        doc: dict[str, Any],
        fewshots: list[dict[str, str]],
        fewshot_as_multiturn: bool,
    ) -> GenerationRequest:
        if self.apply_chat_template:
            if fewshot_as_multiturn:
                messages: list[dict[str, str]] = []
                for fewshot in fewshots:
                    messages.append({"role": "user", "content": spec.prompt_builder(fewshot)})
                    messages.append({"role": "assistant", "content": fewshot["target"]})
                messages.append({"role": "user", "content": spec.prompt_builder(doc)})
            else:
                messages = [{"role": "user", "content": self._build_plain_prompt(spec, doc, fewshots)}]
            return GenerationRequest(
                messages=messages,
                stop=list(spec.stop_strings),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )

        return GenerationRequest(
            prompt=self._build_plain_prompt(spec, doc, fewshots),
            stop=list(spec.stop_strings),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

    def _iter_prepared_samples(
        self,
        *,
        spec: _VariantSpec,
        docs: list[dict[str, Any]] | Any,
        fewshot_docs: list[dict[str, Any]],
        fewshot_as_multiturn: bool,
    ) -> Any:
        for index, doc in enumerate(docs):
            fewshots = self._select_fewshots(
                spec=spec,
                docs=fewshot_docs,
                doc=doc,
                index=index,
            )
            request = self._build_request(
                spec=spec,
                doc=doc,
                fewshots=fewshots,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
            yield _PreparedSample(
                index=index,
                doc=doc,
                target=spec.target_builder(doc),
                request=request,
            )

    def _build_plain_prompt(
        self,
        spec: _VariantSpec,
        doc: dict[str, Any],
        fewshots: list[dict[str, str]],
    ) -> str:
        parts: list[str] = []
        for fewshot in fewshots:
            parts.append(spec.prompt_builder(fewshot))
            parts.append(" ")
            parts.append(fewshot["target"])
            parts.append("\n\n")
        parts.append(spec.prompt_builder(doc))
        return "".join(parts)

    def _select_fewshots(
        self,
        *,
        spec: _VariantSpec,
        docs: list[dict[str, Any]],
        doc: dict[str, Any],
        index: int,
    ) -> list[dict[str, str]]:
        if spec.fewshots:
            return list(spec.fewshots[: spec.num_fewshot])
        if spec.num_fewshot == 0:
            return []

        rng = random.Random(self.fewshot_seed + index)
        population = [candidate for candidate in docs if candidate != doc]
        sampled = rng.sample(population, k=min(spec.num_fewshot, len(population)))
        return [
            {
                "question": str(candidate["question"]),
                "target": spec.target_builder(candidate),
            }
            for candidate in sampled
        ]


def gsm8k_platinum(**kwargs: Any) -> GSM8KPlatinum:
    return GSM8KPlatinum(**kwargs)


def _session_batch_size(
    session: InferenceSession,
    requests: list[GenerationRequest],
) -> int | None:
    resolver = getattr(session, "resolve_batch_size", None)
    if callable(resolver):
        resolved_batch_size = resolver(requests)
        if resolved_batch_size is not None:
            return int(resolved_batch_size)

    batch_size = getattr(session, "batch_size", None)
    if batch_size is not None:
        return int(batch_size)

    config = getattr(session, "config", None)
    config_batch_size = getattr(config, "batch_size", None)
    if config_batch_size is not None:
        return int(config_batch_size)
    return None


def _needs_batch_size_preview(
    suite_batch_size: int | None,
    session: InferenceSession,
) -> bool:
    if suite_batch_size is not None:
        return False

    batch_size = getattr(session, "batch_size", None)
    if isinstance(batch_size, int):
        return False

    config = getattr(session, "config", None)
    config_batch_size = getattr(config, "batch_size", None)
    if isinstance(config_batch_size, int):
        return False

    return True


def _collect_preview_samples(
    prepared_iter: Any,
    *,
    preview_size: int,
    prepare_bar: Any,
) -> list[_PreparedSample]:
    preview_samples: list[_PreparedSample] = []
    for sample in islice(prepared_iter, preview_size):
        preview_samples.append(sample)
        prepare_bar.next().draw()
    return preview_samples


def _prepare_batch_for_session(
    session: InferenceSession,
    batch: list[_PreparedSample],
) -> list[_PreparedSample]:
    if not batch:
        return batch

    prepare_requests = getattr(session, "prepare_requests", None)
    if not callable(prepare_requests):
        return batch

    prepared_requests = prepare_requests([sample.request for sample in batch])
    return [
        replace(sample, request=prepared_request)
        for sample, prepared_request in zip(batch, prepared_requests, strict=True)
    ]


def _pretokenized_pool_size(batch_size: int) -> int:
    return max(batch_size, batch_size * _PRETOKENIZED_POOL_MULTIPLIER)


def _prefetch_executor() -> ThreadPoolExecutor:
    global _PREFETCH_EXECUTOR
    with _PREFETCH_EXECUTOR_LOCK:
        if _PREFETCH_EXECUTOR is None:
            _PREFETCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="evalution-prefetch",
            )
    return _PREFETCH_EXECUTOR


def _shutdown_prefetch_executor() -> None:
    global _PREFETCH_EXECUTOR
    with _PREFETCH_EXECUTOR_LOCK:
        executor = _PREFETCH_EXECUTOR
        _PREFETCH_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=False)


atexit.register(_shutdown_prefetch_executor)


def _iter_prefetched_samples(
    session: InferenceSession,
    preview_samples: list[_PreparedSample],
    prepared_iter: Any,
    *,
    batch_size: int,
    prepare_bar: Any,
    pool_size: int | None = None,
) -> Any:
    sentinel = object()
    queue_maxsize = pool_size or _pretokenized_pool_size(batch_size)
    queue: Queue[Any] = Queue(maxsize=queue_maxsize)
    cancelled = Event()

    def put_prefetched(item: Any) -> bool:
        while not cancelled.is_set():
            try:
                queue.put(item, timeout=_BATCH_PREFETCH_PUT_TIMEOUT_S)
                return True
            except Full:
                continue
        return False

    def worker() -> None:
        try:
            while not cancelled.is_set():
                available_slots = max(0, queue_maxsize - queue.qsize())
                if available_slots <= 0:
                    cancelled.wait(_BATCH_PREFETCH_PUT_TIMEOUT_S)
                    continue
                if 0 < available_slots < batch_size:
                    cancelled.wait(_PRETOKENIZED_REFILL_COALESCE_S)
                    available_slots = max(0, queue_maxsize - queue.qsize())
                    if available_slots <= 0:
                        continue
                chunk_size = min(batch_size, available_slots)
                chunk = list(islice(prepared_iter, chunk_size))
                if not chunk:
                    break
                prepared_chunk = _prepare_batch_for_session(session, chunk)
                for sample in prepared_chunk:
                    if not put_prefetched(sample):
                        return
        except BaseException as exc:
            put_prefetched(_PrefetchFailure(exc))
        finally:
            put_prefetched(sentinel)

    future = _prefetch_executor().submit(worker)
    try:
        for sample in preview_samples:
            yield sample

        while True:
            prefetched = queue.get()
            if prefetched is sentinel:
                break
            if isinstance(prefetched, _PrefetchFailure):
                raise prefetched.error
            yield prefetched
            prepare_bar.next().draw()
    finally:
        cancelled.set()
        close_prepared_iter = getattr(prepared_iter, "close", None)
        if callable(close_prepared_iter):
            close_prepared_iter()
        with suppress(Exception):
            future.result(timeout=1.0)


def _iter_prefetched_batches(
    session: InferenceSession,
    preview_samples: list[_PreparedSample],
    prepared_iter: Any,
    *,
    batch_size: int,
    prepare_bar: Any,
    pool_size: int | None = None,
) -> Any:
    batch: list[_PreparedSample] = []
    for sample in _iter_prefetched_samples(
        session,
        preview_samples,
        prepared_iter,
        batch_size=batch_size,
        prepare_bar=prepare_bar,
        pool_size=pool_size,
    ):
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _limit_docs(docs: Any, limit: int | None) -> Any:
    if limit is None:
        return docs
    if hasattr(docs, "select") and hasattr(docs, "__len__"):
        return docs.select(range(min(limit, len(docs))))
    return islice(docs, limit)


def _doc_count(
    docs: Any,
    *,
    loaded_docs: Any,
    limit: int | None,
    split: str,
) -> int:
    if hasattr(docs, "__len__"):
        count = len(docs)
        return min(limit, count) if limit is not None else count

    split_info = getattr(getattr(loaded_docs, "info", None), "splits", {}).get(split)
    if split_info is not None and getattr(split_info, "num_examples", None) is not None:
        count = int(split_info.num_examples)
        return min(limit, count) if limit is not None else count

    if limit is not None:
        return int(limit)

    raise ValueError(
        "streaming dataset row count is unavailable; set `limit` or use a dataset split with known num_examples"
    )


def _requires_full_doc_materialization(spec: _VariantSpec) -> bool:
    return not spec.fewshots and spec.num_fewshot > 0


def _extract_match(
    text: str,
    pattern: Any,
    *,
    group_select: int,
    fallback: str = "[invalid]",
) -> str:
    matches = pattern.findall(text or "")
    if not matches:
        return fallback
    match = matches[group_select]
    if isinstance(match, tuple):
        match = next((candidate for candidate in match if candidate), fallback)
    return str(match).strip() or fallback


def _exact_match(prediction: str, target: str) -> bool:
    return _normalize(prediction) == _normalize(target)


def _normalize(text: str) -> str:
    normalized = text
    for pattern in _REGEXES_TO_IGNORE:
        normalized = pattern.sub("", normalized)
    normalized = normalized.lower()
    return normalized.strip()
