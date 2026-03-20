from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from datasets import load_dataset

from evalution.engines.base import GenerationRequest, InferenceSession
from evalution.logbar import get_logger, manual_progress, progress, spinner
from evalution.results import SampleResult, TestResult
from evalution.suites.base import TestSuite

GSM8KPlatinumVariant = Literal["base", "cot", "cot_llama", "cot_zeroshot", "default"]

_REGEXES_TO_IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]
_FLEXIBLE_EXTRACT_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"

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
    strict_pattern: str
    strict_group_select: int
    stop_strings: tuple[str, ...]
    prompt_builder: Any
    target_builder: Any
    num_fewshot: int
    fewshots: tuple[dict[str, str], ...]


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
        strict_pattern=r"#### (\-?[0-9\.\,]+)",
        strict_group_select=0,
        stop_strings=("Question:", "</s>", "<|im_end|>"),
        prompt_builder=_base_prompt,
        target_builder=_full_answer,
        num_fewshot=5,
        fewshots=(),
    ),
    "cot": _VariantSpec(
        task_name="gsm8k_platinum_cot",
        strict_pattern=r"The answer is (\-?[0-9\.\,]+).",
        strict_group_select=0,
        stop_strings=("Q:", "</s>", "<|im_end|>"),
        prompt_builder=_cot_prompt,
        target_builder=_numeric_answer,
        num_fewshot=8,
        fewshots=_COT_FEWSHOTS,
    ),
    "cot_zeroshot": _VariantSpec(
        task_name="gsm8k_platinum_cot_zeroshot",
        strict_pattern=r"The answer is (\-?[0-9\.\,]+).",
        strict_group_select=0,
        stop_strings=("Q:", "</s>", "<|im_end|>"),
        prompt_builder=_cot_zeroshot_prompt,
        target_builder=_full_answer,
        num_fewshot=0,
        fewshots=(),
    ),
    "cot_llama": _VariantSpec(
        task_name="gsm8k_platinum_cot_llama",
        strict_pattern=r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))",
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
        with spinner(f"{spec.task_name}: loading dataset"):
            all_docs = list(
                load_dataset(
                    self.dataset_path,
                    self.dataset_name,
                    split=self.split,
                    cache_dir=self.cache_dir,
                )
            )
        docs = all_docs
        if self.limit is not None:
            docs = all_docs[: self.limit]
        logger.info("%s: evaluating %d sample(s)", spec.task_name, len(docs))

        requests: list[GenerationRequest] = []
        targets: list[str] = []
        prepare_iter = progress(docs, title=f"{spec.task_name}: preparing requests")
        for index, doc in enumerate(prepare_iter):
            fewshots = self._select_fewshots(spec=spec, docs=all_docs, doc=doc, index=index)
            request = self._build_request(
                spec=spec,
                doc=doc,
                fewshots=fewshots,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
            requests.append(request)
            targets.append(spec.target_builder(doc))

        aggregate_scores: defaultdict[str, float] = defaultdict(float)
        samples: list[SampleResult] = []
        total = len(requests)
        effective_batch_size = self.batch_size or 1
        invalid_predictions = 0
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
        try:
            for start in range(0, total, effective_batch_size):
                batch_requests = requests[start : start + effective_batch_size]
                batch_outputs = session.generate(batch_requests, batch_size=len(batch_requests))
                for batch_offset, output in enumerate(batch_outputs):
                    index = start + batch_offset
                    doc = docs[index]
                    target = targets[index]
                    strict_prediction = _extract_match(
                        output.text,
                        spec.strict_pattern,
                        group_select=spec.strict_group_select,
                    )
                    flexible_prediction = _extract_match(
                        output.text,
                        _FLEXIBLE_EXTRACT_PATTERN,
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

                    samples.append(
                        SampleResult(
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
                    )

                    processed = len(samples)
                    score_bar.title(
                        self._score_progress_title(
                            task_name=spec.task_name,
                            processed=processed,
                            strict_total=aggregate_scores["exact_match,strict-match"],
                            flexible_total=aggregate_scores["exact_match,flexible-extract"],
                            invalid_predictions=invalid_predictions,
                        )
                    )
                    score_bar.subtitle(
                        f"batch={start // effective_batch_size + 1}/{(total + effective_batch_size - 1) // effective_batch_size}"
                    )
                    score_bar.next().draw()
        finally:
            score_bar.close()

        denominator = len(samples) or 1
        metrics = {
            metric_name: total / denominator
            for metric_name, total in aggregate_scores.items()
        }
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


def _extract_match(
    text: str,
    pattern: str,
    *,
    group_select: int,
    fallback: str = "[invalid]",
) -> str:
    matches = re.findall(pattern, text or "")
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
        normalized = re.sub(pattern, "", normalized)
    normalized = normalized.lower()
    return normalized.strip()
