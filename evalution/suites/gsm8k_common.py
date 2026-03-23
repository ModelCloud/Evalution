# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar, Literal

import pcre

from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.suites.base import BaseTestSuite
from evalution.suites.execution import PreparedSample

GSM8KVariant = Literal["base", "cot", "cot_llama", "cot_zeroshot", "default"]
INVALID_ANSWER = "[invalid]"

# OpenAI's released GSM8K scorer in grade_school_math/dataset.py only accepts
# answers emitted in the `#### number` format. We keep a faithful copy here for
# regression tests against the benchmark-owner parser, but the live suite score
# uses the format-insensitive numeric matcher below.
_GSM8K_REFERENCE_ANSWER_RE = pcre.compile(r"#### (\-?[0-9\.\,]+)")

# MadryLab's GSM8K-Platinum scorer in src/utils.py recovers boxed answers and
# then parses a numeric answer out of the answer section or last line. This is
# also retained for regression tests rather than the live suite score.
_GSM8K_PLATINUM_BOXED_RE = pcre.compile(r"\\boxed\{(?:\\text\{)?([^\\{}]+)\}")

# Format-insensitive numeric extraction used by the live scorer. This keeps
# chat-template wrappers like `The answer is 42.` from zeroing out a correct
# answer purely because it was not rendered as `#### 42`.
_HASH_ANSWER_RE = pcre.compile(r"####\s*(-?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+))")
_ANSWER_LINE_RE = pcre.compile(
    r"(?:the\s+final\s+answer\s+is|final\s+answer\s*:|the\s+answer\s+is|answer\s*:)\s*(.+)",
    pcre.IGNORECASE,
)
_NUMERIC_TOKEN_RE = pcre.compile(r"-?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)")
_PLATINUM_REFERENCE_NUMBER_RE = pcre.compile(r"-?[0-9.]*[0-9]")
_TRAILING_ZERO_RE = pcre.compile(r"\.0+$")


@dataclass(frozen=True, slots=True)
class VariantSpec:
    task_name: str
    stop_strings: tuple[str, ...]
    prompt_builder: Any
    target_builder: Any
    num_fewshot: int
    fewshots: tuple[dict[str, str], ...]


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


def base_prompt(doc: dict[str, Any]) -> str:
    return f"Question: {doc['question']}\nAnswer:"


def cot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA:"


def cot_zeroshot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA: Let's think step by step."


def llama_prompt(doc: dict[str, Any]) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {doc['question']}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


def full_answer(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


def build_variant_specs(task_prefix: str) -> dict[str, VariantSpec]:
    return {
        "base": VariantSpec(
            task_name=task_prefix,
            stop_strings=("Question:", "</s>", "<|im_end|>"),
            prompt_builder=base_prompt,
            target_builder=full_answer,
            num_fewshot=5,
            fewshots=(),
        ),
        "cot": VariantSpec(
            task_name=f"{task_prefix}_cot",
            stop_strings=("Q:", "</s>", "<|im_end|>"),
            prompt_builder=cot_prompt,
            target_builder=full_answer,
            num_fewshot=8,
            fewshots=_COT_FEWSHOTS,
        ),
        "cot_zeroshot": VariantSpec(
            task_name=f"{task_prefix}_cot_zeroshot",
            stop_strings=("Q:", "</s>", "<|im_end|>"),
            prompt_builder=cot_zeroshot_prompt,
            target_builder=full_answer,
            num_fewshot=0,
            fewshots=(),
        ),
        "cot_llama": VariantSpec(
            task_name=f"{task_prefix}_cot_llama",
            stop_strings=("<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"),
            prompt_builder=llama_prompt,
            target_builder=full_answer,
            num_fewshot=8,
            fewshots=_LLAMA_FEWSHOTS,
        ),
    }


def requires_full_doc_materialization(spec: VariantSpec) -> bool:
    return not spec.fewshots and spec.num_fewshot > 0


def extract_gsm8k_reference_answer(text: str) -> str:
    match = _GSM8K_REFERENCE_ANSWER_RE.search(text or "")
    if match is None:
        return INVALID_ANSWER
    return match.group(1).strip().replace(",", "")


def extract_gsm8k_platinum_reference_answer(text: str) -> str:
    output = text or ""
    lowered_without_stars = output.lower().replace("*", "")
    boxed_match = None

    if "answer:" in lowered_without_stars:
        answer_section = output.lower().split("answer: ")[-1]
        boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(answer_section)
        if boxed_match is not None:
            output = f"Answer: {boxed_match.group(1)}"
    else:
        boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(output)
        if boxed_match is not None:
            output = f"Answer: {boxed_match.group(1)}"
        else:
            last_line = output.strip("\n").split("\n")[-1].lower()
            output = f"Answer: {last_line}"

    cleaned = output.replace("*", "").replace("#", "").lower()
    answer_section = cleaned.split("answer: ")[-1].replace(",", "")
    match = _PLATINUM_REFERENCE_NUMBER_RE.search(answer_section)
    if match is None:
        return INVALID_ANSWER
    return _TRAILING_ZERO_RE.sub("", match.group())


def extract_format_insensitive_numeric_answer(text: str) -> str:
    output = text or ""

    hash_match = _HASH_ANSWER_RE.search(output)
    if hash_match is not None:
        return canonicalize_numeric_token(hash_match.group(1))

    answer_matches = list(_ANSWER_LINE_RE.finditer(output))
    if answer_matches:
        extracted = _extract_first_numeric_token(answer_matches[-1].group(1))
        if extracted != INVALID_ANSWER:
            return extracted

    boxed_match = _GSM8K_PLATINUM_BOXED_RE.search(output)
    if boxed_match is not None:
        extracted = _extract_first_numeric_token(boxed_match.group(1))
        if extracted != INVALID_ANSWER:
            return extracted

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        extracted = _extract_last_numeric_token(lines[-1])
        if extracted != INVALID_ANSWER:
            return extracted

    return _extract_last_numeric_token(output)


def canonicalize_numeric_token(token: str) -> str:
    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return INVALID_ANSWER
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return INVALID_ANSWER
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def numbers_equal(left: str, right: str) -> bool:
    if left == INVALID_ANSWER or right == INVALID_ANSWER:
        return False
    return canonicalize_numeric_token(left) == canonicalize_numeric_token(right)


def _extract_first_numeric_token(text: str) -> str:
    match = _NUMERIC_TOKEN_RE.search(text)
    if match is None:
        return INVALID_ANSWER
    return canonicalize_numeric_token(match.group())


def _extract_last_numeric_token(text: str) -> str:
    matches = list(_NUMERIC_TOKEN_RE.finditer(text))
    if not matches:
        return INVALID_ANSWER
    return canonicalize_numeric_token(matches[-1].group())


def gsm8k_reference_target(doc: dict[str, Any]) -> str:
    target = extract_gsm8k_reference_answer(str(doc["answer"]))
    if target == INVALID_ANSWER:
        raise ValueError("GSM8K ground-truth answer is missing the official `#### number` marker")
    return target


def gsm8k_numeric_target(doc: dict[str, Any]) -> str:
    return canonicalize_numeric_token(gsm8k_reference_target(doc))


def gsm8k_platinum_reference_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"]).split("\n#### ")[-1].replace(",", "").strip()


def gsm8k_platinum_numeric_target(doc: dict[str, Any]) -> str:
    target = canonicalize_numeric_token(gsm8k_platinum_reference_target(doc))
    if target == INVALID_ANSWER:
        raise ValueError("GSM8K-Platinum ground-truth answer could not be parsed into a numeric target")
    return target


@dataclass(slots=True)
class BaseGSM8KSuite(BaseTestSuite):
    variant: GSM8KVariant = "cot"
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool | None = None
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    fewshot_seed: int = 0

    VARIANTS: ClassVar[dict[str, VariantSpec]]
    INCLUDE_CLEANING_STATUS: ClassVar[bool] = False
    PRIMARY_METRIC: ClassVar[str] = "accuracy,numeric"
    NUMERIC_EXTRACT_KEY: ClassVar[str] = "numeric-extract"
    SCORING_MODE: ClassVar[str]

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        raise NotImplementedError

    def _resolved_variant(self) -> tuple[str, VariantSpec]:
        variant_name = "base" if self.variant == "default" else self.variant
        return variant_name, self.VARIANTS[variant_name]

    def _resolved_fewshot_as_multiturn(self) -> bool:
        return (
            self.fewshot_as_multiturn
            if self.fewshot_as_multiturn is not None
            else self.apply_chat_template
        )

    def task_name(self) -> str:
        return self._resolved_variant()[1].task_name

    def requires_full_doc_materialization(self) -> bool:
        return requires_full_doc_materialization(self._resolved_variant()[1])

    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        numeric_total = aggregate_scores.get(self.PRIMARY_METRIC, 0.0)
        if processed == 0:
            numeric_score = 0.0
        else:
            numeric_score = numeric_total / processed
        return (
            f"{self.task_name()}: scoring "
            f"numeric={numeric_score:.4f} "
            f"invalid={invalid_predictions}"
        )

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        variant_name, spec = self._resolved_variant()
        return {
            **self.base_result_metadata(
                generation_submission_mode=generation_submission_mode,
            ),
            "variant": variant_name,
            "num_fewshot": spec.num_fewshot,
            "apply_chat_template": self.apply_chat_template,
            "fewshot_as_multiturn": self._resolved_fewshot_as_multiturn(),
            "scoring_mode": self.SCORING_MODE,
            "primary_metric": self.PRIMARY_METRIC,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        _, spec = self._resolved_variant()
        fewshot_docs = docs if requires_full_doc_materialization(spec) else list(spec.fewshots)
        fewshot_as_multiturn = self._resolved_fewshot_as_multiturn()
        for index, doc in enumerate(docs):
            fewshots = self._select_fewshots(
                spec=spec,
                docs=fewshot_docs,
                doc=doc,
                index=index,
            )
            yield PreparedSample(
                index=index,
                doc=doc,
                target=self.numeric_target_from_doc(doc),
                request=self._build_request(
                    spec=spec,
                    doc=doc,
                    fewshots=fewshots,
                    fewshot_as_multiturn=fewshot_as_multiturn,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        numeric_prediction = extract_format_insensitive_numeric_answer(output.text)
        scores = {
            self.PRIMARY_METRIC: float(numbers_equal(numeric_prediction, prepared_sample.target)),
        }
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                self.NUMERIC_EXTRACT_KEY: numeric_prediction,
            },
            scores=scores,
            metadata=self._sample_metadata(prepared_sample.doc),
        )

    def invalid_prediction_count(self, sample: SampleResult) -> int:
        return int(sample.extracted[self.NUMERIC_EXTRACT_KEY] == INVALID_ANSWER)

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        if self.INCLUDE_CLEANING_STATUS:
            return {
                "cleaning_status": doc.get("cleaning_status"),
            }
        return {}

    def _build_request(
        self,
        *,
        spec: VariantSpec,
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
        spec: VariantSpec,
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
        spec: VariantSpec,
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
