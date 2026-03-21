from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pcre

from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.suites.base import BaseTestSuite
from evalution.suites.execution import PreparedSample

GSM8KVariant = Literal["base", "cot", "cot_llama", "cot_zeroshot", "default"]


# Compile regexes through PyPcre so extraction stays on the faster backend.
def compile_regex(pattern: str) -> Any:
    return pcre.compile(pattern)


REGEXES_TO_IGNORE = tuple(
    compile_regex(pattern)
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$")
)
FLEXIBLE_EXTRACT_REGEX = compile_regex(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
FLEXIBLE_EXTRACT_PATTERN = FLEXIBLE_EXTRACT_REGEX

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
class VariantSpec:
    task_name: str
    strict_regex: Any
    strict_group_select: int
    stop_strings: tuple[str, ...]
    prompt_builder: Any
    target_builder: Any
    num_fewshot: int
    fewshots: tuple[dict[str, str], ...]


    # Render the plain base prompt used by the non-CoT variant.
def base_prompt(doc: dict[str, Any]) -> str:
    return f"Question: {doc['question']}\nAnswer:"


# Render the short chain-of-thought prompt used by the default variant.
def cot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA:"


# Render the zero-shot chain-of-thought prompt.
def cot_zeroshot_prompt(doc: dict[str, Any]) -> str:
    return f"Q: {doc['question']}\nA: Let's think step by step."


# Render the Llama-style instruction prompt ending in a final-answer marker.
def llama_prompt(doc: dict[str, Any]) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {doc['question']}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


# Preserve the full reference answer for variants that score on the complete string.
def full_answer(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


# Extract just the numeric answer from GSM8K-style `####` targets.
def numeric_answer(doc: dict[str, Any]) -> str:
    return str(doc["answer"]).split("####")[-1].strip()


# Build the variant table for a concrete suite name prefix.
def build_variant_specs(task_prefix: str) -> dict[str, VariantSpec]:
    return {
        "base": VariantSpec(
            task_name=task_prefix,
            strict_regex=compile_regex(r"#### (\-?[0-9\.\,]+)"),
            strict_group_select=0,
            stop_strings=("Question:", "</s>", "<|im_end|>"),
            prompt_builder=base_prompt,
            target_builder=full_answer,
            num_fewshot=5,
            fewshots=(),
        ),
        "cot": VariantSpec(
            task_name=f"{task_prefix}_cot",
            strict_regex=compile_regex(r"The answer is (\-?[0-9\.\,]+)."),
            strict_group_select=0,
            stop_strings=("Q:", "</s>", "<|im_end|>"),
            prompt_builder=cot_prompt,
            target_builder=numeric_answer,
            num_fewshot=8,
            fewshots=_COT_FEWSHOTS,
        ),
        "cot_zeroshot": VariantSpec(
            task_name=f"{task_prefix}_cot_zeroshot",
            strict_regex=compile_regex(r"The answer is (\-?[0-9\.\,]+)."),
            strict_group_select=0,
            stop_strings=("Q:", "</s>", "<|im_end|>"),
            prompt_builder=cot_zeroshot_prompt,
            target_builder=full_answer,
            num_fewshot=0,
            fewshots=(),
        ),
        "cot_llama": VariantSpec(
            task_name=f"{task_prefix}_cot_llama",
            strict_regex=compile_regex(r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"),
            strict_group_select=-1,
            stop_strings=("<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"),
            prompt_builder=llama_prompt,
            target_builder=numeric_answer,
            num_fewshot=8,
            fewshots=_LLAMA_FEWSHOTS,
        ),
    }


# Determine whether random fewshot sampling requires materialized docs.
def requires_full_doc_materialization(spec: VariantSpec) -> bool:
    return not spec.fewshots and spec.num_fewshot > 0


# Extract a regex match and normalize tuple captures into a single string.
def extract_match(
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


# Compare answers after suite-specific normalization.
def exact_match(prediction: str, target: str) -> bool:
    return normalize(prediction) == normalize(target)


# Strip formatting noise and case differences before exact-match scoring.
def normalize(text: str) -> str:
    normalized = text
    for pattern in REGEXES_TO_IGNORE:
        normalized = pattern.sub("", normalized)
    normalized = normalized.lower()
    return normalized.strip()


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

    # Resolve `default` to the concrete variant used by the suite.
    def _resolved_variant(self) -> tuple[str, VariantSpec]:
        variant_name = "base" if self.variant == "default" else self.variant
        return variant_name, self.VARIANTS[variant_name]

    # Default multiturn fewshot behavior to the chat-template setting.
    def _resolved_fewshot_as_multiturn(self) -> bool:
        return (
            self.fewshot_as_multiturn
            if self.fewshot_as_multiturn is not None
            else self.apply_chat_template
        )

    # Expose the resolved task name through the shared base pipeline.
    def task_name(self) -> str:
        return self._resolved_variant()[1].task_name

    # Materialize docs only for variants that sample fewshots from the dataset itself.
    def requires_full_doc_materialization(self) -> bool:
        return requires_full_doc_materialization(self._resolved_variant()[1])

    # Render the live GSM8K scoring summary shown in the progress bar.
    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        strict_total = aggregate_scores.get("exact_match,strict-match", 0.0)
        flexible_total = aggregate_scores.get("exact_match,flexible-extract", 0.0)
        if processed == 0:
            strict_score = 0.0
            flexible_score = 0.0
        else:
            strict_score = strict_total / processed
            flexible_score = flexible_total / processed
        return (
            f"{self.task_name()}: scoring "
            f"strict={strict_score:.4f} "
            f"flex={flexible_score:.4f} "
            f"invalid={invalid_predictions}"
        )

    # Return result metadata common to all GSM8K-family suites.
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
        }

    # Build generation requests and scoring targets for each dataset row.
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
                target=spec.target_builder(doc),
                request=self._build_request(
                    spec=spec,
                    doc=doc,
                    fewshots=fewshots,
                    fewshot_as_multiturn=fewshot_as_multiturn,
                ),
            )

    # Score a single model output using strict and flexible extraction rules.
    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        _, spec = self._resolved_variant()
        strict_prediction = extract_match(
            output.text,
            spec.strict_regex,
            group_select=spec.strict_group_select,
        )
        flexible_prediction = extract_match(
            output.text,
            FLEXIBLE_EXTRACT_REGEX,
            group_select=-1,
        )
        scores = {
            "exact_match,strict-match": float(exact_match(strict_prediction, prepared_sample.target)),
            "exact_match,flexible-extract": float(exact_match(flexible_prediction, prepared_sample.target)),
        }
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "strict-match": strict_prediction,
                "flexible-extract": flexible_prediction,
            },
            scores=scores,
            metadata=self._sample_metadata(prepared_sample.doc),
        )

    # Count flexible-extract misses as invalid predictions in progress reporting.
    def invalid_prediction_count(self, sample: SampleResult) -> int:
        return int(sample.extracted["flexible-extract"] == "[invalid]")

    # Attach suite-specific row metadata to the per-sample result.
    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        if self.INCLUDE_CLEANING_STATUS:
            return {
                "cleaning_status": doc.get("cleaning_status"),
            }
        return {}

    # Build either plain or chat-formatted requests for the selected variant.
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

    # Concatenate fewshots and the current prompt into the plain-text request body.
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

    # Choose static fewshots or sample deterministic in-dataset fewshots per row.
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
