# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels
from evalution.datasets.wnli_es import load_wnli_es_dataset
from evalution.scorers.classification import matthews_corrcoef

# Keep the supported SpanishBench task names explicit so registry wiring and validation
# stay aligned with the upstream family members that are currently implementable here.
SPANISH_BENCH_TASKS = (
    "copa_es",
    "escola",
    "openbookqa_es",
    "paws_es_spanish_bench",
    "wnli_es",
    "xnli_es_spanish_bench",
)


@dataclass(frozen=True, slots=True)
class _SpanishBenchConfig:
    # Capture the canonical dataset location and split for each SpanishBench task.
    dataset_path: str
    dataset_name: str | None
    split: str


_SPANISH_BENCH_CONFIGS = {
    "copa_es": _SpanishBenchConfig(
        dataset_path="BSC-LT/COPA-es",
        dataset_name=None,
        split="test",
    ),
    "escola": _SpanishBenchConfig(
        dataset_path="nbel/EsCoLA",
        dataset_name=None,
        split="validation",
    ),
    "openbookqa_es": _SpanishBenchConfig(
        dataset_path="BSC-LT/openbookqa-es",
        dataset_name=None,
        split="test",
    ),
    "paws_es_spanish_bench": _SpanishBenchConfig(
        dataset_path="paws-x",
        dataset_name="es",
        split="test",
    ),
    "wnli_es": _SpanishBenchConfig(
        dataset_path="PlanTL-GOB-ES/wnli-es",
        dataset_name=None,
        split="validation",
    ),
    "xnli_es_spanish_bench": _SpanishBenchConfig(
        dataset_path="xnli",
        dataset_name="es",
        split="validation",
    ),
}


def _normalize_inline_text(text: str) -> str:
    # Collapse repeated whitespace so benchmark-specific prompt joins stay stable across dataset rows.
    return " ".join(str(text).split())


def _lowercase_first_letter(text: str) -> str:
    # Lowercasing the first token preserves the benchmark's natural sentence continuation prompts.
    normalized = _normalize_inline_text(text).strip()
    if not normalized:
        return normalized
    return normalized[:1].lower() + normalized[1:]


def _strip_terminal_punctuation(text: str, *, marks: str) -> str:
    # Remove one trailing punctuation mark when the benchmark prompt splices another connector after the text.
    normalized = _normalize_inline_text(text).strip()
    if normalized.endswith(tuple(marks)):
        return normalized[:-1].rstrip()
    return normalized


def _copa_es_connector(question: str) -> str:
    # Mirror the Spanish causal connector wording used by the public benchmark prompts.
    return {
        "cause": "porque",
        "effect": "y por lo tanto",
    }[question]


def _copa_es_prompt(premise: str, question: str) -> str:
    return f"{_strip_terminal_punctuation(premise, marks='.,!?')} {_copa_es_connector(question)}"


def _escola_prompt(sentence: str) -> str:
    return (
        f"{_normalize_inline_text(sentence).strip()}\n"
        "Pregunta: ¿Tiene sentido esta frase?\n"
        "Respuesta:"
    )


def _paws_es_choices(sentence1: str, sentence2: str) -> list[str]:
    sentence1_text = _strip_terminal_punctuation(sentence1, marks=".,;")
    sentence2_text = _lowercase_first_letter(sentence2)
    return [
        f"{sentence1_text}, ¿verdad? No, {sentence2_text}",
        f"{sentence1_text}, ¿verdad? Sí, {sentence2_text}",
    ]


def _xnli_es_choices(premise: str, hypothesis: str) -> list[str]:
    premise_text = _strip_terminal_punctuation(premise, marks=".,!?")
    hypothesis_text = _lowercase_first_letter(hypothesis)
    if hypothesis_text and not hypothesis_text.endswith("."):
        hypothesis_text = f"{hypothesis_text}."
    return [
        f"{premise_text}, ¿correcto? Sí, {hypothesis_text}",
        f"{premise_text}, ¿correcto? Así que, {hypothesis_text}",
        f"{premise_text}, ¿correcto? No, {hypothesis_text}",
    ]


def _wnli_es_prompt(sentence1: str, sentence2: str) -> str:
    # Match the public SpanishBench prompt wording for the translated Winograd NLI task.
    return f"{sentence1.strip()}\nPregunta: {sentence2.strip()} ¿Verdadero o Falso?\nRespuesta:"


@dataclass(slots=True)
class SpanishBench(BaseMultipleChoiceSuite):
    # Evaluate the currently implementable SpanishBench multiple-choice tasks with task-specific prompts.
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = ""
    task: str = "copa_es"

    def __post_init__(self) -> None:
        if self.task not in SPANISH_BENCH_TASKS:
            raise ValueError(f"unsupported spanish_bench task: {self.task!r}")

        config = _SPANISH_BENCH_CONFIGS[self.task]
        if self.dataset_path in {"", config.dataset_path}:
            self.dataset_path = config.dataset_path
        else:
            raise ValueError("spanish_bench dataset_path must match the configured task")

        if self.dataset_name in {None, config.dataset_name}:
            self.dataset_name = config.dataset_name
        else:
            raise ValueError("spanish_bench dataset_name must match the configured task")

        if self.split in {"", config.split}:
            self.split = config.split
        else:
            raise ValueError("spanish_bench split must match the configured task")

    def dataset_loader(self) -> Any:
        if self.task == "wnli_es":
            return load_wnli_es_dataset
        return load_dataset

    def task_name(self) -> str:
        return self.task

    def continuation_for_choice(self, choice: str) -> str:
        if self.task in {"paws_es_spanish_bench", "xnli_es_spanish_bench"}:
            return choice
        return super().continuation_for_choice(choice)

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        if self.task == "copa_es":
            return MultipleChoiceSample(
                index=index,
                prompt=_copa_es_prompt(str(doc["premise"]), str(doc["question"])),
                choices=[
                    _lowercase_first_letter(str(doc["choice1"])),
                    _lowercase_first_letter(str(doc["choice2"])),
                ],
                gold_index=int(doc["label"]),
                metadata={
                    "id": str(doc["id"]),
                    "question": str(doc["question"]),
                    "premise": _normalize_inline_text(str(doc["premise"])).strip(),
                },
            )

        if self.task == "escola":
            return MultipleChoiceSample(
                index=index,
                prompt=_escola_prompt(str(doc["Sentence"])),
                choices=["no", "sí"],
                gold_index=int(doc["Label"]),
                metadata={
                    "id": str(doc["ID"]),
                    "source": str(doc["Source"]),
                    "category": int(doc["Category"]),
                    "split_name": str(doc["Split"]),
                },
            )

        if self.task == "openbookqa_es":
            labels = [str(label) for label in doc["choices"]["label"]]
            choices = [str(choice).strip() for choice in doc["choices"]["text"]]
            return MultipleChoiceSample(
                index=index,
                prompt=_normalize_inline_text(str(doc["question_stem"])).strip(),
                choices=choices,
                gold_index=choice_index_from_labels(labels, str(doc["answerKey"])),
                metadata={
                    "id": str(doc["id"]),
                    "choice_labels": labels,
                },
            )

        if self.task == "paws_es_spanish_bench":
            sentence1 = str(doc["sentence1"])
            sentence2 = str(doc["sentence2"])
            return MultipleChoiceSample(
                index=index,
                prompt="",
                choices=_paws_es_choices(sentence1, sentence2),
                gold_index=int(doc["label"]),
                metadata={
                    "id": int(doc["id"]),
                    "sentence1": _normalize_inline_text(sentence1).strip(),
                    "sentence2": _normalize_inline_text(sentence2).strip(),
                },
            )

        if self.task == "wnli_es":
            return MultipleChoiceSample(
                index=index,
                prompt=_wnli_es_prompt(str(doc["sentence1"]), str(doc["sentence2"])),
                choices=["Falso", "Verdadero"],
                gold_index=int(doc["label"]),
                metadata={"idx": int(doc["index"])},
            )

        if self.task == "xnli_es_spanish_bench":
            premise = str(doc["premise"])
            hypothesis = str(doc["hypothesis"])
            return MultipleChoiceSample(
                index=index,
                prompt="",
                choices=_xnli_es_choices(premise, hypothesis),
                gold_index=int(doc["label"]),
                metadata={
                    "premise": _normalize_inline_text(premise).strip(),
                    "hypothesis": _normalize_inline_text(hypothesis).strip(),
                    "choice_labels": ["A", "B", "C"],
                },
            )

        raise AssertionError(f"unreachable spanish_bench task: {self.task!r}")

    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        if self.task != "escola":
            return {}
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "mcc,ll": matthews_corrcoef(gold_labels, raw_predictions),
            "mcc,ll_avg": matthews_corrcoef(gold_labels, normalized_predictions),
        }


def spanish_bench(*, task: str, **kwargs: Any) -> SpanishBench:
    return SpanishBench(task=task, **kwargs)


def _make_spanish_bench_factory(task: str) -> Any:
    def factory(**kwargs: Any) -> SpanishBench:
        return spanish_bench(task=task, **kwargs)

    factory.__name__ = task
    return factory


for _task_name in SPANISH_BENCH_TASKS:
    globals()[_task_name] = _make_spanish_bench_factory(_task_name)

del _task_name
