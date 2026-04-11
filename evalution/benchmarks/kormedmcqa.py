# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.choice_label import exact_match

# KorMedMCQA exposes four licensed profession subsets and one grouped task in lm-eval.
KORMEDMCQA_SUBSETS = {
    "doctor": "doctor",
    "nurse": "nurse",
    "pharm": "pharm",
    "dentist": "dentist",
}
KORMEDMCQA_TASKS = (
    "kormedmcqa",
    "kormedmcqa_doctor",
    "kormedmcqa_nurse",
    "kormedmcqa_pharm",
    "kormedmcqa_dentist",
)
_KORMEDMCQA_CHOICE_LABELS = ("A", "B", "C", "D", "E")
_KORMEDMCQA_STOP_STRINGS = ("Q:", "</s>", "<|im_end|>", ".", "\n\n")


def _choice_label(answer: int) -> str:
    """Implement choice label for this module."""
    return _KORMEDMCQA_CHOICE_LABELS[int(answer) - 1]


def _format_prompt(doc: dict[str, Any], *, include_answer: bool) -> str:
    # lm-eval's fewshot examples include a space before the gold label; the scored query does not.
    """Format prompt."""
    answer_suffix = f" {_choice_label(int(doc['answer']))}" if include_answer else ""
    return (
        f"{str(doc['question']).strip()}\n"
        f"A. {str(doc['A']).strip()}\n"
        f"B. {str(doc['B']).strip()}\n"
        f"C. {str(doc['C']).strip()}\n"
        f"D. {str(doc['D']).strip()}\n"
        f"E. {str(doc['E']).strip()}\n"
        f"정답：{answer_suffix}"
    )


def _normalize_prediction_label(text: str) -> str:
    """Normalize prediction label. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    for char in text.upper():
        if char in _KORMEDMCQA_CHOICE_LABELS:
            return char
    return text.strip().upper()


def _group_rows(*, dataset_path: str, split: str, cache_dir: str | None = None) -> Dataset:
    """Group rows. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    rows_by_subset: list[list[dict[str, Any]]] = []
    for subset, dataset_name in KORMEDMCQA_SUBSETS.items():
        dataset = load_dataset(dataset_path, dataset_name, split=split, cache_dir=cache_dir)
        rows = []
        for doc in dataset:
            row = dict(doc)
            row["kormedmcqa_subset"] = subset
            rows.append(row)
        rows_by_subset.append(rows)

    combined_rows: list[dict[str, Any]] = []
    max_length = max(len(rows) for rows in rows_by_subset)
    for row_index in range(max_length):
        for rows in rows_by_subset:
            if row_index < len(rows):
                combined_rows.append(rows[row_index])
    return Dataset.from_list(combined_rows)


@dataclass(slots=True)
class KorMedMCQA(BaseTestSuite):
    # KorMedMCQA uses five-shot Korean medical exam prompting and exact-match answer-label scoring.
    """Implement the kor med mcqa benchmark suite."""
    dataset_path: str = "sean0042/KorMedMCQA"
    dataset_name: str | None = "doctor"
    split: str = "test"
    stream: bool = (False)
    subset: str = "doctor"
    fewshot_split: str = "fewshot"
    num_fewshot: int = 5
    # The task target is a single choice label, so a short cap avoids long irrelevant generations.
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 0.0
    apply_chat_template: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset == "kormedmcqa":
            self.dataset_name = None
            return
        if self.subset not in KORMEDMCQA_SUBSETS:
            raise ValueError(f"unsupported kormedmcqa subset: {self.subset!r}")
        expected_dataset_name = KORMEDMCQA_SUBSETS[self.subset]
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("kormedmcqa dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        if self.subset != "kormedmcqa":
            return load_dataset

        dataset_path = self.dataset_path
        split = self.split

        def loader(*_args: Any, cache_dir: str | None = None, **_kwargs: Any) -> Dataset:
            """Implement loader for kor med mcqa."""
            return _group_rows(dataset_path=dataset_path, split=split, cache_dir=cache_dir)

        return loader

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        if self.subset == "kormedmcqa":
            return self.subset
        return f"kormedmcqa_{self.subset}"

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": self.subset,
            "fewshot_split": self.fewshot_split,
            "num_fewshot": self.num_fewshot,
            "apply_chat_template": self.apply_chat_template,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        fewshot_prefix_cache: dict[str, str] = {}

        def fewshot_prefix(subset: str) -> str:
            """Implement fewshot prefix for kor med mcqa."""
            if subset not in fewshot_prefix_cache:
                fewshot_docs = load_dataset(
                    self.dataset_path,
                    KORMEDMCQA_SUBSETS[subset],
                    split=self.fewshot_split,
                    cache_dir=self.cache_dir,
                )
                rendered = [
                    _format_prompt(doc, include_answer=True)
                    for index, doc in enumerate(fewshot_docs)
                    if index < self.num_fewshot
                ]
                fewshot_prefix_cache[subset] = "\n\n".join(rendered)
            return fewshot_prefix_cache[subset]

        for index, doc in enumerate(docs):
            subset = str(doc.get("kormedmcqa_subset") or self.subset)
            prefix = fewshot_prefix(subset)
            prompt = f"{prefix}\n\n{_format_prompt(doc, include_answer=False)}"
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_choice_label(int(doc["answer"])),
                request=GenerationRequest(
                    prompt=prompt,
                    stop=list(_KORMEDMCQA_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        normalized_prediction = _normalize_prediction_label(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": normalized_prediction,
                "target-stripped": prepared_sample.target,
            },
            scores={"em": exact_match(normalized_prediction, prepared_sample.target)},
            metadata={
                "subset": str(prepared_sample.doc.get("kormedmcqa_subset") or self.subset),
                "subject": str(prepared_sample.doc["subject"]),
                "year": int(prepared_sample.doc["year"]),
                "period": int(prepared_sample.doc["period"]),
                "q_number": int(prepared_sample.doc["q_number"]),
                "raw_choices": [
                    str(prepared_sample.doc["A"]).strip(),
                    str(prepared_sample.doc["B"]).strip(),
                    str(prepared_sample.doc["C"]).strip(),
                    str(prepared_sample.doc["D"]).strip(),
                    str(prepared_sample.doc["E"]).strip(),
                ],
            },
        )


def kormedmcqa(**kwargs: Any) -> KorMedMCQA:
    """Implement kormedmcqa for this module."""
    return KorMedMCQA(subset="kormedmcqa", dataset_name=None, **kwargs)


def kormedmcqa_doctor(**kwargs: Any) -> KorMedMCQA:
    """Implement kormedmcqa doctor for this module."""
    kwargs.setdefault("dataset_name", KORMEDMCQA_SUBSETS["doctor"])
    return KorMedMCQA(subset="doctor", **kwargs)


def kormedmcqa_nurse(**kwargs: Any) -> KorMedMCQA:
    """Implement kormedmcqa nurse for this module."""
    kwargs.setdefault("dataset_name", KORMEDMCQA_SUBSETS["nurse"])
    return KorMedMCQA(subset="nurse", **kwargs)


def kormedmcqa_pharm(**kwargs: Any) -> KorMedMCQA:
    """Implement kormedmcqa pharm for this module."""
    kwargs.setdefault("dataset_name", KORMEDMCQA_SUBSETS["pharm"])
    return KorMedMCQA(subset="pharm", **kwargs)


def kormedmcqa_dentist(**kwargs: Any) -> KorMedMCQA:
    """Implement kormedmcqa dentist for this module."""
    kwargs.setdefault("dataset_name", KORMEDMCQA_SUBSETS["dentist"])
    return KorMedMCQA(subset="dentist", **kwargs)
