# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
import zipfile

from datasets import Dataset
from huggingface_hub import hf_hub_download
import pcre

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.qa_text import best_qa_scores, canonicalize_no_answer
from evalution.scorers.summary_rouge import summary_rouge_scores
from evalution.benchmarks.subsets import normalize_subset_token

# Keep the public SCROLLS task list explicit because the family mixes multiple evaluation shapes.
SCROLLS_TASKS = (
    "scrolls_contractnli",
    "scrolls_qasper",
    "scrolls_govreport",
    "scrolls_qmsum",
    "scrolls_quality",
    "scrolls_narrativeqa",
    "scrolls_summscreenfd",
)
SCROLLS_DATASET_NAMES = {
    "contractnli": "contract_nli",
    "qasper": "qasper",
    "govreport": "gov_report",
    "qmsum": "qmsum",
    "quality": "quality",
    "narrativeqa": "narrative_qa",
    "summscreenfd": "summ_screen_fd",
}
_SCROLLS_ALIASES = {
    "contract_nli": "contractnli",
    "gov_report": "govreport",
    "narrative_qa": "narrativeqa",
    "summ_screen_fd": "summscreenfd",
}
_STOP_STRINGS = ("\n", "\nQuestion:", "\nAnswer:")
_QUALITY_CHOICE_PATTERN = pcre.compile(r" *\([A-D]\) *")
_CONTRACT_NLI_CHOICES = ["Not mentioned", "Entailment", "Contradiction"]


def _scrolls_variant_token(value: str) -> str:
    """Implement scrolls variant token for this module."""
    normalized = normalize_subset_token(value)
    return _SCROLLS_ALIASES.get(normalized, normalized)


def _scrolls_dataset_name(variant: str) -> str:
    """Implement scrolls dataset name for this module."""
    dataset_name = SCROLLS_DATASET_NAMES.get(_scrolls_variant_token(variant))
    if dataset_name is None:
        raise ValueError(f"unsupported scrolls variant: {variant!r}")
    return dataset_name


def _scrolls_task_name(variant: str) -> str:
    """Implement scrolls task name for this module."""
    return f"scrolls_{_scrolls_variant_token(variant)}"


def _dedupe_outputs(outputs: list[Any]) -> list[str]:
    """Dedupe outputs."""
    deduped: list[str] = []
    for output in outputs:
        text = str(output).strip()
        if text and text not in deduped:
            deduped.append(text)
    if not deduped:
        raise ValueError("scrolls rows must contain at least one non-empty reference output")
    return deduped


def _group_scrolls_outputs(dataset: list[dict[str, Any]] | Dataset) -> Dataset:
    # Collapse duplicate ids into one row with a list of reference outputs, matching the upstream task grouping.
    """Group scrolls outputs."""
    grouped_rows: dict[str, dict[str, Any]] = {}
    row_order: list[str] = []
    for row in dataset:
        row_id = str(row["id"])
        output_text = str(row["output"]).strip()
        if row_id not in grouped_rows:
            grouped_rows[row_id] = {
                "id": row_id,
                "pid": str(row.get("pid", "")),
                "input": str(row["input"]),
                "outputs": [],
            }
            row_order.append(row_id)
        if output_text and output_text not in grouped_rows[row_id]["outputs"]:
            grouped_rows[row_id]["outputs"].append(output_text)
    return Dataset.from_list([grouped_rows[row_id] for row_id in row_order])


def _load_scrolls_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool | None = None,
) -> Dataset:
    # Materialize SCROLLS splits from the repo-hosted zip archives so no remote dataset scripts execute.
    """Load scrolls dataset. Preserve the fallback order expected by the surrounding caller."""
    effective_stream = False if stream is None else stream
    if effective_stream:
        raise ValueError("scrolls requires stream=False to group duplicate reference rows")
    if dataset_name is None:
        raise ValueError("scrolls dataset_name cannot be None")
    archive_path = hf_hub_download(
        repo_id=dataset_path,
        filename=f"{dataset_name}.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        member_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith(f"/{split}.jsonl") and not name.startswith("__MACOSX/")
            ),
            None,
        )
        if member_name is None:
            raise ValueError(
                f"scrolls archive {dataset_name!r} does not contain split {split!r}"
            )
        with archive.open(member_name) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    return _group_scrolls_outputs(dataset)


def _scrolls_split_question_and_text(input_text: str) -> tuple[str, str]:
    """Implement scrolls split question and text for this module."""
    split_index = input_text.find("\n\n")
    if split_index < 0:
        raise ValueError("scrolls input must contain a double-newline question/text separator")
    question = input_text[:split_index].strip()
    text = input_text[split_index + 2 :].strip()
    return question, text


def _scrolls_qa_prompt(*, text: str, question: str) -> str:
    """Implement scrolls QA prompt for this module."""
    return f"{text.strip()}\n\nQuestion: {question.strip()}\nAnswer:"


def _scrolls_summary_prompt(text: str) -> str:
    """Implement scrolls summary prompt for this module."""
    return (
        f"{text.strip()}\n\n"
        "Question: What is a summary of the preceding text?\n"
        "Answer:"
    )


def _scrolls_best_summary_scores(prediction: str, references: list[str]) -> dict[str, float]:
    """Implement scrolls best summary scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    best_scores: dict[str, float] | None = None
    for reference in references:
        scores = summary_rouge_scores(prediction, reference)
        if best_scores is None:
            best_scores = scores
            continue
        if (scores["rougeLsum"], scores["rouge1"], scores["rouge2"]) > (
            best_scores["rougeLsum"],
            best_scores["rouge1"],
            best_scores["rouge2"],
        ):
            best_scores = scores
    if best_scores is None:
        raise ValueError("scrolls summary scoring requires at least one reference")
    return best_scores


def _quality_choices_and_context(text: str) -> tuple[list[str], str]:
    """Implement quality choices and context for this module."""
    split_index = text.find("\n\n", text.find("(D)"))
    if split_index < 0:
        raise ValueError("scrolls quality rows must contain answer choices followed by the passage text")
    choices_text = text[:split_index]
    passage_text = text[split_index:].strip()
    choices = [
        " ".join(choice.split()).strip()
        for choice in _QUALITY_CHOICE_PATTERN.split(choices_text)[1:]
    ]
    if len(choices) != 4:
        raise ValueError(f"scrolls quality expects four answer choices, got {len(choices)}")
    return choices, passage_text


def _scrolls_outputs(doc: dict[str, Any]) -> list[str]:
    """Implement scrolls outputs for this module."""
    return _dedupe_outputs(list(doc["outputs"]))


def _init_scrolls_variant(instance: Any) -> None:
    # Keep dataset_name and the public task variant locked together because several SCROLLS names normalize.
    """Implement init scrolls variant for this module."""
    canonical_variant = _scrolls_variant_token(instance.variant)
    dataset_name = _scrolls_dataset_name(canonical_variant)
    instance.variant = canonical_variant
    if instance.dataset_name in {None, dataset_name}:
        instance.dataset_name = dataset_name
        return
    raise ValueError("scrolls dataset_name must match the configured variant")


@dataclass(slots=True)
class ScrollsContractNLI(BaseMultipleChoiceSuite):
    # Score the ContractNLI subset with the benchmark's fixed three-way conclusion labels.
    """Implement the scrolls contract nli benchmark suite."""
    dataset_path: str = "tau/scrolls"
    dataset_name: str | None = None
    split: str = "validation"
    variant: str = "contractnli"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        _init_scrolls_variant(self)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_scrolls_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _scrolls_task_name(self.variant)

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        question, text = _scrolls_split_question_and_text(str(doc["input"]))
        outputs = _scrolls_outputs(doc)
        return MultipleChoiceSample(
            index=index,
            prompt=f"{text}\n\nHypothesis: {question}\nConclusion:",
            choices=list(_CONTRACT_NLI_CHOICES),
            gold_index=_CONTRACT_NLI_CHOICES.index(outputs[0]),
            metadata={
                "id": str(doc["id"]),
                "pid": str(doc.get("pid", "")),
                "variant": self.variant,
                "question": question,
                "text": text,
                "choice_texts": list(_CONTRACT_NLI_CHOICES),
                "outputs": outputs,
            },
        )


@dataclass(slots=True)
class ScrollsQuALITY(BaseMultipleChoiceSuite):
    # Score QuALITY rows by parsing the inline choice block and treating the passage as the shared context.
    """Implement the scrolls qu ality benchmark suite."""
    dataset_path: str = "tau/scrolls"
    dataset_name: str | None = None
    split: str = "validation"
    variant: str = "quality"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        _init_scrolls_variant(self)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_scrolls_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _scrolls_task_name(self.variant)

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        question, raw_text = _scrolls_split_question_and_text(str(doc["input"]))
        choices, passage_text = _quality_choices_and_context(raw_text)
        outputs = _scrolls_outputs(doc)
        gold_text = " ".join(outputs[0].split()).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_scrolls_qa_prompt(text=passage_text, question=question),
            choices=choices,
            gold_index=choices.index(gold_text),
            metadata={
                "id": str(doc["id"]),
                "pid": str(doc.get("pid", "")),
                "variant": self.variant,
                "question": question,
                "text": passage_text,
                "choice_texts": choices,
                "outputs": outputs,
            },
        )


@dataclass(slots=True)
class _BaseScrollsQASuite(BaseTestSuite):
    # Share one long-context QA pipeline for the SCROLLS extractive and abstractive question-answering tasks.
    """Implement the base scrolls qasuite benchmark suite."""
    dataset_path: str = "tau/scrolls"
    dataset_name: str | None = None
    split: str = "validation"
    variant: str = ""
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        _init_scrolls_variant(self)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_scrolls_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _scrolls_task_name(self.variant)

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "variant": self.variant,
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            question, text = _scrolls_split_question_and_text(str(doc["input"]))
            outputs = _scrolls_outputs(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=outputs[0],
                request=GenerationRequest(
                    prompt=_scrolls_qa_prompt(text=text, question=question),
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        question, text = _scrolls_split_question_and_text(str(prepared_sample.doc["input"]))
        outputs = _scrolls_outputs(prepared_sample.doc)
        exact, f1_score, best_index = best_qa_scores(output.text, outputs)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": canonicalize_no_answer(output.text),
                "best_answer_index": str(best_index),
                "best_answer": outputs[best_index],
            },
            scores={"em": exact, "f1": f1_score},
            metadata={
                "id": str(prepared_sample.doc["id"]),
                "pid": str(prepared_sample.doc.get("pid", "")),
                "variant": self.variant,
                "question": question,
                "text": text,
                "outputs": outputs,
            },
        )


@dataclass(slots=True)
class ScrollsQasper(_BaseScrollsQASuite):
    """Define the scrolls QASPER helper class."""
    # Keep the class-level state explicit for this helper.
    variant: str = "qasper"


@dataclass(slots=True)
class ScrollsNarrativeQA(_BaseScrollsQASuite):
    """Define the scrolls narrative QA helper class."""
    # Keep the class-level state explicit for this helper.
    variant: str = "narrativeqa"


@dataclass(slots=True)
class _BaseScrollsSummarySuite(BaseTestSuite):
    # Share one long-context summarization pipeline for the SCROLLS summarization subsets.
    """Implement the base scrolls summary suite benchmark suite."""
    dataset_path: str = "tau/scrolls"
    dataset_name: str | None = None
    split: str = "validation"
    variant: str = ""
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        _init_scrolls_variant(self)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_scrolls_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _scrolls_task_name(self.variant)

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "variant": self.variant,
            "scoring_mode": "generated_summary_rouge_best_reference",
            "primary_metric": "rougeLsum",
        }

    def prompt_text(self, doc: dict[str, Any]) -> str:
        """Implement prompt text for base scrolls summary suite."""
        return _scrolls_summary_prompt(str(doc["input"]))

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            outputs = _scrolls_outputs(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=outputs[0],
                request=GenerationRequest(
                    prompt=self.prompt_text(doc),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        outputs = _scrolls_outputs(prepared_sample.doc)
        prediction = output.text.strip()
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": prediction,
                "best_reference": max(outputs, key=len),
                "reference_count": str(len(outputs)),
            },
            scores=_scrolls_best_summary_scores(prediction, outputs),
            metadata={
                "id": str(prepared_sample.doc["id"]),
                "pid": str(prepared_sample.doc.get("pid", "")),
                "variant": self.variant,
                "outputs": outputs,
                "input_chars": len(str(prepared_sample.doc["input"])),
            },
        )


@dataclass(slots=True)
class ScrollsGovReport(_BaseScrollsSummarySuite):
    """Define the scrolls gov report helper class."""
    # Keep the class-level state explicit for this helper.
    variant: str = "govreport"
    max_new_tokens: int = 1024


@dataclass(slots=True)
class ScrollsSummScreenFD(_BaseScrollsSummarySuite):
    """Define the scrolls summ screen fd helper class."""
    # Keep the class-level state explicit for this helper.
    variant: str = "summscreenfd"


@dataclass(slots=True)
class ScrollsQMSum(_BaseScrollsSummarySuite):
    """Define the scrolls qmsum helper class."""
    # Keep the class-level state explicit for this helper.
    variant: str = "qmsum"

    def prompt_text(self, doc: dict[str, Any]) -> str:
        """Implement prompt text for scrolls qmsum."""
        question, text = _scrolls_split_question_and_text(str(doc["input"]))
        return _scrolls_qa_prompt(text=text, question=question)


def scrolls(*, subset: str, **kwargs: Any) -> Any:
    # Dispatch the family-level constructor to the concrete SCROLLS task implementation.
    """Implement scrolls for this module. Preserve the fallback order expected by the surrounding caller."""
    variant = _scrolls_variant_token(subset)
    if variant == "contractnli":
        return ScrollsContractNLI(variant=variant, **kwargs)
    if variant == "quality":
        return ScrollsQuALITY(variant=variant, **kwargs)
    if variant == "qasper":
        return ScrollsQasper(variant=variant, **kwargs)
    if variant == "narrativeqa":
        return ScrollsNarrativeQA(variant=variant, **kwargs)
    if variant == "govreport":
        return ScrollsGovReport(variant=variant, **kwargs)
    if variant == "qmsum":
        return ScrollsQMSum(variant=variant, **kwargs)
    if variant == "summscreenfd":
        return ScrollsSummScreenFD(variant=variant, **kwargs)
    raise ValueError(f"unsupported scrolls variant: {subset!r}")


def _make_scrolls_factory(variant: str) -> Any:
    # Register one import-stable zero-argument factory per SCROLLS task variant.
    """Make scrolls factory."""
    def factory(**kwargs: Any) -> Any:
        """Implement factory for this module."""
        return scrolls(subset=variant, **kwargs)

    factory.__name__ = _scrolls_task_name(variant)
    return factory


for _variant in SCROLLS_DATASET_NAMES:
    globals()[_scrolls_task_name(_variant)] = _make_scrolls_factory(_variant)

del _variant
