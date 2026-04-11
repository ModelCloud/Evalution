# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

from .flores_pt import FloresPT

# Keep benchmark defaults and public task ids explicit at module scope.
PHRASES_ES_DIRECTIONS = (
    "es-va",
    "va-es",
)
PHRASES_ES_TASKS = (
    "phrases_es_va",
    "phrases_va_es",
)
_PHRASES_ES_TASK_BY_DIRECTION = {
    "es-va": "phrases_es_va",
    "va-es": "phrases_va_es",
}
_PHRASES_ES_UPSTREAM_TASK_BY_DIRECTION = {
    "es-va": "phrases_es-va",
    "va-es": "phrases_va-es",
}
_PHRASES_ES_DIRECTION_BY_TASK = {
    task_name: direction
    for direction, task_name in _PHRASES_ES_TASK_BY_DIRECTION.items()
}
_PHRASES_ES_SOURCE_FIELD = {
    "es": "es",
    "va": "va",
}
_PHRASES_ES_LANGUAGE_LABEL = {
    "es": "espanyol",
    "va": "valencià",
}


def _normalize_direction(direction: str) -> str:
    """Normalize direction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    normalized = direction.strip().lower()
    if normalized in PHRASES_ES_DIRECTIONS:
        return normalized
    if normalized in _PHRASES_ES_DIRECTION_BY_TASK:
        return _PHRASES_ES_DIRECTION_BY_TASK[normalized]
    raise ValueError(f"unsupported phrases_es direction: {direction!r}")


def _phrases_es_prompt(source_language: str, target_language: str, text: str) -> str:
    """Implement phrases es prompt for this module."""
    return (
        f"Oració en {_PHRASES_ES_LANGUAGE_LABEL[source_language]}: {text.strip()}\n\n"
        f"Oració en {_PHRASES_ES_LANGUAGE_LABEL[target_language]}:"
    )


@dataclass(slots=True)
class PhrasesES(FloresPT):
    # Evaluate the SpanishBench ES-VA phrase translation pair with corpus translation metrics.
    """Define the phrases es helper class."""
    dataset_path: str = "gplsi/ES-VA_translation_test"
    dataset_name: str | None = None
    split: str = "test"
    direction: str = "es-va"
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        self.direction = _normalize_direction(self.direction)
        if self.dataset_path != "gplsi/ES-VA_translation_test":
            raise ValueError("phrases_es dataset_path must be 'gplsi/ES-VA_translation_test'")
        if self.dataset_name is not None:
            raise ValueError("phrases_es dataset_name must be None")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _PHRASES_ES_TASK_BY_DIRECTION[self.direction]

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
            "upstream_task": _PHRASES_ES_UPSTREAM_TASK_BY_DIRECTION[self.direction],
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        source_language, target_language = self.language_pair_tokens()
        source_field = _PHRASES_ES_SOURCE_FIELD[source_language]
        target_field = _PHRASES_ES_SOURCE_FIELD[target_language]
        for index, doc in enumerate(docs):
            source_text = str(doc[source_field]).strip()
            target_text = str(doc[target_field]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target_text,
                request=GenerationRequest(
                    prompt=_phrases_es_prompt(source_language, target_language, source_text),
                    stop=list(self.stop),
                    max_new_tokens=self.max_new_tokens,
                ),
            )

    def language_pair_tokens(self) -> tuple[str, str]:
        """Implement language pair tokens for phrases es."""
        return tuple(self.direction.split("-", maxsplit=1))  # type: ignore[return-value]

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        source_language, target_language = self.language_pair_tokens()
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
                "direction": self.direction,
                "source_language": source_language,
                "target_language": target_language,
            },
        )


def phrases_es(*, direction: str, **kwargs: Any) -> PhrasesES:
    """Implement phrases es for this module."""
    return PhrasesES(direction=direction, **kwargs)


def _make_phrases_es_factory(direction: str) -> Any:
    """Make phrases es factory."""
    def factory(**kwargs: Any) -> PhrasesES:
        """Implement factory for this module."""
        return phrases_es(direction=direction, **kwargs)

    factory.__name__ = _PHRASES_ES_TASK_BY_DIRECTION[direction]
    return factory


# Register both ES-VA task factories eagerly for import-time discovery.
for _direction in PHRASES_ES_DIRECTIONS:
    globals()[_PHRASES_ES_TASK_BY_DIRECTION[_direction]] = _make_phrases_es_factory(_direction)

del _direction
