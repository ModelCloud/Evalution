# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.datasets.flores200 import (
    FLORES200_ARCHIVE_SHA256,
    FLORES200_ARCHIVE_URL,
    load_flores200_pair,
)
from .flores_pt import FloresPT, _LANGUAGE_CODE_BY_TOKEN

# Keep the SpanishBench FLORES directions explicit so the public factories stay stable.
FLORES_ES_DIRECTIONS = (
    "es-en",
    "en-es",
    "es-eu",
    "eu-es",
    "es-pt",
    "pt-es",
    "es-it",
    "it-es",
    "es-fr",
    "fr-es",
    "es-ca",
    "ca-es",
    "es-gl",
    "gl-es",
    "es-de",
    "de-es",
)
FLORES_ES_TASKS = tuple(f"flores_es_{direction.replace('-', '_')}" for direction in FLORES_ES_DIRECTIONS)
_FLORES_ES_TASK_BY_DIRECTION = {
    direction: f"flores_es_{direction.replace('-', '_')}"
    for direction in FLORES_ES_DIRECTIONS
}
_FLORES_ES_DIRECTION_BY_TASK = {
    task_name: direction
    for direction, task_name in _FLORES_ES_TASK_BY_DIRECTION.items()
}


def _normalize_direction(direction: str) -> str:
    normalized = direction.strip().lower()
    if normalized in FLORES_ES_DIRECTIONS:
        return normalized
    if normalized in _FLORES_ES_DIRECTION_BY_TASK:
        return _FLORES_ES_DIRECTION_BY_TASK[normalized]
    raise ValueError(f"unsupported flores_es direction: {direction!r}")


@dataclass(slots=True)
class FloresES(FloresPT):
    # Reuse the audited FLORES-200 pipeline for the SpanishBench translation directions.
    direction: str = "en-es"

    def __post_init__(self) -> None:
        self.direction = _normalize_direction(self.direction)
        if self.dataset_path != "facebook/flores":
            raise ValueError("flores_es dataset_path must be 'facebook/flores'")
        if self.dataset_name not in {None, "all"}:
            raise ValueError("flores_es dataset_name must be None or 'all'")
        if self.dataset_name is None:
            self.dataset_name = "all"

    def task_name(self) -> str:
        return _FLORES_ES_TASK_BY_DIRECTION[self.direction]

    def dataset_loader(self) -> Any:
        source_language, target_language = self.language_pair_tokens()

        def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
            return load_flores200_pair(
                dataset_path,
                dataset_name,
                source_language=_LANGUAGE_CODE_BY_TOKEN[source_language],
                target_language=_LANGUAGE_CODE_BY_TOKEN[target_language],
                **kwargs,
            )

        return loader

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        source_language, target_language = self.language_pair_tokens()
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": self.direction,
            "source_language": source_language,
            "target_language": target_language,
            "upstream_task": f"spanish_bench_flores_{self.direction}",
            "archive_url": FLORES200_ARCHIVE_URL,
            "archive_sha256": FLORES200_ARCHIVE_SHA256,
        }


def flores_es(*, direction: str, **kwargs: Any) -> FloresES:
    return FloresES(direction=direction, **kwargs)


def _make_flores_es_factory(direction: str) -> Any:
    def factory(**kwargs: Any) -> FloresES:
        return flores_es(direction=direction, **kwargs)

    factory.__name__ = _FLORES_ES_TASK_BY_DIRECTION[direction]
    return factory


# Register all SpanishBench FLORES direction factories eagerly for import-time discovery.
for _direction in FLORES_ES_DIRECTIONS:
    globals()[_FLORES_ES_TASK_BY_DIRECTION[_direction]] = _make_flores_es_factory(_direction)

del _direction
