# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class _AsDictMixin:
    """Provide a consistent dataclass-to-dict conversion for result payloads."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SampleResult(_AsDictMixin):
    """Per-sample evaluation output."""
    # Keep the class-level state explicit for this helper.
    index: int
    prompt: str
    target: str
    prediction: str
    extracted: dict[str, str]
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class TestResult(_AsDictMixin):
    """Aggregated result for one evaluated benchmark."""

    name: str
    metrics: dict[str, float]
    samples: list[SampleResult]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class RunResult(_AsDictMixin):
    """Serialized view of one evaluation run."""

    model: dict[str, Any]
    engine: dict[str, Any]
    tests: list[TestResult]


@dataclass(slots=True)
class CompareMetricResult(_AsDictMixin):
    """Side-by-side metric comparison for a single metric name."""

    left_value: Any
    right_value: Any
    delta: float | None = None
    winner: str | None = None


@dataclass(slots=True)
class CompareTestResult(_AsDictMixin):
    """Side-by-side comparison result for one benchmark."""

    name: str
    left: TestResult
    right: TestResult
    metrics: dict[str, CompareMetricResult]


@dataclass(slots=True)
class CompareRunResult(_AsDictMixin):
    """Serialized view of a two-lane comparison run."""

    left_name: str
    right_name: str
    left: RunResult
    right: RunResult
    tests: list[CompareTestResult]
