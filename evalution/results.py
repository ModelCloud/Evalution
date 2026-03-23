# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleResult:
    index: int
    prompt: str
    target: str
    prediction: str
    extracted: dict[str, str]
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TestResult:
    name: str
    metrics: dict[str, float]
    samples: list[SampleResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    model: dict[str, Any]
    engine: dict[str, Any]
    tests: list[TestResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareMetricResult:
    left_value: Any
    right_value: Any
    delta: float | None = None
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareTestResult:
    name: str
    left: TestResult
    right: TestResult
    metrics: dict[str, CompareMetricResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CompareRunResult:
    left_name: str
    right_name: str
    left: RunResult
    right: RunResult
    tests: list[CompareTestResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
