# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleResult:
    """Define the sample result helper class."""
    # Keep the class-level state explicit for this helper.
    index: int
    prompt: str
    target: str
    prediction: str
    extracted: dict[str, str]
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for sample result."""
        return asdict(self)


@dataclass(slots=True)
class TestResult:
    """Define the test result helper class."""
    # Keep the class-level state explicit for this helper.
    name: str
    metrics: dict[str, float]
    samples: list[SampleResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for test result."""
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    """Define the run result helper class."""
    # Keep the class-level state explicit for this helper.
    model: dict[str, Any]
    engine: dict[str, Any]
    tests: list[TestResult]

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for run result."""
        return asdict(self)


@dataclass(slots=True)
class CompareMetricResult:
    """Define the compare metric result helper class."""
    # Keep the class-level state explicit for this helper.
    left_value: Any
    right_value: Any
    delta: float | None = None
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for compare metric result."""
        return asdict(self)


@dataclass(slots=True)
class CompareTestResult:
    """Define the compare test result helper class."""
    # Keep the class-level state explicit for this helper.
    name: str
    left: TestResult
    right: TestResult
    metrics: dict[str, CompareMetricResult]

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for compare test result."""
        return asdict(self)


@dataclass(slots=True)
class CompareRunResult:
    """Define the compare run result helper class."""
    # Keep the class-level state explicit for this helper.
    left_name: str
    right_name: str
    left: RunResult
    right: RunResult
    tests: list[CompareTestResult]

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for compare run result."""
        return asdict(self)
