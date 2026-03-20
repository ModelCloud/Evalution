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
