from __future__ import annotations

from collections.abc import Sequence

from evalution.config import Model, coerce_model
from evalution.results import RunResult
from evalution.suites.base import TestSuite


def run(
    *,
    model: Model | dict,
    engine: object,
    tests: Sequence[TestSuite],
) -> RunResult:
    model_config = coerce_model(model)
    if not hasattr(engine, "build"):
        raise TypeError("engine must provide a `build(model)` method")

    session = engine.build(model_config)
    try:
        results = [test.evaluate(session) for test in tests]
    finally:
        session.close()

    engine_config = engine.to_dict() if hasattr(engine, "to_dict") else {}
    return RunResult(
        model=model_config.to_dict(),
        engine=engine_config,
        tests=results,
    )
