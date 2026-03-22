# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evalution.config import Model, coerce_model
from evalution.logbar import get_logger, spinner
from evalution.results import RunResult
from evalution.suites.base import TestSuite


@dataclass(slots=True)
class EvaluationRun:
    _engine_impl: object
    _model_config: Model
    _session: Any | None = field(default=None, init=False, repr=False)
    _execution: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _test_results: list[Any] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def run(self, test: TestSuite) -> EvaluationRun:
        if self._closed:
            raise RuntimeError("run is already closed")
        if self._session is None:
            self._session = _build_session(self._engine_impl, self._model_config)
            self._execution = _describe_execution(self._session)
        elif self._test_results:
            _gc_session(self._session)

        logger = get_logger()
        logger.info("running test suite %s", type(test).__name__)
        result = test.evaluate(self._session)
        logger.info("completed test %s", result.name)
        self._test_results.append(result)
        return self

    @property
    def model(self) -> dict[str, Any]:
        return self._materialize_result(close=True).model

    @property
    def engine(self) -> dict[str, Any]:
        return self._materialize_result(close=True).engine

    @property
    def tests(self) -> list[Any]:
        return self._materialize_result(close=True).tests

    def result(self, *, close: bool = True) -> RunResult:
        return self._materialize_result(close=close)

    def to_dict(self) -> dict[str, Any]:
        return self._materialize_result(close=True).to_dict()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            logger = get_logger()
            logger.info("closing engine session")
            self._session.close()
            self._session = None
        logger = get_logger()
        logger.info("finished evaluation run with %d test suite(s)", len(self._test_results))

    def __enter__(self) -> EvaluationRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _materialize_result(self, *, close: bool) -> RunResult:
        if close:
            self.close()

        engine_config = (
            self._engine_impl.to_dict() if hasattr(self._engine_impl, "to_dict") else {}
        )
        if self._execution is not None:
            engine_config = {**engine_config, "execution": self._execution}
        return RunResult(
            model=self._model_config.to_dict(),
            engine=engine_config,
            tests=list(self._test_results),
        )


@dataclass(slots=True)
class EngineBuilder:
    _engine_impl: object

    def model(self, model: Model | dict) -> EvaluationRun:
        return EvaluationRun(
            _engine_impl=self._engine_impl,
            _model_config=coerce_model(model),
        )


def engine(engine: object) -> EngineBuilder:
    if not hasattr(engine, "build"):
        raise TypeError("engine must provide a `build(model)` method")
    return EngineBuilder(_engine_impl=engine)


def run(
    *,
    model: Model | dict,
    engine: object,
    tests: Sequence[TestSuite],
) -> RunResult:
    if not hasattr(engine, "build"):
        raise TypeError("engine must provide a `build(model)` method")

    evaluation = EngineBuilder(_engine_impl=engine).model(model)
    try:
        for test in tests:
            evaluation.run(test)
        return evaluation.result()
    except Exception:
        evaluation.close()
        raise


def _build_session(engine: object, model_config: Model) -> Any:
    logger = get_logger()
    logger.info("building engine %s for model %s", type(engine).__name__, model_config.path)
    with spinner(f"Loading {type(engine).__name__} engine"):
        return engine.build(model_config)


def _describe_execution(session: Any) -> dict[str, Any] | None:
    describe_execution = getattr(session, "describe_execution", None)
    if not callable(describe_execution):
        return None
    execution = describe_execution()
    get_logger().info("engine execution=%s", execution)
    return execution


def _gc_session(session: Any) -> None:
    gc_session = getattr(session, "gc", None)
    if callable(gc_session):
        get_logger().info("running engine gc before next test suite")
        gc_session()
