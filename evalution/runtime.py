# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from evalution.config import Model
from evalution.engines.base import BaseEngine, BaseInferenceSession
from evalution.logbar import (
    LoggingContext,
    get_logger,
    render_test_result_table,
    render_test_summary_table,
    spinner,
    use_logging_context,
)
from evalution.results import RunResult
from evalution.benchmarks.base import TestSuite


@dataclass(slots=True)
class EvaluationRun:
    _engine_impl: BaseEngine
    _model_config: Model
    _session: BaseInferenceSession | None = field(default=None, init=False, repr=False)
    _execution: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _test_results: list[Any] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _logging_context: LoggingContext | None = field(default=None, init=False, repr=False)

    def run(self, test: TestSuite) -> EvaluationRun:
        with self._logging_scope():
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
            self._test_results.append(result)
            logger.info("completed test %s", result.name)
            render_test_result_table(result, logger=logger)
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
        with self._logging_scope():
            if self._closed:
                return
            self._closed = True
            if self._session is not None:
                logger = get_logger()
                logger.info("closing engine session")
                self._session.close()
                self._session = None
            logger = get_logger()
            render_test_summary_table(self._test_results, logger=logger)
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

    def bind_logging_context(self, context: LoggingContext | None) -> EvaluationRun:
        self._logging_context = context
        return self

    def _logging_scope(self):
        if self._logging_context is None:
            return nullcontext()
        return use_logging_context(self._logging_context)


def run(
    *,
    model: Model | dict,
    engine: BaseEngine,
    tests: Sequence[TestSuite],
) -> RunResult:
    if not isinstance(engine, BaseEngine):
        raise TypeError("engine must inherit BaseEngine")

    model_config = model if isinstance(model, Model) else Model(**model)
    evaluation = engine.model(**model_config.to_dict())
    try:
        for test in tests:
            evaluation.run(test)
        return evaluation.result()
    except Exception:
        evaluation.close()
        raise


def _build_session(engine: BaseEngine, model_config: Model) -> BaseInferenceSession:
    logger = get_logger()
    logger.info("building engine %s for model %s", type(engine).__name__, model_config.path)
    with spinner(f"Loading {type(engine).__name__} engine"):
        session = engine.build(model_config)
    if not isinstance(session, BaseInferenceSession):
        raise TypeError("engine.build(model) must return a BaseInferenceSession")
    return session


def _describe_execution(session: BaseInferenceSession) -> dict[str, Any] | None:
    execution = session.describe_execution()
    get_logger().info("engine execution=%s", execution)
    return execution


def _gc_session(session: BaseInferenceSession) -> None:
    get_logger().info("running engine gc before next test suite")
    session.gc()
