# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
import threading
import traceback
from typing import Any

from evalution.logbar import create_split_pane_logging_session, get_logger, render_compare_summary_table
from evalution.results import CompareMetricResult, CompareRunResult, CompareTestResult
from evalution.runtime import EvaluationRun
from evalution.benchmarks.base import TestSuite


@dataclass(slots=True)
class CompareRun:
    """Define the compare run helper class."""
    # Keep the class-level state explicit for this helper.
    _left_name: str
    _right_name: str
    _left_run: EvaluationRun
    _right_run: EvaluationRun
    _test_results: list[CompareTestResult] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _split_logging: Any | None = field(default=None, init=False, repr=False)
    _split_logging_initialized: bool = field(default=False, init=False, repr=False)

    def run(self, test: TestSuite) -> CompareRun:
        """Run run."""
        if self._closed:
            raise RuntimeError("compare run is already closed")

        self._ensure_split_logging()
        left_test = _clone_test(test)
        right_test = _clone_test(test)
        errors: list[tuple[str, BaseException, str]] = []

        left_thread = threading.Thread(
            target=_run_compare_lane,
            args=(self._left_name, self._left_run, left_test, errors),
            name=f"evalution-compare-{self._left_name}",
        )
        right_thread = threading.Thread(
            target=_run_compare_lane,
            args=(self._right_name, self._right_run, right_test, errors),
            name=f"evalution-compare-{self._right_name}",
        )
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()

        if errors:
            self.close()
            lane_name, exc, rendered_traceback = errors[0]
            raise RuntimeError(
                f"compare lane {lane_name!r} failed while running {type(test).__name__}:\n{rendered_traceback}"
            ) from exc

        left_result = self._left_run.result(close=False).tests[-1]
        right_result = self._right_run.result(close=False).tests[-1]
        self._test_results.append(
            _build_compare_test_result(
                left_name=self._left_name,
                right_name=self._right_name,
                left_result=left_result,
                right_result=right_result,
            )
        )
        return self

    @property
    def left(self):
        """Implement left for compare run."""
        return self._materialize_result(close=True).left

    @property
    def right(self):
        """Implement right for compare run."""
        return self._materialize_result(close=True).right

    @property
    def tests(self) -> list[CompareTestResult]:
        """Implement tests for compare run."""
        return self._materialize_result(close=True).tests

    def result(self, *, close: bool = True) -> CompareRunResult:
        """Implement result for compare run."""
        return self._materialize_result(close=close)

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for compare run."""
        return self._materialize_result(close=True).to_dict()

    def close(self) -> None:
        """Release the resources owned by this object."""
        if self._closed:
            return

        self._closed = True
        try:
            self._left_run.close()
        finally:
            try:
                self._right_run.close()
            finally:
                if self._split_logging is not None:
                    self._split_logging.session.close()
                    self._split_logging = None

        render_compare_summary_table(
            self._test_results,
            left_label=self._left_name,
            right_label=self._right_name,
            logger=get_logger(),
        )

    def __enter__(self) -> CompareRun:
        """Enter the managed context for this object."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the managed context for this object."""
        del exc_type, exc, tb
        self.close()

    def _materialize_result(self, *, close: bool) -> CompareRunResult:
        """Implement materialize result for compare run."""
        if close:
            self.close()

        return CompareRunResult(
            left_name=self._left_name,
            right_name=self._right_name,
            left=self._left_run.result(close=False),
            right=self._right_run.result(close=False),
            tests=list(self._test_results),
        )

    def _ensure_split_logging(self) -> None:
        """Ensure split logging."""
        if self._split_logging_initialized:
            return

        self._split_logging_initialized = True
        split_logging = create_split_pane_logging_session(
            left_name=self._left_name,
            right_name=self._right_name,
        )
        if split_logging is None:
            return

        self._split_logging = split_logging
        self._left_run.bind_logging_context(split_logging.left)
        self._right_run.bind_logging_context(split_logging.right)


def compare(
    left: EvaluationRun,
    right: EvaluationRun,
) -> CompareRun:
    """Bind two fresh evaluation handles into one comparison session."""
    left_run = _coerce_compare_lane(left, lane_label="left")
    right_run = _coerce_compare_lane(right, lane_label="right")
    if left_run is right_run:
        raise ValueError("compare requires distinct left and right evaluation handles")

    return CompareRun(
        _left_name=_default_lane_name(left_run, fallback="left"),
        _right_name=_default_lane_name(right_run, fallback="right"),
        _left_run=left_run,
        _right_run=right_run,
    )


def run_compare(
    left: EvaluationRun,
    right: EvaluationRun,
    *,
    tests: Sequence[TestSuite],
) -> CompareRunResult:
    """Execute multiple suites through `compare(...)` and return the closed result."""
    comparison = compare(left, right)
    try:
        for test in tests:
            comparison.run(test)
        return comparison.result()
    except Exception:
        comparison.close()
        raise


def _clone_test(test: TestSuite) -> TestSuite:
    """Implement clone test for this module."""
    try:
        return copy.deepcopy(test)
    except Exception as exc:  # pragma: no cover - defensive guard for custom suite objects.
        raise TypeError(
            "compare requires test suites that can be deep-copied so left and right lanes stay isolated"
        ) from exc


def _coerce_compare_lane(lane: EvaluationRun, *, lane_label: str) -> EvaluationRun:
    """Implement coerce compare lane for this module."""
    if not isinstance(lane, EvaluationRun):
        raise TypeError(f"{lane_label} must be an engine.model(...) handle")
    if lane._closed:
        raise ValueError(f"{lane_label} compare lane is already closed")
    if lane._session is not None or lane._test_results:
        raise ValueError(
            f"{lane_label} compare lane must be a fresh engine.model(...) handle"
        )
    return lane


def _default_lane_name(run: EvaluationRun, *, fallback: str) -> str:
    """Implement default lane name for this module."""
    if run._model_config.label:
        return run._model_config.label
    model_path = run._model_config.path.strip()
    if not model_path:
        return fallback
    return model_path


def _run_compare_lane(
    lane_name: str,
    evaluation: EvaluationRun,
    test: TestSuite,
    errors: list[tuple[str, BaseException, str]],
) -> None:
    """Run compare lane."""
    try:
        evaluation.run(test)
    except BaseException as exc:  # pragma: no cover - exercised via raised RuntimeError in caller.
        errors.append((lane_name, exc, traceback.format_exc()))


def _build_compare_test_result(
    *,
    left_name: str,
    right_name: str,
    left_result,
    right_result,
) -> CompareTestResult:
    """Build compare test result."""
    metrics: dict[str, CompareMetricResult] = {}
    metric_names = sorted(set(left_result.metrics) | set(right_result.metrics))
    for metric_name in metric_names:
        left_value = left_result.metrics.get(metric_name)
        right_value = right_result.metrics.get(metric_name)
        metrics[metric_name] = _build_compare_metric_result(
            left_name=left_name,
            right_name=right_name,
            left_value=left_value,
            right_value=right_value,
        )

    return CompareTestResult(
        name=left_result.name,
        left=left_result,
        right=right_result,
        metrics=metrics,
    )


def _build_compare_metric_result(
    *,
    left_name: str,
    right_name: str,
    left_value: Any,
    right_value: Any,
) -> CompareMetricResult:
    """Build compare metric result. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    left_numeric = _is_numeric_metric(left_value)
    right_numeric = _is_numeric_metric(right_value)
    if not left_numeric or not right_numeric:
        return CompareMetricResult(
            left_value=left_value,
            right_value=right_value,
            delta=None,
            winner=None,
        )

    left_score = float(left_value)
    right_score = float(right_value)
    winner = None
    if left_score > right_score:
        winner = left_name
    elif right_score > left_score:
        winner = right_name

    return CompareMetricResult(
        left_value=left_value,
        right_value=right_value,
        delta=left_score - right_score,
        winner=winner,
    )


def _is_numeric_metric(value: Any) -> bool:
    """Implement is numeric metric for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
