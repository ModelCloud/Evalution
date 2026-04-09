# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

from evalution.logbar import render_compare_result_table, render_compare_summary_table, render_test_result_table, render_test_summary_table
from evalution.results import (
    CompareMetricResult,
    CompareTestResult,
    SampleResult,
    TestResult as EvalutionTestResult,
)


class FakeColumnsLevel:
    def __init__(self) -> None:
        self.simulated_rows: list[tuple[str, ...]] = []
        self.rows: list[tuple[str, ...]] = []
        self.header_calls = 0

    def simulate(self, *values: str) -> None:
        self.simulated_rows.append(tuple(values))

    def header(self) -> None:
        self.header_calls += 1

    def __call__(self, *values: str) -> None:
        self.rows.append(tuple(values))


class FakeColumns:
    def __init__(self) -> None:
        self.info = FakeColumnsLevel()


class FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.columns_calls: list[tuple[object, int]] = []
        self.tables: list[FakeColumns] = []

    def info(self, message: str, *args) -> None:
        self.info_messages.append(message % args if args else message)

    def columns(self, *args, cols=None, padding=2):
        self.columns_calls.append((cols, padding))
        table = FakeColumns()
        self.tables.append(table)
        return table


def _sample_result(index: int = 0) -> SampleResult:
    return SampleResult(
        index=index,
        prompt="prompt",
        target="target",
        prediction="prediction",
        extracted={},
        scores={},
    )


def test_render_test_result_table_uses_logbar_columns() -> None:
    logger = FakeLogger()
    result = EvalutionTestResult(
        name="boolq",
        metrics={
            "accuracy": 1.0,
            "accuracy_norm": 0.75,
        },
        samples=[_sample_result()],
    )

    render_test_result_table(result, logger=logger)

    assert logger.info_messages == ["test suite result: boolq"]
    assert len(logger.tables) == 1
    assert logger.columns_calls[0][1] == 1
    assert logger.tables[0].info.header_calls == 1
    assert logger.tables[0].info.simulated_rows == [
        ("boolq", "1", "accuracy", "1.0000"),
        ("boolq", "1", "accuracy_norm", "0.7500"),
    ]
    assert logger.tables[0].info.rows == [
        ("boolq", "1", "accuracy", "1.0000"),
        ("boolq", "1", "accuracy_norm", "0.7500"),
    ]


def test_render_test_summary_table_skips_single_suite() -> None:
    logger = FakeLogger()
    result = EvalutionTestResult(
        name="boolq",
        metrics={"accuracy": 1.0},
        samples=[_sample_result()],
    )

    render_test_summary_table([result], logger=logger)

    assert logger.info_messages == []
    assert logger.tables == []


def test_render_test_summary_table_consolidates_multiple_suites() -> None:
    logger = FakeLogger()
    results = [
        EvalutionTestResult(
            name="boolq",
            metrics={"accuracy": 1.0},
            samples=[_sample_result()],
        ),
        EvalutionTestResult(
            name="piqa",
            metrics={"accuracy": 0.5, "accuracy_norm": 0.625},
            samples=[_sample_result(1), _sample_result(2)],
        ),
    ]

    render_test_summary_table(results, logger=logger)

    assert logger.info_messages == ["evaluation summary"]
    assert len(logger.tables) == 1
    assert logger.tables[0].info.rows == [
        ("boolq", "1", "accuracy", "1.0000"),
        ("piqa", "2", "accuracy", "0.5000"),
        ("piqa", "2", "accuracy_norm", "0.6250"),
    ]


def test_render_compare_result_table_uses_logbar_columns() -> None:
    logger = FakeLogger()
    result = CompareTestResult(
        name="boolq",
        left=EvalutionTestResult(
            name="boolq",
            metrics={"accuracy": 1.0},
            samples=[_sample_result()],
        ),
        right=EvalutionTestResult(
            name="boolq",
            metrics={"accuracy": 0.75},
            samples=[_sample_result()],
        ),
        metrics={
            "accuracy": CompareMetricResult(
                left_value=1.0,
                right_value=0.75,
                delta=0.25,
                winner="left",
            )
        },
    )

    render_compare_result_table(result, logger=logger)

    assert logger.info_messages == ["comparison result: boolq"]
    assert len(logger.tables) == 1
    assert logger.tables[0].info.rows == [
        ("boolq", "accuracy", "1.0000", "0.7500", "0.2500", "left"),
    ]


def test_render_compare_summary_table_consolidates_multiple_suites() -> None:
    logger = FakeLogger()
    results = [
        CompareTestResult(
            name="boolq",
            left=EvalutionTestResult(
                name="boolq",
                metrics={"accuracy": 1.0},
                samples=[_sample_result()],
            ),
            right=EvalutionTestResult(
                name="boolq",
                metrics={"accuracy": 0.5},
                samples=[_sample_result()],
            ),
            metrics={
                "accuracy": CompareMetricResult(
                    left_value=1.0,
                    right_value=0.5,
                    delta=0.5,
                    winner="left",
                )
            },
        ),
        CompareTestResult(
            name="piqa",
            left=EvalutionTestResult(
                name="piqa",
                metrics={"accuracy": 0.5},
                samples=[_sample_result(1)],
            ),
            right=EvalutionTestResult(
                name="piqa",
                metrics={"accuracy": 0.5},
                samples=[_sample_result(2)],
            ),
            metrics={
                "accuracy": CompareMetricResult(
                    left_value=0.5,
                    right_value=0.5,
                    delta=0.0,
                    winner=None,
                )
            },
        ),
    ]

    render_compare_summary_table(results, left_label="model_a", right_label="model_b", logger=logger)

    assert logger.info_messages == ["comparison summary: model_a vs model_b"]
    assert len(logger.tables) == 1
    assert logger.tables[0].info.rows == [
        ("boolq", "accuracy", "1.0000", "0.5000", "0.5000", "left"),
        ("piqa", "accuracy", "0.5000", "0.5000", "0.0000", "tie"),
    ]
