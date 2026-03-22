# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from contextlib import contextmanager
from typing import Any, Iterator

from logbar import LogBar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evalution.results import TestResult


_RESULT_TABLE_COLUMNS = [
    {"label": "suite", "width": "fit"},
    {"label": "samples", "width": "fit"},
    {"label": "metric", "width": "fit"},
    {"label": "value", "width": "fit"},
]


def get_logger() -> LogBar:
    logger = LogBar.shared()
    logger.setLevel("INFO")
    return logger


def progress_output_interval(total: int) -> int | None:
    if total <= 0:
        return None
    if total <= 100:
        return 1
    return max(1, total // 100)


def progress(iterable: Any, *, title: str) -> Any:
    total = len(iterable) if hasattr(iterable, "__len__") else 0
    bar = get_logger().pb(iterable, output_interval=progress_output_interval(total))
    if title:
        bar.title(title)
    return bar


def manual_progress(total: int, *, title: str, subtitle: str | None = None) -> Any:
    bar = get_logger().pb(total, output_interval=progress_output_interval(total)).manual()
    if title:
        bar.title(title)
    if subtitle:
        bar.subtitle(subtitle)
    return bar


@contextmanager
def spinner(title: str) -> Iterator[None]:
    with get_logger().spinner(title):
        yield


def render_test_result_table(result: TestResult, *, logger: LogBar | None = None) -> None:
    _render_result_table(
        [result],
        title=f"test suite result: {result.name}",
        logger=logger,
    )


def render_test_summary_table(
    results: Sequence[TestResult],
    *,
    logger: LogBar | None = None,
) -> None:
    if len(results) <= 1:
        return

    _render_result_table(
        results,
        title="evaluation summary",
        logger=logger,
    )


def _render_result_table(
    results: Sequence[TestResult],
    *,
    title: str,
    logger: LogBar | None = None,
) -> None:
    rows = _result_rows(results)
    if not rows:
        return

    logger = logger or get_logger()
    logger.info("%s", title)

    columns = logger.columns(cols=_RESULT_TABLE_COLUMNS, padding=1)
    for row in rows:
        columns.info.simulate(*row)
    columns.info.header()
    for row in rows:
        columns.info(*row)


def _result_rows(results: Sequence[TestResult]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for result in results:
        sample_count = str(len(result.samples))
        if not result.metrics:
            rows.append((result.name, sample_count, "-", "-"))
            continue

        for metric_name, metric_value in result.metrics.items():
            rows.append(
                (
                    result.name,
                    sample_count,
                    metric_name,
                    _format_metric_value(metric_value),
                )
            )

    return rows


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)
