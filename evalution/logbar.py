# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import math
from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING
from typing import Any, Iterator

from logbar import LogBar

try:
    from logbar import RegionScreenSession
except ImportError:  # pragma: no cover - older logbar releases do not expose split panes.
    RegionScreenSession = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evalution.results import CompareTestResult, TestResult


# Keep module-level state explicit for this module.
_RESULT_TABLE_COLUMNS = [
    {"label": "suite", "width": "fit"},
    {"label": "samples", "width": "fit"},
    {"label": "metric", "width": "fit"},
    {"label": "value", "width": "fit"},
]
_COMPARE_RESULT_TABLE_BASE_COLUMNS = [
    {"label": "suite", "width": "fit"},
    {"label": "metric", "width": "fit"},
]


@dataclass(frozen=True, slots=True)
class LoggingContext:
    """Define the logging context helper class."""
    # Keep the class-level state explicit for this helper.
    logger: Any
    session: Any | None = None
    region_id: str | None = None


@dataclass(frozen=True, slots=True)
class SplitPaneLoggingSession:
    """Define the split pane logging session helper class."""
    # Keep the class-level state explicit for this helper.
    session: Any
    left: LoggingContext
    right: LoggingContext


# Keep module-level state explicit for this module.
_ACTIVE_LOGGING_CONTEXT: ContextVar[LoggingContext | None] = ContextVar(
    "evalution_active_logging_context",
    default=None,
)
# Reserve one request-metadata key for naming log-likelihood progress bars without exposing a new
# engine API surface.
_LOGLIKELIHOOD_PROGRESS_TITLE_METADATA_KEY = "_evalution_loglikelihood_progress_title"


def get_logger() -> LogBar:
    """Get logger."""
    context = _ACTIVE_LOGGING_CONTEXT.get()
    if context is not None:
        logger = context.logger
        logger.setLevel("INFO")
        return logger

    logger = LogBar.shared()
    logger.setLevel("INFO")
    return logger


def loglikelihood_progress_metadata(
    *,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Implement loglikelihood progress metadata for this module."""
    merged_metadata = dict(metadata or {})
    normalized_title = title.strip()
    if normalized_title:
        merged_metadata[_LOGLIKELIHOOD_PROGRESS_TITLE_METADATA_KEY] = normalized_title
    return merged_metadata


def loglikelihood_progress_title(metadata: dict[str, Any] | None) -> str | None:
    """Implement loglikelihood progress title for this module."""
    if metadata is None:
        return None
    raw_title = metadata.get(_LOGLIKELIHOOD_PROGRESS_TITLE_METADATA_KEY)
    if not isinstance(raw_title, str):
        return None
    normalized_title = raw_title.strip()
    return normalized_title or None


def progress_output_interval(total: int) -> int | None:
    """Implement progress output interval for this module."""
    if total <= 0:
        return None
    if total <= 100:
        return 1
    return max(1, total // 100)


def progress(iterable: Any, *, title: str) -> Any:
    """Implement progress for this module."""
    total = len(iterable) if hasattr(iterable, "__len__") else 0
    context = _ACTIVE_LOGGING_CONTEXT.get()
    if context is not None and context.session is not None:
        bar = context.session.pb(
            iterable,
            region_id=context.region_id,
            output_interval=progress_output_interval(total),
        )
    else:
        bar = get_logger().pb(iterable, output_interval=progress_output_interval(total))
    if title:
        bar.title(title)
    return bar


def manual_progress(total: int, *, title: str, subtitle: str | None = None) -> Any:
    """Implement manual progress for this module."""
    context = _ACTIVE_LOGGING_CONTEXT.get()
    if context is not None and context.session is not None:
        bar = context.session.pb(
            total,
            region_id=context.region_id,
            output_interval=progress_output_interval(total),
        ).manual()
    else:
        bar = get_logger().pb(total, output_interval=progress_output_interval(total)).manual()
    if title:
        bar.title(title)
    if subtitle:
        bar.subtitle(subtitle)
    return bar


@contextmanager
def spinner(title: str) -> Iterator[None]:
    """Implement spinner for this module."""
    context = _ACTIVE_LOGGING_CONTEXT.get()
    if context is not None and context.session is not None:
        with context.session.spinner(region_id=context.region_id, title=title):
            yield
        return

    with get_logger().spinner(title):
        yield


def create_logging_context(
    *,
    logger: Any | None = None,
    session: Any | None = None,
    region_id: str | None = None,
    name: str | None = None,
) -> LoggingContext:
    """Create logging context."""
    if logger is None and session is not None:
        logger = session.create_logger(region_id or "main", name=name)
    if logger is None:
        logger = LogBar.shared()
    logger.setLevel("INFO")
    return LoggingContext(
        logger=logger,
        session=session,
        region_id=region_id,
    )


@contextmanager
def use_logging_context(context: LoggingContext | None) -> Iterator[LoggingContext | None]:
    """Implement use logging context for this module."""
    if context is None:
        yield None
        return

    token: Token[LoggingContext | None] = _ACTIVE_LOGGING_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_LOGGING_CONTEXT.reset(token)


def create_split_pane_logging_session(
    *,
    left_name: str = "left",
    right_name: str = "right",
    gutter: int = 1,
    divider: str | None = None,
    stream: Any | None = None,
    use_alternate_screen: bool | None = None,
    auto_render: bool | None = None,
) -> SplitPaneLoggingSession | None:
    """Create split pane logging session."""
    if RegionScreenSession is None:
        return None

    target_stream = stream if stream is not None else sys.stdout
    stream_supports_tty = bool(getattr(target_stream, "isatty", lambda: False)())
    session = RegionScreenSession.columns(
        "left",
        "right",
        gutter=gutter,
        divider=divider,
        stream=target_stream,
        use_alternate_screen=(
            stream_supports_tty if use_alternate_screen is None else use_alternate_screen
        ),
        auto_render=stream_supports_tty if auto_render is None else auto_render,
    )
    left = create_logging_context(session=session, region_id="left", name=left_name)
    right = create_logging_context(session=session, region_id="right", name=right_name)
    return SplitPaneLoggingSession(
        session=session,
        left=left,
        right=right,
    )


def render_test_result_table(result: TestResult, *, logger: LogBar | None = None) -> None:
    """Render test result table."""
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
    """Render test summary table."""
    if len(results) <= 1:
        return

    _render_result_table(
        results,
        title="evaluation summary",
        logger=logger,
    )


def render_compare_result_table(
    result: CompareTestResult,
    *,
    left_label: str = "left",
    right_label: str = "right",
    logger: LogBar | None = None,
) -> None:
    """Render compare result table."""
    _render_compare_table(
        [result],
        title=f"comparison result: {result.name}",
        left_label=left_label,
        right_label=right_label,
        logger=logger,
    )


def render_compare_summary_table(
    results: Sequence[CompareTestResult],
    *,
    left_label: str = "left",
    right_label: str = "right",
    logger: LogBar | None = None,
) -> None:
    """Render compare summary table."""
    if not results:
        return

    _render_compare_table(
        results,
        title=f"comparison summary: {left_label} vs {right_label}",
        left_label=left_label,
        right_label=right_label,
        logger=logger,
    )


def _render_result_table(
    results: Sequence[TestResult],
    *,
    title: str,
    logger: LogBar | None = None,
) -> None:
    """Render result table."""
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


def _render_compare_table(
    results: Sequence[CompareTestResult],
    *,
    title: str,
    left_label: str,
    right_label: str,
    logger: LogBar | None = None,
) -> None:
    """Render compare table."""
    rows = _compare_result_rows(results)
    if not rows:
        return

    logger = logger or get_logger()
    logger.info("%s", title)

    columns = logger.columns(
        cols=[
            *_COMPARE_RESULT_TABLE_BASE_COLUMNS,
            {"label": left_label, "width": "fit"},
            {"label": right_label, "width": "fit"},
            {"label": "delta", "width": "fit"},
            {"label": "winner", "width": "fit"},
        ],
        padding=1,
    )
    for row in rows:
        columns.info.simulate(*row)
    columns.info.header()
    for row in rows:
        columns.info(*row)


def _result_rows(results: Sequence[TestResult]) -> list[tuple[str, str, str, str]]:
    """Implement result rows for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
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


def _compare_result_rows(
    results: Sequence[CompareTestResult],
) -> list[tuple[str, str, str, str, str, str]]:
    """Compare result rows. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for result in results:
        for metric_name, metric_result in result.metrics.items():
            rows.append(
                (
                    result.name,
                    metric_name,
                    _format_metric_value(metric_result.left_value),
                    _format_metric_value(metric_result.right_value),
                    _format_metric_value(metric_result.delta)
                    if metric_result.delta is not None
                    else "-",
                    metric_result.winner or "tie",
                )
            )
    return rows


def _format_metric_value(value: Any) -> str:
    """Format metric value. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)
