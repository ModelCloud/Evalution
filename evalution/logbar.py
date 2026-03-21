from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from logbar import LogBar


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
