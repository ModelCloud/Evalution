# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

from evalution.logbar import create_logging_context, get_logger, manual_progress, spinner, use_logging_context


class FakeLogger:
    def __init__(self) -> None:
        self.level = None

    def setLevel(self, level: str) -> None:
        self.level = level


class FakeSpinner:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> None:
        self.entered += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.exited += 1


class FakeProgressBar:
    def __init__(self) -> None:
        self.manual_called = False
        self.titles: list[str] = []
        self.subtitles: list[str] = []

    def manual(self) -> FakeProgressBar:
        self.manual_called = True
        return self

    def title(self, value: str) -> FakeProgressBar:
        self.titles.append(value)
        return self

    def subtitle(self, value: str) -> FakeProgressBar:
        self.subtitles.append(value)
        return self


class FakeSession:
    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.progress_calls: list[tuple[int, str | None, int | None]] = []
        self.spinner_calls: list[tuple[str | None, str]] = []
        self.progress_bar = FakeProgressBar()
        self.spinner_bar = FakeSpinner()

    def create_logger(self, region_id: str, name: str | None = None) -> FakeLogger:
        del region_id, name
        return self.logger

    def pb(self, total: int, *, region_id: str | None = None, output_interval: int | None = None) -> FakeProgressBar:
        self.progress_calls.append((total, region_id, output_interval))
        return self.progress_bar

    def spinner(self, *, region_id: str | None = None, title: str = "", interval: float = 0.5, tail_length: int = 4):
        del interval, tail_length
        self.spinner_calls.append((region_id, title))
        return self.spinner_bar


def test_logging_context_routes_logger_progress_and_spinner_to_session() -> None:
    session = FakeSession()
    context = create_logging_context(session=session, region_id="left", name="lane-left")

    with use_logging_context(context):
        assert get_logger() is session.logger
        progress_bar = manual_progress(10, title="prepare", subtitle="batch_size=2")
        with spinner("loading engine"):
            pass

    assert progress_bar is session.progress_bar
    assert session.logger.level == "INFO"
    assert session.progress_calls == [(10, "left", 1)]
    assert session.progress_bar.manual_called is True
    assert session.progress_bar.titles == ["prepare"]
    assert session.progress_bar.subtitles == ["batch_size=2"]
    assert session.spinner_calls == [("left", "loading engine")]
    assert session.spinner_bar.entered == 1
    assert session.spinner_bar.exited == 1
