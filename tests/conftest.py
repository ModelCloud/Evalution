# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import sys

import pytest


# Keep CUDA ordinal assignment stable across test runs so cuda:0/cuda:1 map by PCI bus order.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def pytest_sessionstart(session: pytest.Session) -> None:
    # The no-GIL test matrix must fail fast on the wrong interpreter instead of silently downgrading.
    del session
    if os.environ.get("PYTHON_GIL") != "0":
        raise pytest.UsageError("the test suite must be run with PYTHON_GIL=0")

    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if not callable(is_gil_enabled):
        raise pytest.UsageError(
            "the test suite requires a free-threading Python build that exposes sys._is_gil_enabled()"
        )
    if is_gil_enabled():
        raise pytest.UsageError(
            "the test suite requires Python free-threading with the GIL disabled; run pytest with PYTHON_GIL=0"
        )
