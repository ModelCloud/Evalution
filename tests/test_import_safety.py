# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS_DIR = _REPO_ROOT / "evalution" / "benchmarks"


def test_benchmark_modules_do_not_use_live_dataset_config_discovery() -> None:
    offenders: list[str] = []
    for path in sorted(_BENCHMARKS_DIR.glob("*.py")):
        if "get_dataset_config_names" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert offenders == []


def test_importing_evalution_does_not_call_dataset_config_discovery() -> None:
    script = textwrap.dedent(
        """
        import datasets

        def fail(*args, **kwargs):
            raise RuntimeError("unexpected dataset config discovery during import")

        datasets.get_dataset_config_names = fail

        import evalution  # noqa: F401
        """
    )
    env = dict(os.environ)
    env["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
