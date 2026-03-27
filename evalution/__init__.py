# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Evalution package."""

import sys
from contextlib import redirect_stdout

from evalution._banner import ASCII_LOGO, get_startup_banner
from evalution.compare import CompareRun, compare, run_compare
from evalution.config import Model
from evalution.engines import (
    BaseEngine,
    BaseInferenceSession,
    GPTQModel,
    SGLang,
    Transformers,
    TransformersCompat,
    VLLM,
)
from evalution.logbar import get_logger
from evalution.results import (
    CompareMetricResult,
    CompareRunResult,
    CompareTestResult,
    RunResult,
    SampleResult,
    TestResult,
)
from evalution.runtime import EvaluationRun, run
from evalution.version import __version__
from evalution.yaml import python_from_yaml, run_yaml
from . import benchmarks
from . import engines

__all__ = [
    "BaseEngine",
    "BaseInferenceSession",
    "CompareMetricResult",
    "CompareRun",
    "CompareRunResult",
    "CompareTestResult",
    "EvaluationRun",
    "GPTQModel",
    "Model",
    "RunResult",
    "SGLang",
    "SampleResult",
    "TestResult",
    "Transformers",
    "TransformersCompat",
    "VLLM",
    "benchmarks",
    "compare",
    "engines",
    "python_from_yaml",
    "run_compare",
    "run",
    "run_yaml",
    "__version__",
]

with redirect_stdout(sys.stderr):
    get_logger().info(
        "\n%s",
    get_startup_banner(
            ASCII_LOGO,
            evalution_version=__version__,
        ),
    )
