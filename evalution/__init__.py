# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Evalution package."""

from contextlib import redirect_stdout
import sys
from types import ModuleType

from . import benchmarks
from . import engines
from evalution._banner import ASCII_LOGO, get_startup_banner
from evalution.compare import CompareRun, compare, run_compare
from evalution.config import Model
from evalution.engines import (
    BaseEngine,
    BaseInferenceSession,
    GPTQModel,
    Transformers,
    TransformersCompat,
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
from evalution.runtime import EngineBuilder, EvaluationRun, run
from evalution.version import __version__
from evalution.yaml import python_from_yaml, run_yaml


class _EvalutionModule(ModuleType):
    def __call__(self, engine_impl: BaseEngine) -> EngineBuilder:
        if not isinstance(engine_impl, BaseEngine):
            raise TypeError("engine must inherit BaseEngine")
        return EngineBuilder(_engine_impl=engine_impl)

__all__ = [
    "BaseEngine",
    "BaseInferenceSession",
    "CompareMetricResult",
    "CompareRun",
    "CompareRunResult",
    "CompareTestResult",
    "EngineBuilder",
    "EvaluationRun",
    "GPTQModel",
    "Model",
    "RunResult",
    "SampleResult",
    "TestResult",
    "Transformers",
    "TransformersCompat",
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

sys.modules[__name__].__class__ = _EvalutionModule
