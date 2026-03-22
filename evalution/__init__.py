# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Evalution package."""

from contextlib import redirect_stdout
import sys

from evalution._banner import ASCII_LOGO, get_startup_banner
from evalution.config import Model
from evalution.engines import BaseEngine, BaseInferenceSession, Transformer, TransformerCompat
from evalution.logbar import get_logger
from evalution.results import RunResult, SampleResult, TestResult
from evalution.runtime import EngineBuilder, EvaluationRun, engine, run
from evalution.suites import (
    ARCEasy,
    ARCChallenge,
    BaseMultipleChoiceSuite,
    BaseTestSuite,
    BoolQ,
    CB,
    CoLA,
    COPA,
    GSM8K,
    GSM8KPlatinum,
    HellaSwag,
    MMLU,
    MNLI,
    MRPC,
    MultipleChoiceSample,
    OpenBookQA,
    PIQA,
    QNLI,
    QQP,
    RTE,
    SST2,
    TestSuite,
    WiC,
    WNLI,
    WinoGrande,
    arc_easy,
    arc_challenge,
    boolq,
    cb,
    cola,
    copa,
    choice_index_from_labels,
    f1_for_label,
    gsm8k,
    gsm8k_platinum,
    hellaswag,
    macro_f1,
    matthews_corrcoef,
    mmlu,
    mnli,
    mrpc,
    openbookqa,
    piqa,
    qnli,
    qqp,
    rte,
    question_answer_prompt,
    sst2,
    wic,
    wnli,
    winogrande,
)
from evalution.version import __version__
from evalution.yaml import python_from_yaml, run_yaml

__all__ = [
    "ARCEasy",
    "ARCChallenge",
    "BaseMultipleChoiceSuite",
    "BaseTestSuite",
    "BoolQ",
    "BaseEngine",
    "BaseInferenceSession",
    "CB",
    "CoLA",
    "COPA",
    "EngineBuilder",
    "EvaluationRun",
    "GSM8K",
    "GSM8KPlatinum",
    "HellaSwag",
    "MMLU",
    "MNLI",
    "Model",
    "MRPC",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PIQA",
    "QNLI",
    "QQP",
    "RTE",
    "SST2",
    "RunResult",
    "SampleResult",
    "TestSuite",
    "TestResult",
    "Transformer",
    "TransformerCompat",
    "Transformers",
    "TransformersCompat",
    "WNLI",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "cb",
    "cola",
    "copa",
    "choice_index_from_labels",
    "engine",
    "f1_for_label",
    "gsm8k",
    "gsm8k_platinum",
    "hellaswag",
    "macro_f1",
    "matthews_corrcoef",
    "mmlu",
    "mnli",
    "mrpc",
    "openbookqa",
    "piqa",
    "qnli",
    "qqp",
    "rte",
    "python_from_yaml",
    "question_answer_prompt",
    "sst2",
    "run",
    "run_yaml",
    "wic",
    "wnli",
    "winogrande",
    "__version__",
]

Transformers = Transformer
TransformersCompat = TransformerCompat

with redirect_stdout(sys.stderr):
    get_logger().info(
        "\n%s",
        get_startup_banner(
            ASCII_LOGO,
            evalution_version=__version__,
        ),
    )
