"""Evalution package."""

from evalution.config import Model
from evalution.engines import Transformer
from evalution.results import RunResult, SampleResult, TestResult
from evalution.runtime import EngineBuilder, EvaluationRun, engine, run
from evalution.suites import (
    ARCEasy,
    ARCChallenge,
    BaseMultipleChoiceSuite,
    BaseTestSuite,
    BoolQ,
    GSM8K,
    GSM8KPlatinum,
    HellaSwag,
    MultipleChoiceSample,
    OpenBookQA,
    PIQA,
    TestSuite,
    WinoGrande,
    arc_easy,
    arc_challenge,
    boolq,
    choice_index_from_labels,
    gsm8k,
    gsm8k_platinum,
    hellaswag,
    openbookqa,
    piqa,
    question_answer_prompt,
    winogrande,
)
from evalution.yaml import python_from_yaml, run_yaml

__all__ = [
    "ARCEasy",
    "ARCChallenge",
    "BaseMultipleChoiceSuite",
    "BaseTestSuite",
    "BoolQ",
    "EngineBuilder",
    "EvaluationRun",
    "GSM8K",
    "GSM8KPlatinum",
    "HellaSwag",
    "Model",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PIQA",
    "RunResult",
    "SampleResult",
    "TestSuite",
    "TestResult",
    "Transformer",
    "Transformers",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "choice_index_from_labels",
    "engine",
    "gsm8k",
    "gsm8k_platinum",
    "hellaswag",
    "openbookqa",
    "piqa",
    "python_from_yaml",
    "question_answer_prompt",
    "run",
    "run_yaml",
    "winogrande",
]

Transformers = Transformer
