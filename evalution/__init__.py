"""Evalution package."""

from evalution.config import Model
from evalution.engines import Transformer
from evalution.results import RunResult, SampleResult, TestResult
from evalution.runtime import EngineBuilder, EvaluationRun, engine, run
from evalution.suites import (
    ARCChallenge,
    BaseMultipleChoiceSuite,
    BaseTestSuite,
    GSM8K,
    GSM8KPlatinum,
    HellaSwag,
    MultipleChoiceSample,
    PIQA,
    TestSuite,
    arc_challenge,
    gsm8k,
    gsm8k_platinum,
    hellaswag,
    piqa,
)
from evalution.yaml import python_from_yaml, run_yaml

__all__ = [
    "ARCChallenge",
    "BaseMultipleChoiceSuite",
    "BaseTestSuite",
    "EngineBuilder",
    "EvaluationRun",
    "GSM8K",
    "GSM8KPlatinum",
    "HellaSwag",
    "Model",
    "MultipleChoiceSample",
    "PIQA",
    "RunResult",
    "SampleResult",
    "TestSuite",
    "TestResult",
    "Transformer",
    "Transformers",
    "arc_challenge",
    "engine",
    "gsm8k",
    "gsm8k_platinum",
    "hellaswag",
    "piqa",
    "python_from_yaml",
    "run",
    "run_yaml",
]

Transformers = Transformer
