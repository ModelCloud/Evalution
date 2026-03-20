from __future__ import annotations

from abc import ABC, abstractmethod

from evalution.engines.base import InferenceSession
from evalution.results import TestResult


class TestSuite(ABC):
    @abstractmethod
    def evaluate(self, session: InferenceSession) -> TestResult:
        raise NotImplementedError
