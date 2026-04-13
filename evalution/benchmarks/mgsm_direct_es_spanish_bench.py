# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.benchmarks.mgsm import MGSM


@dataclass(slots=True)
class MGSMDirectESSpanishBench(MGSM):
    # Reuse the clean-room MGSM implementation for the SpanishBench Spanish direct-answer alias.
    """Define the mgsmdirect esspanish bench helper class."""
    dataset_name: str | None = "es"
    language: str = "es"

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "mgsm_direct_es_spanish_bench"


def mgsm_direct_es_spanish_bench(**kwargs: Any) -> MGSMDirectESSpanishBench:
    """Implement mgsm direct es spanish bench for this module."""
    return MGSMDirectESSpanishBench(**kwargs)
