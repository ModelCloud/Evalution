# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _rte_prompt(premise: str, hypothesis: str) -> str:
    # Format the entailment pair using the same True-or-False question style as the upstream benchmark prompt.
    return f"{premise.strip()}\nQuestion: {hypothesis.strip()} True or False?\nAnswer:"


@dataclass(slots=True)
class RTE(BaseMultipleChoiceSuite):
    # Evaluate textual entailment by ranking True/False label continuations with token log-likelihood.
    dataset_path: str = "super_glue"
    dataset_name: str | None = "rte"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the RTE task packaged inside SuperGLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "rte"

    # Convert one RTE row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_rte_prompt(doc["premise"], doc["hypothesis"]),
            choices=["True", "False"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def rte(**kwargs: Any) -> RTE:
    return RTE(**kwargs)
