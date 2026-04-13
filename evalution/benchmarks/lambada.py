# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.single_continuation import (
    BaseSingleContinuationSuite,
    SingleContinuationSample,
)


def _lambada_prompt_target(text: str) -> tuple[str, str]:
    """Implement lambada prompt target for this module."""
    tokens = text.split(" ")
    if len(tokens) < 2:
        raise ValueError("LAMBADA examples must contain at least two space-delimited tokens")
    return " ".join(tokens[:-1]), f" {tokens[-1]}"


@dataclass(slots=True)
class LAMBADA(BaseSingleContinuationSuite):
    # Evaluate the final-word prediction setup used by the LAMBADA benchmark.
    """Define the lambada helper class."""
    dataset_path: str = "EleutherAI/lambada_openai"
    dataset_name: str | None = "default"
    split: str = "test"
    variant_name: str = "openai"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"lambada_{self.variant_name}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> SingleContinuationSample:
        """Build one benchmark sample from a dataset row."""
        text = str(doc["text"])
        prompt, target = _lambada_prompt_target(text)
        metadata = {
            "text": text,
            "target_token": target.strip(),
        }
        if "domain" in doc:
            metadata["domain"] = doc["domain"]
        return SingleContinuationSample(
            index=index,
            prompt=prompt,
            target=target,
            metadata=metadata,
        )


def lambada_openai(**kwargs: Any) -> LAMBADA:
    """Implement lambada openai for this module."""
    return LAMBADA(**kwargs)


def lambada_standard(**kwargs: Any) -> LAMBADA:
    """Implement lambada standard for this module."""
    return LAMBADA(
        dataset_path="cimec/lambada",
        dataset_name=None,
        variant_name="standard",
        **kwargs,
    )
