# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.lambada import LAMBADA, _lambada_prompt_target
from evalution.benchmarks.single_continuation import SingleContinuationSample


@dataclass(slots=True)
class LAMBADACloze(LAMBADA):
    # Evaluate the cloze-prompted LAMBADA variant that appends a blank and arrow marker.
    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"lambada_{self.variant_name}_cloze"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> SingleContinuationSample:
        text = str(doc["text"])
        prompt, target = _lambada_prompt_target(text)
        metadata = {
            "text": text,
            "target_token": target.strip(),
            "prompt_variant": "cloze",
        }
        if "domain" in doc:
            metadata["domain"] = doc["domain"]
        return SingleContinuationSample(
            index=index,
            prompt=f"{prompt} ____. ->",
            target=target,
            metadata=metadata,
        )

    def result_metadata(self) -> dict[str, Any]:
        return {
            **super().result_metadata(),
            "prompt_variant": "cloze",
        }


def lambada_openai_cloze(**kwargs: Any) -> LAMBADACloze:
    return LAMBADACloze(**kwargs)


def lambada_standard_cloze(**kwargs: Any) -> LAMBADACloze:
    return LAMBADACloze(
        dataset_path="cimec/lambada",
        dataset_name=None,
        variant_name="standard",
        **kwargs,
    )
