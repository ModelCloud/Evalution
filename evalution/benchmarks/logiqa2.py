# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.logiqa import _logiqa_prompt
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_LOGIQA2_CHOICE_LABELS = ["A", "B", "C", "D"]


def _parse_logiqa2_row(doc: dict[str, Any]) -> dict[str, Any]:
    """Parse logiqa2 row."""
    payload = doc.get("text")
    if not isinstance(payload, str):
        raise TypeError("LogiQA2 rows must expose a JSON-encoded 'text' field")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError("LogiQA2 row payload must decode to an object")
    return parsed


@dataclass(slots=True)
class LogiQA2(BaseMultipleChoiceSuite):
    """Implement the logi qa2 benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "datatune/LogiQA2.0"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = (False)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "logiqa2"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        parsed = _parse_logiqa2_row(doc)
        options = [str(option).strip() for option in parsed["options"]]
        question_type = parsed.get("type", {})
        return MultipleChoiceSample(
            index=index,
            prompt=_logiqa_prompt(
                str(parsed["text"]).strip(),
                str(parsed["question"]).strip(),
                options,
            ),
            choices=options,
            gold_index=int(parsed["answer"]),
            metadata={
                "question_id": int(parsed["id"]),
                "question_type": sorted(
                    str(name)
                    for name, enabled in dict(question_type).items()
                    if enabled
                ),
                "choice_labels": list(_LOGIQA2_CHOICE_LABELS),
                "choice_texts": options,
            },
        )


def logiqa2(**kwargs: Any) -> LogiQA2:
    """Implement logiqa2 for this module."""
    return LogiQA2(**kwargs)
