# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
mmlu_pro_plus_module = importlib.import_module("evalution.benchmarks.mmlu_pro_plus")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size == 1
        assert len(requests) == 1
        assert "about math." in requests[0].prompt
        assert "Question:\nA prime number larger than 2 is" in requests[0].prompt
        return [GenerationOutput(prompt=requests[0].prompt or "", text="After reasoning, the answer is (B).")]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_mmlu_pro_plus_reuses_mmlu_pro_prompting_and_metadata(monkeypatch) -> None:
    """Verify MMLU pro plus reuses MMLU pro prompting and metadata."""
    validation = Dataset.from_list(
        [
            {
                "question_id": 0,
                "question": "2 + 2 equals?",
                "options": ["4", "3", "2", "1"],
                "answer": "A",
                "answer_index": 0,
                "cot_content": "A: Let's think step by step. 2 + 2 = 4. The answer is (A).",
                "category": "math",
                "src": "val-math",
            }
        ]
    )
    test = Dataset.from_list(
        [
            {
                "question_id": 1,
                "question": "A prime number larger than 2 is",
                "options": ["9", "11", "12", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "math",
                "src": "test-math",
            }
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        """Support the surrounding tests with fake load dataset."""
        del path, name, kwargs
        if split == "validation":
            return validation
        if split == "test":
            return test
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(mmlu_pro_plus_module, "load_dataset", fake_load_dataset)

    result = evalution.benchmarks.mmlu_pro_plus(
        subsets="stem.math",
        num_fewshot=1,
        max_rows=1,
        batch_size=1,
        max_new_tokens=128,
    ).evaluate(FakeSession())

    assert result.name == "mmlu_pro_plus_stem_math"
    assert result.metrics == {"em,choice_label": 1.0}
    assert result.metadata["dataset_path"] == "saeidasgari/mmlu-pro-plus"
    assert result.metadata["subsets"] == ["stem.math"]
    assert result.metadata["subset_paths"] == [["stem", "math"]]
    assert result.metadata["subset_kinds"] == ["leaf"]
    assert result.metadata["selection_mode"] == "single"
    assert result.samples[0].metadata["subset"] == "stem.math"
    assert result.samples[0].metadata["subset_value"] == "math"
