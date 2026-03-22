# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, GenerationRequest

mmlu_pro_module = importlib.import_module("evalution.suites.mmlu_pro")


class FakeSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 2
        assert len(requests) == 2
        assert "about math." in requests[0].prompt
        assert "Question:\n2 + 2 equals?" in requests[0].prompt
        assert "Answer: Let's think step by step. 2 + 2 = 4. The answer is (A)." in requests[0].prompt
        assert requests[0].prompt.endswith(
            "Question:\nA prime number larger than 2 is\nOptions:\nA. 9\nB. 11\nC. 12\nAnswer: Let's think step by step."
        )
        assert "about business." in requests[1].prompt
        assert requests[1].stop[0] == "Question:"
        return [
            GenerationOutput(prompt=requests[0].prompt or "", text="After reasoning, the answer is (B)."),
            GenerationOutput(prompt=requests[1].prompt or "", text="Answer: B"),
        ]

    def close(self) -> None:
        return None


def test_mmlu_pro_uses_category_matched_cot_fewshots(monkeypatch) -> None:
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
            },
            {
                "question_id": 1,
                "question": "Revenue is closest to",
                "options": ["cost", "income", "tax", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "A: Let's think step by step. Revenue is income. The answer is (B).",
                "category": "business",
                "src": "val-business",
            },
        ]
    )
    test = Dataset.from_list(
        [
            {
                "question_id": 70,
                "question": "A prime number larger than 2 is",
                "options": ["9", "11", "12", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "math",
                "src": "test-math",
            },
            {
                "question_id": 71,
                "question": "Profit is revenue minus",
                "options": ["assets", "cost", "equity", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "business",
                "src": "test-business",
            },
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        del path, name, kwargs
        if split == "validation":
            return validation
        if split == "test":
            return test
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(mmlu_pro_module, "load_dataset", fake_load_dataset)

    result = evalution.mmlu_pro(num_fewshot=1, max_rows=2, batch_size=2).evaluate(FakeSession())

    assert result.name == "mmlu_pro"
    assert result.metrics == {"exact_match,choice-label": 1.0}
    assert result.metadata["dataset_path"] == "TIGER-Lab/MMLU-Pro"
    assert result.metadata["fewshot_split"] == "validation"
    assert result.metadata["num_fewshot"] == 1
    assert result.metadata["category"] == "all"
    assert result.metadata["scoring_mode"] == "generated_choice_label_exact_match"
    assert len(result.samples) == 2

    first = result.samples[0]
    assert first.target == "B"
    assert first.extracted["choice-label"] == "B"
    assert first.metadata["category"] == "math"
    assert first.metadata["question_id"] == 70
    assert first.metadata["choice_texts"] == ["9", "11", "12"]
    assert first.metadata["fewshot_count"] == 1

    second = result.samples[1]
    assert second.target == "B"
    assert second.extracted["choice-label"] == "B"
    assert second.metadata["category"] == "business"


def test_mmlu_pro_category_filter_uses_normalized_name(monkeypatch) -> None:
    validation = Dataset.from_list(
        [
            {
                "question_id": 0,
                "question": "A hash table is a",
                "options": ["tree", "array-backed map", "compiler", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "A: Let's think step by step. A hash table is a map. The answer is (B).",
                "category": "computer science",
                "src": "val-cs",
            },
        ]
    )
    test = Dataset.from_list(
        [
            {
                "question_id": 1,
                "question": "A queue is typically",
                "options": ["LIFO", "FIFO", "sorted", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "computer science",
                "src": "test-cs",
            },
            {
                "question_id": 2,
                "question": "An atom contains",
                "options": ["planets", "electrons", "N/A"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "physics",
                "src": "test-physics",
            },
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        del path, name, kwargs
        if split == "validation":
            return validation
        if split == "test":
            return test
        raise AssertionError(f"unexpected split: {split}")

    class CategorySession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 1
            assert len(requests) == 1
            assert "about computer science." in requests[0].prompt
            return [GenerationOutput(prompt=requests[0].prompt or "", text="the answer is (B)")]

        def close(self) -> None:
            return None

    monkeypatch.setattr(mmlu_pro_module, "load_dataset", fake_load_dataset)

    result = evalution.mmlu_pro(category="computer_science", num_fewshot=1, batch_size=1).evaluate(
        CategorySession()
    )

    assert result.name == "mmlu_pro_computer_science"
    assert result.metadata["category"] == "computer science"
    assert len(result.samples) == 1
    assert result.samples[0].metadata["category"] == "computer science"


def test_mmlu_pro_backs_off_fewshots_to_fit_context_window(monkeypatch) -> None:
    validation = Dataset.from_list(
        [
            {
                "question_id": 0,
                "question": "Very long worked example one",
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer": "A",
                "answer_index": 0,
                "cot_content": (
                    "A: Let's think step by step. "
                    "token token token token token token token token token token. "
                    "The answer is (A)."
                ),
                "category": "math",
                "src": "val-math-1",
            },
            {
                "question_id": 1,
                "question": "Very long worked example two",
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": (
                    "A: Let's think step by step. "
                    "token token token token token token token token token token. "
                    "The answer is (B)."
                ),
                "category": "math",
                "src": "val-math-2",
            },
        ]
    )
    test = Dataset.from_list(
        [
            {
                "question_id": 2,
                "question": "Short math question",
                "options": ["wrong", "right", "other", "last"],
                "answer": "B",
                "answer_index": 1,
                "cot_content": "",
                "category": "math",
                "src": "test-math",
            },
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        del path, name, kwargs
        if split == "validation":
            return validation
        if split == "test":
            return test
        raise AssertionError(f"unexpected split: {split}")

    class TinyTokenizer:
        model_max_length = 55

        def __call__(self, text, add_special_tokens=False, padding=False):
            del add_special_tokens, padding
            return {"input_ids": list(range(len(str(text).split())))}

    class TinyModelConfig:
        max_position_embeddings = 55

    class TinyModel:
        config = TinyModelConfig()

    class BackoffSession:
        tokenizer = TinyTokenizer()
        prepare_tokenizer = None
        model = TinyModel()

        def prepare_requests(self, requests: list[GenerationRequest]):
            prepared = []
            for request in requests:
                prompt = request.prompt or ""
                input_ids = list(range(len(prompt.split())))
                prepared.append(
                    GenerationRequest(
                        prompt=request.prompt,
                        messages=request.messages,
                        rendered_prompt=prompt,
                        input_ids=input_ids,
                        add_generation_prompt=request.add_generation_prompt,
                        stop=list(request.stop),
                        max_new_tokens=request.max_new_tokens,
                        do_sample=request.do_sample,
                        temperature=request.temperature,
                        metadata=dict(request.metadata),
                    )
                )
            return prepared

        def generate(self, requests, *, batch_size=None):
            assert batch_size == 1
            assert len(requests) == 1
            assert requests[0].metadata["fewshot_count"] < 2
            assert "Very long worked example one" not in (requests[0].rendered_prompt or requests[0].prompt or "")
            assert "Very long worked example two" not in (requests[0].rendered_prompt or requests[0].prompt or "")
            return [
                GenerationOutput(
                    prompt=requests[0].rendered_prompt or requests[0].prompt or "",
                    text="the answer is (B)",
                )
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(mmlu_pro_module, "load_dataset", fake_load_dataset)

    result = evalution.mmlu_pro(num_fewshot=2, batch_size=1, max_new_tokens=20).evaluate(
        BackoffSession()
    )

    assert result.metrics == {"exact_match,choice-label": 1.0}
    assert result.samples[0].metadata["fewshot_count"] == 0
