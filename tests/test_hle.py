# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
hle_module = importlib.import_module("evalution.benchmarks.hle")


class FakeGenerationSession:
    """Provide the fake generation session helper used by the surrounding tests."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses
        self.requests = []
        self.prompts: list[str] = []

    def generate(self, requests, *, batch_size=None):
        """Implement generate for the fake HLE session."""
        assert batch_size == len(requests)
        self.requests.extend(requests)
        self.prompts.extend(request.prompt or "" for request in requests)
        return [
            GenerationOutput(
                prompt=request.prompt or (request.messages[-1]["content"] if request.messages else ""),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_hle_filters_image_rows_and_scores_exact_and_multiple_choice(monkeypatch) -> None:
    """Verify HLE filters image rows and scores both exact-match and multiple-choice answers."""
    dataset = Dataset.from_list(
        [
            {
                "id": "hle-1",
                "question": "What is 2 + 2?",
                "image": "",
                "image_preview": None,
                "answer": "4",
                "answer_type": "exactMatch",
                "author_name": "Author 1",
                "rationale": "",
                "rationale_image": None,
                "raw_subject": "Math",
                "category": "Science",
                "canary": "canary",
            },
            {
                "id": "hle-image",
                "question": "This image row should be filtered.",
                "image": "data:image/png;base64,deadbeef",
                "image_preview": None,
                "answer": "A",
                "answer_type": "multipleChoice",
                "author_name": "Author 2",
                "rationale": "",
                "rationale_image": None,
                "raw_subject": "Vision",
                "category": "Other",
                "canary": "canary",
            },
            {
                "id": "hle-2",
                "question": "Which choice is correct?\n\nAnswer Choices:\nA. Alpha\nB. Bravo\nC. Charlie\nD. Delta",
                "image": "",
                "image_preview": None,
                "answer": "D",
                "answer_type": "multipleChoice",
                "author_name": "Author 3",
                "rationale": "",
                "rationale_image": None,
                "raw_subject": "Logic",
                "category": "Humanities/Social Science",
                "canary": "canary",
            },
        ]
    )
    monkeypatch.setattr(hle_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeGenerationSession(["Final answer: 4", "The answer is D."])
    result = evalution.benchmarks.hle(max_rows=3, batch_size=2).evaluate(session)

    assert result.name == "hle"
    assert result.metrics == {"acc": 1.0}
    assert result.metadata == {
        "dataset_path": "macabdul9/hle_text_only",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "modality_subset": "text_only",
        "source_benchmark": "cais/hle",
        "apply_chat_template": False,
        "scoring_mode": "generated_hle_answer_accuracy",
        "primary_metric": "acc",
    }
    assert len(result.samples) == 2
    assert all("filtered" not in prompt for prompt in session.prompts)
    assert result.samples[0].scores == {"acc": 1.0}
    assert result.samples[0].extracted == {"answer-extract": "4"}
    assert result.samples[0].metadata["answer_type"] == "exactMatch"
    assert result.samples[1].scores == {"acc": 1.0}
    assert result.samples[1].extracted == {"choice-label": "D", "answer-extract": "D"}
    assert result.samples[1].metadata["answer_type"] == "multipleChoice"


def test_hle_can_wrap_prompts_in_chat_template(monkeypatch) -> None:
    """Verify HLE can emit chat-formatted requests for instruct-style models."""
    dataset = Dataset.from_list(
        [
            {
                "id": "hle-chat",
                "question": "What is 2 + 2?",
                "image": "",
                "image_preview": None,
                "answer": "4",
                "answer_type": "exactMatch",
                "author_name": "Author 1",
                "rationale": "",
                "rationale_image": None,
                "raw_subject": "Math",
                "category": "Science",
                "canary": "canary",
            }
        ]
    )
    monkeypatch.setattr(hle_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeGenerationSession(["Final answer: 4"])
    result = evalution.benchmarks.hle(
        max_rows=1,
        batch_size=1,
        apply_chat_template=True,
    ).evaluate(session)

    request = session.requests[0]
    assert request.prompt is None
    assert request.messages == [{"role": "user", "content": "What is 2 + 2?\n\nAnswer:"}]
    assert result.metadata["apply_chat_template"] is True
    assert result.metrics == {"acc": 1.0}
