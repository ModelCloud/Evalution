from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

arc_challenge_module = importlib.import_module("evalution.suites.arc_challenge")


class FakeSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        del batch_size
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        return None


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "Mercury_7175875",
                "question": (
                    "An astronomer observes that a planet rotates faster after a meteorite "
                    "impact. Which is the most likely effect of this increase in rotation?"
                ),
                "choices": {
                    "text": [
                        "Planetary density will decrease.",
                        "Planetary years will become longer.",
                        "Planetary days will become shorter.",
                        "Planetary gravity will become stronger.",
                    ],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            }
        ]
    )


def test_arc_challenge_suite_scores_choice_labels_and_records_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(max_rows=1)
    session = FakeSession(["The answer is C."])
    result = suite.evaluate(session)

    assert result.name == "arc_challenge"
    assert result.metrics["exact_match,choice-label"] == 1.0
    assert result.metrics["exact_match,choice-text"] == 1.0
    assert result.metadata["dataset_path"] == "allenai/ai2_arc"
    assert result.metadata["dataset_name"] == "ARC-Challenge"
    assert result.metadata["apply_chat_template"] is False
    assert result.samples[0].target == "C. Planetary days will become shorter."
    assert result.samples[0].metadata["id"] == "Mercury_7175875"
    assert result.samples[0].metadata["answer_label"] == "C"
    assert result.samples[0].metadata["choices"][2]["text"] == "Planetary days will become shorter."
    assert "Answer with the correct choice label." in session.requests[0].prompt


def test_arc_challenge_chat_template_uses_single_user_message(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(
        max_rows=1,
        apply_chat_template=True,
    )
    session = FakeSession(["C"])
    result = suite.evaluate(session)

    assert result.metrics["exact_match,choice-label"] == 1.0
    assert session.requests[0].prompt is None
    assert session.requests[0].messages == [
        {
            "role": "user",
            "content": (
                "Question: An astronomer observes that a planet rotates faster after a meteorite "
                "impact. Which is the most likely effect of this increase in rotation?\n"
                "Choices:\n"
                "A. Planetary density will decrease.\n"
                "B. Planetary years will become longer.\n"
                "C. Planetary days will become shorter.\n"
                "D. Planetary gravity will become stronger.\n"
                "Answer with the correct choice label.\n"
                "Answer:"
            ),
        }
    ]


def test_arc_challenge_falls_back_to_choice_text_matching(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(max_rows=1)
    session = FakeSession(["Planetary days will become shorter."])
    result = suite.evaluate(session)

    assert result.metrics["exact_match,choice-label"] == 1.0
    assert result.metrics["exact_match,choice-text"] == 1.0
    assert result.samples[0].extracted["choice-label"] == "C"
    assert result.samples[0].extracted["choice-text"] == "Planetary days will become shorter."


def test_arc_challenge_passes_streaming_flag_to_load_dataset(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args, **kwargs):
        del args
        calls.append(kwargs)
        return _dataset()

    monkeypatch.setattr(arc_challenge_module, "load_dataset", fake_load_dataset)

    suite = evalution.arc_challenge(
        max_rows=1,
        streaming=True,
    )
    session = FakeSession(["C"])
    result = suite.evaluate(session)

    assert result.metadata["streaming"] is True
    assert calls
    assert calls[0]["streaming"] is True


def test_arc_challenge_marks_unparseable_predictions_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(max_rows=1)
    session = FakeSession(["I am not sure."])
    result = suite.evaluate(session)

    assert result.metrics["exact_match,choice-label"] == 0.0
    assert result.metrics["exact_match,choice-text"] == 0.0
    assert result.samples[0].extracted["choice-label"] == "[invalid]"
    assert result.samples[0].extracted["choice-text"] == "[invalid]"
