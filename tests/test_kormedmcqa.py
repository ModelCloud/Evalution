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

kormedmcqa_module = importlib.import_module("evalution.benchmarks.kormedmcqa")


def _row(*, subset: str, question: str, answer: int) -> dict[str, object]:
    return {
        "subject": subset,
        "year": 2022,
        "period": 1,
        "q_number": 1,
        "question": question,
        "A": "첫째",
        "B": "둘째",
        "C": "셋째",
        "D": "넷째",
        "E": "다섯째",
        "answer": answer,
        "cot": "",
    }


class FakeSession:
    def __init__(self, response: str):
        self.response = response
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size == 1
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else "",
                text=self.response,
            )
            for request in requests
        ]

    def close(self) -> None:
        return None


def test_kormedmcqa_doctor_builds_fewshot_prompt_and_scores(monkeypatch) -> None:
    test_dataset = Dataset.from_list([_row(subset="doctor", question="실전 질문", answer=2)])
    fewshot_dataset = Dataset.from_list(
        [_row(subset="doctor", question=f"예시 질문 {index}", answer=1) for index in range(5)]
    )

    def fake_load_dataset(path, name, split, cache_dir=None, streaming=False):
        assert path == "sean0042/KorMedMCQA"
        assert streaming is False
        if split == "fewshot":
            return fewshot_dataset
        return test_dataset

    monkeypatch.setattr(kormedmcqa_module, "load_dataset", fake_load_dataset)
    session = FakeSession(" B.\n")
    result = evalution.benchmarks.kormedmcqa_doctor(max_rows=1, batch_size=1, stream=False).evaluate(session)

    assert result.name == "kormedmcqa_doctor"
    assert result.metrics == {"em": 1.0}
    assert result.metadata["subset"] == "doctor"
    assert result.metadata["num_fewshot"] == 5
    request = session.requests[0]
    assert "예시 질문 0" in request.prompt
    assert request.prompt.endswith("정답：")
    assert request.stop == ["Q:", "</s>", "<|im_end|>", ".", "\n\n"]
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.extracted == {
        "prediction-stripped": "B",
        "target-stripped": "B",
    }
    assert sample.metadata["subset"] == "doctor"
    assert sample.metadata["raw_choices"] == ["첫째", "둘째", "셋째", "넷째", "다섯째"]


def test_kormedmcqa_group_loader_round_robins_subsets(monkeypatch) -> None:
    datasets = {
        "doctor": Dataset.from_list([_row(subset="doctor", question="d1", answer=1)]),
        "nurse": Dataset.from_list([_row(subset="nurse", question="n1", answer=1), _row(subset="nurse", question="n2", answer=1)]),
        "pharm": Dataset.from_list([_row(subset="pharm", question="p1", answer=1)]),
        "dentist": Dataset.from_list([_row(subset="dentist", question="de1", answer=1)]),
    }

    def fake_load_dataset(path, name, split, cache_dir=None, streaming=False):
        assert path == "sean0042/KorMedMCQA"
        assert split == "test"
        assert streaming is False
        return datasets[name]

    monkeypatch.setattr(kormedmcqa_module, "load_dataset", fake_load_dataset)
    suite = evalution.benchmarks.kormedmcqa(max_rows=5, stream=False)
    loaded = suite.dataset_loader()(suite.dataset_path, suite.dataset_name, split=suite.split)
    assert loaded[0]["question"] == "d1"
    assert loaded[1]["question"] == "n1"
    assert loaded[2]["question"] == "p1"
    assert loaded[3]["question"] == "de1"
    assert loaded[4]["question"] == "n2"
