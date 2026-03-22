# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution import yaml as evalution_yaml
from evalution.engines.base import BaseEngine, BaseInferenceSession, GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.suites.gsm8k_platinum")


class FakeEngine(BaseEngine):
    def build(self, model):
        self.model = model
        return FakeSession()

    def to_dict(self):
        return {"name": "fake"}


class FakeSession(BaseInferenceSession):
    def generate(self, requests, *, batch_size=None):
        del batch_size
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text="The answer is 42.",
            )
            for request in requests
        ]

    def describe_execution(self):
        return {"generation_backend": "fake"}

    def loglikelihood(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size=None):
        request_items = list(requests)
        outputs = self.generate([request for _, request in request_items], batch_size=batch_size)
        for (item_id, _request), output in zip(request_items, outputs, strict=True):
            yield item_id, output

    def gc(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_run_yaml_executes_yaml_spec_and_returns_structured_result(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setitem(evalution_yaml._ENGINE_FACTORIES, "fake", FakeEngine)

    result = evalution.run_yaml(
        """
engine:
  type: fake
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 1
"""
    )

    assert result.model["path"] == "/tmp/model"
    assert result.engine["name"] == "fake"
    assert result.engine["execution"]["generation_backend"] == "fake"
    assert len(result.tests) == 1
    assert result.tests[0].name == "gsm8k_platinum_cot"


def test_python_from_yaml_emits_fluent_python_api() -> None:
    script = evalution.python_from_yaml(
        """
engine:
  type: transformers
  dtype: bfloat16
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 128
  - type: boolq
    max_rows: 48
  - type: cb
    max_rows: 56
  - type: cola
    max_rows: 52
  - type: copa
    max_rows: 12
  - type: arc_easy
    max_rows: 40
  - type: arc_challenge
    max_rows: 64
  - type: hellaswag
    max_rows: 32
  - type: mmlu
    subset: stem.abstract_algebra
    num_fewshot: 3
  - type: mmlu_pro
    subset: stem.math
    num_fewshot: 2
  - type: mnli
    max_rows: 44
  - type: mrpc
    max_rows: 28
  - type: openbookqa
    max_rows: 20
  - type: piqa
    max_rows: 16
  - type: qnli
    max_rows: 26
  - type: qqp
    max_rows: 24
  - type: rte
    max_rows: 18
  - type: sst2
    max_rows: 22
  - type: wic
    max_rows: 14
  - type: wnli
    max_rows: 12
  - type: winogrande
    max_rows: 24
"""
    )

    assert "import evalution as eval" in script
    assert "eval.engine(eval.Transformers(" in script
    assert ".model(eval.Model(" in script
    assert ".run(eval.gsm8k_platinum(" in script
    assert ".run(eval.boolq(" in script
    assert ".run(eval.cb(" in script
    assert ".run(eval.cola(" in script
    assert ".run(eval.copa(" in script
    assert ".run(eval.arc_easy(" in script
    assert ".run(eval.arc_challenge(" in script
    assert ".run(eval.hellaswag(" in script
    assert ".run(eval.mmlu(" in script
    assert ".run(eval.mmlu_pro(" in script
    assert ".run(eval.mnli(" in script
    assert ".run(eval.mrpc(" in script
    assert ".run(eval.openbookqa(" in script
    assert ".run(eval.piqa(" in script
    assert ".run(eval.qnli(" in script
    assert ".run(eval.qqp(" in script
    assert ".run(eval.rte(" in script
    assert ".run(eval.sst2(" in script
    assert ".run(eval.wic(" in script
    assert ".run(eval.wnli(" in script
    assert ".run(eval.winogrande(" in script


def test_run_yaml_requires_tests_section() -> None:
    try:
        evalution.run_yaml(
            """
engine:
  type: transformers
model:
  path: /tmp/model
"""
        )
    except KeyError as exc:
        assert str(exc) == "'yaml spec must define a tests section'"
    else:
        raise AssertionError("expected missing tests section to raise KeyError")


def test_python_from_yaml_emits_transformer_compat_alias() -> None:
    script = evalution.python_from_yaml(
        """
engine:
  type: transformer_compat
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "eval.engine(eval.TransformersCompat(" in script
