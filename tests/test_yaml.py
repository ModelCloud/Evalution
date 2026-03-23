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

gsm8k_platinum_module = importlib.import_module("evalution.benchmarks.gsm8k_platinum")
_AIME_TASKS = ["aime", "aime24", "aime25"]
_ARITHMETIC_TASKS = [
    "arithmetic_1dc",
    "arithmetic_2da",
    "arithmetic_2dm",
    "arithmetic_2ds",
    "arithmetic_3da",
    "arithmetic_3ds",
    "arithmetic_4da",
    "arithmetic_4ds",
    "arithmetic_5da",
    "arithmetic_5ds",
]


class FakeEngine(BaseEngine):
    def build(self, model):
        self.model_config = model
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
  type: Transformers
  dtype: bfloat16
model:
  path: /tmp/model
tests:
  - type: anli_r1
    max_rows: 18
  - type: anli_r2
    max_rows: 18
  - type: anli_r3
    max_rows: 18
  - type: asdiv
    max_rows: 18
  - type: asdiv_cot_llama
    max_rows: 18
  - type: babi
    max_rows: 18
  - type: blimp
    subset: adjunct_island
    max_rows: 18
  - type: c4
    max_rows: 8
  - type: gsm8k_platinum
    max_rows: 128
  - type: headqa_en
    max_rows: 18
  - type: headqa_es
    max_rows: 18
  - type: lambada_openai
    max_rows: 18
  - type: lambada_openai_cloze
    max_rows: 18
  - type: lambada_standard
    max_rows: 18
  - type: lambada_standard_cloze
    max_rows: 18
  - type: logiqa
    max_rows: 18
  - type: mathqa
    max_rows: 18
  - type: mc_taco
    max_rows: 18
  - type: medmcqa
    max_rows: 18
  - type: medqa_4options
    max_rows: 18
  - type: boolq
    max_rows: 48
  - type: cb
    max_rows: 56
  - type: cola
    max_rows: 52
  - type: cnn_dailymail
    max_rows: 8
  - type: commonsense_qa
    max_rows: 18
  - type: coqa
    max_rows: 16
  - type: copa
    max_rows: 12
  - type: drop
    max_rows: 16
  - type: ethics_cm
    max_rows: 18
  - type: ethics_deontology
    max_rows: 18
  - type: ethics_justice
    max_rows: 18
  - type: ethics_utilitarianism
    max_rows: 18
  - type: ethics_virtue
    max_rows: 18
  - type: arc_easy
    max_rows: 40
  - type: arc_challenge
    max_rows: 64
  - type: hellaswag
    max_rows: 32
  - type: mmlu
    subsets: stem.abstract_algebra
    num_fewshot: 3
  - type: mmlu_pro
    subsets: stem.math
    num_fewshot: 2
  - type: mnli
    max_rows: 44
  - type: mrpc
    max_rows: 28
  - type: nq_open
    max_rows: 16
  - type: openbookqa
    max_rows: 20
  - type: paws_x_de
    max_rows: 16
  - type: paws_x_en
    max_rows: 16
  - type: paws_x_es
    max_rows: 16
  - type: paws_x_fr
    max_rows: 16
  - type: paws_x_ja
    max_rows: 16
  - type: paws_x_ko
    max_rows: 16
  - type: paws_x_zh
    max_rows: 16
  - type: xcopa_et
    max_rows: 16
  - type: xcopa_ht
    max_rows: 16
  - type: xcopa_id
    max_rows: 16
  - type: xcopa_it
    max_rows: 16
  - type: xcopa_qu
    max_rows: 16
  - type: xcopa_sw
    max_rows: 16
  - type: xcopa_ta
    max_rows: 16
  - type: xcopa_th
    max_rows: 16
  - type: xcopa_tr
    max_rows: 16
  - type: xcopa_vi
    max_rows: 16
  - type: xcopa_zh
    max_rows: 16
  - type: piqa
    max_rows: 16
  - type: pile_10k
    max_rows: 8
  - type: prost
    max_rows: 18
  - type: pubmedqa
    max_rows: 18
  - type: qnli
    max_rows: 26
  - type: qqp
    max_rows: 24
  - type: race
    max_rows: 18
  - type: rte
    max_rows: 18
  - type: sciq
    max_rows: 20
  - type: siqa
    max_rows: 20
  - type: swag
    max_rows: 18
  - type: sst2
    max_rows: 22
  - type: squadv2
    max_rows: 16
  - type: triviaqa
    max_rows: 16
  - type: wic
    max_rows: 14
  - type: webqs
    max_rows: 18
  - type: wikitext
    max_rows: 8
  - type: wsc273
    max_rows: 14
  - type: wnli
    max_rows: 12
  - type: winogrande
    max_rows: 24
"""
    )

    assert "import evalution as eval" in script
    assert "import evalution.benchmarks as benchmarks" in script
    assert "import evalution.engines as engines" in script
    assert "engines.Transformers(" in script
    assert "eval(engines." not in script
    assert ".model(eval.Model(" in script
    assert ".run(benchmarks.anli_r1(" in script
    assert ".run(benchmarks.anli_r2(" in script
    assert ".run(benchmarks.anli_r3(" in script
    assert ".run(benchmarks.asdiv(" in script
    assert ".run(benchmarks.asdiv_cot_llama(" in script
    assert ".run(benchmarks.babi(" in script
    assert ".run(benchmarks.blimp(" in script
    assert ".run(benchmarks.c4(" in script
    assert ".run(benchmarks.gsm8k_platinum(" in script
    assert ".run(benchmarks.headqa_en(" in script
    assert ".run(benchmarks.headqa_es(" in script
    assert ".run(benchmarks.lambada_openai(" in script
    assert ".run(benchmarks.lambada_openai_cloze(" in script
    assert ".run(benchmarks.lambada_standard(" in script
    assert ".run(benchmarks.lambada_standard_cloze(" in script
    assert ".run(benchmarks.logiqa(" in script
    assert ".run(benchmarks.mathqa(" in script
    assert ".run(benchmarks.mc_taco(" in script
    assert ".run(benchmarks.medmcqa(" in script
    assert ".run(benchmarks.medqa_4options(" in script
    assert ".run(benchmarks.boolq(" in script
    assert ".run(benchmarks.cb(" in script
    assert ".run(benchmarks.cola(" in script
    assert ".run(benchmarks.cnn_dailymail(" in script
    assert ".run(benchmarks.commonsense_qa(" in script
    assert ".run(benchmarks.coqa(" in script
    assert ".run(benchmarks.copa(" in script
    assert ".run(benchmarks.drop(" in script
    assert ".run(benchmarks.ethics_cm(" in script
    assert ".run(benchmarks.ethics_deontology(" in script
    assert ".run(benchmarks.ethics_justice(" in script
    assert ".run(benchmarks.ethics_utilitarianism(" in script
    assert ".run(benchmarks.ethics_virtue(" in script
    assert ".run(benchmarks.arc_easy(" in script
    assert ".run(benchmarks.arc_challenge(" in script
    assert ".run(benchmarks.hellaswag(" in script
    assert ".run(benchmarks.mmlu(" in script
    assert ".run(benchmarks.mmlu_pro(" in script
    assert ".run(benchmarks.mnli(" in script
    assert ".run(benchmarks.mrpc(" in script
    assert ".run(benchmarks.nq_open(" in script
    assert ".run(benchmarks.openbookqa(" in script
    assert ".run(benchmarks.paws_x_de(" in script
    assert ".run(benchmarks.paws_x_en(" in script
    assert ".run(benchmarks.paws_x_es(" in script
    assert ".run(benchmarks.paws_x_fr(" in script
    assert ".run(benchmarks.paws_x_ja(" in script
    assert ".run(benchmarks.paws_x_ko(" in script
    assert ".run(benchmarks.paws_x_zh(" in script
    assert ".run(benchmarks.xcopa_et(" in script
    assert ".run(benchmarks.xcopa_ht(" in script
    assert ".run(benchmarks.xcopa_id(" in script
    assert ".run(benchmarks.xcopa_it(" in script
    assert ".run(benchmarks.xcopa_qu(" in script
    assert ".run(benchmarks.xcopa_sw(" in script
    assert ".run(benchmarks.xcopa_ta(" in script
    assert ".run(benchmarks.xcopa_th(" in script
    assert ".run(benchmarks.xcopa_tr(" in script
    assert ".run(benchmarks.xcopa_vi(" in script
    assert ".run(benchmarks.xcopa_zh(" in script
    assert ".run(benchmarks.piqa(" in script
    assert ".run(benchmarks.pile_10k(" in script
    assert ".run(benchmarks.prost(" in script
    assert ".run(benchmarks.pubmedqa(" in script
    assert ".run(benchmarks.qnli(" in script
    assert ".run(benchmarks.qqp(" in script
    assert ".run(benchmarks.race(" in script
    assert ".run(benchmarks.rte(" in script
    assert ".run(benchmarks.sciq(" in script
    assert ".run(benchmarks.siqa(" in script
    assert ".run(benchmarks.swag(" in script
    assert ".run(benchmarks.sst2(" in script
    assert ".run(benchmarks.squadv2(" in script
    assert ".run(benchmarks.triviaqa(" in script
    assert ".run(benchmarks.wic(" in script
    assert ".run(benchmarks.webqs(" in script
    assert ".run(benchmarks.wikitext(" in script
    assert ".run(benchmarks.wsc273(" in script
    assert ".run(benchmarks.wnli(" in script
    assert ".run(benchmarks.winogrande(" in script


def test_python_from_yaml_emits_arithmetic_variants() -> None:
    task_lines = "\n".join(f"  - type: {task}\n    max_rows: 8" for task in _ARITHMETIC_TASKS)
    script = evalution.python_from_yaml(
        f"""
engine:
  type: Transformers
model:
  path: /tmp/model
tests:
{task_lines}
"""
    )

    for task in _ARITHMETIC_TASKS:
        assert f".run(benchmarks.{task}(" in script


def test_python_from_yaml_emits_aime_variants() -> None:
    task_lines = "\n".join(f"  - type: {task}\n    max_rows: 8" for task in _AIME_TASKS)
    script = evalution.python_from_yaml(
        f"""
engine:
  type: Transformers
model:
  path: /tmp/model
tests:
{task_lines}
"""
    )

    for task in _AIME_TASKS:
        assert f".run(benchmarks.{task}(" in script


def test_run_yaml_requires_tests_section() -> None:
    try:
        evalution.run_yaml(
            """
engine:
  type: Transformers
model:
  path: /tmp/model
"""
        )
    except KeyError as exc:
        assert str(exc) == "'yaml spec must define a tests section'"
    else:
        raise AssertionError("expected missing tests section to raise KeyError")


def test_python_from_yaml_emits_transformer_compat_name() -> None:
    script = evalution.python_from_yaml(
        """
engine:
  type: TransformersCompat
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "engines.TransformersCompat(" in script
    assert "eval(engines." not in script


def test_python_from_yaml_emits_gptqmodel_name() -> None:
    script = evalution.python_from_yaml(
        """
engine:
  type: GPTQModel
  backend: auto
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "engines.GPTQModel(" in script
    assert "eval(engines." not in script
