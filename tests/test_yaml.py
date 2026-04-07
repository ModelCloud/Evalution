# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
from dataclasses import fields

from datasets import Dataset

import evalution
from evalution import yaml as evalution_yaml
from evalution.engines import (
    BaseEngineDeviceConfig,
    BaseEnginePagedBatchingConfig,
    BaseEngineQuantizationConfig,
    BaseEngineTokenizerModeConfig,
    BaseEngineTransformersRuntimeConfig,
    GPTQModel,
    OpenVINO,
    SGLang,
    SharedEngineConfig,
    Transformers,
    TransformersCompat,
    VLLM,
)
from evalution.engines.base import BaseEngine, BaseInferenceSession, GenerationOutput

gsm8k_platinum_module = importlib.import_module("evalution.benchmarks.gsm8k_platinum")
_AIME_TASKS = ["aime", "aime24", "aime25"]
_HENDRYCKS_MATH_TASKS = [
    "hendrycks_math_algebra",
    "hendrycks_math_counting_and_probability",
    "hendrycks_math_geometry",
    "hendrycks_math_intermediate_algebra",
    "hendrycks_math_number_theory",
    "hendrycks_math_prealgebra",
    "hendrycks_math_precalculus",
]
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
    monkeypatch.setitem(
        evalution_yaml._ENGINE_REGISTRY,
        "fake",
        FakeEngine,
    )

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
  - type: aexams_biology
    max_rows: 18
  - type: aexams_islamic_studies
    max_rows: 18
  - type: aexams_physics
    max_rows: 18
  - type: aexams_science
    max_rows: 18
  - type: aexams_social
    max_rows: 18
  - type: agieval
    subset: aqua-rat
    max_rows: 16
  - type: agieval_aqua_rat
    max_rows: 16
  - type: afrimgsm
    language: eng
    max_rows: 16
  - type: afrimgsm_eng
    max_rows: 16
  - type: afrimmlu
    language: eng
    max_rows: 16
  - type: afrimmlu_eng
    max_rows: 16
  - type: afrixnli_amh
    max_rows: 16
  - type: afrixnli_eng
    max_rows: 16
  - type: afrixnli_ewe
    max_rows: 16
  - type: afrixnli_fra
    max_rows: 16
  - type: afrixnli_hau
    max_rows: 16
  - type: afrixnli_ibo
    max_rows: 16
  - type: afrixnli_kin
    max_rows: 16
  - type: afrixnli_lin
    max_rows: 16
  - type: afrixnli_lug
    max_rows: 16
  - type: afrixnli_orm
    max_rows: 16
  - type: afrixnli_sna
    max_rows: 16
  - type: afrixnli_sot
    max_rows: 16
  - type: afrixnli_swa
    max_rows: 16
  - type: afrixnli_twi
    max_rows: 16
  - type: afrixnli_wol
    max_rows: 16
  - type: afrixnli_xho
    max_rows: 16
  - type: afrixnli_yor
    max_rows: 16
  - type: afrixnli_zul
    max_rows: 16
  - type: anli_r1
    max_rows: 18
  - type: anli_r2
    max_rows: 18
  - type: anli_r3
    max_rows: 18
  - type: arabicmmlu
    subset: All
    max_rows: 16
  - type: arabicmmlu_islamic_studies
    max_rows: 16
  - type: darijammlu
    subset: accounting
    max_rows: 16
  - type: darijammlu_biology
    max_rows: 16
  - type: egymmlu
    subset: accounting
    max_rows: 16
  - type: egymmlu_biology
    max_rows: 16
  - type: eus_exams
    subset: es_ejadministrativo
    max_rows: 16
  - type: eus_exams_eu_opeosakiadmineu
    max_rows: 16
  - type: careqa
    language: en
    max_rows: 16
  - type: careqa_es
    max_rows: 16
  - type: cabbq
    category: Age
    max_rows: 16
  - type: cabbq_gender
    max_rows: 16
  - type: esbbq
    category: Age
    max_rows: 16
  - type: esbbq_gender
    max_rows: 16
  - type: arc_mt
    language: da
    max_rows: 16
  - type: arc_mt_is
    max_rows: 16
  - type: asdiv
    max_rows: 18
  - type: asdiv_cot_llama
    max_rows: 18
  - type: babi
    max_rows: 18
  - type: babilong
    qa_split: qa1
    max_rows: 18
  - type: babilong_qa2
    max_rows: 18
  - type: bbh
    subset: boolean_expressions
    max_rows: 18
  - type: bbh_date_understanding
    max_rows: 18
  - type: bangla
    subset: boolqa
    max_rows: 16
  - type: bangla_boolqa
    max_rows: 16
  - type: bangla_commonsenseqa
    max_rows: 16
  - type: bangla_mmlu
    max_rows: 16
  - type: bangla_openbookqa
    max_rows: 16
  - type: bangla_piqa
    max_rows: 16
  - type: bear
    max_rows: 18
  - type: bear_big
    max_rows: 18
  - type: belebele
    language: eng_Latn
    max_rows: 16
  - type: bbq
    category: Age
    max_rows: 16
  - type: bbq_gender_identity
    max_rows: 16
  - type: blimp
    subset: adjunct_island
    max_rows: 18
  - type: c4
    max_rows: 8
  - type: ceval
    subset: accountant
    max_rows: 18
  - type: gsm8k_platinum
    max_rows: 128
  - type: headqa_en
    max_rows: 18
  - type: headqa_es
    max_rows: 18
  - type: hendrycks_math
    subset: algebra
    max_rows: 16
  - type: hendrycks_math_geometry
    max_rows: 16
  - type: histoires_morales
    max_rows: 16
  - type: lambada_openai
    max_rows: 18
  - type: lambada_openai_cloze
    max_rows: 18
  - type: lambada_standard
    max_rows: 18
  - type: lambada_standard_cloze
    max_rows: 18
  - type: icelandic_winogrande
    max_rows: 16
  - type: inverse_scaling
    subset: hindsight-neglect
    max_rows: 16
  - type: inverse_scaling_hindsight_neglect
    max_rows: 16
  - type: logiqa
    max_rows: 18
  - type: logiqa2
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
  - type: code2text_go
    max_rows: 8
  - type: code2text_java
    max_rows: 8
  - type: code2text_javascript
    max_rows: 8
  - type: code2text_php
    max_rows: 8
  - type: code2text_python
    max_rows: 8
  - type: code2text_ruby
    max_rows: 8
  - type: commonsense_qa
    max_rows: 18
  - type: copal_id_standard
    max_rows: 18
  - type: copal_id_colloquial
    max_rows: 18
  - type: coqa
    max_rows: 16
  - type: copa
    max_rows: 12
  - type: darijahellaswag
    max_rows: 16
  - type: egyhellaswag
    max_rows: 16
  - type: copa_ar
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
  - type: mutual
    max_rows: 16
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
  - type: xquad
    language: en
    max_rows: 16
  - type: xquad_es
    max_rows: 16
  - type: xstorycloze_ar
    max_rows: 16
  - type: xstorycloze_en
    max_rows: 16
  - type: xstorycloze_es
    max_rows: 16
  - type: xstorycloze_eu
    max_rows: 16
  - type: xstorycloze_hi
    max_rows: 16
  - type: xstorycloze_id
    max_rows: 16
  - type: xstorycloze_my
    max_rows: 16
  - type: xstorycloze_ru
    max_rows: 16
  - type: xstorycloze_sw
    max_rows: 16
  - type: xstorycloze_te
    max_rows: 16
  - type: xstorycloze_zh
    max_rows: 16
  - type: xnli
    language: en
    max_rows: 16
  - type: xnli_fr
    max_rows: 16
  - type: xwinograd_en
    max_rows: 16
  - type: xwinograd_fr
    max_rows: 16
  - type: xwinograd_jp
    max_rows: 16
  - type: xwinograd_pt
    max_rows: 16
  - type: xwinograd_ru
    max_rows: 16
  - type: xwinograd_zh
    max_rows: 16
  - type: piqa
    max_rows: 16
  - type: piqa_ar
    max_rows: 16
  - type: pile_10k
    max_rows: 8
  - type: prost
    max_rows: 18
  - type: pubmedqa
    max_rows: 18
  - type: qa4mre_2011
    max_rows: 16
  - type: qa4mre_2012
    max_rows: 16
  - type: qa4mre_2013
    max_rows: 16
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
  - type: truthfulqa
    variant: mc1
    max_rows: 16
  - type: truthfulqa_mc2
    max_rows: 16
  - type: triviaqa
    max_rows: 16
  - type: wic
    max_rows: 14
  - type: webqs
    max_rows: 18
  - type: wikitext
    max_rows: 8
  - type: wmdp
    subset: bio
    max_rows: 16
  - type: wmdp_chem
    max_rows: 16
  - type: winogender_all
    max_rows: 16
  - type: winogender_female
    max_rows: 16
  - type: winogender_gotcha
    max_rows: 16
  - type: winogender_gotcha_female
    max_rows: 16
  - type: winogender_gotcha_male
    max_rows: 16
  - type: winogender_male
    max_rows: 16
  - type: winogender_neutral
    max_rows: 16
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
    assert ".model(\n        path=" in script
    assert ".run(benchmarks.aexams_biology(" in script
    assert ".run(benchmarks.aexams_islamic_studies(" in script
    assert ".run(benchmarks.aexams_physics(" in script
    assert ".run(benchmarks.aexams_science(" in script
    assert ".run(benchmarks.aexams_social(" in script
    assert ".run(benchmarks.agieval(" in script
    assert ".run(benchmarks.agieval_aqua_rat(" in script
    assert ".run(benchmarks.afrimgsm(" in script
    assert ".run(benchmarks.afrimgsm_eng(" in script
    assert ".run(benchmarks.afrimmlu(" in script
    assert ".run(benchmarks.afrimmlu_eng(" in script
    assert ".run(benchmarks.afrixnli_amh(" in script
    assert ".run(benchmarks.afrixnli_eng(" in script
    assert ".run(benchmarks.afrixnli_ewe(" in script
    assert ".run(benchmarks.afrixnli_fra(" in script
    assert ".run(benchmarks.afrixnli_hau(" in script
    assert ".run(benchmarks.afrixnli_ibo(" in script
    assert ".run(benchmarks.afrixnli_kin(" in script
    assert ".run(benchmarks.afrixnli_lin(" in script
    assert ".run(benchmarks.afrixnli_lug(" in script
    assert ".run(benchmarks.afrixnli_orm(" in script
    assert ".run(benchmarks.afrixnli_sna(" in script
    assert ".run(benchmarks.afrixnli_sot(" in script
    assert ".run(benchmarks.afrixnli_swa(" in script
    assert ".run(benchmarks.afrixnli_twi(" in script
    assert ".run(benchmarks.afrixnli_wol(" in script
    assert ".run(benchmarks.afrixnli_xho(" in script
    assert ".run(benchmarks.afrixnli_yor(" in script
    assert ".run(benchmarks.afrixnli_zul(" in script
    assert ".run(benchmarks.anli_r1(" in script
    assert ".run(benchmarks.anli_r2(" in script
    assert ".run(benchmarks.anli_r3(" in script
    assert ".run(benchmarks.arabicmmlu(" in script
    assert ".run(benchmarks.arabicmmlu_islamic_studies(" in script
    assert ".run(benchmarks.darijammlu(" in script
    assert ".run(benchmarks.darijammlu_biology(" in script
    assert ".run(benchmarks.egymmlu(" in script
    assert ".run(benchmarks.egymmlu_biology(" in script
    assert ".run(benchmarks.eus_exams(" in script
    assert ".run(benchmarks.eus_exams_eu_opeosakiadmineu(" in script
    assert ".run(benchmarks.careqa(" in script
    assert ".run(benchmarks.careqa_es(" in script
    assert ".run(benchmarks.cabbq(" in script
    assert ".run(benchmarks.cabbq_gender(" in script
    assert ".run(benchmarks.esbbq(" in script
    assert ".run(benchmarks.esbbq_gender(" in script
    assert ".run(benchmarks.arc_mt(" in script
    assert ".run(benchmarks.arc_mt_is(" in script
    assert ".run(benchmarks.asdiv(" in script
    assert ".run(benchmarks.asdiv_cot_llama(" in script
    assert ".run(benchmarks.babi(" in script
    assert ".run(benchmarks.babilong(" in script
    assert ".run(benchmarks.babilong_qa2(" in script
    assert ".run(benchmarks.bbh(" in script
    assert ".run(benchmarks.bbh_date_understanding(" in script
    assert ".run(benchmarks.bangla(" in script
    assert ".run(benchmarks.bangla_boolqa(" in script
    assert ".run(benchmarks.bangla_commonsenseqa(" in script
    assert ".run(benchmarks.bangla_mmlu(" in script
    assert ".run(benchmarks.bangla_openbookqa(" in script
    assert ".run(benchmarks.bangla_piqa(" in script
    assert ".run(benchmarks.bear(" in script
    assert ".run(benchmarks.bear_big(" in script
    assert ".run(benchmarks.belebele(" in script
    assert ".run(benchmarks.bbq(" in script
    assert ".run(benchmarks.bbq_gender_identity(" in script
    assert ".run(benchmarks.blimp(" in script
    assert ".run(benchmarks.c4(" in script
    assert ".run(benchmarks.ceval(" in script
    assert ".run(benchmarks.gsm8k_platinum(" in script
    assert ".run(benchmarks.headqa_en(" in script
    assert ".run(benchmarks.headqa_es(" in script
    assert ".run(benchmarks.histoires_morales(" in script
    assert ".run(benchmarks.lambada_openai(" in script
    assert ".run(benchmarks.lambada_openai_cloze(" in script
    assert ".run(benchmarks.lambada_standard(" in script
    assert ".run(benchmarks.lambada_standard_cloze(" in script
    assert ".run(benchmarks.icelandic_winogrande(" in script
    assert ".run(benchmarks.inverse_scaling(" in script
    assert ".run(benchmarks.inverse_scaling_hindsight_neglect(" in script
    assert ".run(benchmarks.logiqa(" in script
    assert ".run(benchmarks.logiqa2(" in script
    assert ".run(benchmarks.mathqa(" in script
    assert ".run(benchmarks.mc_taco(" in script
    assert ".run(benchmarks.medmcqa(" in script
    assert ".run(benchmarks.medqa_4options(" in script
    assert ".run(benchmarks.boolq(" in script
    assert ".run(benchmarks.cb(" in script
    assert ".run(benchmarks.cola(" in script
    assert ".run(benchmarks.cnn_dailymail(" in script
    assert ".run(benchmarks.code2text_go(" in script
    assert ".run(benchmarks.code2text_java(" in script
    assert ".run(benchmarks.code2text_javascript(" in script
    assert ".run(benchmarks.code2text_php(" in script
    assert ".run(benchmarks.code2text_python(" in script
    assert ".run(benchmarks.code2text_ruby(" in script
    assert ".run(benchmarks.commonsense_qa(" in script
    assert ".run(benchmarks.copal_id_standard(" in script
    assert ".run(benchmarks.copal_id_colloquial(" in script
    assert ".run(benchmarks.coqa(" in script
    assert ".run(benchmarks.copa(" in script
    assert ".run(benchmarks.darijahellaswag(" in script
    assert ".run(benchmarks.egyhellaswag(" in script
    assert ".run(benchmarks.copa_ar(" in script
    assert ".run(benchmarks.drop(" in script
    assert ".run(benchmarks.ethics_cm(" in script
    assert ".run(benchmarks.ethics_deontology(" in script
    assert ".run(benchmarks.ethics_justice(" in script
    assert ".run(benchmarks.ethics_utilitarianism(" in script
    assert ".run(benchmarks.ethics_virtue(" in script
    assert ".run(benchmarks.arc_easy(" in script
    assert ".run(benchmarks.arc_challenge(" in script
    assert ".run(benchmarks.hendrycks_math(" in script
    assert ".run(benchmarks.hendrycks_math_geometry(" in script
    assert ".run(benchmarks.hellaswag(" in script
    assert ".run(benchmarks.mmlu(" in script
    assert ".run(benchmarks.mmlu_pro(" in script
    assert ".run(benchmarks.mnli(" in script
    assert ".run(benchmarks.mrpc(" in script
    assert ".run(benchmarks.mutual(" in script
    assert ".run(benchmarks.nq_open(" in script
    assert ".run(benchmarks.openbookqa(" in script
    assert ".run(benchmarks.paws_x_de(" in script
    assert ".run(benchmarks.paws_x_en(" in script
    assert ".run(benchmarks.paws_x_es(" in script
    assert ".run(benchmarks.paws_x_fr(" in script
    assert ".run(benchmarks.paws_x_ja(" in script
    assert ".run(benchmarks.paws_x_ko(" in script
    assert ".run(benchmarks.paws_x_zh(" in script
    assert ".run(benchmarks.piqa(" in script
    assert ".run(benchmarks.piqa_ar(" in script
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
    assert ".run(benchmarks.xquad(" in script
    assert ".run(benchmarks.xquad_es(" in script
    assert ".run(benchmarks.xstorycloze_ar(" in script
    assert ".run(benchmarks.xstorycloze_en(" in script
    assert ".run(benchmarks.xstorycloze_es(" in script
    assert ".run(benchmarks.xstorycloze_eu(" in script
    assert ".run(benchmarks.xstorycloze_hi(" in script
    assert ".run(benchmarks.xstorycloze_id(" in script
    assert ".run(benchmarks.xstorycloze_my(" in script
    assert ".run(benchmarks.xstorycloze_ru(" in script
    assert ".run(benchmarks.xstorycloze_sw(" in script
    assert ".run(benchmarks.xstorycloze_te(" in script
    assert ".run(benchmarks.xstorycloze_zh(" in script
    assert ".run(benchmarks.xnli(" in script
    assert ".run(benchmarks.xnli_fr(" in script
    assert ".run(benchmarks.xwinograd_en(" in script
    assert ".run(benchmarks.xwinograd_fr(" in script
    assert ".run(benchmarks.xwinograd_jp(" in script
    assert ".run(benchmarks.xwinograd_pt(" in script
    assert ".run(benchmarks.xwinograd_ru(" in script
    assert ".run(benchmarks.xwinograd_zh(" in script
    assert ".run(benchmarks.pile_10k(" in script
    assert ".run(benchmarks.prost(" in script
    assert ".run(benchmarks.pubmedqa(" in script
    assert ".run(benchmarks.qa4mre_2011(" in script
    assert ".run(benchmarks.qa4mre_2012(" in script
    assert ".run(benchmarks.qa4mre_2013(" in script
    assert ".run(benchmarks.qnli(" in script
    assert ".run(benchmarks.qqp(" in script
    assert ".run(benchmarks.race(" in script
    assert ".run(benchmarks.rte(" in script
    assert ".run(benchmarks.sciq(" in script
    assert ".run(benchmarks.siqa(" in script
    assert ".run(benchmarks.swag(" in script
    assert ".run(benchmarks.sst2(" in script
    assert ".run(benchmarks.squadv2(" in script
    assert ".run(benchmarks.truthfulqa(" in script
    assert ".run(benchmarks.truthfulqa_mc2(" in script
    assert ".run(benchmarks.triviaqa(" in script
    assert ".run(benchmarks.wic(" in script
    assert ".run(benchmarks.webqs(" in script
    assert ".run(benchmarks.wikitext(" in script
    assert ".run(benchmarks.wmdp(" in script
    assert ".run(benchmarks.wmdp_chem(" in script
    assert ".run(benchmarks.winogender_all(" in script
    assert ".run(benchmarks.winogender_female(" in script
    assert ".run(benchmarks.winogender_gotcha(" in script
    assert ".run(benchmarks.winogender_gotcha_female(" in script
    assert ".run(benchmarks.winogender_gotcha_male(" in script
    assert ".run(benchmarks.winogender_male(" in script
    assert ".run(benchmarks.winogender_neutral(" in script
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


def test_python_from_yaml_emits_openvino_name() -> None:
    script = evalution.python_from_yaml(
        """
engine:
  type: OpenVINO
  device: GPU
  compile: false
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "engines.OpenVINO(" in script
    assert "device='GPU'" in script
    assert "compile=False" in script
    assert "eval(engines." not in script


def test_python_from_yaml_emits_vllm_name() -> None:
    """Verify python_from_yaml emits the explicit VLLM engine constructor name."""

    # What this test is actually verifying:
    # YAML conversion should reference engines.VLLM directly rather than using
    # eval-based engine lookup when the engine type is VLLM.
    script = evalution.python_from_yaml(
        """
engine:
  type: VLLM
  tensor_parallel_size: 2
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "engines.VLLM(" in script
    assert "eval(engines." not in script


def test_non_yaml_engine_api_uses_shared_engine_config_inheritance() -> None:
    """Verify the public engine classes inherit common runtime controls from one shared base."""

    assert issubclass(Transformers, SharedEngineConfig)
    assert issubclass(TransformersCompat, SharedEngineConfig)
    assert issubclass(GPTQModel, SharedEngineConfig)
    assert issubclass(OpenVINO, SharedEngineConfig)
    assert issubclass(VLLM, SharedEngineConfig)
    assert issubclass(SGLang, SharedEngineConfig)
    assert issubclass(OpenVINO, BaseEngineDeviceConfig)
    assert issubclass(TransformersCompat, BaseEngineTransformersRuntimeConfig)
    assert issubclass(Transformers, BaseEngineTransformersRuntimeConfig)
    assert issubclass(GPTQModel, BaseEngineTransformersRuntimeConfig)
    assert issubclass(Transformers, BaseEnginePagedBatchingConfig)
    assert issubclass(GPTQModel, BaseEnginePagedBatchingConfig)
    assert issubclass(VLLM, BaseEngineTokenizerModeConfig)
    assert issubclass(SGLang, BaseEngineTokenizerModeConfig)
    assert issubclass(VLLM, BaseEngineQuantizationConfig)
    assert issubclass(SGLang, BaseEngineQuantizationConfig)
    assert issubclass(SGLang, BaseEngineDeviceConfig)


def test_repeated_engine_keys_live_on_dedicated_base_configs() -> None:
    """Verify repeated engine keys are defined once on dedicated base config classes."""

    assert {field.name for field in fields(BaseEngineDeviceConfig)} == {"device"}
    assert {field.name for field in fields(BaseEngineTokenizerModeConfig)} == {"tokenizer_mode"}
    assert {field.name for field in fields(BaseEngineQuantizationConfig)} == {"quantization"}
    assert {field.name for field in fields(BaseEngineTransformersRuntimeConfig)} == {
        "device",
        "attn_implementation",
        "device_map",
    }
    assert {field.name for field in fields(BaseEnginePagedBatchingConfig)} == {
        "manual_eviction",
        "allow_block_sharing",
        "max_blocks_per_request",
        "use_async_batching",
        "use_cuda_graph",
        "q_padding_interval_size",
        "kv_padding_interval_size",
        "max_cached_graphs",
    }


def test_engine_option_keys_are_inherited_from_engine_dataclass_hierarchy() -> None:
    """Verify YAML engine option keys come from the engine class inheritance tree."""

    # What this test is actually verifying:
    # Shared runtime keys should come from the shared engine config base class,
    # while engine-family-specific keys stay scoped to that family.
    transformers_keys = evalution_yaml._engine_option_keys("transformers")
    gptqmodel_keys = evalution_yaml._engine_option_keys("gptqmodel")
    transformers_compat_keys = evalution_yaml._engine_option_keys("transformerscompat")
    vllm_keys = evalution_yaml._engine_option_keys("vllm")
    sglang_keys = evalution_yaml._engine_option_keys("sglang")
    openvino_keys = evalution_yaml._engine_option_keys("openvino")

    assert "dtype" in transformers_keys
    assert "batch_size" in transformers_keys
    assert "padding_side" in transformers_keys
    assert "attn_implementation" in transformers_keys
    assert "continuous_batching" in transformers_keys

    assert "dtype" in gptqmodel_keys
    assert "batch_size" in gptqmodel_keys
    assert "attn_implementation" in gptqmodel_keys
    assert "manual_eviction" in gptqmodel_keys
    assert "backend" in gptqmodel_keys
    assert "gptqmodel_path" in gptqmodel_keys
    assert "continuous_batching" not in gptqmodel_keys

    assert "dtype" in transformers_compat_keys
    assert "batch_size" in transformers_compat_keys
    assert "attn_implementation" in transformers_compat_keys
    assert "continuous_batching" not in transformers_compat_keys
    assert "manual_eviction" not in transformers_compat_keys

    assert "dtype" in vllm_keys
    assert "batch_size" in vllm_keys
    assert "tensor_parallel_size" in vllm_keys
    assert "vllm_path" in vllm_keys
    assert "attn_implementation" not in vllm_keys
    assert "device" not in vllm_keys

    assert "dtype" in sglang_keys
    assert "batch_size" in sglang_keys
    assert "tp_size" in sglang_keys
    assert "sampling_params" in sglang_keys
    assert "attn_implementation" not in sglang_keys

    assert "dtype" in openvino_keys
    assert "batch_size" in openvino_keys
    assert "device" in openvino_keys
    assert "compile" in openvino_keys
    assert "dynamic_shapes" in openvino_keys
    assert "ov_config" in openvino_keys
    assert "attn_implementation" not in openvino_keys


def test_build_engine_constructs_openvino() -> None:
    engine = evalution_yaml._build_engine(
        {
            "type": "OpenVINO",
            "device": "GPU",
            "compile": False,
        }
    )

    assert isinstance(engine, OpenVINO)
    assert engine.device == "GPU"
    assert engine.compile is False


def test_build_engine_constructs_sglang() -> None:
    """Verify YAML engine construction resolves the registered SGLang engine."""

    engine = evalution_yaml._build_engine(
        {
            "type": "SGLang",
            "tp_size": 2,
            "seed": 7,
        }
    )

    assert isinstance(engine, SGLang)
    assert engine.tp_size == 2
    assert engine.seed == 7


def test_python_from_yaml_rejects_engine_keys_from_the_wrong_engine_family() -> None:
    """Verify YAML emission rejects keys that belong to a different engine family."""

    # What this test is actually verifying:
    # VLLM should not silently accept transformers-only keys just because some
    # engines share other runtime options through inheritance.
    try:
        evalution.python_from_yaml(
            """
engine:
  type: VLLM
  attn_implementation: paged|flash_attention_2
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
        )
    except KeyError as exc:
        message = str(exc)
        assert "does not accept option(s): attn_implementation" in message
        assert "tensor_parallel_size" in message
        assert "attn_implementation" not in message.split("allowed options: ", maxsplit=1)[1]
    else:
        raise AssertionError("expected wrong-family engine options to raise KeyError")


def test_python_from_yaml_emits_sglang_name() -> None:
    """Verify python_from_yaml emits the explicit SGLang engine constructor name."""

    script = evalution.python_from_yaml(
        """
engine:
  type: SGLang
  tp_size: 2
  seed: 7
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
    )

    assert "engines.SGLang(" in script
    assert "tp_size=2" in script
    assert "seed=7" in script
    assert "eval(engines." not in script


def test_python_from_yaml_rejects_unknown_engine_type() -> None:
    """Verify YAML emission rejects unknown engine names instead of emitting invalid code."""

    # What this test is actually verifying:
    # python_from_yaml should fail with the same unknown-engine error shape as
    # run_yaml rather than returning an engines.<unknown>(...) call string.
    try:
        evalution.python_from_yaml(
            """
engine:
  type: HF
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 8
"""
        )
    except KeyError as exc:
        assert str(exc) == "\"unknown engine type: 'hf'\""
    else:
        raise AssertionError("expected unknown engine type to raise KeyError")
