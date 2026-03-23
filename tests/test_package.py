# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import evalution


def test_package_import() -> None:
    assert evalution is not None
    assert not callable(evalution)
    assert evalution.benchmarks is not None
    assert evalution.engines is not None
    assert evalution.BaseEngine is not None
    assert evalution.BaseInferenceSession is not None
    assert evalution.GPTQModel is not None
    assert evalution.Transformers is not None
    assert evalution.TransformersCompat is not None
    assert evalution.engines.GPTQModel is not None
    assert evalution.engines.Transformers is not None
    assert evalution.engines.TransformersCompat is not None


def test_package_exports_benchmarks_namespace() -> None:
    assert evalution.benchmarks.AIME is not None
    assert callable(evalution.benchmarks.aime)
    assert callable(evalution.benchmarks.aime24)
    assert callable(evalution.benchmarks.aime25)
    assert evalution.benchmarks.ANLI is not None
    assert callable(evalution.benchmarks.anli_r1)
    assert callable(evalution.benchmarks.anli_r2)
    assert callable(evalution.benchmarks.anli_r3)
    assert evalution.benchmarks.ARCChallenge is not None
    assert callable(evalution.benchmarks.arc_challenge)
    assert evalution.benchmarks.ARCEasy is not None
    assert callable(evalution.benchmarks.arc_easy)
    assert evalution.benchmarks.Arithmetic is not None
    for factory_name in (
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
    ):
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.ASDiv is not None
    assert evalution.benchmarks.ASDivCoTLlama is not None
    assert callable(evalution.benchmarks.asdiv)
    assert callable(evalution.benchmarks.asdiv_cot_llama)
    assert evalution.benchmarks.BABI is not None
    assert callable(evalution.benchmarks.babi)
    assert evalution.benchmarks.BLiMP is not None
    assert evalution.benchmarks.BLIMP_SUBSETS
    assert callable(evalution.benchmarks.blimp)
    assert evalution.benchmarks.BaseRollingPerplexitySuite is not None
    assert evalution.benchmarks.C4 is not None
    assert callable(evalution.benchmarks.c4)
    assert evalution.benchmarks.CEval is not None
    assert evalution.benchmarks.CEVAL_SUBSETS
    assert callable(evalution.benchmarks.ceval)
    assert evalution.benchmarks.BoolQ is not None
    assert callable(evalution.benchmarks.boolq)
    assert evalution.benchmarks.CB is not None
    assert callable(evalution.benchmarks.cb)
    assert evalution.benchmarks.CoLA is not None
    assert callable(evalution.benchmarks.cola)
    assert evalution.benchmarks.CNNDailyMail is not None
    assert callable(evalution.benchmarks.cnn_dailymail)
    assert evalution.benchmarks.CommonsenseQA is not None
    assert callable(evalution.benchmarks.commonsense_qa)
    assert evalution.benchmarks.COPALID is not None
    assert callable(evalution.benchmarks.copal_id)
    assert callable(evalution.benchmarks.copal_id_standard)
    assert callable(evalution.benchmarks.copal_id_colloquial)
    assert evalution.benchmarks.CoQA is not None
    assert callable(evalution.benchmarks.coqa)
    assert evalution.benchmarks.COPA is not None
    assert callable(evalution.benchmarks.copa)
    assert evalution.benchmarks.DROP is not None
    assert callable(evalution.benchmarks.drop)
    assert evalution.benchmarks.HendrycksEthics is not None
    assert callable(evalution.benchmarks.ethics_cm)
    assert callable(evalution.benchmarks.ethics_deontology)
    assert callable(evalution.benchmarks.ethics_justice)
    assert callable(evalution.benchmarks.ethics_utilitarianism)
    assert callable(evalution.benchmarks.ethics_virtue)
    assert evalution.benchmarks.HEADQA is not None
    assert callable(evalution.benchmarks.headqa_en)
    assert callable(evalution.benchmarks.headqa_es)
    assert evalution.benchmarks.HellaSwag is not None
    assert callable(evalution.benchmarks.hellaswag)
    assert evalution.benchmarks.LAMBADA is not None
    assert evalution.benchmarks.LAMBADACloze is not None
    assert callable(evalution.benchmarks.lambada_openai)
    assert callable(evalution.benchmarks.lambada_openai_cloze)
    assert callable(evalution.benchmarks.lambada_standard)
    assert callable(evalution.benchmarks.lambada_standard_cloze)
    assert evalution.benchmarks.LogiQA is not None
    assert callable(evalution.benchmarks.logiqa)
    assert evalution.benchmarks.MathQA is not None
    assert callable(evalution.benchmarks.mathqa)
    assert evalution.benchmarks.MCTACO is not None
    assert callable(evalution.benchmarks.mc_taco)
    assert evalution.benchmarks.MedMCQA is not None
    assert callable(evalution.benchmarks.medmcqa)
    assert evalution.benchmarks.MedQA is not None
    assert callable(evalution.benchmarks.medqa_4options)
    assert evalution.benchmarks.MMLU is not None
    assert callable(evalution.benchmarks.mmlu)
    assert evalution.benchmarks.MMLUPro is not None
    assert callable(evalution.benchmarks.mmlu_pro)
    assert evalution.benchmarks.MNLI is not None
    assert callable(evalution.benchmarks.mnli)
    assert evalution.benchmarks.MRPC is not None
    assert callable(evalution.benchmarks.mrpc)
    assert evalution.benchmarks.NQOpen is not None
    assert callable(evalution.benchmarks.nq_open)
    assert evalution.benchmarks.OpenBookQA is not None
    assert callable(evalution.benchmarks.openbookqa)
    assert evalution.benchmarks.PAWSX is not None
    assert callable(evalution.benchmarks.paws_x)
    assert callable(evalution.benchmarks.paws_x_de)
    assert callable(evalution.benchmarks.paws_x_en)
    assert callable(evalution.benchmarks.paws_x_es)
    assert callable(evalution.benchmarks.paws_x_fr)
    assert callable(evalution.benchmarks.paws_x_ja)
    assert callable(evalution.benchmarks.paws_x_ko)
    assert callable(evalution.benchmarks.paws_x_zh)
    assert evalution.benchmarks.PIQA is not None
    assert callable(evalution.benchmarks.piqa)
    assert evalution.benchmarks.Pile10K is not None
    assert callable(evalution.benchmarks.pile_10k)
    assert evalution.benchmarks.Prost is not None
    assert callable(evalution.benchmarks.prost)
    assert evalution.benchmarks.PubMedQA is not None
    assert callable(evalution.benchmarks.pubmedqa)
    assert evalution.benchmarks.QNLI is not None
    assert callable(evalution.benchmarks.qnli)
    assert evalution.benchmarks.QQP is not None
    assert callable(evalution.benchmarks.qqp)
    assert evalution.benchmarks.RACE is not None
    assert callable(evalution.benchmarks.race)
    assert evalution.benchmarks.RollingPerplexitySample is not None
    assert evalution.benchmarks.RTE is not None
    assert callable(evalution.benchmarks.rte)
    assert evalution.benchmarks.SciQ is not None
    assert callable(evalution.benchmarks.sciq)
    assert evalution.benchmarks.SIQA is not None
    assert callable(evalution.benchmarks.siqa)
    assert evalution.benchmarks.SWAG is not None
    assert callable(evalution.benchmarks.swag)
    assert evalution.benchmarks.SST2 is not None
    assert callable(evalution.benchmarks.sst2)
    assert evalution.benchmarks.SQuADV2 is not None
    assert callable(evalution.benchmarks.squadv2)
    assert evalution.benchmarks.TriviaQA is not None
    assert callable(evalution.benchmarks.triviaqa)
    assert evalution.benchmarks.WiC is not None
    assert callable(evalution.benchmarks.wic)
    assert evalution.benchmarks.WebQS is not None
    assert callable(evalution.benchmarks.webqs)
    assert evalution.benchmarks.WikiText is not None
    assert callable(evalution.benchmarks.wikitext)
    assert evalution.benchmarks.WSC273 is not None
    assert callable(evalution.benchmarks.wsc273)
    assert evalution.benchmarks.WNLI is not None
    assert callable(evalution.benchmarks.wnli)
    assert evalution.benchmarks.XCOPA is not None
    assert callable(evalution.benchmarks.xcopa)
    assert callable(evalution.benchmarks.xcopa_et)
    assert callable(evalution.benchmarks.xcopa_ht)
    assert callable(evalution.benchmarks.xcopa_id)
    assert callable(evalution.benchmarks.xcopa_it)
    assert callable(evalution.benchmarks.xcopa_qu)
    assert callable(evalution.benchmarks.xcopa_sw)
    assert callable(evalution.benchmarks.xcopa_ta)
    assert callable(evalution.benchmarks.xcopa_th)
    assert callable(evalution.benchmarks.xcopa_tr)
    assert callable(evalution.benchmarks.xcopa_vi)
    assert callable(evalution.benchmarks.xcopa_zh)
    assert evalution.benchmarks.WinoGrande is not None
    assert callable(evalution.benchmarks.winogrande)
    assert callable(evalution.benchmarks.f1_for_label)
    assert callable(evalution.benchmarks.matthews_corrcoef)
    assert callable(evalution.benchmarks.macro_f1)


def test_package_does_not_flatten_benchmarks_into_top_level_namespace() -> None:
    assert not hasattr(evalution, "arc_challenge")
    assert not hasattr(evalution, "arc_easy")
    assert not hasattr(evalution, "EngineBuilder")
    assert not hasattr(evalution, "gsm8k")
    assert not hasattr(evalution, "mmlu")
    assert not hasattr(evalution, "ARCChallenge")
    assert not hasattr(evalution, "ARCEasy")
    assert not hasattr(evalution, "GSM8K")
    assert not hasattr(evalution, "MMLU")
    assert not hasattr(evalution, "f1_for_label")


def test_package_exports_fluent_runtime_api() -> None:
    assert callable(evalution.compare)
    assert callable(evalution.run_compare)
    assert callable(evalution.run_yaml)
    assert callable(evalution.python_from_yaml)


def test_engine_model_starts_evaluation_run() -> None:
    class DummyEngine(evalution.BaseEngine):
        def build(self, model):
            del model
            raise AssertionError("evaluation creation should not build a session")

    evaluation = DummyEngine().model({"path": "/tmp/model"})

    assert isinstance(evaluation, evalution.EvaluationRun)


def test_package_exposes_cli_entrypoint() -> None:
    from evalution import cli

    assert callable(cli.main)
