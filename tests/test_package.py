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
    assert evalution.OpenVINO is not None
    assert evalution.TensorRTLLM is not None
    assert evalution.Transformers is not None
    assert evalution.TransformersCompat is not None
    assert evalution.VLLM is not None
    assert evalution.engines.GPTQModel is not None
    assert evalution.engines.OpenVINO is not None
    assert evalution.engines.TensorRTLLM is not None
    assert evalution.engines.Transformers is not None
    assert evalution.engines.TransformersCompat is not None
    assert evalution.engines.VLLM is not None


def test_package_exports_benchmarks_namespace() -> None:
    assert evalution.benchmarks.AEXAMS is not None
    assert evalution.benchmarks.AEXAMS_SUBJECTS
    assert evalution.benchmarks.AfriXNLI is not None
    assert evalution.benchmarks.AFRIXNLI_LANGUAGES
    assert evalution.benchmarks.AFRIXNLI_TASKS
    assert evalution.benchmarks.AIME is not None
    assert callable(evalution.benchmarks.aexams)
    assert callable(evalution.benchmarks.aexams_biology)
    assert callable(evalution.benchmarks.aexams_islamic_studies)
    assert callable(evalution.benchmarks.aexams_physics)
    assert callable(evalution.benchmarks.aexams_science)
    assert callable(evalution.benchmarks.aexams_social)
    assert evalution.benchmarks.AGIEval is not None
    assert evalution.benchmarks.AGIEVAL_SUBSETS
    assert evalution.benchmarks.AGIEVAL_TASKS
    assert callable(evalution.benchmarks.agieval)
    for factory_name in evalution.benchmarks.AGIEVAL_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.AfriMGSM is not None
    assert evalution.benchmarks.AFRIMGSM_LANGUAGES
    assert evalution.benchmarks.AFRIMGSM_TASKS
    assert callable(evalution.benchmarks.afrimgsm)
    for factory_name in evalution.benchmarks.AFRIMGSM_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.AfriMMLU is not None
    assert evalution.benchmarks.AFRIMMLU_LANGUAGES
    assert evalution.benchmarks.AFRIMMLU_TASKS
    assert callable(evalution.benchmarks.afrimmlu)
    for factory_name in evalution.benchmarks.AFRIMMLU_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert callable(evalution.benchmarks.afrixnli)
    assert callable(evalution.benchmarks.afrixnli_amh)
    assert callable(evalution.benchmarks.afrixnli_eng)
    assert callable(evalution.benchmarks.afrixnli_ewe)
    assert callable(evalution.benchmarks.afrixnli_fra)
    assert callable(evalution.benchmarks.afrixnli_hau)
    assert callable(evalution.benchmarks.afrixnli_ibo)
    assert callable(evalution.benchmarks.afrixnli_kin)
    assert callable(evalution.benchmarks.afrixnli_lin)
    assert callable(evalution.benchmarks.afrixnli_lug)
    assert callable(evalution.benchmarks.afrixnli_orm)
    assert callable(evalution.benchmarks.afrixnli_sna)
    assert callable(evalution.benchmarks.afrixnli_sot)
    assert callable(evalution.benchmarks.afrixnli_swa)
    assert callable(evalution.benchmarks.afrixnli_twi)
    assert callable(evalution.benchmarks.afrixnli_wol)
    assert callable(evalution.benchmarks.afrixnli_xho)
    assert callable(evalution.benchmarks.afrixnli_yor)
    assert callable(evalution.benchmarks.afrixnli_zul)
    assert callable(evalution.benchmarks.aime)
    assert callable(evalution.benchmarks.aime24)
    assert callable(evalution.benchmarks.aime25)
    assert callable(evalution.benchmarks.aime26)
    assert evalution.benchmarks.ANLI is not None
    assert callable(evalution.benchmarks.anli_r1)
    assert callable(evalution.benchmarks.anli_r2)
    assert callable(evalution.benchmarks.anli_r3)
    assert evalution.benchmarks.ArabicMMLU is not None
    assert evalution.benchmarks.ARABICMMLU_SUBSETS
    assert evalution.benchmarks.ARABICMMLU_TASKS
    assert callable(evalution.benchmarks.arabicmmlu)
    for factory_name in evalution.benchmarks.ARABICMMLU_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.DarijaMMLU is not None
    assert evalution.benchmarks.DARIJAMMLU_SUBSETS
    assert evalution.benchmarks.DARIJAMMLU_TASKS
    assert callable(evalution.benchmarks.darijammlu)
    for factory_name in evalution.benchmarks.DARIJAMMLU_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.EgyMMLU is not None
    assert evalution.benchmarks.EGYMMLU_SUBSETS
    assert evalution.benchmarks.EGYMMLU_TASKS
    assert callable(evalution.benchmarks.egymmlu)
    for factory_name in evalution.benchmarks.EGYMMLU_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.EusExams is not None
    assert evalution.benchmarks.EUS_EXAMS_SUBSETS
    assert evalution.benchmarks.EUS_EXAMS_TASKS
    assert callable(evalution.benchmarks.eus_exams)
    for factory_name in evalution.benchmarks.EUS_EXAMS_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.CareQA is not None
    assert evalution.benchmarks.CAREQA_CONFIGS
    assert evalution.benchmarks.CAREQA_TASKS
    assert callable(evalution.benchmarks.careqa)
    assert callable(evalution.benchmarks.careqa_en)
    assert callable(evalution.benchmarks.careqa_es)
    assert evalution.benchmarks.CaBBQ is not None
    assert evalution.benchmarks.CABBQ_CATEGORIES
    assert evalution.benchmarks.CABBQ_TASKS
    assert callable(evalution.benchmarks.cabbq)
    for factory_name in evalution.benchmarks.CABBQ_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.EsBBQ is not None
    assert evalution.benchmarks.ESBBQ_CATEGORIES
    assert evalution.benchmarks.ESBBQ_TASKS
    assert callable(evalution.benchmarks.esbbq)
    for factory_name in evalution.benchmarks.ESBBQ_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.ARCChallenge is not None
    assert evalution.benchmarks.ARCMT is not None
    assert evalution.benchmarks.ARC_MT_LANGUAGES
    assert evalution.benchmarks.ARC_MT_TASKS
    assert callable(evalution.benchmarks.arc_challenge)
    assert evalution.benchmarks.ARCEasy is not None
    assert callable(evalution.benchmarks.arc_easy)
    assert callable(evalution.benchmarks.arc_mt)
    for factory_name in evalution.benchmarks.ARC_MT_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
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
    assert evalution.benchmarks.BABILong is not None
    assert evalution.benchmarks.BABILONG_CONTEXT_LENGTHS
    assert evalution.benchmarks.BABILONG_TASK_SPLITS
    assert evalution.benchmarks.BABILONG_TASKS
    assert callable(evalution.benchmarks.babilong)
    for factory_name in evalution.benchmarks.BABILONG_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.BBH is not None
    assert evalution.benchmarks.BBH_SUBSETS
    assert evalution.benchmarks.BBH_TASKS
    assert callable(evalution.benchmarks.bbh)
    for factory_name in evalution.benchmarks.BBH_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.Bangla is not None
    assert evalution.benchmarks.BANGLA_SUBSETS
    assert evalution.benchmarks.BANGLA_TASKS
    assert callable(evalution.benchmarks.bangla)
    assert callable(evalution.benchmarks.bangla_boolqa)
    assert callable(evalution.benchmarks.bangla_commonsenseqa)
    assert callable(evalution.benchmarks.bangla_mmlu)
    assert callable(evalution.benchmarks.bangla_openbookqa)
    assert callable(evalution.benchmarks.bangla_piqa)
    assert evalution.benchmarks.BEAR is not None
    assert callable(evalution.benchmarks.bear)
    assert callable(evalution.benchmarks.bear_big)
    assert evalution.benchmarks.Belebele is not None
    assert callable(evalution.benchmarks.belebele)
    assert evalution.benchmarks.BBQ is not None
    assert evalution.benchmarks.BBQ_CATEGORIES
    assert evalution.benchmarks.BBQ_TASKS
    assert callable(evalution.benchmarks.bbq)
    for factory_name in evalution.benchmarks.BBQ_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
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
    assert evalution.benchmarks.Click is not None
    assert evalution.benchmarks.CLICK_LANG_SUBSETS == ("text", "grammar", "function")
    assert evalution.benchmarks.CLICK_CUL_SUBSETS == ("economy", "geography", "history", "kpop", "law", "politics", "society", "tradition")
    assert evalution.benchmarks.CLICK_TASKS
    for factory_name in evalution.benchmarks.CLICK_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.CoLA is not None
    assert callable(evalution.benchmarks.cola)
    assert evalution.benchmarks.CNNDailyMail is not None
    assert callable(evalution.benchmarks.cnn_dailymail)
    assert evalution.benchmarks.CMMLU is not None
    assert evalution.benchmarks.CMMLU_SUBSETS
    assert evalution.benchmarks.CMMLU_TASKS
    assert callable(evalution.benchmarks.cmmlu)
    for factory_name in evalution.benchmarks.CMMLU_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.CodeXGLUECodeToText is not None
    assert evalution.benchmarks.CODE_X_GLUE_LANGUAGES
    assert callable(evalution.benchmarks.code_x_glue)
    assert callable(evalution.benchmarks.code2text_go)
    assert callable(evalution.benchmarks.code2text_java)
    assert callable(evalution.benchmarks.code2text_javascript)
    assert callable(evalution.benchmarks.code2text_php)
    assert callable(evalution.benchmarks.code2text_python)
    assert callable(evalution.benchmarks.code2text_ruby)
    assert evalution.benchmarks.CommonsenseQA is not None
    assert callable(evalution.benchmarks.commonsense_qa)
    assert evalution.benchmarks.CrowSPairs is not None
    assert evalution.benchmarks.CROWS_PAIRS_BIAS_TYPES
    assert evalution.benchmarks.CROWS_PAIRS_LANGUAGES
    assert evalution.benchmarks.CROWS_PAIRS_TASKS
    assert callable(evalution.benchmarks.crows_pairs)
    for factory_name in evalution.benchmarks.CROWS_PAIRS_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.DarijaHellaSwag is not None
    assert callable(evalution.benchmarks.darijahellaswag)
    assert evalution.benchmarks.EgyHellaSwag is not None
    assert callable(evalution.benchmarks.egyhellaswag)
    assert evalution.benchmarks.COPAArabic is not None
    assert evalution.benchmarks.COPALID is not None
    assert callable(evalution.benchmarks.copal_id)
    assert callable(evalution.benchmarks.copal_id_standard)
    assert callable(evalution.benchmarks.copal_id_colloquial)
    assert evalution.benchmarks.CoQA is not None
    assert callable(evalution.benchmarks.coqa)
    assert evalution.benchmarks.COPA is not None
    assert callable(evalution.benchmarks.copa)
    assert callable(evalution.benchmarks.copa_ar)
    assert evalution.benchmarks.DROP is not None
    assert callable(evalution.benchmarks.drop)
    assert evalution.benchmarks.FLD is not None
    assert evalution.benchmarks.FLD_LABELS == ("PROVED", "DISPROVED", "UNKNOWN")
    assert callable(evalution.benchmarks.fld)
    assert evalution.benchmarks.FDA is not None
    assert callable(evalution.benchmarks.fda)
    assert evalution.benchmarks.FrenchBenchARCChallenge is not None
    assert callable(evalution.benchmarks.french_bench_arc_challenge)
    assert evalution.benchmarks.EusReading is not None
    assert callable(evalution.benchmarks.eus_reading)
    assert evalution.benchmarks.EusProficiency is not None
    assert callable(evalution.benchmarks.eus_proficiency)
    assert evalution.benchmarks.EusTrivia is not None
    assert callable(evalution.benchmarks.eus_trivia)
    assert evalution.benchmarks.GPQA is not None
    assert evalution.benchmarks.GPQA_SUBSETS
    assert evalution.benchmarks.GPQA_TASKS
    assert callable(evalution.benchmarks.gpqa)
    assert callable(evalution.benchmarks.gpqa_main)
    assert callable(evalution.benchmarks.gpqa_diamond)
    assert callable(evalution.benchmarks.gpqa_extended)
    assert evalution.benchmarks.GSM_PLUS_TASKS == ("gsm_plus", "gsm_plus_mini")
    assert evalution.benchmarks.GSMPlus is not None
    assert evalution.benchmarks.GSMPlusMini is not None
    assert callable(evalution.benchmarks.gsm_plus)
    assert callable(evalution.benchmarks.gsm_plus_mini)
    assert evalution.benchmarks.HAERAE_SUBSETS
    assert evalution.benchmarks.HAERAE_TASKS
    assert evalution.benchmarks.Haerae is not None
    assert callable(evalution.benchmarks.haerae)
    for factory_name in evalution.benchmarks.HAERAE_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.HendrycksEthics is not None
    assert evalution.benchmarks.HendrycksMath is not None
    assert evalution.benchmarks.HENDRYCKS_MATH_SUBSETS
    assert evalution.benchmarks.HENDRYCKS_MATH_TASKS
    assert callable(evalution.benchmarks.hendrycks_math)
    for factory_name in evalution.benchmarks.HENDRYCKS_MATH_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
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
    assert evalution.benchmarks.HistoiresMorales is not None
    assert callable(evalution.benchmarks.histoires_morales)
    assert evalution.benchmarks.MoralStories is not None
    assert callable(evalution.benchmarks.moral_stories)
    assert evalution.benchmarks.IFEval is not None
    assert callable(evalution.benchmarks.ifeval)
    assert evalution.benchmarks.IcelandicWinoGrande is not None
    assert callable(evalution.benchmarks.icelandic_winogrande)
    assert evalution.benchmarks.InverseScaling is not None
    assert evalution.benchmarks.INVERSE_SCALING_SUBSETS
    assert evalution.benchmarks.INVERSE_SCALING_TASKS
    assert callable(evalution.benchmarks.inverse_scaling)
    for factory_name in evalution.benchmarks.INVERSE_SCALING_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.KoBEST is not None
    assert evalution.benchmarks.KOBEST_SUBSETS
    assert evalution.benchmarks.KOBEST_TASKS
    assert callable(evalution.benchmarks.kobest)
    assert callable(evalution.benchmarks.kobest_boolq)
    assert callable(evalution.benchmarks.kobest_copa)
    assert callable(evalution.benchmarks.kobest_hellaswag)
    assert callable(evalution.benchmarks.kobest_sentineg)
    assert callable(evalution.benchmarks.kobest_wic)
    assert evalution.benchmarks.KORMEDMCQA_SUBSETS
    assert evalution.benchmarks.KORMEDMCQA_TASKS
    assert evalution.benchmarks.KorMedMCQA is not None
    assert callable(evalution.benchmarks.kormedmcqa)
    for factory_name in evalution.benchmarks.KORMEDMCQA_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.LAMBADA is not None
    assert evalution.benchmarks.LAMBADACloze is not None
    assert callable(evalution.benchmarks.lambada_openai)
    assert evalution.benchmarks.LAMBADA_OPENAI_MT_LANGUAGES == ("de", "en", "es", "fr", "it")
    assert evalution.benchmarks.LAMBADA_OPENAI_MT_TASKS == (
        "lambada_openai_mt_de",
        "lambada_openai_mt_en",
        "lambada_openai_mt_es",
        "lambada_openai_mt_fr",
        "lambada_openai_mt_it",
    )
    assert evalution.benchmarks.LAMBADA_OPENAI_MT_STABLELM_LANGUAGES == (
        "de",
        "en",
        "es",
        "fr",
        "it",
        "nl",
        "pt",
    )
    assert evalution.benchmarks.LAMBADA_OPENAI_MT_STABLELM_TASKS == (
        "lambada_openai_mt_stablelm_de",
        "lambada_openai_mt_stablelm_en",
        "lambada_openai_mt_stablelm_es",
        "lambada_openai_mt_stablelm_fr",
        "lambada_openai_mt_stablelm_it",
        "lambada_openai_mt_stablelm_nl",
        "lambada_openai_mt_stablelm_pt",
    )
    assert callable(evalution.benchmarks.lambada_openai_mt)
    assert callable(evalution.benchmarks.lambada_openai_mt_de)
    assert callable(evalution.benchmarks.lambada_openai_mt_en)
    assert callable(evalution.benchmarks.lambada_openai_mt_es)
    assert callable(evalution.benchmarks.lambada_openai_mt_fr)
    assert callable(evalution.benchmarks.lambada_openai_mt_it)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_de)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_en)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_es)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_fr)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_it)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_nl)
    assert callable(evalution.benchmarks.lambada_openai_mt_stablelm_pt)
    assert callable(evalution.benchmarks.lambada_openai_cloze)
    assert callable(evalution.benchmarks.lambada_standard)
    assert callable(evalution.benchmarks.lambada_standard_cloze)
    assert evalution.benchmarks.LogiQA is not None
    assert callable(evalution.benchmarks.logiqa)
    assert evalution.benchmarks.LogiQA2 is not None
    assert callable(evalution.benchmarks.logiqa2)
    assert evalution.benchmarks.HumanEval is not None
    assert callable(evalution.benchmarks.humaneval)
    assert evalution.benchmarks.MBPP is not None
    assert callable(evalution.benchmarks.mbpp)
    assert evalution.benchmarks.MathQA is not None
    assert callable(evalution.benchmarks.mathqa)
    assert evalution.benchmarks.Mastermind is not None
    assert evalution.benchmarks.MASTERMIND_VARIANTS == (
        "mastermind_24_easy",
        "mastermind_24_hard",
        "mastermind_35_easy",
        "mastermind_35_hard",
        "mastermind_46_easy",
        "mastermind_46_hard",
    )
    assert callable(evalution.benchmarks.mastermind)
    assert callable(evalution.benchmarks.mastermind_24_easy)
    assert callable(evalution.benchmarks.mastermind_24_hard)
    assert callable(evalution.benchmarks.mastermind_35_easy)
    assert callable(evalution.benchmarks.mastermind_35_hard)
    assert callable(evalution.benchmarks.mastermind_46_easy)
    assert callable(evalution.benchmarks.mastermind_46_hard)
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
    assert evalution.benchmarks.MultiRC is not None
    assert callable(evalution.benchmarks.multirc)
    assert evalution.benchmarks.MuTual is not None
    assert callable(evalution.benchmarks.mutual)
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
    assert evalution.benchmarks.Polemo2 is not None
    assert evalution.benchmarks.POLEMO2_VARIANTS == ("polemo2_in", "polemo2_out")
    assert callable(evalution.benchmarks.polemo2)
    assert callable(evalution.benchmarks.polemo2_in)
    assert callable(evalution.benchmarks.polemo2_out)
    assert evalution.benchmarks.PIQAArabic is not None
    assert evalution.benchmarks.PIQA is not None
    assert callable(evalution.benchmarks.piqa)
    assert callable(evalution.benchmarks.piqa_ar)
    assert evalution.benchmarks.Pile10K is not None
    assert callable(evalution.benchmarks.pile_10k)
    assert evalution.benchmarks.Prost is not None
    assert callable(evalution.benchmarks.prost)
    assert evalution.benchmarks.PubMedQA is not None
    assert callable(evalution.benchmarks.pubmedqa)
    assert evalution.benchmarks.QA4MRE is not None
    assert callable(evalution.benchmarks.qa4mre)
    assert callable(evalution.benchmarks.qa4mre_2011)
    assert callable(evalution.benchmarks.qa4mre_2012)
    assert callable(evalution.benchmarks.qa4mre_2013)
    assert evalution.benchmarks.QNLI is not None
    assert callable(evalution.benchmarks.qnli)
    assert evalution.benchmarks.QQP is not None
    assert callable(evalution.benchmarks.qqp)
    assert evalution.benchmarks.RACE is not None
    assert callable(evalution.benchmarks.race)
    assert evalution.benchmarks.ReCoRD is not None
    assert callable(evalution.benchmarks.record)
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
    assert evalution.benchmarks.ToxiGen is not None
    assert callable(evalution.benchmarks.toxigen)
    assert evalution.benchmarks.TruthfulQAMC is not None
    assert evalution.benchmarks.TRUTHFULQA_TASKS == ("truthfulqa_mc1", "truthfulqa_mc2")
    assert callable(evalution.benchmarks.truthfulqa)
    assert callable(evalution.benchmarks.truthfulqa_mc1)
    assert callable(evalution.benchmarks.truthfulqa_mc2)
    assert evalution.benchmarks.TriviaQA is not None
    assert callable(evalution.benchmarks.triviaqa)
    assert evalution.benchmarks.WiC is not None
    assert callable(evalution.benchmarks.wic)
    assert evalution.benchmarks.WebQS is not None
    assert callable(evalution.benchmarks.webqs)
    assert evalution.benchmarks.WikiText is not None
    assert callable(evalution.benchmarks.wikitext)
    assert evalution.benchmarks.WMDP is not None
    assert evalution.benchmarks.WMDP_SUBSETS
    assert evalution.benchmarks.WMDP_TASKS
    assert callable(evalution.benchmarks.wmdp)
    for factory_name in evalution.benchmarks.WMDP_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.WinoGender is not None
    assert callable(evalution.benchmarks.winogender)
    assert callable(evalution.benchmarks.winogender_all)
    assert callable(evalution.benchmarks.winogender_female)
    assert callable(evalution.benchmarks.winogender_gotcha)
    assert callable(evalution.benchmarks.winogender_gotcha_female)
    assert callable(evalution.benchmarks.winogender_gotcha_male)
    assert callable(evalution.benchmarks.winogender_male)
    assert callable(evalution.benchmarks.winogender_neutral)
    assert evalution.benchmarks.WSC is not None
    assert callable(evalution.benchmarks.wsc)
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
    assert evalution.benchmarks.XQuAD is not None
    assert evalution.benchmarks.XQUAD_LANGUAGES
    assert evalution.benchmarks.XQUAD_TASKS
    assert callable(evalution.benchmarks.xquad)
    for factory_name in evalution.benchmarks.XQUAD_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.XStoryCloze is not None
    assert evalution.benchmarks.XSTORYCLOZE_LANGUAGES
    assert callable(evalution.benchmarks.xstorycloze)
    assert callable(evalution.benchmarks.xstorycloze_ar)
    assert callable(evalution.benchmarks.xstorycloze_en)
    assert callable(evalution.benchmarks.xstorycloze_es)
    assert callable(evalution.benchmarks.xstorycloze_eu)
    assert callable(evalution.benchmarks.xstorycloze_hi)
    assert callable(evalution.benchmarks.xstorycloze_id)
    assert callable(evalution.benchmarks.xstorycloze_my)
    assert callable(evalution.benchmarks.xstorycloze_ru)
    assert callable(evalution.benchmarks.xstorycloze_sw)
    assert callable(evalution.benchmarks.xstorycloze_te)
    assert callable(evalution.benchmarks.xstorycloze_zh)
    assert evalution.benchmarks.XNLI is not None
    assert evalution.benchmarks.XNLI_LANGUAGES
    assert evalution.benchmarks.XNLI_TASKS
    assert evalution.benchmarks.XNLIEU is not None
    assert callable(evalution.benchmarks.xnli)
    assert callable(evalution.benchmarks.xnli_eu)
    for factory_name in evalution.benchmarks.XNLI_TASKS:
        assert callable(getattr(evalution.benchmarks, factory_name))
    assert evalution.benchmarks.XWinograd is not None
    assert callable(evalution.benchmarks.xwinograd)
    assert callable(evalution.benchmarks.xwinograd_en)
    assert callable(evalution.benchmarks.xwinograd_fr)
    assert callable(evalution.benchmarks.xwinograd_jp)
    assert callable(evalution.benchmarks.xwinograd_pt)
    assert callable(evalution.benchmarks.xwinograd_ru)
    assert callable(evalution.benchmarks.xwinograd_zh)
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

    evaluation = DummyEngine().model(path="/tmp/model")

    assert isinstance(evaluation, evalution.EvaluationRun)


def test_package_exposes_cli_entrypoint() -> None:
    from evalution import cli

    assert callable(cli.main)
