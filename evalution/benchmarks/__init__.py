# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from evalution.scorers.classification import f1_for_label, macro_f1, matthews_corrcoef
from .aexams import AEXAMS, AEXAMS_SUBJECTS, aexams, aexams_biology, aexams_islamic_studies, aexams_physics, aexams_science, aexams_social
from . import agieval as _agieval_module
from .agieval import AGIEVAL_SUBSETS, AGIEVAL_TASKS, AGIEval, agieval
from . import afrimgsm as _afrimgsm_module
from .afrimgsm import AFRIMGSM_LANGUAGES, AFRIMGSM_TASKS, AfriMGSM, afrimgsm
from . import afrimmlu as _afrimmlu_module
from .afrimmlu import AFRIMMLU_LANGUAGES, AFRIMMLU_TASKS, AfriMMLU, afrimmlu
from .afrixnli import AFRIXNLI_LANGUAGES, AFRIXNLI_TASKS, AfriXNLI, afrixnli, afrixnli_amh, afrixnli_eng, afrixnli_ewe, afrixnli_fra, afrixnli_hau, afrixnli_ibo, afrixnli_kin, afrixnli_lin, afrixnli_lug, afrixnli_orm, afrixnli_sna, afrixnli_sot, afrixnli_swa, afrixnli_twi, afrixnli_wol, afrixnli_xho, afrixnli_yor, afrixnli_zul
from .alghafa import COPAArabic, PIQAArabic, copa_ar, piqa_ar
from .aime import AIME, aime, aime24, aime25
from .anli import ANLI, anli_r1, anli_r2, anli_r3
from . import arabicmmlu as _arabicmmlu_module
from .arabicmmlu import ARABICMMLU_SUBSETS, ARABICMMLU_TASKS, ArabicMMLU, arabicmmlu
from .arc_easy import ARCEasy, arc_easy
from .arc_challenge import ARCChallenge, arc_challenge
from .arc_mt import ARCMT, ARC_MT_LANGUAGES, ARC_MT_TASKS, arc_mt, arc_mt_da, arc_mt_de, arc_mt_el, arc_mt_es, arc_mt_fi, arc_mt_hu, arc_mt_is, arc_mt_it, arc_mt_nb, arc_mt_pl, arc_mt_pt, arc_mt_sv
from .arithmetic import (
    Arithmetic,
    arithmetic_1dc,
    arithmetic_2da,
    arithmetic_2dm,
    arithmetic_2ds,
    arithmetic_3da,
    arithmetic_3ds,
    arithmetic_4da,
    arithmetic_4ds,
    arithmetic_5da,
    arithmetic_5ds,
)
from .asdiv import ASDiv, ASDivCoTLlama, asdiv, asdiv_cot_llama
from .babi import BABI, babi
from . import babilong as _babilong_module
from .babilong import BABILong, BABILONG_CONTEXT_LENGTHS, BABILONG_TASK_SPLITS, BABILONG_TASKS, babilong
from . import bbh as _bbh_module
from .bbh import BBH, BBH_SUBSETS, BBH_TASKS, bbh
from .bangla import BANGLA_SUBSETS, BANGLA_TASKS, Bangla, bangla, bangla_boolqa, bangla_commonsenseqa, bangla_mmlu, bangla_openbookqa, bangla_piqa
from .bear import BEAR, bear, bear_big
from .belebele import Belebele, belebele
from . import bbq as _bbq_module
from .bbq import BBQ, BBQ_CATEGORIES, BBQ_TASKS, bbq
from .base import BaseTestSuite, TestSuite
from .blimp import BLiMP, BLIMP_SUBSETS, blimp
from .c4 import C4, c4
from . import cabbq as _cabbq_module
from .cabbq import CABBQ_CATEGORIES, CABBQ_TASKS, CaBBQ, cabbq
from . import esbbq as _esbbq_module
from .esbbq import ESBBQ_CATEGORIES, ESBBQ_TASKS, EsBBQ, esbbq
from .ceval import CEVAL_SUBSETS, CEval, ceval
from .careqa import CAREQA_CONFIGS, CAREQA_TASKS, CareQA, careqa, careqa_en, careqa_es
from .boolq import BoolQ, boolq
from .cb import CB, cb
from .cola import CoLA, cola
from .cnn_dailymail import CNNDailyMail, cnn_dailymail
from .code_x_glue import (
    CODE_X_GLUE_LANGUAGES,
    CodeXGLUECodeToText,
    code_x_glue,
    code2text_go,
    code2text_java,
    code2text_javascript,
    code2text_php,
    code2text_python,
    code2text_ruby,
)
from . import crows_pairs as _crows_pairs_module
from .commonsense_qa import CommonsenseQA, commonsense_qa
from .crows_pairs import CROWS_PAIRS_BIAS_TYPES, CROWS_PAIRS_LANGUAGES, CROWS_PAIRS_TASKS, CrowSPairs, crows_pairs
from . import darijammlu as _darijammlu_module
from .darijammlu import DARIJAMMLU_SUBSETS, DARIJAMMLU_TASKS, DarijaMMLU, darijammlu
from .darijahellaswag import DarijaHellaSwag, darijahellaswag
from . import egymmlu as _egymmlu_module
from .egymmlu import EGYMMLU_SUBSETS, EGYMMLU_TASKS, EgyMMLU, egymmlu
from .egyhellaswag import EgyHellaSwag, egyhellaswag
from .copal_id import COPALID, copal_id, copal_id_colloquial, copal_id_standard
from .coqa import CoQA, coqa
from .copa import COPA, copa
from .drop import DROP, drop
from . import eus_exams as _eus_exams_module
from .eus_exams import EUS_EXAMS_SUBSETS, EUS_EXAMS_TASKS, EusExams, eus_exams
from .gpqa import GPQA, GPQA_SUBSETS, GPQA_TASKS, gpqa, gpqa_diamond, gpqa_extended, gpqa_main
from .gsm8k import GSM8K, gsm8k
from .gsm8k_platinum import GSM8KPlatinum, gsm8k_platinum
from .hendrycks_ethics import (
    HendrycksEthics,
    ethics_cm,
    ethics_deontology,
    ethics_justice,
    ethics_utilitarianism,
    ethics_virtue,
)
from .headqa import HEADQA, headqa_en, headqa_es
from .hellaswag import HellaSwag, hellaswag
from .histoires_morales import HistoiresMorales, histoires_morales
from .icelandic_winogrande import IcelandicWinoGrande, icelandic_winogrande
from . import inverse_scaling as _inverse_scaling_module
from .inverse_scaling import INVERSE_SCALING_SUBSETS, INVERSE_SCALING_TASKS, InverseScaling, inverse_scaling
from .kobest import KOBEST_SUBSETS, KOBEST_TASKS, KoBEST, kobest, kobest_boolq, kobest_copa, kobest_hellaswag, kobest_sentineg, kobest_wic
from .lambada import LAMBADA, lambada_openai, lambada_standard
from .lambada_cloze import LAMBADACloze, lambada_openai_cloze, lambada_standard_cloze
from .logiqa import LogiQA, logiqa
from .logiqa2 import LogiQA2, logiqa2
from .mathqa import MathQA, mathqa
from .medmcqa import MedMCQA, medmcqa
from .medqa import MedQA, medqa_4options
from .mc_taco import MCTACO, mc_taco
from .mmlu import MMLU, mmlu
from .mmlu_pro import MMLUPro, mmlu_pro
from .mnli import MNLI, mnli
from .mrpc import MRPC, mrpc
from .nq_open import NQOpen, nq_open
from .multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from .multiple_choice_utils import choice_index_from_labels, question_answer_prompt
from .openbookqa import OpenBookQA, openbookqa
from .paws_x import PAWSX, paws_x, paws_x_de, paws_x_en, paws_x_es, paws_x_fr, paws_x_ja, paws_x_ko, paws_x_zh
from .piqa import PIQA, piqa
from .pile_10k import Pile10K, pile_10k
from .prost import Prost, prost
from .pubmedqa import PubMedQA, pubmedqa
from .qa4mre import QA4MRE, qa4mre, qa4mre_2011, qa4mre_2012, qa4mre_2013
from .qnli import QNLI, qnli
from .qqp import QQP, qqp
from .race import RACE, race
from .rolling_perplexity import BaseRollingPerplexitySuite, RollingPerplexitySample
from .rte import RTE, rte
from .sciq import SciQ, sciq
from .siqa import SIQA, siqa
from .single_continuation import BaseSingleContinuationSuite, SingleContinuationSample
from .swag import SWAG, swag
from .sst2 import SST2, sst2
from .squadv2 import SQuADV2, squadv2
from .truthfulqa import TRUTHFULQA_TASKS, TruthfulQAMC, truthfulqa, truthfulqa_mc1, truthfulqa_mc2
from .triviaqa import TriviaQA, triviaqa
from .wic import WiC, wic
from .webqs import WebQS, webqs
from .wikitext import WikiText, wikitext
from . import wmdp as _wmdp_module
from .wmdp import WMDP, WMDP_SUBSETS, WMDP_TASKS, wmdp
from .winogender import WinoGender, winogender, winogender_all, winogender_female, winogender_gotcha, winogender_gotcha_female, winogender_gotcha_male, winogender_male, winogender_neutral
from .wsc273 import WSC273, wsc273
from .wnli import WNLI, wnli
from .winogrande import WinoGrande, winogrande
from .xcopa import XCOPA, xcopa, xcopa_et, xcopa_ht, xcopa_id, xcopa_it, xcopa_qu, xcopa_sw, xcopa_ta, xcopa_th, xcopa_tr, xcopa_vi, xcopa_zh
from . import xquad as _xquad_module
from .xquad import XQUAD_LANGUAGES, XQUAD_TASKS, XQuAD, xquad
from .xstorycloze import XSTORYCLOZE_LANGUAGES, XStoryCloze, xstorycloze, xstorycloze_ar, xstorycloze_en, xstorycloze_es, xstorycloze_eu, xstorycloze_hi, xstorycloze_id, xstorycloze_my, xstorycloze_ru, xstorycloze_sw, xstorycloze_te, xstorycloze_zh
from . import xnli as _xnli_module
from .xnli import XNLI, XNLI_LANGUAGES, XNLI_TASKS, xnli
from .xwinograd import XWinograd, xwinograd, xwinograd_en, xwinograd_fr, xwinograd_jp, xwinograd_pt, xwinograd_ru, xwinograd_zh

for _crows_pairs_task in CROWS_PAIRS_TASKS:
    globals()[_crows_pairs_task] = getattr(_crows_pairs_module, _crows_pairs_task)

del _crows_pairs_task

for _agieval_task in AGIEVAL_TASKS:
    globals()[_agieval_task] = getattr(_agieval_module, _agieval_task)

del _agieval_task

for _afrimgsm_task in AFRIMGSM_TASKS:
    globals()[_afrimgsm_task] = getattr(_afrimgsm_module, _afrimgsm_task)

del _afrimgsm_task

for _darijammlu_task in DARIJAMMLU_TASKS:
    globals()[_darijammlu_task] = getattr(_darijammlu_module, _darijammlu_task)

del _darijammlu_task

for _egymmlu_task in EGYMMLU_TASKS:
    globals()[_egymmlu_task] = getattr(_egymmlu_module, _egymmlu_task)

del _egymmlu_task

for _eus_exams_task in EUS_EXAMS_TASKS:
    globals()[_eus_exams_task] = getattr(_eus_exams_module, _eus_exams_task)

del _eus_exams_task

for _cabbq_task in CABBQ_TASKS:
    globals()[_cabbq_task] = getattr(_cabbq_module, _cabbq_task)

del _cabbq_task

for _esbbq_task in ESBBQ_TASKS:
    globals()[_esbbq_task] = getattr(_esbbq_module, _esbbq_task)

del _esbbq_task

for _afrimmlu_task in AFRIMMLU_TASKS:
    globals()[_afrimmlu_task] = getattr(_afrimmlu_module, _afrimmlu_task)

del _afrimmlu_task

for _bbh_task in BBH_TASKS:
    globals()[_bbh_task] = getattr(_bbh_module, _bbh_task)

del _bbh_task

for _bbq_task in BBQ_TASKS:
    globals()[_bbq_task] = getattr(_bbq_module, _bbq_task)

del _bbq_task

for _babilong_task in BABILONG_TASKS:
    globals()[_babilong_task] = getattr(_babilong_module, _babilong_task)

del _babilong_task

for _arabicmmlu_task in ARABICMMLU_TASKS:
    globals()[_arabicmmlu_task] = getattr(_arabicmmlu_module, _arabicmmlu_task)

del _arabicmmlu_task

for _xnli_task in XNLI_TASKS:
    globals()[_xnli_task] = getattr(_xnli_module, _xnli_task)

del _xnli_task

for _xquad_task in XQUAD_TASKS:
    globals()[_xquad_task] = getattr(_xquad_module, _xquad_task)

del _xquad_task

for _inverse_scaling_task in INVERSE_SCALING_TASKS:
    globals()[_inverse_scaling_task] = getattr(_inverse_scaling_module, _inverse_scaling_task)

del _inverse_scaling_task

for _wmdp_task in WMDP_TASKS:
    globals()[_wmdp_task] = getattr(_wmdp_module, _wmdp_task)

del _wmdp_task

__all__ = [
    "ANLI",
    "AIME",
    "AEXAMS",
    "AEXAMS_SUBJECTS",
    "AGIEVAL_SUBSETS",
    "AGIEVAL_TASKS",
    "AGIEval",
    "AFRIMGSM_LANGUAGES",
    "AFRIMGSM_TASKS",
    "AfriMGSM",
    "AFRIMMLU_LANGUAGES",
    "AFRIMMLU_TASKS",
    "AfriMMLU",
    "AFRIXNLI_LANGUAGES",
    "AFRIXNLI_TASKS",
    "AfriXNLI",
    "ARABICMMLU_SUBSETS",
    "ARABICMMLU_TASKS",
    "ARCEasy",
    "ARCChallenge",
    "ARCMT",
    "ARC_MT_LANGUAGES",
    "ARC_MT_TASKS",
    "ArabicMMLU",
    "COPAArabic",
    "Arithmetic",
    "ASDiv",
    "ASDivCoTLlama",
    "BABI",
    "BABILong",
    "BABILONG_CONTEXT_LENGTHS",
    "BABILONG_TASK_SPLITS",
    "BABILONG_TASKS",
    "BBH",
    "BBH_SUBSETS",
    "BBH_TASKS",
    "BANGLA_SUBSETS",
    "BANGLA_TASKS",
    "BEAR",
    "Bangla",
    "Belebele",
    "BBQ",
    "BBQ_CATEGORIES",
    "BBQ_TASKS",
    "BLiMP",
    "BLIMP_SUBSETS",
    "BaseRollingPerplexitySuite",
    "BaseMultipleChoiceSuite",
    "BaseSingleContinuationSuite",
    "BaseTestSuite",
    "CABBQ_CATEGORIES",
    "CABBQ_TASKS",
    "C4",
    "CaBBQ",
    "CAREQA_CONFIGS",
    "CAREQA_TASKS",
    "CareQA",
    "CEVAL_SUBSETS",
    "CEval",
    "BoolQ",
    "CB",
    "CoLA",
    "CNNDailyMail",
    "CODE_X_GLUE_LANGUAGES",
    "CodeXGLUECodeToText",
    "CommonsenseQA",
    "CROWS_PAIRS_BIAS_TYPES",
    "CROWS_PAIRS_LANGUAGES",
    "CROWS_PAIRS_TASKS",
    "CrowSPairs",
    "DarijaHellaSwag",
    "DARIJAMMLU_SUBSETS",
    "DARIJAMMLU_TASKS",
    "DarijaMMLU",
    "EGYMMLU_SUBSETS",
    "EGYMMLU_TASKS",
    "EgyHellaSwag",
    "EgyMMLU",
    "ESBBQ_CATEGORIES",
    "ESBBQ_TASKS",
    "EsBBQ",
    "esbbq",
    "EUS_EXAMS_SUBSETS",
    "EUS_EXAMS_TASKS",
    "EusExams",
    "COPALID",
    "CoQA",
    "COPA",
    "DROP",
    "GPQA",
    "GPQA_SUBSETS",
    "GPQA_TASKS",
    "GSM8K",
    "GSM8KPlatinum",
    "HendrycksEthics",
    "HEADQA",
    "HellaSwag",
    "HistoiresMorales",
    "IcelandicWinoGrande",
    "INVERSE_SCALING_SUBSETS",
    "INVERSE_SCALING_TASKS",
    "InverseScaling",
    "KOBEST_SUBSETS",
    "KOBEST_TASKS",
    "KoBEST",
    "LAMBADA",
    "LAMBADACloze",
    "LogiQA",
    "LogiQA2",
    "MathQA",
    "MedMCQA",
    "MedQA",
    "MCTACO",
    "MMLU",
    "MMLUPro",
    "MNLI",
    "MRPC",
    "NQOpen",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PAWSX",
    "PIQAArabic",
    "PIQA",
    "Pile10K",
    "Prost",
    "PubMedQA",
    "QA4MRE",
    "QNLI",
    "QQP",
    "RACE",
    "RollingPerplexitySample",
    "RTE",
    "SciQ",
    "SIQA",
    "SingleContinuationSample",
    "SWAG",
    "SQuADV2",
    "SST2",
    "TestSuite",
    "TRUTHFULQA_TASKS",
    "TruthfulQAMC",
    "TriviaQA",
    "WiC",
    "WebQS",
    "WikiText",
    "WMDP",
    "WMDP_SUBSETS",
    "WMDP_TASKS",
    "WinoGender",
    "WSC273",
    "WNLI",
    "WinoGrande",
    "XCOPA",
    "XQUAD_LANGUAGES",
    "XQUAD_TASKS",
    "XQuAD",
    "XNLI",
    "XNLI_LANGUAGES",
    "XNLI_TASKS",
    "XSTORYCLOZE_LANGUAGES",
    "XStoryCloze",
    "XWinograd",
    "aime",
    "aime24",
    "aime25",
    "aexams",
    "aexams_biology",
    "aexams_islamic_studies",
    "aexams_physics",
    "aexams_science",
    "aexams_social",
    "agieval",
    "afrimgsm",
    "afrimmlu",
    "afrixnli",
    "afrixnli_amh",
    "afrixnli_eng",
    "afrixnli_ewe",
    "afrixnli_fra",
    "afrixnli_hau",
    "afrixnli_ibo",
    "afrixnli_kin",
    "afrixnli_lin",
    "afrixnli_lug",
    "afrixnli_orm",
    "afrixnli_sna",
    "afrixnli_sot",
    "afrixnli_swa",
    "afrixnli_twi",
    "afrixnli_wol",
    "afrixnli_xho",
    "afrixnli_yor",
    "afrixnli_zul",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "arabicmmlu",
    "copa_ar",
    "arc_easy",
    "arc_challenge",
    "arc_mt",
    "arc_mt_da",
    "arc_mt_de",
    "arc_mt_el",
    "arc_mt_es",
    "arc_mt_fi",
    "arc_mt_hu",
    "arc_mt_is",
    "arc_mt_it",
    "arc_mt_nb",
    "arc_mt_pl",
    "arc_mt_pt",
    "arc_mt_sv",
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
    "asdiv",
    "asdiv_cot_llama",
    "babi",
    "babilong",
    "bbh",
    "bangla",
    "bangla_boolqa",
    "bangla_commonsenseqa",
    "bangla_mmlu",
    "bangla_openbookqa",
    "bangla_piqa",
    "bear",
    "bear_big",
    "belebele",
    "bbq",
    "blimp",
    "cabbq",
    "c4",
    "careqa",
    "careqa_en",
    "careqa_es",
    "ceval",
    "boolq",
    "cb",
    "cola",
    "cnn_dailymail",
    "code_x_glue",
    "code2text_go",
    "code2text_java",
    "code2text_javascript",
    "code2text_php",
    "code2text_python",
    "code2text_ruby",
    "commonsense_qa",
    "crows_pairs",
    *CROWS_PAIRS_TASKS,
    "darijahellaswag",
    "darijammlu",
    "egymmlu",
    "egyhellaswag",
    "copal_id",
    "copal_id_colloquial",
    "copal_id_standard",
    "coqa",
    "copa",
    "drop",
    "eus_exams",
    "gpqa",
    "gpqa_diamond",
    "gpqa_extended",
    "gpqa_main",
    "ethics_cm",
    "ethics_deontology",
    "ethics_justice",
    "ethics_utilitarianism",
    "ethics_virtue",
    "choice_index_from_labels",
    "f1_for_label",
    "gsm8k",
    "gsm8k_platinum",
    "headqa_en",
    "headqa_es",
    "hellaswag",
    "histoires_morales",
    "icelandic_winogrande",
    "inverse_scaling",
    "kobest",
    "kobest_boolq",
    "kobest_copa",
    "kobest_hellaswag",
    "kobest_sentineg",
    "kobest_wic",
    "lambada_openai",
    "lambada_openai_cloze",
    "lambada_standard",
    "lambada_standard_cloze",
    "logiqa",
    "logiqa2",
    "mathqa",
    "medmcqa",
    "medqa_4options",
    "mc_taco",
    "macro_f1",
    "matthews_corrcoef",
    "mmlu",
    "mmlu_pro",
    "mnli",
    "mrpc",
    "nq_open",
    "openbookqa",
    "paws_x",
    "paws_x_de",
    "paws_x_en",
    "paws_x_es",
    "paws_x_fr",
    "paws_x_ja",
    "paws_x_ko",
    "paws_x_zh",
    "piqa",
    "piqa_ar",
    "pile_10k",
    "prost",
    "pubmedqa",
    "qa4mre",
    "qa4mre_2011",
    "qa4mre_2012",
    "qa4mre_2013",
    "qnli",
    "qqp",
    "race",
    "rte",
    "sciq",
    "siqa",
    "swag",
    "squadv2",
    "question_answer_prompt",
    "sst2",
    "truthfulqa",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "triviaqa",
    "wic",
    "webqs",
    "wikitext",
    "wmdp",
    "winogender",
    "winogender_all",
    "winogender_female",
    "winogender_gotcha",
    "winogender_gotcha_female",
    "winogender_gotcha_male",
    "winogender_male",
    "winogender_neutral",
    "wsc273",
    "wnli",
    "winogrande",
    "xcopa",
    "xcopa_et",
    "xcopa_ht",
    "xcopa_id",
    "xcopa_it",
    "xcopa_qu",
    "xcopa_sw",
    "xcopa_ta",
    "xcopa_th",
    "xcopa_tr",
    "xcopa_vi",
    "xcopa_zh",
    "xquad",
    "xstorycloze",
    "xstorycloze_ar",
    "xstorycloze_en",
    "xstorycloze_es",
    "xstorycloze_eu",
    "xstorycloze_hi",
    "xstorycloze_id",
    "xstorycloze_my",
    "xstorycloze_ru",
    "xstorycloze_sw",
    "xstorycloze_te",
    "xstorycloze_zh",
    "xnli",
    "xwinograd",
    "xwinograd_en",
    "xwinograd_fr",
    "xwinograd_jp",
    "xwinograd_pt",
    "xwinograd_ru",
    "xwinograd_zh",
]

__all__.extend(BBH_TASKS)
__all__.extend(BABILONG_TASKS)
__all__.extend(AGIEVAL_TASKS)
__all__.extend(AFRIMGSM_TASKS)
__all__.extend(DARIJAMMLU_TASKS)
__all__.extend(EGYMMLU_TASKS)
__all__.extend(EUS_EXAMS_TASKS)
__all__.extend(CAREQA_TASKS)
__all__.extend(CABBQ_TASKS)
__all__.extend(BBQ_TASKS)
__all__.extend(AFRIMMLU_TASKS)
__all__.extend(ARABICMMLU_TASKS)
__all__.extend(WMDP_TASKS)
__all__.extend(XNLI_TASKS)
__all__.extend(XQUAD_TASKS)
__all__.extend(TRUTHFULQA_TASKS)
__all__.extend(INVERSE_SCALING_TASKS)
