# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from evalution.scorers.classification import f1_for_label, macro_f1, matthews_corrcoef
from .aexams import AEXAMS, AEXAMS_SUBJECTS, aexams, aexams_biology, aexams_islamic_studies, aexams_physics, aexams_science, aexams_social
from .afrixnli import AFRIXNLI_LANGUAGES, AFRIXNLI_TASKS, AfriXNLI, afrixnli, afrixnli_amh, afrixnli_eng, afrixnli_ewe, afrixnli_fra, afrixnli_hau, afrixnli_ibo, afrixnli_kin, afrixnli_lin, afrixnli_lug, afrixnli_orm, afrixnli_sna, afrixnli_sot, afrixnli_swa, afrixnli_twi, afrixnli_wol, afrixnli_xho, afrixnli_yor, afrixnli_zul
from .alghafa import COPAArabic, PIQAArabic, copa_ar, piqa_ar
from .aime import AIME, aime, aime24, aime25
from .anli import ANLI, anli_r1, anli_r2, anli_r3
from .arc_easy import ARCEasy, arc_easy
from .arc_challenge import ARCChallenge, arc_challenge
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
from .bear import BEAR, bear, bear_big
from .belebele import Belebele, belebele
from .base import BaseTestSuite, TestSuite
from .blimp import BLiMP, BLIMP_SUBSETS, blimp
from .c4 import C4, c4
from .ceval import CEVAL_SUBSETS, CEval, ceval
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
from .copal_id import COPALID, copal_id, copal_id_colloquial, copal_id_standard
from .coqa import CoQA, coqa
from .copa import COPA, copa
from .drop import DROP, drop
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
from .kobest import KOBEST_SUBSETS, KOBEST_TASKS, KoBEST, kobest, kobest_boolq, kobest_copa, kobest_hellaswag, kobest_sentineg, kobest_wic
from .lambada import LAMBADA, lambada_openai, lambada_standard
from .lambada_cloze import LAMBADACloze, lambada_openai_cloze, lambada_standard_cloze
from .logiqa import LogiQA, logiqa
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
from .triviaqa import TriviaQA, triviaqa
from .wic import WiC, wic
from .webqs import WebQS, webqs
from .wikitext import WikiText, wikitext
from .winogender import WinoGender, winogender, winogender_all, winogender_female, winogender_gotcha, winogender_gotcha_female, winogender_gotcha_male, winogender_male, winogender_neutral
from .wsc273 import WSC273, wsc273
from .wnli import WNLI, wnli
from .winogrande import WinoGrande, winogrande
from .xcopa import XCOPA, xcopa, xcopa_et, xcopa_ht, xcopa_id, xcopa_it, xcopa_qu, xcopa_sw, xcopa_ta, xcopa_th, xcopa_tr, xcopa_vi, xcopa_zh
from .xstorycloze import XSTORYCLOZE_LANGUAGES, XStoryCloze, xstorycloze, xstorycloze_ar, xstorycloze_en, xstorycloze_es, xstorycloze_eu, xstorycloze_hi, xstorycloze_id, xstorycloze_my, xstorycloze_ru, xstorycloze_sw, xstorycloze_te, xstorycloze_zh
from .xwinograd import XWinograd, xwinograd, xwinograd_en, xwinograd_fr, xwinograd_jp, xwinograd_pt, xwinograd_ru, xwinograd_zh

for _crows_pairs_task in CROWS_PAIRS_TASKS:
    globals()[_crows_pairs_task] = getattr(_crows_pairs_module, _crows_pairs_task)

del _crows_pairs_task

__all__ = [
    "ANLI",
    "AIME",
    "AEXAMS",
    "AEXAMS_SUBJECTS",
    "AFRIXNLI_LANGUAGES",
    "AFRIXNLI_TASKS",
    "AfriXNLI",
    "ARCEasy",
    "ARCChallenge",
    "COPAArabic",
    "Arithmetic",
    "ASDiv",
    "ASDivCoTLlama",
    "BABI",
    "BEAR",
    "Belebele",
    "BLiMP",
    "BLIMP_SUBSETS",
    "BaseRollingPerplexitySuite",
    "BaseMultipleChoiceSuite",
    "BaseSingleContinuationSuite",
    "BaseTestSuite",
    "C4",
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
    "KOBEST_SUBSETS",
    "KOBEST_TASKS",
    "KoBEST",
    "LAMBADA",
    "LAMBADACloze",
    "LogiQA",
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
    "TriviaQA",
    "WiC",
    "WebQS",
    "WikiText",
    "WinoGender",
    "WSC273",
    "WNLI",
    "WinoGrande",
    "XCOPA",
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
    "copa_ar",
    "arc_easy",
    "arc_challenge",
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
    "bear",
    "bear_big",
    "belebele",
    "blimp",
    "c4",
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
    "copal_id",
    "copal_id_colloquial",
    "copal_id_standard",
    "coqa",
    "copa",
    "drop",
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
    "triviaqa",
    "wic",
    "webqs",
    "wikitext",
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
    "xwinograd",
    "xwinograd_en",
    "xwinograd_fr",
    "xwinograd_jp",
    "xwinograd_pt",
    "xwinograd_ru",
    "xwinograd_zh",
]
