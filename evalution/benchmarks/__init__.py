# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from evalution.scorers.classification import f1_for_label, macro_f1, matthews_corrcoef
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
from .base import BaseTestSuite, TestSuite
from .c4 import C4, c4
from .boolq import BoolQ, boolq
from .cb import CB, cb
from .cola import CoLA, cola
from .commonsense_qa import CommonsenseQA, commonsense_qa
from .copa import COPA, copa
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
from .wsc273 import WSC273, wsc273
from .wnli import WNLI, wnli
from .winogrande import WinoGrande, winogrande

__all__ = [
    "ANLI",
    "AIME",
    "ARCEasy",
    "ARCChallenge",
    "Arithmetic",
    "ASDiv",
    "ASDivCoTLlama",
    "BABI",
    "BaseRollingPerplexitySuite",
    "BaseMultipleChoiceSuite",
    "BaseSingleContinuationSuite",
    "BaseTestSuite",
    "C4",
    "BoolQ",
    "CB",
    "CoLA",
    "CommonsenseQA",
    "COPA",
    "GSM8K",
    "GSM8KPlatinum",
    "HendrycksEthics",
    "HEADQA",
    "HellaSwag",
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
    "WSC273",
    "WNLI",
    "WinoGrande",
    "aime",
    "aime24",
    "aime25",
    "anli_r1",
    "anli_r2",
    "anli_r3",
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
    "c4",
    "boolq",
    "cb",
    "cola",
    "commonsense_qa",
    "copa",
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
    "piqa",
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
    "wsc273",
    "wnli",
    "winogrande",
]
