# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from evalution.scorers.classification import f1_for_label, macro_f1, matthews_corrcoef
from .anli import ANLI, anli_r1, anli_r2, anli_r3
from .arc_easy import ARCEasy, arc_easy
from .arc_challenge import ARCChallenge, arc_challenge
from .babi import BABI, babi
from .base import BaseTestSuite, TestSuite
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
from .medmcqa import MedMCQA, medmcqa
from .medqa import MedQA, medqa_4options
from .mmlu import MMLU, mmlu
from .mmlu_pro import MMLUPro, mmlu_pro
from .mnli import MNLI, mnli
from .mrpc import MRPC, mrpc
from .multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from .multiple_choice_utils import choice_index_from_labels, question_answer_prompt
from .openbookqa import OpenBookQA, openbookqa
from .piqa import PIQA, piqa
from .qnli import QNLI, qnli
from .qqp import QQP, qqp
from .rte import RTE, rte
from .sciq import SciQ, sciq
from .single_continuation import BaseSingleContinuationSuite, SingleContinuationSample
from .swag import SWAG, swag
from .sst2 import SST2, sst2
from .wic import WiC, wic
from .wnli import WNLI, wnli
from .winogrande import WinoGrande, winogrande

__all__ = [
    "ANLI",
    "ARCEasy",
    "ARCChallenge",
    "BABI",
    "BaseMultipleChoiceSuite",
    "BaseSingleContinuationSuite",
    "BaseTestSuite",
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
    "MedMCQA",
    "MedQA",
    "MMLU",
    "MMLUPro",
    "MNLI",
    "MRPC",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PIQA",
    "QNLI",
    "QQP",
    "RTE",
    "SciQ",
    "SingleContinuationSample",
    "SWAG",
    "SST2",
    "TestSuite",
    "WiC",
    "WNLI",
    "WinoGrande",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "arc_easy",
    "arc_challenge",
    "babi",
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
    "lambada_standard",
    "medmcqa",
    "medqa_4options",
    "macro_f1",
    "matthews_corrcoef",
    "mmlu",
    "mmlu_pro",
    "mnli",
    "mrpc",
    "openbookqa",
    "piqa",
    "qnli",
    "qqp",
    "rte",
    "sciq",
    "swag",
    "question_answer_prompt",
    "sst2",
    "wic",
    "wnli",
    "winogrande",
]
