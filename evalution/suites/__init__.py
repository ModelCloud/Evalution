from .arc_easy import ARCEasy, arc_easy
from .arc_challenge import ARCChallenge, arc_challenge
from .base import BaseTestSuite, TestSuite
from .boolq import BoolQ, boolq
from .cb import CB, cb
from .classification_metrics import f1_for_label, macro_f1
from .copa import COPA, copa
from .gsm8k import GSM8K, gsm8k
from .gsm8k_platinum import GSM8KPlatinum, gsm8k_platinum
from .hellaswag import HellaSwag, hellaswag
from .mmlu import MMLU, mmlu
from .mrpc import MRPC, mrpc
from .multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from .multiple_choice_utils import choice_index_from_labels, question_answer_prompt
from .openbookqa import OpenBookQA, openbookqa
from .piqa import PIQA, piqa
from .qnli import QNLI, qnli
from .rte import RTE, rte
from .sst2 import SST2, sst2
from .wic import WiC, wic
from .winogrande import WinoGrande, winogrande

__all__ = [
    "ARCEasy",
    "ARCChallenge",
    "BaseMultipleChoiceSuite",
    "BaseTestSuite",
    "BoolQ",
    "CB",
    "COPA",
    "GSM8K",
    "GSM8KPlatinum",
    "HellaSwag",
    "MMLU",
    "MRPC",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PIQA",
    "QNLI",
    "RTE",
    "SST2",
    "TestSuite",
    "WiC",
    "WinoGrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "cb",
    "copa",
    "choice_index_from_labels",
    "f1_for_label",
    "gsm8k",
    "gsm8k_platinum",
    "hellaswag",
    "macro_f1",
    "mmlu",
    "mrpc",
    "openbookqa",
    "piqa",
    "qnli",
    "rte",
    "question_answer_prompt",
    "sst2",
    "wic",
    "winogrande",
]
