from .arc_easy import ARCEasy, arc_easy
from .arc_challenge import ARCChallenge, arc_challenge
from .base import BaseTestSuite, TestSuite
from .boolq import BoolQ, boolq
from .copa import COPA, copa
from .gsm8k import GSM8K, gsm8k
from .gsm8k_platinum import GSM8KPlatinum, gsm8k_platinum
from .hellaswag import HellaSwag, hellaswag
from .mmlu import MMLU, mmlu
from .multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from .multiple_choice_utils import choice_index_from_labels, question_answer_prompt
from .openbookqa import OpenBookQA, openbookqa
from .piqa import PIQA, piqa
from .rte import RTE, rte
from .winogrande import WinoGrande, winogrande

__all__ = [
    "ARCEasy",
    "ARCChallenge",
    "BaseMultipleChoiceSuite",
    "BaseTestSuite",
    "BoolQ",
    "COPA",
    "GSM8K",
    "GSM8KPlatinum",
    "HellaSwag",
    "MMLU",
    "MultipleChoiceSample",
    "OpenBookQA",
    "PIQA",
    "RTE",
    "TestSuite",
    "WinoGrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "copa",
    "choice_index_from_labels",
    "gsm8k",
    "gsm8k_platinum",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "rte",
    "question_answer_prompt",
    "winogrande",
]
