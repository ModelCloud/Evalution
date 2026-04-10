# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS
# Resolve subprocess nodeids from the repository root so child pytest runs stay location-agnostic.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBPROCESS_SUITE_ENV = "EVALUTION_LLAMA3_2_SINGLE_SUITE_CHILD"
# Free-threading + upstream paged-attention teardown can abort the whole worker after one CUDA
# fault, so isolate each parametrized suite in its own pytest subprocess in the parent run.
_USE_SUBPROCESS_SUITE_ISOLATION = (
    os.environ.get(_SUBPROCESS_SUITE_ENV) != "1"
    and callable(getattr(sys, "_is_gil_enabled", None))
    and not sys._is_gil_enabled()
)

# Keep these suite keys identical to the standalone single-suite benchmark coverage.
SINGLE_SUITE_SPECS = (
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "arc_challenge_label_perm_0_25",
    "arc_challenge",
    "arc_easy",
    "assin_entailment",
    "assin_paraphrase",
    "asdiv_cot_llama",
    "asdiv",
    "babi",
    "boolq",
    "c4",
    "cb",
    "click_cul_economy",
    "click_cul_geography",
    "click_cul_history",
    "click_cul_kpop",
    "click_cul_law",
    "click_cul",
    "click_cul_politics",
    "click_cul_society",
    "click_cul_tradition",
    "click_lang_function",
    "click_lang_grammar",
    "click_lang",
    "click_lang_text",
    "click",
    "cnn_dailymail",
    "cocoteros_es",
    "cola",
    "commonsense_qa",
    "copa",
    "copa_es",
    "coqa",
    "darijahellaswag",
    "drop",
    "escola",
    "egyhellaswag",
    "ethics_cm",
    "ethics_deontology",
    "ethics_justice",
    "ethics_utilitarianism",
    "ethics_virtue",
    "eus_proficiency",
    "eus_reading",
    "eus_trivia",
    "fda",
    "fld",
    "flores_es_en_es",
    "flores_pt_en_pt",
    "french_bench_arc_challenge",
    "graphwalks_128k",
    "groundcocoa",
    "gsm8k",
    "gsm8k_platinum",
    "gsm_plus",
    "gsm_plus_mini",
    "haerae_general_knowledge",
    "haerae_history",
    "haerae",
    "haerae_loan_word",
    "haerae_rare_word",
    "haerae_standard_nomenclature",
    "headqa_en",
    "headqa_es",
    "hellaswag_label_perm_0_25",
    "hellaswag",
    "histoires_morales",
    "humaneval",
    "icelandic_winogrande",
    "ifeval",
    "kormedmcqa_dentist",
    "kormedmcqa_doctor",
    "kormedmcqa",
    "kormedmcqa_nurse",
    "kormedmcqa_pharm",
    "lambada_openai_cloze",
    "lambada_openai",
    "lambada_openai_mt_de",
    "lambada_openai_mt_en",
    "lambada_openai_mt_es",
    "lambada_openai_mt_fr",
    "lambada_openai_mt_it",
    "lambada_openai_mt_stablelm_de",
    "lambada_openai_mt_stablelm_en",
    "lambada_openai_mt_stablelm_es",
    "lambada_openai_mt_stablelm_fr",
    "lambada_openai_mt_stablelm_it",
    "lambada_openai_mt_stablelm_nl",
    "lambada_openai_mt_stablelm_pt",
    "lambada_standard_cloze",
    "lambada_standard",
    "logiqa",
    "longbench_trec",
    "mastermind_24_easy",
    "mastermind_24_hard",
    "mastermind_35_easy",
    "mastermind_35_hard",
    "mastermind_46_easy",
    "mastermind_46_hard",
    "mathqa",
    "mbpp",
    "mc_taco",
    "medmcqa",
    "medqa_4options",
    "mediqa_qa2019",
    "meqsum",
    "mgsm_direct_es_spanish_bench",
    "mmlu_pro_plus_stem_math",
    "mmlu_redux_stem_abstract_algebra",
    "mnli",
    "moral_stories",
    "mrpc",
    "niah_single_1",
    "nq_open",
    "noticia",
    "openbookqa_label_perm_0_25",
    "openbookqa",
    "pile_10k",
    "piqa",
    "polemo2_in",
    "polemo2_out",
    "phrases_es_va",
    "phrases_va_es",
    "prost",
    "pubmedqa",
    "qnli",
    "qqp",
    "race",
    "record",
    "rte",
    "sciq",
    "siqa",
    "simple_cooccurrence_bias",
    "swde",
    "squadv2",
    "squad_completion",
    "sst2",
    "swag",
    "toxigen",
    "triviaqa",
    "webqs",
    "wic",
    "wikitext",
    "winogrande",
    "wnli",
    "wsc273",
    "wsc",
    "xlsum_es",
    "xnli_eu",
)


def _run_suite_spec_in_subprocess(suite_key: str) -> None:
    """Run one parametrized suite in an isolated pytest child process."""

    nodeid = (
        f"{Path(__file__).resolve()}::"
        "test_llama3_2_transformers_single_suite_spec_full_model_eval"
        f"[{suite_key}]"
    )
    env = dict(os.environ)
    env[_SUBPROCESS_SUITE_ENV] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", nodeid],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    raise AssertionError(
        f"subprocess suite {suite_key!r} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("suite_key", SINGLE_SUITE_SPECS, ids=SINGLE_SUITE_SPECS)
def test_llama3_2_transformers_single_suite_spec_full_model_eval(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> None:
    """Exercise each single-suite spec against the full Llama 3.2 transformer path."""

    if _USE_SUBPROCESS_SUITE_ISOLATION:
        _run_suite_spec_in_subprocess(suite_key)
        return
    run_suite_spec(capsys, suite_key)
