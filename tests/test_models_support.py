from __future__ import annotations

from pathlib import Path

import pcre

from tests.models_support import MIN_REALISTIC_BASELINE_SCORE
from tests.models_support import SUITE_SPECS
from tests.models_support import find_suspicious_low_baselines

# Compile the benchmark-factory pattern once so the audit stays on the repo's PCRE2 backend.
_EVALUTION_BENCHMARK_FACTORY_PATTERN = pcre.compile(r"evalution\.benchmarks\.([a-zA-Z0-9_]+)\s*\(")

# Family dispatchers, alias factories, and helper constructors are covered by representative
# concrete full-model suites in `SUITE_SPECS`, so the audit should not flag them as missing.
_NON_CONCRETE_BENCHMARK_FACTORIES = frozenset(
    {
        "MultipleChoiceSample",
        "afrimgsm",
        "afrimmlu",
        "afrixnli",
        "agieval",
        "arabicmmlu",
        "arc_mt",
        "assin",
        "babilong",
        "bangla",
        "bbh",
        "bbq",
        "bbq_age",
        "belebele",
        "blimp",
        "cabbq",
        "careqa",
        "ceval",
        "claw_eval_avg",
        "claw_eval_pass3",
        "cmmlu",
        "copal_id",
        "crows_pairs",
        "darijammlu",
        "darijammlu_accounting",
        "deepplanning",
        "egymmlu",
        "egymmlu_accounting",
        "esbbq",
        "eus_exams",
        "flores_es",
        "flores_es_es_pt",
        "flores_pt",
        "flores_pt_pt_en",
        "gpqa",
        "haerae",
        "haerae_general_knowledge",
        "hendrycks_math",
        "inverse_scaling",
        "kmmlu",
        "kobest",
        "kormedmcqa",
        "longbench",
        "longbench2",
        "longbench_multifieldqa_zh",
        "longbench_qasper",
        "longbench_samsum",
        "mastermind",
        "mcp_atlas",
        "mcpmark",
        "mgsm",
        "mlqa",
        "mmlu",
        "mmlu_cf",
        "mmlu_pro",
        "mmlu_pro_plus",
        "mmlu_redux",
        "nl2repo",
        "openbookqa_es",
        "paloma",
        "paloma_c4_en",
        "paws_es_spanish_bench",
        "phrases_es",
        "polemo2",
        "qasper",
        "qwenclawbench",
        "qwenwebbench",
        "ruler",
        "ruler_qa_squad",
        "scrolls",
        "skillsbench_avg5",
        "spanish_bench",
        "storycloze",
        "swe_bench_multilingual",
        "swe_bench_pro",
        "swe_bench_verified",
        "tau3_bench",
        "terminal_bench_2",
        "truthfulqa",
        "tool_decathlon",
        "vita_bench",
        "widesearch",
        "wmdp",
        "wnli_es",
        "xcopa",
        "xnli",
        "xnli_es_spanish_bench",
        "xquad",
    }
)

# These concrete unit-tested suites still need real Llama 3.2 full-model baseline captures on
# supported GPU hardware before they can be registered into `SUITE_SPECS`.
_KNOWN_FULL_MODEL_COVERAGE_GAPS = frozenset()


def _unit_tested_benchmark_factories() -> set[str]:
    factories: set[str] = set()
    for path in sorted(Path("tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        factories.update(_EVALUTION_BENCHMARK_FACTORY_PATTERN.findall(text))
    return factories


def test_llama3_2_full_model_specs_store_realistic_score_baselines() -> None:
    suspicious = find_suspicious_low_baselines(SUITE_SPECS)
    assert suspicious == [], (
        f"stored model score baselines must stay above {MIN_REALISTIC_BASELINE_SCORE} "
        f"unless explicitly allowlisted for genuinely hard suites; found {suspicious}"
    )


def test_unit_tested_benchmarks_have_expected_full_model_coverage() -> None:
    missing = {
        factory
        for factory in _unit_tested_benchmark_factories()
        if factory not in SUITE_SPECS and factory not in _NON_CONCRETE_BENCHMARK_FACTORIES
    }
    assert missing == _KNOWN_FULL_MODEL_COVERAGE_GAPS, (
        "unit-tested concrete benchmark suites must have stored full-model regression baselines; "
        f"unexpected coverage gaps: {sorted(missing - _KNOWN_FULL_MODEL_COVERAGE_GAPS)}, "
        f"remaining known gaps: {sorted(_KNOWN_FULL_MODEL_COVERAGE_GAPS & missing)}"
    )
