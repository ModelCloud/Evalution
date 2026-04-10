# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from pprint import pformat
from typing import Any

import yaml

import evalution.benchmarks as benchmarks
from evalution.config import Model
from evalution.engines import (
    BaseEngine,
    GPTQModel,
    OpenVINO,
    SGLang,
    TensorRTLLM,
    Transformers,
    TransformersCompat,
    VLLM,
)
from evalution.runtime import EvaluationRun


@dataclass(frozen=True, slots=True)
class _EngineSpec:
    # Preserve the legacy registry shape used by CLI/YAML tests while still allowing bare classes.
    factory: type[BaseEngine]
    emit_alias: str | None = None


_EngineRegistryEntry = type[BaseEngine] | _EngineSpec

# Keep engine lookup centralized, but derive YAML option inheritance directly
# from the concrete engine dataclass hierarchy so Python and YAML stay aligned.
_ENGINE_REGISTRY: dict[str, _EngineRegistryEntry] = {
    "transformers": _EngineSpec(factory=Transformers, emit_alias="Transformers"),
    "transformerscompat": _EngineSpec(factory=TransformersCompat, emit_alias="TransformersCompat"),
    "gptqmodel": _EngineSpec(factory=GPTQModel, emit_alias="GPTQModel"),
    "openvino": _EngineSpec(factory=OpenVINO, emit_alias="OpenVINO"),
    "tensorrtllm": _EngineSpec(factory=TensorRTLLM, emit_alias="TensorRTLLM"),
    "vllm": _EngineSpec(factory=VLLM, emit_alias="VLLM"),
    "sglang": _EngineSpec(factory=SGLang, emit_alias="SGLang"),
}

# Map every YAML/CLI benchmark name to the corresponding benchmark factory.
_TEST_FACTORIES: dict[str, Any] = {
    "aexams_biology": benchmarks.aexams_biology,
    "aexams_islamic_studies": benchmarks.aexams_islamic_studies,
    "aexams_physics": benchmarks.aexams_physics,
    "aexams_science": benchmarks.aexams_science,
    "aexams_social": benchmarks.aexams_social,
    "agieval": benchmarks.agieval,
    "afrimgsm": benchmarks.afrimgsm,
    "afrimmlu": benchmarks.afrimmlu,
    "afrixnli": benchmarks.afrixnli,
    "afrixnli_amh": benchmarks.afrixnli_amh,
    "afrixnli_eng": benchmarks.afrixnli_eng,
    "afrixnli_ewe": benchmarks.afrixnli_ewe,
    "afrixnli_fra": benchmarks.afrixnli_fra,
    "afrixnli_hau": benchmarks.afrixnli_hau,
    "afrixnli_ibo": benchmarks.afrixnli_ibo,
    "afrixnli_kin": benchmarks.afrixnli_kin,
    "afrixnli_lin": benchmarks.afrixnli_lin,
    "afrixnli_lug": benchmarks.afrixnli_lug,
    "afrixnli_orm": benchmarks.afrixnli_orm,
    "afrixnli_sna": benchmarks.afrixnli_sna,
    "afrixnli_sot": benchmarks.afrixnli_sot,
    "afrixnli_swa": benchmarks.afrixnli_swa,
    "afrixnli_twi": benchmarks.afrixnli_twi,
    "afrixnli_wol": benchmarks.afrixnli_wol,
    "afrixnli_xho": benchmarks.afrixnli_xho,
    "afrixnli_yor": benchmarks.afrixnli_yor,
    "afrixnli_zul": benchmarks.afrixnli_zul,
    "aime": benchmarks.aime,
    "aime24": benchmarks.aime24,
    "aime25": benchmarks.aime25,
    "aime26": benchmarks.aime26,
    "anli_r1": benchmarks.anli_r1,
    "anli_r2": benchmarks.anli_r2,
    "anli_r3": benchmarks.anli_r3,
    "arabicmmlu": benchmarks.arabicmmlu,
    "arc_challenge": benchmarks.arc_challenge,
    "arc_easy": benchmarks.arc_easy,
    "arc_mt": benchmarks.arc_mt,
    "arc_mt_da": benchmarks.arc_mt_da,
    "arc_mt_de": benchmarks.arc_mt_de,
    "arc_mt_el": benchmarks.arc_mt_el,
    "arc_mt_es": benchmarks.arc_mt_es,
    "arc_mt_fi": benchmarks.arc_mt_fi,
    "arc_mt_hu": benchmarks.arc_mt_hu,
    "arc_mt_is": benchmarks.arc_mt_is,
    "arc_mt_it": benchmarks.arc_mt_it,
    "arc_mt_nb": benchmarks.arc_mt_nb,
    "arc_mt_pl": benchmarks.arc_mt_pl,
    "arc_mt_pt": benchmarks.arc_mt_pt,
    "arc_mt_sv": benchmarks.arc_mt_sv,
    "assin": benchmarks.assin,
    "assin_entailment": benchmarks.assin_entailment,
    "assin_paraphrase": benchmarks.assin_paraphrase,
    "arithmetic_1dc": benchmarks.arithmetic_1dc,
    "arithmetic_2da": benchmarks.arithmetic_2da,
    "arithmetic_2dm": benchmarks.arithmetic_2dm,
    "arithmetic_2ds": benchmarks.arithmetic_2ds,
    "arithmetic_3da": benchmarks.arithmetic_3da,
    "arithmetic_3ds": benchmarks.arithmetic_3ds,
    "arithmetic_4da": benchmarks.arithmetic_4da,
    "arithmetic_4ds": benchmarks.arithmetic_4ds,
    "arithmetic_5da": benchmarks.arithmetic_5da,
    "arithmetic_5ds": benchmarks.arithmetic_5ds,
    "asdiv": benchmarks.asdiv,
    "asdiv_cot_llama": benchmarks.asdiv_cot_llama,
    "babi": benchmarks.babi,
    "babilong": benchmarks.babilong,
    "bbh": benchmarks.bbh,
    "bangla": benchmarks.bangla,
    "bangla_boolqa": benchmarks.bangla_boolqa,
    "bangla_commonsenseqa": benchmarks.bangla_commonsenseqa,
    "bangla_mmlu": benchmarks.bangla_mmlu,
    "bangla_openbookqa": benchmarks.bangla_openbookqa,
    "bangla_piqa": benchmarks.bangla_piqa,
    "bear": benchmarks.bear,
    "bear_big": benchmarks.bear_big,
    "belebele": benchmarks.belebele,
    "belebele_por_Latn": benchmarks.belebele_por_Latn,
    "belebele_spa_Latn": benchmarks.belebele_spa_Latn,
    "bbq": benchmarks.bbq,
    "blimp": benchmarks.blimp,
    "c4": benchmarks.c4,
    "careqa": benchmarks.careqa,
    "careqa_en": benchmarks.careqa_en,
    "careqa_es": benchmarks.careqa_es,
    "cabbq": benchmarks.cabbq,
    "esbbq": benchmarks.esbbq,
    "ceval": benchmarks.ceval,
    "boolq": benchmarks.boolq,
    "cb": benchmarks.cb,
    "cola": benchmarks.cola,
    "cnn_dailymail": benchmarks.cnn_dailymail,
    "cmmlu": benchmarks.cmmlu,
    "code2text_go": benchmarks.code2text_go,
    "code2text_java": benchmarks.code2text_java,
    "code2text_javascript": benchmarks.code2text_javascript,
    "code2text_php": benchmarks.code2text_php,
    "code2text_python": benchmarks.code2text_python,
    "code2text_ruby": benchmarks.code2text_ruby,
    "cocoteros_es": benchmarks.cocoteros_es,
    "commonsense_qa": benchmarks.commonsense_qa,
    "copa_ar": benchmarks.copa_ar,
    "copal_id_colloquial": benchmarks.copal_id_colloquial,
    "copal_id_standard": benchmarks.copal_id_standard,
    "coqa": benchmarks.coqa,
    "copa": benchmarks.copa,
    "darijahellaswag": benchmarks.darijahellaswag,
    "egyhellaswag": benchmarks.egyhellaswag,
    "drop": benchmarks.drop,
    "fld": benchmarks.fld,
    "fda": benchmarks.fda,
    "flores_es": benchmarks.flores_es,
    "flores_pt": benchmarks.flores_pt,
    "french_bench_arc_challenge": benchmarks.french_bench_arc_challenge,
    "darijammlu": benchmarks.darijammlu,
    "egymmlu": benchmarks.egymmlu,
    "eus_exams": benchmarks.eus_exams,
    "eus_reading": benchmarks.eus_reading,
    "eus_proficiency": benchmarks.eus_proficiency,
    "eus_trivia": benchmarks.eus_trivia,
    "gpqa": benchmarks.gpqa,
    "gpqa_main": benchmarks.gpqa_main,
    "gpqa_diamond": benchmarks.gpqa_diamond,
    "gpqa_extended": benchmarks.gpqa_extended,
    "ethics_cm": benchmarks.ethics_cm,
    "ethics_deontology": benchmarks.ethics_deontology,
    "ethics_justice": benchmarks.ethics_justice,
    "ethics_utilitarianism": benchmarks.ethics_utilitarianism,
    "ethics_virtue": benchmarks.ethics_virtue,
    "gsm8k": benchmarks.gsm8k,
    "gsm8k_fr": benchmarks.gsm8k_fr,
    "gsm8k_ko": benchmarks.gsm8k_ko,
    "groundcocoa": benchmarks.groundcocoa,
    "gsm_plus": benchmarks.gsm_plus,
    "gsm_plus_mini": benchmarks.gsm_plus_mini,
    "gsm8k_platinum": benchmarks.gsm8k_platinum,
    "headqa_en": benchmarks.headqa_en,
    "headqa_es": benchmarks.headqa_es,
    "hendrycks_math": benchmarks.hendrycks_math,
    "hellaswag": benchmarks.hellaswag,
    "histoires_morales": benchmarks.histoires_morales,
    "moral_stories": benchmarks.moral_stories,
    "icelandic_winogrande": benchmarks.icelandic_winogrande,
    "ifeval": benchmarks.ifeval,
    "ifeval_pt": benchmarks.ifeval_pt,
    "inverse_scaling": benchmarks.inverse_scaling,
    "kmmlu": benchmarks.kmmlu,
    "kobest": benchmarks.kobest,
    "kobest_boolq": benchmarks.kobest_boolq,
    "kobest_copa": benchmarks.kobest_copa,
    "kobest_hellaswag": benchmarks.kobest_hellaswag,
    "kobest_sentineg": benchmarks.kobest_sentineg,
    "kobest_wic": benchmarks.kobest_wic,
    "longbench": benchmarks.longbench,
    "longbench2": benchmarks.longbench2,
    "lambada_openai": benchmarks.lambada_openai,
    "lambada_openai_mt_de": benchmarks.lambada_openai_mt_de,
    "lambada_openai_mt_en": benchmarks.lambada_openai_mt_en,
    "lambada_openai_mt_es": benchmarks.lambada_openai_mt_es,
    "lambada_openai_mt_fr": benchmarks.lambada_openai_mt_fr,
    "lambada_openai_mt_it": benchmarks.lambada_openai_mt_it,
    "lambada_openai_mt_stablelm_de": benchmarks.lambada_openai_mt_stablelm_de,
    "lambada_openai_mt_stablelm_en": benchmarks.lambada_openai_mt_stablelm_en,
    "lambada_openai_mt_stablelm_es": benchmarks.lambada_openai_mt_stablelm_es,
    "lambada_openai_mt_stablelm_fr": benchmarks.lambada_openai_mt_stablelm_fr,
    "lambada_openai_mt_stablelm_it": benchmarks.lambada_openai_mt_stablelm_it,
    "lambada_openai_mt_stablelm_nl": benchmarks.lambada_openai_mt_stablelm_nl,
    "lambada_openai_mt_stablelm_pt": benchmarks.lambada_openai_mt_stablelm_pt,
    "lambada_openai_cloze": benchmarks.lambada_openai_cloze,
    "lambada_standard": benchmarks.lambada_standard,
    "lambada_standard_cloze": benchmarks.lambada_standard_cloze,
    "logiqa": benchmarks.logiqa,
    "logiqa2": benchmarks.logiqa2,
    "humaneval": benchmarks.humaneval,
    "mbpp": benchmarks.mbpp,
    "mathqa": benchmarks.mathqa,
    "mgsm": benchmarks.mgsm,
    "mgsm_direct_es_spanish_bench": benchmarks.mgsm_direct_es_spanish_bench,
    "mlqa": benchmarks.mlqa,
    "mastermind_24_easy": benchmarks.mastermind_24_easy,
    "mastermind_24_hard": benchmarks.mastermind_24_hard,
    "mastermind_35_easy": benchmarks.mastermind_35_easy,
    "mastermind_35_hard": benchmarks.mastermind_35_hard,
    "mastermind_46_easy": benchmarks.mastermind_46_easy,
    "mastermind_46_hard": benchmarks.mastermind_46_hard,
    "mc_taco": benchmarks.mc_taco,
    "medmcqa": benchmarks.medmcqa,
    "medqa_4options": benchmarks.medqa_4options,
    "mmlu": benchmarks.mmlu,
    "mmlu_cf": benchmarks.mmlu_cf,
    "mmlu_pro": benchmarks.mmlu_pro,
    "mmlu_pro_plus": benchmarks.mmlu_pro_plus,
    "mmlu_redux": benchmarks.mmlu_redux,
    "mnli": benchmarks.mnli,
    "mrpc": benchmarks.mrpc,
    "multirc": benchmarks.multirc,
    "mutual": benchmarks.mutual,
    "nq_open": benchmarks.nq_open,
    "noticia": benchmarks.noticia,
    "openbookqa": benchmarks.openbookqa,
    "paloma": benchmarks.paloma,
    "paws_x_de": benchmarks.paws_x_de,
    "paws_x_en": benchmarks.paws_x_en,
    "paws_x_es": benchmarks.paws_x_es,
    "paws_x_fr": benchmarks.paws_x_fr,
    "paws_x_ja": benchmarks.paws_x_ja,
    "paws_x_ko": benchmarks.paws_x_ko,
    "paws_x_zh": benchmarks.paws_x_zh,
    "phrases_es": benchmarks.phrases_es,
    "piqa": benchmarks.piqa,
    "piqa_ar": benchmarks.piqa_ar,
    "pile_10k": benchmarks.pile_10k,
    "polemo2_in": benchmarks.polemo2_in,
    "polemo2_out": benchmarks.polemo2_out,
    "prost": benchmarks.prost,
    "pubmedqa": benchmarks.pubmedqa,
    "qasper": benchmarks.qasper,
    "qasper_bool": benchmarks.qasper_bool,
    "qasper_freeform": benchmarks.qasper_freeform,
    "qa4mre_2011": benchmarks.qa4mre_2011,
    "qa4mre_2012": benchmarks.qa4mre_2012,
    "qa4mre_2013": benchmarks.qa4mre_2013,
    "qnli": benchmarks.qnli,
    "qqp": benchmarks.qqp,
    "race": benchmarks.race,
    "record": benchmarks.record,
    "ruler": benchmarks.ruler,
    "rte": benchmarks.rte,
    "sciq": benchmarks.sciq,
    "scrolls": benchmarks.scrolls,
    "spanish_bench": benchmarks.spanish_bench,
    "siqa": benchmarks.siqa,
    "simple_cooccurrence_bias": benchmarks.simple_cooccurrence_bias,
    "storycloze": benchmarks.storycloze,
    "swag": benchmarks.swag,
    "sst2": benchmarks.sst2,
    "swde": benchmarks.swde,
    "squad_completion": benchmarks.squad_completion,
    "squadv2": benchmarks.squadv2,
    "toxigen": benchmarks.toxigen,
    "truthfulqa": benchmarks.truthfulqa,
    "truthfulqa_mc1": benchmarks.truthfulqa_mc1,
    "truthfulqa_mc2": benchmarks.truthfulqa_mc2,
    "triviaqa": benchmarks.triviaqa,
    "wic": benchmarks.wic,
    "webqs": benchmarks.webqs,
    "wikitext": benchmarks.wikitext,
    "wmdp": benchmarks.wmdp,
    "winogender_all": benchmarks.winogender_all,
    "winogender_female": benchmarks.winogender_female,
    "winogender_gotcha": benchmarks.winogender_gotcha,
    "winogender_gotcha_female": benchmarks.winogender_gotcha_female,
    "winogender_gotcha_male": benchmarks.winogender_gotcha_male,
    "winogender_male": benchmarks.winogender_male,
    "winogender_neutral": benchmarks.winogender_neutral,
    "wsc": benchmarks.wsc,
    "wsc273": benchmarks.wsc273,
    "wnli": benchmarks.wnli,
    "winogrande": benchmarks.winogrande,
    "xcopa_et": benchmarks.xcopa_et,
    "xcopa_ht": benchmarks.xcopa_ht,
    "xcopa_id": benchmarks.xcopa_id,
    "xcopa_it": benchmarks.xcopa_it,
    "xcopa_qu": benchmarks.xcopa_qu,
    "xcopa_sw": benchmarks.xcopa_sw,
    "xcopa_ta": benchmarks.xcopa_ta,
    "xcopa_th": benchmarks.xcopa_th,
    "xcopa_tr": benchmarks.xcopa_tr,
    "xcopa_vi": benchmarks.xcopa_vi,
    "xcopa_zh": benchmarks.xcopa_zh,
    "xquad": benchmarks.xquad,
    "xstorycloze_ar": benchmarks.xstorycloze_ar,
    "xstorycloze_en": benchmarks.xstorycloze_en,
    "xstorycloze_es": benchmarks.xstorycloze_es,
    "xstorycloze_eu": benchmarks.xstorycloze_eu,
    "xstorycloze_hi": benchmarks.xstorycloze_hi,
    "xstorycloze_id": benchmarks.xstorycloze_id,
    "xstorycloze_my": benchmarks.xstorycloze_my,
    "xstorycloze_ru": benchmarks.xstorycloze_ru,
    "xstorycloze_sw": benchmarks.xstorycloze_sw,
    "xstorycloze_te": benchmarks.xstorycloze_te,
    "xstorycloze_zh": benchmarks.xstorycloze_zh,
    "xnli": benchmarks.xnli,
    "xnli_eu": benchmarks.xnli_eu,
    "xlsum_es": benchmarks.xlsum_es,
    "xwinograd_en": benchmarks.xwinograd_en,
    "xwinograd_fr": benchmarks.xwinograd_fr,
    "xwinograd_jp": benchmarks.xwinograd_jp,
    "xwinograd_pt": benchmarks.xwinograd_pt,
    "xwinograd_ru": benchmarks.xwinograd_ru,
    "xwinograd_zh": benchmarks.xwinograd_zh,
}

for _click_task in benchmarks.CLICK_TASKS:
    _TEST_FACTORIES[_click_task] = getattr(benchmarks, _click_task)

del _click_task

for _task_name in benchmarks.CABBQ_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

for _task_name in benchmarks.ESBBQ_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.BBQ_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.XNLI_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.XQUAD_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.TRUTHFULQA_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.INVERSE_SCALING_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.WMDP_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.DARIJAMMLU_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.EGYMMLU_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.EUS_EXAMS_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _agieval_task in benchmarks.AGIEVAL_TASKS:
    _TEST_FACTORIES[_agieval_task] = getattr(benchmarks, _agieval_task)

del _agieval_task

for _afrimgsm_task in benchmarks.AFRIMGSM_TASKS:
    _TEST_FACTORIES[_afrimgsm_task] = getattr(benchmarks, _afrimgsm_task)

del _afrimgsm_task

for _afrimmlu_task in benchmarks.AFRIMMLU_TASKS:
    _TEST_FACTORIES[_afrimmlu_task] = getattr(benchmarks, _afrimmlu_task)

del _afrimmlu_task

for _crows_pairs_task in benchmarks.CROWS_PAIRS_TASKS:
    _TEST_FACTORIES[_crows_pairs_task] = getattr(benchmarks, _crows_pairs_task)

del _crows_pairs_task

for _bbh_task in benchmarks.BBH_TASKS:
    _TEST_FACTORIES[_bbh_task] = getattr(benchmarks, _bbh_task)

del _bbh_task

for _babilong_task in benchmarks.BABILONG_TASKS:
    _TEST_FACTORIES[_babilong_task] = getattr(benchmarks, _babilong_task)

del _babilong_task

for _arabicmmlu_task in benchmarks.ARABICMMLU_TASKS:
    _TEST_FACTORIES[_arabicmmlu_task] = getattr(benchmarks, _arabicmmlu_task)

del _arabicmmlu_task

for _hendrycks_math_task in benchmarks.HENDRYCKS_MATH_TASKS:
    _TEST_FACTORIES[_hendrycks_math_task] = getattr(benchmarks, _hendrycks_math_task)

del _hendrycks_math_task

for _task_name in benchmarks.CMMLU_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.KMMLU_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.MGSM_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.MMLU_CF_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.MLQA_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.PALOMA_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.LONG_BENCH_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.LONG_BENCH2_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.STORYCLOZE_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.SCROLLS_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name

for _task_name in benchmarks.RULER_TASKS:
    _TEST_FACTORIES[_task_name] = getattr(benchmarks, _task_name)

del _task_name


def run_yaml(source: str | Path) -> EvaluationRun:
    spec = _load_yaml_spec(source)
    model_config = _build_model(spec["model"])
    evaluation = _build_engine(spec["engine"]).model(**model_config.to_dict())
    for test in _build_tests(spec["tests"]):
        evaluation.run(test)
    return evaluation


def python_from_yaml(source: str | Path) -> str:
    spec = _load_yaml_spec(source)
    engine_spec = _coerce_named_mapping(spec.get("engine"), label="engine")
    engine_name = _normalize_engine_name(_extract_name(engine_spec, label="engine"))
    engine_kwargs = _mapping_without_name(engine_spec)
    _validate_engine_option_keys(engine_name, engine_kwargs)
    model_spec = _coerce_named_mapping(spec.get("model"), label="model")
    test_specs = spec.get("tests")
    if not isinstance(test_specs, list) or not test_specs:
        raise TypeError("tests must be a non-empty list of test suite mappings")

    engine_alias = _resolve_engine_emit_alias(engine_name)
    model_config = _build_model(model_spec)
    model_kwargs = _build_model_emit_kwargs(model_config)
    lines = [
        "import evalution as eval",
        "import evalution.benchmarks as benchmarks",
        "import evalution.engines as engines",
        "",
        "result = (",
        f"    {_emit_call(f'engines.{engine_alias}', engine_kwargs, indent='    ')}",
    ]
    lines.extend(_emit_keyword_call("model", model_kwargs, indent="    "))
    for test_spec in test_specs:
        test_mapping = _coerce_named_mapping(test_spec, label="test")
        test_name = _extract_name(test_mapping, label="test")
        lines.append(
            f"    .run({_emit_call(f'benchmarks.{test_name}', _mapping_without_name(test_mapping), indent='    ')})"
        )
    lines.append(")")
    return "\n".join(lines)


def _load_yaml_spec(source: str | Path) -> dict[str, Any]:
    path: Path | None = None
    if isinstance(source, Path):
        path = source
    elif isinstance(source, str):
        maybe_path = Path(source)
        try:
            if maybe_path.exists():
                path = maybe_path
        except OSError:
            # Treat long or multiline strings as inline YAML instead of filesystem paths.
            path = None
    if path is not None:
        text = path.read_text(encoding="utf-8")
    elif isinstance(source, str):
        text = source
    else:
        raise TypeError("source must be a YAML string or filesystem path")

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("yaml spec must decode to a mapping")
    if "engine" not in loaded:
        raise KeyError("yaml spec must define an engine section")
    if "model" not in loaded:
        raise KeyError("yaml spec must define a model section")
    if "tests" not in loaded:
        raise KeyError("yaml spec must define a tests section")
    return loaded


def _build_engine(spec: Any) -> BaseEngine:
    mapping = _coerce_named_mapping(spec, label="engine")
    engine_name = _normalize_engine_name(_extract_name(mapping, label="engine"))
    engine_kwargs = _mapping_without_name(mapping)
    _validate_engine_option_keys(engine_name, engine_kwargs)
    factory = _engine_factory(engine_name)
    return factory(**engine_kwargs)


def _build_model(spec: Any) -> Model:
    if not isinstance(spec, dict):
        raise TypeError("model must be a mapping of model options")
    return Model(**spec)


def _build_tests(specs: Any) -> list[Any]:
    if not isinstance(specs, list) or not specs:
        raise TypeError("tests must be a non-empty list of test suite mappings")

    built_tests: list[Any] = []
    for spec in specs:
        mapping = _coerce_named_mapping(spec, label="test")
        test_name = _extract_name(mapping, label="test")
        factory = _TEST_FACTORIES.get(test_name)
        if factory is None:
            raise KeyError(f"unknown test type: {test_name!r}")
        built_tests.append(factory(**_mapping_without_name(mapping)))
    return built_tests


def _coerce_named_mapping(spec: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(spec)


def _extract_name(mapping: dict[str, Any], *, label: str) -> str:
    raw_name = mapping.get("type", mapping.get("name"))
    if not isinstance(raw_name, str) or not raw_name:
        raise KeyError(f"{label} must define a non-empty 'type' or 'name'")
    return raw_name


def _normalize_engine_name(name: str) -> str:
    """Normalize engine lookup keys so CLI, YAML, and Python inputs share one registry."""

    return name.strip().lower()


def _resolve_engine_emit_alias(engine_name: str) -> str:
    """Resolve the public Python constructor name used by python_from_yaml."""

    spec = _engine_spec(engine_name)
    return spec.emit_alias or spec.factory.__name__


def _engine_factory(engine_name: str) -> type[BaseEngine]:
    """Resolve the registered engine factory or raise the standard unknown-engine error."""

    return _engine_spec(engine_name).factory


def _engine_spec(engine_name: str) -> _EngineSpec:
    """Normalize legacy and new registry entries to one internal shape."""

    entry = _ENGINE_REGISTRY.get(engine_name)
    if entry is None:
        raise KeyError(f"unknown engine type: {engine_name!r}")
    if isinstance(entry, _EngineSpec):
        return entry
    return _EngineSpec(factory=entry, emit_alias=entry.__name__)


@lru_cache(maxsize=None)
def _engine_option_keys(engine_name: str) -> frozenset[str]:
    """Resolve YAML option keys from one engine dataclass init signature."""

    factory = _engine_factory(engine_name)
    if not is_dataclass(factory):
        return frozenset()
    return frozenset(field.name for field in fields(factory) if field.init)


def _validate_engine_option_keys(engine_name: str, kwargs: dict[str, Any]) -> None:
    """Reject engine kwargs that do not belong to the selected engine family."""

    allowed_keys = _engine_option_keys(engine_name)
    if not allowed_keys:
        return

    unexpected_keys = sorted(key for key in kwargs if key not in allowed_keys)
    if not unexpected_keys:
        return

    allowed_list = ", ".join(sorted(allowed_keys))
    unexpected_list = ", ".join(unexpected_keys)
    raise KeyError(
        f"engine {engine_name!r} does not accept option(s): {unexpected_list}; "
        f"allowed options: {allowed_list}"
    )


def _mapping_without_name(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mapping.items()
        if key not in {"type", "name"}
    }


def _emit_call(name: str, kwargs: dict[str, Any], *, indent: str) -> str:
    if not kwargs:
        return f"{name}()"

    lines = [f"{name}("]
    for key, value in kwargs.items():
        rendered_value = pformat(value, sort_dicts=False)
        rendered_lines = rendered_value.splitlines() or [rendered_value]
        lines.append(f"    {key}={rendered_lines[0]},")
        for continuation in rendered_lines[1:]:
            lines.append(f"    {continuation}")
    lines.append(")")
    return "\n".join(f"{indent}{line}" if index else line for index, line in enumerate(lines))


def _emit_keyword_call(name: str, kwargs: dict[str, Any], *, indent: str) -> list[str]:
    lines: list[str] = []
    if not kwargs:
        lines.append(f"{indent}.{name}()")
        return lines

    lines.append(f"{indent}.{name}(")
    for key, value in kwargs.items():
        rendered_value = pformat(value, sort_dicts=False)
        rendered_lines = rendered_value.splitlines() or [rendered_value]
        lines.append(f"        {key}={rendered_lines[0]},")
        for continuation in rendered_lines[1:]:
            lines.append(f"        {continuation}")
    lines.append("    )")
    return lines


def _build_model_emit_kwargs(model: Model) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"path": model.path}
    if model.tokenizer is not None:
        kwargs["tokenizer"] = model.tokenizer
    if model.tokenizer_path is not None:
        kwargs["tokenizer_path"] = model.tokenizer_path
    if model.revision is not None:
        kwargs["revision"] = model.revision
    if model.trust_remote_code:
        kwargs["trust_remote_code"] = model.trust_remote_code
    if model.model_kwargs:
        kwargs["model_kwargs"] = model.model_kwargs
    if model.tokenizer_kwargs:
        kwargs["tokenizer_kwargs"] = model.tokenizer_kwargs
    return kwargs

for _haerae_task in benchmarks.HAERAE_TASKS:
    _TEST_FACTORIES[_haerae_task] = getattr(benchmarks, _haerae_task)

del _haerae_task

for _kormedmcqa_task in benchmarks.KORMEDMCQA_TASKS:
    _TEST_FACTORIES[_kormedmcqa_task] = getattr(benchmarks, _kormedmcqa_task)

del _kormedmcqa_task

for _spanish_bench_task in benchmarks.SPANISH_BENCH_TASKS:
    _TEST_FACTORIES[_spanish_bench_task] = getattr(benchmarks, _spanish_bench_task)

del _spanish_bench_task

for _flores_es_task in benchmarks.FLORES_ES_TASKS:
    _TEST_FACTORIES[_flores_es_task] = getattr(benchmarks, _flores_es_task)

del _flores_es_task

for _phrases_es_task in benchmarks.PHRASES_ES_TASKS:
    _TEST_FACTORIES[_phrases_es_task] = getattr(benchmarks, _phrases_es_task)

del _phrases_es_task

for _flores_pt_task in benchmarks.FLORES_PT_TASKS:
    _TEST_FACTORIES[_flores_pt_task] = getattr(benchmarks, _flores_pt_task)

del _flores_pt_task
