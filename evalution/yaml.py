# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any

import yaml

import evalution.benchmarks as benchmarks
from evalution.config import Model
from evalution.engines import BaseEngine, GPTQModel, Transformers, TransformersCompat
from evalution.runtime import EvaluationRun

_ENGINE_FACTORIES: dict[str, Any] = {
    "gptqmodel": GPTQModel,
    "transformers": Transformers,
    "transformerscompat": TransformersCompat,
}
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
    "code2text_go": benchmarks.code2text_go,
    "code2text_java": benchmarks.code2text_java,
    "code2text_javascript": benchmarks.code2text_javascript,
    "code2text_php": benchmarks.code2text_php,
    "code2text_python": benchmarks.code2text_python,
    "code2text_ruby": benchmarks.code2text_ruby,
    "commonsense_qa": benchmarks.commonsense_qa,
    "copa_ar": benchmarks.copa_ar,
    "copal_id_colloquial": benchmarks.copal_id_colloquial,
    "copal_id_standard": benchmarks.copal_id_standard,
    "coqa": benchmarks.coqa,
    "copa": benchmarks.copa,
    "darijahellaswag": benchmarks.darijahellaswag,
    "egyhellaswag": benchmarks.egyhellaswag,
    "drop": benchmarks.drop,
    "darijammlu": benchmarks.darijammlu,
    "egymmlu": benchmarks.egymmlu,
    "eus_exams": benchmarks.eus_exams,
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
    "gsm8k_platinum": benchmarks.gsm8k_platinum,
    "headqa_en": benchmarks.headqa_en,
    "headqa_es": benchmarks.headqa_es,
    "hellaswag": benchmarks.hellaswag,
    "histoires_morales": benchmarks.histoires_morales,
    "icelandic_winogrande": benchmarks.icelandic_winogrande,
    "inverse_scaling": benchmarks.inverse_scaling,
    "kobest": benchmarks.kobest,
    "kobest_boolq": benchmarks.kobest_boolq,
    "kobest_copa": benchmarks.kobest_copa,
    "kobest_hellaswag": benchmarks.kobest_hellaswag,
    "kobest_sentineg": benchmarks.kobest_sentineg,
    "kobest_wic": benchmarks.kobest_wic,
    "lambada_openai": benchmarks.lambada_openai,
    "lambada_openai_cloze": benchmarks.lambada_openai_cloze,
    "lambada_standard": benchmarks.lambada_standard,
    "lambada_standard_cloze": benchmarks.lambada_standard_cloze,
    "logiqa": benchmarks.logiqa,
    "logiqa2": benchmarks.logiqa2,
    "mathqa": benchmarks.mathqa,
    "mc_taco": benchmarks.mc_taco,
    "medmcqa": benchmarks.medmcqa,
    "medqa_4options": benchmarks.medqa_4options,
    "mmlu": benchmarks.mmlu,
    "mmlu_pro": benchmarks.mmlu_pro,
    "mnli": benchmarks.mnli,
    "mrpc": benchmarks.mrpc,
    "nq_open": benchmarks.nq_open,
    "openbookqa": benchmarks.openbookqa,
    "paws_x_de": benchmarks.paws_x_de,
    "paws_x_en": benchmarks.paws_x_en,
    "paws_x_es": benchmarks.paws_x_es,
    "paws_x_fr": benchmarks.paws_x_fr,
    "paws_x_ja": benchmarks.paws_x_ja,
    "paws_x_ko": benchmarks.paws_x_ko,
    "paws_x_zh": benchmarks.paws_x_zh,
    "piqa": benchmarks.piqa,
    "piqa_ar": benchmarks.piqa_ar,
    "pile_10k": benchmarks.pile_10k,
    "prost": benchmarks.prost,
    "pubmedqa": benchmarks.pubmedqa,
    "qa4mre_2011": benchmarks.qa4mre_2011,
    "qa4mre_2012": benchmarks.qa4mre_2012,
    "qa4mre_2013": benchmarks.qa4mre_2013,
    "qnli": benchmarks.qnli,
    "qqp": benchmarks.qqp,
    "race": benchmarks.race,
    "rte": benchmarks.rte,
    "sciq": benchmarks.sciq,
    "siqa": benchmarks.siqa,
    "swag": benchmarks.swag,
    "sst2": benchmarks.sst2,
    "squadv2": benchmarks.squadv2,
    "truthfulqa": benchmarks.truthfulqa,
    "truthfulqa_mc1": benchmarks.truthfulqa_mc1,
    "truthfulqa_mc2": benchmarks.truthfulqa_mc2,
    "triviaqa": benchmarks.triviaqa,
    "wic": benchmarks.wic,
    "webqs": benchmarks.webqs,
    "wikitext": benchmarks.wikitext,
    "winogender_all": benchmarks.winogender_all,
    "winogender_female": benchmarks.winogender_female,
    "winogender_gotcha": benchmarks.winogender_gotcha,
    "winogender_gotcha_female": benchmarks.winogender_gotcha_female,
    "winogender_gotcha_male": benchmarks.winogender_gotcha_male,
    "winogender_male": benchmarks.winogender_male,
    "winogender_neutral": benchmarks.winogender_neutral,
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
    "xwinograd_en": benchmarks.xwinograd_en,
    "xwinograd_fr": benchmarks.xwinograd_fr,
    "xwinograd_jp": benchmarks.xwinograd_jp,
    "xwinograd_pt": benchmarks.xwinograd_pt,
    "xwinograd_ru": benchmarks.xwinograd_ru,
    "xwinograd_zh": benchmarks.xwinograd_zh,
}

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
    model_spec = _coerce_named_mapping(spec.get("model"), label="model")
    test_specs = spec.get("tests")
    if not isinstance(test_specs, list) or not test_specs:
        raise TypeError("tests must be a non-empty list of test suite mappings")

    if engine_name == "transformers":
        engine_alias = "Transformers"
    elif engine_name == "gptqmodel":
        engine_alias = "GPTQModel"
    elif engine_name == "transformerscompat":
        engine_alias = "TransformersCompat"
    else:
        engine_alias = engine_name
    model_config = _build_model(model_spec)
    model_kwargs = _build_model_emit_kwargs(model_config)
    lines = [
        "import evalution as eval",
        "import evalution.benchmarks as benchmarks",
        "import evalution.engines as engines",
        "",
        "result = (",
        f"    {_emit_call(f'engines.{engine_alias}', _mapping_without_name(engine_spec), indent='    ')}",
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
        if maybe_path.exists():
            path = maybe_path
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
    factory = _ENGINE_FACTORIES.get(engine_name)
    if factory is None:
        raise KeyError(f"unknown engine type: {engine_name!r}")
    return factory(**_mapping_without_name(mapping))


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
    return name.strip().lower()


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
