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
    "arc_challenge": benchmarks.arc_challenge,
    "arc_easy": benchmarks.arc_easy,
    "boolq": benchmarks.boolq,
    "cb": benchmarks.cb,
    "cola": benchmarks.cola,
    "copa": benchmarks.copa,
    "gsm8k": benchmarks.gsm8k,
    "gsm8k_platinum": benchmarks.gsm8k_platinum,
    "hellaswag": benchmarks.hellaswag,
    "mmlu": benchmarks.mmlu,
    "mmlu_pro": benchmarks.mmlu_pro,
    "mnli": benchmarks.mnli,
    "mrpc": benchmarks.mrpc,
    "openbookqa": benchmarks.openbookqa,
    "piqa": benchmarks.piqa,
    "qnli": benchmarks.qnli,
    "qqp": benchmarks.qqp,
    "rte": benchmarks.rte,
    "sst2": benchmarks.sst2,
    "wic": benchmarks.wic,
    "wnli": benchmarks.wnli,
    "winogrande": benchmarks.winogrande,
}


def run_yaml(source: str | Path) -> EvaluationRun:
    spec = _load_yaml_spec(source)
    evaluation = _build_engine(spec["engine"]).model(_build_model(spec["model"]))
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
    lines = [
        "import evalution as eval",
        "import evalution.benchmarks as benchmarks",
        "import evalution.engines as engines",
        "",
        "result = (",
        f"    {_emit_call(f'engines.{engine_alias}', _mapping_without_name(engine_spec), indent='    ')}",
        f"    .model({_emit_call('eval.Model', model_spec, indent='    ')})",
    ]
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
