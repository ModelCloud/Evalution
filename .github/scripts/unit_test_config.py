from __future__ import annotations

from dataclasses import asdict, dataclass

from common import normalize_test_file, test_requires_gpu, to_safe_name


@dataclass(frozen=True)
class UnitTestConfig:
    test_file: str
    safe_name: str
    requires_gpu: bool
    python_version: str
    uv_python: str


def resolve_unit_test_config(test_file: str) -> UnitTestConfig:
    normalized = normalize_test_file(test_file)
    python_version = "3.14t"
    uv_python = "3.14t"

    if normalized == "tests/test_tensorrt_llm_engine.py":
        python_version = "3.12"
        uv_python = "3.12"

    return UnitTestConfig(
        test_file=normalized,
        safe_name=to_safe_name(normalized),
        requires_gpu=test_requires_gpu(normalized),
        python_version=python_version,
        uv_python=uv_python,
    )


def resolve_unit_test_config_dict(test_file: str) -> dict[str, str | bool]:
    return asdict(resolve_unit_test_config(test_file))
