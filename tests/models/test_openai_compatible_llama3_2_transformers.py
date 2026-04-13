# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pcre
import pytest

from tests.models_support import (
    LLAMA3_2_1B_INSTRUCT,
    LLAMA3_2_TRANSFORMERS_DEVICE,
    LLAMA3_2_TRANSFORMERS_TEST_MARKS,
    SUITE_SPECS,
    assert_metrics_match_baseline,
)

# Reuse the standard Llama 3.2 transformer integration marks for the HTTP proving-ground test.
pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS

# Keep the shared GSM8K Platinum settings explicit so the native and HTTP runs stay identical.
_SUITE_KEY = "gsm8k_platinum"
_OPENAI_SERVER_BATCH_SIZE = 24
_OPENAI_SERVER_BATCH_WINDOW_S = 0.01
_OPENAI_HTTP_NATIVE_DELTA_ABS_TOLERANCE = 4 / 128
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULT_JSON_RE = pcre.compile(r"RESULT_JSON_START\n(.*?)\nRESULT_JSON_END", pcre.DOTALL)


def _select_single_gpu_index() -> str:
    """Pick one low-usage physical GPU and expose only that device to the child run."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    candidates: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        index_text, used_text, total_text = [part.strip() for part in line.split(",")]
        candidates.append((int(index_text), int(used_text), int(total_text)))
    if not candidates:
        raise RuntimeError("nvidia-smi did not report any GPUs")
    candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
    return str(candidates[0][0])


def _run_suite_subprocess(mode: str, *, gpu_index: str) -> dict[str, object]:
    """Execute one isolated benchmark run and parse the printed JSON payload."""

    script = textwrap.dedent(
        f"""
        import json

        import evalution
        import evalution.engines as engines
        from evalution.config import Model
        from evalution.engines.openai_server import build_openai_compatible_server
        from tests.models_support import LLAMA3_2_1B_INSTRUCT, LLAMA3_2_TRANSFORMERS_DEVICE, SUITE_SPECS

        suite_key = {_SUITE_KEY!r}
        mode = {mode!r}
        suite = SUITE_SPECS[suite_key].suite_factory()
        model_path = str(LLAMA3_2_1B_INSTRUCT)

        if mode == "native":
            result = (
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="paged|flash_attention_2",
                    device=LLAMA3_2_TRANSFORMERS_DEVICE,
                    batch_size="auto",
                )
                .model(path=model_path)
                .run(suite)
                .result()
            )
        elif mode == "openai_http":
            backend_engine = engines.Transformers(
                dtype="bfloat16",
                attn_implementation="paged|flash_attention_2",
                device=LLAMA3_2_TRANSFORMERS_DEVICE,
                batch_size="auto",
            )
            with build_openai_compatible_server(
                engine=backend_engine,
                model=Model(path=model_path),
                model_name=model_path,
                max_batch_size={_OPENAI_SERVER_BATCH_SIZE!r},
                batch_window_s={_OPENAI_SERVER_BATCH_WINDOW_S!r},
            ) as server:
                result = (
                    engines.OpenAICompatible(
                        base_url=server.base_url,
                        batch_size={_OPENAI_SERVER_BATCH_SIZE!r},
                        max_parallel_requests={_OPENAI_SERVER_BATCH_SIZE!r},
                    )
                    .model(path=model_path)
                    .run(suite)
                    .result()
                )
        else:
            raise ValueError(mode)

        test = result.tests[0]
        print("RESULT_JSON_START")
        print(
            json.dumps(
                {{
                    "mode": mode,
                    "model": result.model,
                    "engine": result.engine,
                    "metrics": test.metrics,
                    "metadata": test.metadata,
                    "sample_count": len(test.samples),
                    "test_name": test.name,
                }},
                sort_keys=True,
            )
        )
        print("RESULT_JSON_END")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env={
            **dict(os.environ),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu_index,
            "EVALUTION_TEST_DEVICE": "cuda:0",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    match = _RESULT_JSON_RE.search(completed.stdout)
    assert match is not None, completed.stdout
    return json.loads(match.group(1))


def _render_ab_table(*, native_accuracy: float, openai_accuracy: float) -> str:
    """Render the required ASCII comparison table for the native-vs-HTTP benchmark run."""

    rows = [
        ("native_transformers", "N/A", f"{native_accuracy:.7f}", "N/A", "N/A", "N/A", "N/A", f"{0.0:+.7f}"),
        (
            "openai_http",
            "N/A",
            f"{openai_accuracy:.7f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            f"{openai_accuracy - native_accuracy:+.7f}",
        ),
    ]
    headers = (
        "variant",
        "performance",
        "accuracy",
        "cpu_time_s",
        "gpu_time_s",
        "cpu_ram_gib",
        "gpu_vram_gib",
        "delta_acc",
    )
    widths = [
        max(len(header), max(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = "| " + " | ".join(
        header.ljust(width) for header, width in zip(headers, widths, strict=True)
    ) + " |"
    data_rows = [
        "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ) + " |"
        for row in rows
    ]
    return "\n".join([separator, header_row, separator, *data_rows, separator])


def test_llama3_2_transformers_openai_compatible_gsm8k_platinum_matches_native_baseline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the OpenAI-compatible path preserves the native GSM8K Platinum score envelope."""

    del capsys
    spec = SUITE_SPECS[_SUITE_KEY]
    gpu_index = _select_single_gpu_index()

    native_result = _run_suite_subprocess("native", gpu_index=gpu_index)
    openai_result = _run_suite_subprocess("openai_http", gpu_index=gpu_index)

    assert native_result["test_name"] == spec.expected_name
    assert openai_result["test_name"] == spec.expected_name
    assert native_result["sample_count"] == spec.expected_sample_count
    assert openai_result["sample_count"] == spec.expected_sample_count
    assert set(native_result["metrics"]) == spec.expected_metrics
    assert set(openai_result["metrics"]) == spec.expected_metrics

    for key, expected_value in spec.expected_metadata.items():
        assert native_result["metadata"][key] == expected_value
        assert openai_result["metadata"][key] == expected_value

    assert native_result["engine"]["resolved_engine"] == "Transformers"
    assert (
        native_result["engine"]["execution"]["generation_backend"]
        == "continuous_batching"
    )
    assert openai_result["engine"]["resolved_engine"] == "OpenAICompatible"
    assert openai_result["engine"]["batch_size"] == _OPENAI_SERVER_BATCH_SIZE
    assert openai_result["engine"]["execution"]["generation_backend"] == "openai_http"
    assert openai_result["engine"]["execution"]["batch_size"] == _OPENAI_SERVER_BATCH_SIZE
    assert openai_result["engine"]["execution"]["model_name"] == str(LLAMA3_2_1B_INSTRUCT)
    assert (
        openai_result["engine"]["execution"]["max_parallel_requests"]
        == _OPENAI_SERVER_BATCH_SIZE
    )

    assert_metrics_match_baseline(
        native_result["metrics"],
        spec.baseline,
        abs_tolerance=spec.abs_tolerance,
    )
    assert_metrics_match_baseline(
        openai_result["metrics"],
        spec.baseline,
        abs_tolerance=spec.abs_tolerance,
    )
    assert_metrics_match_baseline(
        openai_result["metrics"],
        native_result["metrics"],
        abs_tolerance=_OPENAI_HTTP_NATIVE_DELTA_ABS_TOLERANCE,
    )

    print(
        _render_ab_table(
            native_accuracy=float(native_result["metrics"]["acc,num"]),
            openai_accuracy=float(openai_result["metrics"]["acc,num"]),
        )
    )
