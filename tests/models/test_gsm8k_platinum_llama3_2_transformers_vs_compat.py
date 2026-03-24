# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from pathlib import Path
import pcre
import subprocess
import sys
import textwrap

from tests.models_support import (
    LLAMA3_2_1B_INSTRUCT,
    LLAMA3_2_TRANSFORMERS_DEVICE,
    LLAMA3_2_TRANSFORMERS_TEST_MARKS,
    assert_metrics_match_baseline,
)

pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS

_GSM8K_PLATINUM_SUITE_KWARGS = {
    "variant": "cot",
    "apply_chat_template": True,
    "batch_size": 24,
    "max_new_tokens": 96,
    "streaming": True,
    "max_rows": 128,
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DUAL_ENGINE_SCORE_ABS_TOLERANCE = 3 / 128


def _run_gsm8k_platinum(engine_name: str) -> dict[str, object]:
    script = textwrap.dedent(
        f"""
        import json

        import evalution

        model_path = {str(LLAMA3_2_1B_INSTRUCT)!r}
        device = {LLAMA3_2_TRANSFORMERS_DEVICE!r}
        engine_name = {engine_name!r}
        suite_kwargs = {dict(_GSM8K_PLATINUM_SUITE_KWARGS)!r}

        if engine_name == "transformers":
            engine = evalution.Transformers(
                dtype="bfloat16",
                attn_implementation="paged|flash_attention_2",
                device=device,
                batch_size="auto",
            )
        elif engine_name == "compat":
            engine = evalution.TransformersCompat(
                dtype="bfloat16",
                device=device,
                batch_size="auto",
            )
        else:
            raise ValueError(engine_name)

        result = (
            engine
            .model(evalution.Model(path=model_path))
            .run(evalution.benchmarks.gsm8k_platinum(**suite_kwargs))
            .result()
        )
        test = result.tests[0]
        print("RESULT_JSON_START")
        print(
            json.dumps(
                {{
                    "engine": result.engine,
                    "metrics": test.metrics,
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
        capture_output=True,
        text=True,
        check=True,
    )
    match = pcre.search(
        r"RESULT_JSON_START\n(.*?)\nRESULT_JSON_END",
        completed.stdout,
        pcre.DOTALL,
    )
    assert match is not None, completed.stdout
    return json.loads(match.group(1))


def test_llama3_2_transformers_vs_transformers_compat_gsm8k_platinum_baselines():
    transformers_result = _run_gsm8k_platinum("transformers")
    compat_result = _run_gsm8k_platinum("compat")

    assert transformers_result["test_name"] == "gsm8k_platinum_cot"
    assert compat_result["test_name"] == "gsm8k_platinum_cot"
    assert set(transformers_result["metrics"]) == {"acc,num"}
    assert set(compat_result["metrics"]) == {"acc,num"}
    assert transformers_result["sample_count"] == 128
    assert compat_result["sample_count"] == 128

    assert transformers_result["engine"]["resolved_engine"] == "Transformers"
    assert compat_result["engine"]["resolved_engine"] == "TransformersCompat"
    assert (
        transformers_result["engine"]["execution"]["generation_backend"]
        == "continuous_batching"
    )
    assert compat_result["engine"]["execution"]["generation_backend"] == "generate_compat"
    assert transformers_result["engine"]["execution"]["paged_attention"] is True
    assert compat_result["engine"]["execution"]["paged_attention"] is False

    assert_metrics_match_baseline(
        transformers_result["metrics"],
        {"acc,num": 0.390625},
        abs_tolerance=_DUAL_ENGINE_SCORE_ABS_TOLERANCE,
    )
    assert_metrics_match_baseline(
        compat_result["metrics"],
        {"acc,num": 0.390625},
        abs_tolerance=_DUAL_ENGINE_SCORE_ABS_TOLERANCE,
    )
