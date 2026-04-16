# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import pcre
import pytest

import evalution


@pytest.mark.parametrize(
    ("factory_name", "benchmark_name"),
    [
        ("swe_bench_verified", "SWE-bench Verified"),
        ("swe_bench_multilingual", "SWE-bench Multilingual"),
        ("swe_bench_pro", "SWE-bench Pro"),
        ("terminal_bench_2", "Terminal-Bench 2.0"),
        ("claw_eval_avg", "Claw-Eval Avg"),
        ("claw_eval_pass3", "Claw-Eval Pass^3"),
        ("skillsbench_avg5", "SkillsBench Avg5"),
        ("qwenclawbench", "QwenClawBench"),
        ("nl2repo", "NL2Repo"),
        ("qwenwebbench", "QwenWebBench"),
        ("tau3_bench", "TAU3-Bench"),
        ("vita_bench", "VITA-Bench"),
        ("deepplanning", "DeepPlanning"),
        ("tool_decathlon", "Tool Decathlon"),
        ("mcpmark", "MCPMark"),
        ("mcp_atlas", "MCP-Atlas"),
        ("widesearch", "WideSearch"),
    ],
)
def test_capability_gated_benchmarks_raise_clear_runtime_errors(
    factory_name: str,
    benchmark_name: str,
) -> None:
    """Verify capability-gated benchmarks fail with explicit unsupported-runtime errors."""
    suite = getattr(evalution.benchmarks, factory_name)()

    with pytest.raises(NotImplementedError, match=pcre.escape(benchmark_name)):
        suite.evaluate(object())
