# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalution.benchmarks.base import TestSuite

# Keep benchmark defaults and public task ids explicit at module scope.
CAPABILITY_GATED_TASKS = (
    "swe_bench_verified",
    "swe_bench_multilingual",
    "swe_bench_pro",
    "terminal_bench_2",
    "claw_eval_avg",
    "claw_eval_pass3",
    "skillsbench_avg5",
    "qwenclawbench",
    "nl2repo",
    "qwenwebbench",
    "tau3_bench",
    "vita_bench",
    "deepplanning",
    "tool_decathlon",
    "mcpmark",
    "mcp_atlas",
    "widesearch",
)
_CAPABILITY_GATED_BENCHMARKS = {
    "swe_bench_verified": {
        "benchmark_name": "SWE-bench Verified",
        "required_capabilities": (
            "repository patch execution",
            "project-specific test harness execution",
        ),
    },
    "swe_bench_multilingual": {
        "benchmark_name": "SWE-bench Multilingual",
        "required_capabilities": (
            "repository patch execution",
            "project-specific multilingual test harness execution",
        ),
    },
    "swe_bench_pro": {
        "benchmark_name": "SWE-bench Pro",
        "required_capabilities": (
            "repository patch execution",
            "project-specific test harness execution",
        ),
    },
    "terminal_bench_2": {
        "benchmark_name": "Terminal-Bench 2.0",
        "required_capabilities": ("interactive terminal runtime",),
    },
    "claw_eval_avg": {
        "benchmark_name": "Claw-Eval Avg",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "claw_eval_pass3": {
        "benchmark_name": "Claw-Eval Pass^3",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "skillsbench_avg5": {
        "benchmark_name": "SkillsBench Avg5",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "qwenclawbench": {
        "benchmark_name": "QwenClawBench",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "nl2repo": {
        "benchmark_name": "NL2Repo",
        "required_capabilities": (
            "repository patch execution",
            "project-specific test harness execution",
        ),
    },
    "qwenwebbench": {
        "benchmark_name": "QwenWebBench",
        "required_capabilities": ("web browsing runtime",),
    },
    "tau3_bench": {
        "benchmark_name": "TAU3-Bench",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "vita_bench": {
        "benchmark_name": "VITA-Bench",
        "required_capabilities": ("multimodal vision runtime",),
    },
    "deepplanning": {
        "benchmark_name": "DeepPlanning",
        "required_capabilities": ("agentic planning runtime",),
    },
    "tool_decathlon": {
        "benchmark_name": "Tool Decathlon",
        "required_capabilities": ("agentic tool-use runtime",),
    },
    "mcpmark": {
        "benchmark_name": "MCPMark",
        "required_capabilities": ("MCP client and tool-use runtime",),
    },
    "mcp_atlas": {
        "benchmark_name": "MCP-Atlas",
        "required_capabilities": ("MCP client and tool-use runtime",),
    },
    "widesearch": {
        "benchmark_name": "WideSearch",
        "required_capabilities": ("web search runtime",),
    },
}


@dataclass(slots=True)
class CapabilityGatedSuite(TestSuite):
    """Implement the capability-gated benchmark suite placeholder."""
    # Keep the class-level state explicit so unsupported benchmark registrations stay auditable.
    benchmark_name: str
    task_id: str
    required_capabilities: tuple[str, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, session: Any) -> Any:
        """Raise a clear capability error for benchmarks that need unsupported runtimes."""
        del session
        capabilities = ", ".join(self.required_capabilities)
        raise NotImplementedError(
            f"{self.benchmark_name} is registered in Evalution as `{self.task_id}`, "
            f"but it requires unsupported runtime capabilities: {capabilities}."
        )


def _capability_gated(task_id: str, **kwargs: Any) -> CapabilityGatedSuite:
    """Implement capability gated factory for this module."""
    spec = _CAPABILITY_GATED_BENCHMARKS[task_id]
    return CapabilityGatedSuite(
        benchmark_name=spec["benchmark_name"],
        task_id=task_id,
        required_capabilities=tuple(spec["required_capabilities"]),
        config=dict(kwargs),
    )


def swe_bench_verified(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement swe_bench_verified for this module."""
    return _capability_gated("swe_bench_verified", **kwargs)


def swe_bench_multilingual(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement swe_bench_multilingual for this module."""
    return _capability_gated("swe_bench_multilingual", **kwargs)


def swe_bench_pro(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement swe_bench_pro for this module."""
    return _capability_gated("swe_bench_pro", **kwargs)


def terminal_bench_2(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement terminal_bench_2 for this module."""
    return _capability_gated("terminal_bench_2", **kwargs)


def claw_eval_avg(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement claw_eval_avg for this module."""
    return _capability_gated("claw_eval_avg", **kwargs)


def claw_eval_pass3(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement claw_eval_pass3 for this module."""
    return _capability_gated("claw_eval_pass3", **kwargs)


def skillsbench_avg5(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement skillsbench_avg5 for this module."""
    return _capability_gated("skillsbench_avg5", **kwargs)


def qwenclawbench(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement qwenclawbench for this module."""
    return _capability_gated("qwenclawbench", **kwargs)


def nl2repo(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement nl2repo for this module."""
    return _capability_gated("nl2repo", **kwargs)


def qwenwebbench(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement qwenwebbench for this module."""
    return _capability_gated("qwenwebbench", **kwargs)


def tau3_bench(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement tau3_bench for this module."""
    return _capability_gated("tau3_bench", **kwargs)


def vita_bench(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement vita_bench for this module."""
    return _capability_gated("vita_bench", **kwargs)


def deepplanning(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement deepplanning for this module."""
    return _capability_gated("deepplanning", **kwargs)


def tool_decathlon(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement tool_decathlon for this module."""
    return _capability_gated("tool_decathlon", **kwargs)


def mcpmark(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement mcpmark for this module."""
    return _capability_gated("mcpmark", **kwargs)


def mcp_atlas(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement mcp_atlas for this module."""
    return _capability_gated("mcp_atlas", **kwargs)


def widesearch(**kwargs: Any) -> CapabilityGatedSuite:
    """Implement widesearch for this module."""
    return _capability_gated("widesearch", **kwargs)
