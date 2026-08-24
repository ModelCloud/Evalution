# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Command extraction helpers for agentic benchmarks.

Agentic suites such as Terminal-Bench and DeepSWE execute model-generated
commands through an isolated :mod:`evalution.agent_runtime`; this module only
parses the raw generation into the shell command to run.
"""

from __future__ import annotations

import pcre

_BASH_TAG_RE = pcre.compile(r"<bash>(.*?)</bash>", pcre.DOTALL | pcre.IGNORECASE)


def extract_command(text: str) -> str:
    """Pull a shell command out of a model generation, stripping code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    bash_match = _BASH_TAG_RE.search(text)
    if bash_match:
        return bash_match.group(1).strip()

    return text
