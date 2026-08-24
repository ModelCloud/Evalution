# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Docker sandbox helpers for agentic benchmarks.

Agentic suites such as Terminal-Bench and DeepSWE ship tasks as containerized
environments.  The :class:`DockerSandbox` utility provides a thin wrapper
around ``docker run`` so that benchmark scorers can execute generated commands
in an isolated container when Docker is available.
"""

from __future__ import annotations

import dataclasses

import pcre
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

_BASH_TAG_RE = pcre.compile(r"<bash>(.*?)</bash>", pcre.DOTALL | pcre.IGNORECASE)


@dataclasses.dataclass(slots=True)
class DockerRunResult:
    """Result of a command executed inside a Docker container."""

    stdout: str
    stderr: str
    exit_code: int
    command: list[str]
    duration_s: float


class DockerSandbox:
    """Run shell commands inside a throw-away Docker container."""

    def __init__(
        self,
        image: str = "alpine:latest",
        timeout: float = 60.0,
        network: str = "none",
        pull: str = "never",
        shell: str = "sh",
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.network = network
        self.pull = pull
        self.shell = shell

    def run(
        self,
        command: str,
        *,
        image: str | None = None,
        timeout: float | None = None,
        network: str | None = None,
        pull: str | None = None,
        env: Mapping[str, str] | None = None,
        volumes: Mapping[str, str] | None = None,
        workdir: str | None = None,
        shell: str | None = None,
    ) -> DockerRunResult:
        """Execute ``command`` in a fresh container and return its result."""
        image = image or self.image
        timeout = timeout or self.timeout
        network = network or self.network
        pull = pull or self.pull
        shell = shell or self.shell

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--pull",
            pull,
            "--network",
            network,
        ]
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        if volumes:
            for host_path, container_path in volumes.items():
                docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
        if workdir:
            docker_cmd.extend(["-w", workdir])

        docker_cmd.extend([image, shell, "-c", command])

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return DockerRunResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                command=docker_cmd,
                duration_s=time.perf_counter() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return DockerRunResult(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=-1,
                command=docker_cmd,
                duration_s=timeout,
            )


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
