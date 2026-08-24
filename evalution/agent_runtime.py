# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Isolated execution runtimes for agentic benchmark workloads.

Agentic benchmarks execute model-generated commands, so every tool-calling
suite must run them through a sandboxed :class:`BaseAgentRuntime` such as
:class:`DockerAgentRuntime` or :class:`SmolVmAgentRuntime`.  Running on the
bare host is only allowed through the explicit, warning-emitting
:class:`UnsafeLocalRuntime` escape hatch.

Every runtime shares two common settings: ``path`` selects the runtime
binary (``"auto"`` resolves through the current environment ``PATH``) and
``image`` selects the default execution image.
"""

from __future__ import annotations

import dataclasses
import subprocess
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping

AUTO_PATH = "auto"


@dataclasses.dataclass(slots=True)
class AgentRuntimeResult:
    """Result of one command executed by an agent runtime."""

    stdout: str
    stderr: str
    exit_code: int
    command: list[str]
    duration_s: float


def _resolve_path(path: str, default_binary: str) -> str:
    """Resolve ``path="auto"`` to the runtime's default binary name."""
    return default_binary if path == AUTO_PATH else path


class BaseAgentRuntime(ABC):
    """Common configuration and interface for isolated agent execution.

    ``path`` locates the runtime CLI binary; the default ``"auto"`` uses the
    binary name resolved from the current configured bin environment.
    ``image`` is the default image used when a call does not override it.
    """

    DEFAULT_BINARY: str = ""

    def __init__(
        self,
        *,
        path: str = AUTO_PATH,
        image: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.path = path
        self.image = image
        self.timeout = timeout

    @property
    def resolved_path(self) -> str:
        """Return the concrete runtime binary used for execution."""
        return _resolve_path(self.path, self.DEFAULT_BINARY)

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        image: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        volumes: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentRuntimeResult:
        """Run ``command`` inside the isolated runtime."""
        raise NotImplementedError


class DockerAgentRuntime(BaseAgentRuntime):
    """Run agent commands in disposable Docker containers."""

    DEFAULT_BINARY = "docker"
    DEFAULT_IMAGE = "alpine:latest"

    def __init__(
        self,
        *,
        path: str = AUTO_PATH,
        image: str | None = None,
        timeout: float = 60.0,
        network: str = "none",
        pull: str = "never",
        shell: str = "sh",
    ) -> None:
        super().__init__(path=path, image=image, timeout=timeout)
        self.network = network
        self.pull = pull
        self.shell = shell

    def run(
        self,
        command: str,
        *,
        image: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        volumes: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentRuntimeResult:
        """Run ``command`` in a fresh container."""
        resolved_image = image or self.image or self.DEFAULT_IMAGE
        resolved_timeout = self.timeout if timeout is None else timeout
        docker_cmd = [
            self.resolved_path,
            "run",
            "--rm",
            "-i",
            "--pull",
            self.pull,
            "--network",
            self.network,
        ]
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        if volumes:
            for host_path, container_path in volumes.items():
                docker_cmd.extend(["-v", f"{host_path}:{container_path}"])
        if workdir:
            docker_cmd.extend(["-w", workdir])
        docker_cmd.extend([resolved_image, self.shell, "-c", command])
        return _run_process(docker_cmd, timeout=resolved_timeout)


class SmolVmAgentRuntime(BaseAgentRuntime):
    """Run agent commands in disposable smolvm microVMs.

    See https://github.com/smol-machines/smolvm; each command boots an
    ephemeral hardware-isolated VM that is removed after exit.
    """

    DEFAULT_BINARY = "smolvm"
    DEFAULT_IMAGE = "alpine"

    def __init__(
        self,
        *,
        path: str = AUTO_PATH,
        image: str | None = None,
        timeout: float = 60.0,
        network: bool = False,
        cpus: int | None = None,
        memory_mib: int | None = None,
        shell: str = "sh",
    ) -> None:
        super().__init__(path=path, image=image, timeout=timeout)
        self.network = network
        self.cpus = cpus
        self.memory_mib = memory_mib
        self.shell = shell

    def run(
        self,
        command: str,
        *,
        image: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        volumes: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentRuntimeResult:
        """Run ``command`` in an ephemeral smolvm machine."""
        resolved_image = image or self.image or self.DEFAULT_IMAGE
        resolved_timeout = self.timeout if timeout is None else timeout
        smolvm_cmd = [self.resolved_path, "machine", "run"]
        if self.network:
            smolvm_cmd.append("--net")
        if self.cpus is not None:
            smolvm_cmd.extend(["--cpus", str(self.cpus)])
        if self.memory_mib is not None:
            smolvm_cmd.extend(["--mem", str(self.memory_mib)])
        smolvm_cmd.extend(["--image", resolved_image])
        if env:
            for key, value in env.items():
                smolvm_cmd.extend(["--env", f"{key}={value}"])
        if volumes:
            for host_path, container_path in volumes.items():
                smolvm_cmd.extend(["--volume", f"{host_path}:{container_path}"])
        if workdir:
            smolvm_cmd.extend(["--workdir", workdir])
        smolvm_cmd.extend(["--", self.shell, "-c", command])
        return _run_process(smolvm_cmd, timeout=resolved_timeout)


class UnsafeLocalRuntime(BaseAgentRuntime):
    """Run agent commands directly on the host without any isolation."""

    DEFAULT_BINARY = "sh"

    def __init__(self, *, shell: str = "sh", timeout: float = 60.0) -> None:
        super().__init__(path=AUTO_PATH, image=None, timeout=timeout)
        self.shell = shell
        warnings.warn(
            "UnsafeLocalRuntime executes agent commands directly on the host "
            "without sandboxing; only use it for fully trusted workloads.",
            RuntimeWarning,
            stacklevel=2,
        )

    def run(
        self,
        command: str,
        *,
        image: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        volumes: Mapping[str, str] | None = None,
        workdir: str | None = None,
    ) -> AgentRuntimeResult:
        """Run ``command`` on the host; isolation options are ignored."""
        del image, env, volumes, workdir
        resolved_timeout = self.timeout if timeout is None else timeout
        return _run_process([self.shell, "-c", command], timeout=resolved_timeout)


def _run_process(command: list[str], *, timeout: float) -> AgentRuntimeResult:
    """Execute a runtime CLI command and normalize timeout results."""
    start = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return AgentRuntimeResult(
            stdout=process.stdout,
            stderr=process.stderr,
            exit_code=process.returncode,
            command=command,
            duration_s=time.perf_counter() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return AgentRuntimeResult(
            stdout=_text_output(exc.stdout),
            stderr=_text_output(exc.stderr),
            exit_code=-1,
            command=command,
            duration_s=timeout,
        )


def _text_output(value: str | bytes | None) -> str:
    """Normalize subprocess output, including timeout byte strings."""
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
