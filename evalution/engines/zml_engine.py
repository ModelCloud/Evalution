# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ZML/LLMD inference engine integration.

LLMD is distributed as a standalone server rather than a Python runtime.  This
adapter deliberately uses its OpenAI-compatible API so the ZML scheduler and
attention implementation stay in the server process.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from evalution.config import Model
from evalution.engines.base import BaseInferenceSession
from evalution.engines.openai_engine import OpenAICompatible, OpenAICompatibleSession

# Keep the LLMD process and health-check defaults explicit at module scope.
_DEFAULT_LLMD_EXECUTABLE = "llmd"
_DEFAULT_HEALTH_PATH = "/v1/models"
_DEFAULT_STARTUP_TIMEOUT_S = 120.0
_DEFAULT_HEALTH_INTERVAL_S = 0.25


@dataclass(slots=True)
class ZML(OpenAICompatible):
    """Configure Evalution to benchmark a ZML/LLMD server.

    LLMD enables its optimized serving path in the server: continuous batching,
    paged attention, prefix caching, automatic tensor/expert sharding, and the
    platform-appropriate attention kernels (including FlashAttention 2/3 on
    supported CUDA devices).  ``batch_size`` and ``max_parallel_requests``
    control Evalution's client-side admission pressure so those server features
    can observe concurrent benchmark requests.

    Set ``launch_server=True`` to start a local ``llmd`` executable for the
    model passed to ``.model(...)``.  Containerized deployments can instead
    provide a complete ``server_command`` such as a ``docker run`` command.
    ``server_args`` is intentionally passed through unchanged for LLMD releases
    that add further runtime controls.
    """

    # LLMD's current documented serving path is OpenAI-compatible and supports
    # up to a substantially wider native batch than Evalution's conservative
    # HTTP default.  Users can raise either limit explicitly for their device.
    batch_size: int = 16
    max_parallel_requests: int = 64

    # Report the server-side capabilities in execution metadata.  These are
    # enabled by LLMD itself rather than translated into undocumented flags.
    continuous_batching: bool = True
    paged_attention: bool = True
    prefix_caching: bool = True
    attention_backend: str = "auto"
    tensor_parallel_size: int | str = "auto"

    # DFlash is the one documented optional generation accelerator exposed as a
    # server launch flag.  It is not applicable to the Llama 3.2 smoke test but
    # remains available for supported model families.
    dflash_model: str | None = None

    # Optional local process management.  With ``server_command`` set, the
    # command is treated as complete and is not modified except for environment
    # and lifecycle handling.
    launch_server: bool = False
    executable: str = _DEFAULT_LLMD_EXECUTABLE
    server_command: list[str] | None = None
    server_args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    health_path: str = _DEFAULT_HEALTH_PATH
    startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S
    health_interval_s: float = _DEFAULT_HEALTH_INTERVAL_S

    def build(self, model: Model) -> BaseInferenceSession:
        """Construct a ZML/LLMD-backed inference session."""

        self.resolved_engine = "ZML"
        return ZMLSession.from_config(self, model)


@dataclass(slots=True)
class ZMLSession(OpenAICompatibleSession):
    """Drive Evalution requests through one ZML/LLMD HTTP server."""

    config: ZML
    _server_process: subprocess.Popen[Any] | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: ZML, model_config: Model) -> ZMLSession:
        """Create a session and optionally own the LLMD server process."""

        model_name = config.model_name or Path(model_config.path).name or model_config.path
        server_process: subprocess.Popen[Any] | None = None
        if config.launch_server:
            server_process = _launch_server(config, model_config)
            try:
                _wait_for_server(config, server_process=server_process)
            except Exception:
                _terminate_process(server_process)
                raise
        return cls(
            config=config,
            model_config=model_config,
            model_name=model_name,
            _server_process=server_process,
        )

    def describe_execution(self) -> dict[str, Any]:
        """Expose ZML serving and client-admission settings in results."""

        return {
            "generation_backend": "zml_llmd_openai_http",
            "base_url": self.config.base_url.rstrip("/"),
            "model_name": self.model_name,
            "batch_size": self.config.batch_size,
            "max_parallel_requests": self.config.max_parallel_requests,
            "continuous_batching": self.config.continuous_batching,
            "paged_attention": self.config.paged_attention,
            "prefix_caching": self.config.prefix_caching,
            "attention_backend": self.config.attention_backend,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "dflash_model": self.config.dflash_model,
            "server_managed": self._server_process is not None,
        }

    def gc(self) -> None:
        """Leave cache lifetime to LLMD, which owns the compiled runtime."""

        return

    def _effective_batch_size(self, batch_size: int | None) -> int:
        """Cap client refill concurrency before requests enter LLMD's native scheduler."""

        configured = super()._effective_batch_size(batch_size)
        limit = int(self.config.max_parallel_requests)
        if limit <= 0:
            return 1
        return min(configured, limit)

    def close(self) -> None:
        """Close the HTTP session and stop a server started by this session."""

        super().close()
        process = self._server_process
        self._server_process = None
        if process is not None:
            _terminate_process(process)


def _launch_server(config: ZML, model_config: Model) -> subprocess.Popen[Any]:
    """Start LLMD using either a complete command or the local executable form."""

    if config.server_command:
        command = list(config.server_command)
    else:
        command = [config.executable, f"--model={model_config.path}"]
        if config.dflash_model is not None:
            command.append(f"--dflash-model={config.dflash_model}")
        command.extend(config.server_args)

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in config.environment.items()})
    try:
        return subprocess.Popen(command, env=environment)
    except FileNotFoundError as exc:
        command_name = command[0] if command else config.executable
        raise RuntimeError(
            f"ZML engine could not start {command_name!r}; install LLMD or provide "
            "server_command=['docker', 'run', ...]"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"ZML engine failed to start LLMD command {command!r}: {exc}") from exc


def _wait_for_server(
    config: ZML,
    *,
    server_process: subprocess.Popen[Any] | None = None,
) -> None:
    """Wait until LLMD's documented model route accepts requests."""

    deadline = time.monotonic() + max(float(config.startup_timeout_s), 0.0)
    endpoint = f"{config.base_url.rstrip('/')}/{config.health_path.lstrip('/')}"
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        if server_process is not None and server_process.poll() is not None:
            raise RuntimeError(
                f"ZML/LLMD exited before becoming ready with return code "
                f"{server_process.returncode}"
            )
        try:
            with request.urlopen(endpoint, timeout=max(float(config.health_interval_s), 0.1)) as response:
                if 200 <= response.status < 300:
                    return
                last_error = RuntimeError(f"health route returned HTTP {response.status}")
        except (OSError, error.URLError, error.HTTPError) as exc:
            last_error = exc
        time.sleep(max(float(config.health_interval_s), 0.01))
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(
        f"ZML/LLMD did not become ready at {endpoint!r} within "
        f"{config.startup_timeout_s:.1f}s{detail}"
    )


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Terminate a managed LLMD process without masking the original error."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


__all__ = ["ZML", "ZMLSession"]
