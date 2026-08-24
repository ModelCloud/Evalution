# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from evalution.agent_runtime import BaseAgentRuntime


@dataclass(slots=True, frozen=True)
class Model:
    # Keep the class-level state explicit for this helper.
    """Serializable model configuration shared by all engine backends."""
    path: str
    label: str | None = None
    tokenizer: Any | None = None
    tokenizer_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for model."""
        return asdict(self)


def coerce_model(model: Model | dict[str, Any]) -> Model:
    """Normalize user-provided model config inputs to one dataclass shape."""
    if isinstance(model, Model):
        return model
    if isinstance(model, dict):
        return Model(**model)
    raise TypeError("model must be an evalution.Model or a plain dict of model options")


def model_with_label(model: Model, *, label: str | None) -> Model:
    """Return a labeled copy without mutating the original model config."""
    if label is None:
        return model
    return replace(model, label=label)


@dataclass(slots=True, frozen=True)
class AgentRuntimeConfig:
    """Select the isolated runtime used by an agentic benchmark.

    ``agent_runtime`` must be a sandboxed :class:`BaseAgentRuntime` such as
    :class:`DockerAgentRuntime` or :class:`SmolVmAgentRuntime`; pass
    :class:`UnsafeLocalRuntime` to explicitly allow unisolated host execution.
    """

    agent_runtime: BaseAgentRuntime | None = None

    @property
    def runtime(self) -> BaseAgentRuntime | None:
        """Return the configured agent runtime."""
        return self.agent_runtime
