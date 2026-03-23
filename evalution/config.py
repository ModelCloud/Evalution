# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(slots=True, frozen=True)
class Model:
    path: str
    label: str | None = None
    tokenizer_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_model(model: Model | dict[str, Any]) -> Model:
    if isinstance(model, Model):
        return model
    if isinstance(model, dict):
        return Model(**model)
    raise TypeError("model must be an evalution.Model or a plain dict of model options")


def model_with_label(model: Model, *, label: str | None) -> Model:
    if label is None:
        return model
    return replace(model, label=label)
