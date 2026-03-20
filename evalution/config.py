from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Model:
    path: str
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
