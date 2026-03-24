# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import tokenicer
from types import SimpleNamespace

from evalution.engines import transformers_common as common


def test_tokenicer_is_imported_from_installed_package() -> None:
    assert common.Tokenicer is tokenicer.Tokenicer


def test_load_tokenizer_from_model_uses_tokenicer_load(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeTokenicer:
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            observed["path"] = path
            observed["strict"] = strict
            observed["kwargs"] = kwargs
            return "tokenizer"

    monkeypatch.setattr(common, "Tokenicer", FakeTokenicer)

    tokenizer = common._load_tokenizer_from_model(
        "model/path",
        trust_remote_code=True,
        revision="main",
    )

    assert tokenizer == "tokenizer"
    assert observed["path"] == "model/path"
    assert observed["strict"] is False
    assert observed["kwargs"] == {"trust_remote_code": True, "revision": "main"}


def test_normalize_tokenizer_special_tokens_calls_auto_fix_with_model_when_present() -> None:
    calls: list[tuple[object | None, bool, object]] = []
    model = SimpleNamespace()

    class FakeTokenizer:
        def auto_fix_pad_token(self, model=None, strict: bool = True, pad_tokens: object = None) -> None:
            calls.append((model, strict, pad_tokens))

    common._normalize_tokenizer_special_tokens(tokenizer=FakeTokenizer(), model=model)

    assert calls == [(model, False, None)]


def test_normalize_tokenizer_special_tokens_calls_auto_fix_without_model_when_absent() -> None:
    calls: list[tuple[object | None, bool]] = []

    class FakeTokenizer:
        def auto_fix_pad_token(self, strict: bool = True) -> None:
            calls.append((None, strict))

    common._normalize_tokenizer_special_tokens(tokenizer=FakeTokenizer(), model=None)

    assert calls == [(None, False)]
