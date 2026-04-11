# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import tokenicer
from types import SimpleNamespace

import pytest

from evalution.engines import transformers_common as common


def test_tokenicer_is_imported_from_installed_package() -> None:
    """Verify tokenicer is imported from installed package."""
    assert common.Tokenicer is tokenicer.Tokenicer


def test_load_tokenizer_from_model_uses_tokenicer_load(monkeypatch) -> None:
    """Verify load tokenizer from model uses tokenicer load."""
    observed: dict[str, object] = {}

    class FakeTokenicer:
        """Provide the fake tokenicer helper used by the surrounding tests."""
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            """Load load."""
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


def test_load_tokenizer_from_model_retries_direct_local_gguf_file(monkeypatch, tmp_path) -> None:
    """Verify load tokenizer from model retries direct local gguf file."""
    calls: list[tuple[str, bool, dict[str, object]]] = []
    gguf_path = tmp_path / "Bonsai-1.7B.gguf"
    gguf_path.write_bytes(b"GGUF")

    class FakeTokenicer:
        """Provide the fake tokenicer helper used by the surrounding tests."""
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            """Load load."""
            calls.append((path, strict, dict(kwargs)))
            if len(calls) == 1:
                raise ValueError("missing gguf_file")
            return "tokenizer"

    monkeypatch.setattr(common, "Tokenicer", FakeTokenicer)

    tokenizer = common._load_tokenizer_from_model(str(gguf_path))

    assert tokenizer == "tokenizer"
    assert calls == [
        (str(gguf_path), False, {}),
        (str(tmp_path), False, {"gguf_file": "Bonsai-1.7B.gguf"}),
    ]


def test_load_tokenizer_from_model_retries_single_gguf_directory(monkeypatch, tmp_path) -> None:
    """Verify load tokenizer from model retries single gguf directory."""
    calls: list[tuple[str, bool, dict[str, object]]] = []
    (tmp_path / "Bonsai-1.7B.gguf").write_bytes(b"GGUF")
    (tmp_path / "README.md").write_text("docs")

    class FakeTokenicer:
        """Provide the fake tokenicer helper used by the surrounding tests."""
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            """Load load."""
            calls.append((path, strict, dict(kwargs)))
            if len(calls) == 1:
                raise ValueError("missing gguf_file")
            return "tokenizer"

    monkeypatch.setattr(common, "Tokenicer", FakeTokenicer)

    tokenizer = common._load_tokenizer_from_model(str(tmp_path))

    assert tokenizer == "tokenizer"
    assert calls == [
        (str(tmp_path), False, {}),
        (str(tmp_path), False, {"gguf_file": "Bonsai-1.7B.gguf"}),
    ]


def test_load_tokenizer_from_model_retries_single_gguf_hub_repo(monkeypatch) -> None:
    """Verify load tokenizer from model retries single gguf hub repo."""
    calls: list[tuple[str, bool, dict[str, object]]] = []

    class FakeTokenicer:
        """Provide the fake tokenicer helper used by the surrounding tests."""
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            """Load load."""
            calls.append((path, strict, dict(kwargs)))
            if len(calls) == 1:
                raise ValueError("missing gguf_file")
            return "tokenizer"

    monkeypatch.setattr(common, "Tokenicer", FakeTokenicer)
    monkeypatch.setattr(
        common,
        "_list_hub_repo_entries",
        lambda repo_id, *, revision: [
            "Bonsai-1.7B.gguf",
            "README.md",
            "assets/hero.png",
        ],
    )

    tokenizer = common._load_tokenizer_from_model(
        "prism-ml/Bonsai-1.7B-gguf",
        revision="main",
    )

    assert tokenizer == "tokenizer"
    assert calls == [
        ("prism-ml/Bonsai-1.7B-gguf", False, {"revision": "main"}),
        (
            "prism-ml/Bonsai-1.7B-gguf",
            False,
            {"revision": "main", "gguf_file": "Bonsai-1.7B.gguf"},
        ),
    ]


def test_load_tokenizer_from_model_does_not_retry_when_dense_weights_are_present(monkeypatch) -> None:
    """Verify load tokenizer from model does not retry when dense weights are present."""
    calls: list[tuple[str, bool, dict[str, object]]] = []

    class FakeTokenicer:
        """Provide the fake tokenicer helper used by the surrounding tests."""
        @classmethod
        def load(cls, path: str, strict: bool = False, **kwargs: object) -> object:
            """Load load."""
            calls.append((path, strict, dict(kwargs)))
            raise RuntimeError("original failure")

    monkeypatch.setattr(common, "Tokenicer", FakeTokenicer)
    monkeypatch.setattr(
        common,
        "_list_hub_repo_entries",
        lambda repo_id, *, revision: [
            "Bonsai-1.7B.gguf",
            "model.safetensors",
        ],
    )

    with pytest.raises(RuntimeError, match="original failure"):
        common._load_tokenizer_from_model("prism-ml/Bonsai-1.7B-mixed", revision="main")

    assert calls == [("prism-ml/Bonsai-1.7B-mixed", False, {"revision": "main"})]


def test_normalize_tokenizer_special_tokens_calls_auto_fix_with_model_when_present() -> None:
    """Verify normalize tokenizer special tokens calls auto fix with model when present. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    calls: list[tuple[object | None, bool, object]] = []
    model = SimpleNamespace()

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        def auto_fix_pad_token(self, model=None, strict: bool = True, pad_tokens: object = None) -> None:
            """Implement auto fix pad token for fake tokenizer."""
            calls.append((model, strict, pad_tokens))

    common._normalize_tokenizer_special_tokens(tokenizer=FakeTokenizer(), model=model)

    assert calls == [(model, False, None)]


def test_normalize_tokenizer_special_tokens_calls_auto_fix_without_model_when_absent() -> None:
    """Verify normalize tokenizer special tokens calls auto fix without model when absent. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    calls: list[tuple[object | None, bool]] = []

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        def auto_fix_pad_token(self, strict: bool = True) -> None:
            """Implement auto fix pad token for fake tokenizer."""
            calls.append((None, strict))

    common._normalize_tokenizer_special_tokens(tokenizer=FakeTokenizer(), model=None)

    assert calls == [(None, False)]
