from __future__ import annotations

from evalution.engines.transformers_common import _loader_import_compat_fallback


def test_transformer_common_loader_retries_without_attention_backend_on_flash_attn_import_error() -> None:
    # A broken flash-attn wheel should fail loudly so the environment mismatch is visible.
    fallback_kwargs, retry_action = _loader_import_compat_fallback(
        {
            "attn_implementation": "flash_attention_2",
            "dtype": "bfloat16",
        },
        ImportError("flash_attn_2_cuda.so: undefined symbol: some_cuda_symbol"),
    )

    assert fallback_kwargs is None
    assert retry_action is None


def test_transformer_common_loader_keeps_attention_backend_for_unrelated_import_errors() -> None:
    fallback_kwargs, retry_action = _loader_import_compat_fallback(
        {
            "attn_implementation": "flash_attention_2",
            "dtype": "bfloat16",
        },
        ImportError("some unrelated import failure"),
    )

    assert fallback_kwargs is None
    assert retry_action is None
