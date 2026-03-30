import os
import sys


if os.environ.get("EVALUTION_DISABLE_FA_VARLEN_CTX_PATCH") == "1":
    try:
        from evalution.engines import transformers as transformers_engine
    except Exception as exc:
        print(
            f"[sitecustomize] failed to import evalution.engines.transformers: {exc!r}",
            file=sys.stderr,
        )
    else:
        def _disabled_flash_attn_varlen_ctx_patch() -> None:
            print(
                "[sitecustomize] disabled evalution flash-attn varlen CUDA-context monkeypatch",
                file=sys.stderr,
            )

        transformers_engine._patch_flash_attn_varlen_fwd_cuda_context_once = _disabled_flash_attn_varlen_ctx_patch
