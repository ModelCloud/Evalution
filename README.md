# Evalution

Modern LLM evaluation with a small, explicit runtime API.

Install:

```bash
pip install Evalution
```

Runtime dependencies include `transformers`, `datasets`, `logbar`, and `PyPcre`.

Simple usage:

```python
import evalution

result = evalution.run(
    model={"path": "/monster/data/model/Llama-3.2-1B-Instruct"},
    engine=evalution.Transformer(),
    tests=[evalution.gsm8k_platinum()],
)
```

Advanced usage:

```python
import evalution

result = evalution.run(
    model=evalution.Model(
        path="/monster/data/model/Llama-3.2-1B-Instruct",
    ),
    engine=evalution.Transformer(
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        device="cuda:0",
        batch_size="auto",
        paged_attention="auto",
        max_new_tokens=256,
    ),
    tests=[
        evalution.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            max_new_tokens=96,
            batch_size=64,
            limit=128,
        ),
    ],
)
```

`Transformer()` defaults to auto behavior for batching, paged attention, dtype resolution, and
attention selection. If you do not set `attn_implementation`, the backend leaves attention
selection on its default auto behavior.

Evalution chooses a per-suite batch size from the suite row count, rendered prompt token lengths,
GPU VRAM, and dtype. On compatible CUDA `transformers` models it also switches to paged continuous
batching automatically for `flash_attention_2`, using `paged|flash_attention_2` and logging the
effective attention backend before execution. Callers can still force paged attention on other
supported backends with `paged_attention=True`, force plain static generation with
`paged_attention=False`, or pin an explicit attention implementation with
`attn_implementation=...`. A suite can override engine batch sizing with
`gsm8k_platinum(batch_size=...)`.

For `gsm8k_platinum`, answer extraction and normalization use precompiled `pcre.compile(...)`
patterns so regex work stays on the `PyPcre` path and avoids stdlib `re`.

Current built-in coverage:

- Hugging Face `transformers` inference engine
- `gsm8k_platinum` suite ported from `lm-eval`
- `logbar`-powered runtime logging and evaluation progress bars
