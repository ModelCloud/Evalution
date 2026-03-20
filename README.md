# Evalution

Modern LLM evaluation with a small, explicit runtime API.

```python
import evalution

result = evalution.run(
    model=evalution.Model(
        path="/monster/data/model/Llama-3.2-1B-Instruct",
    ),
    engine=evalution.Transformer(
        dtype="bfloat16",
        attn_implementation="sdpa",
        device="cuda:0",
        batch_size="auto",
        paged_attention="auto",
    ),
    tests=[
        evalution.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            limit=8,
        ),
    ],
)
```

`Transformer(batch_size="auto", paged_attention="auto")` is the default. Evalution will choose a
per-suite batch size from the suite row count, rendered prompt token lengths, GPU VRAM, and dtype.
On compatible CUDA `transformers` models it will also switch to paged continuous batching
automatically for `flash_attention_2`, using `paged|flash_attention_2` and logging the effective
attention backend before execution. Callers can still force paged attention on other supported
backends with `paged_attention=True`, or force plain static generation with `paged_attention=False`.
A suite can still override batch sizing with `gsm8k_platinum(batch_size=...)`.

Current built-in coverage:

- Hugging Face `transformers` inference engine
- `gsm8k_platinum` suite ported from `lm-eval`
- `logbar`-powered runtime logging and evaluation progress bars
