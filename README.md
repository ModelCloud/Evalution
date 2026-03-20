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

`Transformer(batch_size="auto")` is the default. Evalution will choose a per-suite batch size from
the suite row count, rendered prompt token lengths, GPU VRAM, and dtype. A suite can still override
that with `gsm8k_platinum(batch_size=...)` when it needs a smaller or larger batch than the engine
default.

Current built-in coverage:

- Hugging Face `transformers` inference engine
- `gsm8k_platinum` suite ported from `lm-eval`
- `logbar`-powered runtime logging and evaluation progress bars
