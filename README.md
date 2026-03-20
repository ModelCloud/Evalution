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
        batch_size=1,
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

Current built-in coverage:

- Hugging Face `transformers` inference engine
- `gsm8k_platinum` suite ported from `lm-eval`
