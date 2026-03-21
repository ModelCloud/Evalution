# Evalution

Modern LLM evaluation with a small, explicit runtime API.

Install:

```bash
pip install Evalution
```

Install from source:

```bash
git clone https://github.com/modelcloud/Evalution.git
cd Evalution
pip install .
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

## Citation

If you use Evalution or the built-in `gsm8k_platinum` suite, please cite:

```bibtex
# Evalution
@misc{modelcloud2026evalution,
  author = {ModelCloud and @qubitium},
  title = {Evalution},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/modelcloud/Evalution}},
  note = {Contact: x.com/qubitium},
  year = {2026},
}

# GSM8K-Platinum
@article{vendrow2025largelanguagemodelbenchmarks,
  title = {Do Large Language Model Benchmarks Test Reliability?},
  author = {Joshua Vendrow and Edward Vendrow and Sara Beery and Aleksander Madry},
  journal = {arXiv preprint arXiv:2502.03461},
  year = {2025},
}

# GSM8K
@article{cobbe2021trainingverifierssolvemath,
  title = {Training Verifiers to Solve Math Word Problems},
  author = {Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
  journal = {arXiv preprint arXiv:2110.14168},
  year = {2021},
}
```
