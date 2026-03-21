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
import evalution as eval

result = (
    eval.engine(eval.Transformers())
    .model({"path": "/monster/data/model/Llama-3.2-1B-Instruct"})
    .run(eval.gsm8k_platinum())
    .run(eval.boolq())
    .run(eval.cb())
    .run(eval.copa())
    .run(eval.arc_easy())
    .run(eval.arc_challenge())
    .run(eval.hellaswag())
    .run(eval.mmlu(subject="abstract_algebra", num_fewshot=5))
    .run(eval.openbookqa())
    .run(eval.piqa())
    .run(eval.rte())
    .run(eval.sst2())
    .run(eval.wic())
    .run(eval.winogrande())
)
```

Advanced usage:

```python
import evalution as eval

result = (
    eval.engine(
        eval.Transformers(
            dtype="bfloat16",
            attn_implementation="flash_attention_2",
            device="cuda:0",
            batch_size="auto",
            paged_attention="auto",
            allow_block_sharing=True,
            use_async_batching=None,
            max_new_tokens=256,
        )
    )
    .model(
        eval.Model(
            path="/monster/data/model/Llama-3.2-1B-Instruct",
        )
    )
    .run(
        eval.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            max_new_tokens=96,
            batch_size=64,
            max_rows=128,
        )
    )
    .run(
        eval.arc_challenge(
            apply_chat_template=True,
            max_new_tokens=8,
            max_rows=128,
        )
    )
)
```

The chained object is already the completed run handle. Accessing `result.model`, `result.engine`,
`result.tests`, or `result.to_dict()` finalizes the run and closes the engine session implicitly.

YAML usage:

```yaml
engine:
  type: transformers
  dtype: bfloat16
  attn_implementation: flash_attention_2
  device: cuda:0
  paged_attention: true

model:
  path: /monster/data/model/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
    variant: cot
    apply_chat_template: true
    max_new_tokens: 96
    batch_size: 64
    max_rows: 128
  - type: arc_challenge
    apply_chat_template: true
    max_new_tokens: 8
    max_rows: 128
```

```python
import evalution as eval

result = eval.run_yaml("evalution.yaml")

python_script = eval.python_from_yaml("evalution.yaml")
```

CLI usage:

```bash
evalution evalution.yaml
evalution run evalution.yaml
evalution run evalution.yaml --output result.json
evalution emit-python evalution.yaml
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

For multiple-choice suites such as `hellaswag` and `piqa`, Evalution scores every option with
token-level log-likelihood and reports both raw-choice accuracy and length-normalized accuracy.

For `transformers` continuous batching, `Transformer(...)` also exposes the upstream manager knobs
`manual_eviction`, `allow_block_sharing`, `use_async_batching`, `q_padding_interval_size`,
`kv_padding_interval_size`, and `max_cached_graphs`. Evalution keeps a session-owned continuous
batching manager alive while stop strings and sampling settings stay compatible, then tears it down
on `gc()` between suites or on `close()`.

For `gsm8k_platinum`, answer extraction and normalization use precompiled `pcre.compile(...)`
patterns so regex work stays on the `PyPcre` path and avoids stdlib `re`.

For `arc_challenge`, prompts ask for a single multiple-choice label and scoring first extracts a
choice label, then falls back to an exact choice-text match when the model returns the option text
instead of the label.

Current built-in coverage:

- Hugging Face `transformers` inference engine
- `arc_challenge` suite for `allenai/ai2_arc` `ARC-Challenge`
- `arc_easy` suite for `allenai/ai2_arc` `ARC-Easy`
- `boolq` suite for `super_glue` `boolq`
- `cb` suite for `super_glue` `cb`
- `copa` suite for `super_glue` `copa`
- `gsm8k` suite for `openai/gsm8k`
- `gsm8k_platinum` suite ported from `lm-eval`
- `hellaswag` suite for `Rowan/hellaswag`
- `mmlu` suite for `cais/mmlu`
- `openbookqa` suite for `allenai/openbookqa` `main`
- `piqa` suite for `baber/piqa`
- `rte` suite for `super_glue` `rte`
- `sst2` suite for `nyu-mll/glue` `sst2`
- `wic` suite for `super_glue` `wic`
- `winogrande` suite for `winogrande` `winogrande_xl`
- `logbar`-powered runtime logging and evaluation progress bars

## Citation

If you use Evalution or the built-in `gsm8k`, `gsm8k_platinum`, `arc_challenge`, `mmlu`, or `piqa`
suites, please cite:

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

# ARC
@article{clark2018arc,
  title = {Think you have Solved Question Answering? Try {ARC}, the {AI2} Reasoning Challenge},
  author = {Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal = {arXiv preprint arXiv:1803.05457},
  year = {2018},
}

# PIQA
@inproceedings{bisk2020piqa,
  title = {PIQA: Reasoning about Physical Commonsense in Natural Language},
  author = {Yonatan Bisk and Rowan Zellers and Ronan Le Bras and Jianfeng Gao and Yejin Choi},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2020},
}

# MMLU
@article{hendryckstest2021,
  title = {Measuring Massive Multitask Language Understanding},
  author = {Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal = {International Conference on Learning Representations},
  year = {2021},
}
```
