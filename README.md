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
    .run(eval.mrpc())
    .run(eval.openbookqa())
    .run(eval.piqa())
    .run(eval.qnli())
    .run(eval.rte())
    .run(eval.sst2())
    .run(eval.wic())
    .run(eval.wnli())
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

## Supported Suites

Evalution currently ships the following built-in suites:

| Suite | Hugging Face dataset | Default split | Scoring | Original benchmark |
| --- | --- | --- | --- | --- |
| `arc_challenge` | `allenai/ai2_arc` / `ARC-Challenge` | `test` | Generated label exact match, choice-text fallback | ARC `clark2018arc` |
| `arc_easy` | `allenai/ai2_arc` / `ARC-Easy` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ARC `clark2018arc` |
| `boolq` | `super_glue` / `boolq` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `cb` | `super_glue` / `cb` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, macro F1 | SuperGLUE `wang2019superglue` |
| `cola` | `nyu-mll/glue` / `cola` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, MCC | GLUE `wang-etal-2018-glue` |
| `copa` | `super_glue` / `copa` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `gsm8k` | `openai/gsm8k` / `main` | `test` | Generated answer exact match, strict + flexible extraction | GSM8K `cobbe2021trainingverifierssolvemath` |
| `gsm8k_platinum` | `madrylab/gsm8k-platinum` / `main` | `test` | Generated answer exact match, strict + flexible extraction | GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks` |
| `hellaswag` | `Rowan/hellaswag` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | HellaSwag `zellers2019hellaswag` |
| `mmlu` | `cais/mmlu` / `<subject>` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | MMLU `hendryckstest2021` |
| `mrpc` | `nyu-mll/glue` / `mrpc` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | GLUE `wang-etal-2018-glue` |
| `openbookqa` | `allenai/openbookqa` / `main` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | OpenBookQA `mihaylov2018openbookqa` |
| `piqa` | `baber/piqa` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | PIQA `bisk2020piqa` |
| `qnli` | `nyu-mll/glue` / `qnli` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `rte` | `super_glue` / `rte` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `sst2` | `nyu-mll/glue` / `sst2` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `wic` | `super_glue` / `wic` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `wnli` | `nyu-mll/glue` / `wnli` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `winogrande` | `winogrande` / `winogrande_xl` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | WinoGrande `sakaguchi2019winogrande` |

Evalution also includes the Hugging Face `transformers` inference engine, YAML execution, a packaged CLI, and `logbar`-powered runtime progress reporting.

## Citation

If you use Evalution, cite the project itself. If you use one or more built-in suites, also cite the
original benchmark papers below.

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
```

The current built-in suite coverage maps to these benchmark citations:

- `arc_challenge`, `arc_easy`: ARC `clark2018arc`
- `boolq`, `cb`, `copa`, `rte`, `wic`: SuperGLUE `wang2019superglue`
- `gsm8k`: GSM8K `cobbe2021trainingverifierssolvemath`
- `gsm8k_platinum`: GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks`
- `hellaswag`: HellaSwag `zellers2019hellaswag`
- `mmlu`: MMLU `hendryckstest2021`
- `mrpc`, `qnli`, `sst2`, `wnli`: GLUE `wang-etal-2018-glue`
- `openbookqa`: OpenBookQA `mihaylov2018openbookqa`
- `piqa`: PIQA `bisk2020piqa`
- `winogrande`: WinoGrande `sakaguchi2019winogrande`

```bibtex

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

# GLUE
@inproceedings{wang-etal-2018-glue,
  title = {{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author = {Alex Wang and Amanpreet Singh and Julian Michael and Felix Hill and Omer Levy and Samuel Bowman},
  booktitle = {Proceedings of the 2018 EMNLP Workshop BlackboxNLP},
  year = {2018},
}

# SuperGLUE
@inproceedings{wang2019superglue,
  title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
  author = {Alex Wang and Yada Pruksachatkun and Nikita Nangia and Amanpreet Singh and Julian Michael and Felix Hill and Omer Levy and Samuel Bowman},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019},
}

# HellaSwag
@inproceedings{zellers2019hellaswag,
  title = {HellaSwag: Can a Machine Really Finish Your Sentence?},
  author = {Rowan Zellers and Ari Holtzman and Yonatan Bisk and Ali Farhadi and Yejin Choi},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year = {2019},
}

# OpenBookQA
@inproceedings{mihaylov2018openbookqa,
  title = {Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
  author = {Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
}

# WinoGrande
@article{sakaguchi2019winogrande,
  title = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
  author = {Keisuke Sakaguchi and Ronan Le Bras and Chandra Bhagavatula and Yejin Choi},
  journal = {arXiv preprint arXiv:1907.10641},
  year = {2019},
}
```
