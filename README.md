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

Engine implementation notes for backend authors live in [docs/engine.md](docs/engine.md).

Simple usage:

```python
import evalution as eval

result = (
    eval.engine(eval.Transformer())
    .model({"path": "/monster/data/model/Llama-3.2-1B-Instruct"})
    .run(eval.gsm8k_platinum())
)
```

Advanced usage:

```python
import evalution as eval

result = (
    eval.engine(
        eval.Transformer(
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

Compare usage:

```python
import evalution as eval

result = (
    eval.compare(
        eval.engine(eval.Transformer(dtype="bfloat16", device="cuda:0")).model(
            {"path": "/monster/data/model/Llama-3.2-1B-Instruct"},
            label="llama",
        ),
        eval.engine(eval.TransformerCompat(device="cuda:1")).model(
            {"path": "/monster/data/model/Qwen2.5-1.5B-Instruct"},
            label="qwen",
        ),
    )
    .run(eval.gsm8k_platinum(max_rows=128))
    .run(eval.arc_challenge(max_rows=128))
)
```

`compare(...)` takes the same `eval.engine(...).model(...)` handles used for single-model runs, so
single and compare flows share one fluent entry shape. Compare lane labels come from
`.model(..., label="...")`; when omitted, Evalution falls back to the model path. It runs the same
suite list on both lanes while allowing different engines and model configs on the left and right.
When the terminal supports LogBar split panes, Evalution binds each lane to its own pane and
renders a consolidated compare summary when the run closes.

YAML usage:

```yaml
engine:
  type: Transformer
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

`Transformer(...)` accepts runtime options such as `dtype`, `device`, `batch_size`,
`paged_attention`, `attn_implementation`, and `max_new_tokens`.

Per-suite options such as `apply_chat_template`, `batch_size`, `max_new_tokens`, and `max_rows`
can be set directly on each suite call or in each YAML `tests` entry.

## Subset Selection

Subset-aware suites use a `subsets` selector instead of suite-specific selector names.
Currently this applies to `mmlu` and `mmlu_pro`.

- `subsets: all` runs the full suite.
- `subsets: stem` runs the full `stem` subtree.
- `subsets: stem.math` or `subsets: stem.abstract_algebra` runs a single leaf path.
- `subsets: [stem, humanities]` runs the union of multiple selections.
- Deeper paths are supported by the same syntax when a suite defines them.

Python:

```python
import evalution as eval

result = (
    eval.engine(eval.Transformer())
    .model(eval.Model(path="/monster/data/model/Llama-3.2-1B-Instruct"))
    .run(eval.mmlu(subsets=["stem.abstract_algebra", "humanities.philosophy"]))
    .run(eval.mmlu_pro(subsets="stem.math"))
)
```

YAML:

```yaml
tests:
  - type: mmlu
    subsets:
      - stem.abstract_algebra
      - humanities.philosophy
    num_fewshot: 5
  - type: mmlu_pro
    subsets: stem.math
    num_fewshot: 5
```

Use `TransformerCompat()` in Python or `engine.type: TransformerCompat` in YAML when you want
the compatibility engine explicitly.

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
| `mmlu` | `cais/mmlu` / `<subsets>` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | MMLU `hendryckstest2021` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` / `<subsets>` | `test` | Generated choice-label exact match with CoT prompting | MMLU-Pro `wang2024mmlupro` |
| `mnli` | `nyu-mll/glue` / `mnli` | `validation_matched` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `mrpc` | `nyu-mll/glue` / `mrpc` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | GLUE `wang-etal-2018-glue` |
| `openbookqa` | `allenai/openbookqa` / `main` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | OpenBookQA `mihaylov2018openbookqa` |
| `piqa` | `baber/piqa` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | PIQA `bisk2020piqa` |
| `qnli` | `nyu-mll/glue` / `qnli` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `qqp` | `nyu-mll/glue` / `qqp` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | GLUE `wang-etal-2018-glue` |
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
- `mmlu_pro`: MMLU-Pro `wang2024mmlupro`
- `cola`, `mnli`, `mrpc`, `qnli`, `qqp`, `sst2`, `wnli`: GLUE `wang-etal-2018-glue`
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

# MMLU-Pro
@article{wang2024mmlupro,
  title = {MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author = {Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
  journal = {arXiv preprint arXiv:2406.01574},
  year = {2024},
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
