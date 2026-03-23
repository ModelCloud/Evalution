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
Metric-key glossary lives in [docs/scores.md](docs/scores.md). Scoring implementation notes and
scorer-module mapping live in [docs/scorers.md](docs/scorers.md).

Simple usage:

```python
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers()
    .model({"path": "/monster/data/model/Llama-3.2-1B-Instruct"})
    .run(benchmarks.gsm8k_platinum())
)
```

Advanced usage:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers(
        dtype="bfloat16",
        attn_implementation="paged|flash_attention_2",
        device="cuda:0",
        batch_size="auto",
        allow_block_sharing=True,
        use_async_batching=None,
        max_new_tokens=256,
    )
    .model(
        eval.Model(
            path="/monster/data/model/Llama-3.2-1B-Instruct",
        )
    )
    .run(
        benchmarks.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            max_new_tokens=96,
            batch_size=64,
            max_rows=128,
        )
    )
    .run(
        benchmarks.arc_challenge(
            batch_size=64,
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
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    eval.compare(
        engines.Transformers(dtype="bfloat16", device="cuda:0").model(
            {"path": "/monster/data/model/Llama-3.2-1B-Instruct"},
            label="llama",
        ),
        engines.TransformersCompat(device="cuda:1").model(
            {"path": "/monster/data/model/Qwen2.5-1.5B-Instruct"},
            label="qwen",
        ),
    )
    .run(benchmarks.gsm8k_platinum(max_rows=128))
    .run(benchmarks.arc_challenge(max_rows=128))
)
```

`compare(...)` takes the same `engine.model(...)` handles used for single-model runs, so
single and compare flows share one fluent entry shape. Compare lane labels come from
`.model(..., label="...")`; when omitted, Evalution falls back to the model path. It runs the same
suite list on both lanes while allowing different engines and model configs on the left and right.
When the terminal supports LogBar split panes, Evalution binds each lane to its own pane and
renders a consolidated compare summary when the run closes.

YAML usage:

```yaml
engine:
  type: Transformers
  dtype: bfloat16
  attn_implementation: paged|flash_attention_2
  device: cuda:0

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
    batch_size: 64
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

`engines.Transformers(...)` accepts runtime options such as `dtype`, `device`, `batch_size`,
`attn_implementation`, and `max_new_tokens`.

Per-benchmark options such as `apply_chat_template`, `batch_size`, `max_new_tokens`, `max_rows`,
and scorer-specific options like `label_permutations` can be set directly on each benchmark call
or in each YAML `tests` entry.

## Subset Selection

Subset-aware benchmarks use a `subsets` selector instead of benchmark-specific selector names.
Currently this applies to `mmlu` and `mmlu_pro`.

- `subsets: all` runs the full benchmark.
- `subsets: stem` runs the full `stem` subtree.
- `subsets: stem.math` or `subsets: stem.abstract_algebra` runs a single leaf path.
- `subsets: [stem, humanities]` runs the union of multiple selections.
- Deeper paths are supported by the same syntax when a suite defines them.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers()
    .model(eval.Model(path="/monster/data/model/Llama-3.2-1B-Instruct"))
    .run(benchmarks.mmlu(subsets=["stem.abstract_algebra", "humanities.philosophy"]))
    .run(benchmarks.mmlu_pro(subsets="stem.math"))
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

Use `engines.TransformersCompat()` in Python or `engine.type: TransformersCompat` in YAML when you
want the compatibility engine explicitly.

## Supported Benchmarks

Evalution currently ships the following built-in benchmarks:

Evalution aims to align each built-in suite's default split, prompting shape, and scoring logic as
closely as practical with the original benchmark paper and any released reference code from the
benchmark authors. Users should compare scores across different LLM evaluation projects with care:
some frameworks do not match the original benchmark scoring exactly, which can make headline
numbers look comparable when they are not. This matters most for researchers reporting results in
papers or otherwise making cross-project claims. Reported scores are also affected by runtime and
numerics details such as hardware behavior, dtype and normalization choices, kernel differences,
and attention or matmul approximation and accumulation behavior. Even with the same benchmark
logic, those implementation details can shift results.

| Suite | Hugging Face dataset | Default split | Scoring | Original benchmark |
| --- | --- | --- | --- | --- |
| `arc_challenge` | `allenai/ai2_arc` / `ARC-Challenge` | `test` | Multiple-choice exam score with tie-aware partial credit | ARC `clark2018arc` |
| `arc_easy` | `allenai/ai2_arc` / `ARC-Easy` | `test` | Multiple-choice exam score with tie-aware partial credit | ARC `clark2018arc` |
| `boolq` | `super_glue` / `boolq` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `cb` | `super_glue` / `cb` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, macro F1 | SuperGLUE `wang2019superglue` |
| `cola` | `nyu-mll/glue` / `cola` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy, MCC | GLUE `wang-etal-2018-glue` |
| `commonsense_qa` | `tau/commonsense_qa` | `validation` | Multiple-choice log-likelihood over answer labels, raw + length-normalized accuracy | CommonsenseQA `talmor2019commonsenseqa` |
| `copa` | `super_glue` / `copa` | `validation` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `gsm8k` | `openai/gsm8k` / `main` | `test` | Format-insensitive numeric accuracy | GSM8K `cobbe2021trainingverifierssolvemath` |
| `gsm8k_platinum` | `madrylab/gsm8k-platinum` / `main` | `test` | Format-insensitive numeric accuracy | GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks` |
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

`arc_challenge` and `arc_easy`: choose among answer options and score the question as an exam
item, including partial credit when multiple top-scoring choices tie.

For selected multiple-choice suites, `label_permutations` can be set to any float in `[0.0, 1.0]`
to add an extra permutation-averaged label-only metric. This does not replace the default
benchmark score. It adds extra inference work on purpose so users can compare the benchmark-native
score against a label-bias-mitigated alternative when option length is a concern. Metric names
carry the exact configured fraction after `:`, for example `acc,label_perm:0.25`. See
[docs/scores.md](docs/scores.md) for the short-label glossary and [docs/scorers.md](docs/scorers.md)
for the exact math, metric names, and compute tradeoffs.

Metric key glossary:

- `acc`: accuracy-like credit on a `0.0` to `1.0` scale for each sample, then averaged.
- `ll`: raw summed continuation log-likelihood over the scored answer tokens.
- `ll_avg`: average continuation log-likelihood per scored answer token to reduce length bias.
- `exam`: ARC exam-style tie-aware partial credit.
- `num`: numeric-answer match after numeric extraction and canonicalization.
- `em`: exact match after the suite's task-specific extraction step.
- `choice_label`: extracted option-label match such as `A/B/C/D`.
- `label_perm:<fraction>`: permutation-averaged label-only accuracy using the configured fraction
  of all label permutations.
- `f1`: F1 score derived from the suite's predicted labels.
- `mcc`: Matthews correlation coefficient derived from the suite's predicted labels.
- `macro`: macro-average across labels rather than a single positive class.
- `yes`: positive-class metric using the suite's `yes` or equivalent positive label.

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
- `commonsense_qa`: CommonsenseQA `talmor2019commonsenseqa`
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

# CommonsenseQA
@inproceedings{talmor2019commonsenseqa,
  title = {CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author = {Alon Talmor and Jonathan Herzig and Nicholas Lourie and Jonathan Berant},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019},
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
