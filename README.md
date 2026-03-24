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
The package also depends on `tokenicer` for tokenizer loading and normalization.

Engine implementation notes for backend authors live in [docs/engine.md](docs/engine.md).
Metric-key glossary lives in [docs/scores.md](docs/scores.md). Scoring implementation notes and
scorer-module mapping live in [docs/scorers.md](docs/scorers.md).

Simple usage:

```python
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers()
    .model(path="/monster/data/model/Llama-3.2-1B-Instruct")
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
    .model(path="/monster/data/model/Llama-3.2-1B-Instruct")
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
            path="/monster/data/model/Llama-3.2-1B-Instruct",
            label="llama",
        ),
        engines.TransformersCompat(device="cuda:1").model(
            path="/monster/data/Qwen2.5-1.5B-Instruct",
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
    .model(path="/monster/data/model/Llama-3.2-1B-Instruct")
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

`Tokenicer` is used to load tokenizers for the transformer and transformer-compat engines. When
`engine.model(...)` is called with a model config, Evalution resolves tokenizer loading in this order:
`tokenizer` (preinitialized object), `tokenizer_path`, then `path`.
`Tokenicer` also applies its normalization stage so pad/eos/bos token IDs are corrected before evaluation.
To inject a custom tokenizer, pass it through `.model(...)` on the model config:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

custom_tokenizer = ...

result = (
    engines.Transformers()
    .model(
        path="/monster/data/model/Llama-3.2-1B-Instruct",
        tokenizer=custom_tokenizer,
    )
    .run(benchmarks.gsm8k_platinum(max_rows=128))
)
```

YAML flows can only configure `tokenizer_path`; passing a live tokenizer object is Python-only.

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

| Suite | Scoring | Original benchmark |
| --- | --- | --- |
| `aexams` | Multiple-choice log-likelihood over answer labels across `biology/islamic_studies/physics/science/social` subjects, raw + length-normalized accuracy | EXAMS `hardalov-etal-2020-exams` |
| `agieval` | Multiple-choice log-likelihood across supported AGIEval subject subsets, raw + length-normalized accuracy | AGIEval `zhong2023agieval` |
| `afrimgsm` | Generated numeric exact match across `amh/eng/ewe/fra/hau/ibo/kin/lin/lug/orm/sna/sot/swa/twi/vai/wol/xho/yor/zul` translated grade-school math subsets | IrokoBench AfriMGSM `adelani2025irokobench` |
| `afrimmlu` | Multiple-choice log-likelihood across `amh/eng/ewe/fra/hau/ibo/kin/lin/lug/orm/sna/sot/swa/twi/wol/xho/yor/zul` translated MMLU subsets, raw + length-normalized accuracy | IrokoBench AfriMMLU `adelani2025irokobench` |
| `aime` | Generated math-normalized exact match with boxed-answer extraction | AIME `aime_1983_2024` |
| `aime24` | Generated math-normalized exact match with boxed-answer extraction | AIME `aime_2024` |
| `aime25` | Generated math-normalized exact match with boxed-answer extraction | AIME `aime_2025` |
| `afrixnli` | Three-way NLI multiple-choice log-likelihood across `amh/eng/ewe/fra/hau/ibo/kin/lin/lug/orm/sna/sot/swa/twi/wol/xho/yor/zul` language subsets, raw + length-normalized accuracy | IrokoBench AfriXNLI `adelani2025irokobench` |
| `anli_r1` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ANLI `nie-etal-2020-adversarial` |
| `anli_r2` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ANLI `nie-etal-2020-adversarial` |
| `anli_r3` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ANLI `nie-etal-2020-adversarial` |
| `arabicmmlu` | Multiple-choice log-likelihood across configurable ArabicMMLU subject subsets, raw + length-normalized accuracy | ArabicMMLU `koto2024arabicmmlu` |
| `darijammlu` | Multiple-choice log-likelihood across configurable DarijaMMLU subject subsets, raw + length-normalized accuracy | DarijaMMLU `shang2024atlaschatadaptinglargelanguage` |
| `egymmlu` | Multiple-choice log-likelihood across configurable EgyMMLU subject subsets, raw + length-normalized accuracy | EgyMMLU `el-mekki-etal-2025-nilechat` |
| `eus_exams` | Multiple-choice log-likelihood across configurable Basque and Spanish civil-service exam subsets, raw + length-normalized accuracy | EusExams `etxaniz2024latxa` |
| `arc_challenge` | Multiple-choice exam score with tie-aware partial credit | ARC `clark2018arc` |
| `arc_easy` | Multiple-choice exam score with tie-aware partial credit | ARC `clark2018arc` |
| `arc_mt` | Multiple-choice exam score with tie-aware partial credit across translated ARC Challenge subsets `da/de/el/es/fi/hu/is/it/nb/pl/pt/sv` | ARC `clark2018arc` |
| `arithmetic_1dc` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_2da` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_2dm` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_2ds` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_3da` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_3ds` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_4da` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_4ds` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_5da` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `arithmetic_5ds` | Single-continuation log-likelihood, greedy accuracy with no added target delimiter | GPT-3 arithmetic `brown2020gpt3` |
| `asdiv` | Single-continuation log-likelihood, greedy accuracy over canonical numeric answers | ASDiv `miao2021diverse` |
| `asdiv_cot_llama` | Few-shot CoT generation with format-insensitive numeric accuracy | ASDiv `miao2021diverse` |
| `babi` | Generated exact match | bAbI `weston2015towards` |
| `babilong` | Generated normalized exact match across `qa1` to `qa20` with configurable context lengths | BABILong `kuratov2024babilong` |
| `bbh` | Generated exact match across 27 BIG-Bench Hard subsets selected through the `subset` parameter | BIG-Bench Hard `suzgun2022challenging` |
| `bangla` | Multiple-choice log-likelihood across `boolqa/commonsenseqa/mmlu/openbookqa/piqa` Bangla subsets, raw + length-normalized accuracy | TituLLMs Bangla benchmarks `nahin2025titullmsfamilybanglallms` |
| `bear` | Full-statement multiple-choice log-likelihood over balanced relational distractors, raw + length-normalized accuracy | BEAR `wiland2024bear` |
| `bear_big` | Full-statement multiple-choice log-likelihood over the larger BEAR probe, raw + length-normalized accuracy | BEAR `wiland2024bear` |
| `belebele` | Multiple-choice reading-comprehension log-likelihood across 122 language variants selected through the `language` parameter, raw + length-normalized accuracy | Belebele `bandarkar2023belebele` |
| `blimp` | Minimal-pair full-sentence log-likelihood over configurable BLiMP subsets, raw + length-normalized accuracy | BLiMP `warstadt2020blimp` |
| `c4` | Rolling log-likelihood with word perplexity, byte perplexity, and bits per byte | C4 `raffel2020exploring` |
| `careqa` | Multiple-choice log-likelihood across supported closed-ended healthcare QA subsets `en/es`, raw + length-normalized accuracy | CareQA `arias-duart-etal-2025-automatic` |
| `cabbq` | Multiple-choice log-likelihood across Catalan BBQ bias categories, raw + length-normalized accuracy | CaBBQ `ruizfernández2025esbbqcabbqspanishcatalan` |
| `esbbq` | Multiple-choice log-likelihood across Spanish BBQ bias categories, raw + length-normalized accuracy | EsBBQ `ruizfernández2025esbbqcabbqspanishcatalan` |
| `ceval` | Multiple-choice log-likelihood over configurable C-Eval subsets, raw + length-normalized accuracy | C-Eval `huang2023ceval` |
| `boolq` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `cb` | Multiple-choice log-likelihood, raw + length-normalized accuracy, macro F1 | SuperGLUE `wang2019superglue` |
| `cola` | Multiple-choice log-likelihood, raw + length-normalized accuracy, MCC | GLUE `wang-etal-2018-glue` |
| `cnn_dailymail` | Generated summarization with ROUGE-1, ROUGE-2, and ROUGE-Lsum F1 | CNN/DailyMail `nallapati2016abstractive` |
| `code_x_glue` | Code-to-docstring generation with corpus smoothed BLEU-4 across `go/java/javascript/php/python/ruby` language subsets | CodeXGLUE `lu2021codexglue` |
| `commonsense_qa` | Multiple-choice log-likelihood over answer labels, raw + length-normalized accuracy | CommonsenseQA `talmor2019commonsenseqa` |
| `coqa` | Generated conversational QA exact match and token-overlap F1 with gold history turns | CoQA `reddy2019coqa` |
| `copa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `copa_ar` | Multiple-choice log-likelihood over the AlGhafa Arabic COPA translation, raw + length-normalized accuracy | AlGhafa `almazrouei-etal-2023-alghafa` |
| `copal_id` | Multiple-choice log-likelihood across `standard/colloquial` variants, raw + length-normalized accuracy | COPAL-ID `wibowo-etal-2024-copal` |
| `crows_pairs` | Pairwise full-sentence log-likelihood across `english/french` language splits and supported bias-category subsets, with stereotype-preference rate and average absolute log-likelihood gap | CrowS-Pairs `nangia-etal-2020-crows`, French CrowS-Pairs `neveol-etal-2022-french` |
| `darijahellaswag` | Multiple-choice log-likelihood over Moroccan Darija translated HellaSwag endings, raw + length-normalized accuracy | DarijaHellaSwag `shang2024atlaschatadaptinglargelanguage` |
| `egyhellaswag` | Multiple-choice log-likelihood over Egyptian Arabic translated HellaSwag endings, raw + length-normalized accuracy | EgyHellaSwag `mekki2025nilechatlinguisticallydiverseculturally` |
| `drop` | Generated QA exact match and token-overlap F1 over accepted answer spans | DROP `dua2019drop` |
| `gpqa` | Generated answer-label exact match across the `main/diamond/extended` subsets, with seeded answer-order shuffling and author-style zero-shot prompting | GPQA `rein2024gpqa` |
| `ethics_cm` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ETHICS `hendrycks2021ethics` |
| `ethics_deontology` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ETHICS `hendrycks2021ethics` |
| `ethics_justice` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ETHICS `hendrycks2021ethics` |
| `ethics_utilitarianism` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ETHICS `hendrycks2021ethics` |
| `ethics_virtue` | Multiple-choice log-likelihood, raw + length-normalized accuracy | ETHICS `hendrycks2021ethics` |
| `gsm8k` | Format-insensitive numeric accuracy | GSM8K `cobbe2021trainingverifierssolvemath` |
| `gsm8k_platinum` | Format-insensitive numeric accuracy | GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks` |
| `hellaswag` | Multiple-choice log-likelihood, raw + length-normalized accuracy | HellaSwag `zellers2019hellaswag` |
| `headqa_en` | Multiple-choice log-likelihood, raw + length-normalized accuracy | HEAD-QA `vilares-gomez-rodriguez-2019-head` |
| `headqa_es` | Multiple-choice log-likelihood, raw + length-normalized accuracy | HEAD-QA `vilares-gomez-rodriguez-2019-head` |
| `histoires_morales` | Multiple-choice log-likelihood over moral versus norm-divergent actions, raw + length-normalized accuracy | Histoires Morales `leteno2025histoiresmorales` |
| `kobest` | Multiple-choice log-likelihood across `boolq/copa/hellaswag/sentineg/wic` Korean subsets, raw + length-normalized accuracy | KoBEST `kim2022kobest` |
| `icelandic_winogrande` | Partial-evaluation multiple-choice log-likelihood over blank replacements, raw + length-normalized accuracy | Icelandic WinoGrande `snaebjarnarson-etal-2022-warm` |
| `lambada_openai` | Single-continuation log-likelihood, greedy accuracy + perplexity | LAMBADA `paperno2016lambada` |
| `lambada_openai_cloze` | Single-continuation log-likelihood, greedy accuracy + perplexity | LAMBADA `paperno2016lambada` |
| `lambada_standard` | Single-continuation log-likelihood, greedy accuracy + perplexity | LAMBADA `paperno2016lambada` |
| `lambada_standard_cloze` | Single-continuation log-likelihood, greedy accuracy + perplexity | LAMBADA `paperno2016lambada` |
| `logiqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | LogiQA `liu2020logiqa` |
| `mathqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | MathQA `amini2019mathqa` |
| `mc_taco` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | MC-TACO `zhou2019mctaco` |
| `medmcqa` | Multiple-choice log-likelihood over answer labels, raw + length-normalized accuracy | MedMCQA `pmlr-v174-pal22a` |
| `medqa_4options` | Multiple-choice log-likelihood over answer labels, raw + length-normalized accuracy | MedQA `jin2020disease` |
| `mmlu` | Multiple-choice log-likelihood, raw + length-normalized accuracy | MMLU `hendryckstest2021` |
| `mmlu_pro` | Generated choice-label exact match with CoT prompting | MMLU-Pro `wang2024mmlupro` |
| `mnli` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `mrpc` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | GLUE `wang-etal-2018-glue` |
| `nq_open` | Generated QA exact match and token-overlap F1 over answer aliases | Natural Questions `kwiatkowski2019natural` |
| `openbookqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | OpenBookQA `mihaylov2018openbookqa` |
| `paws_x` | Multiple-choice log-likelihood across `de/en/es/fr/ja/ko/zh` language subsets, raw + length-normalized accuracy, positive-class F1 | PAWS-X `yang2019pawsx` |
| `xcopa` | Multiple-choice log-likelihood over option labels across `et/ht/id/it/qu/sw/ta/th/tr/vi/zh` language subsets, raw + length-normalized accuracy | XCOPA `ponti2020xcopa` |
| `xstorycloze` | Multiple-choice log-likelihood over translated StoryCloze endings across `ar/en/es/eu/hi/id/my/ru/sw/te/zh` subsets, raw + length-normalized accuracy | XStoryCloze `lin2021fewshotmultilingual` |
| `xnli` | Three-way NLI multiple-choice log-likelihood across `ar/bg/de/el/en/es/fr/hi/ru/sw/th/tr/ur/vi/zh` language subsets, raw + length-normalized accuracy | XNLI `conneau2018xnli` |
| `xwinograd` | Partial-evaluation multiple-choice log-likelihood over blank replacements across `en/fr/jp/pt/ru/zh` language subsets, raw + length-normalized accuracy | XWinograd `tikhonov2021heads` |
| `piqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | PIQA `bisk2020piqa` |
| `piqa_ar` | Multiple-choice log-likelihood over the AlGhafa Arabic PIQA translation, raw + length-normalized accuracy | AlGhafa `almazrouei-etal-2023-alghafa` |
| `pile_10k` | Rolling log-likelihood with word perplexity, byte perplexity, and bits per byte | The Pile `gao2020pile` |
| `prost` | Multiple-choice log-likelihood, raw + length-normalized accuracy | PROST `aroca-ouellette-etal-2021-prost` |
| `pubmedqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | PubMedQA `jin2019pubmedqa` |
| `qa4mre` | Multiple-choice log-likelihood across `2011/2012/2013` English machine reading evaluation sets, raw + length-normalized accuracy | QA4MRE `Peas2013QA4MRE2O` |
| `qnli` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `qqp` | Multiple-choice log-likelihood, raw + length-normalized accuracy, positive-class F1 | GLUE `wang-etal-2018-glue` |
| `race` | Multiple-choice log-likelihood, raw + length-normalized accuracy | RACE `lai-etal-2017-race` |
| `rte` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `sciq` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SciQ `welbl2017crowdsourcing` |
| `siqa` | Multiple-choice log-likelihood, raw + length-normalized accuracy | Social IQA `sap2019social` |
| `swag` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SWAG `zellers2018swagaf` |
| `sst2` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `squadv2` | Generated QA exact match and token-overlap F1 with explicit no-answer handling | SQuAD 2.0 `rajpurkar2018know` |
| `triviaqa` | Generated QA exact match and token-overlap F1 over answer aliases | TriviaQA `joshi2017triviaqa` |
| `wic` | Multiple-choice log-likelihood, raw + length-normalized accuracy | SuperGLUE `wang2019superglue` |
| `webqs` | Accepted-alias log-likelihood, greedy exact match over any accepted answer | WebQuestions `berant-etal-2013-semantic` |
| `wikitext` | Rolling log-likelihood with word perplexity, byte perplexity, and bits per byte | WikiText-2 `merity2016pointer` |
| `winogender` | Multiple-choice log-likelihood over pronoun-reference prompts across `all/gotcha` variants and `female/male/neutral` gender filters, raw + length-normalized accuracy | WinoGender `rudinger2018winogender` |
| `wsc273` | Partial-evaluation multiple-choice log-likelihood, raw + length-normalized accuracy | WSC273 `levesque2012winograd` |
| `wnli` | Multiple-choice log-likelihood, raw + length-normalized accuracy | GLUE `wang-etal-2018-glue` |
| `winogrande` | Multiple-choice log-likelihood, raw + length-normalized accuracy | WinoGrande `sakaguchi2019winogrande` |

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
- `ppl`: perplexity from exponentiated negative mean continuation log-likelihood. Lower is better.
- `word_perplexity`: exponentiated negative mean log-likelihood per original-document word, weighted across samples.
- `byte_perplexity`: exponentiated negative mean log-likelihood per original-document byte, weighted across samples.
- `bits_per_byte`: negative mean log-likelihood per original-document byte in base-2 units.
- `choice_label`: extracted option-label match such as `A/B/C/D`.
- `pct_stereotype`: fraction of sentence pairs where the more stereotypical sentence receives the higher score.
- `likelihood_diff`: average absolute log-likelihood gap between paired candidate sentences.
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

- `aexams_biology`, `aexams_islamic_studies`, `aexams_physics`, `aexams_science`, `aexams_social`: EXAMS `hardalov-etal-2020-exams`
- `agieval_<subset>` for the built-in single-answer AGIEval subsets: AGIEval `zhong2023agieval`
- `afrimgsm_<language>` for the built-in AfriMGSM language subsets: IrokoBench AfriMGSM `adelani2025irokobench`
- `afrimmlu_<language>` for the built-in AfriMMLU language subsets: IrokoBench AfriMMLU `adelani2025irokobench`
- `afrixnli_amh`, `afrixnli_eng`, `afrixnli_ewe`, `afrixnli_fra`, `afrixnli_hau`, `afrixnli_ibo`, `afrixnli_kin`, `afrixnli_lin`, `afrixnli_lug`, `afrixnli_orm`, `afrixnli_sna`, `afrixnli_sot`, `afrixnli_swa`, `afrixnli_twi`, `afrixnli_wol`, `afrixnli_xho`, `afrixnli_yor`, `afrixnli_zul`: IrokoBench AfriXNLI `adelani2025irokobench`
- `anli_r1`, `anli_r2`, `anli_r3`: ANLI `nie-etal-2020-adversarial`
- `arabicmmlu_<subset>` for the built-in ArabicMMLU subsets: ArabicMMLU `koto2024arabicmmlu`
- `darijammlu_<subset>` for the built-in DarijaMMLU subsets: DarijaMMLU `shang2024atlaschatadaptinglargelanguage`
- `egymmlu_<subset>` for the built-in EgyMMLU subsets: EgyMMLU `el-mekki-etal-2025-nilechat`
- `eus_exams_<subset>` for the built-in EusExams subsets: EusExams `etxaniz2024latxa`
- `aime`, `aime24`, `aime25`: AIME `aime_1983_2024`, `aime_2024`, `aime_2025`
- `arc_challenge`, `arc_easy`: ARC `clark2018arc`
- `arc_mt_da`, `arc_mt_de`, `arc_mt_el`, `arc_mt_es`, `arc_mt_fi`, `arc_mt_hu`, `arc_mt_is`, `arc_mt_it`, `arc_mt_nb`, `arc_mt_pl`, `arc_mt_pt`, `arc_mt_sv`: ARC `clark2018arc`
- `arithmetic_1dc`, `arithmetic_2da`, `arithmetic_2dm`, `arithmetic_2ds`, `arithmetic_3da`, `arithmetic_3ds`, `arithmetic_4da`, `arithmetic_4ds`, `arithmetic_5da`, `arithmetic_5ds`: GPT-3 arithmetic `brown2020gpt3`
- `asdiv`, `asdiv_cot_llama`: ASDiv `miao2021diverse`
- `babi`: bAbI `weston2015towards`
- `babilong_<qa_split>` for `qa1` through `qa20`: BABILong `kuratov2024babilong`
- `bbh_<subset>` for all BIG-Bench Hard subsets: BIG-Bench Hard `suzgun2022challenging`
- `bangla_boolqa`, `bangla_commonsenseqa`, `bangla_mmlu`, `bangla_openbookqa`, `bangla_piqa`: TituLLMs Bangla benchmarks `nahin2025titullmsfamilybanglallms`
- `bear`, `bear_big`: BEAR `wiland2024bear`
- `belebele`: Belebele `bandarkar2023belebele`
- `blimp`: BLiMP `warstadt2020blimp`
- `c4`: C4 `raffel2020exploring`
- `careqa_en`, `careqa_es`: CareQA `arias-duart-etal-2025-automatic`
- `cabbq_<category>` for the built-in Catalan BBQ categories: CaBBQ `ruizfernández2025esbbqcabbqspanishcatalan`
- `esbbq_<category>` for the built-in Spanish BBQ categories: EsBBQ `ruizfernández2025esbbqcabbqspanishcatalan`
- `ceval`: C-Eval `huang2023ceval`
- `boolq`, `cb`, `copa`, `rte`, `wic`: SuperGLUE `wang2019superglue`
- `cnn_dailymail`: CNN/DailyMail `nallapati2016abstractive`
- `code2text_go`, `code2text_java`, `code2text_javascript`, `code2text_php`, `code2text_python`, `code2text_ruby`: CodeXGLUE `lu2021codexglue`
- `commonsense_qa`: CommonsenseQA `talmor2019commonsenseqa`
- `coqa`: CoQA `reddy2019coqa`
- `copa_ar`, `piqa_ar`: AlGhafa Arabic translations `almazrouei-etal-2023-alghafa`
- `copal_id_standard`, `copal_id_colloquial`: COPAL-ID `wibowo-etal-2024-copal`
- `crows_pairs_english`, `crows_pairs_english_age`, `crows_pairs_english_autre`, `crows_pairs_english_disability`, `crows_pairs_english_gender`, `crows_pairs_english_nationality`, `crows_pairs_english_physical_appearance`, `crows_pairs_english_race_color`, `crows_pairs_english_religion`, `crows_pairs_english_sexual_orientation`, `crows_pairs_english_socioeconomic`: CrowS-Pairs `nangia-etal-2020-crows`
- `crows_pairs_french`, `crows_pairs_french_age`, `crows_pairs_french_autre`, `crows_pairs_french_disability`, `crows_pairs_french_gender`, `crows_pairs_french_nationality`, `crows_pairs_french_physical_appearance`, `crows_pairs_french_race_color`, `crows_pairs_french_religion`, `crows_pairs_french_sexual_orientation`, `crows_pairs_french_socioeconomic`: French CrowS-Pairs `neveol-etal-2022-french`
- `darijahellaswag`: DarijaHellaSwag `shang2024atlaschatadaptinglargelanguage`
- `egyhellaswag`: EgyHellaSwag `mekki2025nilechatlinguisticallydiverseculturally`
- `drop`: DROP `dua2019drop`
- `gpqa_main`, `gpqa_diamond`, `gpqa_extended`: GPQA `rein2024gpqa`
- `ethics_cm`, `ethics_deontology`, `ethics_justice`, `ethics_utilitarianism`, `ethics_virtue`: ETHICS `hendrycks2021ethics`
- `gsm8k`: GSM8K `cobbe2021trainingverifierssolvemath`
- `gsm8k_platinum`: GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks`
- `hellaswag`: HellaSwag `zellers2019hellaswag`
- `headqa_en`, `headqa_es`: HEAD-QA `vilares-gomez-rodriguez-2019-head`
- `histoires_morales`: Histoires Morales `leteno2025histoiresmorales`
- `icelandic_winogrande`: Icelandic WinoGrande `snaebjarnarson-etal-2022-warm`
- `kobest_boolq`, `kobest_copa`, `kobest_hellaswag`, `kobest_sentineg`, `kobest_wic`: KoBEST `kim2022kobest`
- `lambada_openai`, `lambada_openai_cloze`, `lambada_standard`, `lambada_standard_cloze`: LAMBADA `paperno2016lambada`
- `logiqa`: LogiQA `liu2020logiqa`
- `mathqa`: MathQA `amini2019mathqa`
- `mc_taco`: MC-TACO `zhou2019mctaco`
- `medmcqa`: MedMCQA `pmlr-v174-pal22a`
- `medqa_4options`: MedQA `jin2020disease`
- `mmlu`: MMLU `hendryckstest2021`
- `mmlu_pro`: MMLU-Pro `wang2024mmlupro`
- `cola`, `mnli`, `mrpc`, `qnli`, `qqp`, `sst2`, `wnli`: GLUE `wang-etal-2018-glue`
- `nq_open`: Natural Questions `kwiatkowski2019natural`
- `openbookqa`: OpenBookQA `mihaylov2018openbookqa`
- `paws_x_de`, `paws_x_en`, `paws_x_es`, `paws_x_fr`, `paws_x_ja`, `paws_x_ko`, `paws_x_zh`: PAWS-X `yang2019pawsx`
- `piqa`: PIQA `bisk2020piqa`
- `pile_10k`: The Pile `gao2020pile`
- `prost`: PROST `aroca-ouellette-etal-2021-prost`
- `pubmedqa`: PubMedQA `jin2019pubmedqa`
- `qa4mre_2011`, `qa4mre_2012`, `qa4mre_2013`: QA4MRE `Peas2013QA4MRE2O`
- `squadv2`: SQuAD 2.0 `rajpurkar2018know`
- `triviaqa`: TriviaQA `joshi2017triviaqa`
- `race`: RACE `lai-etal-2017-race`
- `sciq`: SciQ `welbl2017crowdsourcing`
- `siqa`: Social IQA `sap2019social`
- `swag`: SWAG `zellers2018swagaf`
- `webqs`: WebQuestions `berant-etal-2013-semantic`
- `wikitext`: WikiText-2 `merity2016pointer`
- `winogender_all`, `winogender_female`, `winogender_gotcha`, `winogender_gotcha_female`, `winogender_gotcha_male`, `winogender_male`, `winogender_neutral`: WinoGender `rudinger2018winogender`
- `wsc273`: WSC273 `levesque2012winograd`
- `winogrande`: WinoGrande `sakaguchi2019winogrande`
- `xcopa_et`, `xcopa_ht`, `xcopa_id`, `xcopa_it`, `xcopa_qu`, `xcopa_sw`, `xcopa_ta`, `xcopa_th`, `xcopa_tr`, `xcopa_vi`, `xcopa_zh`: XCOPA `ponti2020xcopa`
- `xstorycloze_ar`, `xstorycloze_en`, `xstorycloze_es`, `xstorycloze_eu`, `xstorycloze_hi`, `xstorycloze_id`, `xstorycloze_my`, `xstorycloze_ru`, `xstorycloze_sw`, `xstorycloze_te`, `xstorycloze_zh`: XStoryCloze `lin2021fewshotmultilingual`
- `xnli_<language>` for the built-in XNLI language subsets: XNLI `conneau2018xnli`
- `xwinograd_en`, `xwinograd_fr`, `xwinograd_jp`, `xwinograd_pt`, `xwinograd_ru`, `xwinograd_zh`: XWinograd `tikhonov2021heads`

```bibtex

# EXAMS
@inproceedings{hardalov-etal-2020-exams,
  title = {EXAMS: A Multi-Subject High School Examinations Dataset for Cross-Lingual and Multilingual Question Answering},
  author = {Momchil Hardalov and Todor Mihaylov and Vassil Momchev and Pepa Atanasova and Preslav Nakov and Iryna Gurevych},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020},
  pages = {3407--3414},
  url = {https://aclanthology.org/2020.emnlp-main.438/},
  doi = {10.18653/v1/2020.emnlp-main.438},
}

# AIME
@dataset{aime_1983_2024,
  author = {Hemish Veeraboina},
  title = {AIME Problem Set 1983-2024},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024},
}

@dataset{aime_2024,
  author = {Maxwell Jia},
  title = {AIME Problem Set 2024},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/Maxwell-Jia/AIME_2024},
}

@dataset{aime_2025,
  author = {math-ai},
  title = {AIME Problem Set 2025},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/math-ai/aime25},
}

# AGIEval
@article{zhong2023agieval,
  title = {AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
  author = {Wanjun Zhong and Zijie Huang and Shirong Ma and Angelica Chen and Yuxin Wang and Li Dong and Jie Tang and Nan Duan},
  journal = {arXiv preprint arXiv:2304.06364},
  year = {2023},
  url = {https://arxiv.org/abs/2304.06364},
}

# ANLI
@inproceedings{nie-etal-2020-adversarial,
  title = {Adversarial NLI: A New Benchmark for Natural Language Understanding},
  author = {Yixin Nie and Adina Williams and Emily Dinan and Mohit Bansal and Jason Weston and Douwe Kiela},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year = {2020},
}

# bAbI
@article{weston2015towards,
  title = {Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks},
  author = {Jason Weston and Antoine Bordes and Sumit Chopra and Alexander M. Rush and Bart van Merri{\"e}nboer and Armand Joulin and Tomas Mikolov},
  journal = {arXiv preprint arXiv:1502.05698},
  year = {2015},
}

# Bangla
@misc{nahin2025titullmsfamilybanglallms,
  title = {TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking},
  author = {Shahriar Kabir Nahin and Rabindra Nath Nandi and Sagor Sarker and Quazi Sarwar Muhtaseem and Md Kowsher and Apu Chandraw Shill and Md Ibrahim and Mehadi Hasan Menon and Tareq Al Muntasir and Firoj Alam},
  year = {2025},
  eprint = {2502.11187},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2502.11187},
}

# BEAR
@article{wiland2024bear,
  title = {BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models},
  author = {Jacek Wiland and Max Ploner and Alan Akbik},
  journal = {arXiv preprint arXiv:2404.04113},
  year = {2024},
  url = {https://arxiv.org/abs/2404.04113},
}

# Belebele
@misc{bandarkar2023belebele,
  title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  author = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
  year = {2023},
  eprint = {2308.16884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2308.16884},
}

# BLiMP
@article{warstadt2020blimp,
  title = {BLiMP: The Benchmark of Linguistic Minimal Pairs for English},
  author = {Alex Warstadt and Alicia Parrish and Haokun Liu and Anhad Mohananey and Wei Peng and Sheng-Fu Wang and Samuel R. Bowman},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {8},
  pages = {377--392},
  year = {2020},
  doi = {10.1162/tacl_a_00321},
  url = {https://doi.org/10.1162/tacl_a_00321},
}

# BIG-Bench Hard
@article{suzgun2022challenging,
  title = {Challenging {BIG-Bench} Tasks and Whether Chain-of-Thought Can Solve Them},
  author = {Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V. and Chi, Ed H. and Zhou, Denny and Wei, Jason},
  journal = {arXiv preprint arXiv:2210.09261},
  year = {2022},
  url = {https://arxiv.org/abs/2210.09261},
}

# BABILong
@article{kuratov2024babilong,
  title = {BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
  author = {Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Burtsev, Mikhail},
  journal = {arXiv preprint arXiv:2406.10149},
  year = {2024},
  url = {https://arxiv.org/abs/2406.10149},
}

# C4
@article{raffel2020exploring,
  title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  journal = {Journal of Machine Learning Research},
  volume = {21},
  number = {140},
  pages = {1--67},
  year = {2020},
  url = {https://jmlr.org/papers/v21/20-074.html},
}

# C-Eval
@article{huang2023ceval,
  title = {C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
  author = {Yuzhen Huang and Yidong Wang and Chunyang Wang and Lei Chen and Xiaonan Luo and Yuxin Tang and Zhilin Yang and Qianqian Wang and Cheng Li and Weijian Yin and others},
  journal = {arXiv preprint arXiv:2305.08322},
  year = {2023},
  url = {https://arxiv.org/abs/2305.08322},
}

# DarijaHellaSwag
@article{shang2024atlaschatadaptinglargelanguage,
  title = {Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect},
  author = {Shang, Guokan and others},
  journal = {arXiv preprint arXiv:2409.17912},
  year = {2024},
  url = {https://arxiv.org/abs/2409.17912},
}

# EgyHellaSwag
@article{mekki2025nilechatlinguisticallydiverseculturally,
  title = {NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities},
  author = {El Mekki, Abdellah and Atou, Houdaifa and Nacar, Omer and Shehata, Shady and Abdul-Mageed, Muhammad},
  journal = {arXiv preprint arXiv:2505.18383},
  year = {2025},
  url = {https://arxiv.org/abs/2505.18383},
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

# Histoires Morales
@article{leteno2025histoiresmorales,
  title = {Histoires Morales: A French Dataset for Assessing Moral Alignment},
  author = {Leteno, Thibaud and Proskurina, Irina and Gourru, Antoine and Velcin, Julien and Laclau, Charlotte and Metzler, Guillaume and Gravier, Christophe},
  journal = {arXiv preprint arXiv:2501.17117},
  year = {2025},
  url = {https://arxiv.org/abs/2501.17117},
}

# Icelandic WinoGrande
@inproceedings{snaebjarnarson-etal-2022-warm,
  title = {A Warm Start and a Clean Crawled Corpus - A Recipe for Good Language Models},
  author = {Sn{\ae}bjarnarson, V{\'e}steinn and S{\'i}monarson, Haukur Barri and Ragnarsson, P{\'e}tur Orri and Ing{\'o}lfsd{\'o}ttir, Svanhv{\'i}t Lilja and J{\'o}nsson, Haukur and Thorsteinsson, Vilhjalmur and Einarsson, Hafsteinn},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year = {2022},
  address = {Marseille, France},
  publisher = {European Language Resources Association},
  url = {https://aclanthology.org/2022.lrec-1.464/},
  pages = {4356--4366},
}

# GPQA
@inproceedings{rein2024gpqa,
  title = {{GPQA}: A Graduate-Level Google-Proof {Q\&A} Benchmark},
  author = {David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
  booktitle = {First Conference on Language Modeling},
  year = {2024},
  url = {https://openreview.net/forum?id=Ti67584b98},
}

# ARC
@article{clark2018arc,
  title = {Think you have Solved Question Answering? Try {ARC}, the {AI2} Reasoning Challenge},
  author = {Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal = {arXiv preprint arXiv:1803.05457},
  year = {2018},
}

# GPT-3 Arithmetic
@inproceedings{brown2020gpt3,
  title = {Language Models are Few-Shot Learners},
  author = {Tom Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel Ziegler and Jeffrey Wu and Clemens Winter and Chris Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}

# CareQA
@inproceedings{arias-duart-etal-2025-automatic,
  title = {Automatic Evaluation of Healthcare LLMs Beyond Question-Answering},
  author = {Arias-Duart, Anna and Bernabeu, Pablo and Lopez, Adria and Hadj Taieb, Meriem and Villegas, Marta and Gonzalez-Agirre, Aitor},
  booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2025},
  url = {https://arxiv.org/abs/2502.06666},
}

# CaBBQ / EsBBQ
@misc{ruizfernández2025esbbqcabbqspanishcatalan,
  title = {EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering},
  author = {Valle Ruiz-Fernández and Mario Mina and Júlia Falcão and Luis Vasquez-Reina and Anna Sallés and Aitor Gonzalez-Agirre and Olatz Perez-de-Viñaspre},
  year = {2025},
  eprint = {2507.11216},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2507.11216},
}

# XNLI
@inproceedings{conneau2018xnli,
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  author = {Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel R. and Schwenk, Holger and Stoyanov, Veselin},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
  pages = {2475--2485},
  url = {https://aclanthology.org/D18-1269/},
  doi = {10.18653/v1/D18-1269},
}

# ASDiv
@article{miao2021diverse,
  title = {A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
  author = {Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
  journal = {arXiv preprint arXiv:2106.15772},
  year = {2021},
}

# PIQA
@inproceedings{bisk2020piqa,
  title = {PIQA: Reasoning about Physical Commonsense in Natural Language},
  author = {Yonatan Bisk and Rowan Zellers and Ronan Le Bras and Jianfeng Gao and Yejin Choi},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2020},
}

# AlGhafa
@inproceedings{almazrouei-etal-2023-alghafa,
  title = {AlGhafa Evaluation Benchmark for Arabic Language Models},
  author = {Almazrouei, Ebtesam and Cojocaru, Ruxandra and Baldo, Michele and Malartic, Quentin and Alobeidli, Hamza and Mazzotta, Daniele and Penedo, Guilherme and Campesan, Giulia and Farooq, Mugariya and Alhammadi, Maitha and Launay, Julien and Noune, Badreddine},
  booktitle = {Proceedings of ArabicNLP 2023},
  month = dec,
  year = {2023},
  address = {Singapore (Hybrid)},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2023.arabicnlp-1.21},
  doi = {10.18653/v1/2023.arabicnlp-1.21},
  pages = {244--275},
}

# The Pile
@article{gao2020pile,
  title = {The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
  author = {Leo Gao and Stella Biderman and Sid Black and Laurence Golding and Travis Hoppe and Charles Foster and Jason Phang and Horace He and Anish Thite and Noa Nabeshima and Shawn Presser and Connor Leahy},
  journal = {arXiv preprint arXiv:2101.00027},
  year = {2020},
}

# PROST
@inproceedings{aroca-ouellette-etal-2021-prost,
  title = {{PROST}: Physical Reasoning about Objects through Space and Time},
  author = {St{\'e}phane Aroca-Ouellette and Cory Paik and Alessandro Roncone and Katharina Kann},
  booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  year = {2021},
  pages = {4597--4608},
  url = {https://aclanthology.org/2021.findings-acl.404},
}

# SQuAD 2.0
@article{rajpurkar2018know,
  title = {Know What You Don’t Know: Unanswerable Questions for SQuAD},
  author = {Pranav Rajpurkar and Robin Jia and Percy Liang},
  journal = {arXiv preprint arXiv:1806.03822},
  year = {2018},
}

# TriviaQA
@article{joshi2017triviaqa,
  title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
  author = {Mandar Joshi and Eunsol Choi and Daniel Weld and Luke Zettlemoyer},
  journal = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year = {2017},
}

# Natural Questions
@article{kwiatkowski2019natural,
  title = {Natural Questions: A Benchmark for Question Answering Research},
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Jacob Devlin and Kenton Lee and Kristina Toutanova and Llion Jones and Matthew Kelcey and Ming-Wei Chang and Andrew M. Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  pages = {452--466},
  year = {2019},
}

# PubMedQA
@inproceedings{jin2019pubmedqa,
  title = {PubMedQA: A Dataset for Biomedical Research Question Answering},
  author = {Qiao Jin and Bhuwan Dhingra and Zhengping Liu and William Cohen and Xinghua Lu},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
  pages = {2567--2577},
}

# QA4MRE
@inproceedings{Peas2013QA4MRE2O,
  title = {QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation},
  author = {Pe{\~n}as, Anselmo and Hovy, Eduard H. and Forner, Pamela and Rodrigo, {\'A}lvaro and Sutcliffe, Richard F. E. and Morante, Roser},
  booktitle = {CLEF},
  year = {2013},
}

# CommonsenseQA
@inproceedings{talmor2019commonsenseqa,
  title = {CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author = {Alon Talmor and Jonathan Herzig and Nicholas Lourie and Jonathan Berant},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019},
}

# CoQA
@article{reddy2019coqa,
  title = {CoQA: A Conversational Question Answering Challenge},
  author = {Siva Reddy and Danqi Chen and Christopher D. Manning},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  pages = {249--266},
  year = {2019},
}

# CrowS-Pairs
@inproceedings{nangia-etal-2020-crows,
  title = {CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models},
  author = {Nikita Nangia and Clara Vania and Rasika Bhalerao and Samuel R. Bowman},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020},
  pages = {1953--1967},
  url = {https://aclanthology.org/2020.emnlp-main.154/},
  doi = {10.18653/v1/2020.emnlp-main.154},
}

@inproceedings{neveol-etal-2022-french,
  title = {French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English},
  author = {Aur{\'e}lie N{\'e}v{\'e}ol and Yoann Dupont and Julien Bezan{\c{c}}on and Kar{\"e}n Fort},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2022},
  pages = {8521--8531},
  url = {https://aclanthology.org/2022.acl-long.583/},
  doi = {10.18653/v1/2022.acl-long.583},
}

# COPAL-ID
@inproceedings{wibowo-etal-2024-copal,
  title = {COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances},
  author = {Haryo Akbarianto Wibowo and Swandana Rama Sandhiyudha and Genta Indra Winata and Ayu Purwarianti and Sebastian Ruder and Rahmad Mahardhika and Pascale Fung},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {1393--1410},
  year = {2024},
  url = {https://aclanthology.org/2024.naacl-long.77/},
}

# DROP
@inproceedings{dua2019drop,
  title = {DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
  author = {Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019},
}

# ETHICS
@article{hendrycks2021ethics,
  title = {Aligning AI With Shared Human Values},
  author = {Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal = {International Conference on Learning Representations},
  year = {2021},
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

# ArabicMMLU
@misc{koto2024arabicmmlu,
  title = {ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic},
  author = {Koto, Fajri and Li, Haonan and Shatnawi, Sara and Doughman, Jad and Sadallah, Abdelrahman Boda and Alraeesi, Aisha and Almubarak, Khalid and Alyafeai, Zaid and Sengupta, Neha and Shehata, Shady and Habash, Nizar and Nakov, Preslav and Baldwin, Timothy},
  year = {2024},
  eprint = {2402.12840},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2402.12840},
}

# DarijaMMLU
@article{shang2024atlaschatadaptinglargelanguage,
  title = {Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect},
  author = {Guokan Shang and Hadi Abdine and Yousef Khoubrane and Amr Mohamed and Yassine Abbahaddou and Sofiane Ennadir and Imane Momayiz and Xuguang Ren and Eric Moulines and Preslav Nakov and Michalis Vazirgiannis and Eric Xing},
  year = {2024},
  eprint = {2409.17912},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2409.17912},
}

# EgyMMLU
@inproceedings{el-mekki-etal-2025-nilechat,
  title = {NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities},
  author = {El Mekki, Abdellah and Atou, Houdaifa and Nacar, Omer and Shehata, Shady and Abdul-Mageed, Muhammad},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year = {2025},
  address = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2025.emnlp-main.556/},
  doi = {10.18653/v1/2025.emnlp-main.556},
}

# EusExams
@misc{etxaniz2024latxa,
  title = {Latxa: An Open Language Model and Evaluation Suite for Basque},
  author = {Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
  year = {2024},
  eprint = {2403.20266},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2403.20266},
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

# HEAD-QA
@inproceedings{vilares-gomez-rodriguez-2019-head,
  title = {HEAD-QA: A Healthcare Dataset for Complex Reasoning},
  author = {David Vilares and Carlos G{\'o}mez-Rodr{\'i}guez},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year = {2019},
}

# KoBEST
@misc{kim2022kobest,
  title = {KoBEST: Korean Balanced Evaluation of Significant Tasks},
  author = {Dohyeong Kim and Myeongjun Jang and Deuk Sin Kwon and Eric Davis},
  year = {2022},
  eprint = {2204.04541},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2204.04541},
}

# IrokoBench
@inproceedings{adelani2025irokobench,
  title = {IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models},
  author = {David Ifeoluwa Adelani and Jessica Ojo and Israel Abebe Azime and Jian Yun Zhuang and Jesujoba O. Alabi and others},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages = {2732--2757},
  year = {2025},
  url = {https://aclanthology.org/2025.naacl-long.139/},
}

# LAMBADA
@misc{paperno2016lambada,
  title = {The LAMBADA dataset: Word prediction requiring a broad discourse context},
  author = {Denis Paperno and Germ{\'a}n Kruszewski and Angeliki Lazaridou and Quan Ngoc Pham and Raffaella Bernardi and Sandro Pezzelle and Marco Baroni and Gemma Boleda and Raquel Fern{\'a}ndez},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.2630551},
  year = {2016},
}

# LogiQA
@misc{liu2020logiqa,
  title = {LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
  author = {Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
  year = {2020},
  eprint = {2007.08124},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

# MathQA
@misc{amini2019mathqa,
  title = {MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms},
  author = {Aida Amini and Saadia Gabriel and Peter Lin and Rik Koncel-Kedziorski and Yejin Choi and Hannaneh Hajishirzi},
  year = {2019},
  eprint = {1905.13319},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

# MC-TACO
@inproceedings{zhou2019mctaco,
  title = {Going on a vacation takes longer than going for a walk: A Study of Temporal Commonsense Understanding},
  author = {Ben Zhou and Daniel Khashabi and Qiang Ning and Dan Roth},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}

# MedMCQA
@inproceedings{pmlr-v174-pal22a,
  title = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author = {Ankit Pal and Logesh Kumar Umapathi and Malaikannan Sankarasubbu},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  year = {2022},
}

# MedQA
@article{jin2020disease,
  title = {What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author = {Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
  journal = {arXiv preprint arXiv:2009.13081},
  year = {2020},
}

# OpenBookQA
@inproceedings{mihaylov2018openbookqa,
  title = {Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
  author = {Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
}

# PAWS-X
@inproceedings{yang2019pawsx,
  title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
  author = {Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
}

# CNN/DailyMail
@article{nallapati2016abstractive,
  title = {Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond},
  author = {Ramesh Nallapati and Bowen Zhou and Cicero dos Santos and {\c{C}}aglar Gul{\c{c}}ehre and Bing Xiang},
  journal = {arXiv preprint arXiv:1602.06023},
  year = {2016},
}

# CodeXGLUE
@inproceedings{lu2021codexglue,
  title = {CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author = {Shuai Lu and Daya Guo and Shuo Ren and Junjie Huang and Alexey Svyatkovskiy and Ambrosio Blanco and Colin B. Clement and Dawn Drain and Daxin Jiang and Duyu Tang and Ge Li and Lidong Zhou and Linjun Shou and Long Zhou and Michele Tufano and Ming Gong and Ming Zhou and Nan Duan and Neel Sundaresan and Shao Kun Deng and Shengyu Fu and Shujie Liu},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year = {2021},
}

# XCOPA
@inproceedings{ponti2020xcopa,
  title = {XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning},
  author = {Edoardo M. Ponti and Rahul Gupta and Ivan Vuli{\'c} and Goran Glava{\v{s}} and Anna Korhonen},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year = {2020},
}

# SciQ
@inproceedings{welbl2017crowdsourcing,
  title = {Crowdsourcing Multiple Choice Science Questions},
  author = {Johannes Welbl and Nelson F. Liu and Matt Gardner},
  booktitle = {Proceedings of the 3rd Workshop on Noisy User-generated Text},
  year = {2017},
}

# Social IQA
@inproceedings{sap2019social,
  title = {Social IQa: Commonsense Reasoning about Social Interactions},
  author = {Maarten Sap and Hannah Rashkin and Derek Chen and Ronan Le Bras and Yejin Choi},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
}

# SWAG
@inproceedings{zellers2018swagaf,
  title = {SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
  author = {Rowan Zellers and Yonatan Bisk and Roy Schwartz and Yejin Choi},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
}

# WebQuestions
@inproceedings{berant-etal-2013-semantic,
  title = {Semantic Parsing on Freebase from Question-Answer Pairs},
  author = {Jonathan Berant and Andrew Chou and Roy Frostig and Percy Liang},
  booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
  year = {2013},
  pages = {1533--1544},
  url = {https://aclanthology.org/D13-1160},
}

# WikiText-2
@misc{merity2016pointer,
  title = {Pointer Sentinel Mixture Models},
  author = {Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
  year = {2016},
  eprint = {1609.07843},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

# WinoGender
@inproceedings{rudinger2018winogender,
  title = {Gender Bias in Coreference Resolution},
  author = {Rachel Rudinger and Jason Naradowsky and Brian Leonard and Benjamin Van Durme},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2018},
}

# WSC273
@inproceedings{levesque2012winograd,
  title = {The Winograd Schema Challenge},
  author = {Hector Levesque and Ernest Davis and Leora Morgenstern},
  booktitle = {Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning},
  year = {2012},
}

# XWinograd
@misc{tikhonov2021heads,
  title = {It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
  author = {Alexey Tikhonov and Max Ryabinin},
  year = {2021},
  eprint = {2106.12066},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

# XStoryCloze
@article{lin2021fewshotmultilingual,
  title = {Few-shot Learning with Multilingual Language Models},
  author = {Xi Victoria Lin and Todor Mihaylov and Mikel Artetxe and Tianlu Wang and Shuohui Chen and Daniel Simig and Myle Ott and Naman Goyal and Shruti Bhosale and Jingfei Du and Ramakanth Pasunuru and Sam Shleifer and Punit Singh Koura and Vishrav Chaudhary and Brian O'Horo and Jeff Wang and Luke Zettlemoyer and Zornitsa Kozareva and Mona T. Diab and Veselin Stoyanov and Xian Li},
  journal = {arXiv preprint arXiv:2112.10668},
  year = {2021},
}

# WinoGrande
@article{sakaguchi2019winogrande,
  title = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
  author = {Keisuke Sakaguchi and Ronan Le Bras and Chandra Bhagavatula and Yejin Choi},
  journal = {arXiv preprint arXiv:1907.10641},
  year = {2019},
}
```
