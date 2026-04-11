<p align=center>
<div align=center>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/cda318d5-7f4c-4dc9-8fa8-2b48afcd4c29" />


</div>
<h1 align="center">Evalution</h1>

Evalution is a modern LLM evaluation toolkit for fast, benchmark-faithful, multi-engine model evaluation. 🚀

Install ⚡

```bash
pip install Evalution
```

Install from source 🛠️

```bash
git clone https://github.com/modelcloud/Evalution.git
cd Evalution
pip install .
```

Core runtime dependencies stay lean: `transformers`, `datasets`, `logbar`, `PyPcre`, and `tokenicer`. 🪶

## Why Evalution ✨

**8 engines. 126 built-in benchmark families. 213 in-repo GPU benchmark regression tests.**

- 🚂 Multi-engine out of the box: `Transformers`, `TransformersCompat`, `VLLM`, `SGLang`, `TensorRTLLM`, `OpenAICompatible`, `GPTQModel`, and `OpenVINO`.
- 📚 Broad benchmark coverage: 126 documented built-in benchmark families spanning reasoning, multilingual evals, coding, long-context, QA, perplexity, safety, and more.
- 🧪 GPU validated: the repo includes 213 in-repo GPU benchmark regression tests for individual Llama 3.2 benchmark runs, with RTX 4090 and A100-aware baselines where scores are pinned.
- ⚡ Speed: Evalution is faster than many evaluators and attempts continuous batching by default when the selected engine supports it.
- 🪶 Minimal core deps: the default install stays focused and avoids dragging in every backend dependency up front.
- 🧼 Clean API: Python, YAML, CLI, single-model runs, and compare flows all share the same readable shape.
- 🧩 Extensible: Evalution has a clean extension API for execution engines, benchmark suites, and scorers.
- 📝 YAML support: run configs from YAML and emit Python back from YAML when you want a code path.
- 🔌 Easy to extend: adding new engines and new benchmark suites follows a clear public contract with contributor docs in-tree.
- ⚖️ Side-by-side compare mode: run threaded left/right model lanes against the same suite list and get one consolidated summary.
- 🎯 Benchmark-faithful by default: suites aim to stay close to original papers and released reference behavior, not just headline numbers.

Engine implementation notes for backend authors live in [docs/engine.md](docs/engine.md).
Metric-key glossary lives in [docs/scores.md](docs/scores.md). Scoring implementation notes and
scorer-module mapping live in [docs/scorers.md](docs/scorers.md).
Contributor guidance for adding new eval/benchmark/test suites lives in
[docs/benchmark_suite_guidelines.md](docs/benchmark_suite_guidelines.md).

Simple usage 🧪

```python
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers()
    .model(path="meta-llama/Llama-3.2-1B-Instruct")
    .run(benchmarks.gsm8k_platinum())
)
```

Advanced usage 🧠

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
        max_blocks_per_request=None,
        use_async_batching=None,
        use_cuda_graph=None,
        max_new_tokens=256,
    )
    .model(path="meta-llama/Llama-3.2-1B-Instruct")
    .run(
        benchmarks.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            max_new_tokens=96,
            batch_size=64,
        )
    )
    .run(
        benchmarks.gsm8k_platinum(
            batch_size=64,
        )
    )
)
```

The chained object is already the completed run handle. Accessing `result.model`, `result.engine`,
`result.tests`, or `result.to_dict()` finalizes the run and closes the engine session implicitly.

Compare usage ⚖️

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    eval.compare(
        engines.Transformers(dtype="bfloat16", device="cuda:0").model(
            path="meta-llama/Llama-3.2-1B-Instruct",
            label="llama",
        ),
        engines.TransformersCompat(device="cuda:1").model(
            path="/monster/data/Qwen2.5-1.5B-Instruct",
            label="qwen",
        ),
    )
    .run(benchmarks.gsm8k_platinum())
    .run(benchmarks.gsm8k_platinum(variant="cot"))
)
```

`compare(...)` takes the same `engine.model(...)` handles used for single-model runs, so
single and compare flows share one fluent entry shape. Compare lane labels come from
`.model(..., label="...")`; when omitted, Evalution falls back to the model path. It runs the same
suite list on both lanes while allowing different engines and model configs on the left and right.
When the terminal supports LogBar split panes, Evalution binds each lane to its own pane and
renders a consolidated compare summary when the run closes.

YAML usage 📝

```yaml
engine:
  type: Transformers
  dtype: bfloat16
  attn_implementation: paged|flash_attention_2
  device: cuda:0
  max_blocks_per_request: null
  use_cuda_graph: null

model:
  path: meta-llama/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
    variant: cot
    apply_chat_template: true
    max_new_tokens: 96
    batch_size: 64
  - type: gsm8k_platinum
    batch_size: 64
```

```python
import evalution as eval

result = eval.run_yaml("evalution.yaml")

python_script = eval.python_from_yaml("evalution.yaml")
```

CLI usage 💻

```bash
evalution evalution.yaml
evalution run evalution.yaml
evalution run evalution.yaml --output result.json
evalution emit-python evalution.yaml
```

`engines.Transformers(...)` accepts runtime options such as `dtype`, `device`, `batch_size`,
`attn_implementation`, and `max_new_tokens`.

`engines.VLLM(...)` accepts vLLM runtime options such as `tensor_parallel_size`,
`gpu_memory_utilization`, `max_model_len`, `quantization`, `tokenizer_mode`, and
`enforce_eager`.

`engines.SGLang(...)` accepts SGLang runtime options such as `tp_size`,
`mem_fraction_static`, `context_length`, `quantization`, `attention_backend`, `sampling_backend`, `tokenizer_mode`,
and `max_running_requests`.

`engines.LlamaCpp(...)` accepts llama.cpp runtime options such as `device`, `n_ctx`,
`n_gpu_layers`, `flash_attn`, `main_gpu`, `llama_cpp_path`, and `llama_kwargs`.
Its `continuous_batching=True` mode schedules multiple in-flight requests together but still
returns one final completion per request rather than streaming partial tokens to the caller.

`engines.OpenVINO(...)` accepts OpenVINO runtime options such as `dtype`, `device`, `batch_size`, `attn_implementation`, `max_new_tokens` and `ov_config`.

Per-benchmark options such as `apply_chat_template`, `batch_size`, `max_new_tokens`, `max_rows`,
`order`, and scorer-specific options like `label_permutations` can be set directly on each benchmark
call or in each YAML `tests` entry.

## Benchmark Row Order 🔀

Dataset-backed benchmarks support an `order` override that controls benchmark row traversal order.
This is a benchmark-level dataset ordering control, separate from any engine-internal request
reordering used for batching efficiency.

Supported values:

- `native`: preserve the dataset loader's row order. This is the default.
- `shuffle`: shuffle rows deterministically with an implicit seed of `7`.
- `shuffle|245`: shuffle rows deterministically with the provided integer seed.
- `length|asc`: execute shorter prepared requests first.
- `length|desc`: execute longer prepared requests first.

Python:

```python
import evalution.benchmarks as benchmarks

suite = benchmarks.gsm8k(order="length|desc")
suite = benchmarks.gsm8k_platinum(order="shuffle|245")
```

YAML:

```yaml
tests:
  - type: gsm8k
    order: length|desc
  - type: gsm8k_platinum
    order: shuffle|245
```

Notes:

- `shuffle` without an explicit seed is normalized to `shuffle|7`.
- Ordering is applied after the benchmark's selected rows are loaded and capped by `max_rows`.
- When `stream=True`, `order` must stay `native`.

## Subset Selection 🧭

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
    .model(path="meta-llama/Llama-3.2-1B-Instruct")
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

## Engine Samples 🔌

Each exported engine has a minimal sample below. `Transformers` covers both the modern
`Transformers` engine and the fixed-batch `TransformersCompat` engine; swap `Transformers` for
`TransformersCompat` or `type: Transformers` for `type: TransformersCompat` when you need the
compatibility backend explicitly.

### Transformers / TransformersCompat 🤗

Use `engines.Transformers()` in Python or `engine.type: Transformers` in YAML when you want the
preferred Hugging Face runtime.

Python:

```python
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.Transformers(
        dtype="bfloat16",
        device="cuda:0",
    )
    .model(path="meta-llama/Llama-3.2-1B-Instruct")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: Transformers
  dtype: bfloat16
  device: cuda:0

model:
  path: meta-llama/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
```

### SGLang ⚡

Use `engines.SGLang()` in Python or `engine.type: SGLang` in YAML when you want the SGLang runtime.
Evalution will preserve `generate(...)`, `generate_continuous(...)`, `loglikelihood(...)`, and
`loglikelihood_rolling(...)` through the same shared engine contract. The current sglang backend
expects `num_beams=1`.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

# Extra dependency: `pip install sglang`
if __name__ == '__main__':
    result = (
        engines.SGLang(
            batch_size=16,
            mem_fraction_static=0.8,
        )
        .model(path="/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")
        .run(benchmarks.gsm8k_platinum())
    )
```

YAML:

```yaml
engine:
  type: SGLang
  batch_size: 16
  tp_size: 1

model:
  path: /monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit

tests:
  - type: gsm8k_platinum
```

### VLLM 🚀

Use `engines.VLLM()` in Python or `engine.type: VLLM` in YAML when you want the vLLM runtime.
Evalution will preserve `generate(...)`, `generate_continuous(...)`, `loglikelihood(...)`, and
`loglikelihood_rolling(...)` through the same shared engine contract. The current vLLM backend
expects `num_beams=1`.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

# Extra dependency: `pip install vllm`
result = (
    engines.VLLM(
        batch_size=16,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    .model(path="/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: VLLM
  batch_size: 16
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  enforce_eager: true

model:
  path: /monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit

tests:
  - type: gsm8k_platinum
```

### TensorRTLLM 🧠

Use `engines.TensorRTLLM()` in Python or `engine.type: TensorRTLLM` in YAML when you want the
TensorRT-LLM runtime. Configure `tensorrt_llm_path` only when `tensorrt_llm` is not importable
from the active environment. The current backend expects `num_beams=1`.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

# Extra dependency: install `tensorrt_llm`, or set `tensorrt_llm_path` to a local checkout.
result = (
    engines.TensorRTLLM(
        batch_size=16,
        tensor_parallel_size=1,
    )
    .model(path="/monster/data/model/Llama-3.2-1B-Instruct")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: TensorRTLLM
  batch_size: 16
  tensor_parallel_size: 1

model:
  path: /monster/data/model/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
```

### OpenAICompatible 🌐

Use `engines.OpenAICompatible()` in Python or `engine.type: OpenAICompatible` in YAML when you
want to evaluate through an OpenAI-compatible HTTP endpoint. Evalution expects generation routes
such as `/v1/chat/completions` or `/v1/completions`, plus the Evalution scoring routes
`/v1/eval/loglikelihood` and `/v1/eval/loglikelihood/rolling`. Evalution still uses
`.model(...)` for its shared run API, and this engine converts `.model(path=...)` into the remote
OpenAI-compatible HTTP `model` argument.

Python:

```python
import os

import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.OpenAICompatible(
        base_url="http://127.0.0.1:8000",
        api_key=os.environ["OPENAI_API_KEY"],
        batch_size=4,
    )
    .model(path="meta-llama/Llama-3.2-1B-Instruct")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: OpenAICompatible
  base_url: http://127.0.0.1:8000
  api_key: ${OPENAI_API_KEY}
  batch_size: 4

model:
  path: meta-llama/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
```

### GPTQModel 🪶

Use `engines.GPTQModel()` in Python or `engine.type: GPTQModel` in YAML when you want to load a
quantized checkpoint through GPTQModel's native loader. Configure `gptqmodel_path` only when the
runtime is not importable from the active environment.

Python:

```python
import evalution.benchmarks as benchmarks
import evalution.engines as engines

# Extra dependency: `pip install gptqmodel`
result = (
    engines.GPTQModel(
        device="cuda:0",
        backend="auto",
        batch_size=16,
    )
    .model(path="/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: GPTQModel
  device: cuda:0
  backend: auto
  batch_size: 16

model:
  path: /monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit

tests:
  - type: gsm8k_platinum
```

### LlamaCpp 🦙

Use `engines.LlamaCpp()` in Python or `engine.type: LlamaCpp` in YAML when you want a
`llama.cpp` backend through `llama-cpp-python`. Evalution keeps generation, native
`generate_continuous(...)`, `loglikelihood(...)`, and `loglikelihood_rolling(...)` on the same
shared engine contract. The current backend expects `num_beams=1`.

Install from source when you need CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on -DCUDAToolkit_ROOT=/usr/local/cuda-12.8" \
FORCE_CMAKE=1 \
pip install --no-binary=:all: --force-reinstall llama-cpp-python
```

Notes:

- `device` can be `auto`, `cuda`, `cpu`, or `mlx`. When a GPU-backed request is not available in
  the installed binding, Evalution falls back to CPU instead of aborting engine construction.
- `continuous_batching` defaults to `True` and uses llama.cpp's native multi-sequence batch API.
  Evalution enables the required unified-KV multi-sequence runtime internally and admits requests
  by shared `n_ctx` budget, so large prompts do not overcommit the native scheduler. Set
  `continuous_batching=False` if you want regular fixed-size batching instead.
- `LlamaCpp` uses llama.cpp's native tokenizer for prompt tokenization and scoring. An optional
  Hugging Face tokenizer is only loaded when needed for chat template rendering.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

result = (
    engines.LlamaCpp(
        device="auto",
        continuous_batching=True,
        n_ctx=4096,
        n_gpu_layers=-1,
    )
    .model(
        path="/monster/data/model/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        tokenizer_path="/monster/data/model/Llama-3.2-1B-Instruct",
    )
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: LlamaCpp
  device: auto
  continuous_batching: true
  n_ctx: 4096
  n_gpu_layers: -1

model:
  path: /monster/data/model/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  tokenizer_path: /monster/data/model/Llama-3.2-1B-Instruct

tests:
  - type: gsm8k_platinum
```

For the shared llama.cpp integration test artifact, download
`bartowski/Llama-3.2-1B-Instruct-GGUF` with the `Llama-3.2-1B-Instruct-Q4_K_M.gguf` file and keep
the original Hugging Face tokenizer checkout at `/monster/data/model/Llama-3.2-1B-Instruct`:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    local_dir="/monster/data/model/Llama-3.2-1B-Instruct-GGUF",
)
PY
```

### OpenVINO 🔧

Use `engines.OpenVINO()` in Python or `engine.type: OpenVINO` in YAML when you want to run an
Optimum Intel `OVModelForCausalLM` backend.

Python:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

# Extra dependency: `pip install "optimum[openvino]"`
result = (
    engines.OpenVINO(
        device="cpu",
    )
    .model(path="/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")
    .run(benchmarks.gsm8k_platinum())
)
```

YAML:

```yaml
engine:
  type: OpenVINO
  device: cpu

model:
  path: /monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit

tests:
  - type: gsm8k_platinum
```

`Tokenicer` is used to load tokenizers for the transformer, transformer-compat, OpenVINO, GPTQModel,
vLLM, and optionally LlamaCpp engines. When `engine.model(...)` is called with a model config,
Evalution resolves tokenizer loading in this order:
`tokenizer` (preinitialized object), `tokenizer_path`, then `path`.
`Tokenicer` also applies its normalization stage so pad/eos/bos token IDs are corrected before evaluation.
`LlamaCpp` still uses llama.cpp's native tokenizer for scoring and prompt tokenization; the optional
loaded tokenizer is only used to render chat templates when the caller supplies one.
To inject a custom tokenizer, pass it through `.model(...)` on the model config:

```python
import evalution as eval
import evalution.benchmarks as benchmarks
import evalution.engines as engines

custom_tokenizer = ...

result = (
    engines.Transformers()
    .model(
        path="meta-llama/Llama-3.2-1B-Instruct",
        tokenizer=custom_tokenizer,
    )
    .run(benchmarks.gsm8k_platinum())
)
```

YAML flows can only configure `tokenizer_path`; passing a live tokenizer object is Python-only.

## Supported Benchmarks 📚

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

Variant-heavy families are folded into a single row below. Brace notation indicates the concrete
built-in suite names covered by that row.

| Suite | Original benchmark |
| --- | --- |
| `aexams` | EXAMS `hardalov-etal-2020-exams` |
| `agieval` | AGIEval `zhong2023agieval` |
| `afrimgsm` | IrokoBench AfriMGSM `adelani2025irokobench` |
| `afrimmlu` | IrokoBench AfriMMLU `adelani2025irokobench` |
| `aime`, `aime24`, `aime25`, `aime26` | AIME `aime_1983_2024`, `aime_2024`, `aime_2025`, `aime_2026` |
| `afrixnli` | IrokoBench AfriXNLI `adelani2025irokobench` |
| `anli_{r1,r2,r3}` | ANLI `nie-etal-2020-adversarial` |
| `arabicmmlu` | ArabicMMLU `koto2024arabicmmlu` |
| `darijammlu` | DarijaMMLU `shang2024atlaschatadaptinglargelanguage` |
| `egymmlu` | EgyMMLU `el-mekki-etal-2025-nilechat` |
| `eus_exams` | EusExams `etxaniz2024latxa` |
| `eus_reading` | EusReading `etxaniz2024latxa` |
| `eus_proficiency` | EusProficiency `etxaniz2024latxa` |
| `eus_trivia` | EusTrivia `etxaniz2024latxa` |
| <code>arc_challenge</code>, <code>arc_easy</code>, <code>arc_mt_{da,<wbr>de,<wbr>el,<wbr>es,<wbr>fi,<wbr>hu,<wbr>is,<wbr>it,<wbr>nb,<wbr>pl,<wbr>pt,<wbr>sv}</code> | ARC `clark2018arc` |
| <code>arithmetic_{1dc,<wbr>2da,<wbr>2dm,<wbr>2ds,<wbr>3da,<wbr>3ds,<wbr>4da,<wbr>4ds,<wbr>5da,<wbr>5ds}</code> | GPT-3 arithmetic `brown2020gpt3` |
| `asdiv` | ASDiv `miao2021diverse` |
| `asdiv_cot_llama` | ASDiv `miao2021diverse` |
| `babi` | bAbI `weston2015towards` |
| `babilong` | BABILong `kuratov2024babilong` |
| `bbh` | BIG-Bench Hard `suzgun2022challenging` |
| `bangla` | TituLLMs Bangla benchmarks `nahin2025titullmsfamilybanglallms` |
| `bear`, `bear_big` | BEAR `wiland2024bear` |
| `belebele` | Belebele `bandarkar2023belebele` |
| `bbq` | BBQ `parrish2022bbq` |
| `blimp` | BLiMP `warstadt2020blimp` |
| `c4` | C4 `raffel2020exploring` |
| `careqa` | CareQA `arias-duart-etal-2025-automatic` |
| `cabbq` | CaBBQ `ruizfernández2025esbbqcabbqspanishcatalan` |
| `esbbq` | EsBBQ `ruizfernández2025esbbqcabbqspanishcatalan` |
| `ceval` | C-Eval `huang2023ceval` |
| `cmmlu` | CMMLU `li2023cmmlu` |
| `boolq` | SuperGLUE `wang2019superglue` |
| `cb` | SuperGLUE `wang2019superglue` |
| `multirc` | SuperGLUE `wang2019superglue` |
| `click` | CLIcK `kim-etal-2024-click` |
| `cola` | GLUE `wang-etal-2018-glue` |
| `cnn_dailymail` | CNN/DailyMail `nallapati2016abstractive` |
| `code_x_glue` | CodeXGLUE `lu2021codexglue` |
| `commonsense_qa` | CommonsenseQA `talmor2019commonsenseqa` |
| `coqa` | CoQA `reddy2019coqa` |
| `copa` | SuperGLUE `wang2019superglue` |
| `copa_ar` | AlGhafa `almazrouei-etal-2023-alghafa` |
| `copal_id` | COPAL-ID `wibowo-etal-2024-copal` |
| `crows_pairs` | CrowS-Pairs `nangia-etal-2020-crows`, French CrowS-Pairs `neveol-etal-2022-french` |
| `darijahellaswag` | DarijaHellaSwag `shang2024atlaschatadaptinglargelanguage` |
| `egyhellaswag` | NileChat `el-mekki-etal-2025-nilechat` |
| `drop` | DROP `dua2019drop` |
| `fld` | FLD `morishita2023learning` |
| `fda` | BASED / FDA `arora2024simple` |
| `french_bench_arc_challenge` | FrenchBench ARC-Challenge |
| `gpqa` | GPQA `rein2024gpqa` |
| `gsm_plus`, `gsm_plus_mini` | GSM-Plus `li2024gsmpluscomprehensivebenchmarkevaluating` |
| <code>ethics_{cm,<wbr>deontology,<wbr>justice,<wbr>utilitarianism,<wbr>virtue}</code> | ETHICS `hendrycks2021ethics` |
| `gsm8k` | GSM8K `cobbe2021trainingverifierssolvemath` |
| `gsm8k_fr` | GSM8K `cobbe2021trainingverifierssolvemath`, French translation dataset `cmh2025gsm8kfr` |
| `gsm8k_ko` | GSM8K `cobbe2021trainingverifierssolvemath`, Korean translation dataset `kuotient2024gsm8kko` |
| `gsm8k_platinum` | GSM8K-Platinum `vendrow2025largelanguagemodelbenchmarks` |
| `mgsm` | MGSM `shi2022multilingualchainofthought` |
| `haerae` | HAE-RAE `son-etal-2024-hae` |
| `hellaswag` | HellaSwag `zellers2019hellaswag` |
| `headqa_{en,es}` | HEAD-QA `vilares-gomez-rodriguez-2019-head` |
| `ifeval` | IFEval `zhou2023instruction` |
| `ifeval_pt` | IFEval `zhou2023instruction`, Tucano 2 Cool `klugecorrea2026tucano2cool` |
| `hendrycks_math` | MATH `hendrycks2021measuring` |
| `histoires_morales` | Histoires Morales `leteno2025histoiresmorales` |
| `moral_stories` | Moral Stories `emelin-etal-2021-moral` |
| `kobest` | KoBEST `kim2022kobest` |
| `kmmlu` | KMMLU `son2024kmmlu` |
| `kormedmcqa` | KorMedMCQA `kweon2024kormedmcqa` |
| `icelandic_winogrande` | Icelandic WinoGrande `snaebjarnarson-etal-2022-warm` |
| <code>lambada_{openai,<wbr>openai_cloze,<wbr>standard,<wbr>standard_cloze}</code> | LAMBADA `paperno2016lambada` |
| <code>lambada_openai_mt_{de,<wbr>en,<wbr>es,<wbr>fr,<wbr>it}</code>, <code>lambada_openai_mt_stablelm_{de,<wbr>en,<wbr>es,<wbr>fr,<wbr>it,<wbr>nl,<wbr>pt}</code> | LAMBADA-MT |
| `inverse_scaling` | Inverse Scaling Prize `mckenzie2023inverse` |
| `logiqa` | LogiQA `liu2020logiqa` |
| `logiqa2` | LogiQA 2.0 `liu2022logiqa2` |
| `humaneval` | HumanEval `chen2021evaluatinglargelanguagemodels` |
| `mbpp` | MBPP `austin2021program` |
| `mastermind` | Mastermind |
| `mathqa` | MathQA `amini2019mathqa` |
| `mc_taco` | MC-TACO `zhou2019mctaco` |
| `medmcqa` | MedMCQA `pmlr-v174-pal22a` |
| `medqa_4options` | MedQA `jin2020disease` |
| `mmlu` | MMLU `hendryckstest2021` |
| `mmlu_cf` | MMLU-CF `zhao2024mmlucf` |
| `mmlu_pro` | MMLU-Pro `wang2024mmlupro` |
| `mnli` | GLUE `wang-etal-2018-glue` |
| `mrpc` | GLUE `wang-etal-2018-glue` |
| `mutual` | MuTual `cui2020mutual` |
| `nq_open` | Natural Questions `kwiatkowski2019natural` |
| `openbookqa` | OpenBookQA `mihaylov2018openbookqa` |
| `paws_x` | PAWS-X `yang2019pawsx` |
| `xcopa` | XCOPA `ponti2020xcopa` |
| `polemo2` | KLEJ POLEMO 2.0 `kocon-etal-2019-multi` |
| `xquad` | XQuAD `artetxe2020crosslingual` |
| `xstorycloze` | XStoryCloze `lin2021fewshotmultilingual` |
| `xnli` | XNLI `conneau2018xnli` |
| `xnli_eu` | XNLI-EU `heredia-etal-2024-xnlieu` |
| `xwinograd` | XWinograd `tikhonov2021heads` |
| `piqa` | PIQA `bisk2020piqa` |
| `piqa_ar` | AlGhafa `almazrouei-etal-2023-alghafa` |
| `pile_10k` | The Pile `gao2020pile` |
| `prost` | PROST `aroca-ouellette-etal-2021-prost` |
| `pubmedqa` | PubMedQA `jin2019pubmedqa` |
| `qa4mre` | QA4MRE `Peas2013QA4MRE2O` |
| `qnli` | GLUE `wang-etal-2018-glue` |
| `qqp` | GLUE `wang-etal-2018-glue` |
| `race` | RACE `lai-etal-2017-race` |
| `record` | SuperGLUE ReCoRD `wang2019superglue` |
| `rte` | SuperGLUE `wang2019superglue` |
| `sciq` | SciQ `welbl2017crowdsourcing` |
| `siqa` | Social IQA `sap2019social` |
| `swag` | SWAG `zellers2018swagaf` |
| `toxigen` | ToxiGen `hartvigsen-etal-2022-toxigen` |
| `sst2` | GLUE `wang-etal-2018-glue` |
| `squadv2` | SQuAD 2.0 `rajpurkar2018know` |
| `truthfulqa` | TruthfulQA `lin-etal-2022-truthfulqa` |
| `triviaqa` | TriviaQA `joshi2017triviaqa` |
| `wic` | SuperGLUE `wang2019superglue` |
| `webqs` | WebQuestions `berant-etal-2013-semantic` |
| `wikitext` | WikiText-2 `merity2016pointer` |
| `winogender` | WinoGender `rudinger2018winogender` |
| `wsc` | SuperGLUE WSC `wang2019superglue` |
| `wsc273` | WSC273 `levesque2012winograd` |
| `wnli` | GLUE `wang-etal-2018-glue` |
| `winogrande` | WinoGrande `sakaguchi2019winogrande` |

ARC suites (`arc_challenge`, `arc_easy`, and `arc_mt_*`) choose among answer options and score the
question as an exam item, including partial credit when multiple top-scoring choices tie.

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
- `ll_diff`: average absolute log-likelihood gap between paired candidate sentences.
- `label_perm:<fraction>`: permutation-averaged label-only accuracy using the configured fraction
  of all label permutations.
- `f1`: F1 score derived from the suite's predicted labels.
- `mcc`: Matthews correlation coefficient derived from the suite's predicted labels.
- `macro`: macro-average across labels rather than a single positive class.
- `boolean`: positive-class metric using the suite's positive boolean label.

Evalution also includes the Hugging Face `transformers` inference engine, YAML execution, a packaged CLI, and `logbar`-powered runtime progress reporting. 📈

## Citation 📎

If you use Evalution, cite the project itself. If you use one or more built-in suites, also cite the
original benchmark papers below.

```bibtex
% Evalution project citation.
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

Comments inside the BibTeX block below note which built-in suites each citation covers.

```bibtex
% EXAMS. Suites: aexams_{biology,islamic_studies,physics,science,social}.
@inproceedings{hardalov-etal-2020-exams,
  title = {EXAMS: A Multi-Subject High School Examinations Dataset for Cross-Lingual and Multilingual Question Answering},
  author = {Momchil Hardalov and Todor Mihaylov and Vassil Momchev and Pepa Atanasova and Preslav Nakov and Iryna Gurevych},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020},
  pages = {3407--3414},
  url = {https://aclanthology.org/2020.emnlp-main.438/},
  doi = {10.18653/v1/2020.emnlp-main.438},
}

% AIME suites: aime -> aime_1983_2024, aime24 -> aime_2024, aime25 -> aime_2025, aime26 -> aime_2026.
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

@dataset{aime_2026,
  author = {math-ai},
  title = {AIME Problem Set 2026},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/math-ai/aime26},
}

% CMMLU. Suites: cmmlu_<subset>.
@article{li2023cmmlu,
  title = {CMMLU: Measuring massive multitask language understanding in Chinese},
  author = {Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin},
  journal = {arXiv preprint arXiv:2306.09212},
  year = {2023},
  url = {https://arxiv.org/abs/2306.09212},
}

% AGIEval. Suites: agieval_<subset>.
@article{zhong2023agieval,
  title = {AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
  author = {Wanjun Zhong and Zijie Huang and Shirong Ma and Angelica Chen and Yuxin Wang and Li Dong and Jie Tang and Nan Duan},
  journal = {arXiv preprint arXiv:2304.06364},
  year = {2023},
  url = {https://arxiv.org/abs/2304.06364},
}

% ANLI. Suites: anli_{r1,r2,r3}.
@inproceedings{nie-etal-2020-adversarial,
  title = {Adversarial NLI: A New Benchmark for Natural Language Understanding},
  author = {Yixin Nie and Adina Williams and Emily Dinan and Mohit Bansal and Jason Weston and Douwe Kiela},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year = {2020},
}

% bAbI
@article{weston2015towards,
  title = {Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks},
  author = {Jason Weston and Antoine Bordes and Sumit Chopra and Alexander M. Rush and Bart van Merri{\"e}nboer and Armand Joulin and Tomas Mikolov},
  journal = {arXiv preprint arXiv:1502.05698},
  year = {2015},
}

% Bangla. Suites: bangla_{boolqa,commonsenseqa,mmlu,openbookqa,piqa}.
@misc{nahin2025titullmsfamilybanglallms,
  title = {TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking},
  author = {Shahriar Kabir Nahin and Rabindra Nath Nandi and Sagor Sarker and Quazi Sarwar Muhtaseem and Md Kowsher and Apu Chandraw Shill and Md Ibrahim and Mehadi Hasan Menon and Tareq Al Muntasir and Firoj Alam},
  year = {2025},
  eprint = {2502.11187},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2502.11187},
}

% BEAR. Suites: bear, bear_big.
@article{wiland2024bear,
  title = {BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked Language Models},
  author = {Jacek Wiland and Max Ploner and Alan Akbik},
  journal = {arXiv preprint arXiv:2404.04113},
  year = {2024},
  url = {https://arxiv.org/abs/2404.04113},
}

% Belebele
@misc{bandarkar2023belebele,
  title = {The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants},
  author = {Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
  year = {2023},
  eprint = {2308.16884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2308.16884},
}

% BLiMP
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

% BIG-Bench Hard. Suites: bbh_<subset>.
@article{suzgun2022challenging,
  title = {Challenging {BIG-Bench} Tasks and Whether Chain-of-Thought Can Solve Them},
  author = {Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V. and Chi, Ed H. and Zhou, Denny and Wei, Jason},
  journal = {arXiv preprint arXiv:2210.09261},
  year = {2022},
  url = {https://arxiv.org/abs/2210.09261},
}

% BABILong. Suites: babilong_{qa1..qa20}.
@article{kuratov2024babilong,
  title = {BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
  author = {Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Burtsev, Mikhail},
  journal = {arXiv preprint arXiv:2406.10149},
  year = {2024},
  url = {https://arxiv.org/abs/2406.10149},
}

% C4
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

% C-Eval
@article{huang2023ceval,
  title = {C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
  author = {Yuzhen Huang and Yidong Wang and Chunyang Wang and Lei Chen and Xiaonan Luo and Yuxin Tang and Zhilin Yang and Qianqian Wang and Cheng Li and Weijian Yin and others},
  journal = {arXiv preprint arXiv:2305.08322},
  year = {2023},
  url = {https://arxiv.org/abs/2305.08322},
}

% Atlas-Chat. Suites: darijahellaswag, darijammlu_<subset>.
@article{shang2024atlaschatadaptinglargelanguage,
  title = {Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect},
  author = {Shang, Guokan and others},
  journal = {arXiv preprint arXiv:2409.17912},
  year = {2024},
  url = {https://arxiv.org/abs/2409.17912},
}

% GSM8K-Platinum
@article{vendrow2025largelanguagemodelbenchmarks,
  title = {Do Large Language Model Benchmarks Test Reliability?},
  author = {Joshua Vendrow and Edward Vendrow and Sara Beery and Aleksander Madry},
  journal = {arXiv preprint arXiv:2502.03461},
  year = {2025},
}

% GSM8K
@article{cobbe2021trainingverifierssolvemath,
  title = {Training Verifiers to Solve Math Word Problems},
  author = {Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
  journal = {arXiv preprint arXiv:2110.14168},
  year = {2021},
}

% GSM8K French. Suites: gsm8k_fr.
@misc{cmh2025gsm8kfr,
  title = {gsm8k\_fr},
  author = {cmh},
  year = {2025},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/cmh/gsm8k_fr},
}

% GSM8K Korean. Suites: gsm8k_ko.
@misc{kuotient2024gsm8kko,
  title = {gsm8k-ko},
  author = {kuotient},
  year = {2024},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/kuotient/gsm8k-ko},
}

% MGSM. Suites: mgsm_direct_{bn,de,en,es,fr,ja,ru,sw,te,th,zh}.
@article{shi2022multilingualchainofthought,
  title = {Language Models are Multilingual Chain-of-Thought Reasoners},
  author = {Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
  journal = {arXiv preprint arXiv:2210.03057},
  year = {2022},
  url = {https://arxiv.org/abs/2210.03057},
}

% GSM-Plus. Suites: gsm_plus, gsm_plus_mini.
@misc{li2024gsmpluscomprehensivebenchmarkevaluating,
  title = {GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers},
  author = {Qintong Li and Leyang Cui and Xueliang Zhao and Lingpeng Kong and Wei Bi},
  year = {2024},
  eprint = {2402.19255},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2402.19255},
}

% Histoires Morales
@article{leteno2025histoiresmorales,
  title = {Histoires Morales: A French Dataset for Assessing Moral Alignment},
  author = {Leteno, Thibaud and Proskurina, Irina and Gourru, Antoine and Velcin, Julien and Laclau, Charlotte and Metzler, Guillaume and Gravier, Christophe},
  journal = {arXiv preprint arXiv:2501.17117},
  year = {2025},
  url = {https://arxiv.org/abs/2501.17117},
}

% Moral Stories
@inproceedings{emelin-etal-2021-moral,
  title = {Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences},
  author = {Emelin, Denis and Le Bras, Ronan and Hwang, Jena D. and Forbes, Maxwell and Choi, Yejin},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year = {2021},
  url = {https://aclanthology.org/2021.emnlp-main.54},
}

% Icelandic WinoGrande
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

% GPQA. Suites: gpqa_{main,diamond,extended}.
@inproceedings{rein2024gpqa,
  title = {{GPQA}: A Graduate-Level Google-Proof {Q\&A} Benchmark},
  author = {David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
  booktitle = {First Conference on Language Modeling},
  year = {2024},
  url = {https://openreview.net/forum?id=Ti67584b98},
}

% ARC. Suites: arc_challenge, arc_easy, arc_mt_{da,de,el,es,fi,hu,is,it,nb,pl,pt,sv}.
@article{clark2018arc,
  title = {Think you have Solved Question Answering? Try {ARC}, the {AI2} Reasoning Challenge},
  author = {Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal = {arXiv preprint arXiv:1803.05457},
  year = {2018},
}

% GPT-3 Arithmetic. Suites: arithmetic_{1dc,2da,2dm,2ds,3da,3ds,4da,4ds,5da,5ds}.
@inproceedings{brown2020gpt3,
  title = {Language Models are Few-Shot Learners},
  author = {Tom Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel Ziegler and Jeffrey Wu and Clemens Winter and Chris Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}

% CareQA. Suites: careqa_{en,es}.
@inproceedings{arias-duart-etal-2025-automatic,
  title = {Automatic Evaluation of Healthcare LLMs Beyond Question-Answering},
  author = {Arias-Duart, Anna and Bernabeu, Pablo and Lopez, Adria and Hadj Taieb, Meriem and Villegas, Marta and Gonzalez-Agirre, Aitor},
  booktitle = {Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics},
  year = {2025},
  url = {https://arxiv.org/abs/2502.06666},
}

% CaBBQ / EsBBQ. Suites: cabbq_<category>, esbbq_<category>.
@misc{ruizfernández2025esbbqcabbqspanishcatalan,
  title = {EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering},
  author = {Valle Ruiz-Fernández and Mario Mina and Júlia Falcão and Luis Vasquez-Reina and Anna Sallés and Aitor Gonzalez-Agirre and Olatz Perez-de-Viñaspre},
  year = {2025},
  eprint = {2507.11216},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2507.11216},
}

% BBQ. Suites: bbq_<category>.
@inproceedings{parrish2022bbq,
  title = {BBQ: A Hand-Built Bias Benchmark for Question Answering},
  author = {Parrish, Alicia and Chen, Angelica and Nangia, Nikita and Padmakumar, Vishakh and Phang, Jason and Thompson, John and Htut, Phu Mon and Bowman, Samuel R.},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
  year = {2022},
  url = {https://aclanthology.org/2022.findings-acl.165/},
}

% XNLI. Suites: xnli_<language>.
@inproceedings{conneau2018xnli,
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  author = {Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel R. and Schwenk, Holger and Stoyanov, Veselin},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
  pages = {2475--2485},
  url = {https://aclanthology.org/D18-1269/},
  doi = {10.18653/v1/D18-1269},
}

% XQuAD. Suites: xquad_<language>.
@inproceedings{artetxe2020crosslingual,
  title = {Cross-lingual Question Answering},
  author = {Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year = {2020},
  pages = {1198--1207},
  url = {https://aclanthology.org/2020.acl-main.119/},
  doi = {10.18653/v1/2020.acl-main.119},
}

% TruthfulQA. Suites: truthfulqa_{mc1,mc2}.
@inproceedings{lin-etal-2022-truthfulqa,
  title = {TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author = {Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2022},
  pages = {3214--3252},
  url = {https://aclanthology.org/2022.acl-long.229},
  doi = {10.18653/v1/2022.acl-long.229},
}

% ASDiv. Suites: asdiv, asdiv_cot_llama.
@article{miao2021diverse,
  title = {A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
  author = {Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
  journal = {arXiv preprint arXiv:2106.15772},
  year = {2021},
}

% PIQA
@inproceedings{bisk2020piqa,
  title = {PIQA: Reasoning about Physical Commonsense in Natural Language},
  author = {Yonatan Bisk and Rowan Zellers and Ronan Le Bras and Jianfeng Gao and Yejin Choi},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2020},
}

% AlGhafa. Suites: copa_ar, piqa_ar.
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

% The Pile
@article{gao2020pile,
  title = {The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
  author = {Leo Gao and Stella Biderman and Sid Black and Laurence Golding and Travis Hoppe and Charles Foster and Jason Phang and Horace He and Anish Thite and Noa Nabeshima and Shawn Presser and Connor Leahy},
  journal = {arXiv preprint arXiv:2101.00027},
  year = {2020},
}

% PROST
@inproceedings{aroca-ouellette-etal-2021-prost,
  title = {{PROST}: Physical Reasoning about Objects through Space and Time},
  author = {St{\'e}phane Aroca-Ouellette and Cory Paik and Alessandro Roncone and Katharina Kann},
  booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  year = {2021},
  pages = {4597--4608},
  url = {https://aclanthology.org/2021.findings-acl.404},
}

% SQuAD 2.0
@article{rajpurkar2018know,
  title = {Know What You Don’t Know: Unanswerable Questions for SQuAD},
  author = {Pranav Rajpurkar and Robin Jia and Percy Liang},
  journal = {arXiv preprint arXiv:1806.03822},
  year = {2018},
}

% TriviaQA
@article{joshi2017triviaqa,
  title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
  author = {Mandar Joshi and Eunsol Choi and Daniel Weld and Luke Zettlemoyer},
  journal = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year = {2017},
}

% Natural Questions
@article{kwiatkowski2019natural,
  title = {Natural Questions: A Benchmark for Question Answering Research},
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Jacob Devlin and Kenton Lee and Kristina Toutanova and Llion Jones and Matthew Kelcey and Ming-Wei Chang and Andrew M. Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  pages = {452--466},
  year = {2019},
}

% PubMedQA
@inproceedings{jin2019pubmedqa,
  title = {PubMedQA: A Dataset for Biomedical Research Question Answering},
  author = {Qiao Jin and Bhuwan Dhingra and Zhengping Liu and William Cohen and Xinghua Lu},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
  pages = {2567--2577},
}

% QA4MRE
@inproceedings{Peas2013QA4MRE2O,
  title = {QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation},
  author = {Pe{\~n}as, Anselmo and Hovy, Eduard H. and Forner, Pamela and Rodrigo, {\'A}lvaro and Sutcliffe, Richard F. E. and Morante, Roser},
  booktitle = {CLEF},
  year = {2013},
}

% CommonsenseQA
@inproceedings{talmor2019commonsenseqa,
  title = {CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author = {Alon Talmor and Jonathan Herzig and Nicholas Lourie and Jonathan Berant},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019},
}

% CoQA
@article{reddy2019coqa,
  title = {CoQA: A Conversational Question Answering Challenge},
  author = {Siva Reddy and Danqi Chen and Christopher D. Manning},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  pages = {249--266},
  year = {2019},
}

% CrowS-Pairs. Suites: crows_pairs_english and crows_pairs_english_<bias_type>; French CrowS-Pairs suites: crows_pairs_french and crows_pairs_french_<bias_type>.
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

% COPAL-ID. Suites: copal_id_{standard,colloquial}.
@inproceedings{wibowo-etal-2024-copal,
  title = {COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances},
  author = {Haryo Akbarianto Wibowo and Swandana Rama Sandhiyudha and Genta Indra Winata and Ayu Purwarianti and Sebastian Ruder and Rahmad Mahardhika and Pascale Fung},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {1393--1410},
  year = {2024},
  url = {https://aclanthology.org/2024.naacl-long.77/},
}

% DROP
@inproceedings{dua2019drop,
  title = {DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
  author = {Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019},
}

% ETHICS. Suites: ethics_{cm,deontology,justice,utilitarianism,virtue}.
@article{hendrycks2021ethics,
  title = {Aligning AI With Shared Human Values},
  author = {Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal = {International Conference on Learning Representations},
  year = {2021},
}

% MMLU
@article{hendryckstest2021,
  title = {Measuring Massive Multitask Language Understanding},
  author = {Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal = {International Conference on Learning Representations},
  year = {2021},
}

% MMLU-CF. Suites: mmlu_cf_<subject>.
@article{zhao2024mmlucf,
  title = {MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark},
  author = {Qihao Zhao and Yangyu Huang and Tengchao Lv and Lei Cui and Qinzheng Sun and Shaoguang Mao and Xin Zhang and Ying Xin and Qiufeng Yin and Scarlett Li and Furu Wei},
  journal = {arXiv preprint arXiv:2412.15194},
  year = {2024},
  url = {https://arxiv.org/abs/2412.15194},
}

% MMLU-Pro
@article{wang2024mmlupro,
  title = {MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author = {Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
  journal = {arXiv preprint arXiv:2406.01574},
  year = {2024},
}

% ArabicMMLU. Suites: arabicmmlu_<subset>.
@misc{koto2024arabicmmlu,
  title = {ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic},
  author = {Koto, Fajri and Li, Haonan and Shatnawi, Sara and Doughman, Jad and Sadallah, Abdelrahman Boda and Alraeesi, Aisha and Almubarak, Khalid and Alyafeai, Zaid and Sengupta, Neha and Shehata, Shady and Habash, Nizar and Nakov, Preslav and Baldwin, Timothy},
  year = {2024},
  eprint = {2402.12840},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2402.12840},
}
% NileChat. Suites: egyhellaswag, egymmlu_<subset>.
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

% Latxa / EusExams. Suites: eus_exams_<subset>, eus_reading, eus_proficiency, eus_trivia.
@misc{etxaniz2024latxa,
  title = {Latxa: An Open Language Model and Evaluation Suite for Basque},
  author = {Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
  year = {2024},
  eprint = {2403.20266},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2403.20266},
}

% GLUE. Suites: cola, mnli, mrpc, qnli, qqp, sst2, wnli.
@inproceedings{wang-etal-2018-glue,
  title = {{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author = {Alex Wang and Amanpreet Singh and Julian Michael and Felix Hill and Omer Levy and Samuel Bowman},
  booktitle = {Proceedings of the 2018 EMNLP Workshop BlackboxNLP},
  year = {2018},
}

% SuperGLUE. Suites: boolq, cb, copa, multirc, record, rte, wic, wsc.
@inproceedings{wang2019superglue,
  title = {SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems},
  author = {Alex Wang and Yada Pruksachatkun and Nikita Nangia and Amanpreet Singh and Julian Michael and Felix Hill and Omer Levy and Samuel Bowman},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019},
}

% HellaSwag
@inproceedings{zellers2019hellaswag,
  title = {HellaSwag: Can a Machine Really Finish Your Sentence?},
  author = {Rowan Zellers and Ari Holtzman and Yonatan Bisk and Ali Farhadi and Yejin Choi},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year = {2019},
}

% HEAD-QA. Suites: headqa_{en,es}.
@inproceedings{vilares-gomez-rodriguez-2019-head,
  title = {HEAD-QA: A Healthcare Dataset for Complex Reasoning},
  author = {David Vilares and Carlos G{\'o}mez-Rodr{\'i}guez},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year = {2019},
}

% KoBEST. Suites: kobest_{boolq,copa,hellaswag,sentineg,wic}.
@misc{kim2022kobest,
  title = {KoBEST: Korean Balanced Evaluation of Significant Tasks},
  author = {Dohyeong Kim and Myeongjun Jang and Deuk Sin Kwon and Eric Davis},
  year = {2022},
  eprint = {2204.04541},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2204.04541},
}

% KMMLU. Suites: kmmlu_<subset>.
@article{son2024kmmlu,
  title = {KMMLU: Measuring Massive Multitask Language Understanding in Korean},
  author = {Guijin Son and Hanwool Lee and Sungdong Kim and Seungone Kim and Niklas Muennighoff and Taekyoon Choi and Cheonbok Park and Kang Min Yoo and Stella Biderman},
  journal = {arXiv preprint arXiv:2402.11548},
  year = {2024},
  url = {https://arxiv.org/abs/2402.11548},
}

% IrokoBench. Suites: afrimgsm_<language>, afrimmlu_<language>, afrixnli_<language>.
@inproceedings{adelani2025irokobench,
  title = {IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models},
  author = {David Ifeoluwa Adelani and Jessica Ojo and Israel Abebe Azime and Jian Yun Zhuang and Jesujoba O. Alabi and others},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages = {2732--2757},
  year = {2025},
  url = {https://aclanthology.org/2025.naacl-long.139/},
}

% LAMBADA. Suites: lambada_{openai,openai_cloze,standard,standard_cloze}, lambada_openai_mt_{de,en,es,fr,it}, lambada_openai_mt_stablelm_{de,en,es,fr,it,nl,pt}.
@misc{paperno2016lambada,
  title = {The LAMBADA dataset: Word prediction requiring a broad discourse context},
  author = {Denis Paperno and Germ{\'a}n Kruszewski and Angeliki Lazaridou and Quan Ngoc Pham and Raffaella Bernardi and Sandro Pezzelle and Marco Baroni and Gemma Boleda and Raquel Fern{\'a}ndez},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.2630551},
  year = {2016},
}

% Inverse Scaling Prize. Suites: inverse_scaling_<subset>.
@article{mckenzie2023inverse,
  title = {Inverse Scaling Prize: First Round Winners},
  author = {Robert McKenzie and Ethan Perez and Jan Leike and others},
  year = {2023},
  journal = {arXiv preprint arXiv:2306.09479},
  url = {https://arxiv.org/abs/2306.09479},
}

% LogiQA. Suites: logiqa.
@misc{liu2020logiqa,
  title = {LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
  author = {Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
  year = {2020},
  eprint = {2007.08124},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

% LogiQA 2.0. Suites: logiqa2.
@misc{liu2022logiqa2,
  title = {LogiQA 2.0: An Improved Dataset for Logical Reasoning in Natural Language Understanding},
  author = {Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
  year = {2022},
  eprint = {2203.15796},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

% HumanEval
@misc{chen2021evaluatinglargelanguagemodels,
  title = {Evaluating Large Language Models Trained on Code},
  author = {Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Krzysztof Misztal and John Schulman and Dario Amodei},
  year = {2021},
  eprint = {2107.03374},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
}

% MBPP
@misc{austin2021program,
  title = {Program Synthesis with Large Language Models},
  author = {Jacob Austin and Augustus Odena and Maxwell Nye and Maarten Bosma and Henryk Michalewski and David Dohan and Ellen Jiang and Carrie Cai and Michael Terry and Quoc Le and Charles Sutton},
  year = {2021},
  eprint = {2108.07732},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
}

% MuTual. Suites: mutual.
@inproceedings{cui2020mutual,
  title = {MuTual: A Dataset for Multi-Turn Dialogue Reasoning},
  author = {Cui, Leyang and Wu, Yu and Liu, Shujie and Zhang, Yue and Zhou, Ming},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year = {2020},
  url = {https://aclanthology.org/2020.acl-main.130/},
}

% MathQA
@misc{amini2019mathqa,
  title = {MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms},
  author = {Aida Amini and Saadia Gabriel and Peter Lin and Rik Koncel-Kedziorski and Yejin Choi and Hannaneh Hajishirzi},
  year = {2019},
  eprint = {1905.13319},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

% MATH. Suites: hendrycks_math_<subset>.
@article{hendrycks2021measuring,
  title = {Measuring Mathematical Problem Solving With the MATH Dataset},
  author = {Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal = {Advances in Neural Information Processing Systems},
  volume = {34},
  pages = {5325--5337},
  year = {2021},
  url = {https://proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html},
}

% MC-TACO
@inproceedings{zhou2019mctaco,
  title = {Going on a vacation takes longer than going for a walk: A Study of Temporal Commonsense Understanding},
  author = {Ben Zhou and Daniel Khashabi and Qiang Ning and Dan Roth},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
}

% MedMCQA
@inproceedings{pmlr-v174-pal22a,
  title = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author = {Ankit Pal and Logesh Kumar Umapathi and Malaikannan Sankarasubbu},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  year = {2022},
}

% MedQA
@article{jin2020disease,
  title = {What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author = {Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
  journal = {arXiv preprint arXiv:2009.13081},
  year = {2020},
}

% OpenBookQA
@inproceedings{mihaylov2018openbookqa,
  title = {Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
  author = {Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
}

% PAWS-X. Suites: paws_x_{de,en,es,fr,ja,ko,zh}.
@inproceedings{yang2019pawsx,
  title = {PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification},
  author = {Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
}

% CNN/DailyMail
@article{nallapati2016abstractive,
  title = {Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond},
  author = {Ramesh Nallapati and Bowen Zhou and Cicero dos Santos and {\c{C}}aglar Gul{\c{c}}ehre and Bing Xiang},
  journal = {arXiv preprint arXiv:1602.06023},
  year = {2016},
}

% CodeXGLUE. Suites: code2text_{go,java,javascript,php,python,ruby}.
@inproceedings{lu2021codexglue,
  title = {CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author = {Shuai Lu and Daya Guo and Shuo Ren and Junjie Huang and Alexey Svyatkovskiy and Ambrosio Blanco and Colin B. Clement and Dawn Drain and Daxin Jiang and Duyu Tang and Ge Li and Lidong Zhou and Linjun Shou and Long Zhou and Michele Tufano and Ming Gong and Ming Zhou and Nan Duan and Neel Sundaresan and Shao Kun Deng and Shengyu Fu and Shujie Liu},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year = {2021},
}

% XCOPA. Suites: xcopa_{et,ht,id,it,qu,sw,ta,th,tr,vi,zh}.
@inproceedings{ponti2020xcopa,
  title = {XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning},
  author = {Edoardo M. Ponti and Rahul Gupta and Ivan Vuli{\'c} and Goran Glava{\v{s}} and Anna Korhonen},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year = {2020},
}

% SciQ
@inproceedings{welbl2017crowdsourcing,
  title = {Crowdsourcing Multiple Choice Science Questions},
  author = {Johannes Welbl and Nelson F. Liu and Matt Gardner},
  booktitle = {Proceedings of the 3rd Workshop on Noisy User-generated Text},
  year = {2017},
}

% Social IQA
@inproceedings{sap2019social,
  title = {Social IQa: Commonsense Reasoning about Social Interactions},
  author = {Maarten Sap and Hannah Rashkin and Derek Chen and Ronan Le Bras and Yejin Choi},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  year = {2019},
}

% SWAG
@inproceedings{zellers2018swagaf,
  title = {SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
  author = {Rowan Zellers and Yonatan Bisk and Roy Schwartz and Yejin Choi},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year = {2018},
}

% WebQuestions
@inproceedings{berant-etal-2013-semantic,
  title = {Semantic Parsing on Freebase from Question-Answer Pairs},
  author = {Jonathan Berant and Andrew Chou and Roy Frostig and Percy Liang},
  booktitle = {Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},
  year = {2013},
  pages = {1533--1544},
  url = {https://aclanthology.org/D13-1160},
}

% WikiText-2
@misc{merity2016pointer,
  title = {Pointer Sentinel Mixture Models},
  author = {Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
  year = {2016},
  eprint = {1609.07843},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

% WinoGender. Suites: winogender_{all,female,gotcha,gotcha_female,gotcha_male,male,neutral}.
@inproceedings{rudinger2018winogender,
  title = {Gender Bias in Coreference Resolution},
  author = {Rachel Rudinger and Jason Naradowsky and Brian Leonard and Benjamin Van Durme},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year = {2018},
}

% WSC273
@inproceedings{levesque2012winograd,
  title = {The Winograd Schema Challenge},
  author = {Hector Levesque and Ernest Davis and Leora Morgenstern},
  booktitle = {Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning},
  year = {2012},
}

% XWinograd. Suites: xwinograd_{en,fr,jp,pt,ru,zh}.
@misc{tikhonov2021heads,
  title = {It's All in the Heads: Using Attention Heads as a Baseline for Cross-Lingual Transfer in Commonsense Reasoning},
  author = {Alexey Tikhonov and Max Ryabinin},
  year = {2021},
  eprint = {2106.12066},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
}

% XStoryCloze. Suites: xstorycloze_{ar,en,es,eu,hi,id,my,ru,sw,te,zh}.
@article{lin2021fewshotmultilingual,
  title = {Few-shot Learning with Multilingual Language Models},
  author = {Xi Victoria Lin and Todor Mihaylov and Mikel Artetxe and Tianlu Wang and Shuohui Chen and Daniel Simig and Myle Ott and Naman Goyal and Shruti Bhosale and Jingfei Du and Ramakanth Pasunuru and Sam Shleifer and Punit Singh Koura and Vishrav Chaudhary and Brian O'Horo and Jeff Wang and Luke Zettlemoyer and Zornitsa Kozareva and Mona T. Diab and Veselin Stoyanov and Xian Li},
  journal = {arXiv preprint arXiv:2112.10668},
  year = {2021},
}

% WinoGrande
@article{sakaguchi2019winogrande,
  title = {WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
  author = {Keisuke Sakaguchi and Ronan Le Bras and Chandra Bhagavatula and Yejin Choi},
  journal = {arXiv preprint arXiv:1907.10641},
  year = {2019},
}

% BASED / FDA. Suites: fda.
@article{arora2024simple,
  title = {Simple linear attention language models balance the recall-throughput tradeoff},
  author = {Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher R{'e}},
  journal = {arXiv preprint arXiv:2402.18668},
  year = {2024},
  url = {https://arxiv.org/abs/2402.18668},
}

% FLD. Suites: fld.
@article{morishita2023learning,
  title = {Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic},
  author = {Terufumi Morishita and Gaku Morio and Atsuki Yamaguchi and Yasuhiro Sogawa},
  journal = {arXiv preprint arXiv:2308.07336},
  year = {2023},
  url = {https://arxiv.org/abs/2308.07336},
}

% IFEval. Suites: ifeval.
@article{zhou2023instruction,
  title = {Instruction-Following Evaluation for Large Language Models},
  author = {Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
  journal = {arXiv preprint arXiv:2311.07911},
  year = {2023},
  url = {https://arxiv.org/abs/2311.07911},
}

% IFEval-PT. Suites: ifeval_pt.
@article{klugecorrea2026tucano2cool,
  title = {Tucano 2 Cool: Better Open Source LLMs for Portuguese},
  author = {Nicholas Kluge Correa and Aniket Sen and Shiza Fatimah and Sophia Falk and Lennard Landgraf and Julia Kastner and Lucie Flek},
  journal = {arXiv preprint arXiv:2603.03543},
  year = {2026},
  url = {https://arxiv.org/abs/2603.03543},
}

% CLIcK. Suites: click, click_lang, click_lang_{text,grammar,function}, click_cul, click_cul_{economy,geography,history,kpop,law,politics,society,tradition}.
@inproceedings{kim-etal-2024-click,
  title = "{CLI}c{K}: A Benchmark Dataset of Cultural and Linguistic Intelligence in {K}orean",
  author = "Kim, Eunsu  and
    Suk, Juyoung  and
    Oh, Philhoon  and
    Yoo, Haneul  and
    Thorne, James  and
    Oh, Alice",
  booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
  year = "2024",
  address = "Torino, Italia",
  publisher = "ELRA and ICCL",
  url = "https://aclanthology.org/2024.lrec-main.296/",
  pages = "3335--3346",
}

% HAE-RAE. Suites: haerae, haerae_{general_knowledge,history,loan_word,rare_word,standard_nomenclature}.
@inproceedings{son-etal-2024-hae,
  title = "{HAE}-{RAE} Bench: Evaluation of {K}orean Knowledge in Language Models",
  author = "Son, Guijin  and
    Lee, Hanwool  and
    Kim, Suwan  and
    Kim, Huiseo  and
    Lee, Jae cheol  and
    Yeom, Je Won  and
    Jung, Jihyu  and
    Kim, Jung woo  and
    Kim, Songseong",
  booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
  year = "2024",
  address = "Torino, Italia",
  publisher = "ELRA and ICCL",
  url = "https://aclanthology.org/2024.lrec-main.704/",
  pages = "7993--8007",
}

% KorMedMCQA. Suites: kormedmcqa, kormedmcqa_{doctor,nurse,pharm,dentist}.
@misc{kweon2024kormedmcqa,
  title = {KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations},
  author = {Sunjun Kweon and Byungjin Choi and Gyouk Chu and Junyeong Song and Daeun Hyeon and Sujin Gan and Jueon Kim and Minkyu Kim and Rae Woong Park and Edward Choi},
  year = {2024},
  eprint = {2403.01469},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2403.01469},
}

% KLEJ POLEMO 2.0. Suites: polemo2_{in,out}.
@inproceedings{kocon-etal-2019-multi,
  title = "Multi-Level Sentiment Analysis of {P}ol{E}mo 2.0: Extended Corpus of Multi-Domain Consumer Reviews",
  author = "Koco{'n}, Jan  and
    Mi{\l}kowski, Piotr  and
    Za{'s}ko-Zieli{'n}ska, Monika",
  booktitle = "Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)",
  year = "2019",
  address = "Hong Kong, China",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/K19-1092/",
  doi = "10.18653/v1/K19-1092",
  pages = "980--991",
}

% RACE. Suites: race.
@inproceedings{lai-etal-2017-race,
  title = "{RACE}: Large-scale {R}e{A}ding Comprehension Dataset From Examinations",
  author = "Lai, Guokun  and
    Xie, Qizhe  and
    Liu, Hanxiao  and
    Yang, Yiming  and
    Hovy, Eduard",
  booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
  year = "2017",
  address = "Copenhagen, Denmark",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D17-1082/",
  doi = "10.18653/v1/D17-1082",
  pages = "785--794",
}

% ToxiGen. Suites: toxigen.
@inproceedings{hartvigsen-etal-2022-toxigen,
  title = "{T}oxi{G}en: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection",
  author = "Hartvigsen, Thomas  and
    Gabriel, Saadia  and
    Palangi, Hamid  and
    Sap, Maarten  and
    Ray, Dipankar  and
    Kamar, Ece",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2022",
  address = "Dublin, Ireland",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2022.acl-long.234/",
  doi = "10.18653/v1/2022.acl-long.234",
  pages = "3309--3326",
}

% XNLI-EU. Suites: xnli_eu.
@inproceedings{heredia-etal-2024-xnlieu,
  title = "{XNLI}eu: a dataset for cross-lingual {NLI} in {B}asque",
  author = "Heredia, Maite  and
    Etxaniz, Julen  and
    Zulaika, Muitze  and
    Saralegi, Xabier  and
    Barnes, Jeremy  and
    Soroa, Aitor",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  year = "2024",
  address = "Mexico City, Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.naacl-long.234/",
  doi = "10.18653/v1/2024.naacl-long.234",
  pages = "4177--4188",
}

```
