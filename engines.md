# Engines

The canonical engine guide lives in [docs/engine.md](docs/engine.md).

Built-in engines at a glance:

| Engine | Runtime | Model formats | Continuous batching | Notes |
| --- | --- | --- | --- | --- |
| `Transformers` | `transformers` | Hugging Face checkpoints | Native when the installed runtime supports it | Modern default backend |
| `TransformersCompat` | `transformers` | Hugging Face checkpoints | Emulated fixed batches | Compatibility fallback path |
| `GPTQModel` | `gptqmodel` | GPTQ checkpoints | Shared transformer-style path | Quantized local runtime |
| `Tinygrad` | `tinygrad.llm` | Local GGUF checkpoints | Emulated fixed-batch submission over static batching | Use `tokenizer_path` for chat templates; CUDA defaults to `jit=2`, `jitbeam=0` |
| `LlamaCpp` | `llama-cpp-python` | GGUF checkpoints | Native multi-sequence batch API | In-process llama.cpp backend |
| `OpenAICompatible` | OpenAI-style HTTP API | Remote model ids | Client-side queued batching | Requires Evalution loglikelihood endpoints |
| `OpenVINO` | `optimum-intel` / OpenVINO | Hugging Face checkpoints | Fixed batches | Intel-focused local runtime |
| `SGLang` | `sglang` | Hugging Face checkpoints | Runtime-dependent | In-process SGLang backend |
| `TensorRTLLM` | `tensorrt_llm` | TensorRT-LLM exports | Runtime-dependent | NVIDIA TensorRT-LLM backend |
| `VLLM` | `vllm` | Hugging Face checkpoints | Native when low-level request APIs are available | vLLM local runtime |

Use [docs/engine.md](docs/engine.md) for backend contracts, built-in engine behavior, and
implementation guidance for new runtimes.
