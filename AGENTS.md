# Evalution AGENTS

## GPU Test Policy

- Keep CUDA ordinal assignment stable across test runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- Run every GPU test as a single-GPU test process. Each process should see exactly one GPU via `CUDA_VISIBLE_DEVICES` masking against the PCI-bus-ordered device list.
- If a task provides a list of GPUs, treat that as permission to shard independent backward or parallel test runs across those GPUs to accelerate A/B testing and sweeps. Each shard still remains single-GPU.
- Do not interpret a GPU list as a request to run one test in multi-GPU mode unless the task explicitly asks for multi-GPU coverage.
- When a task truly needs multi-GPU coverage, call that out explicitly and keep it separate from the default single-GPU regression path.

## Benchmarking and Testing

- Pair every benchmark test with its own standalone `Llama 3.2 1B Instruct` model test to capture the performance envelope exhaustively.
- Embed real RTX 4090 and A100 baseline scores into every quantization-method unit test, sourced from actual runs, so regressions are detectable later.
- Have quantization regression tests detect the GPU in use, select the corresponding RTX 4090 or A100 baseline, and assert against that hardware-specific value.
- Run both the `lm-eval` score and this repository's model unit test score, explain any large disparities, and confirm they remain in sync before accepting results.
- Do not cite or copy `lm-eval` code; keep implementations clean-room and derived from the original author or paper.

## A/B Test Reporting

- Every A/B test, backend comparison, and sweep summary must include an ASCII table in the logs or report. Do not summarize A/B results only in prose.
- Keep one row per variant or test case. Include the baseline row and each candidate row being compared.
- Required tracking columns are `variant`, `performance`, `accuracy`, `cpu_time_s`, `gpu_time_s`, `cpu_ram_gib`, and `gpu_vram_gib`.
- If a metric is not applicable or unavailable for a specific run, keep the column and write `N/A` rather than removing it.
- When useful, add delta columns such as `delta_perf`, `delta_acc`, or `%_change`, but do not remove the required core columns.
- Use a plain ASCII table format such as:

```text
+-----------+-------------+----------+------------+------------+-------------+--------------+
| variant   | performance | accuracy | cpu_time_s | gpu_time_s | cpu_ram_gib | gpu_vram_gib |
+-----------+-------------+----------+------------+------------+-------------+--------------+
| baseline  | 123.4 tok/s | 0.7421   | 12.40      | 11.98      | 3.21        | 9.88         |
| candidate | 131.7 tok/s | 0.7415   | 11.02      | 10.61      | 3.34        | 10.12        |
+-----------+-------------+----------+------------+------------+-------------+--------------+
```
