# AGENTS.MD

This file defines the required implementation and testing rules for agents and contributors working in this repository.

## Required Rules

1. Always add clear, concise code comments to every class, `@classmethod`, and all global or class-level state. Comments should explain intent or constraints, not restate obvious syntax.
2. Never copy code from another repository without explicit authorization.
3. Always perform a clean-room re-implementation with priority ordered as: correctness first, then efficiency across GPU, CPU, and memory usage.
4. Every benchmark test must also have its own standalone model test based on `Llama 3.2 1B Instruct`.
5. Quantization-method model unit tests must include baseline scores captured from real execution on both `RTX 4090` and `A100` hardware, and those baseline values must be encoded in the test code for future regression checks.
6. Quantization-method regression tests must detect which GPU is in use, choose the appropriate `RTX 4090` or `A100` baseline, and assert against that hardware-specific baseline.
7. Use the original author or benchmark implementation and the original scoring logic as the primary reference source. If a research paper exists, read it to understand the method and intent before implementing.
8. Use `lm-eval` only as a reference for quality and logic when there is no upstream implementation from the original author or entity.
9. Always run both the `lm-eval` score and this repository's model unit test score, and verify that there is no large disparity between them before accepting results.
10. Do not copy, embed, or cite `lm-eval` code in the implementation. Keep the implementation clean-room and derived first from the original author's codebase and/or paper.
