# New Suite Guidelines

This document defines the minimum bar for adding new Evalution benchmark, eval, or test suites.

## Reuse Existing Scores First

- Reuse an existing scorer or score shape whenever the benchmark semantics already match something in `evalution/scorers/`.
- Extend an existing scorer only when the new suite has a real scoring gap that cannot be covered cleanly by configuration or light composition.
- Do not invent a new metric name, extraction rule, or scoring pipeline when an existing Evalution score already represents the benchmark faithfully.
- If the benchmark owner publishes a scoring rule, implement that rule clean-room and map it onto the closest existing Evalution score surface when possible.

## Regex Policy For New Suites

- Use the `pcre` module from `PyPcre` for all regex work. Do not add `import re` or `import regex`.
- Compile reusable patterns once with `pcre.compile(...)` and call pattern methods such as `.search()`, `.match()`, `.findall()`, `.sub()`, or `.split()`.
- Keep patterns PCRE2-compatible from the start so suite code, tests, and future ports all share one regex dialect.
- Prefer `pcre.escape(...)` for any dynamic pattern fragment instead of interpolating user or dataset text directly into a regex.

## Why Evalution Standardizes On PCRE2

- PCRE2 gives the project one consistent regex engine across code and tests.
- It is feature-rich and mature, which reduces pressure to build benchmark-specific parsing hacks around stdlib `re`.
- It aligns with Evalution's preference for compiled, reusable patterns that are fast to execute repeatedly inside dataset loaders, scorers, and answer extractors.
- It is the project standard for safer regex construction practices such as escaping dynamic text before compilation.

## Contributor Checklist

- Add the suite under `evalution/benchmarks/` and reuse shared scorers before creating new score code.
- Add or update unit tests under `tests/` and model-backed coverage under `tests/models/` when the suite is intended to ship as a supported benchmark.
- Keep regex usage on compiled `pcre` patterns only.
- Document any new metric or scoring behavior in `docs/scorers.md` or `docs/scores.md` when it changes user-visible outputs.
