# Scorers

Evalution keeps benchmark orchestration under `evalution/suites/` and shared scoring logic under
`evalution/scorers/`.

The split is:

- `evalution/suites/`: dataset loading, prompt construction, few-shot selection, request execution,
  and result packaging.
- `evalution/scorers/`: score math, canonical answer extraction, benchmark-specific parser rules,
  and reusable metric helpers.

This separation keeps benchmark semantics inspectable without mixing them into engine or prompting
code.

## Package Layout

```text
evalution/scorers/
  __init__.py
  choice_label.py
  classification.py
  gsm8k.py
  multiple_choice.py
```

## Scorer Modules

### `multiple_choice.py`

Shared helpers for log-likelihood-ranked answer options.

Implements:

- raw multiple-choice accuracy from summed continuation log-likelihood
- length-normalized multiple-choice accuracy from `logprob / token_count`
- ARC-style exam score with tie-aware partial credit
- permutation-averaged label-only scoring for optional bias-mitigation runs

Primary entry points:

- `build_choice_score(...)`
- `build_choice_scores(...)`
- `multiple_choice_outcome(...)`
- `exam_score_outcome(...)`
- `label_permutations_for_mode(...)`
- `label_permutation_outcome(...)`

### `gsm8k.py`

Shared numeric-answer parsing and equality for GSM8K-family suites.

Implements:

- benchmark-owner GSM8K `#### number` parser
- benchmark-owner GSM8K-Platinum parser
- format-insensitive numeric extraction used by the live Evalution scorer
- numeric canonicalization and equality
- ground-truth target extraction helpers

### `classification.py`

Task-agnostic classification metrics used by GLUE and SuperGLUE style suites.

Implements:

- `f1_for_label(...)`
- `macro_f1(...)`
- `matthews_corrcoef(...)`

### `choice_label.py`

Exact-match helpers for extracted choice labels.

Implements:

- `exact_match(...)`
- `choice_label_exact_match(...)`

## Suite To Scorer Mapping

| Suite family | Scorer module | Runtime metric shape |
| --- | --- | --- |
| Generic multiple-choice suites | `multiple_choice.py` | `accuracy,loglikelihood`, `accuracy,loglikelihood_norm` |
| ARC | `multiple_choice.py` | `accuracy,exam_score` |
| Optional label-bias mitigation on multiple-choice suites | `multiple_choice.py` | `accuracy,label_perm_<fraction>` |
| GSM8K / GSM8K-Platinum | `gsm8k.py` | `accuracy,numeric` |
| GLUE / SuperGLUE extra classification metrics | `classification.py` | `f1,...`, `mcc,...` |
| MMLU-Pro | `choice_label.py` | `exact_match,choice-label` |

## `label_permutations`

`label_permutations` is an opt-in extra scorer config for relevant multiple-choice suites. It uses
numeric fractions:

- `0.0`: disabled
- any float in `(0.0, 1.0)`: balanced subset of permutations sized from that fraction
- `1.0`: all permutations

It is additive by design:

- the benchmark-native default score still runs
- the permutation-averaged label-only score is added as an extra metric
- the extra metric costs extra log-likelihood calls

Metric names:

- `label_permutations=0.25` -> `accuracy,label_perm_0.25`
- `label_permutations=0.5` -> `accuracy,label_perm_0.5`
- `label_permutations=0.75` -> `accuracy,label_perm_0.75`
- `label_permutations=1.0` -> `accuracy,label_perm_1.0`

### Why this exists

Full-choice continuation scoring can favor shorter options because it scores the entire answer
string. Label-only scoring removes most of that option-length effect, but a fixed label mapping can
introduce label prior bias, where models prefer `A` or `B` regardless of content.

Permutation averaging reduces that by evaluating several relabelings of the same question and
averaging scores back onto the original options.

## Label Permutation Averaging Math

Let the original options be `x_1, ..., x_K`.

Let `pi` be a permutation that maps original options onto label positions.

For one relabeling `pi`, the model scores labels `A/B/C/...` under a prompt with the permuted
option order. If `s_pi(j)` is the log-likelihood of label position `j`, then the score contributed
to original option `x_i` under `pi` is:

```text
score_pi(i) = s_pi(pi(i))
```

Across a selected permutation set `P`, Evalution averages those scores:

```text
score_avg(i) = (1 / |P|) * sum_{pi in P} score_pi(i)
```

Prediction is then:

```text
y_hat = argmax_i score_avg(i)
```

and the extra metric is:

```text
accuracy = 1[y_hat = y]
```

## Compute Tradeoff

If a suite has `K` answer options and uses a permutation set `P`, the extra label-only scorer adds:

```text
|P| * K
```

extra log-likelihood requests per sample.

Examples:

- 4-choice task with `label_permutations=0.25`: `6 * 4 = 24` extra requests
- 4-choice task with `label_permutations=0.5`: `12 * 4 = 48` extra requests
- 4-choice task with `label_permutations=1.0`: `24 * 4 = 96` extra requests

For small choice counts, Evalution rounds nonzero fractions up to a balanced minimum set so the
extra scorer actually averages label positions instead of collapsing to a single fixed labeling.
For binary tasks, any nonzero value therefore uses both permutations.

## Reference Alignment

When a benchmark ships official scoring code from the benchmark authors or the affiliated research
organization, Evalution prefers to encode that scoring rule directly and cite it in suite comments.
ARC and GSM8K are the clearest examples:

- ARC uses tie-aware exam scoring aligned with the AllenAI ARC solver release.
- GSM8K keeps the benchmark-owner parsers available for regression tests, while the live runtime
  score uses format-insensitive numeric matching to avoid answer-template lock-in.

The optional `label_permutations` scorer is intentionally not the benchmark-native default metric.
It exists as an extra diagnostic score for users who want to study how much fixed-answer formatting
or option length may be influencing a model.
