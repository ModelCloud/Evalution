# Scores

This file documents the short metric sub-labels used in Evalution score keys.

Metric keys use a compact comma-separated shape:

```text
<metric_family>,<score_variant>
```

Examples:

- `acc,ll`
- `acc,ll_avg`
- `acc,exam`
- `acc,num`
- `acc,label_perm:0.25`
- `em,choice_label`
- `f1,ll_avg_macro`
- `pct_stereotype`
- `likelihood_diff`
- `mcc,ll_avg`

## Metric Key Glossary

- `acc`: accuracy-like credit on a `0.0` to `1.0` scale for each sample, then averaged.
- `ll`: raw summed continuation log-likelihood over the scored answer tokens.
- `ll_avg`: average continuation log-likelihood per scored answer token to reduce length bias.
- `exam`: ARC exam-style tie-aware partial credit.
- `num`: numeric-answer match after numeric extraction and canonicalization.
- `em`: exact match after the suite's task-specific extraction step.
- `pct_stereotype`: fraction of minimal pairs where the more stereotypical sentence receives the higher score.
- `likelihood_diff`: average absolute log-likelihood gap between paired candidate sentences.
- `choice_label`: extracted option-label match such as `A/B/C/D`.
- `label_perm:<fraction>`: permutation-averaged label-only accuracy using the configured fraction
  of all label permutations.
- `f1`: F1 score derived from the suite's predicted labels.
- `mcc`: Matthews correlation coefficient derived from the suite's predicted labels.
- `macro`: macro-average across labels rather than a single positive class.
- `yes`: positive-class metric using the suite's `yes` or equivalent positive label.

## Notes

- `ll` and `ll_avg` differ only in how answer choice scores are normalized. `ll` sums token log
  probabilities. `ll_avg` divides by the number of scored answer tokens to reduce bias toward
  shorter options.
- `label_perm:<fraction>` is opt-in and additive. It does not replace the suite's default
  benchmark score.
- In keys like `f1,ll_avg_yes`, the `yes` suffix is not part of `ll_avg`. It states which class
  the F1 is computed for.
- Some suites expose only one metric family. For example, ARC uses `acc,exam`, GSM8K uses
  `acc,num`, and MMLU-Pro uses `em,choice_label`.

For scorer-module layout, benchmark-specific scoring comments, and the label permutation math, see
[scorers.md](scorers.md).
