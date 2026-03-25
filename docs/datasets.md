# Dataset Split Conventions

This document records the dataset identifier and split semantics for each Evalution benchmark.

## Policy

- Every dataset-backed benchmark exposes `split=` as an override.
- Every dataset-backed benchmark exposes `order=` as an override for row traversal order.
- Benchmark defaults are set per benchmark, not globally.
- Use the benchmark's canonical evaluation split for default reporting when the dataset exposes one.
- Keep non-`test` defaults when the public benchmark rows live under another name such as `validation`, `val`, `train`, `eval`, `test_r1`, or task-specific names like `qa1`.
- Internal split-like fields such as `fewshot_split` and `qa_split` are documented alongside the main evaluation split.
- Supported `order=` values are `native`, `shuffle`, `shuffle|<seed>`, `length|asc`, and `length|desc`.
- `order=` changes dataset row traversal only. It does not replace engine-internal request reordering used for batching efficiency.
- `shuffle` uses an implicit seed of `7` when no explicit `shuffle|<seed>` suffix is provided.
- When `stream=True`, `order` must stay `native`.

## Notable Cases

- `mmlu`: defaults to `split="test"` and `fewshot_split="dev"`. This matches benchmark-style reporting and keeps few-shot examples on the development split.
- `mmlu_pro`: defaults to `split="test"` and `fewshot_split="validation"`.
- `openbookqa`, `sciq`, `paws_x`, `xcopa`, `mathqa`, and `mc_taco` default to `split="test"` after auditing against the current `lm_eval` task configs, which declare an explicit `test_split`.
- `ceval`: uses `split="val"` because that dataset names its public evaluation rows `val`, not `validation` or `test`.
- `babilong`: uses task-specific split names such as `qa1`; both `split` and `qa_split` must agree.
- `anli`: uses round-specific splits like `test_r1`.
- Some benchmarks intentionally evaluate on `validation` because the dataset does not publish a public test target in the Hugging Face configuration used here.

## Benchmark Inventory

| Benchmark | Dataset Path | Dataset Name Default | Eval Split Default | Internal Splits |
| --- | --- | --- | --- | --- |
| `aexams.AEXAMS` | `Hennara/aexams` | `None` | `test` | none |
| `afrimgsm.AfriMGSM` | `masakhane/afrimgsm` | `eng` | `test` | none |
| `afrimmlu.AfriMMLU` | `masakhane/afrimmlu` | `eng` | `test` | none |
| `afrixnli.AfriXNLI` | `masakhane/afrixnli` | `eng` | `test` | none |
| `agieval.AGIEval` | `RUCAIBox/AGIEval` | `None` | `test` | none |
| `aime.AIME` | `gneubig/aime-1983-2024` | `None` | `train` | none |
| `alghafa.COPAArabic` | `Hennara/copa_ar` | `None` | `test` | none |
| `alghafa.PIQAArabic` | `Hennara/pica_ar` | `None` | `test` | none |
| `anli.ANLI` | `facebook/anli` | `None` | `test_r1` | none |
| `arabicmmlu.ArabicMMLU` | `MBZUAI/ArabicMMLU` | `None` | `test` | none |
| `arc_challenge.ARCChallenge` | `allenai/ai2_arc` | `ARC-Challenge` | `test` | none |
| `arc_easy.ARCEasy` | `allenai/ai2_arc` | `ARC-Easy` | `test` | none |
| `arc_mt.ARCMT` | `LumiOpen/arc_challenge_mt` or `mideind/icelandic-arc-challenge` | language code or `None` | `test` | none |
| `arithmetic.Arithmetic` | `EleutherAI/arithmetic` | `arithmetic_1dc` | `validation` | none |
| `asdiv.ASDiv` | `EleutherAI/asdiv` | `None` | `validation` | none |
| `asdiv.ASDivCoTLlama` | `EleutherAI/asdiv` | `None` | `validation` | none |
| `babi.BABI` | `Muennighoff/babi` | `None` | `test` | none |
| `babilong.BABILong` | `RMT-team/babilong` | `0k` | `qa1` | `qa_split=qa1` |
| `bbh.BBH` | `lukaemon/bbh` | `None` | `test` | none |
| `bear.BEAR` | `lm-pub-quiz/BEAR` | `BEAR` | `test` | none |
| `belebele.Belebele` | `facebook/belebele` | `eng_Latn` | `test` | none |
| `blimp.BLiMP` | `blimp` | `None` | `train` | none |
| `boolq.BoolQ` | `super_glue` | `boolq` | `validation` | none |
| `c4.C4` | `allenai/c4` | `en` | `validation` | none |
| `cabbq.CaBBQ` | `BSC-LT/CaBBQ` | `Age` | `test` | none |
| `careqa.CareQA` | `HPAI-BSC/CareQA` | `CareQA_en` | `test` | none |
| `cb.CB` | `super_glue` | `cb` | `validation` | none |
| `click.Click` | `EunsuKim/CLIcK` | `None` | `train` | none |
| `ceval.CEval` | `ceval/ceval-exam` | `None` | `val` | none |
| `cnn_dailymail.CNNDailyMail` | `cnn_dailymail` | `3.0.0` | `validation` | none |
| `code_x_glue.CodeXGLUECodeToText` | language-specific `CM/codexglue_code2text_*` | `None` | `test` | none |
| `cola.CoLA` | `nyu-mll/glue` | `cola` | `validation` | none |
| `commonsense_qa.CommonsenseQA` | `tau/commonsense_qa` | `None` | `validation` | none |
| `copa.COPA` | `super_glue` | `copa` | `validation` | none |
| `copal_id.COPALID` | `haryoaw/COPAL` | `id` | `test` | none |
| `coqa.CoQA` | `coqa` | `None` | `validation` | none |
| `crows_pairs.CrowSPairs` | `jannalu/crows_pairs_multilingual` | `english` | `test` | none |
| `darijammlu.DarijaMMLU` | `MBZUAI-Paris/DarijaMMLU` | `None` | `test` | none |
| `drop.DROP` | `drop` | `None` | `validation` | none |
| `fld.FLD` | `hitachi-nlp/FLD.v2` | `default` | `test` | none |
| `fda.FDA` | `hazyresearch/based-fda` | `default` | `validation` | none |
| `moral_stories.MoralStories` | `LabHC/moral_stories` | `None` | `train` | none |
| `french_bench_arc_challenge.FrenchBenchARCChallenge` | `manu/french_bench_arc_challenge` | `None` | `test` | none |
| `egymmlu.EgyMMLU` | `UBC-NLP/EgyMMLU` | `None` | `test` | none |
| `esbbq.EsBBQ` | `BSC-LT/EsBBQ` | `Age` | `test` | none |
| `eus_exams.EusExams` | `HiTZ/EusExams` | `None` | `test` | none |
| `eus_reading.EusReading` | `HiTZ/EusReading` | `default` | `test` | none |
| `eus_proficiency.EusProficiency` | `HiTZ/EusProficiency` | `default` | `test` | none |
| `eus_trivia.EusTrivia` | `HiTZ/EusTrivia` | `default` | `test` | none |
| `gpqa.GPQA` | `Idavidrein/gpqa` | `None` | `train` | none |
| `gsm_plus.GSMPlus` | `qintongli/GSM-Plus` | `None` | `test` | none |
| `gsm_plus.GSMPlusMini` | `qintongli/GSM-Plus` | `None` | `testmini` | none |
| `haerae.Haerae` | `HAERAE-HUB/HAE_RAE_BENCH` | `general_knowledge` or `None` | `test` | `subset=...` |
| `kormedmcqa.KorMedMCQA` | `sean0042/KorMedMCQA` | `doctor` or `None` | `test` | `subset=...`, `fewshot_split=fewshot` |
| `graphwalks.graphwalks_128k` | `openai/graphwalks` | `None` | `train` | `data_file=graphwalks_128k_and_shorter.parquet` |
| `graphwalks.graphwalks_1M` | `openai/graphwalks` | `None` | `train` | `data_file=graphwalks_256k_to_1mil.parquet` |
| `gsm8k.GSM8K` | `openai/gsm8k` | `main` | `test` | none |
| `gsm8k_platinum.GSM8KPlatinum` | `madrylab/gsm8k-platinum` | `main` | `test` | none |
| `headqa.HEADQA` | `EleutherAI/headqa` | `en` | `test` | none |
| `hellaswag.HellaSwag` | `Rowan/hellaswag` | `None` | `validation` | none |
| `hendrycks_ethics.HendrycksEthics` | `EleutherAI/hendrycks_ethics` | `commonsense` | `test` | none |
| `histoires_morales.HistoiresMorales` | `LabHC/histoires_morales` | `None` | `train` | none |
| `icelandic_winogrande.IcelandicWinoGrande` | `mideind/icelandic-winogrande` | `None` | `train` | none |
| `inverse_scaling.InverseScaling` | `pminervini/inverse-scaling` | `hindsight-neglect` | `data` | none |
| `kobest.KoBEST` | `skt/kobest_v1` | `None` | `test` | none |
| `lambada.LAMBADA` | `EleutherAI/lambada_openai` | `default` | `test` | none |
| `lambada.LAMBADA` | `EleutherAI/lambada_openai` | `de`, `en`, `es`, `fr`, or `it` | `test` | `variant_name="openai_mt_<lang>"` |
| `lambada.LAMBADA` | `EleutherAI/lambada_multilingual_stablelm` | `de`, `en`, `es`, `fr`, `it`, `nl`, or `pt` | `test` | `variant_name="openai_mt_stablelm_<lang>"` |
| `logiqa.LogiQA` | `EleutherAI/logiqa` | `logiqa` | `validation` | none |
| `logiqa2.LogiQA2` | `datatune/LogiQA2.0` | `None` | `test` | none |
| `mathqa.MathQA` | `math_qa` | `None` | `test` | none |
| `mc_taco.MCTACO` | `CogComp/mc_taco` | `None` | `test` | none |
| `medmcqa.MedMCQA` | `openlifescienceai/medmcqa` | `None` | `validation` | none |
| `medqa.MedQA` | `GBaker/MedQA-USMLE-4-options-hf` | `None` | `test` | none |
| `mmlu.MMLU` | `cais/mmlu` | `all` or leaf subset name | `test` | `fewshot_split=dev` |
| `multirc.MultiRC` | `super_glue` | `multirc` | `validation` | none |
| `mastermind.Mastermind` | variant-dependent: `flair/mastermind_24_mcq_random`, `flair/mastermind_24_mcq_close`, `flair/mastermind_35_mcq_random`, `flair/mastermind_35_mcq_close`, `flair/mastermind_46_mcq_random`, `flair/mastermind_46_mcq_close` | `None` | `test` | none |
| `mbpp.MBPP` | `mbpp` | `sanitized` | `test` | none |
| `mmlu_pro.MMLUPro` | `TIGER-Lab/MMLU-Pro` | `None` | `test` | `fewshot_split=validation` |
| `mnli.MNLI` | `nyu-mll/glue` | `mnli` | `validation_matched` | none |
| `mrpc.MRPC` | `nyu-mll/glue` | `mrpc` | `validation` | none |
| `multirc.MultiRC` | `super_glue` | `multirc` | `validation` | none |
| `nq_open.NQOpen` | `nq_open` | `nq_open` | `validation` | none |
| `openbookqa.OpenBookQA` | `allenai/openbookqa` | `main` | `test` | none |
| `paws_x.PAWSX` | `paws-x` | `en` | `test` | none |
| `pile_10k.Pile10K` | `monology/pile-uncopyrighted` | `None` | `train` | none |
| `polemo2.Polemo2` | variant-dependent: `allegro/klej-polemo2-in`, `allegro/klej-polemo2-out` | `None` | `test` | none |
| `piqa.PIQA` | `baber/piqa` | `None` | `validation` | none |
| `prost.Prost` | `corypaik/prost` | `None` | `test` | none |
| `pubmedqa.PubMedQA` | `bigbio/pubmed_qa` | `pubmed_qa_labeled_fold0_source` | `test` | none |
| `qa4mre.QA4MRE` | `qa4mre` | `2011.main.EN` | `train` | none |
| `qnli.QNLI` | `nyu-mll/glue` | `qnli` | `validation` | none |
| `qqp.QQP` | `nyu-mll/glue` | `qqp` | `validation` | none |
| `race.RACE` | `EleutherAI/race` | `high` | `test` | none |
| `record.ReCoRD` | `super_glue` | `record` | `validation` | none |
| `rte.RTE` | `super_glue` | `rte` | `validation` | none |
| `sciq.SciQ` | `allenai/sciq` | `None` | `test` | none |
| `siqa.SIQA` | `allenai/social_i_qa` | `None` | `validation` | none |
| `squadv2.SQuADV2` | `squad_v2` | `squad_v2` | `validation` | none |
| `sst2.SST2` | `nyu-mll/glue` | `sst2` | `validation` | none |
| `swag.SWAG` | `swag` | `regular` | `validation` | none |
| `toxigen.ToxiGen` | `skg/toxigen-data` | `annotated` | `test` | none |
| `triviaqa.TriviaQA` | `trivia_qa` | `rc.nocontext` | `validation` | none |
| `truthfulqa.TruthfulQAMC` | `truthfulqa/truthful_qa` | `multiple_choice` | `validation` | none |
| `webqs.WebQS` | `web_questions` | `None` | `test` | none |
| `wic.WiC` | `super_glue` | `wic` | `validation` | none |
| `wikitext.WikiText` | `EleutherAI/wikitext_document_level` | `wikitext-2-raw-v1` | `test` | none |
| `winogender.WinoGender` | `oskarvanderwal/winogender` | `all` | `test` | none |
| `winogrande.WinoGrande` | `winogrande` | `winogrande_xl` | `validation` | none |
| `wsc.WSC` | `super_glue` | `wsc.fixed` | `validation` | none |
| `wnli.WNLI` | `nyu-mll/glue` | `wnli` | `validation` | none |
| `wsc273.WSC273` | `winograd_wsc` | `wsc273` | `test` | none |
| `xcopa.XCOPA` | `xcopa` | `it` | `test` | none |
| `xnli.XNLI` | `facebook/xnli` | `en` | `validation` | none |
| `xnli_eu.XNLIEU` | `HiTZ/xnli-eu` | `eu` | `test` | none |
| `xquad.XQuAD` | `google/xquad` | `xquad.en` | `validation` | none |
| `xstorycloze.XStoryCloze` | `juletxara/xstory_cloze` | `en` | `eval` | none |
| `xwinograd.XWinograd` | `Muennighoff/xwinograd` | `en` | `test` | none |
| `bangla.Bangla` | subset-dependent: `hishab/boolq_bn`, `hishab/commonsenseqa-bn`, `hishab/titulm-bangla-mmlu`, `hishab/openbookqa-bn`, `hishab/piqa-bn` | subset-dependent | subset-dependent: `validation` or `test` | none |
| `darijahellaswag.DarijaHellaSwag` | `MBZUAI-Paris/DarijaHellaSwag` | `None` | `validation` | none |
| `egyhellaswag.EgyHellaSwag` | `UBC-NLP/EgyHellaSwag` | `None` | `validation` | none |

Entries with dynamic dataset paths, names, or split defaults are documented by their runtime convention rather than a single literal value.
