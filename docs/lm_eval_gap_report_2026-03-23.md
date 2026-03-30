# lm-eval Gap Report (2026-03-23)

## Snapshot

- Local repo: `Evalution` at `/root/Evalution`
- Upstream repo: `EleutherAI/lm-evaluation-harness` main @ `ee7e8f4fe58e13d6760c066474f0d01477317d1d` (2026-03-18, Intel Gaudi support via optimum-habana (#3550))
- Open PRs scanned: 199 (`page 1 = 100`, `page 2 = 99`)
- Local exported benchmarks: 22
- Upstream task families with YAMLs on main: 208
- Comparable upstream families after excluding obvious meta/config wrappers ['benchmarks', 'leaderboard', 'llama3', 'metabench', 'score', 'tinyBenchmarks']: 202
- Covered upstream families locally: 11
- Missing comparable families from upstream main: 191
- Open PR family rows missing locally: 138
- Open PR rows introducing families not yet on main: 85
- Open PR rows extending families already on main but still missing locally: 53

## Local Coverage

`arc_challenge`, `arc_easy`, `boolq`, `cb`, `cola`, `copa`, `gsm8k`, `gsm8k_platinum`, `hellaswag`, `mmlu`, `mmlu_pro`, `mnli`, `mrpc`, `openbookqa`, `piqa`, `qnli`, `qqp`, `rte`, `sst2`, `wic`, `winogrande`, `wnli`

These 22 exported benchmarks map to 11 upstream family roots: `arc`, `glue`, `super_glue`, `gsm8k`, `gsm8k_platinum`, `hellaswag`, `mmlu`, `mmlu_pro`, `openbookqa`, `piqa`, and `winogrande`.

## Partial Gaps Inside Covered Families

| Upstream family | Missing benchmark/task | Upstream ref | Note |
| --- | --- | --- | --- |
| `glue` | `rte` | `glue/rte/default.yaml` | Local `rte` targets SuperGLUE (`super_glue/rte`), not GLUE RTE. |
| `super_glue` | `multirc` | `super_glue/multirc/default.yaml` | Benchmark present upstream; not implemented locally. |
| `super_glue` | `record` | `super_glue/record/default.yaml` | Benchmark present upstream; not implemented locally. |
| `super_glue` | `wsc` | `super_glue/wsc/default.yaml` | Benchmark present upstream; not implemented locally. |

Prompt variants and smoke-test bundles were not counted as benchmark gaps here. For example, `gsm8k-cot` and `mmlu` prompt wrappers exist upstream, but they are not separate benchmark families.

## Missing Families From Upstream Main

Full CSV: [`lm_eval_missing_main_families_2026-03-23.csv`](./lm_eval_missing_main_families_2026-03-23.csv)

| Family | YAML files | Example task/config 1 | Example task/config 2 |
| --- | ---: | --- | --- |
| `aclue` | 16 | `aclue/aclue_ancient_chinese_culture.yaml` | `aclue/aclue_ancient_literature.yaml` |
| `acpbench` | 30 | `acpbench/boolq_cot_2shot/act_reach.yaml` | `acpbench/boolq_cot_2shot/app.yaml` |
| `aexams` | 6 | `aexams/aexams_Biology.yaml` | `aexams/aexams_IslamicStudies.yaml` |
| `afrimgsm` | 369 | `afrimgsm/direct/afrimgsm.yaml` | `afrimgsm/direct/prompt_1/afrimgsm_amh.yaml` |
| `afrimmlu` | 177 | `afrimmlu/direct/afrimmlu.yaml` | `afrimmlu/direct/prompt_1/afrimmlu_direct_amh.yaml` |
| `afrixnli` | 265 | `afrixnli/anli prompt/en-direct/afrixnli_en_direct_amh.yaml` | `afrixnli/anli prompt/en-direct/afrixnli_en_direct_eng.yaml` |
| `afrobench` | 1818 | `afrobench/adr/afridiacritics.yaml` | `afrobench/adr/prompt_1/afridiacritics_bbj.yaml` |
| `agieval` | 25 | `agieval/agieval.yaml` | `agieval/agieval_cn.yaml` |
| `aime` | 3 | `aime/aime.yaml` | `aime/aime24.yaml` |
| `alghafa` | 2 | `alghafa/copa_ar/copa_ar.yaml` | `alghafa/piqa_ar/piqa_ar.yaml` |
| `anli` | 3 | `anli/anli_r1.yaml` | `anli/anli_r2.yaml` |
| `arab_culture` | 18 | `arab_culture/arab_culture_algeria.yaml` | `arab_culture/arab_culture_egypt.yaml` |
| `arab_culture_completion` | 18 | `arab_culture_completion/arab_culture_completion_algeria.yaml` | `arab_culture_completion/arab_culture_completion_egypt.yaml` |
| `arabic_leaderboard_complete` | 152 | `arabic_leaderboard_complete/arabic_leaderboard_alghafa/arabic_leaderboard_alghafa.yaml` | `arabic_leaderboard_complete/arabic_leaderboard_alghafa/arabic_leaderboard_alghafa_mcq_exams_test_ar.yaml` |
| `arabic_leaderboard_light` | 152 | `arabic_leaderboard_light/arabic_leaderboard_alghafa_light/arabic_leaderboard_alghafa_light.yaml` | `arabic_leaderboard_light/arabic_leaderboard_alghafa_light/arabic_leaderboard_alghafa_mcq_exams_test_ar_light.yaml` |
| `arabicmmlu` | 46 | `arabicmmlu/arabicmmlu_accounting_university.yaml` | `arabicmmlu/arabicmmlu_arabic_language_general.yaml` |
| `aradice` | 109 | `aradice/ArabicMMLU/EGY/AraDiCE_ArabicMMLU.yaml` | `aradice/ArabicMMLU/EGY/AraDiCE_ArabicMMLU_high_humanities_history.yaml` |
| `arc_mt` | 12 | `arc_mt/arc_challenge_mt_da.yaml` | `arc_mt/arc_challenge_mt_de.yaml` |
| `arithmetic` | 10 | `arithmetic/arithmetic_1dc.yaml` | `arithmetic/arithmetic_2da.yaml` |
| `asdiv` | 2 | `asdiv/asdiv-cot-llama.yaml` | `asdiv/default.yaml` |
| `babi` | 1 | `babi/babi.yaml` |  |
| `babilong` | 22 | `babilong/babilong.yaml` | `babilong/babilong_longctx.yaml` |
| `bangla` | 5 | `bangla/bangla_boolqa.yaml` | `bangla/bangla_commonsenseqa.yaml` |
| `basque_bench` | 26 | `basque_bench/arc_eu_challenge.yaml` | `basque_bench/arc_eu_easy.yaml` |
| `basqueglue` | 6 | `basqueglue/bec.yaml` | `basqueglue/bhtc.yaml` |
| `bbh` | 113 | `bbh/cot_fewshot/boolean_expressions.yaml` | `bbh/cot_fewshot/causal_judgement.yaml` |
| `bbq` | 6 | `bbq/bbq_generate.yaml` | `bbq/bbq_generate_ambig.yaml` |
| `bear` | 2 | `bear/bear.yaml` | `bear/bear_big.yaml` |
| `belebele` | 123 | `belebele/belebele_acm_Arab.yaml` | `belebele/belebele_afr_Latn.yaml` |
| `bertaqa` | 16 | `bertaqa/bertaqa_en.yaml` | `bertaqa/bertaqa_en_mt_gemma-7b.yaml` |
| `bhs` | 25 | `bhs/basque-DO-S_DO_V_AUX.yaml` | `bhs/basque-DO-S_IO_DO_V_AUX.yaml` |
| `bigbench` | 286 | `bigbench/generate_until/abstract_narrative_understanding.yaml` | `bigbench/generate_until/anachronisms.yaml` |
| `blimp` | 68 | `blimp/adjunct_island.yaml` | `blimp/anaphor_gender_agreement.yaml` |
| `blimp_nl` | 85 | `blimp_nl/adpositional_phrases__argument_r_extraction.yaml` | `blimp_nl/adpositional_phrases__argument_scrambling.yaml` |
| `c4` | 1 | `c4/c4.yaml` |  |
| `cabbq` | 11 | `cabbq/cabbq.yaml` | `cabbq/cabbq_age.yaml` |
| `careqa` | 4 | `careqa/careqa_en.yaml` | `careqa/careqa_es.yaml` |
| `catalan_bench` | 43 | `catalan_bench/arc_ca_challenge.yaml` | `catalan_bench/arc_ca_easy.yaml` |
| `ceval` | 53 | `ceval/ceval-valid_accountant.yaml` | `ceval/ceval-valid_advanced_mathematics.yaml` |
| `chartqa` | 3 | `chartqa/chartqa.yaml` | `chartqa/chartqa_llama.yaml` |
| `click` | 14 | `click/click.yaml` | `click/click_cul/click_cul_economy.yaml` |
| `cmmlu` | 68 | `cmmlu/cmmlu_agronomy.yaml` | `cmmlu/cmmlu_anatomy.yaml` |
| `cnn_dailymail` | 1 | `cnn_dailymail/cnn_dailymail.yaml` |  |
| `code_x_glue` | 7 | `code_x_glue/code-text/go.yaml` | `code_x_glue/code-text/java.yaml` |
| `common_voice` | 1 | `common_voice/common_voice_en.yaml` |  |
| `commonsense_qa` | 1 | `commonsense_qa/default.yaml` |  |
| `copal_id` | 2 | `copal_id/colloquial.yaml` | `copal_id/standard.yaml` |
| `coqa` | 1 | `coqa/default.yaml` |  |
| `crows_pairs` | 22 | `crows_pairs/crows_pairs_english.yaml` | `crows_pairs/crows_pairs_english_age.yaml` |
| `csatqa` | 7 | `csatqa/csatqa_gr.yaml` | `csatqa/csatqa_li.yaml` |
| `darija_bench` | 37 | `darija_bench/darija_sentiment/darija_sentiment.yaml` | `darija_bench/darija_sentiment/darija_sentiment_electrom.yaml` |
| `darijahellaswag` | 1 | `darijahellaswag/darijahellaswag.yaml` |  |
| `darijammlu` | 47 | `darijammlu/darijammlu_accounting.yaml` | `darijammlu/darijammlu_arabic_language.yaml` |
| `discrim_eval` | 2 | `discrim_eval/discrim_eval_explicit.yaml` | `discrim_eval/discrim_eval_implicit.yaml` |
| `drop` | 1 | `drop/default.yaml` |  |
| `e2lmc` | 20 | `e2lmc/mmlu_early_training/mmlu_early_training.yaml` | `e2lmc/noor/noor_abstract_algebra.yaml` |
| `egyhellaswag` | 1 | `egyhellaswag/egyhellaswag.yaml` |  |
| `egymmlu` | 47 | `egymmlu/egymmlu_accounting.yaml` | `egymmlu/egymmlu_arabic_language.yaml` |
| `eq_bench` | 3 | `eq_bench/default.yaml` | `eq_bench/multilingual/eqbench_ca.yaml` |
| `esbbq` | 11 | `esbbq/esbbq.yaml` | `esbbq/esbbq_age.yaml` |
| `eus_exams` | 62 | `eus_exams/eus_exams_es_ejadministrativo.yaml` | `eus_exams/eus_exams_es_ejauxiliar.yaml` |
| `eus_proficiency` | 1 | `eus_proficiency/eus_proficiency.yaml` |  |
| `eus_reading` | 1 | `eus_reading/eus_reading.yaml` |  |
| `eus_trivia` | 1 | `eus_trivia/eus_trivia.yaml` |  |
| `evalita_llm` | 73 |  |  |
| `fda` | 1 | `fda/fda.yaml` |  |
| `fld` | 4 | `fld/fld_default.yaml` | `fld/fld_logical_formula_default.yaml` |
| `french_bench` | 18 | `french_bench/french_bench_arc_challenge.yaml` | `french_bench/french_bench_boolqa.yaml` |
| `galician_bench` | 30 | `galician_bench/belebele_glg_Latn.yaml` | `galician_bench/flores_gl/flores_ca-gl.yaml` |
| `glianorex` | 3 | `glianorex/glianorex.yaml` | `glianorex/glianorex_en.yaml` |
| `global_mmlu` | 2709 | `global_mmlu/default/ar/global_mmlu_ar_business.yaml` | `global_mmlu/default/ar/global_mmlu_ar_humanities.yaml` |
| `global_piqa` | 234 | `global_piqa/completions/acm_arab.yaml` | `global_piqa/completions/acq_arab.yaml` |
| `gpqa` | 15 | `gpqa/cot_n_shot/gpqa_diamond_cot_n_shot.yaml` | `gpqa/cot_n_shot/gpqa_extended_cot_n_shot.yaml` |
| `graphwalks` | 3 | `graphwalks/graphwalks.yaml` | `graphwalks/graphwalks_128k.yaml` |
| `groundcocoa` | 1 | `groundcocoa/groundcocoa.yaml` |  |
| `gsm_plus` | 2 | `gsm_plus/gsm_plus.yaml` | `gsm_plus/gsm_plus_mini.yaml` |
| `haerae` | 6 | `haerae/haerae_gk.yaml` | `haerae/haerae_hi.yaml` |
| `headqa` | 2 | `headqa/headqa_en.yaml` | `headqa/headqa_es.yaml` |
| `hendrycks_ethics` | 5 | `hendrycks_ethics/commonsense.yaml` | `hendrycks_ethics/deontology.yaml` |
| `hendrycks_math` | 9 | `hendrycks_math/hendrycks_math.yaml` | `hendrycks_math/hendrycks_math500.yaml` |
| `histoires_morales` | 1 | `histoires_morales/histoires_morales.yaml` |  |
| `hrm8k` | 12 | `hrm8k/default/hrm8k.yaml` | `hrm8k/default/hrm8k_gsm8k.yaml` |
| `humaneval` | 5 | `humaneval/humaneval.yaml` | `humaneval/humaneval_64.yaml` |
| `humaneval_infilling` | 5 | `humaneval_infilling/humaneval_infilling.yaml` | `humaneval_infilling/multi_line_infilling.yaml` |
| `icelandic_winogrande` | 1 | `icelandic_winogrande/default.yaml` |  |
| `ifeval` | 3 | `ifeval/ifeval.yaml` | `ifeval/multilingual/ifeval_ca.yaml` |
| `include` | 666 | `include/default/Albanian/include_base_44_albanian_arts_humanities.yaml` | `include/default/Albanian/include_base_44_albanian_business_commerce.yaml` |
| `inverse_scaling` | 11 | `inverse_scaling/inverse_scaling_hindsight_neglect.yaml` | `inverse_scaling/inverse_scaling_into_the_unknown.yaml` |
| `japanese_leaderboard` | 9 | `japanese_leaderboard/ja_leaderboard_jaqket_v2.yaml` | `japanese_leaderboard/ja_leaderboard_jcommonsenseqa.yaml` |
| `jfinqa` | 4 | `jfinqa/jfinqa_consistency.yaml` | `jfinqa/jfinqa_numerical.yaml` |
| `jsonschema_bench` | 3 | `jsonschema_bench/jsonschema_bench_easy.yaml` | `jsonschema_bench/jsonschema_bench_hard.yaml` |
| `kbl` | 65 | `kbl/bar_exam/civil/kbl_bar_exam_em_civil_2012.yaml` | `kbl/bar_exam/civil/kbl_bar_exam_em_civil_2013.yaml` |
| `kmmlu` | 250 | `kmmlu/cot_hard/kmmlu_cot_hard_accounting.yaml` | `kmmlu/cot_hard/kmmlu_cot_hard_agricultural_sciences.yaml` |
| `kobest` | 6 | `kobest/kobest_boolq.yaml` | `kobest/kobest_copa.yaml` |
| `kormedmcqa` | 5 | `kormedmcqa/dentist.yaml` | `kormedmcqa/doctor.yaml` |
| `lambada` | 2 | `lambada/lambada_openai.yaml` | `lambada/lambada_standard.yaml` |
| `lambada_cloze` | 2 | `lambada_cloze/lambada_openai_cloze.yaml` | `lambada_cloze/lambada_standard_cloze.yaml` |
| `lambada_multilingual` | 5 | `lambada_multilingual/lambada_mt_de.yaml` | `lambada_multilingual/lambada_mt_en.yaml` |
| `lambada_multilingual_stablelm` | 7 | `lambada_multilingual_stablelm/lambada_mt_stablelm_de.yaml` | `lambada_multilingual_stablelm/lambada_mt_stablelm_en.yaml` |
| `libra` | 22 | `libra/librusec_history.yaml` | `libra/librusec_mhqa.yaml` |
| `lingoly` | 3 | `lingoly/lingoly_context.yaml` | `lingoly/lingoly_group.yaml` |
| `lm_syneval` | 73 | `lm_syneval/lm_syneval__agreement__long_vp_coord__plur_MS_LMV_LMV.yaml` | `lm_syneval/lm_syneval__agreement__long_vp_coord__sing_MS_LMV_LMV.yaml` |
| `logiqa` | 1 | `logiqa/logiqa.yaml` |  |
| `logiqa2` | 2 | `logiqa2/logieval.yaml` | `logiqa2/logiqa2.yaml` |
| `longbench` | 48 | `longbench/2wikimqa.yaml` | `longbench/2wikimqa_e.yaml` |
| `longbench2` | 26 | `longbench2/academic_multi_doc.yaml` | `longbench2/academic_single.yaml` |
| `mastermind` | 6 | `mastermind/mastermind_24_easy.yaml` | `mastermind/mastermind_24_hard.yaml` |
| `mathqa` | 1 | `mathqa/mathqa.yaml` |  |
| `mbpp` | 4 | `mbpp/mbpp.yaml` | `mbpp/mbpp_instruct.yaml` |
| `mc_taco` | 1 | `mc_taco/default.yaml` |  |
| `med_concepts_qa` | 21 | `med_concepts_qa/med_concepts_qa_atc_easy.yaml` | `med_concepts_qa/med_concepts_qa_atc_hard.yaml` |
| `med_prescriptions` | 2 | `med_prescriptions/med_prescriptions_easy.yaml` | `med_prescriptions/med_prescriptions_hard.yaml` |
| `med_text_classification` | 2 | `med_text_classification/med_text_classification_easy.yaml` | `med_text_classification/med_text_classification_hard.yaml` |
| `meddialog` | 4 | `meddialog/meddialog_qsumm.yaml` | `meddialog/meddialog_qsumm_perplexity.yaml` |
| `mediqa_qa2019` | 2 | `mediqa_qa2019/mediqa_qa2019.yaml` | `mediqa_qa2019/mediqa_qa2019_perplexity.yaml` |
| `medmcqa` | 1 | `medmcqa/medmcqa.yaml` |  |
| `medqa` | 1 | `medqa/medqa.yaml` |  |
| `medtext` | 2 | `medtext/medtext.yaml` | `medtext/medtext_perplexity.yaml` |
| `mela` | 11 | `mela/mela_ar.yaml` | `mela/mela_de.yaml` |
| `meqsum` | 1 | `meqsum/meqsum.yaml` |  |
| `mgsm` | 33 | `mgsm/direct/mgsm_direct_bn.yaml` | `mgsm/direct/mgsm_direct_de.yaml` |
| `mimic_repsum` | 2 | `mimic_repsum/mimic_repsum.yaml` | `mimic_repsum/mimic_repsum_perplexity.yaml` |
| `minerva_math` | 8 | `minerva_math/minerva_math500.yaml` | `minerva_math/minerva_math_algebra.yaml` |
| `mlqa` | 49 | `mlqa/mlqa_ar_ar.yaml` | `mlqa/mlqa_ar_de.yaml` |
| `mmlu-pro-plus` | 15 | `mmlu-pro-plus/mmlu_pro_plus_biology.yaml` | `mmlu-pro-plus/mmlu_pro_plus_business.yaml` |
| `mmlu-redux` | 58 | `mmlu-redux/generative/mmlu_abstract_algebra.yaml` | `mmlu-redux/generative/mmlu_anatomy.yaml` |
| `mmlu-redux-spanish` | 59 | `mmlu-redux-spanish/generative/mmlu_abstract_algebra.yaml` | `mmlu-redux-spanish/generative/mmlu_anatomy.yaml` |
| `mmlu_prox` | 870 | `mmlu_prox/af/mmlu_prox_af_biology.yaml` | `mmlu_prox/af/mmlu_prox_af_business.yaml` |
| `mmlusr` | 174 | `mmlusr/answer_only/answer_only_abstract_algebra.yaml` | `mmlusr/answer_only/answer_only_anatomy.yaml` |
| `mmmu` | 37 | `mmmu/mmmu_accounting.yaml` | `mmmu/mmmu_agriculture.yaml` |
| `model_written_evals` | 187 | `model_written_evals/advanced_ai_risk/fewshot-coordinate-itself.yaml` | `model_written_evals/advanced_ai_risk/fewshot-coordinate-other-ais.yaml` |
| `moral_stories` | 1 | `moral_stories/moral_stories.yaml` |  |
| `mts_dialog` | 2 | `mts_dialog/mts_dialog.yaml` | `mts_dialog/mts_dialog_perplexity.yaml` |
| `multiblimp` | 101 | `multiblimp/multiblimp_abk.yaml` | `multiblimp/multiblimp_aln.yaml` |
| `mutual` | 2 | `mutual/multual_plus.yaml` | `mutual/mutual.yaml` |
| `noreval` | 116 | `noreval/ask_gec/ask_gec_p0.yaml` | `noreval/ask_gec/ask_gec_p1.yaml` |
| `noticia` | 1 | `noticia/noticia.yaml` |  |
| `nq_open` | 1 | `nq_open/nq_open.yaml` |  |
| `okapi` | 157 | `okapi/arc_multilingual/arc_ar.yaml` | `okapi/arc_multilingual/arc_bn.yaml` |
| `olaph` | 2 | `olaph/olaph.yaml` | `olaph/olaph_perplexity.yaml` |
| `openai-mmmlu` | 869 | `openai-mmmlu/default/mmmlu_ar_xy_abstract_algebra.yaml` | `openai-mmmlu/default/mmmlu_ar_xy_anatomy.yaml` |
| `paloma` | 16 | `paloma/paloma_4chan_meta_sep.yaml` | `paloma/paloma_c4_100_domains.yaml` |
| `paws-x` | 8 | `paws-x/paws_de.yaml` | `paws-x/paws_en.yaml` |
| `pile` | 22 | `pile/pile_arxiv.yaml` | `pile/pile_bookcorpus2.yaml` |
| `pile_10k` | 1 | `pile_10k/pile_10k.yaml` |  |
| `pisa` | 14 | `pisa/pisa_ch.yaml` | `pisa/pisa_ch_llm_judged.yaml` |
| `polemo2` | 2 | `polemo2/polemo2_in.yaml` | `polemo2/polemo2_out.yaml` |
| `portuguese_bench` | 20 | `portuguese_bench/assin_entailment.yaml` | `portuguese_bench/assin_paraphrase.yaml` |
| `prost` | 1 | `prost/corypaik_prost.yaml` |  |
| `pubmedqa` | 1 | `pubmedqa/pubmedqa.yaml` |  |
| `qa4mre` | 3 | `qa4mre/qa4mre_2011.yaml` | `qa4mre/qa4mre_2012.yaml` |
| `qasper` | 2 | `qasper/bool.yaml` | `qasper/freeform.yaml` |
| `race` | 1 | `race/race.yaml` |  |
| `realtoxicityprompts` | 1 | `realtoxicityprompts/realtoxicityprompts.yaml` |  |
| `ruler` | 14 | `ruler/cwe.yaml` | `ruler/fwe.yaml` |
| `sciq` | 1 | `sciq/sciq.yaml` |  |
| `scrolls` | 7 | `scrolls/scrolls_contractnli.yaml` | `scrolls/scrolls_govreport.yaml` |
| `simple_cooccurrence_bias` | 2 | `simple_cooccurrence_bias/simple_cooccurrence_bias.yaml` | `simple_cooccurrence_bias/simple_cooccurrence_bias_gen.yaml` |
| `siqa` | 1 | `siqa/siqa.yaml` |  |
| `slr_bench` | 6 | `slr_bench/slr_bench_all.yaml` | `slr_bench/slr_bench_basic.yaml` |
| `spanish_bench` | 29 | `spanish_bench/cocoteros_es.yaml` | `spanish_bench/copa_es.yaml` |
| `squad_completion` | 1 | `squad_completion/squad_completion.yaml` |  |
| `squadv2` | 1 | `squadv2/squadv2.yaml` |  |
| `storycloze` | 2 | `storycloze/storycloze_2016.yaml` | `storycloze/storycloze_2018.yaml` |
| `swag` | 1 | `swag/swag.yaml` |  |
| `swde` | 1 | `swde/swde.yaml` |  |
| `tmlu` | 32 | `tmlu/default/tmlu_AST_biology.yaml` | `tmlu/default/tmlu_AST_chemistry.yaml` |
| `tmmluplus` | 72 | `tmmluplus/default/tmmluplus_accounting.yaml` | `tmmluplus/default/tmmluplus_administrative_law.yaml` |
| `toxigen` | 1 | `toxigen/toxigen.yaml` |  |
| `translation` | 8 | `translation/iwslt2017_ar-en.yaml` | `translation/iwslt2017_en-ar.yaml` |
| `triviaqa` | 1 | `triviaqa/default.yaml` |  |
| `truthfulqa` | 3 | `truthfulqa/truthfulqa_gen.yaml` | `truthfulqa/truthfulqa_mc1.yaml` |
| `truthfulqa-multi` | 15 | `truthfulqa-multi/truthfulqa-multi_gen_ca.yaml` | `truthfulqa-multi/truthfulqa-multi_gen_en.yaml` |
| `turblimp` | 17 | `turblimp/anaphor_agreement.yaml` | `turblimp/argument_structure_ditransitive.yaml` |
| `turkishmmlu` | 18 | `turkishmmlu/config/Biology.yaml` | `turkishmmlu/config/Chemistry.yaml` |
| `ulqa` | 12 | `ulqa/celep1.yaml` | `ulqa/celep2.yaml` |
| `unitxt` | 19 | `unitxt/20_newsgroups.yaml` | `unitxt/ag_news.yaml` |
| `unscramble` | 5 | `unscramble/anagrams1.yaml` | `unscramble/anagrams2.yaml` |
| `webqs` | 1 | `webqs/webqs.yaml` |  |
| `wikitext` | 1 | `wikitext/wikitext.yaml` |  |
| `winogender` | 7 | `winogender/winogender.yaml` | `winogender/winogender_female.yaml` |
| `wmdp` | 4 | `wmdp/wmdp_bio.yaml` | `wmdp/wmdp_chem.yaml` |
| `wmt2016` | 1 | `wmt2016/ro_en-t5_prompt.yaml` |  |
| `wsc273` | 1 | `wsc273/default.yaml` |  |
| `xcopa` | 12 | `xcopa/default_et.yaml` | `xcopa/default_ht.yaml` |
| `xnli` | 16 | `xnli/xnli_ar.yaml` | `xnli/xnli_bg.yaml` |
| `xnli_eu` | 3 | `xnli_eu/xnli_eu.yaml` | `xnli_eu/xnli_eu_mt.yaml` |
| `xquad` | 12 | `xquad/xquad_ar.yaml` | `xquad/xquad_de.yaml` |
| `xstorycloze` | 12 | `xstorycloze/default_ar.yaml` | `xstorycloze/default_en.yaml` |
| `xwinograd` | 7 | `xwinograd/xwinograd_en.yaml` | `xwinograd/xwinograd_fr.yaml` |
| `zhoblimp` | 119 | `zhoblimp/BA_BEI_subj_drop.yaml` | `zhoblimp/BA_deletion.yaml` |

## Open PR Benchmark Additions Missing Locally

Full CSV: [`lm_eval_missing_open_pr_families_2026-03-23.csv`](./lm_eval_missing_open_pr_families_2026-03-23.csv)

### Novel Families Not Yet On Main

| Family | PR | Draft | New task files | Title |
| --- | ---: | --- | ---: | --- |
| `3LM` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `afridoc_mt` | [#3506](https://github.com/EleutherAI/lm-evaluation-harness/pull/3506) | no | 20 | Hineni |
| `afrihate` | [#3506](https://github.com/EleutherAI/lm-evaluation-harness/pull/3506) | no | 20 | Hineni |
| `ai2d` | [#2542](https://github.com/EleutherAI/lm-evaluation-harness/pull/2542) | yes | 2 | [MM] Ai2d |
| `aibe` | [#2712](https://github.com/EleutherAI/lm-evaluation-harness/pull/2712) | no | 2 | Add AIBE task and utilities |
| `arastem` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `arc_vi` | [#1123](https://github.com/EleutherAI/lm-evaluation-harness/pull/1123) | yes | 9 | add all vlsp |
| `based_drop` | [#3507](https://github.com/EleutherAI/lm-evaluation-harness/pull/3507) | no | 13 | [TASKS] add tasks from GDN paper |
| `based_nq` | [#3507](https://github.com/EleutherAI/lm-evaluation-harness/pull/3507) | no | 13 | [TASKS] add tasks from GDN paper |
| `based_triviaqa` | [#3507](https://github.com/EleutherAI/lm-evaluation-harness/pull/3507) | no | 13 | [TASKS] add tasks from GDN paper |
| `biology_ds` | [#2486](https://github.com/EleutherAI/lm-evaluation-harness/pull/2486) | no | 2 | Biology ds |
| `bluebench` | [#2369](https://github.com/EleutherAI/lm-evaluation-harness/pull/2369) | no | 72 | Add the BlueBench benchmark |
| `boxes` | [#1557](https://github.com/EleutherAI/lm-evaluation-harness/pull/1557) | no | 2 | Adding new task: Boxes |
| `casehold` | [#2570](https://github.com/EleutherAI/lm-evaluation-harness/pull/2570) | no | 2 | Added caseHOLD task |
| `caselawqa` | [#2739](https://github.com/EleutherAI/lm-evaluation-harness/pull/2739) | no | 9 | New benchmark: CaselawQA |
| `countdown` | [#3384](https://github.com/EleutherAI/lm-evaluation-harness/pull/3384) | no | 2 | [feat] Add Countdown Task |
| `coverbench` | [#2207](https://github.com/EleutherAI/lm-evaluation-harness/pull/2207) | no | 2 | CoverBench |
| `CrowsPairs` | [#2488](https://github.com/EleutherAI/lm-evaluation-harness/pull/2488) | no | 1951 | Yaml crowspairs tasks |
| `cuisine` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `dharma2` | [#1753](https://github.com/EleutherAI/lm-evaluation-harness/pull/1753) | no | 1 | Create task `dharma2` - a small (300 qs) & wide (many topics) dataset |
| `dialect_identification` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `dream_interpretation` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `dynamic_ifeval` | [#3149](https://github.com/EleutherAI/lm-evaluation-harness/pull/3149) | no | 10 | Add task dynamic_ifeval |
| `e3c_v3_ner` | [#2812](https://github.com/EleutherAI/lm-evaluation-harness/pull/2812) | no | 16 | E3 c v3 name entity recognition |
| `e3c_v3_re` | [#2806](https://github.com/EleutherAI/lm-evaluation-harness/pull/2806) | no | 8 | Add new task named e3c_v3_re |
| `e3c_v3_re` | [#2812](https://github.com/EleutherAI/lm-evaluation-harness/pull/2812) | no | 16 | E3 c v3 name entity recognition |
| `fake_video` | [#3049](https://github.com/EleutherAI/lm-evaluation-harness/pull/3049) | no | 2 | add video modality and video demo task |
| `finance` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `financial_mmlu_ko` | [#2699](https://github.com/EleutherAI/lm-evaluation-harness/pull/2699) | no | 2 | Add Task (Financial mmlu ko) |
| `flores200` | [#1706](https://github.com/EleutherAI/lm-evaluation-harness/pull/1706) | no | 41618 | Implementing Flores 200 translation evaluation benchmark across 200 languages |
| `flores200` | [#3534](https://github.com/EleutherAI/lm-evaluation-harness/pull/3534) | no | 3 | Adds llm-as-a-judge support via new metric |
| `fpb` | [#1815](https://github.com/EleutherAI/lm-evaluation-harness/pull/1815) | no | 2 | Financial PhraseBank (FPB) Eval Metric |
| `greekmmlu` | [#3581](https://github.com/EleutherAI/lm-evaluation-harness/pull/3581) | no | 52 | add GreekMMLU (official native-sourced benchmark) task configuration |
| `gsm8k_symbolic` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `gsm_symbolic` | [#3354](https://github.com/EleutherAI/lm-evaluation-harness/pull/3354) | no | 3 | Add gsm_symbolic and gsm_symbolic_cot tasks |
| `hellaswag_vi` | [#1123](https://github.com/EleutherAI/lm-evaluation-harness/pull/1123) | yes | 9 | add all vlsp |
| `hu_collocation_lambada ` | [#3043](https://github.com/EleutherAI/lm-evaluation-harness/pull/3043) | no | 1 | first commit of Hungarian generative benchmark |
| `ifbench` | [#3642](https://github.com/EleutherAI/lm-evaluation-harness/pull/3642) | no | 12 | [Task] IFBench |
| `ifeval_pt` | [#3622](https://github.com/EleutherAI/lm-evaluation-harness/pull/3622) | no | 5 | Add ifeval_pt to harness |
| `infinitebench` | [#3256](https://github.com/EleutherAI/lm-evaluation-harness/pull/3256) | no | 68 | Add long-context evaluation benchmarks (LongBench v2, Babilong, InfiniteBench, Phonebook) |
| `kazmmlu` | [#3037](https://github.com/EleutherAI/lm-evaluation-harness/pull/3037) | no | 40 | Added KazMMLU Task |
| `klokan-qa` | [#1657](https://github.com/EleutherAI/lm-evaluation-harness/pull/1657) | no | 1 | Klokan-qa task |
| `kmmlu_pro` | [#3198](https://github.com/EleutherAI/lm-evaluation-harness/pull/3198) | no | 8 | Add new task: kmmlu_pro, kmmlu_redux |
| `kmmlu_redux` | [#3198](https://github.com/EleutherAI/lm-evaluation-harness/pull/3198) | no | 8 | Add new task: kmmlu_pro, kmmlu_redux |
| `ko_commongen_v2` | [#2208](https://github.com/EleutherAI/lm-evaluation-harness/pull/2208) | no | 8 |  Add KoCommonGEN v2 benchmark |
| `kobalt` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `legalbench` | [#1878](https://github.com/EleutherAI/lm-evaluation-harness/pull/1878) | no | 5 | Add LegalBench tasks |
| `longbench_v2` | [#3256](https://github.com/EleutherAI/lm-evaluation-harness/pull/3256) | no | 68 | Add long-context evaluation benchmarks (LongBench v2, Babilong, InfiniteBench, Phonebook) |
| `longproc` | [#3544](https://github.com/EleutherAI/lm-evaluation-harness/pull/3544) | no | 25 | feat(tasks): add LongProc benchmark (6 task types, 16 configs) |
| `m_ifeval` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `masakhanews` | [#3506](https://github.com/EleutherAI/lm-evaluation-harness/pull/3506) | no | 20 | Hineni |
| `math500` | [#2556](https://github.com/EleutherAI/lm-evaluation-harness/pull/2556) | no | 28 | add llama3 tasks |
| `math_500` | [#3381](https://github.com/EleutherAI/lm-evaluation-harness/pull/3381) | no | 5 | Math 500 |
| `medical_specialities` | [#2113](https://github.com/EleutherAI/lm-evaluation-harness/pull/2113) | no | 36 | Medical specialities |
| `milu` | [#2482](https://github.com/EleutherAI/lm-evaluation-harness/pull/2482) | no | 13 | MILU dataset from AI4Bharat for Indic LLM eval |
| `mmlu_cf` | [#3542](https://github.com/EleutherAI/lm-evaluation-harness/pull/3542) | no | 19 | feat(task): add MMLU-CF contamination-free benchmark |
| `mmlu_ru` | [#2378](https://github.com/EleutherAI/lm-evaluation-harness/pull/2378) | no | 413 | add Russian mmlu |
| `mmlu_vi` | [#1123](https://github.com/EleutherAI/lm-evaluation-harness/pull/1123) | yes | 9 | add all vlsp |
| `mmmlu` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `nq` | [#1649](https://github.com/EleutherAI/lm-evaluation-harness/pull/1649) | yes | 13 | Add natural questions in a the closedbook setup. |
| `numeric_bench` | [#2835](https://github.com/EleutherAI/lm-evaluation-harness/pull/2835) | no | 13 | feat: Numeric bench |
| `openai_mmmlu` | [#2312](https://github.com/EleutherAI/lm-evaluation-harness/pull/2312) | no | 873 | mmlu translated professionally by OpenAI |
| `opengptx` | [#2488](https://github.com/EleutherAI/lm-evaluation-harness/pull/2488) | no | 1951 | Yaml crowspairs tasks |
| `permutation_benchmark` | [#3157](https://github.com/EleutherAI/lm-evaluation-harness/pull/3157) | no | 99 | Feat/add permutation benchmark/task to lm-evaluation-harness |
| `persianmmlu` | [#1979](https://github.com/EleutherAI/lm-evaluation-harness/pull/1979) | no | 23 | add persianmmlu benchmark for assessing Persian Language understanding |
| `phonebook` | [#3256](https://github.com/EleutherAI/lm-evaluation-harness/pull/3256) | no | 68 | Add long-context evaluation benchmarks (LongBench v2, Babilong, InfiniteBench, Phonebook) |
| `physics_gre` | [#1655](https://github.com/EleutherAI/lm-evaluation-harness/pull/1655) | no | 5 | Physics GRE task added |
| `poetry_analysis` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `polymath` | [#3623](https://github.com/EleutherAI/lm-evaluation-harness/pull/3623) | no | 97 | Add evaluation task for PolyMath (multilingual math reasoning benchmark) |
| `proof-pile` | [#2132](https://github.com/EleutherAI/lm-evaluation-harness/pull/2132) | no | 1 | Introduce perplexity per token in loglikelihood_rolling |
| `putnam_axiom` | [#2946](https://github.com/EleutherAI/lm-evaluation-harness/pull/2946) | no | 5 | Final putnam axiom bm |
| `redlite` | [#2020](https://github.com/EleutherAI/lm-evaluation-harness/pull/2020) | no | 8 | Add Redlite tasks for safety benchmarking |
| `religious_qa` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `selfcheckgpt` | [#1080](https://github.com/EleutherAI/lm-evaluation-harness/pull/1080) | no | 1 | Add Selfcheckgpt evaluation to tasks |
| `sib200` | [#1705](https://github.com/EleutherAI/lm-evaluation-harness/pull/1705) | no | 206 | Implement Sib200 evaluation benchmark - text classification in 200 languages  |
| `sparc` | [#3262](https://github.com/EleutherAI/lm-evaluation-harness/pull/3262) | no | 2 | Adding SPaRC to lm eval harness |
| `summarization` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `swahili_bench` | [#2031](https://github.com/EleutherAI/lm-evaluation-harness/pull/2031) | no | 1 | swahili_ARC_Challenge |
| `triviaqa_closedbook` | [#1649](https://github.com/EleutherAI/lm-evaluation-harness/pull/1649) | yes | 13 | Add natural questions in a the closedbook setup. |
| `trump_tts` | [#2957](https://github.com/EleutherAI/lm-evaluation-harness/pull/2957) | no | 2 | add ultravox models support for audio tasks |
| `truthfulqa_vi` | [#1123](https://github.com/EleutherAI/lm-evaluation-harness/pull/1123) | yes | 9 | add all vlsp |
| `uae_knowledge` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `uncheatable_eval` | [#3442](https://github.com/EleutherAI/lm-evaluation-harness/pull/3442) | no | 18 | Add Uncheatable Eval |
| `wmt24pp` | [#3480](https://github.com/EleutherAI/lm-evaluation-harness/pull/3480) | no | 58 | Implement new translation tasks for google WMT24++ datasets |
| `zebralogic` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |

### Existing Main-Branch Families With New Open-PR Configs

| Family | PR | Draft | New task files | Title |
| --- | ---: | --- | ---: | --- |
| `aclue` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `aexams` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `afrimgsm` | [#2106](https://github.com/EleutherAI/lm-evaluation-harness/pull/2106) | no | 5 | IrokoBench edit |
| `afrimmlu` | [#2106](https://github.com/EleutherAI/lm-evaluation-harness/pull/2106) | no | 5 | IrokoBench edit |
| `afrixnli` | [#2106](https://github.com/EleutherAI/lm-evaluation-harness/pull/2106) | no | 5 | IrokoBench edit |
| `agieval` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `aime` | [#3510](https://github.com/EleutherAI/lm-evaluation-harness/pull/3510) | no | 3 | Added pass@k and avg@k metrics to AIME benchmark |
| `aime` | [#3603](https://github.com/EleutherAI/lm-evaluation-harness/pull/3603) | no | 1 | feat: Add AIME 2026 task configuration and documentation. |
| `alghafa` | [#1946](https://github.com/EleutherAI/lm-evaluation-harness/pull/1946) | no | 8 | Alghafa benchmark |
| `arabicmmlu` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `aradice` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `arithmetic` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `babilong` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `babilong` | [#3256](https://github.com/EleutherAI/lm-evaluation-harness/pull/3256) | no | 68 | Add long-context evaluation benchmarks (LongBench v2, Babilong, InfiniteBench, Phonebook) |
| `bbh` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `bbh` | [#1481](https://github.com/EleutherAI/lm-evaluation-harness/pull/1481) | yes | 2 | Transfer zero-shot BBH parsing improvements to few-shot BBH |
| `bbh` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `belebele` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `bigbench` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `bigbench` | [#3556](https://github.com/EleutherAI/lm-evaluation-harness/pull/3556) | no | 1 | fix(bigbench): add group for task discovery (bigbench) |
| `blimp` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `blimp` | [#2951](https://github.com/EleutherAI/lm-evaluation-harness/pull/2951) | no | 47 | Added RuBLiMP, a Russian benchmark of linguistic minimal pairs |
| `ceval` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `click` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `cmmlu` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `csatqa` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `darija_bench` | [#3031](https://github.com/EleutherAI/lm-evaluation-harness/pull/3031) | no | 3 | Tarjma bench |
| `fld` | [#2022](https://github.com/EleutherAI/lm-evaluation-harness/pull/2022) | no | 10 | [add] multiple-choice-question versions of fld benchmark |
| `gpqa` | [#3547](https://github.com/EleutherAI/lm-evaluation-harness/pull/3547) | no | 4 | feat (tasks): add AAII GPQA Diamond tasks, extraction regex, and reasoning/non-reasoning wrappers |
| `haerae` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `ifeval` | [#3240](https://github.com/EleutherAI/lm-evaluation-harness/pull/3240) | no | 1 | Trim thinking content from model output in IFEval |
| `kormedmcqa` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `leaderboard` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `llama3` | [#2556](https://github.com/EleutherAI/lm-evaluation-harness/pull/2556) | no | 28 | add llama3 tasks |
| `llama3` | [#3593](https://github.com/EleutherAI/lm-evaluation-harness/pull/3593) | no | 23 | Add llama 3 eval configs for GPQA, IFEval, MATH, MGSM, and HellaSwag |
| `mathqa` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `mmlu-redux` | [#3250](https://github.com/EleutherAI/lm-evaluation-harness/pull/3250) | no | 201 | Main |
| `noreval` | [#3572](https://github.com/EleutherAI/lm-evaluation-harness/pull/3572) | no | 52 | New NorEval tasks |
| `okapi` | [#1860](https://github.com/EleutherAI/lm-evaluation-harness/pull/1860) | no | 2 | mmlu-pro for the  Italian language |
| `okapi` | [#2488](https://github.com/EleutherAI/lm-evaluation-harness/pull/2488) | no | 1951 | Yaml crowspairs tasks |
| `paws-x` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `sciq` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `siqa` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `translation` | [#3241](https://github.com/EleutherAI/lm-evaluation-harness/pull/3241) | no | 127 | Adding 3LM to lm eval harness |
| `truthfulqa` | [#925](https://github.com/EleutherAI/lm-evaluation-harness/pull/925) | yes | 1949 | Alternative Worlds Prompts for Various Tasks and Benchmarks |
| `wmdp` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `xcopa` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `xnli` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `xnli` | [#3553](https://github.com/EleutherAI/lm-evaluation-harness/pull/3553) | no | 1 | feat(tasks): add Persian XNLI evaluation task |
| `xnli_eu` | [#2869](https://github.com/EleutherAI/lm-evaluation-harness/pull/2869) | yes | 1 | Fix formatting issues in XNLI tasks in Basque, Catalan, Galician and Spanish |
| `xquad` | [#3609](https://github.com/EleutherAI/lm-evaluation-harness/pull/3609) | no | 1 | feat(xquad): add group YAML, fix Arabic prompt, clean imports |
| `xstorycloze` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |
| `xwinograd` | [#1922](https://github.com/EleutherAI/lm-evaluation-harness/pull/1922) | yes | 105 | Multiprompt |

## Notes

- Excluded from the main-family count as meta/config wrappers rather than standalone benchmarks: `benchmarks`, `leaderboard`, `llama3`, `metabench`, `score`, `tinyBenchmarks`.
- Open-PR analysis only counted PRs that add new files under `lm_eval/tasks/`; pure bug-fix PRs without new task files were excluded.
- Some open PRs add multiple families in a single PR, so the PR CSV is one row per family, not one row per PR.
- A few PR families are model-specific or modality-specific task packages; they were left in the PR CSV because they still represent benchmark/task surface area missing locally.
