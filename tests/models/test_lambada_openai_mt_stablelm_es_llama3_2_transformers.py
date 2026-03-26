# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from tests.models_support import run_suite_spec


def test_llama3_2_transformers_lambada_openai_mt_stablelm_es_full_model_eval(capsys):
    run_suite_spec(capsys, "lambada_openai_mt_stablelm_es")
