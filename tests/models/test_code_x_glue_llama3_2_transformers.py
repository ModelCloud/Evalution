# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from tests.models_support import CODE_X_GLUE_TASKS, LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_specs

pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_llama3_2_transformers_code_x_glue_full_model_eval(capsys):
    run_suite_specs(capsys, CODE_X_GLUE_TASKS)
