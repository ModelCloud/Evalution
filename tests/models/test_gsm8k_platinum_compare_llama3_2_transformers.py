# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from tests.models_support import (
    LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS,
    run_compare_suite_spec,
)

pytestmark = LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS


def test_llama3_2_transformers_gsm8k_platinum_full_model_compare_eval(capsys):
    run_compare_suite_spec(capsys, "gsm8k_platinum")
