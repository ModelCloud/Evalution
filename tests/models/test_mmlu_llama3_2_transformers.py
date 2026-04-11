# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

# Keep shared test fixtures and expectations explicit at module scope.
pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_llama3_2_transformers_mmlu_full_model_eval(capsys):
    """Verify llama3 2 transformers MMLU full model eval."""
    run_suite_spec(capsys, "mmlu_all")


def test_llama3_2_transformers_mmlu_stem_full_model_eval(capsys):
    """Verify llama3 2 transformers MMLU stem full model eval."""
    run_suite_spec(capsys, "mmlu_stem")
