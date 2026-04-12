# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_llama3_2_transformers_multirc_full_model_eval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the MultiRC full-model regression on the shared Llama 3.2 baseline harness."""

    run_suite_spec(capsys, "multirc")
