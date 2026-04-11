# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

# Reuse the shared Llama 3.2 Transformers integration marks for this one-suite regression.
pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_llama3_2_transformers_ifeval_pt_full_model_eval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify llama3 2 transformers IFEval pt full model eval."""
    run_suite_spec(capsys, "ifeval_pt")
