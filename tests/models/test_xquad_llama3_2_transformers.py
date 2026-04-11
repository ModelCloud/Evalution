# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, XQUAD_TASKS, run_suite_spec

# Keep shared test fixtures and expectations explicit at module scope.
pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


@pytest.mark.parametrize("suite_key", XQUAD_TASKS)
def test_llama3_2_transformers_xquad_full_model_eval(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> None:
    """Verify llama3 2 transformers XQuAD full model eval."""
    run_suite_spec(capsys, suite_key)
