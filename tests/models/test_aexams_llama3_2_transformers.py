# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from tests.models_support import AEXAMS_TASKS, LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_specs

# Keep shared test fixtures and expectations explicit at module scope.
pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_aexams_llama3_2_transformers(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify aexams llama3 2 transformers."""
    run_suite_specs(capsys, AEXAMS_TASKS)
