# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, WINOGENDER_TASKS, run_suite_specs

pytestmark = LLAMA3_2_TRANSFORMERS_TEST_MARKS


def test_winogender_llama3_2_transformers(capsys: pytest.CaptureFixture[str]) -> None:
    run_suite_specs(capsys, WINOGENDER_TASKS)
