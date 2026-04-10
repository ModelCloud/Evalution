# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest
import torch

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

_LONG_BENCH2_A100_TASKS = ("longbench2_legal_single",)
_MIN_LONG_BENCH2_VRAM_BYTES = 90 * 1024**3

pytestmark = [
    *LLAMA3_2_TRANSFORMERS_TEST_MARKS,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < _MIN_LONG_BENCH2_VRAM_BYTES,
        reason="the LongBench2 legal_single regression requires a 96 GB-class CUDA device",
    ),
]


@pytest.mark.parametrize("suite_key", _LONG_BENCH2_A100_TASKS, ids=_LONG_BENCH2_A100_TASKS)
def test_llama3_2_transformers_longbench2_full_model_eval(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> None:
    run_suite_spec(capsys, suite_key)
