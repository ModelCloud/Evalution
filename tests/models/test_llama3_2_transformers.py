from __future__ import annotations

from pathlib import Path

import pytest
import torch

import evalution

LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.skipif(not LLAMA3_2_1B_INSTRUCT.exists(), reason="local Llama 3.2 1B Instruct weights are not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the llama 3.2 integration test")
def test_llama3_2_transformers_gsm8k_platinum_cot() -> None:
    result = evalution.run(
        model=evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)),
        engine=evalution.Transformer(
            dtype="bfloat16",
            attn_implementation="sdpa",
            device="cuda:0",
            batch_size=1,
        ),
        tests=[
            evalution.gsm8k_platinum(
                variant="cot",
                apply_chat_template=True,
                limit=1,
                max_new_tokens=96,
            )
        ],
    )

    test_result = result.tests[0]
    assert test_result.name == "gsm8k_platinum_cot"
    assert len(test_result.samples) == 1
    assert "exact_match,strict-match" in test_result.metrics
    assert "exact_match,flexible-extract" in test_result.metrics
    assert test_result.samples[0].prediction != ""
