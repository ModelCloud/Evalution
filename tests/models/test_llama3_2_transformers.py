from __future__ import annotations

from pathlib import Path

import pytest
import torch

import evalution

LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.skipif(
    not LLAMA3_2_1B_INSTRUCT.exists(),
    reason="local Llama 3.2 1B Instruct weights are not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the llama 3.2 integration test",
)
def test_llama3_2_transformers_full_model_eval_run(capsys: pytest.CaptureFixture[str]) -> None:
    # Disable pytest capture for the actual eval so LogBar output stays visible.
    with capsys.disabled():
        result = evalution.run(
            model=evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)),
            engine=evalution.Transformer(
                dtype="bfloat16",
                attn_implementation="sdpa",
                device="cuda:0",
                batch_size=2,
            ),
            tests=[
                evalution.gsm8k_platinum(
                    variant="cot",
                    apply_chat_template=True,
                    limit=2,
                    max_new_tokens=96,
                )
            ],
        )

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "sdpa"
    assert result.engine["batch_size"] == 2
    assert len(result.tests) == 1

    test_result = result.tests[0]
    assert test_result.name == "gsm8k_platinum_cot"
    assert test_result.metadata["variant"] == "cot"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["fewshot_as_multiturn"] is True
    assert test_result.metadata["num_fewshot"] == 8
    assert len(test_result.samples) == 2
    assert set(test_result.metrics) == {
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    }

    for metric_value in test_result.metrics.values():
        assert isinstance(metric_value, float)
        assert 0.0 <= metric_value <= 1.0

    for index, sample in enumerate(test_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
        assert "Q:" in sample.prompt
        assert "A:" in sample.prompt
        assert set(sample.extracted) == {"strict-match", "flexible-extract"}
        assert set(sample.scores) == {
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        }
        for score in sample.scores.values():
            assert score in {0.0, 1.0}

    serialized = result.to_dict()
    assert serialized["tests"][0]["name"] == "gsm8k_platinum_cot"
    assert len(serialized["tests"][0]["samples"]) == 2
    assert serialized["tests"][0]["samples"][0]["prediction"]
