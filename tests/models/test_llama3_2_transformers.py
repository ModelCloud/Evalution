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
                batch_size="auto",
            ),
            tests=[
                evalution.gsm8k_platinum(
                    variant="cot_llama",
                    apply_chat_template=True,
                    max_new_tokens=96,
                )
            ],
        )

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "sdpa"
    assert result.engine["batch_size"] == "auto"
    assert len(result.tests) == 1

    test_result = result.tests[0]
    assert test_result.name == "gsm8k_platinum_cot_llama"
    assert test_result.metadata["variant"] == "cot_llama"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["fewshot_as_multiturn"] is True
    assert test_result.metadata["num_fewshot"] == 8
    assert len(test_result.samples) > 1000
    assert set(test_result.metrics) == {
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    }
    flexible_score = test_result.metrics["exact_match,flexible-extract"]
    strict_score = test_result.metrics["exact_match,strict-match"]
    assert isinstance(flexible_score, float)
    assert isinstance(strict_score, float)
    assert 0.0 <= flexible_score <= 1.0
    assert 0.0 <= strict_score <= 1.0
    assert flexible_score >= 0.20
    assert strict_score >= 0.10

    invalid_predictions = 0
    exact_matches = 0
    for index, sample in enumerate(test_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
        assert "Given the following problem, reason and give a final answer to the problem." in sample.prompt
        assert "Problem:" in sample.prompt
        assert "The final answer is" in sample.prompt
        assert set(sample.extracted) == {"strict-match", "flexible-extract"}
        assert set(sample.scores) == {
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        }
        if sample.extracted["flexible-extract"] == "[invalid]":
            invalid_predictions += 1
        if sample.scores["exact_match,flexible-extract"] == 1.0:
            exact_matches += 1

    assert exact_matches > 0
    assert invalid_predictions / len(test_result.samples) < 0.20

    serialized = result.to_dict()
    assert serialized["tests"][0]["name"] == "gsm8k_platinum_cot_llama"
    assert len(serialized["tests"][0]["samples"]) == len(test_result.samples)
    assert serialized["tests"][0]["samples"][0]["prediction"]
