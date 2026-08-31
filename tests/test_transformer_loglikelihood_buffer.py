from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from evalution.engines.transformers_common import BaseTransformerSession, _ScoringChunk


def test_score_chunks_batches_cuda_scalar_reductions() -> None:
    class Model:
        def __call__(self, input_ids, **_kwargs):
            batch, length = input_ids.shape
            logits = torch.full((batch, length, 16), -4.0)
            for row in range(batch):
                for position in range(length - 1):
                    logits[row, position, int(input_ids[row, position + 1])] = 4.0
            return SimpleNamespace(logits=logits)

    session = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        input_device=torch.device("cpu"),
        model=Model(),
        _scoring_attention_context=lambda: nullcontext(),
    )
    chunks = [
        _ScoringChunk(
            request_index=index,
            input_ids=[index + 1, index + 2],
            score_start=1,
            score_count=1,
            metadata={"_evalution_disable_loglikelihood_chunk_progress": True},
        )
        for index in range(3)
    ]

    outputs = BaseTransformerSession._score_chunks(session, chunks, batch_size=2)

    assert len(outputs) == len(chunks)
    assert all(output.is_greedy for output in outputs)
    assert all(output.token_count == 1 for output in outputs)
    assert outputs[0].logprob == outputs[1].logprob == outputs[2].logprob
