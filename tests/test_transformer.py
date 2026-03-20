from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformer import Transformer, TransformerSession


def test_transformer_defaults_batch_size_to_auto() -> None:
    engine = Transformer()

    assert engine.batch_size == "auto"
    assert engine.paged_attention == "auto"
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["paged_attention"] == "auto"


def test_transformer_session_resolves_auto_batch_size_once_per_suite(monkeypatch) -> None:
    session = TransformerSession(
        config=Transformer(batch_size="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
    )
    requests = [GenerationRequest(prompt="Question: 1 + 1\nAnswer:")]
    stats = {
        "row_count": 1,
        "min_prompt_tokens": 12,
        "avg_prompt_tokens": 12.0,
        "max_prompt_tokens": 12,
        "max_new_tokens": 32,
        "dtype_name": "bfloat16",
        "dtype_bytes": 2,
        "total_vram_gib": 0.0,
        "parameter_count_billions": 1.0,
    }
    calls = {"estimate": 0}

    monkeypatch.setattr(
        TransformerSession,
        "_batch_size_stats",
        lambda self, batch: stats,
    )

    def fake_estimate(self, batch_stats):
        calls["estimate"] += 1
        assert batch_stats is stats
        return 16

    monkeypatch.setattr(
        TransformerSession,
        "_estimate_auto_batch_size",
        fake_estimate,
    )

    assert session.resolve_batch_size(requests) == 16
    assert session.resolve_batch_size(requests) == 16
    assert calls["estimate"] == 1


def test_transformer_session_describes_auto_paged_attention_on_cuda_like_session() -> None:
    session = TransformerSession(
        config=Transformer(attn_implementation="flash_attention_2", paged_attention="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
            generate_batch=lambda *args, **kwargs: {},
            set_attn_implementation=lambda value: None,
        ),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": "flash_attention_2",
        "effective_attn_implementation": "paged|flash_attention_2",
        "paged_attention": True,
        "generation_backend": "continuous_batching",
        "standard_batch_size_cap": None,
    }


def test_transformer_session_prepare_requests_batches_tokenization() -> None:
    class FakePrepareTokenizer:
        def __init__(self) -> None:
            self.encode_calls: list[list[str]] = []

        def __call__(self, prompts, *, add_special_tokens=False, padding=False):
            assert add_special_tokens is False
            assert padding is False
            assert isinstance(prompts, list)
            self.encode_calls.append(list(prompts))
            return {
                "input_ids": [
                    [index + 1, index + 2, index + 3]
                    for index, _prompt in enumerate(prompts)
                ]
            }

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"chat::{messages[0]['content']}"

    prepare_tokenizer = FakePrepareTokenizer()
    session = TransformerSession(
        config=Transformer(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        prepare_tokenizer=prepare_tokenizer,
        input_device=SimpleNamespace(type="cpu"),
    )

    prepared = session.prepare_requests(
        [
            GenerationRequest(prompt="Q: 1 + 1\nA:"),
            GenerationRequest(messages=[{"role": "user", "content": "Q: 2 + 2\nA:"}]),
            GenerationRequest(
                prompt="Q: 3 + 3\nA:",
                rendered_prompt="Q: 3 + 3\nA:",
                input_ids=[9, 9, 9],
            ),
        ]
    )

    assert prepare_tokenizer.encode_calls == [["Q: 1 + 1\nA:", "chat::Q: 2 + 2\nA:"]]
    assert prepared[0].rendered_prompt == "Q: 1 + 1\nA:"
    assert prepared[0].input_ids == [1, 2, 3]
    assert prepared[1].rendered_prompt == "chat::Q: 2 + 2\nA:"
    assert prepared[1].input_ids == [2, 3, 4]
    assert prepared[2].rendered_prompt == "Q: 3 + 3\nA:"
    assert prepared[2].input_ids == [9, 9, 9]


def test_transformer_session_generate_reuses_pretokenized_requests() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            self.pad_calls = 0
            self.encode_calls = 0

        def __call__(self, prompts, **kwargs):
            del prompts, kwargs
            self.encode_calls += 1
            raise AssertionError("pretokenized requests should not be re-encoded")

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            assert return_tensors == "pt"
            assert padding is True
            self.pad_calls += 1
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    tokenizer = FakeTokenizer()
    session = TransformerSession(
        config=Transformer(),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=tokenizer,
        input_device=torch.device("cpu"),
    )

    outputs = session.generate(
        [
            GenerationRequest(
                prompt="Q: 40 + 2\nA:",
                rendered_prompt="Q: 40 + 2\nA:",
                input_ids=[1, 2, 3],
            )
        ],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "The answer is 42."
    assert tokenizer.pad_calls == 1
    assert tokenizer.encode_calls == 0


def test_transformer_session_generate_uses_generate_batch_when_paged_attention_is_enabled() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompts, *, add_special_tokens=False, **kwargs):
            assert add_special_tokens is False
            del kwargs
            if isinstance(prompts, str):
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeGenerateBatchOutput:
        def __init__(self, tokens):
            self.generated_tokens = tokens

    class FakeModel:
        def __init__(self) -> None:
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"
            self.calls: list[dict[str, object]] = []

        def generate_batch(self, inputs, *, generation_config=None, progress_bar=False):
            self.calls.append(
                {
                    "inputs": inputs,
                    "generation_config": generation_config,
                    "progress_bar": progress_bar,
                }
            )
            return {"req_0": FakeGenerateBatchOutput([101, 102, 103])}

        def set_attn_implementation(self, value: str) -> None:
            self.config._attn_implementation = value

    model = FakeModel()
    session = TransformerSession(
        config=Transformer(attn_implementation="flash_attention_2", paged_attention="auto"),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session.generate(
        [GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "The answer is 42."
    assert model.calls
    call = model.calls[0]
    assert call["inputs"] == [[11, 12, 13]]
    assert call["progress_bar"] is False
    assert call["generation_config"].stop_strings == ["Q:"]


def test_transformer_session_serializes_shared_tokenizer_access_between_prepare_and_generate() -> None:
    class BlockingTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            self._state_lock = threading.Lock()
            self._active_calls = 0
            self.overlap_detected = False
            self.prepare_started = threading.Event()
            self.pad_started = threading.Event()
            self.allow_prepare_finish = threading.Event()

        def _enter(self) -> None:
            with self._state_lock:
                self._active_calls += 1
                if self._active_calls > 1:
                    self.overlap_detected = True

        def _exit(self) -> None:
            with self._state_lock:
                self._active_calls -= 1

        def __call__(self, prompts, *, add_special_tokens=False, padding=False, **kwargs):
            del kwargs
            assert add_special_tokens is False
            assert padding is False
            assert isinstance(prompts, list)
            self._enter()
            try:
                self.prepare_started.set()
                assert self.allow_prepare_finish.wait(timeout=1.0)
                return {"input_ids": [[1, 2, 3] for _ in prompts]}
            finally:
                self._exit()

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            del encoded_inputs
            assert return_tensors == "pt"
            assert padding is True
            self._enter()
            try:
                self.pad_started.set()
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            finally:
                self._exit()

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    tokenizer = BlockingTokenizer()
    session = TransformerSession(
        config=Transformer(),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=tokenizer,
        prepare_tokenizer=None,
        input_device=torch.device("cpu"),
    )

    prepare_errors: list[BaseException] = []
    generate_errors: list[BaseException] = []

    def run_prepare() -> None:
        try:
            session.prepare_requests([GenerationRequest(prompt="Q: 1 + 1\nA:")])
        except BaseException as exc:  # pragma: no cover - asserted below
            prepare_errors.append(exc)

    def run_generate() -> None:
        try:
            session.generate(
                [
                    GenerationRequest(
                        prompt="Q: 40 + 2\nA:",
                        rendered_prompt="Q: 40 + 2\nA:",
                        input_ids=[1, 2, 3],
                    )
                ],
                batch_size=1,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            generate_errors.append(exc)

    prepare_thread = threading.Thread(target=run_prepare)
    generate_thread = threading.Thread(target=run_generate)

    prepare_thread.start()
    assert tokenizer.prepare_started.wait(timeout=1.0)
    generate_thread.start()

    assert not tokenizer.pad_started.wait(timeout=0.1)

    tokenizer.allow_prepare_finish.set()
    prepare_thread.join(timeout=1.0)
    generate_thread.join(timeout=1.0)

    assert not prepare_errors
    assert not generate_errors
    assert tokenizer.overlap_detected is False


def test_transformer_session_serializes_generate_calls() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            del encoded_inputs
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class BlockingModel:
        def __init__(self) -> None:
            self._state_lock = threading.Lock()
            self._active_calls = 0
            self._call_count = 0
            self.overlap_detected = False
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.allow_first_finish = threading.Event()

        def generate(self, **kwargs):
            del kwargs
            with self._state_lock:
                self._active_calls += 1
                self._call_count += 1
                call_number = self._call_count
                if self._active_calls > 1:
                    self.overlap_detected = True
            try:
                if call_number == 1:
                    self.first_started.set()
                    assert self.allow_first_finish.wait(timeout=1.0)
                else:
                    self.second_started.set()
                return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            finally:
                with self._state_lock:
                    self._active_calls -= 1

    model = BlockingModel()
    session = TransformerSession(
        config=Transformer(),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )
    requests = [
        GenerationRequest(
            prompt="Q: 40 + 2\nA:",
            rendered_prompt="Q: 40 + 2\nA:",
            input_ids=[1, 2, 3],
        )
    ]
    errors: list[BaseException] = []

    def run_generate() -> None:
        try:
            session.generate(requests, batch_size=1)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=run_generate)
    second_thread = threading.Thread(target=run_generate)

    first_thread.start()
    assert model.first_started.wait(timeout=1.0)

    second_thread.start()
    assert not model.second_started.wait(timeout=0.1)

    model.allow_first_finish.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not errors
    assert model.overlap_detected is False


def test_transformer_session_falls_back_to_standard_generate_when_paged_generation_fails(monkeypatch) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        set_attn_implementation=lambda value: None,
    )
    session = TransformerSession(
        config=Transformer(attn_implementation="sdpa", paged_attention=True),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="sdpa",
        effective_attn_implementation="paged|sdpa",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )
    requests = [GenerationRequest(prompt=f"Q: {index}\nA:") for index in range(4)]
    calls: dict[str, object] = {}

    def fake_generate_paged(self, batch, *, batch_size):
        del self, batch, batch_size
        raise RuntimeError("paged failure")

    def fake_generate_standard(self, batch, *, batch_size):
        calls["batch_size"] = batch_size
        return [
            GenerationOutput(
                prompt=request.prompt or "",
                text="The answer is 42.",
            )
            for request in batch
        ]

    monkeypatch.setattr(TransformerSession, "_generate_paged", fake_generate_paged)
    monkeypatch.setattr(TransformerSession, "_generate_standard", fake_generate_standard)

    outputs = session.generate(requests, batch_size=4)

    assert len(outputs) == 4
    assert calls["batch_size"] == 2
    assert session.paged_attention_enabled is False
    assert session.generation_backend == "generate"
    assert session.effective_attn_implementation == "sdpa"
    assert session.standard_batch_size_cap == 2
