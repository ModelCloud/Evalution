# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json

from datasets import Dataset

from evalution import cli
from evalution import yaml as evalution_yaml
from evalution.engines.base import BaseEngine, BaseInferenceSession


class FakeEngine(BaseEngine):
    def build(self, model):
        del model
        return FakeSession()

    def to_dict(self):
        return {"name": "fake"}


class FakeSession(BaseInferenceSession):
    def generate(self, requests, *, batch_size=None):
        del batch_size
        return [
            type("Output", (), {"prompt": request.prompt, "text": "The answer is 42."})()
            for request in requests
        ]

    def describe_execution(self):
        return {"generation_backend": "fake"}

    def loglikelihood(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        del requests, batch_size
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size=None):
        request_items = list(requests)
        outputs = self.generate([request for _, request in request_items], batch_size=batch_size)
        for (item_id, _request), output in zip(request_items, outputs, strict=True):
            yield item_id, output

    def gc(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_cli_run_executes_yaml_and_prints_json(monkeypatch, tmp_path, capsys) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )
    gsm8k_module = __import__("evalution.benchmarks.gsm8k_platinum", fromlist=["load_dataset"])
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setitem(evalution_yaml._ENGINE_FACTORIES, "fake", FakeEngine)

    spec = tmp_path / "evalution.yaml"
    spec.write_text(
        """
engine:
  type: fake
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main([str(spec)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["model"]["path"] == "/tmp/model"
    assert payload["engine"]["name"] == "fake"
    assert payload["tests"][0]["name"] == "gsm8k_platinum_cot"


def test_cli_emit_python_prints_equivalent_script(tmp_path, capsys) -> None:
    spec = tmp_path / "evalution.yaml"
    spec.write_text(
        """
engine:
  type: transformers
model:
  path: /tmp/model
tests:
  - type: arc_challenge
    max_rows: 8
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["emit-python", str(spec)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "import evalution as eval" in captured.out
    assert "import evalution.benchmarks as benchmarks" in captured.out
    assert "import evalution.engines as engines" in captured.out
    assert "eval(engines.Transformers(" in captured.out
    assert ".run(benchmarks.arc_challenge(" in captured.out


def test_cli_run_can_write_json_to_output_file(monkeypatch, tmp_path) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )
    gsm8k_module = __import__("evalution.benchmarks.gsm8k_platinum", fromlist=["load_dataset"])
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setitem(evalution_yaml._ENGINE_FACTORIES, "fake", FakeEngine)

    spec = tmp_path / "evalution.yaml"
    spec.write_text(
        """
engine:
  type: fake
model:
  path: /tmp/model
tests:
  - type: gsm8k_platinum
    max_rows: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    exit_code = cli.main(["run", str(spec), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["tests"][0]["name"] == "gsm8k_platinum_cot"
