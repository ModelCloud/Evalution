# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubertcloud

from tests.models_support import run_suite_spec


def test_llama3_2_transformers_ifeval_full_model_eval(capsys):
    run_suite_spec(capsys, "ifeval")
