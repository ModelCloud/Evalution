# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import replace

import tests.models_support as model_support

pytestmark = model_support.LLAMA3_2_TRANSFORMERS_TEST_MARKS


def _override_batch_size(
    spec: model_support.SuiteSpec,
    *,
    batch_size: int,
) -> model_support.SuiteSpec:
    def suite_factory(spec: model_support.SuiteSpec = spec):
        suite = spec.suite_factory()
        suite.batch_size = batch_size
        return suite

    return replace(spec, suite_factory=suite_factory)


def test_llama3_2_transformers_bangla_full_model_eval(capsys):
    specs = [
        _override_batch_size(model_support.SUITE_SPECS[suite_key], batch_size=4)
        for suite_key in model_support.BANGLA_TASKS
    ]
    result, test_results = model_support.run_llama3_2_suites(
        capsys,
        [spec.suite_factory() for spec in specs],
    )
    for test_result, spec in zip(test_results, specs, strict=True):
        model_support._assert_suite_matches_spec(test_result, spec)
    serialized = result.to_dict()
    assert len(serialized["tests"]) == len(test_results)
    for serialized_test, test_result in zip(serialized["tests"], test_results, strict=True):
        assert serialized_test["name"] == test_result.name
        assert len(serialized_test["samples"]) == len(test_result.samples)
        if test_result.samples:
            assert serialized_test["samples"][0]["prediction"]
