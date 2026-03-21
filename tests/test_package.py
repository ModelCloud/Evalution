import evalution


def test_package_import() -> None:
    assert evalution is not None


def test_package_exports_arc_challenge_suite() -> None:
    assert evalution.ARCChallenge is not None
    assert callable(evalution.arc_challenge)
    assert evalution.ARCEasy is not None
    assert callable(evalution.arc_easy)
    assert evalution.BoolQ is not None
    assert callable(evalution.boolq)
    assert evalution.CB is not None
    assert callable(evalution.cb)
    assert evalution.COPA is not None
    assert callable(evalution.copa)
    assert evalution.HellaSwag is not None
    assert callable(evalution.hellaswag)
    assert evalution.MMLU is not None
    assert callable(evalution.mmlu)
    assert evalution.OpenBookQA is not None
    assert callable(evalution.openbookqa)
    assert evalution.PIQA is not None
    assert callable(evalution.piqa)
    assert evalution.QNLI is not None
    assert callable(evalution.qnli)
    assert evalution.RTE is not None
    assert callable(evalution.rte)
    assert evalution.SST2 is not None
    assert callable(evalution.sst2)
    assert evalution.WiC is not None
    assert callable(evalution.wic)
    assert evalution.WinoGrande is not None
    assert callable(evalution.winogrande)
    assert callable(evalution.f1_for_label)
    assert callable(evalution.macro_f1)


def test_package_exports_fluent_runtime_api() -> None:
    assert callable(evalution.engine)
    assert evalution.Transformers is evalution.Transformer
    assert callable(evalution.run_yaml)
    assert callable(evalution.python_from_yaml)


def test_package_exposes_cli_entrypoint() -> None:
    from evalution import cli

    assert callable(cli.main)
