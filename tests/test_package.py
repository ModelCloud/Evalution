import evalution


def test_package_import() -> None:
    assert evalution is not None


def test_package_exports_arc_challenge_suite() -> None:
    assert evalution.ARCChallenge is not None
    assert callable(evalution.arc_challenge)
    assert evalution.BoolQ is not None
    assert callable(evalution.boolq)
    assert evalution.HellaSwag is not None
    assert callable(evalution.hellaswag)
    assert evalution.PIQA is not None
    assert callable(evalution.piqa)


def test_package_exports_fluent_runtime_api() -> None:
    assert callable(evalution.engine)
    assert evalution.Transformers is evalution.Transformer
    assert callable(evalution.run_yaml)
    assert callable(evalution.python_from_yaml)


def test_package_exposes_cli_entrypoint() -> None:
    from evalution import cli

    assert callable(cli.main)
