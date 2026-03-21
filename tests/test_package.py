import evalution


def test_package_import() -> None:
    assert evalution is not None


def test_package_exports_arc_challenge_suite() -> None:
    assert evalution.ARCChallenge is not None
    assert callable(evalution.arc_challenge)
