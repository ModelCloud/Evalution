from evalution import __version__, get_version


def test_package_version() -> None:
    assert __version__ == "0.1.0"
    assert get_version() == __version__
