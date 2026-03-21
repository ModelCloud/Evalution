from __future__ import annotations

from collections.abc import Sequence

from evalution.config import Model, coerce_model
from evalution.logbar import get_logger, spinner
from evalution.results import RunResult
from evalution.suites.base import TestSuite


def run(
    *,
    model: Model | dict,
    engine: object,
    tests: Sequence[TestSuite],
) -> RunResult:
    model_config = coerce_model(model)
    if not hasattr(engine, "build"):
        raise TypeError("engine must provide a `build(model)` method")

    logger = get_logger()
    logger.info("building engine %s for model %s", type(engine).__name__, model_config.path)
    with spinner(f"Loading {type(engine).__name__} engine"):
        session = engine.build(model_config)
    describe_execution = getattr(session, "describe_execution", None)
    execution = None
    if callable(describe_execution):
        execution = describe_execution()
        logger.info("engine execution=%s", execution)
    try:
        results = []
        total_tests = len(tests)
        for index, test in enumerate(tests):
            logger.info("running test suite %s", type(test).__name__)
            result = test.evaluate(session)
            logger.info("completed test %s", result.name)
            results.append(result)
            gc_session = getattr(session, "gc", None)
            if index + 1 < total_tests and callable(gc_session):
                logger.info("running engine gc before next test suite")
                gc_session()
    finally:
        logger.info("closing engine session")
        session.close()

    engine_config = engine.to_dict() if hasattr(engine, "to_dict") else {}
    if execution is not None:
        engine_config = {**engine_config, "execution": execution}
    logger.info("finished evaluation run with %d test suite(s)", len(results))
    return RunResult(
        model=model_config.to_dict(),
        engine=engine_config,
        tests=results,
    )
