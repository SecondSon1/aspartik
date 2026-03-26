import pytest
from pytest import Config, TestReport, fixture

import os
import sys

from aspartik.rng import RNG

_rng = RNG(4)


sys.path.insert(0, os.path.abspath("python/"))


def pytest_report_teststatus(report: TestReport, config: Config):
    global _rng

    if report.outcome != "passed" or report.when != "call":
        return None  # handled by pytest

    letter = ""
    if _rng.random_bool(0.01):
        letter = "."
    return report.outcome, letter, report.outcome.upper()


_SKIP_MARKERS = {
    "manual": "need -m manual option to run",
    "likelihood": "need -m likelihood option to run (requires BEAST1 + BEAGLE)",
}


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("-m")

    for marker, reason in _SKIP_MARKERS.items():
        if selected == marker:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@fixture
def rng():
    return RNG(4)
