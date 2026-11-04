"""Each verified session still reports the problems it is known to have.

A dataset module's ``EXPECTED_WATCH_OUTS`` maps verified sessions to
patterns, each searched for (``re.search``) in the loaded recording's
``watch_outs``. The patterns name findings computed from the data, by the
checks or by the loader's own detectors, not text the loader writes for that
session by hand: a check that stops reporting a problem a real file is known
to have fails here.
"""

from __future__ import annotations

import re
from importlib import import_module

import pytest

from neurospatial.datasets import load
from neurospatial.datasets._catalog import dataset_ids, sessions


def _expected(dataset: str) -> dict[str, tuple[str, ...]]:
    return import_module(f"neurospatial.datasets.{dataset}").EXPECTED_WATCH_OUTS


CASES = [
    pytest.param(dataset, session, patterns, id=f"{dataset}/{session}")
    for dataset in dataset_ids()
    for session, patterns in _expected(dataset).items()
]


@pytest.mark.parametrize("dataset", dataset_ids())
def test_expected_watch_outs_cover_the_verified_sessions(dataset):
    verified = {entry.session for entry in sessions(dataset) if entry.verified}
    expected = _expected(dataset)
    assert set(expected) == verified
    assert all(expected.values()), "every verified session names a finding"
    for patterns in expected.values():
        for pattern in patterns:
            re.compile(pattern)


@pytest.mark.realdata
# Several NWB files' devices carry the 'manufacturer' field pynwb now deprecates.
@pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize(("dataset", "session", "patterns"), CASES)
def test_session_reports_its_known_problems(dataset, session, patterns):
    rec = load(dataset, session)
    missing = [
        pattern
        for pattern in patterns
        if not any(re.search(pattern, text) for text in rec.watch_outs)
    ]
    assert not missing, f"{dataset}/{session} no longer reports {missing}"
