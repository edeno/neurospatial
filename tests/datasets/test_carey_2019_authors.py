"""Carey et al. (2019): the authors' sequence pipeline, ported, against the authors' own run.

The reference is the authors' MATLAB code run in Octave on the paper's 19
sessions with the paper's own candidate events
(``scripts/reproductions/carey_2019``), committed as
``tests/data/carey_2019/authors_octave_sequences.csv``. The Python port
(``carey_authors_route.py``) reproduces it exactly: same units, time grid,
MAP and sequences, to the last bit of every sequence's start and end.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from carey_authors_route import session_sequences
from test_carey_2019 import _intervals_of, _paper_candidates

from neurospatial.datasets import load
from neurospatial.datasets._fetch import cache_root

REFERENCE = Path(__file__).parents[1] / "data" / "carey_2019"
#: Supplementary Fig. 6 of the paper: sequences per rat, and the total in the Results.
PAPER_PER_RAT = {"R042": 1257, "R044": 95, "R050": 429, "R064": 1999}
PAPER_TOTAL = 3780
#: The authors' code in Octave, shuffles off (the identity-shuffle filter
#: removes nothing: see scripts/reproductions/carey_2019/README.md).
AUTHORS_PER_RAT = {"R042": 1210, "R044": 91, "R050": 393, "R064": 1963}


def _reference() -> dict[tuple[str, str], np.ndarray]:
    rows: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(REFERENCE / "authors_octave_sequences.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["session"], r["side"]), []).append(
                (float(r["start_s"]), float(r["end_s"]))
            )
    return {k: np.array(v) for k, v in rows.items()}


def _sessions() -> list[str]:
    return sorted(json.loads((REFERENCE / "authors_octave_sessions.json").read_text()))


def test_reference_counts_are_the_octave_run():
    """The committed table holds the Octave run's 3,657 sequences, per rat as recorded."""
    ref = _reference()
    per_rat = Counter()
    for (session, _), rows in ref.items():
        per_rat[session[:4]] += len(rows)
    assert dict(per_rat) == AUTHORS_PER_RAT
    assert sum(per_rat.values()) == 3657
    info = json.loads((REFERENCE / "authors_octave_sessions.json").read_text())
    assert len(info) == 19
    assert all(
        info[s][side]["eligible_sequences"] == len(ref.get((s, side), ()))
        for s in info
        for side in ("left", "right")
    )


def _check_session(session: str, ref: dict[tuple[str, str], np.ndarray]) -> dict[str, int]:
    rec = load("carey_2019", session=session)
    events = _intervals_of(_paper_candidates(session))
    eligible, sides = session_sequences(rec, cache_root() / "carey_2019" / session, events)
    info = json.loads((REFERENCE / "authors_octave_sessions.json").read_text())[session]
    folder = cache_root() / "carey_2019" / session
    names = sorted(p.name for p in list(folder.glob("*.t")) + list(folder.glob("*._t")))
    for side in ("left", "right"):
        assert len(sides[side].units) == len(info[side]["units"])
        assert len(sides[side].raw_sequences) == info[side]["raw_sequences"]
        theirs = ref.get((session, side), np.empty((0, 2)))
        ours = eligible[side][np.argsort(eligible[side][:, 0])] if len(eligible[side]) else eligible[side]
        np.testing.assert_array_equal(ours, theirs)
    assert names  # the session's spike files are cached
    return {side: len(eligible[side]) for side in ("left", "right")}


@pytest.mark.realdata
def test_the_port_reproduces_the_authors_run_on_the_default_session():
    """R050-2014-03-29: every eligible sequence identical to the authors' code's (11 left, 30 right)."""
    counts = _check_session("R050-2014-03-29", _reference())
    assert counts == {"left": 11, "right": 30}


@pytest.mark.realdata
@pytest.mark.slow
def test_the_port_reproduces_the_authors_run_on_all_paper_sessions():
    """All 19 sessions, about 25 minutes: 3,657 sequences, each identical to the authors' code's.

    The paper reports 3,780 (Results) and per rat 1,257 / 95 / 429 / 1,999
    (Supplementary Fig. 6). The authors' code gives 3% fewer on the public
    data; the cause is not found (scripts/reproductions/carey_2019/README.md).
    """
    ref = _reference()
    per_rat: Counter[str] = Counter()
    for session in _sessions():
        counts = _check_session(session, ref)
        per_rat[session[:4]] += counts["left"] + counts["right"]
    assert dict(per_rat) == AUTHORS_PER_RAT
    for rat, n in per_rat.items():
        assert 0.90 * PAPER_PER_RAT[rat] <= n <= PAPER_PER_RAT[rat]
