"""add_positions never interpolates across a declared recording break."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurospatial.events import add_positions

BREAK = 8
OFFSET = 1024.0


@pytest.fixture
def joined():
    """Two 4 Hz recordings on one clock; the animal is moved at the join."""
    times = np.arange(16) / 4
    positions = np.c_[np.r_[np.arange(8.0), 50 + np.arange(8.0)], np.zeros(16)]
    events = np.array([0.5, 1.75, 1.875, 2.0, 2.125, 3.75])  # s, dyadic
    return times, positions, events, np.arange(16) == BREAK


def _x(times, positions, event_times, **kwargs):
    events = pd.DataFrame({"timestamp": event_times})
    return add_positions(events, times=times, positions=positions, **kwargs)[
        "x"
    ].to_numpy()


def test_declared_break_equals_long_gap(joined):
    times, positions, events, break_before = joined
    declared = _x(times, positions, events, break_before=break_before)
    after = events >= times[BREAK]
    reference = _x(
        np.where(np.arange(times.size) >= BREAK, times + OFFSET, times),
        positions,
        np.where(after, events + OFFSET, events),
    )
    np.testing.assert_array_equal(declared, reference)
    # Inside the break the position is unknown; at either sample it is observed.
    np.testing.assert_array_equal(declared[1:5], [7.0, np.nan, 50.0, 50.5])
    undeclared = _x(times, positions, events)
    assert np.isfinite(undeclared[2])


def test_no_breaks_matches_omitting_break_before(joined):
    times, positions, events, _ = joined
    expected = _x(times, positions, events)
    for flags in (np.zeros(times.size, dtype=bool), np.arange(times.size) == 0):
        np.testing.assert_array_equal(
            _x(times, positions, events, break_before=flags), expected
        )


@pytest.mark.parametrize(
    "flags",
    [np.zeros(15, dtype=bool), np.zeros(16, dtype=int)],
    ids=["wrong length", "not Boolean"],
)
def test_invalid_break_before_names_function(joined, flags):
    times, positions, events, _ = joined
    with pytest.raises(ValueError, match=r"add_positions\(\) needs break_before"):
        _x(times, positions, events, break_before=flags)
