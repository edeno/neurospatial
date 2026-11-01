"""Event-aligned analyses of a deliberately empty selection of intervals."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Intervals
from neurospatial.events import (
    peri_event_distance_histogram,
    peri_event_histogram,
    population_peri_event_histogram,
)

TIMES = np.arange(0.0, 60.0, 0.05)
POSITIONS = np.column_stack([np.linspace(0.0, 120.0, TIMES.size), np.zeros(TIMES.size)])
SPIKES = np.sort(np.random.default_rng(3).uniform(0.0, 60.0, 400))
EVENTS = np.array([10.0, 20.0, 30.0])
TRIALS = Intervals.from_arrays(
    [8.0, 18.0, 28.0], [12.0, 22.0, 32.0], cue=["left", "right", "left"]
)

CALLS = {
    "peri_event_histogram": lambda epochs: peri_event_histogram(
        SPIKES, EVENTS, (-1.0, 1.0), bin_size=0.25, epochs=epochs
    ),
    "population_peri_event_histogram": lambda epochs: population_peri_event_histogram(
        [SPIKES, SPIKES[::2]], EVENTS, (-1.0, 1.0), bin_size=0.25, epochs=epochs
    ),
    "peri_event_distance_histogram": lambda epochs: peri_event_distance_histogram(
        SPIKES, EVENTS, TIMES, POSITIONS, (-10.0, 10.0), bin_size=2.0, epochs=epochs
    ),
}


def _rates(result) -> np.ndarray:
    rates = getattr(result, "firing_rates", None)
    return np.asarray(result.firing_rate if rates is None else rates)


@pytest.mark.parametrize("name", list(CALLS))
def test_event_aligned_empty_selection(name):
    call = CALLS[name]
    # A real condition selects events and gives defined values.
    selected = call(TRIALS[TRIALS.cue == "left"])
    assert selected.n_events == 2
    assert not np.isnan(_rates(selected)).all()

    # A condition with no trials: NaN, no events, one warning.
    none = TRIALS[TRIALS.cue == "up"]
    with pytest.warns(UserWarning, match="epochs selects no time") as record:
        result = call(none)
    assert len(record) == 1
    assert result.n_events == 0
    assert result.n_events_dropped == EVENTS.size
    assert result.event_indices.size == 0
    assert np.isnan(_rates(result)).all()
    assert isinstance(result.summary(), dict)

    # Windows with duration that contain no event usually mean another clock.
    elsewhere = Intervals.from_arrays([40.0], [45.0])
    with pytest.raises(
        ValueError, match=r"All 3 events were dropped|none of the 3 events falls"
    ):
        call(elsewhere)


def test_empty_selection_still_validates_the_recording():
    none = TRIALS[TRIALS.cue == "up"]
    with pytest.raises(ValueError, match="times"):
        peri_event_distance_histogram(
            SPIKES,
            EVENTS,
            TIMES[::-1],
            POSITIONS,
            (-10.0, 10.0),
            bin_size=2.0,
            epochs=none,
        )
