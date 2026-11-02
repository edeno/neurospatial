"""Event-aligned analyses of a deliberately empty selection of intervals."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Intervals
from neurospatial.events import (
    align_continuous_to_events,
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


def test_continuous_alignment_empty_selection():
    signal = np.sin(TIMES)
    selected = align_continuous_to_events(
        TIMES, signal, EVENTS, (-1.0, 1.0), dt=0.05, epochs=TRIALS
    )
    assert selected.n_events == 3

    none = TRIALS[TRIALS.cue == "up"]
    with pytest.warns(UserWarning, match="epochs selects no time") as record:
        result = align_continuous_to_events(
            TIMES, signal, EVENTS, (-1.0, 1.0), dt=0.05, epochs=none
        )
    assert len(record) == 1
    assert result.n_events == 0
    assert result.event_status.tolist() == ["outside_epochs"] * EVENTS.size
    assert np.isnan(result.means).all()
    assert result.values.shape[0] == 0

    elsewhere = Intervals.from_arrays([40.0], [45.0])
    with pytest.raises(ValueError, match="All 3 events were excluded"):
        align_continuous_to_events(
            TIMES, signal, EVENTS, (-1.0, 1.0), dt=0.05, epochs=elsewhere
        )


NO_TRIALS = TRIALS[TRIALS.cue == "up"]
EVENT_SELECTIONS = {"no rows": NO_TRIALS, "empty array": np.array([])}
PSTH_CALLS = {
    "peri_event_histogram": lambda events, **kw: peri_event_histogram(
        SPIKES, events, (-1.0, 1.0), bin_size=0.25, **kw
    ),
    "population_peri_event_histogram": lambda events, **kw: (
        population_peri_event_histogram(
            [SPIKES, SPIKES[::2]], events, (-1.0, 1.0), bin_size=0.25, **kw
        )
    ),
}


@pytest.mark.parametrize("selection", list(EVENT_SELECTIONS))
@pytest.mark.parametrize("name", list(PSTH_CALLS))
def test_psth_of_no_events_is_nan_with_a_warning(name, selection):
    events = EVENT_SELECTIONS[selection]
    with pytest.warns(UserWarning, match="event_times holds no event") as record:
        result = PSTH_CALLS[name](events)
    assert len(record) == 1
    assert (result.n_events, result.n_events_dropped) == (0, 0)
    assert result.event_indices.size == 0
    assert np.isnan(_rates(result)).all()
    assert np.isnan(result.sem).all()
    assert isinstance(result.summary(), dict)
    assert len(result.to_dataframe()) == _rates(result).size
    if selection == "no rows":
        assert result.align == "start"
        np.testing.assert_array_equal(result.events_per_bin, 0)
    else:
        assert result.align is None
        assert result.events_per_bin is None


@pytest.mark.parametrize("selection", list(EVENT_SELECTIONS))
def test_response_plot_of_no_events_draws_an_empty_raster(selection):
    import matplotlib.pyplot as plt

    from neurospatial.events import plot_peri_event_response

    events = EVENT_SELECTIONS[selection]
    with pytest.warns(UserWarning, match="event_times holds no event"):
        result = peri_event_histogram(SPIKES, events, (-1.0, 1.0), bin_size=0.25)
    raster_ax, _ = plot_peri_event_response(SPIKES, events, result)
    assert raster_ax.get_ylim() == (-0.5, 0.5)
    assert raster_ax.get_title() == "0 retained events"
    plt.close(raster_ax.figure)


def test_distance_histogram_of_no_events_is_nan_with_a_warning():
    with pytest.warns(UserWarning, match="event_times holds no event") as record:
        result = peri_event_distance_histogram(
            SPIKES, np.array([]), TIMES, POSITIONS, (-10.0, 10.0), bin_size=2.0
        )
    assert len(record) == 1
    assert (result.n_events, result.n_events_dropped) == (0, 0)
    assert result.event_indices.size == 0
    assert np.isnan(result.firing_rate).all()


@pytest.mark.parametrize("name", list(PSTH_CALLS))
def test_psth_of_no_trials_aligned_at_their_stop(name):
    with pytest.warns(UserWarning, match="event_times holds no event"):
        result = PSTH_CALLS[name](NO_TRIALS, align="stop")
    assert result.align == "stop"
    assert result.n_events == 0


@pytest.mark.parametrize("selection", list(EVENT_SELECTIONS))
def test_raster_of_no_events_is_empty(selection):
    from neurospatial.events import align_spikes_to_events

    assert (
        align_spikes_to_events(SPIKES, EVENT_SELECTIONS[selection], (-1.0, 1.0)) == []
    )


@pytest.mark.parametrize("selection", list(EVENT_SELECTIONS))
def test_continuous_alignment_of_no_events_is_nan_with_a_warning(selection):
    events = EVENT_SELECTIONS[selection]
    with pytest.warns(UserWarning, match="event_times holds no event") as record:
        result = align_continuous_to_events(
            TIMES, np.sin(TIMES), events, (-1.0, 1.0), dt=0.05
        )
    assert len(record) == 1
    assert (result.n_events, result.n_events_dropped) == (0, 0)
    assert result.event_status.size == 0
    assert np.isnan(result.means).all()
    assert result.values.shape == (0, result.relative_times.size, 1)
    assert isinstance(result.summary(), dict)
    if selection == "no rows":
        assert result.align == "start"
        assert result.event_ids.size == 0
    else:
        assert result.align is None
        assert result.event_ids is None
