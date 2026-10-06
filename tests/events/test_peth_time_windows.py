"""Only fully observed event windows contribute to peri-event rates."""

import warnings

import numpy as np
import pytest

from neurospatial.events.alignment import (
    peri_event_histogram,
    population_peri_event_histogram,
)


@pytest.fixture
def event_recording():
    return (
        np.r_[np.arange(0.05, 100, 0.1), np.arange(1100.05, 1200, 0.1)],
        np.array([10, 50, 99.5, 1100.2, 1150]),
        [(0.0, 100.0), (1100.0, 1200.0)],
    )


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize(
    "keyword, kept, dropped", [("epochs", 3, 2), ("spike_window", 2, 3)]
)
def test_drops_events_whose_window_leaves_recording(
    event_recording, population, keyword, kept, dropped
):
    spikes, events, epochs = event_recording
    function = population_peri_event_histogram if population else peri_event_histogram
    values = epochs if keyword == "epochs" else (0.0, 100.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(
            [spikes, spikes] if population else spikes,
            events,
            (-0.5, 1.0),
            **{keyword: values},
        )
    assert not caught
    assert result.n_events == kept
    assert result.n_events_dropped == dropped
    assert result.summary()["n_events_dropped"] == dropped
    if population:
        assert result[0].n_events_dropped == dropped
        assert all(unit.n_events == kept for unit in result)


@pytest.mark.parametrize("population", [False, True])
def test_flat_rate_recovered_at_recording_edges(event_recording, population):
    spikes, _, epochs = event_recording
    events = np.random.default_rng(0).uniform(0, 1200, 200)
    function = population_peri_event_histogram if population else peri_event_histogram
    spike_input = [spikes, spikes] if population else spikes
    observed = function(spike_input, events, (-1, 1), bin_size=0.1, epochs=epochs)
    naive = function(spike_input, events, (-1, 1), bin_size=0.1)
    rate = observed.mean_firing_rate if population else observed.firing_rate
    naive_rate = naive.mean_firing_rate if population else naive.firing_rate
    np.testing.assert_allclose(rate, 10.0, rtol=0.01)
    assert np.min(naive_rate) < 9.0


@pytest.mark.parametrize("population", [False, True])
def test_all_events_dropped_error(event_recording, population):
    spikes, events, _ = event_recording
    function = population_peri_event_histogram if population else peri_event_histogram
    with pytest.raises(ValueError) as error:
        function(
            [spikes] if population else spikes, events, (-0.5, 1), epochs=(5000, 6000)
        )
    message = str(error.value)
    assert "peri-event" in message
    assert "5 events" in message
    assert "Why:" in message
    assert any(line.startswith("Fix:") for line in message.splitlines())


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_event_still_raises(event_recording, population, invalid):
    spikes, _, epochs = event_recording
    function = population_peri_event_histogram if population else peri_event_histogram
    with pytest.raises(ValueError, match=r"NaN|Inf"):
        function(
            [spikes] if population else spikes,
            np.array([invalid, 10.0]),
            (-0.5, 1),
            epochs=epochs,
        )


def test_both_windows_intersect(event_recording):
    spikes, events, epochs = event_recording
    result = peri_event_histogram(
        spikes, events, (-0.5, 1), epochs=epochs, spike_window=(0, 100)
    )
    assert result.n_events == 2
    assert result.n_events_dropped == 3


def test_defaults_keep_all_events(event_recording):
    spikes, events, _ = event_recording
    result = peri_event_histogram(spikes, events, (-0.5, 1))
    assert result.n_events == events.size
    assert result.n_events_dropped == 0


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize(
    "window,bin_size,spikes,expected",
    [
        ((0.1, 0.3), 0.1, [0.1, 0.2, 0.3], [1, 1]),
        ((0.0, 1.05), 0.25, [0.99, 1.0, 1.04], [0, 0, 0, 1]),
        ((1e9 + 0.1, 1e9 + 0.3), 0.1, [1e9 + 0.1, 1e9 + 0.3], [1, 0]),
    ],
)
def test_peth_bins_are_half_open(population, window, bin_size, spikes, expected):
    """Every whole bin excludes its right edge, including the last bin."""
    # Duplicate events avoid the single-event warning without shifting boundaries.
    function = population_peri_event_histogram if population else peri_event_histogram
    spike_array = np.asarray(spikes)
    result = function(
        [spike_array] if population else spike_array,
        np.array([0.0, 0.0]),
        window,
        bin_size=bin_size,
    )
    counts = result.histograms[0] if population else result.histogram
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(result.sem, np.zeros_like(result.sem))
    assert np.all(result.bin_centers < window[1])


@pytest.mark.parametrize("population", [False, True])
def test_no_whole_peth_bin_preserves_empty_result(population):
    function = population_peri_event_histogram if population else peri_event_histogram
    spikes = np.array([0.1])
    result = function(
        [spikes] if population else spikes,
        np.array([0.0, 0.0]),
        (0, 0.1),
        bin_size=0.25,
    )
    counts = result.histograms[0] if population else result.histogram
    assert counts.shape == (0,)
    assert result.bin_centers.shape == (0,)
