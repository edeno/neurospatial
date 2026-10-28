"""Scientific invariants of peri-event histograms.

A PSTH must recover a known rate, not depend on the clock origin or event
order, and report the uncertainty of the curve it shows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial.events import (
    peri_event_histogram,
    population_peri_event_histogram,
)
from neurospatial.simulation import generate_poisson_spikes

WINDOW = (-0.5, 0.5)
BIN_SIZE = 0.125  # exact in binary, so brute-force edges match exactly


def _brute_force_counts(spike_times, event_times):
    """Counts per event and bin, shape (n_events, n_bins), from np.histogram."""
    edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE / 2, BIN_SIZE)
    return np.array([np.histogram(spike_times - e, bins=edges)[0] for e in event_times])


@pytest.fixture(scope="module")
def population():
    """Three units with different rates and an event-locked response."""
    rng = np.random.default_rng(1)
    event_times = np.sort(rng.uniform(5.0, 195.0, 40))
    trains = [np.sort(rng.uniform(0.0, 200.0, n)) for n in (300, 800, 1500)]
    trains[0] = np.sort(np.r_[trains[0], event_times + 0.05])
    return trains, event_times


def test_population_mean_sem_is_across_events(population):
    """mean_sem is the SEM across events of each event's mean over units."""
    trains, event_times = population
    result = population_peri_event_histogram(
        trains, event_times, WINDOW, bin_size=BIN_SIZE
    )

    counts = np.stack([_brute_force_counts(t, event_times) for t in trains])
    per_event_mean = counts.mean(axis=0)
    expected = per_event_mean.std(axis=0, ddof=1) / np.sqrt(event_times.size)

    assert_allclose(result.mean_histogram, per_event_mean.mean(axis=0))
    assert_allclose(result.mean_sem, expected)


def test_population_mean_sem_is_zero_when_units_alternate():
    """One unit fires on even events and one on odd: the mean never varies.

    The mean of the per-unit SEMs, drawn before, gave a band of +/-0.17
    counts around a curve with no event-to-event variability.
    """
    events = np.arange(10.0, 110.0, 10.0)
    trains = [events[0::2] + 0.1, events[1::2] + 0.1]

    result = population_peri_event_histogram(trains, events, WINDOW, bin_size=0.5)

    assert_allclose(result.mean_histogram, [0.0, 0.5])
    assert_allclose(result.mean_sem, [0.0, 0.0], atol=1e-15)
    assert (result.sem[:, 1] > 0).all()


def test_population_plot_band_is_mean_sem(population):
    import matplotlib.pyplot as plt

    trains, event_times = population
    result = population_peri_event_histogram(
        trains, event_times, WINDOW, bin_size=BIN_SIZE
    )
    ax = result.plot()
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    rate, rate_sem = result.mean_firing_rate, result.mean_sem / BIN_SIZE

    assert_allclose(band.max(), (rate + rate_sem).max())
    assert_allclose(band.min(), (rate - rate_sem).min())
    plt.close(ax.figure)


def test_each_population_unit_matches_its_single_unit_psth(population):
    trains, event_times = population
    result = population_peri_event_histogram(
        trains, event_times, WINDOW, bin_size=BIN_SIZE
    )
    for unit, train in enumerate(trains):
        single = peri_event_histogram(train, event_times, WINDOW, bin_size=BIN_SIZE)
        assert_array_equal(result.histograms[unit], single.histogram)
        assert_array_equal(result.sem[unit], single.sem)
        assert_array_equal(
            result.histograms[unit], _brute_force_counts(train, event_times).mean(0)
        )


def test_psth_does_not_depend_on_the_clock_origin(population):
    trains, event_times = population
    shift = 1234.0  # a whole number of bins keeps every edge exact
    base = peri_event_histogram(trains[0], event_times, WINDOW, bin_size=BIN_SIZE)
    shifted = peri_event_histogram(
        trains[0] + shift, event_times + shift, WINDOW, bin_size=BIN_SIZE
    )
    assert_array_equal(shifted.histogram, base.histogram)
    assert_array_equal(shifted.sem, base.sem)


def test_psth_does_not_depend_on_event_order(population):
    trains, event_times = population
    order = np.random.default_rng(2).permutation(event_times.size)
    base = population_peri_event_histogram(
        trains, event_times, WINDOW, bin_size=BIN_SIZE
    )
    permuted = population_peri_event_histogram(
        trains, event_times[order], WINDOW, bin_size=BIN_SIZE
    )
    assert_allclose(permuted.histograms, base.histograms, rtol=1e-12)
    assert_allclose(permuted.sem, base.sem, rtol=1e-12)
    assert_allclose(permuted.mean_sem, base.mean_sem, rtol=1e-12)
    assert_array_equal(permuted.event_indices, np.arange(event_times.size))


def test_psth_recovers_a_known_poisson_rate():
    """5 Hz baseline with a 40 Hz response in [0, 200) ms after each event."""
    dt = 0.001
    times = np.arange(0.0, 1000.0, dt)
    event_times = np.arange(2.0, 998.0, 2.0)
    since_event = np.full(times.size, np.inf)
    nearest = np.searchsorted(event_times, times, side="right") - 1
    has_event = nearest >= 0
    since_event[has_event] = times[has_event] - event_times[nearest[has_event]]
    rate = np.where((since_event >= 0) & (since_event < 0.2), 40.0, 5.0)
    spikes = generate_poisson_spikes(rate, times, refractory_period=0.0, seed=7)

    result = peri_event_histogram(spikes, event_times, WINDOW, bin_size=0.05)

    expected = np.where(
        (result.bin_centers > 0) & (result.bin_centers < 0.2), 40.0, 5.0
    )
    # Poisson count per bin has variance equal to its mean.
    standard_error = np.sqrt(expected * 0.05 / event_times.size) / 0.05
    assert np.all(np.abs(result.firing_rate - expected) < 4 * standard_error)
    assert_allclose(result.firing_rate_sem, standard_error, rtol=0.2)
    assert abs(result.summary()["baseline_rate"] - 5.0) < 4 * standard_error[0] / 3


@pytest.mark.parametrize("population_call", [False, True])
def test_millisecond_window_warns_once(population_call):
    spikes = np.arange(0.0, 5000.0, 0.5)
    trains = [spikes, spikes + 0.1, spikes + 0.2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if population_call:
            population_peri_event_histogram(
                trains, np.array([2000.0, 3000.0]), (-500.0, 500.0), bin_size=10.0
            )
        else:
            peri_event_histogram(
                spikes, np.array([2000.0, 3000.0]), (-500.0, 500.0), bin_size=10.0
            )
    wide = [w for w in caught if "more than 60 seconds" in str(w.message)]
    assert len(wide) == 1
    assert wide[0].filename == __file__


@pytest.mark.parametrize("population_call", [False, True])
def test_two_dimensional_event_times_are_refused(population_call):
    """A (2, 2) array was flattened into four events."""
    spikes = np.array([1.0, 2.0])
    events = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="event_times must be a 1-D"):
        if population_call:
            population_peri_event_histogram([spikes], events, (-0.5, 0.5))
        else:
            peri_event_histogram(spikes, events, (-0.5, 0.5))


def test_bad_event_time_errors_name_the_first_bad_entry():
    spikes = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match=r"event_times must be numbers"):
        peri_event_histogram(spikes, ["1.0", "soon"], (-0.5, 0.5))
    with pytest.raises(ValueError, match=r"2 NaN value\(s\), first at index 1"):
        peri_event_histogram(spikes, [1.0, np.nan, np.nan], (-0.5, 0.5))
