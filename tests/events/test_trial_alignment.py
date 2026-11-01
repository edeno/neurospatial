"""Aligning trials (rows of an Intervals) at their start or stop.

Each trial's window ends at the trial's other bound, so nothing from the next
(or previous) trial is sampled; the side of the alignment point is sampled as
for any event.
"""

import numpy as np
import pytest

from neurospatial import Intervals
from neurospatial.events import (
    align_continuous_to_events,
    align_spikes_to_events,
    peri_event_histogram,
    population_peri_event_histogram,
)


@pytest.fixture
def clock():
    """10 Hz samples from 0 to 20 s, frames at 6.3, 6.4 and 6.5 s dropped."""
    return np.delete(np.arange(200) / 10, [63, 64, 65])


@pytest.fixture
def trials():
    """Three trials; the last has no duration."""
    return Intervals.from_arrays([1.0, 5.0, 9.0], [3.0, 8.5, 9.0])


SPIKES = np.array([0.95, 1.05, 1.5, 2.95, 3.05, 5.2, 8.4, 8.6, 9.2])


def test_values_follow_time_since_start_across_dropped_frames(clock, trials):
    # The signal is 2 * time, so the value t s after a trial starts is
    # 2 * (start + t) whatever frames were dropped; an alignment by sample
    # index would be off by three frames after 6.2 s.
    result = align_continuous_to_events(
        clock, 2 * clock, trials, (-0.5, 4.0), dt=0.5, max_gap=0.5
    )
    rel = result.relative_times
    values = result.values[..., 0]
    starts, stops = trials.start, trials.stop
    expected = 2 * (starts[:, None] + rel[None, :])
    inside = starts[:, None] + rel[None, :] < stops[:, None]
    np.testing.assert_allclose(values[inside], expected[inside], rtol=0, atol=1e-12)
    assert np.isnan(values[~inside]).all()
    # Trial 1 at 1.5 s after its start is 6.5 s, a dropped frame.
    assert values[1, rel.tolist().index(1.5)] == pytest.approx(13.0)
    np.testing.assert_array_equal(result.event_ids, trials.interval_ids)
    assert result.align == "start"
    # Each cell averages the trials under way at that time.
    np.testing.assert_array_equal(result.counts[:, 0], inside.sum(axis=0))


def test_aligned_at_stop_keeps_the_time_after_arrival(clock, trials):
    result = align_continuous_to_events(
        clock, clock, trials, (-4.0, 1.0), dt=0.5, align="stop"
    )
    rel = result.relative_times
    absolute = trials.stop[:, None] + rel[None, :]
    inside = absolute >= trials.start[:, None]
    np.testing.assert_allclose(result.values[..., 0][inside], absolute[inside])
    assert np.isnan(result.values[..., 0][~inside]).all()
    # The second after each arrival (rel >= 0) is kept for every trial.
    assert inside[:, rel >= 0].all()


def test_trial_windows_cut_at_their_own_stop_do_not_overlap(clock, trials):
    result = align_continuous_to_events(clock, clock, trials, (0.0, 10.0), dt=0.5)
    assert not result.overlapping.any()


def test_trial_table_rows_are_labelled_by_interval_id(clock):
    trials = Intervals.from_arrays([1.0, 5.0], [3.0, 8.0], interval_ids=[7, 3])
    result = align_continuous_to_events(clock, clock, trials, (0.0, 2.0), dt=0.5)
    assert result.events_table()["event_id"].tolist() == [7, 3]
    assert result.events_table().attrs["align"] == "start"


def test_values_match_pynapple_build_tensor_before_each_stop():
    nap = pytest.importorskip("pynapple")
    times = np.arange(100) / 10
    trials = Intervals.from_arrays([1.0, 5.0], [3.0, 8.5])
    ours = align_continuous_to_events(times, times, trials, (0.0, 3.5), dt=0.1).values[
        ..., 0
    ]
    theirs = nap.build_tensor(
        nap.Tsd(t=times, d=times), nap.IntervalSet(trials.start, trials.stop)
    )
    n = ours.shape[1]
    # pynapple's intervals are closed, so it also holds the sample at each
    # stop; before it the two agree on regularly sampled data.
    before_stop = ~np.isnan(ours)
    np.testing.assert_allclose(ours[before_stop], theirs[:, :n][before_stop])
    np.testing.assert_allclose(theirs[0, 20], 3.0)
    assert np.isnan(ours[0, 20])


def test_psth_averages_each_bin_over_the_trials_under_way(trials):
    result = peri_event_histogram(SPIKES, trials, (-0.5, 4.0), bin_size=0.5)
    # By hand: trial [1, 3) counts 1 1 1 0 1 in its bins from 0.5 s; trial
    # [5, 8.5) counts 0 1 0 0 0 0 0 1 from 4.5 s; trial [9, 9) only has the
    # bin before its start, holding the spike at 8.6 s.
    per_trial = np.array(
        [
            [1, 1, 1, 0, 1, np.nan, np.nan, np.nan, np.nan],
            [0, 1, 0, 0, 0, 0, 0, 1, np.nan],
            [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    n = (~np.isnan(per_trial)).sum(axis=0)
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmean(per_trial, axis=0)
    np.testing.assert_allclose(result.histogram, expected)
    np.testing.assert_array_equal(result.events_per_bin, n)
    sem = np.full(n.size, np.nan)
    for b in np.flatnonzero(n > 1):
        column = per_trial[:, b][~np.isnan(per_trial[:, b])]
        sem[b] = column.std(ddof=1) / np.sqrt(column.size)
    np.testing.assert_allclose(result.sem, sem)
    assert result.n_events == 3
    assert result.align == "start"
    assert result.to_dataframe()["n_events"].tolist() == n.tolist()


def test_psth_of_trials_longer_than_the_window_equals_plain_times():
    trials = Intervals.from_arrays([1.0, 5.0, 9.0], [4.0, 8.0, 12.0])
    spikes = np.sort(np.random.default_rng(0).uniform(0, 13, 80))
    from_trials = population_peri_event_histogram(
        [spikes, spikes[::2]], trials, (-0.5, 2.0), bin_size=0.25
    )
    from_times = population_peri_event_histogram(
        [spikes, spikes[::2]], trials.start, (-0.5, 2.0), bin_size=0.25
    )
    np.testing.assert_allclose(from_trials.histograms, from_times.histograms)
    np.testing.assert_allclose(from_trials.sem, from_times.sem)
    np.testing.assert_allclose(from_trials.mean_sem, from_times.mean_sem)
    np.testing.assert_array_equal(from_trials.events_per_bin, 3)
    assert (
        from_trials[1].events_per_bin is from_trials.events_per_bin
        or (from_trials[1].events_per_bin == from_trials.events_per_bin).all()
    )
    assert from_times.events_per_bin is None
    assert from_times.align is None


def test_psth_window_outside_every_trial_raises(trials):
    with pytest.raises(ValueError, match="no interval holds a whole bin"):
        # Every trial is shorter than 4 s, and its window ends at its stop.
        peri_event_histogram(SPIKES, trials, (4.0, 5.0), bin_size=0.5)


def test_raster_is_cut_at_each_trial_s_other_bound(trials):
    from_start = align_spikes_to_events(SPIKES, trials, (-0.5, 4.0))
    expected = [[-0.05, 0.05, 0.5, 1.95], [0.2, 3.4], [-0.4]]
    for got, want in zip(from_start, expected, strict=True):
        np.testing.assert_allclose(got, want)
    from_stop = align_spikes_to_events(SPIKES, trials, (-1.0, 1.0), align="stop")
    expected = [[-0.05, 0.05], [-0.1, 0.1, 0.7], [0.2]]
    for got, want in zip(from_stop, expected, strict=True):
        np.testing.assert_allclose(got, want)


@pytest.mark.parametrize(
    ("events", "kwargs", "match"),
    [
        (np.array([1.0, 2.0]), {"align": "stop"}, "given with plain event times"),
        (None, {"align": "middle"}, "align='middle' is not supported"),
        (None, {"event_ids": ["a", "b", "c"]}, "event_ids was given with an Intervals"),
    ],
)
def test_misuse_is_refused(clock, trials, events, kwargs, match):
    with pytest.raises(ValueError, match=match):
        align_continuous_to_events(
            clock, clock, trials if events is None else events, (0, 1), dt=0.5, **kwargs
        )


def test_an_intervals_without_rows_is_refused(clock, trials):
    with pytest.raises(ValueError, match="Intervals with no rows"):
        peri_event_histogram(SPIKES, trials[np.zeros(3, dtype=bool)], (0.0, 1.0))


def test_continuous_cut_is_exact_at_float_rounded_trial_bounds():
    times = np.round(np.arange(0, 5, 0.1), 10)
    # 0.7 + 0.1 rounds below 0.8: that query is the trial's stop, not inside.
    result = align_continuous_to_events(
        times,
        times,
        Intervals.from_arrays([0.7], [0.8]),
        (0.0, 1.0),
        dt=0.1,
        window_coverage="partial",
    )
    np.testing.assert_allclose(
        result.values[0, :, 0][~np.isnan(result.values[0, :, 0])], [0.7]
    )
    # 0.3 - 0.2 rounds below 0.1: that query is the trial's start, inside
    # (and after the stop is kept when aligned at the stop).
    result = align_continuous_to_events(
        times,
        times,
        Intervals.from_arrays([0.1], [0.3]),
        (-0.2, 0.1),
        dt=0.1,
        align="stop",
        window_coverage="partial",
    )
    np.testing.assert_allclose(result.values[0, :, 0], [0.1, 0.2, 0.3])


def test_psth_leaves_a_spike_at_a_trial_stop_to_the_next_trial():
    trials = Intervals.from_arrays([0.1, 0.3], [0.3, 0.6])
    result = peri_event_histogram(np.array([0.3]), trials, (0.0, 0.3), bin_size=0.1)
    np.testing.assert_allclose(result.histogram, [0.5, 0.0, 0.0])
    raster = align_spikes_to_events(np.array([0.3]), trials, (0.0, 0.3))
    assert [r.tolist() for r in raster] == [[], [0.0]]


def test_response_plot_cuts_raster_rows_like_the_histogram():
    from neurospatial.events import plot_peri_event_response

    trials = Intervals.from_arrays([1.0, 2.0], [2.0, 3.0])
    spikes = np.array([1.5, 2.1, 2.5])
    result = peri_event_histogram(spikes, trials, (0.0, 2.0), bin_size=0.5)
    raster_ax, _ = plot_peri_event_response(spikes, trials, result)
    # Trial [1, 2) shows only its spike at 1.5; 2.1 and 2.5 are the next
    # trial's, in its own row.
    rows = [list(np.round(c.get_positions(), 9)) for c in raster_ax.collections]
    assert rows == [[0.5], [0.1, 0.5]]
    with pytest.raises(ValueError, match="computed from intervals"):
        plot_peri_event_response(spikes, trials.start, result)
