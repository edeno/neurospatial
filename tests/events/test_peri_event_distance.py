"""Firing aligned to the distance travelled from an event."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.events import PeriEventDistanceResult, peri_event_distance_histogram


def _hand_path():
    """1-D path 0 -> 10 -> 20, a 1 s pause, then 20 -> 40, at 1 s steps."""
    times = np.arange(5.0)
    positions = np.array([[0.0], [10.0], [20.0], [20.0], [40.0]])
    return times, positions


def test_rate_is_spikes_over_time_spent_at_each_distance():
    """Worked by hand: event at t = 1 s (10 cm along the path).

    Distances from the event are -10, 0, 10, 10, 30 at the samples. Time per
    10 cm bin from -10: 1 s, 1 s, 1 s pausing at 10 plus 0.5 s moving on,
    and 0.5 s. Spikes at 0.5, 2.5, 3.25, 3.75 s are at -5, 10, 15, 25 cm.
    """
    times, positions = _hand_path()
    result = peri_event_distance_histogram(
        np.array([0.5, 2.5, 3.25, 3.75]),
        np.array([1.0]),
        times,
        positions,
        (-10.0, 30.0),
        bin_size=10.0,
        max_gap=None,
    )
    assert isinstance(result, PeriEventDistanceResult)
    np.testing.assert_allclose(result.bin_centers, [-5.0, 5.0, 15.0, 25.0])
    np.testing.assert_allclose(result.occupancy, [[1.0, 1.0, 1.5, 0.5]])
    np.testing.assert_allclose(result.spike_counts, [[1, 0, 2, 1]])
    np.testing.assert_allclose(result.firing_rate, [1.0, 0.0, 2 / 1.5, 2.0])


def test_rate_does_not_depend_on_running_speed():
    """A cell firing 20 Hz within 10 cm after a cue, 2 Hz elsewhere.

    Laps run at 10 or 40 cm/s cover the distance in different times; the
    rate per distance bin is the same.
    """
    rng = np.random.default_rng(0)
    lap_times, lap_positions, cues, spikes = [], [], [], []
    clock = 0.0
    for lap in range(80):
        speed = 10.0 if lap % 2 else 40.0
        t = clock + np.arange(0.0, 100.0 / speed, 1 / 120)
        x = (t - clock) * speed
        cue_time = clock + 50.0 / speed  # the cue is passed at 50 cm
        rate = np.where((x >= 50.0) & (x < 60.0), 20.0, 2.0)
        dt = np.diff(t, append=t[-1] + 1 / 120)
        n = rng.poisson(rate * dt)
        spikes.append(np.repeat(t, n) + rng.uniform(0, 1 / 120, n.sum()))
        lap_times.append(t)
        lap_positions.append(x)
        cues.append(cue_time)
        clock = t[-1] + 5.0  # a break between laps
    result = peri_event_distance_histogram(
        np.sort(np.concatenate(spikes)),
        np.array(cues),
        np.concatenate(lap_times),
        np.concatenate(lap_positions)[:, None],
        (-30.0, 30.0),
        bin_size=10.0,
    )
    np.testing.assert_allclose(
        result.firing_rate, [2.0, 2.0, 2.0, 20.0, 2.0, 2.0], rtol=0.15
    )


def test_distance_never_seen_has_no_rate():
    """Missing is not zero: a bin no event's path reached is NaN."""
    times, positions = _hand_path()
    result = peri_event_distance_histogram(
        np.array([0.5]),
        np.array([1.0]),
        times,
        positions,
        (-20.0, 40.0),
        bin_size=10.0,
        max_gap=None,
    )
    assert np.isnan(result.firing_rate[0])  # [-20, -10): before the path
    assert np.isnan(result.firing_rate[-1])  # [30, 40): past its end


def test_event_in_a_tracking_gap_is_dropped_and_distance_stops_at_a_gap():
    times = np.r_[np.arange(0.0, 5.0, 0.5), np.arange(20.0, 25.0, 0.5)]
    positions = np.r_[np.arange(10.0) * 5, 100 + np.arange(10.0) * 5][:, None]
    result = peri_event_distance_histogram(
        np.array([1.0, 21.0]),
        np.array([2.0, 10.0]),  # 10 s is in the 15 s tracking gap
        times,
        positions,
        (-10.0, 40.0),
        bin_size=10.0,
        max_gap=1.0,
    )
    assert result.n_events == 1 and result.n_events_dropped == 1
    np.testing.assert_array_equal(result.event_indices, [0])
    # The path of the first event ends 25 cm after it, at the gap; the run
    # after the gap is not part of its distance axis.
    assert result.occupancy[0, -1] == 0.0
    assert result.spike_counts.sum() == 1


def test_spike_window_limits_the_time_counted():
    times, positions = _hand_path()
    result = peri_event_distance_histogram(
        np.array([0.5, 2.5, 3.25, 3.75]),
        np.array([1.0]),
        times,
        positions,
        (-10.0, 30.0),
        bin_size=10.0,
        max_gap=None,
        spike_window=(0.0, 2.0),
    )
    np.testing.assert_allclose(result.occupancy, [[1.0, 1.0, 0.0, 0.0]])
    assert np.isnan(result.firing_rate[2:]).all()


def test_tables_and_summary():
    times, positions = _hand_path()
    result = peri_event_distance_histogram(
        np.array([0.5, 2.5, 3.25, 3.75]),
        np.array([1.0]),
        times,
        positions,
        (-10.0, 30.0),
        bin_size=10.0,
        max_gap=None,
    )
    df = result.to_dataframe()
    assert list(df.columns) == [
        "unit_id",
        "bin_center",
        "firing_rate",
        "spike_count",
        "occupancy",
        "n_events_covering",
    ]
    table = result.summary_table()
    assert table.index.name == "unit_id"
    assert table["peak_rate"].iloc[0] == pytest.approx(2.0)
    assert table["peak_distance"].iloc[0] == pytest.approx(25.0)
    assert table["baseline_rate"].iloc[0] == pytest.approx(1.0)
    assert result.summary()["n_events"] == 1
    with pytest.raises(ValueError, match="read-only"):
        result.occupancy[0, 0] = 5.0
