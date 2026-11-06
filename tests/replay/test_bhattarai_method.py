"""BHATTARAI_2020: a line fit tested against circular shifts of the decoded positions."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    detect_replay,
    fit_linear_trajectory,
    replay_significance,
)
from neurospatial.replay.methods import BHATTARAI_2020, PUBLISHED_METHODS


@pytest.fixture
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    )


def _decoded(track, bins):
    """A posterior with all its mass on one bin per time bin, 10 ms apart."""
    bins = np.asarray(bins)
    return np.eye(track.n_bins)[bins], 0.01 * np.arange(bins.size)


def test_a_sweep_is_replay_and_needs_five_cells(track):
    posterior, times = _decoded(track, 2 * np.arange(25))
    five = np.zeros((25, 6), dtype=int)
    five[np.arange(25), np.arange(25) % 5] = 1
    result = detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=five, rng=0
    )
    assert result.is_replay
    assert result.score == pytest.approx(1.0)  # R² of a straight line
    four = np.zeros((25, 6), dtype=int)
    four[np.arange(25), np.arange(25) % 4] = 1
    assert not detect_replay(
        track,
        posterior,
        times,
        replay_method=BHATTARAI_2020,
        spike_counts=four,
        rng=0,
    ).is_replay


def _null_r_squared(track, positions, times, shifts):
    """R² of the positions rolled by each shift, by np.corrcoef and np.roll."""
    return np.array(
        [np.corrcoef(np.roll(positions, k), times)[0, 1] ** 2 for k in shifts]
    )


def test_null_shifts_the_decoded_positions_in_time(track):
    """Each shuffle is the event rolled by 1 to n - 1 bins; the null is its R²."""
    rng = np.random.default_rng(0)
    bins = rng.integers(0, track.n_bins, 21)
    posterior, times = _decoded(track, bins)
    result = replay_significance(
        track,
        posterior,
        times,
        score=fit_linear_trajectory,
        score_options={"method": "map"},
        nulls="time_bin_circular",
        n_shuffles=200,
        rng=1,
    )
    centers = np.ravel(track.to_linear(track.bin_centers))[bins]
    # Every shuffle's R² is the R² of one of the 20 rolls, and all 20 occur.
    by_shift = _null_r_squared(track, centers, times, range(1, 21))
    scores = result.shuffles["time_bin_circular"].null_scores
    distance = np.abs(scores[:, None] - by_shift[None, :]).min(axis=1)
    assert distance.max() < 1e-9
    assert len(np.unique(np.round(scores, 9))) == len(np.unique(np.round(by_shift, 9)))
    observed = _null_r_squared(track, centers, times, [0])[0]
    assert result.observed.score == pytest.approx(observed, abs=1e-9)


def test_a_straight_line_beats_its_shifts_and_a_scrambled_one_does_not(track):
    line = np.arange(30)
    scrambled = np.random.default_rng(3).permutation(line)
    p_values = {}
    for name, bins in {"line": line, "scrambled": scrambled}.items():
        posterior, times = _decoded(track, bins)
        result = replay_significance(
            track,
            posterior,
            times,
            score=fit_linear_trajectory,
            score_options={"method": "map"},
            nulls="time_bin_circular",
            n_shuffles=1000,
            rng=0,
        )
        p_values[name] = result.p_value
        null = result.shuffles["time_bin_circular"].null_scores
        if name == "line":
            # No shift of a straight line fits it as well.
            assert (null < result.observed.score - 1e-9).all()
        else:
            assert result.observed.score <= np.percentile(null, 95)
    assert p_values["line"] < 0.05 < p_values["scrambled"]


def test_a_sweep_has_a_p_value(track):
    posterior, times = _decoded(track, 2 * np.arange(25))
    counts = np.eye(6, dtype=int)[np.arange(25) % 6]
    result = detect_replay(
        track,
        posterior,
        times,
        replay_method=BHATTARAI_2020,
        spike_counts=counts,
        rng=0,
    )
    assert 0.0 < result.p_value < 0.05
    assert result.n_shuffles_used == 1000
    assert result.is_replay


def test_a_stationary_event_has_no_verdict(track):
    posterior, times = _decoded(track, np.full(12, 7))
    counts = np.eye(6, dtype=int)[np.arange(12) % 6]
    result = detect_replay(
        track,
        posterior,
        times,
        replay_method=BHATTARAI_2020,
        spike_counts=counts,
        rng=0,
    )
    assert not result.is_replay
    assert np.isnan(result.score)
    assert "undefined" in result.error


def test_time_bins_without_spikes_are_dropped_before_the_shifts(track):
    """A line with silent time bins: their posteriors, however wild, do not count."""
    bins = 2 * np.arange(24)
    posterior, times = _decoded(track, bins)
    counts = np.eye(6, dtype=int)[np.arange(24) % 6]
    silent = np.array([3, 4, 11])
    counts[silent] = 0
    posterior[silent] = np.eye(track.n_bins)[track.n_bins - 1]
    result = detect_replay(
        track,
        posterior,
        times,
        replay_method=BHATTARAI_2020,
        spike_counts=counts,
        rng=0,
    )
    assert result.is_replay
    assert result.score == pytest.approx(1.0)  # R² of the 21 kept bins' line


def test_recipe_records_the_papers_settings():
    assert BHATTARAI_2020 in PUBLISHED_METHODS
    assert BHATTARAI_2020.decoding["time_bin"] == 0.02
    assert BHATTARAI_2020.decoding["time_step"] == 0.01
    assert BHATTARAI_2020.units == "cm"
    assert BHATTARAI_2020.nulls == ("time_bin_circular",)
    assert BHATTARAI_2020.n_shuffles == 1000
    assert BHATTARAI_2020.alpha == 0.05
