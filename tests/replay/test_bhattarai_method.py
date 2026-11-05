"""BHATTARAI_2020: a line fit tested against circular shifts of the decoded positions."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import detect_replay
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


def _share_by_loop(positions, times):
    """The share of circular shifts beating the event's R², shift by shift."""

    def r_squared(values):
        return np.corrcoef(values, times)[0, 1] ** 2

    observed = r_squared(positions)
    shifted = [r_squared(np.roll(positions, k)) for k in range(positions.size)]
    return float(np.mean(np.array(shifted) > observed * (1 + 1e-12)))


def test_a_sweep_is_replay_and_needs_five_cells(track):
    posterior, times = _decoded(track, 2 * np.arange(25))
    five = np.zeros((25, 6), dtype=int)
    five[np.arange(25), np.arange(25) % 5] = 1
    result = detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=five
    )
    assert result.is_replay
    assert result.score == 0.0  # no shift of a straight line fits better
    four = np.zeros((25, 6), dtype=int)
    four[np.arange(25), np.arange(25) % 4] = 1
    assert not detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=four
    ).is_replay


@pytest.mark.parametrize("seed", range(6))
def test_score_is_the_share_of_shifts_with_a_larger_r_squared(track, seed):
    """The vectorized share equals the one from np.corrcoef and np.roll."""
    rng = np.random.default_rng(seed)
    bins = rng.integers(0, track.n_bins, 21)
    posterior, times = _decoded(track, bins)
    counts = np.eye(6, dtype=int)[np.arange(21) % 6]
    result = detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=counts
    )
    centers = np.ravel(track.to_linear(track.bin_centers))
    assert result.score == pytest.approx(
        _share_by_loop(centers[bins], times), abs=1e-12
    )
    assert result.is_replay == (result.score < 0.05)


def test_a_stationary_event_has_no_verdict(track):
    posterior, times = _decoded(track, np.full(12, 7))
    counts = np.eye(6, dtype=int)[np.arange(12) % 6]
    result = detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=counts
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
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=counts
    )
    assert result.is_replay
    assert result.score == 0.0


def test_a_period_of_the_event_is_not_a_larger_r_squared(track):
    """Shifts by a whole period rebuild the event: rounding must not count them."""
    bins = np.tile([3, 9, 20, 41], 6)
    posterior, times = _decoded(track, bins)
    counts = np.eye(6, dtype=int)[np.arange(24) % 6]
    result = detect_replay(
        track, posterior, times, replay_method=BHATTARAI_2020, spike_counts=counts
    )
    centers = np.ravel(track.to_linear(track.bin_centers))
    assert result.score == pytest.approx(
        _share_by_loop(centers[bins], times), abs=1e-12
    )


def test_recipe_records_the_papers_settings():
    assert BHATTARAI_2020 in PUBLISHED_METHODS
    assert BHATTARAI_2020.decoding["time_bin"] == 0.02
    assert BHATTARAI_2020.decoding["time_step"] == 0.01
    assert BHATTARAI_2020.units == "cm"
