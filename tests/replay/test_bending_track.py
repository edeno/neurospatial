"""Replay scores on a bending track use positions along the track.

On a U-shaped track neither x nor y orders the bins, so a score that used
a coordinate instead of the distance along the track would see a sweep
around the U as standing still on each arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    rank_order_correlation,
    replay_significance,
    weighted_correlation,
)

NODES = np.array([[0.0, 0.0], [0.0, 100.0], [50.0, 100.0], [50.0, 0.0]])
DT = 0.02


@pytest.fixture(scope="module")
def u_track():
    return Environment.linear_track(node_positions=NODES, bin_size=10.0, units="cm")


def _along(points):
    """Distance along the U from its first node, by projection onto each piece."""
    starts, stops = NODES[:-1], NODES[1:]
    lengths = np.linalg.norm(stops - starts, axis=1)
    offsets = np.r_[0.0, np.cumsum(lengths)[:-1]]
    along = []
    for point in points:
        for start, stop, length, offset in zip(
            starts, stops, lengths, offsets, strict=True
        ):
            t = np.dot(point - start, stop - start) / length**2
            if -1e-9 <= t <= 1 + 1e-9 and np.allclose(
                start + t * (stop - start), point
            ):
                along.append(offset + t * length)
                break
    return np.asarray(along)


def _double_sum_correlation(posterior, times, positions):
    """Weighted Pearson correlation written as sums over (time, position) pairs."""
    w = posterior / posterior.sum(axis=1, keepdims=True)
    w = w / w.sum()
    t_mean = sum(
        w[i, j] * times[i] for i in range(w.shape[0]) for j in range(w.shape[1])
    )
    x_mean = sum(
        w[i, j] * positions[j] for i in range(w.shape[0]) for j in range(w.shape[1])
    )
    cov = var_t = var_x = 0.0
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            cov += w[i, j] * (times[i] - t_mean) * (positions[j] - x_mean)
            var_t += w[i, j] * (times[i] - t_mean) ** 2
            var_x += w[i, j] * (positions[j] - x_mean) ** 2
    return cov / np.sqrt(var_t * var_x)


@pytest.fixture(scope="module")
def sweep_around_the_u(u_track):
    """A noisy sweep from one end of the U to the other."""
    rng = np.random.default_rng(0)
    n = u_track.n_bins
    posterior = 30.0 * np.eye(n) + rng.uniform(0.0, 1.0, (n, n))
    return posterior / posterior.sum(axis=1, keepdims=True), np.arange(n) * DT


def test_the_bins_run_around_the_u(u_track):
    np.testing.assert_allclose(_along(u_track.bin_centers), 5.0 + 10.0 * np.arange(25))


def test_a_clean_sweep_has_correlation_one(u_track):
    times = np.arange(u_track.n_bins) * DT
    assert weighted_correlation(u_track, np.eye(u_track.n_bins), times).score == 1.0


def test_weighted_correlation_uses_distance_along_the_track(
    u_track, sweep_around_the_u
):
    posterior, times = sweep_around_the_u
    expected = _double_sum_correlation(posterior, times, _along(u_track.bin_centers))
    assert expected > 0.5
    result = weighted_correlation(u_track, posterior, times)
    assert result.score == pytest.approx(expected, abs=1e-10)
    tested = replay_significance(
        u_track,
        posterior,
        times,
        score=weighted_correlation,
        nulls="column_cycle",
        n_shuffles=9,
        rng=0,
    )
    assert tested.observed.score == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("field_order", ["peak", "center"])
def test_rank_order_uses_distance_along_the_track(u_track, field_order):
    """Cells with fields around the U, firing in that order: rho is 1."""
    n = u_track.n_bins
    rates = 1.0 + 19.0 * np.eye(n)
    spikes = [np.array([1.0 + 0.005 * cell]) for cell in range(n)]
    result = rank_order_correlation(
        u_track,
        spikes,
        (0.99, 1.2),
        rates,
        spike_statistic="first",
        field_order=field_order,
    )
    assert result.n_active_cells == n
    assert result.score == pytest.approx(1.0)
