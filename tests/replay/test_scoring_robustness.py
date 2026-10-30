"""Replay scoring of events whose score or shuffles are undefined."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import weighted_correlation
from neurospatial.replay.correlation import _weighted_correlations

DT = 0.02
N_CELLS = 20


@pytest.fixture(scope="module")
def track():
    """A 100 cm track with 20 place cells evenly along it."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    centers = np.linspace(2.5, 97.5, N_CELLS)[:, None]
    rates = 0.5 + 20.0 * np.exp(-0.5 * ((x - centers) / 6.0) ** 2)
    return env, rates


class TestWeightedCorrelationDegeneracy:
    @pytest.mark.parametrize("t0", [0.0, 1.0, 37.3, 3600.123])
    @pytest.mark.parametrize("row_normalization", ["per_time_bin", "none"])
    def test_mass_in_one_time_bin_is_undefined(self, track, t0, row_normalization):
        env, _ = track
        rng = np.random.default_rng(1)
        times = t0 + DT * (np.arange(10) + 0.5)
        for row in range(10):
            posterior = np.zeros((10, env.n_bins))
            posterior[row] = rng.random(env.n_bins)
            score = weighted_correlation(
                env, posterior, times, row_normalization=row_normalization
            ).score
            assert np.isnan(score), (row, score)

    @pytest.mark.parametrize("t0", [0.0, 3600.123])
    @pytest.mark.parametrize("row_normalization", ["per_time_bin", "none"])
    def test_mass_at_one_position_is_undefined(self, track, t0, row_normalization):
        env, _ = track
        rng = np.random.default_rng(2)
        times = t0 + DT * (np.arange(10) + 0.5)
        for column in range(env.n_bins):
            posterior = np.zeros((10, env.n_bins))
            posterior[:, column] = rng.random(10)
            score = weighted_correlation(
                env, posterior, times, row_normalization=row_normalization
            ).score
            assert np.isnan(score), (column, score)

    def test_small_real_variation_is_kept(self, track):
        env, _ = track
        posterior = np.zeros((10, env.n_bins))
        posterior[4, 2] = 1.0
        posterior[5, 3] = 1e-6  # little mass, but a real second time bin
        times = 3600.0 + DT * np.arange(10)
        assert weighted_correlation(
            env, posterior, times, row_normalization="none"
        ).score == pytest.approx(1.0)

    def test_batch_matches_single_scores(self, track):
        env, _ = track
        rng = np.random.default_rng(3)
        times = 100.0 + DT * np.arange(8)
        batch = rng.random((6, 8, env.n_bins))
        batch[0] = 0.0
        batch[0, 2] = rng.random(env.n_bins)  # one time bin: undefined
        positions = env.bin_centers[:, 0]
        batched = _weighted_correlations(batch, times, positions)
        single = [
            weighted_correlation(env, weights, times, row_normalization="none").score
            for weights in batch
        ]
        np.testing.assert_array_equal(batched, single)
        assert np.isnan(batched[0]) and np.isfinite(batched[1:]).all()
