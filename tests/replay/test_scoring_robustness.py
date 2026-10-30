"""Replay scoring of events whose score or shuffles are undefined."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_position
from neurospatial.replay import (
    fit_linear_trajectory,
    replay_significance,
    weighted_correlation,
)
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


def _event(env, rates, counts, t0=0.0, prior=None):
    counts = np.asarray(counts)
    times = t0 + DT * (np.arange(len(counts)) + 0.5)
    return decode_position(env, counts, rates, dt=DT, times=times, prior=prior)


def _sweep_counts(n_time=10):
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    for t in range(n_time):
        cell = int(t * N_CELLS / n_time)
        counts[t, cell] = 2
        counts[t, min(cell + 1, N_CELLS - 1)] = 1
    return counts


def _stationary_counts(n_time=10):
    """Five neighboring cells fire the same way in every time bin.

    The MAP position never changes, so its R² is undefined, while the event
    has enough active cells for the methods' cell criteria.
    """
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    counts[:, 8:13] = [1, 2, 4, 2, 1]
    return counts


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


class TestUndefinedEventScore:
    def test_replay_significance_names_the_score_that_is_undefined(self, track):
        env, rates = track
        event = _event(env, rates, _stationary_counts())
        with pytest.raises(ValueError, match="fit_linear_trajectory is undefined"):
            replay_significance(
                env,
                event.posterior,
                event.times,
                score=fit_linear_trajectory,
                score_options={"method": "map"},
                nulls="time_bin_permutation",
                n_shuffles=10,
                rng=0,
            )


def _parity_score(env, posterior, times):
    """A score that is undefined when the first row peaks at an even bin."""
    first = int(np.argmax(posterior[0]))
    score = np.nan if first % 2 == 0 else float(np.argmax(posterior[-1]) - first)
    return SimpleNamespace(score=score, summary=lambda: {"score": score})


class TestUndefinedShuffles:
    @pytest.fixture
    def posterior(self, track):
        env, _ = track
        return np.eye(env.n_bins)[[1, 3, 5, 7, 9, 11, 13, 15]]

    def test_counts_used_shuffles_and_names_the_null(self, track, posterior):
        env, _ = track
        times = np.arange(len(posterior)) * DT
        with pytest.warns(UserWarning, match="column_cycle shuffles have an undefined"):
            result = replay_significance(
                env,
                posterior,
                times,
                score=_parity_score,
                nulls="column_cycle",
                n_shuffles=200,
                rng=0,
            )
        test = result.shuffles["column_cycle"]
        finite = np.isfinite(test.null_scores)
        used = result.n_shuffles_used["column_cycle"]
        assert 0 < used < 200 and used == finite.sum()
        assert result.summary()["column_cycle_n_shuffles_used"] == used
        assert np.isfinite(test.z_score)
        observed = test.observed_score
        expected = (np.sum(test.null_scores[finite] >= observed) + 1) / (used + 1)
        assert test.p_value == pytest.approx(expected)


class TestCustomScoreSignature:
    def test_unreadable_signature_is_refused(self, track):
        env, _ = track

        class Unreadable:
            __signature__ = "not a signature"

            def __call__(self, env, posterior, times):
                return SimpleNamespace(score=1.0)

        posterior = np.eye(env.n_bins)[[1, 3, 5, 7]]
        with pytest.raises(TypeError, match="cannot read the signature"):
            replay_significance(
                env,
                posterior,
                np.arange(4) * DT,
                score=Unreadable(),
                nulls="column_cycle",
                n_shuffles=5,
                rng=0,
            )
