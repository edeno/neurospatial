"""Tests for regression_line_score, the band mass under the MAP regression line."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.replay import (
    RegressionLineScoreResult,
    regression_line_score,
    replay_significance,
)


def _track(length: float = 30.0, bin_size: float = 3.0, units: str = "cm"):
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [length, 0.0]], bin_size=bin_size, units=units
    )


def _brute_force(env, posterior, times, band, wrap_band):
    """The score from its definition, one time bin and one bin at a time."""
    edges = env.bin_centers[:, 0]
    half = np.diff(edges).mean() / 2
    starts, stops = edges - half, edges + half
    loop = stops.max() - starts.min()
    rows = [k for k in range(len(posterior)) if posterior[k].sum() > 0]
    map_positions = [edges[np.argmax(posterior[k])] for k in rows]
    slope, intercept = np.polyfit(times[rows], map_positions, 1)
    masses = []
    for k in rows:
        center = slope * times[k] + intercept
        shifts = (-loop, 0.0, loop) if wrap_band else (0.0,)
        mass = 0.0
        for j in range(env.n_bins):
            for shift in shifts:
                low = max(starts[j], center - band + shift)
                high = min(stops[j], center + band + shift)
                if high > low:
                    mass += posterior[k, j] * (high - low) / (stops[j] - starts[j])
        masses.append(min(mass, 1.0))
    return float(np.mean(masses)), slope, intercept


def _hand_posterior():
    """Four time bins on a 10-bin track, the third without spikes.

    The MAP bins 1, 3 and 9 do not lie on a line, so the fitted line passes
    between bin centers and its band cuts bins partway; the last row's band
    reaches past the track's end.
    """
    posterior = np.zeros((4, 10))
    posterior[0, [0, 1, 2, 7]] = [0.2, 0.5, 0.2, 0.1]
    posterior[1, [2, 3, 4, 9]] = [0.25, 0.4, 0.25, 0.1]
    posterior[3, [0, 1, 7, 8, 9]] = [0.15, 0.1, 0.2, 0.25, 0.3]
    return posterior, np.array([0.0, 0.02, 0.04, 0.06])


class TestScoreDefinition:
    def test_matches_hand_calculation(self):
        env = _track()
        posterior, times = _hand_posterior()
        result = regression_line_score(env, posterior, times, band=4.0)
        assert isinstance(result, RegressionLineScoreResult)
        expected, slope, intercept = _brute_force(env, posterior, times, 4.0, False)
        assert result.score == pytest.approx(expected, abs=1e-12)
        assert result.slope == pytest.approx(slope, rel=1e-12)
        assert result.intercept == pytest.approx(intercept, rel=1e-12)
        rows = [0, 1, 3]
        map_positions = [4.5, 10.5, 28.5]  # bins 1, 3 and 9 of 3 cm
        r = np.corrcoef(times[rows], map_positions)[0, 1]
        assert result.r_squared == pytest.approx(r**2, rel=1e-12)

    def test_wrapping_takes_band_past_the_end_from_the_other_end(self):
        env = _track()
        posterior, times = _hand_posterior()
        opened = regression_line_score(env, posterior, times, band=4.0)
        wrapped = regression_line_score(env, posterior, times, band=4.0, wrap_band=True)
        expected, _, _ = _brute_force(env, posterior, times, 4.0, True)
        assert wrapped.score == pytest.approx(expected, abs=1e-12)
        # The last row's band reaches past the track's end, where wrapping
        # takes part of bin 0, so the wrapped score is higher.
        assert wrapped.score > opened.score
        assert wrapped.wrap_band and not opened.wrap_band

    def test_band_longer_than_the_track_holds_every_row_when_wrapped(self):
        env = _track()
        posterior, times = _hand_posterior()
        assert regression_line_score(
            env, posterior, times, band=40.0, wrap_band=True
        ).score == pytest.approx(1.0)

    def test_nelpy_bins_agree_for_lines_through_bin_centers(self):
        """With the line on bin centers, the band is the 2w + 1 bins nelpy sums.

        nelpy's trajectory_score_array (w=3, normalize=True) rounds the line
        to a bin and sums bins line - 3 to line + 3, wrapping around the
        track; 10.5 cm on 3 cm bins is the same band.
        """
        env = _track(length=60.0)  # 20 bins of 3 cm
        rng = np.random.default_rng(0)
        posterior = rng.dirichlet(np.full(env.n_bins, 0.2), size=6)
        map_bins = 14 + np.arange(6)  # one bin per time bin, up to the end
        posterior[np.arange(6), map_bins] += 5.0
        posterior /= posterior.sum(axis=1, keepdims=True)
        times = np.arange(6) * 0.02
        result = regression_line_score(env, posterior, times, band=10.5, wrap_band=True)
        nelpy = np.mean(
            [
                posterior[k, (map_bins[k] + np.arange(-3, 4)) % env.n_bins].sum()
                for k in range(6)
            ]
        )
        assert result.score == pytest.approx(nelpy, abs=1e-12)
        opened = regression_line_score(env, posterior, times, band=10.5)
        assert opened.score < result.score  # the last rows' bands leave the track

    def test_time_bins_without_mass_are_left_out(self):
        env = _track()
        posterior, times = _hand_posterior()
        kept = [0, 1, 3]
        with_zero = regression_line_score(env, posterior, times, band=4.0)
        without = regression_line_score(env, posterior[kept], times[kept], band=4.0)
        assert with_zero.score == without.score
        assert with_zero.slope == without.slope

    def test_fewer_than_two_time_bins_with_mass_has_no_score(self):
        env = _track()
        posterior = np.zeros((3, env.n_bins))
        posterior[1, 4] = 1.0
        result = regression_line_score(env, posterior, np.arange(3) * 0.02, band=4.0)
        assert np.isnan(result.score)
        assert np.isnan(result.slope)

    def test_stationary_decode_scores_its_mass_with_undefined_r_squared(self):
        env = _track()
        posterior = np.zeros((5, env.n_bins))
        posterior[:, 4] = 0.8
        posterior[:, 9] = 0.2
        result = regression_line_score(env, posterior, np.arange(5) * 0.02, band=4.0)
        assert result.score == pytest.approx(0.8)
        assert result.slope == pytest.approx(0.0, abs=1e-9)
        assert np.isnan(result.r_squared)

    def test_bin_numbering_does_not_matter(self):
        """The score reads positions along the track, not bin indices."""
        env = _track()
        posterior, times = _hand_posterior()
        flipped = Environment.linear_track(
            endpoints=[[30.0, 0.0], [0.0, 0.0]], bin_size=3.0, units="cm"
        )
        order = np.argsort(flipped.bin_centers[:, 0])  # index of each track bin
        renumbered = np.zeros_like(posterior)
        renumbered[:, order] = posterior
        assert (
            regression_line_score(flipped, renumbered, times, band=4.0).score
            == regression_line_score(env, posterior, times, band=4.0).score
        )


class TestScoreInputs:
    def test_default_band_is_21_cm_full_width(self):
        env = _track()
        posterior, times = _hand_posterior()
        assert regression_line_score(env, posterior, times).band == 10.5
        in_meters = _track(length=0.3, bin_size=0.03, units="m")
        assert regression_line_score(in_meters, posterior, times).band == (
            pytest.approx(0.105)
        )

    def test_default_band_needs_metric_units(self):
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [30.0, 0.0]], bin_size=3.0
        )
        posterior, times = _hand_posterior()
        with pytest.raises(ValueError, match="band"):
            regression_line_score(env, posterior, times)
        assert np.isfinite(regression_line_score(env, posterior, times, band=4.0).score)

    @pytest.mark.parametrize("band", [0.0, -1.0, np.nan, np.inf])
    def test_band_must_be_positive(self, band):
        env = _track()
        posterior, times = _hand_posterior()
        with pytest.raises(ValueError, match="band > 0"):
            regression_line_score(env, posterior, times, band=band)

    def test_wrap_band_must_be_a_bool(self):
        env = _track()
        posterior, times = _hand_posterior()
        with pytest.raises(TypeError, match="wrap_band"):
            regression_line_score(env, posterior, times, wrap_band="yes")  # type: ignore[arg-type]

    def test_rows_must_be_a_mass_or_zero(self):
        env = _track()
        posterior, times = _hand_posterior()
        posterior[2, 0] = 0.5
        with pytest.raises(ValueError, match="probability mass"):
            regression_line_score(env, posterior, times)


class TestBatchedShuffles:
    def test_batched_scores_equal_scoring_each_shuffle(self):
        from neurospatial.replay.significance import (
            _RegressionLineScorer,
            _scorer_for,
        )

        env = _track(length=60.0)
        times = np.arange(9) * 0.02
        rng = np.random.default_rng(4)
        posteriors = []
        for _ in range(300):  # more than one batch of 256
            posterior = rng.dirichlet(np.full(env.n_bins, 0.3), size=9)
            posterior[rng.random(9) < 0.2] = 0.0  # time bins without spikes
            posteriors.append(posterior)
        for options in ({}, {"band": 4.0}, {"band": 7.5, "wrap_band": True}):
            scorer = _scorer_for(
                regression_line_score, env, posteriors[0], times, dict(options)
            )
            assert isinstance(scorer, _RegressionLineScorer)
            batched = scorer.scores(iter(posteriors), len(posteriors))
            one_by_one = [
                regression_line_score(env, p, times, **options).score
                for p in posteriors
            ]
            assert_allclose(batched, one_by_one, rtol=0, atol=0, equal_nan=True)
            assert scorer.observe().score == one_by_one[0]


class TestSignificance:
    def test_sweep_beats_column_cycles_and_spread_does_not(self):
        env = _track(length=150.0)
        times = np.arange(12) * 0.02
        sweep = np.full((12, env.n_bins), 0.1 / (env.n_bins - 1))
        sweep[np.arange(12), 8 + 2 * np.arange(12)] = 0.9
        common = {
            "score": regression_line_score,
            "nulls": "column_cycle",
            "n_shuffles": 199,
            "rng": 0,
        }
        assert replay_significance(env, sweep, times, **common).p_value == (
            pytest.approx(1 / 200)
        )
        rng = np.random.default_rng(1)
        scattered = np.full((12, env.n_bins), 0.1 / (env.n_bins - 1))
        scattered[np.arange(12), rng.integers(0, env.n_bins, 12)] = 0.9
        assert replay_significance(env, scattered, times, **common).p_value > 0.05
