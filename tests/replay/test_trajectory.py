"""Tests for trajectory analysis in neurospatial.replay.trajectory.

These tests verify trajectory fitting and detection functions for analyzing
decoded position sequences, including isotonic regression, linear regression,
and Radon transform detection.
"""

import numpy as np
import pytest
from environment_helpers import renumbered, strips_1d
from replay_helpers import wrapped_bump

from neurospatial import Environment


def _track_for(posterior: np.ndarray) -> Environment:
    """A straight track with one 1 cm bin per posterior column, in order."""
    n_bins = posterior.shape[1]
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [float(n_bins), 0.0]], bin_size=1.0, units="cm"
    )


# =============================================================================
# Result Dataclasses
# =============================================================================


class TestIsotonicFitResult:
    """Test IsotonicFitResult dataclass."""

    def test_isotonic_fit_result_frozen(self):
        """IsotonicFitResult should be immutable (frozen dataclass)."""
        from neurospatial.replay.trajectory import IsotonicFitResult

        result = IsotonicFitResult(
            fitted_positions=np.array([1.0, 2.0]),
            r_squared=0.9,
            mean_map_probability=0.5,
            direction="decreasing",
            residuals=np.array([0.0, 0.0]),
        )

        with pytest.raises((AttributeError, TypeError)):
            result.r_squared = 0.5


class TestLinearFitResult:
    """Test LinearFitResult dataclass."""

    def test_linear_fit_result_frozen(self):
        """LinearFitResult should be immutable (frozen dataclass)."""
        from neurospatial.replay.trajectory import LinearFitResult

        result = LinearFitResult(
            slope=1.0,
            intercept=0.0,
            r_squared=0.9,
            slope_std=0.1,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.slope = 2.0


class TestRadonDetectionResult:
    """Test RadonDetectionResult dataclass."""

    def test_radon_detection_result_frozen(self):
        """RadonDetectionResult should be immutable (frozen dataclass)."""
        from neurospatial.replay.trajectory import RadonDetectionResult

        result = RadonDetectionResult(
            speed=100.0,
            start_position=2.0,
            score=0.7,
            band=15.0,
            sinogram=np.zeros((10, 10)),
            sinogram_speeds=np.zeros(10),
            sinogram_start_positions=np.zeros(10),
        )

        with pytest.raises((AttributeError, TypeError)):
            result.speed = 60.0


class TestFitIsotonicTrajectory:
    """Test fit_isotonic_trajectory function."""

    def test_fit_isotonic_trajectory_fitted_positions_shape(self):
        """Fitted positions should have shape (n_time_bins,)."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 15
        n_bins = 30
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(_track_for(posterior), posterior, times)

        assert result.fitted_positions.shape == (n_time_bins,)

    def test_fit_isotonic_trajectory_residuals_shape(self):
        """Residuals should have shape (n_time_bins,)."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 15
        n_bins = 30
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(_track_for(posterior), posterior, times)

        assert result.residuals.shape == (n_time_bins,)

    def test_fit_isotonic_trajectory_increasing_direction(self):
        """Should detect increasing direction for monotonically increasing data."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with strictly increasing MAP positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(_track_for(posterior), posterior, times)

        assert result.direction == "increasing"
        assert result.r_squared > 0.99  # Should be near-perfect fit

    def test_fit_isotonic_trajectory_decreasing_direction(self):
        """Should detect decreasing direction for monotonically decreasing data."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with strictly decreasing MAP positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(45, 5, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(_track_for(posterior), posterior, times)

        assert result.direction == "decreasing"
        assert result.r_squared > 0.99  # Should be near-perfect fit

    def test_fit_isotonic_trajectory_force_increasing(self):
        """Should force increasing direction when specified."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with decreasing positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(45, 5, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(
            _track_for(posterior), posterior, times, increasing=True
        )

        assert result.direction == "increasing"

    def test_fit_isotonic_trajectory_force_decreasing(self):
        """Should force decreasing direction when specified."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with increasing positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(
            _track_for(posterior), posterior, times, increasing=False
        )

        assert result.direction == "decreasing"

    def test_fit_isotonic_trajectory_fitted_is_monotonic(self):
        """Fitted positions should be monotonic."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 30
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        # Add some noise to positions
        posterior = np.zeros((n_time_bins, n_bins))
        base_positions = np.linspace(5, 45, n_time_bins)
        noisy_positions = base_positions + rng.normal(0, 3, n_time_bins)
        noisy_positions = np.clip(noisy_positions, 0, n_bins - 1).astype(int)
        for t, pos in enumerate(noisy_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(_track_for(posterior), posterior, times)

        # Check monotonicity
        diffs = np.diff(result.fitted_positions)
        if result.direction == "increasing":
            assert np.all(diffs >= 0)
        else:
            assert np.all(diffs <= 0)

    def test_fit_isotonic_trajectory_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        n_time_bins = 10
        n_bins = 20
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match=r"method.*map.*expected"):
            fit_isotonic_trajectory(
                _track_for(posterior), posterior, times, method="invalid"
            )


@pytest.fixture
def simple_env():
    """Create a simple 1D environment for testing."""
    from neurospatial import Environment

    positions = np.linspace(0, 100, 1000).reshape(-1, 1)
    return Environment.from_samples(positions, bin_size=2.0)


class TestFitLinearTrajectory:
    """Test fit_linear_trajectory function."""

    def test_fit_linear_trajectory_reproducible_with_rng_int(self, simple_env):
        """Results should be reproducible with integer rng seed."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with some posterior_entropy
        posterior = np.zeros((n_time_bins, n_bins))
        base_positions = np.linspace(5, n_bins - 5, n_time_bins)
        for t, pos in enumerate(base_positions):
            pos = int(pos)
            pos = np.clip(pos, 1, n_bins - 2)
            posterior[t, pos] = 0.7
            posterior[t, pos - 1] = 0.15
            posterior[t, pos + 1] = 0.15

        result1 = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", rng=42
        )
        result2 = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", rng=42
        )

        assert result1.slope == result2.slope
        assert result1.intercept == result2.intercept

    def test_fit_linear_trajectory_reproducible_with_rng_generator(self, simple_env):
        """Results should be reproducible with Generator rng."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with some posterior_entropy
        posterior = np.zeros((n_time_bins, n_bins))
        base_positions = np.linspace(5, n_bins - 5, n_time_bins)
        for t, pos in enumerate(base_positions):
            pos = int(pos)
            pos = np.clip(pos, 1, n_bins - 2)
            posterior[t, pos] = 0.7
            posterior[t, pos - 1] = 0.15
            posterior[t, pos + 1] = 0.15

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", rng=rng1
        )
        result2 = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", rng=rng2
        )

        assert result1.slope == result2.slope
        assert result1.intercept == result2.intercept

    def test_fit_linear_trajectory_r_squared_range(self, simple_env):
        """R² should be in [0, 1]."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with linear positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, n_bins - 5, n_time_bins).astype(int)
        map_positions = np.clip(map_positions, 0, n_bins - 1)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_linear_trajectory(simple_env, posterior, times, rng=42)

        assert 0.0 <= result.r_squared <= 1.0

    def test_fit_linear_trajectory_perfect_linear_high_r_squared(self, simple_env):
        """Perfect linear data should have R² ≈ 1."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with perfect linear positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, n_bins - 5, n_time_bins).astype(int)
        map_positions = np.clip(map_positions, 0, n_bins - 1)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_linear_trajectory(
            simple_env, posterior, times, method="map", rng=42
        )

        assert result.r_squared > 0.99

    def test_fit_linear_trajectory_non_uniform_times(self, simple_env):
        """Should work with non-uniformly spaced time bins."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins

        # Non-uniform times
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 1, n_time_bins))

        # Create posterior with linear positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, n_bins - 5, n_time_bins).astype(int)
        map_positions = np.clip(map_positions, 0, n_bins - 1)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_linear_trajectory(simple_env, posterior, times, rng=42)

        assert np.isfinite(result.slope)
        assert np.isfinite(result.intercept)

    def test_sampled_fit_of_certain_posterior_has_no_spread(self, simple_env):
        """One-hot rows, equal samples per time bin: every resample is the MAP fit."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 20
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(7)
        posterior = np.zeros((n_time_bins, simple_env.n_bins))
        posterior[np.arange(n_time_bins), rng.integers(0, 50, n_time_bins)] = 1.0

        sampled = fit_linear_trajectory(
            simple_env, posterior, times, n_samples=60, n_resamples=5, rng=0
        )
        map_fit = fit_linear_trajectory(simple_env, posterior, times, method="map")

        assert sampled.slope_std == pytest.approx(0.0, abs=1e-9)
        assert sampled.slope == pytest.approx(map_fit.slope)
        assert sampled.intercept == pytest.approx(map_fit.intercept)
        assert sampled.r_squared == pytest.approx(map_fit.r_squared)

    def test_slope_std_is_the_monte_carlo_spread_of_the_pooled_slope(self):
        """slope_std is the sampling SD of one pooled least-squares slope.

        Each row is a Gaussian of width sigma on 1 cm bins, so a sampled
        position is the true line plus independent noise of variance
        sigma**2 + 1/12 (the bin width). With m samples at each of the
        times t, the pooled slope has standard deviation
        sqrt(sigma**2 + 1/12) / sqrt(m * sum((t - mean(t))**2)).
        """
        from scipy.stats import norm

        from neurospatial.replay.trajectory import fit_linear_trajectory

        sigma = 8.0
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [400.0, 0.0]], bin_size=1.0, units="cm"
        )
        edges = np.arange(0.0, 401.0)
        times = np.arange(20) * 0.01
        center = 100.0 + 500.0 * times
        posterior = np.diff(norm.cdf((edges - center[:, None]) / sigma), axis=1)
        posterior /= posterior.sum(axis=1, keepdims=True)
        per_bin, n_resamples = 20, 400

        result = fit_linear_trajectory(
            env,
            posterior,
            times,
            n_samples=per_bin * len(times),
            n_resamples=n_resamples,
            rng=11,
        )

        expected = np.sqrt(sigma**2 + 1 / 12) / np.sqrt(
            per_bin * np.sum((times - times.mean()) ** 2)
        )
        # The SD of 400 resamples is within about 3.5% of the truth.
        assert result.slope_std == pytest.approx(expected, rel=0.15)
        assert result.slope == pytest.approx(
            500.0, abs=4 * expected / np.sqrt(n_resamples)
        )

    def test_fit_linear_trajectory_invalid_method_raises(self, simple_env):
        """Invalid method should raise ValueError."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        n_time_bins = 10
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        with pytest.raises(ValueError, match=r"method.*sample.*map"):
            fit_linear_trajectory(simple_env, posterior, times, method="invalid")

    def test_fit_linear_trajectory_1d_posterior_raises(self, simple_env):
        """1D posterior should raise ValueError."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        times = np.linspace(0, 1, 10)
        posterior_1d = np.ones(10)  # 1D array - invalid

        with pytest.raises(ValueError, match=r"posterior must be 2D"):
            fit_linear_trajectory(simple_env, posterior_1d, times)


def _gaussian_sweep(bin_size, *, speed=200.0, start=20.0, sigma=3.0, n_time_bins=25):
    """A Gaussian posterior moving at ``speed`` cm/s on a 100 cm track.

    Each row is the mass of a Gaussian of width ``sigma`` cm in each bin, so
    every bin size decodes the same physical posterior.
    """
    from scipy.stats import norm

    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=bin_size, units="cm"
    )
    edges = np.arange(0.0, 100.0 + bin_size / 2, bin_size)
    times = np.arange(n_time_bins) * 0.01
    center = start + speed * times
    posterior = np.diff(norm.cdf((edges - center[:, None]) / sigma), axis=1)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return env, posterior, times


def _speed_step(result):
    return float(np.diff(result.sinogram_speeds)[0])


def _start_step(result):
    return float(np.diff(result.sinogram_start_positions)[0])


class TestDetectTrajectoryRadon:
    """The detected trajectory is a physical speed and start position."""

    N = 50

    @pytest.fixture
    def forward(self):
        """One 1 cm bin per 10 ms time bin: 100 cm/s from the 0.5 cm center."""
        posterior = np.eye(self.N)
        return _track_for(posterior), posterior, np.arange(self.N) * 0.01

    def test_forward_sweep_speed_and_start(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        result = detect_trajectory_radon(*forward)

        # The sweep ends at the end of the track, where a faster grid speed
        # would leave it, so the speed is within one grid step.
        assert abs(result.speed - 100.0) <= _speed_step(result)
        assert abs(result.start_position - 0.5) <= _start_step(result)
        assert result.score == pytest.approx(1.0)

    def test_constant_position_has_zero_speed(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, _, times = forward
        posterior = np.zeros((self.N, self.N))
        posterior[:, 20] = 1.0
        result = detect_trajectory_radon(env, posterior, times)

        assert result.speed == 0.0
        assert abs(result.start_position - 20.5) <= _start_step(result) / 2
        assert result.score == pytest.approx(1.0)

    def test_mirrored_sweep_has_mirrored_speed_and_equal_score(self):
        """Reversing the posterior along the track flips only the direction."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = _gaussian_sweep(2.0, speed=150.0)
        forward = detect_trajectory_radon(env, posterior, times)
        reverse = detect_trajectory_radon(env, posterior[:, ::-1], times)

        assert abs(forward.speed - 150.0) <= _speed_step(forward)
        assert reverse.speed == pytest.approx(-forward.speed)
        assert reverse.start_position == pytest.approx(100.0 - forward.start_position)
        assert reverse.score == pytest.approx(forward.score, abs=1e-12)

    def test_time_reversed_sweep_scores_the_same(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        reverse = detect_trajectory_radon(env, posterior[::-1], times)

        assert abs(reverse.speed + 100.0) <= _speed_step(reverse)
        assert reverse.score == pytest.approx(1.0)

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 4.0])
    def test_speed_does_not_depend_on_bin_size(self, bin_size):
        """A 200 cm/s replay is 200 cm/s on 1, 2 and 4 cm bins."""
        from neurospatial.replay.trajectory import (
            detect_trajectory_radon,
            fit_linear_trajectory,
        )

        env, posterior, times = _gaussian_sweep(bin_size)
        radon = detect_trajectory_radon(env, posterior, times)
        linear = fit_linear_trajectory(env, posterior, times, method="map")

        assert abs(radon.speed - 200.0) <= _speed_step(radon)
        assert radon.speed == pytest.approx(linear.slope, rel=0.02)
        assert abs(radon.start_position - 20.0) <= _start_step(radon)

    @pytest.mark.parametrize("tail", ["uniform", "gaussian"])
    def test_partial_sweep_scores_the_same_reversed_in_time(self, tail):
        """A sweep that fills part of the event may enter as well as leave the track.

        Rows 0-5 sweep from 60 to 98 cm; the rest are spread out. Reversed in
        time, the best line enters the track partway through the event.
        """
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
        )
        times = np.arange(12) * 0.02
        posterior = np.zeros((12, env.n_bins))
        posterior[np.arange(6), 30 + 3 * np.arange(6) + 1] = 1.0
        if tail == "uniform":
            posterior[6:] = 1.0 / env.n_bins
        else:
            posterior[6:] = wrapped_bump(env, np.full(6, 50.0), 25.0)

        forward = detect_trajectory_radon(env, posterior, times)
        position_reversed = detect_trajectory_radon(env, posterior[:, ::-1], times)
        time_reversed = detect_trajectory_radon(env, posterior[::-1], times)

        assert position_reversed.score == pytest.approx(forward.score, abs=1e-12)
        assert time_reversed.score == pytest.approx(forward.score, abs=1e-12)
        assert time_reversed.speed == pytest.approx(-forward.speed)

    def test_uniform_posterior_scores_below_a_sharp_sweep(self, forward):
        """A uniform posterior puts 2 * band / track length within the band."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        uniform = np.full((self.N, self.N), 1.0 / self.N)

        assert detect_trajectory_radon(env, uniform, times).score == pytest.approx(
            2 * 15.0 / self.N
        )
        assert detect_trajectory_radon(env, posterior, times).score == pytest.approx(
            1.0
        )

    def test_dropped_time_bins_keep_the_speed(self, forward):
        """Undecodable time bins can be left out with their times."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        keep = np.ones(self.N, dtype=bool)
        keep[[5, 17, 30]] = False
        result = detect_trajectory_radon(env, posterior[keep], times[keep])

        assert abs(result.speed - 100.0) <= _speed_step(result)
        assert result.score == pytest.approx(1.0)

    def test_search_grid_is_dense_and_symmetric(self, forward):
        """Starts at most band / 5 apart; speeds whose ends differ by one start step."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        result = detect_trajectory_radon(env, posterior, times, max_speed=400.0)

        starts, speeds = result.sinogram_start_positions, result.sinogram_speeds
        np.testing.assert_allclose(np.diff(starts), _start_step(result))
        assert _start_step(result) <= 15.0 / 5
        # Starts are symmetric about the track's center and reach as far
        # beyond its ends as the fastest trajectory travels in the event.
        np.testing.assert_allclose(starts + starts[::-1], self.N)
        reach = speeds[-1] * (times[-1] - times[0])
        assert starts[0] <= -reach + _start_step(result)
        assert np.any(np.isclose(starts, _start_step(result) / 2))
        np.testing.assert_allclose(np.diff(speeds), _speed_step(result))
        assert _speed_step(result) * (times[-1] - times[0]) == pytest.approx(
            _start_step(result)
        )
        np.testing.assert_allclose(speeds, -speeds[::-1])
        assert 0.0 in speeds
        assert 400.0 - _speed_step(result) < speeds[-1] <= 400.0
        assert result.sinogram.shape == (len(speeds), len(starts))

    def test_max_speed_limits_the_detected_speed(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        # The sweep covers 49 cm in 0.49 s; a 20 cm/s line strays beyond the
        # 15 cm band for part of it, while a 48 cm/s line would not.
        result = detect_trajectory_radon(*forward, max_speed=20.0)

        assert result.speed <= 20.0
        assert result.score < 1.0

    def test_default_max_speed_is_30_m_per_s(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        speeds = detect_trajectory_radon(*forward).sinogram_speeds
        assert 3000.0 - np.diff(speeds)[0] < speeds[-1] <= 3000.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_speed": 0.0}, "max_speed > 0"),
            ({"max_speed": np.inf}, "max_speed > 0"),
            ({"band": -1.0}, "band > 0"),
        ],
    )
    def test_search_settings_must_be_positive(self, forward, kwargs, match):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        with pytest.raises(ValueError, match=match):
            detect_trajectory_radon(*forward, **kwargs)

    def test_times_in_milliseconds_are_refused(self, forward):
        """Times 1000x too large make a search far beyond any replay event."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        with pytest.raises(ValueError, match=r"(?s)would score.*in seconds"):
            detect_trajectory_radon(env, posterior, times * 1000)

    def test_times_must_increase(self, forward):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        with pytest.raises(ValueError, match="strictly increasing"):
            detect_trajectory_radon(env, posterior, times[::-1])

    def test_detect_trajectory_radon_empty_posterior(self, forward):
        """An all-zero posterior holds no probability and is refused."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        with pytest.raises(ValueError, match=r"50 of 50 rows.*sums to 0"):
            detect_trajectory_radon(env, np.zeros_like(posterior), times)


# =============================================================================
# Positions along the track
# =============================================================================


class TestFitsReadPositionAlongTrack:
    """Fits use each bin's position along the track, not its bin number."""

    @pytest.fixture
    def sweep(self):
        """20 bins of 5 cm, and a posterior sweeping 5 cm per 10 ms."""
        env = strips_1d(np.arange(0.0, 101.0, 5.0))
        times = np.arange(20) * 0.01
        posterior = np.eye(20)
        order = np.random.default_rng(3).permutation(20)
        return env, renumbered(env, order), posterior, posterior[:, order], times

    @pytest.mark.parametrize("method", ["map", "sample"])
    def test_linear_fit_is_a_speed_that_ignores_bin_numbering(self, sweep, method):
        from neurospatial.replay.trajectory import fit_linear_trajectory

        env, shuffled_env, posterior, shuffled, times = sweep
        original = fit_linear_trajectory(env, posterior, times, method=method, rng=0)
        renumbered = fit_linear_trajectory(
            shuffled_env, shuffled, times, method=method, rng=0
        )

        reverse = fit_linear_trajectory(
            env, posterior[:, ::-1], times, method=method, rng=0
        )

        assert original.slope == pytest.approx(500.0)  # cm/s
        assert original.intercept == pytest.approx(2.5)  # first bin center
        assert reverse.slope == pytest.approx(-500.0)
        assert reverse.intercept == pytest.approx(97.5)  # last bin center
        assert renumbered.slope == pytest.approx(original.slope)
        assert renumbered.r_squared == pytest.approx(original.r_squared)
        assert original.slope_std is None  # one resample, or the MAP fit

    @pytest.mark.parametrize("method", ["map", "expected"])
    def test_isotonic_fit_ignores_bin_numbering(self, sweep, method):
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        env, shuffled_env, posterior, shuffled, times = sweep
        original = fit_isotonic_trajectory(env, posterior, times, method=method)
        renumbered = fit_isotonic_trajectory(
            shuffled_env, shuffled, times, method=method
        )

        np.testing.assert_allclose(original.fitted_positions, 2.5 + 500.0 * times)
        np.testing.assert_allclose(
            renumbered.fitted_positions, original.fitted_positions
        )
        assert renumbered.r_squared == pytest.approx(1.0)

    def test_radon_ignores_bin_numbering(self, sweep):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, shuffled_env, posterior, shuffled, times = sweep
        original = detect_trajectory_radon(env, posterior, times)
        renumbered = detect_trajectory_radon(shuffled_env, shuffled, times)

        assert abs(original.speed - 500.0) <= _speed_step(original) / 2  # cm/s
        assert abs(original.start_position - 2.5) <= _start_step(original) / 2
        np.testing.assert_allclose(renumbered.sinogram, original.sinogram)
        assert renumbered.speed == original.speed

    def test_constant_speed_over_unequal_bins_is_a_straight_line(self):
        """Bins of 2 cm then 8 cm: visiting each center at 1000 cm/s is linear."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        edges = np.r_[np.arange(0.0, 21.0, 2.0), 20.0 + 8.0 * np.arange(1, 11)]
        env = strips_1d(edges)
        centers = (edges[:-1] + edges[1:]) / 2
        times = centers / 1000.0
        posterior = np.eye(env.n_bins)

        result = fit_linear_trajectory(env, posterior, times, method="map")

        assert result.slope == pytest.approx(1000.0)
        assert result.r_squared == pytest.approx(1.0)

    def test_map_is_the_bin_of_highest_density(self):
        """A 4 cm bin holding the most mass is not the MAP over 1 cm bins.

        Each row puts 0.4 in the 4 cm bin (density 0.1 per cm) and 0.3 in a
        1 cm bin that moves 1 cm per 10 ms (density 0.3 per cm), so the MAP
        moves at 100 cm/s although the largest mass never moves.
        """
        from neurospatial.decoding.estimates import map_bin
        from neurospatial.replay.trajectory import (
            fit_isotonic_trajectory,
            fit_linear_trajectory,
        )

        env = strips_1d(np.array([0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
        n_time_bins = 5
        posterior = np.zeros((n_time_bins, env.n_bins))
        posterior[:, 0] = 0.4
        posterior[:, 1:] = 0.3 / 4
        posterior[np.arange(n_time_bins), 1 + np.arange(n_time_bins)] = 0.3
        times = np.arange(n_time_bins) * 0.01
        map_positions = env.bin_centers[map_bin(env, posterior), 0]

        linear = fit_linear_trajectory(env, posterior, times, method="map")
        isotonic = fit_isotonic_trajectory(env, posterior, times, method="map")

        assert linear.slope == pytest.approx(100.0)
        np.testing.assert_allclose(isotonic.fitted_positions, map_positions)

    def test_radon_unchanged_when_a_bin_is_split_into_equal_halves(self):
        """Splitting an 8 cm bin, mass shared by length, leaves the image as is."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        rng = np.random.default_rng(0)
        coarse = strips_1d(np.array([0.0, 2.0, 4.0, 12.0, 14.0]))
        fine = strips_1d(np.array([0.0, 2.0, 4.0, 8.0, 12.0, 14.0]))
        posterior = rng.random((10, 4))
        posterior /= posterior.sum(axis=1, keepdims=True)
        split = np.column_stack(
            [
                posterior[:, :2],
                posterior[:, [2]] / 2,
                posterior[:, [2]] / 2,
                posterior[:, 3:],
            ]
        )

        times = np.arange(10) * 0.01
        np.testing.assert_allclose(
            detect_trajectory_radon(coarse, posterior, times).sinogram,
            detect_trajectory_radon(fine, split, times).sinogram,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        "fit",
        ["fit_linear_trajectory", "fit_isotonic_trajectory", "detect_trajectory_radon"],
    )
    def test_open_arena_is_refused(self, fit):
        import neurospatial.replay.trajectory as trajectory

        arena = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 20, (500, 2)), bin_size=5.0
        )
        posterior = np.full((5, arena.n_bins), 1.0 / arena.n_bins)
        with pytest.raises(ValueError, match="needs a position along a track"):
            getattr(trajectory, fit)(arena, posterior, np.arange(5.0))


# =============================================================================
# Posterior rows must be probability masses
# =============================================================================


class TestPosteriorRowsAreValidated:
    """NaN, negative or unnormalized rows are refused, not read as bin 0."""

    @pytest.fixture
    def sweep(self):
        """A forward sweep, one bin per 10 ms on a 50 cm track of 1 cm bins."""
        posterior = np.eye(50)
        return _track_for(posterior), posterior, np.arange(50) * 0.01

    @staticmethod
    def _call(fit, env, posterior, times):
        import neurospatial.replay.trajectory as trajectory

        return getattr(trajectory, fit)(env, posterior, times)

    @pytest.mark.parametrize(
        "fit",
        ["fit_linear_trajectory", "fit_isotonic_trajectory", "detect_trajectory_radon"],
    )
    def test_float32_rounding_cast_to_float64_is_accepted(self, sweep, fit):
        """Rows a DecodingResult accepts (within 1e-6 of 1) are accepted here."""
        env, _, times = sweep
        smooth = np.exp(-0.5 * ((np.arange(50) - np.arange(50)[:, None]) / 3.0) ** 2)
        rounded = (smooth / smooth.sum(axis=1, keepdims=True)).astype(np.float32)
        assert np.abs(rounded.astype(np.float64).sum(axis=1) - 1).max() > 1e-8
        self._call(fit, env, rounded.astype(np.float64), times)

    @pytest.mark.parametrize(
        "fit",
        ["fit_linear_trajectory", "fit_isotonic_trajectory", "detect_trajectory_radon"],
    )
    def test_unnormalized_rows_report_the_deviation(self, sweep, fit):
        env, posterior, times = sweep
        posterior = posterior * (1 + 2e-5)
        with pytest.raises(ValueError, match=r"\|row sum - 1\| = 2e-05"):
            self._call(fit, env, posterior, times)

    @pytest.mark.parametrize(
        "fit",
        ["fit_linear_trajectory", "fit_isotonic_trajectory", "detect_trajectory_radon"],
    )
    def test_nan_rows_are_refused(self, sweep, fit):
        env, posterior, times = sweep
        posterior = posterior.copy()
        posterior[[10, 20]] = np.nan
        with pytest.raises(ValueError, match=r"2 of 50 rows.*NaN") as excinfo:
            self._call(fit, env, posterior, times)
        message = str(excinfo.value)
        assert "track start" in message  # why
        assert "together with their times" in message  # fix

    @pytest.mark.parametrize(
        "fit",
        ["fit_linear_trajectory", "fit_isotonic_trajectory", "detect_trajectory_radon"],
    )
    def test_rows_missing_mass_are_refused(self, sweep, fit):
        env, posterior, times = sweep
        with pytest.raises(ValueError, match=r"50 of 50 rows.*0\.7") as excinfo:
            self._call(fit, env, 0.7 * posterior, times)
        assert "renormalize" in str(excinfo.value)

    def test_negative_rows_are_refused(self, sweep):
        from neurospatial.replay.trajectory import fit_linear_trajectory

        env, posterior, times = sweep
        posterior = posterior.copy()
        posterior[3, 0] = -0.5
        posterior[3, 1] += 0.5
        with pytest.raises(ValueError, match=r"1 of 50 rows.*negative"):
            fit_linear_trajectory(env, posterior, times)

    def test_rounding_error_in_row_sums_is_accepted(self, sweep):
        """A float32 posterior sums to 1 only to single precision."""
        from neurospatial.replay.trajectory import fit_linear_trajectory

        env, _, times = sweep
        rng = np.random.default_rng(0)
        posterior = rng.dirichlet(np.ones(50), size=50).astype(np.float32)
        assert not np.all(posterior.sum(axis=1) == 1.0)
        result = fit_linear_trajectory(env, posterior, times, rng=1)
        assert np.isfinite(result.slope)


# =============================================================================
# Stationary decodes and degenerate times
# =============================================================================


class TestStationaryDecode:
    """A decode that never moves has no fit quality and no direction."""

    @pytest.fixture
    def stationary(self):
        posterior = np.zeros((20, 50))
        posterior[:, 10] = 1.0
        return _track_for(posterior), posterior, np.arange(20) * 0.01

    @pytest.mark.parametrize("method", ["map", "sample"])
    def test_linear_fit_r_squared_is_undefined(self, stationary, method):
        from neurospatial.replay.trajectory import fit_linear_trajectory

        env, posterior, times = stationary
        result = fit_linear_trajectory(env, posterior, times, method=method, rng=0)

        assert result.slope == pytest.approx(0.0)
        assert np.isnan(result.r_squared)

    @pytest.mark.parametrize("increasing", [None, True, False])
    def test_isotonic_fit_has_no_direction(self, stationary, increasing):
        from neurospatial.replay.trajectory import fit_isotonic_trajectory

        env, posterior, times = stationary
        result = fit_isotonic_trajectory(env, posterior, times, increasing=increasing)

        assert result.direction is None
        assert np.isnan(result.r_squared)
        np.testing.assert_allclose(result.fitted_positions, 10.5)

    def test_identical_times_are_refused(self):
        from neurospatial.replay.trajectory import fit_linear_trajectory

        posterior = np.eye(5, 10)
        with pytest.raises(ValueError, match="identical"):
            fit_linear_trajectory(
                _track_for(posterior), posterior, np.zeros(5), method="map"
            )


class TestArgumentChecks:
    """Times and sample counts are checked before fitting."""

    @pytest.fixture
    def sweep(self):
        posterior = np.eye(10, 20)
        return _track_for(posterior), posterior, np.arange(10) * 0.01

    @pytest.mark.parametrize(
        "fit", ["fit_linear_trajectory", "fit_isotonic_trajectory"]
    )
    def test_times_must_match_posterior_rows(self, sweep, fit):
        import neurospatial.replay.trajectory as trajectory

        env, posterior, _ = sweep
        with pytest.raises(ValueError, match=r"times has 15 entries.*10 rows"):
            getattr(trajectory, fit)(env, posterior, np.arange(15) * 0.01)

    @pytest.mark.parametrize(
        "fit", ["fit_linear_trajectory", "fit_isotonic_trajectory"]
    )
    def test_times_must_be_finite(self, sweep, fit):
        import neurospatial.replay.trajectory as trajectory

        env, posterior, times = sweep
        times = times.copy()
        times[4] = np.nan
        with pytest.raises(ValueError, match=r"times.*NaN or Inf"):
            getattr(trajectory, fit)(env, posterior, times)

    @pytest.mark.parametrize("n_samples", [0, 1])
    def test_sampling_needs_two_samples(self, sweep, n_samples):
        from neurospatial.replay.trajectory import fit_linear_trajectory

        env, posterior, times = sweep
        with pytest.raises(ValueError, match=r"n_samples.*at least 2"):
            fit_linear_trajectory(env, posterior, times, n_samples=n_samples)


def _bumps(n_time_bins: int, n_bins: int, seed: int) -> np.ndarray:
    """A noisy forward sweep: rows are 3-bin-wide Gaussian bumps on 1 cm bins."""
    noise = np.random.default_rng(seed).normal(0, 0.1 * n_bins, n_time_bins)
    centers = np.linspace(0.2, 0.8, n_time_bins) * n_bins + noise
    posterior = np.exp(-0.5 * ((np.arange(n_bins) + 0.5 - centers[:, None]) / 3.0) ** 2)
    return posterior / posterior.sum(axis=1, keepdims=True)


class TestLinearFitFollowsThePapers:
    """Karlsson & Frank (2009), Jadhav-lab code and Wu et al. (2017)."""

    def test_sampled_fit_is_one_least_squares_line_through_the_samples(self):
        """The fit equals scipy's linregress on the same sampled points.

        The draws are rebuilt independently: equal shares per time bin in
        time-bin order, each the first bin whose cumulative mass exceeds the
        uniform times the row total.
        """
        from scipy.stats import linregress

        from neurospatial.replay import fit_linear_trajectory

        posterior = _bumps(8, 40, seed=1)
        env = _track_for(posterior)
        times = np.arange(8) * 0.015
        n_samples = 8 * 37 + 5  # remainder of 5 goes to the earliest bins

        result = fit_linear_trajectory(
            env, posterior, times, n_samples=n_samples, rng=3
        )

        shares = np.full(8, n_samples // 8)
        shares[:5] += 1
        time_bins = np.repeat(np.arange(8), shares)
        uniforms = np.random.default_rng(3).random(n_samples)
        cumulative = np.cumsum(posterior, axis=1)
        drawn = [
            np.argmax(cumulative[t] > u * cumulative[t, -1])
            for t, u in zip(time_bins, uniforms, strict=True)
        ]
        expected = linregress(times[time_bins], env.bin_centers[drawn, 0])

        assert result.slope == pytest.approx(expected.slope, rel=1e-10)
        assert result.intercept == pytest.approx(expected.intercept, rel=1e-10)
        assert result.r_squared == pytest.approx(expected.rvalue**2, rel=1e-10)
        assert result.slope_std is None

    def test_equal_split_gives_the_remainder_to_the_earliest_time_bins(self):
        """One-hot rows: 7 samples over 5 time bins are points 2, 2, 1, 1, 1."""
        from scipy.stats import linregress

        from neurospatial.replay import fit_linear_trajectory

        bins = np.array([3, 9, 4, 15, 12])
        posterior = np.eye(20)[bins]
        env = _track_for(posterior)
        times = np.arange(5) * 0.02

        result = fit_linear_trajectory(env, posterior, times, n_samples=7, rng=0)

        repeats = [2, 2, 1, 1, 1]
        expected = linregress(
            np.repeat(times, repeats), np.repeat(env.bin_centers[bins, 0], repeats)
        )
        assert result.slope == pytest.approx(expected.slope)
        assert result.r_squared == pytest.approx(expected.rvalue**2)

    def test_r_squared_slope_and_intercept_are_means_over_resamples(self):
        """Three resamples are three consecutive one-resample fits, averaged."""
        from neurospatial.replay import fit_linear_trajectory

        posterior = _bumps(10, 50, seed=2)
        env = _track_for(posterior)
        times = np.arange(10) * 0.02

        pooled = fit_linear_trajectory(
            env, posterior, times, n_samples=500, n_resamples=3, rng=5
        )
        shared = np.random.default_rng(5)
        single = [
            fit_linear_trajectory(env, posterior, times, n_samples=500, rng=shared)
            for _ in range(3)
        ]

        slopes = [fit.slope for fit in single]
        assert pooled.r_squared == pytest.approx(
            np.mean([fit.r_squared for fit in single])
        )
        assert pooled.slope == pytest.approx(np.mean(slopes))
        assert pooled.intercept == pytest.approx(
            np.mean([fit.intercept for fit in single])
        )
        assert pooled.slope_std == pytest.approx(np.std(slopes, ddof=1))
        assert len(set(slopes)) == 3  # the resamples differ

    def test_spike_counts_split_samples_in_proportion_to_spikes(self):
        """Each resample splits n_samples multinomially by spike count."""
        from neurospatial.replay.trajectory import _samples_per_time_bin

        spikes = np.array([0, 3, 1, 0, 6])
        split = _samples_per_time_bin(
            1000, 2000, 5, spikes, "spikes", np.random.default_rng(0)
        )

        assert split.shape == (2000, 5)
        np.testing.assert_array_equal(split.sum(axis=1), 1000)
        np.testing.assert_array_equal(split[:, [0, 3]], 0)
        assert len(np.unique(split[:, 1])) > 1  # random, not a fixed share
        # Multinomial mean n p, standard error sqrt(n p (1 - p) / 2000) < 0.4.
        np.testing.assert_allclose(split.mean(axis=0), 1000 * spikes / 10, atol=2.0)

    def test_equal_split_totals_are_exact(self):
        from neurospatial.replay.trajectory import _samples_per_time_bin

        split = _samples_per_time_bin(
            10_005, 3, 7, None, "equal", np.random.default_rng(0)
        )

        np.testing.assert_array_equal(split.sum(axis=1), 10_005)
        np.testing.assert_array_equal(
            split[0], [1430, 1430, 1429, 1429, 1429, 1429, 1429]
        )

    @pytest.mark.parametrize("method", ["sample", "map"])
    def test_time_bins_without_spikes_are_left_out(self, method):
        """Spike-free time bins decode a far-off position that would ruin the line."""
        from neurospatial.replay import fit_linear_trajectory

        bins = np.array([2, 4, 39, 8, 10, 39, 14])
        posterior = np.eye(40)[bins]
        env = _track_for(posterior)
        times = np.arange(7) * 0.02
        spikes = np.array([[1, 0], [2, 1], [0, 0], [0, 1], [3, 0], [0, 0], [1, 1]])

        result = fit_linear_trajectory(
            env,
            posterior,
            times,
            method=method,
            n_samples=1000,
            n_resamples=4,
            spike_counts=spikes,
            rng=0,
        )

        assert result.r_squared == pytest.approx(1.0)
        assert result.slope == pytest.approx(100.0)  # 2 cm per 20 ms

    def test_spike_counts_per_neuron_are_summed(self):
        from neurospatial.replay import fit_linear_trajectory

        posterior = _bumps(6, 30, seed=4)
        env = _track_for(posterior)
        times = np.arange(6) * 0.02
        per_neuron = np.random.default_rng(1).poisson(1.0, size=(6, 4))

        kwargs = {"n_samples": 300, "n_resamples": 3, "rng": 2}
        assert fit_linear_trajectory(
            env, posterior, times, spike_counts=per_neuron, **kwargs
        ) == fit_linear_trajectory(
            env, posterior, times, spike_counts=per_neuron.sum(axis=1), **kwargs
        )

    def test_map_fit_is_least_squares_through_the_map_positions(self):
        """Wu et al. (2017): regress the MAP position on time."""
        from scipy.stats import linregress

        from neurospatial.decoding.estimates import map_bin
        from neurospatial.replay import fit_linear_trajectory

        posterior = _bumps(12, 50, seed=6)
        env = _track_for(posterior)
        times = np.arange(12) * 0.02
        spikes = np.array([1, 0, 2, 3, 0, 1, 1, 0, 4, 1, 2, 1])
        positions = env.bin_centers[map_bin(env, posterior), 0]

        result = fit_linear_trajectory(env, posterior, times, method="map")
        with_spikes = fit_linear_trajectory(
            env, posterior, times, method="map", spike_counts=spikes
        )

        expected = linregress(times, positions)
        assert result.slope == pytest.approx(expected.slope)
        assert result.intercept == pytest.approx(expected.intercept)
        assert result.r_squared == pytest.approx(expected.rvalue**2)
        active = linregress(times[spikes > 0], positions[spikes > 0])
        assert with_spikes.slope == pytest.approx(active.slope)
        assert with_spikes.r_squared == pytest.approx(active.rvalue**2)


class TestLinearFitRefusals:
    """Inputs that leave no line to fit, or are not spike counts, are refused."""

    @pytest.fixture
    def sweep(self):
        posterior = np.eye(10, 20)
        return _track_for(posterior), posterior, np.arange(10) * 0.01

    @pytest.mark.parametrize("method", ["sample", "map"])
    def test_one_time_bin_with_spikes_is_refused(self, sweep, method):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        spikes = np.zeros(10, dtype=int)
        spikes[4] = 3
        with pytest.raises(ValueError, match=r"only 1 of 10 time bins have spikes"):
            fit_linear_trajectory(
                env, posterior, times, method=method, spike_counts=spikes
            )

    @pytest.mark.parametrize(
        ("spikes", "match"),
        [
            (np.ones(9, dtype=int), r"spike_counts has shape \(9,\)"),
            (np.ones((10, 2, 2), dtype=int), r"spike_counts has shape"),
            (np.full(10, -1), r"non-negative whole numbers"),
            (np.full(10, 0.5), r"non-negative whole numbers"),
            (np.full(10, np.nan), r"non-negative whole numbers"),
        ],
    )
    def test_spike_counts_must_be_counts_per_time_bin(self, sweep, spikes, match):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        with pytest.raises(ValueError, match=match):
            fit_linear_trajectory(env, posterior, times, spike_counts=spikes)

    def test_whole_number_floats_are_accepted(self, sweep):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        result = fit_linear_trajectory(
            env, posterior, times, spike_counts=np.full(10, 2.0), rng=0
        )
        assert result.r_squared == pytest.approx(1.0)

    def test_resamples_with_all_samples_at_one_time_are_refused(self, sweep):
        """Two samples split at random over two spiking time bins often coincide."""
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        spikes = np.zeros(10, dtype=int)
        spikes[[2, 7]] = 1
        with pytest.raises(ValueError, match=r"raise n_samples"):
            fit_linear_trajectory(
                env,
                posterior,
                times,
                n_samples=2,
                n_resamples=50,
                spike_counts=spikes,
                allocation="spikes",
                rng=0,
            )

    def test_n_resamples_must_be_positive(self, sweep):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        with pytest.raises(ValueError, match=r"n_resamples of at least 1"):
            fit_linear_trajectory(env, posterior, times, n_resamples=0)

    def test_n_samples_must_be_an_integer(self, sweep):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times = sweep
        with pytest.raises(TypeError, match=r"n_samples must be an integer"):
            fit_linear_trajectory(env, posterior, times, n_samples=1e4)


class TestIsotonicFitFollowsMichon:
    """Michon et al. (2019): weighted isotonic fit of the MAP positions."""

    @staticmethod
    def _weighted_r_squared(y, fitted, w):
        mean = np.sum(w * y) / np.sum(w)
        return 1 - np.sum(w * (y - fitted) ** 2) / np.sum(w * (y - mean) ** 2)

    def test_fit_is_sklearn_weighted_isotonic_with_map_weights(self):
        from sklearn.isotonic import IsotonicRegression

        from neurospatial.decoding.estimates import map_bin
        from neurospatial.replay import fit_isotonic_trajectory

        rng = np.random.default_rng(8)
        n_time_bins, n_bins = 15, 40
        centers = np.linspace(5, 35, n_time_bins) + rng.normal(0, 4, n_time_bins)
        posterior = np.exp(
            -0.5 * ((np.arange(n_bins) + 0.5 - centers[:, None]) / 2.0) ** 2
        )
        posterior += rng.uniform(0, 0.3, posterior.shape)  # uneven MAP mass
        posterior /= posterior.sum(axis=1, keepdims=True)
        env = _track_for(posterior)
        times = np.arange(n_time_bins) * 0.01

        result = fit_isotonic_trajectory(env, posterior, times)

        modes = map_bin(env, posterior)
        x = env.bin_centers[modes, 0]
        w = posterior[np.arange(n_time_bins), modes]
        fits = {
            increasing: IsotonicRegression(increasing=increasing).fit(
                times, x, sample_weight=w
            )
            for increasing in (True, False)
        }
        r_squared = {
            increasing: self._weighted_r_squared(x, fit.predict(times), w)
            for increasing, fit in fits.items()
        }
        best = max(r_squared, key=r_squared.get)

        assert result.direction == ("increasing" if best else "decreasing")
        np.testing.assert_allclose(result.fitted_positions, fits[best].predict(times))
        assert result.r_squared == pytest.approx(r_squared[best])
        assert result.mean_map_probability == pytest.approx(np.mean(w))
        assert result.score == pytest.approx(np.mean(w) * r_squared[best])
        assert result.r_squared < 1  # the noise is not monotonic
        assert result.mean_map_probability < 0.5  # weights matter

    def test_reversed_sweep_is_the_decreasing_fit_with_the_same_score(self):
        from neurospatial.replay import fit_isotonic_trajectory

        posterior = _bumps(12, 50, seed=9)
        env = _track_for(posterior)
        times = np.arange(12) * 0.01

        forward = fit_isotonic_trajectory(env, posterior, times, increasing=True)
        reversed_ = fit_isotonic_trajectory(env, posterior[::-1], times)

        assert reversed_.direction == "decreasing"
        assert reversed_.score == pytest.approx(forward.score)
        np.testing.assert_allclose(
            reversed_.fitted_positions, forward.fitted_positions[::-1]
        )

    def test_expected_positions_share_the_map_weights(self):
        from neurospatial.replay import fit_isotonic_trajectory

        posterior = _bumps(10, 30, seed=10)
        env = _track_for(posterior)
        times = np.arange(10) * 0.01

        by_map = fit_isotonic_trajectory(env, posterior, times)
        by_mean = fit_isotonic_trajectory(env, posterior, times, method="expected")

        assert by_mean.mean_map_probability == by_map.mean_map_probability
        np.testing.assert_allclose(
            by_mean.fitted_positions + by_mean.residuals,
            posterior @ env.bin_centers[:, 0],
        )

    def test_summary_reports_the_score_and_its_parts(self):
        from neurospatial.replay import fit_isotonic_trajectory

        posterior = 0.5 * np.eye(10, 20) + 0.5 * np.eye(10, 20, k=10)
        env = _track_for(posterior)
        result = fit_isotonic_trajectory(env, posterior, np.arange(10) * 0.01)

        assert result.summary() == {
            "score": result.score,
            "r_squared": result.r_squared,
            "mean_map_probability": 0.5,
            "direction": result.direction,
        }
        assert result.score == pytest.approx(0.5 * result.r_squared)


class TestReplayShuffleCalibration:
    """A replay screen against a circular-shift null keeps its error rate.

    Events are decoded from Poisson spikes of 40 place cells (8 cm fields,
    20 Hz peak, 1 Hz baseline) on a 100 cm track, ten 20 ms time bins each.
    The score is the R² of the MAP line fit, and the null circularly shifts
    each time bin's posterior along the track.
    """

    DT = 0.02
    N_TIME_BINS = 10
    FIELD_CENTERS = np.linspace(0.0, 100.0, 40)

    @pytest.fixture(scope="class")
    def track(self):
        return Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
        )

    def _rates(self, positions):
        """Firing rate of each cell at each position, shape (n, n_cells), Hz."""
        offset = positions[:, None] - self.FIELD_CENTERS[None, :]
        return 1.0 + 20.0 * np.exp(-0.5 * (offset / 8.0) ** 2)

    def _decode(self, env, positions, spike_rng):
        from neurospatial.decoding import decode_position

        counts = spike_rng.poisson(self._rates(positions) * self.DT)
        models = self._rates(env.bin_centers[:, 0]).T
        return decode_position(env, counts, models, dt=self.DT).posterior

    def _p_value(self, env, posterior, *, n_shuffles, seed):
        from neurospatial.replay.nulls import shuffle_posterior_circular
        from neurospatial.replay.trajectory import fit_linear_trajectory
        from neurospatial.stats import compute_shuffle_pvalue

        times = np.arange(self.N_TIME_BINS) * self.DT

        def score(rows):
            return fit_linear_trajectory(env, rows, times, method="map").r_squared

        null = [
            score(shuffled)
            for shuffled in shuffle_posterior_circular(
                env, posterior, n_shuffles=n_shuffles, rng=seed
            )
        ]
        return compute_shuffle_pvalue(score(posterior), np.array(null))

    def test_events_without_sequences_are_rarely_significant(self, track):
        stimulus_rng = np.random.default_rng(1)
        spike_rng = np.random.default_rng(2)
        p_values = [
            self._p_value(
                track,
                self._decode(
                    track, stimulus_rng.uniform(0, 100, self.N_TIME_BINS), spike_rng
                ),
                n_shuffles=99,
                seed=seed,
            )
            for seed in range(200)
        ]

        assert np.mean(np.array(p_values) <= 0.05) <= 0.08

    def test_planted_sweep_is_significant(self, track):
        posterior = self._decode(
            track, np.linspace(10.0, 90.0, self.N_TIME_BINS), np.random.default_rng(2)
        )

        assert self._p_value(track, posterior, n_shuffles=199, seed=999) < 0.01


# =============================================================================
# Replay score (Davidson et al. 2009) and its significance
# =============================================================================


def _band_mass(starts, stops, row, position, band):
    """Mass of one posterior row within ``band`` of ``position``, by bin overlap."""
    overlap = np.clip(
        np.minimum(stops, position + band) - np.maximum(starts, position - band),
        0.0,
        None,
    )
    return float(row @ (overlap / (stops - starts)))


class TestReplayScore:
    """The score is Davidson et al.'s equation 5 with their off-track rule."""

    def test_sinogram_is_mean_band_mass_with_median_off_track(self):
        """Every line's score matches a direct evaluation on unequal bins."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        edges = np.r_[0.0, np.cumsum(np.random.default_rng(0).uniform(2, 6, 25))]
        env = strips_1d(edges)
        starts, stops = edges[:-1], edges[1:]
        rng = np.random.default_rng(1)
        posterior = rng.dirichlet(np.full(env.n_bins, 0.3), size=12)
        times = np.sort(rng.uniform(0, 0.3, 12))
        band = 7.0

        result = detect_trajectory_radon(
            env, posterior, times, band=band, max_speed=400.0
        )

        starts_grid = result.sinogram_start_positions
        on_track = starts_grid[(starts_grid >= edges[0]) & (starts_grid <= edges[-1])]
        off_track_mass = np.array(
            [
                np.median([_band_mass(starts, stops, row, x, band) for x in on_track])
                for row in posterior
            ]
        )
        checked_off_track = False
        for i in range(0, len(result.sinogram_speeds), 7):
            for j in range(0, len(starts_grid), 5):
                path = starts_grid[j] + result.sinogram_speeds[i] * (times - times[0])
                off = (path < edges[0]) | (path > edges[-1])
                if off.all():
                    # Never on the track: not a candidate.
                    assert np.isnan(result.sinogram[i, j])
                    continue
                checked_off_track |= bool(off.any())
                expected = np.mean(
                    [
                        off_track_mass[k]
                        if off[k]
                        else _band_mass(starts, stops, posterior[k], path[k], band)
                        for k in range(len(times))
                    ]
                )
                assert result.sinogram[i, j] == pytest.approx(expected, abs=1e-12)
        assert checked_off_track

    def test_score_does_not_depend_on_bin_size(self):
        """A spread-out posterior scores the same on 1, 2 and 4 cm bins."""
        from neurospatial.replay.trajectory import detect_trajectory_radon

        scores = [
            detect_trajectory_radon(*_gaussian_sweep(bin_size, sigma=8.0)).score
            for bin_size in (1.0, 2.0, 4.0)
        ]

        np.testing.assert_allclose(scores, scores[0], atol=0.01)
        assert 0.5 < scores[0] < 1.0

    def test_wider_band_holds_more_mass(self):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = _gaussian_sweep(1.0, sigma=8.0)
        narrow = detect_trajectory_radon(env, posterior, times, band=5.0)
        wide = detect_trajectory_radon(env, posterior, times, band=25.0)

        assert narrow.band == 5.0
        assert narrow.score < wide.score

    def test_default_band_is_15_cm_in_the_environment_units(self):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [1.0, 0.0]], bin_size=0.02, units="m"
        )
        posterior = np.eye(env.n_bins)
        result = detect_trajectory_radon(env, posterior, np.arange(env.n_bins) * 0.01)

        assert result.band == pytest.approx(0.15)
        assert abs(result.speed - 2.0) <= _speed_step(result) / 2  # m/s
        assert result.sinogram_speeds[-1] <= 30.0  # the 30 m/s default

    def test_band_must_be_a_positive_length(self):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env, posterior, times = _gaussian_sweep(2.0)
        with pytest.raises(ValueError, match="band > 0"):
            detect_trajectory_radon(env, posterior, times, band=0.0)

    def test_default_band_needs_metric_units(self):
        from neurospatial.replay.trajectory import detect_trajectory_radon

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [50.0, 0.0]], bin_size=1.0, units="px"
        )
        posterior = np.eye(env.n_bins)
        with pytest.raises(ValueError, match=r"needs band.*15 cm"):
            detect_trajectory_radon(env, posterior, np.arange(env.n_bins) * 0.01)


class TestSampledFitAllocation:
    """Karlsson & Frank split samples equally; the Jadhav-lab code by spikes."""

    @pytest.fixture
    def mixed_event(self):
        """Alternating sharp and wide posteriors with unequal spike counts."""
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=5.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        centers = np.linspace(10.0, 110.0, 12)
        widths = np.where(np.arange(12) % 2 == 0, 4.0, 20.0)
        posterior = np.exp(-0.5 * ((x - centers[:, None]) / widths[:, None]) ** 2)
        posterior /= posterior.sum(axis=1, keepdims=True)
        spikes = np.array([1, 0, 9, 1, 0, 8, 1, 1, 7, 0, 1, 6])
        return env, posterior, np.arange(12) * 0.02, spikes

    def test_sampled_r_squared_converges_to_the_weighted_correlation(self, mixed_event):
        """Pooled R² tends to the squared weighted correlation of the sampled bins."""
        from neurospatial.replay import fit_linear_trajectory, weighted_correlation

        env, posterior, times, spikes = mixed_event
        keep = spikes > 0
        equal = weighted_correlation(env, posterior[keep], times[keep]).score ** 2
        by_spikes = (
            weighted_correlation(
                env,
                posterior[keep] * spikes[keep, None],
                times[keep],
                row_normalization="none",
            ).score
            ** 2
        )
        assert abs(equal - by_spikes) > 0.005  # the two estimands differ

        fit_equal = fit_linear_trajectory(
            env, posterior, times, n_samples=400_000, spike_counts=spikes, rng=0
        )
        fit_spikes = fit_linear_trajectory(
            env,
            posterior,
            times,
            n_samples=400_000,
            spike_counts=spikes,
            allocation="spikes",
            rng=0,
        )
        assert fit_equal.r_squared == pytest.approx(equal, abs=0.003)
        assert fit_spikes.r_squared == pytest.approx(by_spikes, abs=0.003)

    def test_equal_split_skips_time_bins_without_spikes(self, mixed_event):
        from neurospatial.replay.trajectory import _samples_per_time_bin

        _, _, _, spikes = mixed_event
        split = _samples_per_time_bin(
            90, 1, len(spikes), spikes, "equal", np.random.default_rng(0)
        )[0]
        assert np.all(split[spikes == 0] == 0)
        assert split.sum() == 90
        assert set(split[spikes > 0]) == {10}

    def test_spike_allocation_needs_spike_counts(self, mixed_event):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times, _ = mixed_event
        with pytest.raises(ValueError, match="needs spike_counts"):
            fit_linear_trajectory(env, posterior, times, allocation="spikes")

    def test_fewer_samples_than_sampled_time_bins_is_refused(self, mixed_event):
        from neurospatial.replay import fit_linear_trajectory

        env, posterior, times, spikes = mixed_event
        with pytest.raises(ValueError, match="fewer than the 9 time bins"):
            fit_linear_trajectory(
                env, posterior, times, n_samples=5, spike_counts=spikes
            )

    def test_blocked_draws_match_one_block(self, mixed_event, monkeypatch):
        """Drawing and fitting in blocks gives the same result as one block."""
        import neurospatial.replay.trajectory as trajectory

        env, posterior, times, _ = mixed_event
        whole = trajectory.fit_linear_trajectory(
            env, posterior, times, n_samples=1200, n_resamples=6, rng=3
        )
        monkeypatch.setattr(trajectory, "_SAMPLE_BLOCK", 2400)  # 2 resamples a block
        blocked = trajectory.fit_linear_trajectory(
            env, posterior, times, n_samples=1200, n_resamples=6, rng=3
        )
        assert blocked.r_squared == whole.r_squared
        assert blocked.slope_std == whole.slope_std
