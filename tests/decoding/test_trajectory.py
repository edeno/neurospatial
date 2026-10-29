"""Tests for trajectory analysis in neurospatial.decoding.trajectory.

These tests verify trajectory fitting and detection functions for analyzing
decoded position sequences, including isotonic regression, linear regression,
and Radon transform detection.
"""

import numpy as np
import pytest
from environment_helpers import renumbered, strips_1d

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
        from neurospatial.decoding.trajectory import IsotonicFitResult

        result = IsotonicFitResult(
            fitted_positions=np.array([1.0, 2.0]),
            r_squared=0.9,
            direction="decreasing",
            residuals=np.array([0.0, 0.0]),
        )

        with pytest.raises((AttributeError, TypeError)):
            result.r_squared = 0.5


class TestLinearFitResult:
    """Test LinearFitResult dataclass."""

    def test_linear_fit_result_frozen(self):
        """LinearFitResult should be immutable (frozen dataclass)."""
        from neurospatial.decoding.trajectory import LinearFitResult

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
        from neurospatial.decoding.trajectory import RadonDetectionResult

        result = RadonDetectionResult(
            speed=100.0,
            start_position=2.0,
            score=0.7,
            angle=0.5,
            sinogram=np.zeros((10, 10)),
            sinogram_angles=np.zeros(10),
            sinogram_positions=np.zeros(10),
            time_bin_size=0.01,
            column_width=1.0,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.speed = 60.0


class TestFitIsotonicTrajectory:
    """Test fit_isotonic_trajectory function."""

    def test_fit_isotonic_trajectory_fitted_positions_shape(self):
        """Fitted positions should have shape (n_time_bins,)."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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

    def test_fit_linear_trajectory_method_map(self, simple_env):
        """method='map' should use argmax positions for fitting."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with clear MAP positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, n_bins - 5, n_time_bins).astype(int)
        map_positions = np.clip(map_positions, 0, n_bins - 1)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_linear_trajectory(simple_env, posterior, times, method="map")

        assert result.slope_std is None  # No posterior_entropy for method="map"
        assert result.slope > 0  # Increasing positions

    def test_fit_linear_trajectory_method_sample(self, simple_env):
        """method='sample' should provide posterior_entropy via Monte Carlo."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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

        result = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", rng=42
        )

        assert result.slope_std is not None  # Should have posterior_entropy
        assert result.slope_std >= 0  # Non-negative std

    def test_fit_linear_trajectory_reproducible_with_rng_int(self, simple_env):
        """Results should be reproducible with integer rng seed."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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

    def test_fit_linear_trajectory_negative_slope(self, simple_env):
        """Decreasing positions should yield negative slope."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with decreasing positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(n_bins - 5, 5, n_time_bins).astype(int)
        map_positions = np.clip(map_positions, 0, n_bins - 1)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_linear_trajectory(simple_env, posterior, times, method="map")

        assert result.slope < 0

    def test_fit_linear_trajectory_non_uniform_times(self, simple_env):
        """Should work with non-uniformly spaced time bins."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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

    def test_fit_linear_trajectory_n_samples_affects_std(self, simple_env):
        """More samples should generally yield more stable estimates."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        n_time_bins = 20
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with some posterior_entropy
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result_100 = fit_linear_trajectory(
            simple_env, posterior, times, method="sample", n_samples=100, rng=42
        )
        result_1000 = fit_linear_trajectory(
            simple_env,
            posterior,
            times,
            method="sample",
            n_samples=1000,
            rng=43,
        )

        # Both should have posterior_entropy estimates
        assert result_100.slope_std is not None
        assert result_1000.slope_std is not None

    def test_fit_linear_trajectory_invalid_method_raises(self, simple_env):
        """Invalid method should raise ValueError."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        n_time_bins = 10
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins)

        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        with pytest.raises(ValueError, match=r"method.*map.*sample"):
            fit_linear_trajectory(simple_env, posterior, times, method="invalid")

    def test_fit_linear_trajectory_1d_posterior_raises(self, simple_env):
        """1D posterior should raise ValueError."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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


class TestDetectTrajectoryRadon:
    """The detected line is a physical speed and start position."""

    N = 50

    @pytest.fixture
    def forward(self):
        """One 1 cm bin per 10 ms time bin: 100 cm/s from the 0.5 cm center."""
        posterior = np.eye(self.N)
        return _track_for(posterior), posterior, np.arange(self.N) * 0.01

    def test_forward_sweep_speed_and_start(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        result = detect_trajectory_radon(*forward)

        assert result.speed == pytest.approx(100.0)
        assert result.start_position == pytest.approx(0.5)
        assert result.angle == pytest.approx(np.pi / 4)
        assert result.score == pytest.approx(1.0)
        assert result.time_bin_size == pytest.approx(0.01)
        assert result.column_width == pytest.approx(1.0)

    def test_constant_position_has_zero_speed(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, _, times = forward
        posterior = np.zeros((self.N, self.N))
        posterior[:, 20] = 1.0
        result = detect_trajectory_radon(env, posterior, times)

        assert result.speed == 0.0
        assert result.start_position == pytest.approx(20.5)
        assert result.score == pytest.approx(1.0)

    def test_mirrored_sweep_has_mirrored_speed_and_equal_score(self):
        """Reversing the posterior along the track flips only the direction."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, posterior, times = _gaussian_sweep(2.0, speed=150.0)
        forward = detect_trajectory_radon(env, posterior, times)
        reverse = detect_trajectory_radon(env, posterior[:, ::-1], times)

        assert forward.speed == pytest.approx(150.0, rel=0.04)
        assert reverse.speed == pytest.approx(-forward.speed)
        assert reverse.start_position == pytest.approx(100.0 - forward.start_position)
        assert reverse.score == pytest.approx(forward.score, abs=1e-12)

    def test_time_reversed_sweep_scores_the_same(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        reverse = detect_trajectory_radon(env, posterior[::-1], times)

        assert reverse.speed == pytest.approx(-100.0)
        assert reverse.score == pytest.approx(1.0)

    @pytest.mark.parametrize("bin_size", [1.0, 2.0, 4.0])
    def test_speed_does_not_depend_on_bin_size(self, bin_size):
        """A 200 cm/s replay is 200 cm/s on 1, 2 and 4 cm bins."""
        from neurospatial.decoding.trajectory import (
            detect_trajectory_radon,
            fit_linear_trajectory,
        )

        env, posterior, times = _gaussian_sweep(bin_size)
        radon = detect_trajectory_radon(
            env, posterior, times, angle_range=(0.0, 1.5), angle_step=np.pi / 720
        )
        linear = fit_linear_trajectory(env, posterior, times, method="map")

        assert radon.speed == pytest.approx(200.0, rel=0.01)
        assert radon.speed == pytest.approx(linear.slope, rel=0.01)
        assert radon.start_position == pytest.approx(20.0, abs=bin_size / 2)

    def test_uniform_posterior_scores_below_a_sharp_sweep(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        uniform = np.full((self.N, self.N), 1.0 / self.N)

        assert detect_trajectory_radon(env, uniform, times).score == pytest.approx(
            1.0 / self.N
        )
        assert detect_trajectory_radon(env, posterior, times).score == pytest.approx(
            1.0
        )

    def test_dropped_time_bins_keep_the_speed(self, forward):
        """Undecodable time bins can be left out with their times."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        keep = np.ones(self.N, dtype=bool)
        keep[[5, 17, 30]] = False
        result = detect_trajectory_radon(env, posterior[keep], times[keep])

        assert result.speed == pytest.approx(100.0)
        assert result.start_position == pytest.approx(0.5)
        assert result.score == pytest.approx(1.0)

    def test_angle_grid_honours_the_step(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        step = np.deg2rad(0.7)
        result = detect_trajectory_radon(
            *forward, angle_range=(-np.pi / 4, np.pi / 4), angle_step=step
        )

        np.testing.assert_allclose(np.diff(result.sinogram_angles), step)
        assert result.sinogram_angles[0] == pytest.approx(-np.pi / 4)
        assert result.sinogram_angles[-1] <= np.pi / 4
        assert result.sinogram.shape == (len(result.sinogram_angles), self.N)
        assert result.sinogram_positions == pytest.approx(np.arange(self.N) + 0.5)

    def test_default_grid_leaves_out_vertical_lines(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        angles = detect_trajectory_radon(*forward).sinogram_angles

        assert len(angles) == 179  # every degree from -89 to 89
        np.testing.assert_allclose(np.rad2deg(angles[[0, -1]]), [-89.0, 89.0])

    def test_angle_range_limits_the_detected_angle(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        result = detect_trajectory_radon(*forward, angle_range=(1.0, 1.2))

        assert 1.0 <= result.angle <= 1.2

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"angle_step": 4.0}, r"angle_step <= 3\.14159"),
            ({"angle_step": 0.0}, "0 < angle_step"),
            ({"angle_range": (0.5, 0.2)}, "low < high"),
            ({"angle_range": (-90.0, 90.0)}, "in radians"),
            ({"angle_step": np.pi}, r"only \+/-pi/2"),
        ],
    )
    def test_angles_that_leave_nothing_to_search_are_refused(
        self, forward, kwargs, match
    ):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        with pytest.raises(ValueError, match=match):
            detect_trajectory_radon(*forward, **kwargs)

    def test_times_must_increase(self, forward):
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, posterior, times = forward
        with pytest.raises(ValueError, match="strictly increasing"):
            detect_trajectory_radon(env, posterior, times[::-1])

    def test_detect_trajectory_radon_empty_posterior(self, forward):
        """An all-zero posterior holds no probability and is refused."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        env, shuffled_env, posterior, shuffled, times = sweep
        original = fit_linear_trajectory(env, posterior, times, method=method, rng=0)
        renumbered = fit_linear_trajectory(
            shuffled_env, shuffled, times, method=method, rng=0
        )

        assert original.slope == pytest.approx(500.0)  # cm/s
        assert original.intercept == pytest.approx(2.5)  # first bin center
        assert renumbered.slope == pytest.approx(original.slope)
        assert renumbered.r_squared == pytest.approx(original.r_squared)

    @pytest.mark.parametrize("method", ["map", "expected"])
    def test_isotonic_fit_ignores_bin_numbering(self, sweep, method):
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

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
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        env, shuffled_env, posterior, shuffled, times = sweep
        original = detect_trajectory_radon(env, posterior, times)
        renumbered = detect_trajectory_radon(shuffled_env, shuffled, times)

        assert original.speed == pytest.approx(500.0)  # cm/s
        assert original.start_position == pytest.approx(2.5)
        np.testing.assert_allclose(renumbered.sinogram, original.sinogram)

    def test_constant_speed_over_unequal_bins_is_a_straight_line(self):
        """Bins of 2 cm then 8 cm: visiting each center at 1000 cm/s is linear."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        from neurospatial.decoding.trajectory import (
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
        from neurospatial.decoding.trajectory import detect_trajectory_radon

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
        import neurospatial.decoding.trajectory as trajectory

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
        import neurospatial.decoding.trajectory as trajectory

        return getattr(trajectory, fit)(env, posterior, times)

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        env, posterior, times = sweep
        posterior = posterior.copy()
        posterior[3, 0] = -0.5
        posterior[3, 1] += 0.5
        with pytest.raises(ValueError, match=r"1 of 50 rows.*negative"):
            fit_linear_trajectory(env, posterior, times)

    def test_rounding_error_in_row_sums_is_accepted(self, sweep):
        """A float32 posterior sums to 1 only to single precision."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        env, posterior, times = stationary
        result = fit_linear_trajectory(env, posterior, times, method=method, rng=0)

        assert result.slope == pytest.approx(0.0)
        assert np.isnan(result.r_squared)

    @pytest.mark.parametrize("increasing", [None, True, False])
    def test_isotonic_fit_has_no_direction(self, stationary, increasing):
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        env, posterior, times = stationary
        result = fit_isotonic_trajectory(env, posterior, times, increasing=increasing)

        assert result.direction is None
        assert np.isnan(result.r_squared)
        np.testing.assert_allclose(result.fitted_positions, 10.5)

    def test_identical_times_are_refused(self):
        from neurospatial.decoding.trajectory import fit_linear_trajectory

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
        import neurospatial.decoding.trajectory as trajectory

        env, posterior, _ = sweep
        with pytest.raises(ValueError, match=r"times has 15 entries.*10 rows"):
            getattr(trajectory, fit)(env, posterior, np.arange(15) * 0.01)

    @pytest.mark.parametrize(
        "fit", ["fit_linear_trajectory", "fit_isotonic_trajectory"]
    )
    def test_times_must_be_finite(self, sweep, fit):
        import neurospatial.decoding.trajectory as trajectory

        env, posterior, times = sweep
        times = times.copy()
        times[4] = np.nan
        with pytest.raises(ValueError, match=r"times.*NaN or Inf"):
            getattr(trajectory, fit)(env, posterior, times)

    @pytest.mark.parametrize("n_samples", [0, 1])
    def test_sampling_needs_two_samples(self, sweep, n_samples):
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        env, posterior, times = sweep
        with pytest.raises(ValueError, match=r"n_samples.*at least 2"):
            fit_linear_trajectory(env, posterior, times, n_samples=n_samples)
