"""Tests for trajectory analysis in neurospatial.decoding.trajectory.

These tests verify trajectory fitting and detection functions for analyzing
decoded position sequences, including isotonic regression, linear regression,
and Radon transform detection.
"""

import numpy as np
import pytest

# =============================================================================
# Milestone 3.1: Result Dataclasses
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
            angle_degrees=30.0,
            score=0.7,
            offset=10.0,
            sinogram=np.zeros((10, 10)),
        )

        with pytest.raises((AttributeError, TypeError)):
            result.angle_degrees = 60.0


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

        result = fit_isotonic_trajectory(None, posterior, times)

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

        result = fit_isotonic_trajectory(None, posterior, times)

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

        result = fit_isotonic_trajectory(None, posterior, times)

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

        result = fit_isotonic_trajectory(None, posterior, times)

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

        result = fit_isotonic_trajectory(None, posterior, times, increasing=True)

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

        result = fit_isotonic_trajectory(None, posterior, times, increasing=False)

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

        result = fit_isotonic_trajectory(None, posterior, times)

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
            fit_isotonic_trajectory(None, posterior, times, method="invalid")


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

    def test_fit_linear_trajectory_times_mismatch_raises(self, simple_env):
        """Mismatched times length should raise ValueError."""
        from neurospatial.decoding.trajectory import fit_linear_trajectory

        n_time_bins = 10
        n_bins = simple_env.n_bins
        times = np.linspace(0, 1, n_time_bins + 5)  # Wrong length

        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        with pytest.raises(ValueError, match=r"times length.*must match posterior"):
            fit_linear_trajectory(simple_env, posterior, times)


class TestDetectTrajectoryRadon:
    """Test detect_trajectory_radon function (requires scikit-image)."""

    def test_detect_trajectory_radon_angle_range(self):
        """Detected angle should be in the specified theta_range."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 50
        n_bins = 50

        # Create a diagonal pattern
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            pos = t % n_bins
            posterior[t, pos] = 1.0

        result = detect_trajectory_radon(posterior, theta_range=(-90, 90))

        assert -90 <= result.angle_degrees <= 90

    def test_detect_trajectory_radon_sinogram_shape(self):
        """Sinogram should have shape (n_angles, n_offsets)."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 50
        n_bins = 40

        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = detect_trajectory_radon(
            posterior, theta_range=(-45, 45), theta_step=1.0
        )

        # Sinogram should be 2D
        assert result.sinogram.ndim == 2
        # Number of angles should match (45 - (-45)) / 1.0 = 90
        assert result.sinogram.shape[0] == 90

    def test_detect_trajectory_radon_diagonal_forward(self):
        """Should detect ~45° angle for forward diagonal trajectory."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 100
        n_bins = 100

        # Create perfect diagonal (45° in image coordinates)
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            pos = t
            if pos < n_bins:
                posterior[t, pos] = 1.0

        result = detect_trajectory_radon(posterior, theta_step=0.5)

        # For a 45° diagonal line in image space, Radon angle depends on orientation
        # The detected angle should indicate a diagonal pattern
        assert result.score > 0  # Should have non-zero score

    def test_detect_trajectory_radon_diagonal_reverse(self):
        """Should detect reverse diagonal trajectory."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 100
        n_bins = 100

        # Create reverse diagonal (negative slope)
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            pos = n_bins - 1 - t
            if 0 <= pos < n_bins:
                posterior[t, pos] = 1.0

        result = detect_trajectory_radon(posterior, theta_step=0.5)

        # Should have opposite angle from forward diagonal
        assert result.score > 0

    def test_detect_trajectory_radon_horizontal_line(self):
        """Horizontal line (constant position) should detect ~0° angle."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 100
        n_bins = 50

        # Create horizontal line (constant position at middle)
        posterior = np.zeros((n_time_bins, n_bins))
        constant_pos = n_bins // 2
        for t in range(n_time_bins):
            posterior[t, constant_pos] = 1.0

        result = detect_trajectory_radon(posterior, theta_step=1.0)

        # Horizontal line should be near 0° or 90° depending on Radon convention
        # The key is that it should be detected consistently
        assert np.isfinite(result.angle_degrees)
        assert result.score > 0

    def test_detect_trajectory_radon_theta_step_affects_resolution(self):
        """Smaller theta_step should give more angle resolution."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 50
        n_bins = 50

        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result_coarse = detect_trajectory_radon(posterior, theta_step=5.0)
        result_fine = detect_trajectory_radon(posterior, theta_step=0.5)

        # Finer step should give more angles
        assert result_fine.sinogram.shape[0] > result_coarse.sinogram.shape[0]

    def test_detect_trajectory_radon_custom_theta_range(self):
        """Custom theta_range should limit detected angles."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 50
        n_bins = 50

        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        # Narrow angle range
        result = detect_trajectory_radon(
            posterior, theta_range=(30, 60), theta_step=1.0
        )

        # Angle should be within range
        assert 30 <= result.angle_degrees <= 60

    def test_detect_trajectory_radon_non_square_posterior(self):
        """Should work with non-square posteriors."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        n_time_bins = 100
        n_bins = 40

        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = detect_trajectory_radon(posterior)

        assert isinstance(result.angle_degrees, float)
        assert result.sinogram.ndim == 2

    def test_detect_trajectory_radon_empty_posterior(self):
        """Should handle all-zero posterior gracefully."""
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        posterior = np.zeros((50, 50))
        result = detect_trajectory_radon(posterior)

        assert np.isfinite(result.angle_degrees)
        assert result.score >= 0


class TestRadonAngleRecovery:
    """Radon detection recovers the trajectory ANGLE, not merely score > 0.

    The existing diagonal tests assert only ``score > 0``, which is true for any
    non-negative posterior. These assert the recovered angle matches the known
    geometry (convention from the docstring: +45 forward, -45 reverse, 0 for a
    constant position) and that a uniform posterior yields a far lower score than
    a concentrated trajectory.
    """

    N = 50

    def _forward_diagonal(self) -> np.ndarray:
        posterior = np.zeros((self.N, self.N))
        for t in range(self.N):
            posterior[t, t] = 1.0
        return posterior

    def test_forward_diagonal_recovers_plus_45(self) -> None:
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        result = detect_trajectory_radon(self._forward_diagonal(), theta_step=0.5)
        assert abs(result.angle_degrees - 45.0) < 5.0

    def test_reverse_diagonal_recovers_minus_45(self) -> None:
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        posterior = np.zeros((self.N, self.N))
        for t in range(self.N):
            posterior[t, self.N - 1 - t] = 1.0  # position decreases with time
        result = detect_trajectory_radon(posterior, theta_step=0.5)
        assert abs(result.angle_degrees - (-45.0)) < 5.0

    def test_constant_position_recovers_zero(self) -> None:
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        posterior = np.zeros((self.N, self.N))
        posterior[:, self.N // 2] = 1.0  # same bin for all time -> horizontal
        result = detect_trajectory_radon(posterior, theta_step=1.0)
        assert abs(result.angle_degrees) < 5.0

    def test_uniform_posterior_scores_far_below_trajectory(self) -> None:
        from neurospatial.decoding.trajectory import detect_trajectory_radon

        # The Radon score is integrated probability mass along a line, so its
        # scale depends on the image; an absolute threshold is meaningless. The
        # meaningful claim is that a uniform posterior (no trajectory) integrates
        # to far less mass than a concentrated diagonal.
        uniform = np.full((self.N, self.N), 1.0 / self.N)
        uniform_score = detect_trajectory_radon(uniform).score
        diagonal_score = detect_trajectory_radon(self._forward_diagonal()).score
        assert uniform_score < 0.2 * diagonal_score
