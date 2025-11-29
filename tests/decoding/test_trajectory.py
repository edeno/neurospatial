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

    def test_isotonic_fit_result_creation(self):
        """IsotonicFitResult should be creatable with required fields."""
        from neurospatial.decoding.trajectory import IsotonicFitResult

        fitted = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        residuals = np.array([0.1, -0.1, 0.05, -0.05, 0.0])

        result = IsotonicFitResult(
            fitted_positions=fitted,
            r_squared=0.95,
            direction="increasing",
            residuals=residuals,
        )

        np.testing.assert_array_equal(result.fitted_positions, fitted)
        assert result.r_squared == 0.95
        assert result.direction == "increasing"
        np.testing.assert_array_equal(result.residuals, residuals)

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

    def test_isotonic_fit_result_direction_values(self):
        """IsotonicFitResult direction should accept 'increasing' or 'decreasing'."""
        from neurospatial.decoding.trajectory import IsotonicFitResult

        # Both directions should work
        for direction in ["increasing", "decreasing"]:
            result = IsotonicFitResult(
                fitted_positions=np.array([1.0]),
                r_squared=0.8,
                direction=direction,
                residuals=np.array([0.0]),
            )
            assert result.direction == direction


class TestLinearFitResult:
    """Test LinearFitResult dataclass."""

    def test_linear_fit_result_creation(self):
        """LinearFitResult should be creatable with required fields."""
        from neurospatial.decoding.trajectory import LinearFitResult

        result = LinearFitResult(
            slope=2.5,
            intercept=10.0,
            r_squared=0.92,
            slope_std=0.15,
        )

        assert result.slope == 2.5
        assert result.intercept == 10.0
        assert result.r_squared == 0.92
        assert result.slope_std == 0.15

    def test_linear_fit_result_slope_std_none(self):
        """LinearFitResult slope_std can be None (for method='map')."""
        from neurospatial.decoding.trajectory import LinearFitResult

        result = LinearFitResult(
            slope=1.0,
            intercept=0.0,
            r_squared=0.85,
            slope_std=None,
        )

        assert result.slope_std is None

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

    def test_linear_fit_result_negative_slope(self):
        """LinearFitResult should accept negative slopes (reverse trajectories)."""
        from neurospatial.decoding.trajectory import LinearFitResult

        result = LinearFitResult(
            slope=-3.0,
            intercept=100.0,
            r_squared=0.88,
            slope_std=0.2,
        )

        assert result.slope == -3.0


class TestRadonDetectionResult:
    """Test RadonDetectionResult dataclass."""

    def test_radon_detection_result_creation(self):
        """RadonDetectionResult should be creatable with required fields."""
        from neurospatial.decoding.trajectory import RadonDetectionResult

        sinogram = np.random.rand(180, 100)

        result = RadonDetectionResult(
            angle_degrees=45.0,
            score=0.85,
            offset=25.0,
            sinogram=sinogram,
        )

        assert result.angle_degrees == 45.0
        assert result.score == 0.85
        assert result.offset == 25.0
        np.testing.assert_array_equal(result.sinogram, sinogram)

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

    def test_radon_detection_result_negative_angle(self):
        """RadonDetectionResult should accept negative angles."""
        from neurospatial.decoding.trajectory import RadonDetectionResult

        result = RadonDetectionResult(
            angle_degrees=-45.0,
            score=0.9,
            offset=0.0,
            sinogram=np.zeros((10, 10)),
        )

        assert result.angle_degrees == -45.0


class TestResultDataclassesExport:
    """Test that result dataclasses are properly exported."""

    def test_import_from_trajectory_module(self):
        """Result dataclasses should be importable from trajectory module."""
        from neurospatial.decoding.trajectory import (
            IsotonicFitResult,
            LinearFitResult,
            RadonDetectionResult,
        )

        assert IsotonicFitResult is not None
        assert LinearFitResult is not None
        assert RadonDetectionResult is not None

    def test_import_from_decoding_package(self):
        """Result dataclasses should be importable from decoding package."""
        from neurospatial.decoding import (
            IsotonicFitResult,
            LinearFitResult,
            RadonDetectionResult,
        )

        assert IsotonicFitResult is not None
        assert LinearFitResult is not None
        assert RadonDetectionResult is not None


# =============================================================================
# Milestone 3.2: Isotonic Regression
# =============================================================================


class TestFitIsotonicTrajectory:
    """Test fit_isotonic_trajectory function."""

    def test_fit_isotonic_trajectory_returns_result(self):
        """fit_isotonic_trajectory should return IsotonicFitResult."""
        from neurospatial.decoding.trajectory import (
            IsotonicFitResult,
            fit_isotonic_trajectory,
        )

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with increasing MAP positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(posterior, times)

        assert isinstance(result, IsotonicFitResult)

    def test_fit_isotonic_trajectory_fitted_positions_shape(self):
        """Fitted positions should have shape (n_time_bins,)."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        n_time_bins = 15
        n_bins = 30
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(posterior, times)

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

        result = fit_isotonic_trajectory(posterior, times)

        assert result.residuals.shape == (n_time_bins,)

    def test_fit_isotonic_trajectory_r_squared_range(self):
        """R² should be in [0, 1]."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 40
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(posterior, times)

        assert 0.0 <= result.r_squared <= 1.0

    def test_fit_isotonic_trajectory_direction_values(self):
        """Direction should be 'increasing' or 'decreasing'."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 40
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(posterior, times)

        assert result.direction in ("increasing", "decreasing")

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

        result = fit_isotonic_trajectory(posterior, times)

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

        result = fit_isotonic_trajectory(posterior, times)

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

        result = fit_isotonic_trajectory(posterior, times, increasing=True)

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

        result = fit_isotonic_trajectory(posterior, times, increasing=False)

        assert result.direction == "decreasing"

    def test_fit_isotonic_trajectory_method_map(self):
        """method='map' should use argmax positions."""
        from neurospatial.decoding.trajectory import (
            IsotonicFitResult,
            fit_isotonic_trajectory,
        )

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with clear MAP positions
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 0.9
            # Add some noise
            posterior[t, (pos + 1) % n_bins] = 0.1

        result = fit_isotonic_trajectory(posterior, times, method="map")

        assert isinstance(result, IsotonicFitResult)

    def test_fit_isotonic_trajectory_method_expected(self):
        """method='expected' should use weighted mean positions."""
        from neurospatial.decoding.trajectory import (
            IsotonicFitResult,
            fit_isotonic_trajectory,
        )

        n_time_bins = 20
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)
        rng = np.random.default_rng(42)

        posterior = rng.random((n_time_bins, n_bins))
        posterior /= posterior.sum(axis=1, keepdims=True)

        result = fit_isotonic_trajectory(posterior, times, method="expected")

        assert isinstance(result, IsotonicFitResult)

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

        result = fit_isotonic_trajectory(posterior, times)

        # Check monotonicity
        diffs = np.diff(result.fitted_positions)
        if result.direction == "increasing":
            assert np.all(diffs >= 0)
        else:
            assert np.all(diffs <= 0)

    def test_fit_isotonic_trajectory_non_uniform_times(self):
        """Should work with non-uniformly spaced time bins."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        n_time_bins = 20
        n_bins = 50

        # Non-uniform times
        times = np.sort(np.random.default_rng(42).uniform(0, 1, n_time_bins))

        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(posterior, times)

        assert result.fitted_positions.shape == (n_time_bins,)

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
            fit_isotonic_trajectory(posterior, times, method="invalid")


class TestFitIsotonicTrajectorySuccessCriteria:
    """Test success criteria from TASKS.md for fit_isotonic_trajectory."""

    def test_success_criteria(self):
        """Verify success criteria from TASKS.md for Milestone 3.2."""
        from neurospatial.decoding.trajectory import fit_isotonic_trajectory

        n_time_bins = 25
        n_bins = 50
        times = np.linspace(0, 1, n_time_bins)

        # Create posterior with monotonic pattern
        posterior = np.zeros((n_time_bins, n_bins))
        map_positions = np.linspace(5, 45, n_time_bins).astype(int)
        for t, pos in enumerate(map_positions):
            posterior[t, pos] = 1.0

        result = fit_isotonic_trajectory(posterior, times)

        # Success criteria from TASKS.md
        assert result.fitted_positions.shape == (n_time_bins,)
        assert 0.0 <= result.r_squared <= 1.0
        assert result.direction in ("increasing", "decreasing")
