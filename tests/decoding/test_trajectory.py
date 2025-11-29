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
