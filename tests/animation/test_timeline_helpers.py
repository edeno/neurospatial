"""Tests for timeline and interpolation helper functions.

This module tests the private helper functions used for temporal alignment
of overlay data to animation frame times.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neurospatial.animation.overlays import (
    _build_frame_times,
    _interp_linear,
    _interp_nearest,
)


class TestBuildFrameTimes:
    """Test _build_frame_times() function."""

    def test_with_provided_frame_times(self):
        """Test using provided frame_times array."""
        frame_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        n_frames = 5

        result = _build_frame_times(
            frame_times=frame_times, fps=None, n_frames=n_frames
        )

        assert_array_equal(result, frame_times)

    def test_synthesize_from_fps(self):
        """Test synthesizing frame_times from fps and n_frames."""
        fps = 30
        n_frames = 90

        result = _build_frame_times(frame_times=None, fps=fps, n_frames=n_frames)

        # Should create timestamps at 30 Hz: 0, 1/30, 2/30, ..., 89/30
        expected = np.arange(n_frames) / fps
        assert_array_almost_equal(result, expected)

    def test_monotonic_validation_passes(self):
        """Test that monotonic frame_times pass validation."""
        frame_times = np.array([0.0, 1.0, 2.0, 3.0])
        n_frames = 4

        # Should not raise
        result = _build_frame_times(
            frame_times=frame_times, fps=None, n_frames=n_frames
        )
        assert_array_equal(result, frame_times)

    def test_non_monotonic_raises_error(self):
        """Test that non-monotonic frame_times raise ValueError."""
        frame_times = np.array([0.0, 2.0, 1.0, 3.0])  # Not monotonic
        n_frames = 4

        with pytest.raises(ValueError, match="frame_times must be monotonically"):
            _build_frame_times(frame_times=frame_times, fps=None, n_frames=n_frames)

    def test_length_mismatch_raises_error(self):
        """Test that frame_times length must match n_frames."""
        frame_times = np.array([0.0, 1.0, 2.0])  # 3 elements
        n_frames = 5  # But need 5

        with pytest.raises(ValueError, match="frame_times length"):
            _build_frame_times(frame_times=frame_times, fps=None, n_frames=n_frames)

    def test_neither_provided_raises_error(self):
        """Test that either frame_times or fps must be provided."""
        with pytest.raises(ValueError, match="Either frame_times or fps"):
            _build_frame_times(frame_times=None, fps=None, n_frames=10)

    def test_both_provided_prefers_frame_times(self):
        """Test that frame_times takes precedence when both provided."""
        frame_times = np.array([0.0, 0.5, 1.0])
        fps = 10  # Would create different times
        n_frames = 3

        result = _build_frame_times(frame_times=frame_times, fps=fps, n_frames=n_frames)

        # Should use frame_times, not fps
        assert_array_equal(result, frame_times)


class TestInterpLinear:
    """Test _interp_linear() vectorized linear interpolation."""

    def test_basic_interpolation_1d(self):
        """Test linear interpolation of 1D data."""
        t_src = np.array([0.0, 1.0, 2.0, 3.0])
        x_src = np.array([0.0, 10.0, 20.0, 30.0])
        t_frame = np.array([0.5, 1.5, 2.5])

        result = _interp_linear(t_src, x_src, t_frame)

        expected = np.array([5.0, 15.0, 25.0])
        assert_array_almost_equal(result, expected)

    def test_interpolation_2d_data(self):
        """Test linear interpolation of 2D position data."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        t_frame = np.array([0.5, 1.5])

        result = _interp_linear(t_src, x_src, t_frame)

        expected = np.array([[5.0, 2.5], [15.0, 7.5]])
        assert_array_almost_equal(result, expected)

    def test_extrapolation_returns_nan(self):
        """Test that extrapolation beyond source times returns NaN."""
        t_src = np.array([1.0, 2.0, 3.0])
        x_src = np.array([10.0, 20.0, 30.0])
        t_frame = np.array([0.0, 1.5, 4.0])  # 0.0 and 4.0 are out of bounds

        result = _interp_linear(t_src, x_src, t_frame)

        assert np.isnan(result[0])  # Before source range
        assert_array_almost_equal(result[1], 15.0)  # Within range
        assert np.isnan(result[2])  # After source range

    def test_exact_match_on_source_times(self):
        """Test interpolation at exact source time points."""
        t_src = np.array([0.0, 1.0, 2.0, 3.0])
        x_src = np.array([0.0, 10.0, 20.0, 30.0])
        t_frame = np.array([0.0, 2.0, 3.0])

        result = _interp_linear(t_src, x_src, t_frame)

        expected = np.array([0.0, 20.0, 30.0])
        assert_array_almost_equal(result, expected)

    def test_3d_data_interpolation(self):
        """Test linear interpolation of 3D position data."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 2.0], [20.0, 10.0, 4.0]])
        t_frame = np.array([0.5, 1.5])

        result = _interp_linear(t_src, x_src, t_frame)

        expected = np.array([[5.0, 2.5, 1.0], [15.0, 7.5, 3.0]])
        assert_array_almost_equal(result, expected)

    def test_empty_frame_times(self):
        """Test interpolation with empty frame times array."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([0.0, 10.0, 20.0])
        t_frame = np.array([])

        result = _interp_linear(t_src, x_src, t_frame)

        assert result.shape == (0,)


class TestInterpNearest:
    """Test _interp_nearest() vectorized nearest neighbor interpolation."""

    def test_basic_nearest_interpolation_1d(self):
        """Test nearest neighbor interpolation of 1D data."""
        t_src = np.array([0.0, 1.0, 2.0, 3.0])
        x_src = np.array([0.0, 10.0, 20.0, 30.0])
        t_frame = np.array([0.4, 0.6, 1.4, 1.6])

        result = _interp_nearest(t_src, x_src, t_frame)

        # 0.4 closer to 0.0 → 0.0
        # 0.6 closer to 1.0 → 10.0
        # 1.4 closer to 1.0 → 10.0
        # 1.6 closer to 2.0 → 20.0
        expected = np.array([0.0, 10.0, 10.0, 20.0])
        assert_array_almost_equal(result, expected)

    def test_nearest_interpolation_2d_data(self):
        """Test nearest neighbor interpolation of 2D position data."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
        t_frame = np.array([0.4, 1.4])

        result = _interp_nearest(t_src, x_src, t_frame)

        # 0.4 closer to 0.0
        # 1.4 closer to 1.0
        expected = np.array([[0.0, 0.0], [10.0, 5.0]])
        assert_array_almost_equal(result, expected)

    def test_extrapolation_returns_nan(self):
        """Test that extrapolation beyond source times returns NaN."""
        t_src = np.array([1.0, 2.0, 3.0])
        x_src = np.array([10.0, 20.0, 30.0])
        t_frame = np.array([0.0, 1.5, 4.0])  # 0.0 and 4.0 are out of bounds

        result = _interp_nearest(t_src, x_src, t_frame)

        assert np.isnan(result[0])  # Before source range
        assert_array_almost_equal(result[1], 10.0)  # Within range (closer to 1.0)
        assert np.isnan(result[2])  # After source range

    def test_exact_match_on_source_times(self):
        """Test interpolation at exact source time points."""
        t_src = np.array([0.0, 1.0, 2.0, 3.0])
        x_src = np.array([0.0, 10.0, 20.0, 30.0])
        t_frame = np.array([0.0, 2.0, 3.0])

        result = _interp_nearest(t_src, x_src, t_frame)

        expected = np.array([0.0, 20.0, 30.0])
        assert_array_almost_equal(result, expected)

    def test_midpoint_behavior(self):
        """Test behavior when frame time is exactly between two source times."""
        t_src = np.array([0.0, 2.0])
        x_src = np.array([0.0, 20.0])
        t_frame = np.array([1.0])  # Exactly at midpoint

        result = _interp_nearest(t_src, x_src, t_frame)

        # Should pick one consistently (implementation dependent)
        # Just verify it returns one of the values, not NaN
        assert not np.isnan(result[0])
        assert result[0] in [0.0, 20.0]

    def test_3d_data_interpolation(self):
        """Test nearest neighbor interpolation of 3D position data."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 2.0], [20.0, 10.0, 4.0]])
        t_frame = np.array([0.4, 1.6])

        result = _interp_nearest(t_src, x_src, t_frame)

        # 0.4 closer to 0.0
        # 1.6 closer to 2.0
        expected = np.array([[0.0, 0.0, 0.0], [20.0, 10.0, 4.0]])
        assert_array_almost_equal(result, expected)

    def test_empty_frame_times(self):
        """Test interpolation with empty frame times array."""
        t_src = np.array([0.0, 1.0, 2.0])
        x_src = np.array([0.0, 10.0, 20.0])
        t_frame = np.array([])

        result = _interp_nearest(t_src, x_src, t_frame)

        assert result.shape == (0,)


class TestInterpolationEdgeCases:
    """Test edge cases for interpolation functions."""

    def test_single_source_point(self):
        """Test interpolation with only one source point."""
        t_src = np.array([1.0])
        x_src = np.array([10.0])
        t_frame = np.array([0.5, 1.0, 1.5])

        # Linear should return NaN for all (can't interpolate with 1 point)
        result_linear = _interp_linear(t_src, x_src, t_frame)
        assert np.isnan(result_linear[0])  # Before
        assert_array_almost_equal(result_linear[1], 10.0)  # Exact match
        assert np.isnan(result_linear[2])  # After

        # Nearest should return value for exact match, NaN otherwise
        result_nearest = _interp_nearest(t_src, x_src, t_frame)
        assert np.isnan(result_nearest[0])  # Before
        assert_array_almost_equal(result_nearest[1], 10.0)  # Exact match
        assert np.isnan(result_nearest[2])  # After

    def test_nan_in_source_data_propagates(self):
        """Test that NaN values in source data are handled."""
        t_src = np.array([0.0, 1.0, 2.0, 3.0])
        x_src = np.array([0.0, np.nan, 20.0, 30.0])
        t_frame = np.array([0.5, 1.0, 1.5])

        result_linear = _interp_linear(t_src, x_src, t_frame)
        # Interpolation involving NaN should produce NaN
        assert np.isnan(result_linear[0])  # Interpolates between 0 and NaN
        assert np.isnan(result_linear[1])  # Exact NaN point
        assert np.isnan(result_linear[2])  # Interpolates between NaN and 20

        result_nearest = _interp_nearest(t_src, x_src, t_frame)
        # Nearest picks exact values, so 1.0 should be NaN
        assert np.isnan(result_nearest[1])  # Exact NaN point


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
