"""Tests for timeline and interpolation helper functions.

This module tests the private helper functions used for temporal alignment
of overlay data to animation frame times.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.animation.overlays import _validate_frame_times


class TestValidateFrameTimes:
    """Test _validate_frame_times() function.

    This function validates that frame_times array is correctly shaped
    and monotonically increasing. It does NOT synthesize frame_times
    from fps - that responsibility is handled by the caller.
    """

    def test_returns_ndarray(self):
        """Test that function returns NDArray."""
        frame_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        n_frames = 5

        result = _validate_frame_times(frame_times, n_frames)

        assert isinstance(result, np.ndarray)

    def test_returns_same_array(self):
        """Test that function returns the same frame_times array."""
        frame_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        n_frames = 5

        result = _validate_frame_times(frame_times, n_frames)

        assert_array_equal(result, frame_times)

    def test_monotonic_validation_passes(self):
        """Test that monotonically increasing frame_times pass validation."""
        frame_times = np.array([0.0, 1.0, 2.0, 3.0])
        n_frames = 4

        # Should not raise
        result = _validate_frame_times(frame_times, n_frames)
        assert_array_equal(result, frame_times)

    def test_non_monotonic_raises_error(self):
        """Test that non-monotonic frame_times raise ValueError."""
        frame_times = np.array([0.0, 2.0, 1.0, 3.0])  # Not monotonic
        n_frames = 4

        with pytest.raises(ValueError, match="monotonically increasing"):
            _validate_frame_times(frame_times, n_frames)

    def test_equal_values_raises_error(self):
        """Test that duplicate values (non-strictly monotonic) raise ValueError."""
        frame_times = np.array([0.0, 1.0, 1.0, 2.0])  # Duplicate 1.0
        n_frames = 4

        with pytest.raises(ValueError, match="monotonically increasing"):
            _validate_frame_times(frame_times, n_frames)

    def test_length_mismatch_raises_error(self):
        """Test that frame_times length must match n_frames."""
        frame_times = np.array([0.0, 1.0, 2.0])  # 3 elements
        n_frames = 5  # But need 5

        with pytest.raises(ValueError, match="frame_times length"):
            _validate_frame_times(frame_times, n_frames)

    def test_single_frame_passes(self):
        """Test that single frame timestamps pass validation."""
        frame_times = np.array([5.0])  # Single frame
        n_frames = 1

        result = _validate_frame_times(frame_times, n_frames)
        assert_array_equal(result, frame_times)

    def test_reports_non_increasing_count(self):
        """Test that error message includes count of non-increasing intervals."""
        frame_times = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0])  # 2 bad intervals
        n_frames = 6

        with pytest.raises(ValueError, match=r"\d+ non-increasing"):
            _validate_frame_times(frame_times, n_frames)

    def test_irregular_spacing_passes(self):
        """Test that irregularly spaced but monotonic frame_times pass."""
        frame_times = np.array([0.0, 0.1, 0.5, 2.0, 10.0])  # Irregular spacing
        n_frames = 5

        result = _validate_frame_times(frame_times, n_frames)
        assert_array_equal(result, frame_times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
