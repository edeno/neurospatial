"""Tests for RegularGridLayout class, especially error messages."""

import numpy as np
import pytest

from neurospatial.layout.engines.regular_grid import RegularGridLayout


class TestNoActiveBinsError:
    """Tests for the 'No active bins found' error message."""

    def test_no_active_bins_error_bin_size_too_large(self):
        """Test error message when bin_size is too large for data range."""
        # Create small data range (0-10)
        data = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])

        # Use bin_size that's too large (50 > 10)
        # This creates a 1x1 grid, and with threshold=2, no bins have enough samples
        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=50.0,  # Much larger than data range
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,  # Require more samples than we have
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Check for WHAT went wrong
        assert "No active bins found" in error_msg

        # Check for diagnostic information
        assert "bin_size" in error_msg
        assert "50" in error_msg or "50.0" in error_msg

        # Check for data range information
        assert (
            "data range" in error_msg.lower() or "dimension_ranges" in error_msg.lower()
        )

        # Check for WHY explanation (common causes)
        assert "too large" in error_msg.lower() or "larger" in error_msg.lower()

        # Check for HOW to fix (suggestions)
        assert "reduce" in error_msg.lower() or "smaller" in error_msg.lower()

    def test_no_active_bins_error_threshold_too_high(self):
        """Test error message when bin_count_threshold is too high."""
        # Create sparse data
        data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])  # 3 samples

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=1.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=10,  # Higher than any bin can have
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Check for WHAT went wrong
        assert "No active bins found" in error_msg

        # Check for diagnostic information
        assert "bin_count_threshold" in error_msg
        assert "10" in error_msg

        # Check for WHY explanation
        assert "threshold" in error_msg.lower()

        # Check for HOW to fix
        assert "reduce" in error_msg.lower() or "lower" in error_msg.lower()

    def test_no_active_bins_error_no_morphological_ops(self):
        """Test error message when data is sparse and no morphological operations used."""
        # Create very sparse data - only corner points
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=1.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=2,  # Require at least 2 samples per bin (only 2 total)
                dilate=False,  # No dilation
                fill_holes=False,  # No hole filling
                close_gaps=False,  # No gap closing
            )

        error_msg = str(exc_info.value)

        # Check for WHAT went wrong
        assert "No active bins found" in error_msg

        # Check for suggestions about morphological operations
        assert (
            "dilate" in error_msg.lower()
            or "fill_holes" in error_msg.lower()
            or "close_gaps" in error_msg.lower()
            or "morphological" in error_msg.lower()
        )

    def test_no_active_bins_error_shows_actual_data_range(self):
        """Test that error message shows the actual data range."""
        # Create data with known range
        data = np.array([[5.0, 10.0], [15.0, 20.0], [10.0, 15.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=100.0,  # Too large
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,  # Too high for any bin
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should show data range information
        # X range: 5.0 to 15.0 (range = 10.0)
        # Y range: 10.0 to 20.0 (range = 10.0)
        assert "5" in error_msg or "15" in error_msg  # X range
        assert "10" in error_msg or "20" in error_msg  # Y range

    def test_no_active_bins_error_shows_grid_shape(self):
        """Test that error message shows the resulting grid shape."""
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=100.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,  # Too high
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should mention grid shape or number of bins
        assert (
            "grid" in error_msg.lower()
            or "bin" in error_msg.lower()
            or "shape" in error_msg.lower()
        )

    def test_no_active_bins_error_shows_parameters_used(self):
        """Test that error message shows key parameters that were used."""
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=50.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,
                dilate=True,
                fill_holes=True,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should show the parameters that affect active bin inference
        assert "bin_size" in error_msg
        assert "bin_count_threshold" in error_msg

        # Should show morphological operation settings
        assert (
            "dilate" in error_msg.lower()
            or "fill_holes" in error_msg.lower()
            or "morphological" in error_msg.lower()
        )

    def test_no_active_bins_error_provides_actionable_suggestions(self):
        """Test that error message provides specific, actionable suggestions."""
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=50.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=10,
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should provide at least 2 actionable suggestions
        suggestion_count = 0

        if "reduce bin_size" in error_msg.lower() or "smaller bin" in error_msg.lower():
            suggestion_count += 1

        if (
            "reduce bin_count_threshold" in error_msg.lower()
            or "lower threshold" in error_msg.lower()
        ):
            suggestion_count += 1

        if (
            "dilate=true" in error_msg.lower()
            or "enable morphological" in error_msg.lower()
        ):
            suggestion_count += 1

        assert suggestion_count >= 2, (
            f"Expected at least 2 suggestions, error was:\n{error_msg}"
        )

    def test_no_active_bins_error_multiline_format(self):
        """Test that error message is well-formatted with multiple lines."""
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=50.0,
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,  # Too high
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should be multi-line (more informative than single line)
        lines = error_msg.strip().split("\n")
        assert len(lines) >= 3, f"Expected multi-line error message, got:\n{error_msg}"

    def test_no_active_bins_error_with_sequence_bin_size(self):
        """Test error message when bin_size is a sequence."""
        data = np.array([[0.0, 0.0], [10.0, 10.0]])

        layout = RegularGridLayout()

        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=[100.0, 50.0],  # Different sizes per dimension
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=5,
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should show list representation of bin_size
        assert "[100" in error_msg or "100.0" in error_msg
        assert "50" in error_msg or "50.0" in error_msg

    def test_no_active_bins_error_all_nan_data(self):
        """Test error message when all data samples are NaN."""
        # Create data with all NaN values
        data = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

        layout = RegularGridLayout()

        # With all NaN data, the active_mask will be all False after filtering
        with pytest.raises(ValueError) as exc_info:
            layout.build(
                bin_size=1.0,
                dimension_ranges=[(0, 10), (0, 10)],  # Must provide explicit range
                data_samples=data,
                infer_active_bins=True,
                bin_count_threshold=0,
                dilate=False,
                fill_holes=False,
                close_gaps=False,
            )

        error_msg = str(exc_info.value)

        # Should specifically mention NaN issue
        assert "NaN" in error_msg or "nan" in error_msg.lower()
        assert "no valid data" in error_msg.lower() or "all nan" in error_msg.lower()
