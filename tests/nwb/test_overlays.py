"""
Tests for NWB overlay factory functions.

Tests the factory functions that create animation overlays from NWB data:
- position_overlay_from_nwb()
- bodypart_overlay_from_nwb()
- head_direction_overlay_from_nwb()
- environment_from_position()
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestPositionOverlayFromNwb:
    """Tests for position_overlay_from_nwb() function."""

    def test_basic_overlay_creation(self, sample_nwb_with_position):
        """Test PositionOverlay creation from NWB Position data."""
        from neurospatial.animation.overlays import PositionOverlay
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position)

        # Should return PositionOverlay instance
        assert isinstance(overlay, PositionOverlay)

        # Data should have correct shape (1000 samples, 2D)
        assert overlay.data.shape == (1000, 2)
        assert overlay.data.dtype == np.float64

        # Times should be populated from timestamps
        assert overlay.times is not None
        assert overlay.times.shape == (1000,)
        assert overlay.times.dtype == np.float64
        assert overlay.times[0] == 0.0
        assert overlay.times[-1] > 0.0

    def test_data_matches_original(self, sample_nwb_with_position):
        """Test that overlay data matches original NWB Position data."""
        from neurospatial.nwb import position_overlay_from_nwb, read_position

        overlay = position_overlay_from_nwb(sample_nwb_with_position)

        # Get original data for comparison
        positions, timestamps = read_position(sample_nwb_with_position)

        np.testing.assert_array_almost_equal(overlay.data, positions)
        np.testing.assert_array_almost_equal(overlay.times, timestamps)

    def test_color_parameter_passed_through(self, sample_nwb_with_position):
        """Test that color parameter is passed to PositionOverlay."""
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, color="blue")

        assert overlay.color == "blue"

    def test_size_parameter_passed_through(self, sample_nwb_with_position):
        """Test that size parameter is passed to PositionOverlay."""
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, size=20.0)

        assert overlay.size == 20.0

    def test_trail_length_parameter_passed_through(self, sample_nwb_with_position):
        """Test that trail_length parameter is passed to PositionOverlay."""
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, trail_length=15)

        assert overlay.trail_length == 15

    def test_default_parameters(self, sample_nwb_with_position):
        """Test that default parameters are applied correctly."""
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position)

        # Check defaults match function signature
        assert overlay.color == "red"
        assert overlay.size == 12.0
        assert overlay.trail_length == 0

    def test_processing_module_forwarded(self, empty_nwb):
        """Test that processing_module parameter is forwarded to read_position."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.nwb import position_overlay_from_nwb

        nwbfile = empty_nwb

        # Create Position in a custom module
        custom_module = nwbfile.create_processing_module(
            name="tracking", description="Tracking data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((50, 2)) * 42.0,
                timestamps=np.arange(50) / 10.0,
                reference_frame="test",
                unit="cm",
            )
        )
        custom_module.add(position)

        # Read with explicit module
        overlay = position_overlay_from_nwb(nwbfile, processing_module="tracking")

        assert overlay.data.shape == (50, 2)
        np.testing.assert_array_almost_equal(overlay.data, np.ones((50, 2)) * 42.0)

    def test_position_name_forwarded(self, sample_nwb_with_position_multiple_series):
        """Test that position_name parameter is forwarded to read_position."""
        from neurospatial.nwb import position_overlay_from_nwb

        # Request specific series by name
        overlay = position_overlay_from_nwb(
            sample_nwb_with_position_multiple_series,
            position_name="body",
        )

        # Check we got the 'body' series (500 samples)
        assert overlay.data.shape == (500, 2)

    def test_additional_kwargs_passed_through(self, sample_nwb_with_position):
        """Test that additional kwargs are passed to PositionOverlay."""
        from neurospatial.nwb import position_overlay_from_nwb

        # interp is a valid PositionOverlay parameter
        overlay = position_overlay_from_nwb(
            sample_nwb_with_position,
            interp="nearest",
        )

        assert overlay.interp == "nearest"

    def test_multiple_parameters_combined(self, sample_nwb_with_position):
        """Test creating overlay with multiple custom parameters."""
        from neurospatial.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(
            sample_nwb_with_position,
            color="green",
            size=15.0,
            trail_length=20,
            interp="nearest",
        )

        assert overlay.color == "green"
        assert overlay.size == 15.0
        assert overlay.trail_length == 20
        assert overlay.interp == "nearest"

    def test_error_when_no_position_found(self, empty_nwb):
        """Test KeyError when no Position container found."""
        from neurospatial.nwb import position_overlay_from_nwb

        with pytest.raises(KeyError, match=r"No Position.*found"):
            position_overlay_from_nwb(empty_nwb)
