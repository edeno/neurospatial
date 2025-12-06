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
        from neurospatial.io.nwb import position_overlay_from_nwb

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
        from neurospatial.io.nwb import position_overlay_from_nwb, read_position

        overlay = position_overlay_from_nwb(sample_nwb_with_position)

        # Get original data for comparison
        positions, timestamps = read_position(sample_nwb_with_position)

        np.testing.assert_array_almost_equal(overlay.data, positions)
        np.testing.assert_array_almost_equal(overlay.times, timestamps)

    def test_color_parameter_passed_through(self, sample_nwb_with_position):
        """Test that color parameter is passed to PositionOverlay."""
        from neurospatial.io.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, color="blue")

        assert overlay.color == "blue"

    def test_size_parameter_passed_through(self, sample_nwb_with_position):
        """Test that size parameter is passed to PositionOverlay."""
        from neurospatial.io.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, size=20.0)

        assert overlay.size == 20.0

    def test_trail_length_parameter_passed_through(self, sample_nwb_with_position):
        """Test that trail_length parameter is passed to PositionOverlay."""
        from neurospatial.io.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position, trail_length=15)

        assert overlay.trail_length == 15

    def test_default_parameters(self, sample_nwb_with_position):
        """Test that default parameters are applied correctly."""
        from neurospatial.io.nwb import position_overlay_from_nwb

        overlay = position_overlay_from_nwb(sample_nwb_with_position)

        # Check defaults match function signature
        assert overlay.color == "red"
        assert overlay.size == 12.0
        assert overlay.trail_length == 0

    def test_processing_module_forwarded(self, empty_nwb):
        """Test that processing_module parameter is forwarded to read_position."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import position_overlay_from_nwb

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
        from neurospatial.io.nwb import position_overlay_from_nwb

        # Request specific series by name
        overlay = position_overlay_from_nwb(
            sample_nwb_with_position_multiple_series,
            position_name="body",
        )

        # Check we got the 'body' series (500 samples)
        assert overlay.data.shape == (500, 2)

    def test_additional_kwargs_passed_through(self, sample_nwb_with_position):
        """Test that additional kwargs are passed to PositionOverlay."""
        from neurospatial.io.nwb import position_overlay_from_nwb

        # interp is a valid PositionOverlay parameter
        overlay = position_overlay_from_nwb(
            sample_nwb_with_position,
            interp="nearest",
        )

        assert overlay.interp == "nearest"

    def test_multiple_parameters_combined(self, sample_nwb_with_position):
        """Test creating overlay with multiple custom parameters."""
        from neurospatial.io.nwb import position_overlay_from_nwb

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
        from neurospatial.io.nwb import position_overlay_from_nwb

        with pytest.raises(KeyError, match=r"No Position.*found"):
            position_overlay_from_nwb(empty_nwb)


# Skip all pose tests if ndx-pose is not installed
ndx_pose = pytest.importorskip("ndx_pose")


class TestBodypartOverlayFromNwb:
    """Tests for bodypart_overlay_from_nwb() function."""

    def test_basic_overlay_creation(self, sample_nwb_with_pose):
        """Test BodypartOverlay creation from NWB PoseEstimation data."""
        from neurospatial.animation.overlays import BodypartOverlay
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose)

        # Should return BodypartOverlay instance
        assert isinstance(overlay, BodypartOverlay)

        # Data should be a dict of bodypart names to arrays
        assert isinstance(overlay.data, dict)
        assert set(overlay.data.keys()) == {"nose", "body", "tail"}

        # Each bodypart should have correct shape (500 samples, 2D)
        for name, positions in overlay.data.items():
            assert positions.shape == (500, 2), f"{name} has wrong shape"
            assert positions.dtype == np.float64, f"{name} has wrong dtype"

        # Times should be populated from timestamps
        assert overlay.times is not None
        assert overlay.times.shape == (500,)
        assert overlay.times.dtype == np.float64
        assert overlay.times[0] == 0.0
        assert overlay.times[-1] > 0.0

    def test_data_matches_original(self, sample_nwb_with_pose):
        """Test that overlay data matches original NWB PoseEstimation data."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb, read_pose

        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose)

        # Get original data for comparison
        bodyparts, timestamps, _ = read_pose(sample_nwb_with_pose)

        # Verify all bodyparts match
        assert set(overlay.data.keys()) == set(bodyparts.keys())
        for name in overlay.data:
            np.testing.assert_array_almost_equal(overlay.data[name], bodyparts[name])

        np.testing.assert_array_almost_equal(overlay.times, timestamps)

    def test_skeleton_auto_extracted(self, sample_nwb_with_pose):
        """Test that skeleton is automatically extracted from PoseEstimation."""
        from neurospatial.animation.skeleton import Skeleton
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose)

        # Skeleton should be extracted
        assert overlay.skeleton is not None
        assert isinstance(overlay.skeleton, Skeleton)

        # Skeleton should have correct nodes (from fixture: nose, body, tail)
        assert set(overlay.skeleton.nodes) == {"nose", "body", "tail"}

        # Skeleton should have correct edges (nose-body, body-tail)
        # Edges are canonicalized (sorted alphabetically)
        assert len(overlay.skeleton.edges) == 2

    def test_skeleton_matches_read_pose(self, sample_nwb_with_pose):
        """Test that extracted skeleton matches skeleton from read_pose()."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb, read_pose

        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose)
        _, _, skeleton = read_pose(sample_nwb_with_pose)

        # Should be equivalent
        assert overlay.skeleton.name == skeleton.name
        assert overlay.skeleton.nodes == skeleton.nodes
        assert overlay.skeleton.edges == skeleton.edges

    def test_colors_parameter_passed_through(self, sample_nwb_with_pose):
        """Test that colors parameter is passed to BodypartOverlay."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        custom_colors = {"nose": "yellow", "body": "green", "tail": "blue"}
        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose, colors=custom_colors)

        assert overlay.colors == custom_colors

    def test_default_colors_none(self, sample_nwb_with_pose):
        """Test that default colors is None (uses skeleton colors)."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        overlay = bodypart_overlay_from_nwb(sample_nwb_with_pose)

        # Default should be None (defer to skeleton colors)
        assert overlay.colors is None

    def test_pose_estimation_name_forwarded(self, sample_nwb_with_pose):
        """Test that pose_estimation_name parameter is forwarded to read_pose."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        # Request specific pose estimation by name
        overlay = bodypart_overlay_from_nwb(
            sample_nwb_with_pose,
            pose_estimation_name="PoseEstimation",
        )

        # Should return overlay successfully
        assert set(overlay.data.keys()) == {"nose", "body", "tail"}

    def test_additional_kwargs_passed_through(self, sample_nwb_with_pose):
        """Test that additional kwargs are passed to BodypartOverlay."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        # interp is a valid BodypartOverlay parameter
        overlay = bodypart_overlay_from_nwb(
            sample_nwb_with_pose,
            interp="nearest",
        )

        assert overlay.interp == "nearest"

    def test_multiple_parameters_combined(self, sample_nwb_with_pose):
        """Test creating overlay with multiple custom parameters."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        custom_colors = {"nose": "red", "body": "blue", "tail": "green"}
        overlay = bodypart_overlay_from_nwb(
            sample_nwb_with_pose,
            colors=custom_colors,
            interp="nearest",
        )

        assert overlay.colors == custom_colors
        assert overlay.interp == "nearest"
        assert overlay.skeleton is not None

    def test_error_when_no_pose_estimation_found(self, empty_nwb):
        """Test KeyError when no PoseEstimation container found."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        with pytest.raises(KeyError, match=r"No PoseEstimation.*found"):
            bodypart_overlay_from_nwb(empty_nwb)

    def test_error_when_named_pose_not_found(self, sample_nwb_with_pose):
        """Test KeyError when specified PoseEstimation name not found."""
        from neurospatial.io.nwb import bodypart_overlay_from_nwb

        with pytest.raises(KeyError, match=r"not found.*Available"):
            bodypart_overlay_from_nwb(
                sample_nwb_with_pose,
                pose_estimation_name="NonexistentPose",
            )


class TestHeadDirectionOverlayFromNwb:
    """Tests for head_direction_overlay_from_nwb() function."""

    def test_basic_overlay_creation(self, sample_nwb_with_head_direction):
        """Test HeadDirectionOverlay creation from NWB CompassDirection data."""
        from neurospatial.animation.overlays import HeadDirectionOverlay
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        overlay = head_direction_overlay_from_nwb(sample_nwb_with_head_direction)

        # Should return HeadDirectionOverlay instance
        assert isinstance(overlay, HeadDirectionOverlay)

        # Data should have correct shape (1000 samples, 1D angles)
        assert overlay.data.shape == (1000,)
        assert overlay.data.dtype == np.float64

        # Times should be populated from timestamps
        assert overlay.times is not None
        assert overlay.times.shape == (1000,)
        assert overlay.times.dtype == np.float64
        assert overlay.times[0] == 0.0
        assert overlay.times[-1] > 0.0

    def test_data_matches_original(self, sample_nwb_with_head_direction):
        """Test that overlay data matches original NWB CompassDirection data."""
        from neurospatial.io.nwb import (
            head_direction_overlay_from_nwb,
            read_head_direction,
        )

        overlay = head_direction_overlay_from_nwb(sample_nwb_with_head_direction)

        # Get original data for comparison
        angles, timestamps = read_head_direction(sample_nwb_with_head_direction)

        np.testing.assert_array_almost_equal(overlay.data, angles)
        np.testing.assert_array_almost_equal(overlay.times, timestamps)

    def test_color_parameter_passed_through(self, sample_nwb_with_head_direction):
        """Test that color parameter is passed to HeadDirectionOverlay."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        overlay = head_direction_overlay_from_nwb(
            sample_nwb_with_head_direction, color="cyan"
        )

        assert overlay.color == "cyan"

    def test_length_parameter_passed_through(self, sample_nwb_with_head_direction):
        """Test that length parameter is passed to HeadDirectionOverlay."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        overlay = head_direction_overlay_from_nwb(
            sample_nwb_with_head_direction, length=25.0
        )

        assert overlay.length == 25.0

    def test_default_parameters(self, sample_nwb_with_head_direction):
        """Test that default parameters are applied correctly."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        overlay = head_direction_overlay_from_nwb(sample_nwb_with_head_direction)

        # Check defaults match function signature
        assert overlay.color == "yellow"
        assert overlay.length == 15.0

    def test_processing_module_forwarded(self, empty_nwb):
        """Test that processing_module parameter is forwarded to read_head_direction."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        nwbfile = empty_nwb

        # Create CompassDirection in a custom module
        custom_module = nwbfile.create_processing_module(
            name="tracking", description="Tracking data"
        )
        compass_direction = CompassDirection(name="CompassDirection")
        compass_direction.add_spatial_series(
            SpatialSeries(
                name="head_direction",
                data=np.ones(50) * 1.5,  # All pointing in same direction
                timestamps=np.arange(50) / 10.0,
                reference_frame="test",
                unit="radians",
            )
        )
        custom_module.add(compass_direction)

        # Read with explicit module
        overlay = head_direction_overlay_from_nwb(nwbfile, processing_module="tracking")

        assert overlay.data.shape == (50,)
        np.testing.assert_array_almost_equal(overlay.data, np.ones(50) * 1.5)

    def test_compass_name_forwarded(self, empty_nwb):
        """Test that compass_name parameter is forwarded to read_head_direction."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        nwbfile = empty_nwb

        # Create CompassDirection with multiple SpatialSeries
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavioral data"
        )
        compass_direction = CompassDirection(name="CompassDirection")
        compass_direction.add_spatial_series(
            SpatialSeries(
                name="head_angle",
                data=np.ones(30) * 0.5,
                timestamps=np.arange(30) / 10.0,
                reference_frame="test",
                unit="radians",
            )
        )
        compass_direction.add_spatial_series(
            SpatialSeries(
                name="body_angle",
                data=np.ones(40) * 2.0,
                timestamps=np.arange(40) / 10.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass_direction)

        # Request specific series by name
        overlay = head_direction_overlay_from_nwb(nwbfile, compass_name="body_angle")

        # Check we got the 'body_angle' series (40 samples)
        assert overlay.data.shape == (40,)
        np.testing.assert_array_almost_equal(overlay.data, np.ones(40) * 2.0)

    def test_additional_kwargs_passed_through(self, sample_nwb_with_head_direction):
        """Test that additional kwargs are passed to HeadDirectionOverlay."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        # interp is a valid HeadDirectionOverlay parameter
        overlay = head_direction_overlay_from_nwb(
            sample_nwb_with_head_direction,
            interp="nearest",
        )

        assert overlay.interp == "nearest"

    def test_multiple_parameters_combined(self, sample_nwb_with_head_direction):
        """Test creating overlay with multiple custom parameters."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        overlay = head_direction_overlay_from_nwb(
            sample_nwb_with_head_direction,
            color="magenta",
            length=30.0,
            interp="nearest",
        )

        assert overlay.color == "magenta"
        assert overlay.length == 30.0
        assert overlay.interp == "nearest"

    def test_error_when_no_compass_direction_found(self, empty_nwb):
        """Test KeyError when no CompassDirection container found."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        with pytest.raises(KeyError, match=r"No CompassDirection.*found"):
            head_direction_overlay_from_nwb(empty_nwb)

    def test_error_when_named_compass_not_found(self, sample_nwb_with_head_direction):
        """Test KeyError when specified compass_name not found."""
        from neurospatial.io.nwb import head_direction_overlay_from_nwb

        with pytest.raises(KeyError, match=r"not found.*Available"):
            head_direction_overlay_from_nwb(
                sample_nwb_with_head_direction,
                compass_name="NonexistentSeries",
            )
