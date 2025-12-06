"""Tests for neurospatial.ops.egocentric module.

These tests verify the new import path `neurospatial.ops.egocentric` works correctly.
The module provides coordinate reference frame transformations between allocentric
(world-centered) and egocentric (animal-centered) coordinate systems.
"""

import numpy as np
import pytest

# Test imports from the new location
from neurospatial.ops.egocentric import (
    EgocentricFrame,
    allocentric_to_egocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    egocentric_to_allocentric,
    heading_from_body_orientation,
    heading_from_velocity,
)


class TestEgocentricFrame:
    """Tests for EgocentricFrame dataclass."""

    def test_to_egocentric_facing_east(self):
        """Test transformation when facing East (heading=0)."""
        frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=0.0)
        # Point 10 units East is 10 units ahead
        result = frame.to_egocentric(np.array([[10.0, 0.0]]))
        np.testing.assert_allclose(result, [[10.0, 0.0]])

    def test_to_egocentric_facing_north(self):
        """Test transformation when facing North (heading=pi/2)."""
        frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=np.pi / 2)
        # Point 10 units East is 10 units to the right
        result = frame.to_egocentric(np.array([[10.0, 0.0]]))
        np.testing.assert_allclose(result, [[0.0, -10.0]], atol=1e-10)

    def test_roundtrip(self):
        """Test round-trip transformation preserves coordinates."""
        frame = EgocentricFrame(position=np.array([5.0, 3.0]), heading=np.pi / 4)
        allocentric = np.array([[10.0, 7.0], [2.0, -1.0]])
        egocentric = frame.to_egocentric(allocentric)
        recovered = frame.to_allocentric(egocentric)
        np.testing.assert_allclose(recovered, allocentric, atol=1e-10)


class TestAllocentricToEgocentric:
    """Tests for allocentric_to_egocentric function."""

    def test_basic_transform(self):
        """Test basic transformation at multiple timepoints."""
        landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # 2 landmarks
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Animal at origin
        headings = np.array([0.0, np.pi / 2])  # Facing East, then North
        ego = allocentric_to_egocentric(landmarks, positions, headings)
        assert ego.shape == (2, 2, 2)

        # At t=0 (facing East), landmark (10, 0) is ahead
        np.testing.assert_allclose(ego[0, 0], [10.0, 0.0], atol=1e-10)
        # At t=1 (facing North), landmark (10, 0) is to the right
        np.testing.assert_allclose(ego[1, 0], [0.0, -10.0], atol=1e-10)

    def test_invalid_points_shape(self):
        """Test error on invalid points shape."""
        with pytest.raises(ValueError, match="Cannot transform points"):
            allocentric_to_egocentric(
                np.array([10.0]),  # 1D, invalid
                np.array([[0.0, 0.0]]),
                np.array([0.0]),
            )

    def test_positions_headings_mismatch(self):
        """Test error when positions and headings lengths differ."""
        with pytest.raises(ValueError, match="length mismatch"):
            allocentric_to_egocentric(
                np.array([[10.0, 0.0]]),
                np.array([[0.0, 0.0], [1.0, 1.0]]),  # 2 positions
                np.array([0.0]),  # 1 heading
            )


class TestEgocentricToAllocentric:
    """Tests for egocentric_to_allocentric function."""

    def test_roundtrip(self):
        """Test that allocentric->egocentric->allocentric preserves coordinates."""
        landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])
        positions = np.array([[5.0, 5.0]])
        headings = np.array([np.pi / 4])
        ego = allocentric_to_egocentric(landmarks, positions, headings)
        recovered = egocentric_to_allocentric(ego, positions, headings)
        np.testing.assert_allclose(recovered[0], landmarks, atol=1e-10)


class TestComputeEgocentricBearing:
    """Tests for compute_egocentric_bearing function."""

    def test_ahead_bearing_zero(self):
        """Test that target directly ahead has bearing 0."""
        target = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])  # Facing East
        bearing = compute_egocentric_bearing(target, position, heading)
        np.testing.assert_allclose(bearing, [[0.0]], atol=1e-10)

    def test_left_bearing_pi_half(self):
        """Test that target to the left has bearing pi/2."""
        target = np.array([[0.0, 10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])  # Facing East
        bearing = compute_egocentric_bearing(target, position, heading)
        np.testing.assert_allclose(bearing, [[np.pi / 2]], atol=1e-10)

    def test_right_bearing_negative_pi_half(self):
        """Test that target to the right has bearing -pi/2."""
        target = np.array([[0.0, -10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])  # Facing East
        bearing = compute_egocentric_bearing(target, position, heading)
        np.testing.assert_allclose(bearing, [[-np.pi / 2]], atol=1e-10)


class TestComputeEgocentricDistance:
    """Tests for compute_egocentric_distance function."""

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        targets = np.array([[10.0, 0.0], [0.0, 10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])
        distances = compute_egocentric_distance(
            targets, position, heading, metric="euclidean"
        )
        np.testing.assert_allclose(distances, [[10.0, 10.0]])

    def test_invalid_metric(self):
        """Test error on invalid metric."""
        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])
        with pytest.raises(ValueError, match="Invalid distance metric"):
            compute_egocentric_distance(targets, position, heading, metric="invalid")

    def test_geodesic_without_env(self):
        """Test error when geodesic metric used without environment."""
        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])
        with pytest.raises(ValueError, match="missing environment"):
            compute_egocentric_distance(targets, position, heading, metric="geodesic")


class TestHeadingFromVelocity:
    """Tests for heading_from_velocity function."""

    def test_moving_east(self):
        """Test heading when moving East."""
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([t * 10, np.zeros_like(t)])
        headings = heading_from_velocity(positions, dt=t[1] - t[0])
        # Middle samples should be close to 0 (facing East)
        np.testing.assert_allclose(headings[10:-10], 0.0, atol=0.1)

    def test_moving_north(self):
        """Test heading when moving North."""
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([np.zeros_like(t), t * 10])
        headings = heading_from_velocity(positions, dt=t[1] - t[0])
        # Middle samples should be close to pi/2 (facing North)
        np.testing.assert_allclose(headings[10:-10], np.pi / 2, atol=0.1)

    def test_insufficient_samples(self):
        """Test error when fewer than 2 samples provided."""
        with pytest.raises(ValueError, match="insufficient trajectory data"):
            heading_from_velocity(np.array([[0.0, 0.0]]), dt=0.1)


class TestHeadingFromBodyOrientation:
    """Tests for heading_from_body_orientation function."""

    def test_facing_east(self):
        """Test heading from nose/tail pointing East."""
        n = 50
        nose = np.tile([10.0, 0.0], (n, 1))
        tail = np.tile([0.0, 0.0], (n, 1))
        headings = heading_from_body_orientation(nose, tail)
        np.testing.assert_allclose(headings, 0.0)

    def test_facing_north(self):
        """Test heading from nose/tail pointing North."""
        n = 50
        nose = np.tile([0.0, 10.0], (n, 1))
        tail = np.tile([0.0, 0.0], (n, 1))
        headings = heading_from_body_orientation(nose, tail)
        np.testing.assert_allclose(headings, np.pi / 2)

    def test_all_nan_raises(self):
        """Test error when all keypoints are NaN."""
        n = 10
        nose = np.full((n, 2), np.nan)
        tail = np.full((n, 2), np.nan)
        with pytest.raises(ValueError, match="all keypoints are NaN"):
            heading_from_body_orientation(nose, tail)


class TestImportsFromOps:
    """Tests for imports from neurospatial.ops module."""

    def test_import_from_ops_init(self):
        """Test that egocentric functions can be imported from ops/__init__.py."""
        from neurospatial.ops import (
            EgocentricFrame,
            allocentric_to_egocentric,
            compute_egocentric_bearing,
            compute_egocentric_distance,
            egocentric_to_allocentric,
            heading_from_body_orientation,
            heading_from_velocity,
        )

        # Verify these are the correct types
        assert EgocentricFrame is not None
        assert callable(allocentric_to_egocentric)
        assert callable(compute_egocentric_bearing)
        assert callable(compute_egocentric_distance)
        assert callable(egocentric_to_allocentric)
        assert callable(heading_from_body_orientation)
        assert callable(heading_from_velocity)
