"""Tests for reference_frames module.

Tests for coordinate transformations between allocentric and egocentric
reference frames, heading computation utilities, and egocentric polar
environment creation.

TDD: These tests are written BEFORE implementation to define expected behavior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestModuleSetup:
    """Test module imports and structure."""

    def test_module_imports(self):
        """Module can be imported successfully."""
        from neurospatial import reference_frames

        assert reference_frames is not None

    def test_all_exports_defined(self):
        """Module defines __all__ with expected exports."""
        from neurospatial import reference_frames

        expected_exports = {
            "EgocentricFrame",
            "allocentric_to_egocentric",
            "egocentric_to_allocentric",
            "compute_egocentric_bearing",
            "compute_egocentric_distance",
            "heading_from_velocity",
            "heading_from_body_orientation",
        }
        assert hasattr(reference_frames, "__all__")
        assert set(reference_frames.__all__) == expected_exports

    def test_module_docstring_exists(self):
        """Module has a docstring."""
        from neurospatial import reference_frames

        assert reference_frames.__doc__ is not None
        assert len(reference_frames.__doc__) > 100  # Non-trivial docstring


class TestEgocentricFrame:
    """Tests for EgocentricFrame dataclass."""

    def test_dataclass_creation(self):
        """EgocentricFrame can be created with position and heading."""
        from neurospatial.reference_frames import EgocentricFrame

        frame = EgocentricFrame(
            position=np.array([10.0, 20.0]),
            heading=np.pi / 4,
        )
        assert_allclose(frame.position, [10.0, 20.0])
        assert_allclose(frame.heading, np.pi / 4)

    def test_to_egocentric_heading_zero_identity(self):
        """With heading=0, egocentric x-axis aligns with allocentric x-axis.

        When the animal faces East (heading=0):
        - A point East of the animal is ahead (+x in egocentric)
        - A point North of the animal is to the left (+y in egocentric)
        """
        from neurospatial.reference_frames import EgocentricFrame

        # Animal at origin facing East
        frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=0.0)

        # Point 10 units East is 10 units ahead
        point_east = np.array([[10.0, 0.0]])
        ego = frame.to_egocentric(point_east)
        assert_allclose(ego, [[10.0, 0.0]], atol=1e-10)

        # Point 10 units North is 10 units left
        point_north = np.array([[0.0, 10.0]])
        ego = frame.to_egocentric(point_north)
        assert_allclose(ego, [[0.0, 10.0]], atol=1e-10)

    def test_to_egocentric_heading_pi_half(self):
        """With heading=π/2, egocentric x-axis aligns with allocentric y-axis.

        When the animal faces North (heading=π/2):
        - A point North of the animal is ahead (+x in egocentric)
        - A point East of the animal is to the right (-y in egocentric)
        """
        from neurospatial.reference_frames import EgocentricFrame

        # Animal at origin facing North
        frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=np.pi / 2)

        # Point 10 units North is 10 units ahead
        point_north = np.array([[0.0, 10.0]])
        ego = frame.to_egocentric(point_north)
        assert_allclose(ego, [[10.0, 0.0]], atol=1e-10)

        # Point 10 units East is 10 units right (-y)
        point_east = np.array([[10.0, 0.0]])
        ego = frame.to_egocentric(point_east)
        assert_allclose(ego, [[0.0, -10.0]], atol=1e-10)

    def test_to_allocentric_inverse(self):
        """to_allocentric is the inverse of to_egocentric."""
        from neurospatial.reference_frames import EgocentricFrame

        frame = EgocentricFrame(
            position=np.array([5.0, 3.0]),
            heading=np.pi / 3,  # 60 degrees
        )

        # Transform to egocentric and back
        allocentric_orig = np.array([[10.0, 20.0], [30.0, 40.0]])
        egocentric = frame.to_egocentric(allocentric_orig)
        allocentric_recovered = frame.to_allocentric(egocentric)

        assert_allclose(allocentric_recovered, allocentric_orig, atol=1e-10)

    def test_round_trip_preserves_coordinates(self):
        """Round-trip transformation preserves coordinates for various headings."""
        from neurospatial.reference_frames import EgocentricFrame

        rng = np.random.default_rng(42)

        for heading in [0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2, 3 * np.pi / 4]:
            position = rng.uniform(-100, 100, size=2)
            frame = EgocentricFrame(position=position, heading=heading)

            allocentric = rng.uniform(-100, 100, size=(10, 2))
            egocentric = frame.to_egocentric(allocentric)
            recovered = frame.to_allocentric(egocentric)

            assert_allclose(recovered, allocentric, atol=1e-10)


class TestAllocentricToEgocentric:
    """Tests for allocentric_to_egocentric batch function."""

    def test_batch_transform_multiple_timepoints(self):
        """Batch transform handles multiple timepoints correctly."""
        from neurospatial.reference_frames import allocentric_to_egocentric

        # 3 timepoints, 2 landmarks
        landmarks = np.array([[10.0, 0.0], [0.0, 10.0]])  # Shape: (2, 2)
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])  # Shape: (3, 2)
        headings = np.array([0.0, np.pi / 2, np.pi])  # Shape: (3,)

        ego = allocentric_to_egocentric(landmarks, positions, headings)

        # Output shape: (n_time, n_points, 2) = (3, 2, 2)
        assert ego.shape == (3, 2, 2)

        # At t=0: animal at origin, facing East (heading=0)
        # Landmark (10, 0) is 10 units ahead
        assert_allclose(ego[0, 0], [10.0, 0.0], atol=1e-10)
        # Landmark (0, 10) is 10 units left
        assert_allclose(ego[0, 1], [0.0, 10.0], atol=1e-10)

    def test_broadcasting_2d_points_to_3d(self):
        """2D point input broadcasts to 3D output."""
        from neurospatial.reference_frames import allocentric_to_egocentric

        # Same landmarks transformed at each timepoint
        landmarks = np.array([[10.0, 0.0]])  # Shape: (1, 2)
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Shape: (2, 2)
        headings = np.array([0.0, np.pi / 2])  # Shape: (2,)

        ego = allocentric_to_egocentric(landmarks, positions, headings)

        # Output: (n_time, n_points, 2) = (2, 1, 2)
        assert ego.shape == (2, 1, 2)

        # t=0, heading=0: (10, 0) is ahead
        assert_allclose(ego[0, 0], [10.0, 0.0], atol=1e-10)

        # t=1, heading=π/2: (10, 0) is right
        assert_allclose(ego[1, 0], [0.0, -10.0], atol=1e-10)

    def test_shape_validation_error_messages(self):
        """Invalid shapes produce clear error messages."""
        from neurospatial.reference_frames import allocentric_to_egocentric

        points = np.array([10.0, 0.0])  # Wrong: 1D instead of 2D
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="points"):
            allocentric_to_egocentric(points, positions, headings)

    def test_positions_headings_length_mismatch(self):
        """Positions and headings must have matching length."""
        from neurospatial.reference_frames import allocentric_to_egocentric

        points = np.array([[10.0, 0.0]])
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2 timepoints
        headings = np.array([0.0])  # 1 timepoint

        with pytest.raises(ValueError, match=r"positions.*headings"):
            allocentric_to_egocentric(points, positions, headings)


class TestEgocentricToAllocentric:
    """Tests for egocentric_to_allocentric batch function."""

    def test_inverse_of_allocentric_to_egocentric(self):
        """egocentric_to_allocentric is the inverse of allocentric_to_egocentric."""
        from neurospatial.reference_frames import (
            allocentric_to_egocentric,
            egocentric_to_allocentric,
        )

        rng = np.random.default_rng(42)

        points = rng.uniform(-50, 50, size=(5, 2))
        positions = rng.uniform(-10, 10, size=(10, 2))
        headings = rng.uniform(-np.pi, np.pi, size=10)

        ego = allocentric_to_egocentric(points, positions, headings)
        recovered = egocentric_to_allocentric(ego, positions, headings)

        # Output should broadcast back to match input shape
        assert recovered.shape == (10, 5, 2)

        # Each recovered point should match original
        for t in range(10):
            assert_allclose(recovered[t], points, atol=1e-10)


class TestComputeEgocentricBearing:
    """Tests for compute_egocentric_bearing function."""

    def test_bearing_zero_when_target_ahead(self):
        """Bearing is 0 when target is directly ahead."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # Target at (10, 0), animal at origin facing East
        target = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(target, position, heading)
        assert_allclose(bearing, [[0.0]], atol=1e-10)

    def test_bearing_pi_half_when_target_left(self):
        """Bearing is π/2 when target is to the left."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # Target at (0, 10), animal at origin facing East
        # Target is 90 degrees left
        target = np.array([[0.0, 10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(target, position, heading)
        assert_allclose(bearing, [[np.pi / 2]], atol=1e-10)

    def test_bearing_negative_pi_half_when_target_right(self):
        """Bearing is -π/2 when target is to the right."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # Target at (0, -10), animal at origin facing East
        target = np.array([[0.0, -10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(target, position, heading)
        assert_allclose(bearing, [[-np.pi / 2]], atol=1e-10)

    def test_bearing_behind_target(self):
        """Bearing is ±π when target is behind."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # Target at (-10, 0), animal at origin facing East
        target = np.array([[-10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing = compute_egocentric_bearing(target, position, heading)
        # Should be π or -π (both valid)
        assert np.abs(np.abs(bearing[0, 0]) - np.pi) < 1e-10

    def test_angle_wrapping_near_pi(self):
        """Angles wrap correctly near ±π boundary."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # Test points slightly behind, should still give angles near ±π
        target1 = np.array([[-10.0, 0.1]])  # Slightly to the left of behind
        target2 = np.array([[-10.0, -0.1]])  # Slightly to the right of behind
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        bearing1 = compute_egocentric_bearing(target1, position, heading)
        bearing2 = compute_egocentric_bearing(target2, position, heading)

        # Both should be close to π in absolute value
        assert np.abs(bearing1[0, 0]) > 0.99 * np.pi
        assert np.abs(bearing2[0, 0]) > 0.99 * np.pi

        # bearing1 (slightly left) should be positive, bearing2 negative
        assert bearing1[0, 0] > 0  # Slightly left of behind
        assert bearing2[0, 0] < 0  # Slightly right of behind

    def test_batch_multiple_targets_and_times(self):
        """Handles multiple targets and timepoints."""
        from neurospatial.reference_frames import compute_egocentric_bearing

        # 3 targets, 5 timepoints
        targets = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
        positions = np.zeros((5, 2))
        headings = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2])

        bearing = compute_egocentric_bearing(targets, positions, headings)

        assert bearing.shape == (5, 3)


class TestComputeEgocentricDistance:
    """Tests for compute_egocentric_distance function."""

    def test_euclidean_distance(self):
        """Euclidean distance computed correctly."""
        from neurospatial.reference_frames import compute_egocentric_distance

        targets = np.array([[10.0, 0.0], [0.0, 10.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])  # Heading doesn't affect distance

        distances = compute_egocentric_distance(
            targets, position, heading, metric="euclidean"
        )

        assert distances.shape == (1, 2)
        assert_allclose(distances, [[10.0, 10.0]], atol=1e-10)

    def test_geodesic_distance_with_environment(self):
        """Geodesic distance uses environment's distance_field."""
        from neurospatial import Environment
        from neurospatial.reference_frames import compute_egocentric_distance

        # Create a simple 2D grid environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, size=(500, 2))
        env = Environment.from_samples(positions, bin_size=5.0)

        # Targets and position within the environment
        targets = np.array([[50.0, 50.0], [80.0, 80.0]])
        position = np.array([[10.0, 10.0]])
        heading = np.array([0.0])

        distances = compute_egocentric_distance(
            targets, position, heading, metric="geodesic", env=env
        )

        assert distances.shape == (1, 2)
        # Geodesic distances should be finite and positive
        assert np.all(np.isfinite(distances))
        assert np.all(distances > 0)
        # Geodesic distances should be within reasonable range of Euclidean
        # (can be slightly different due to graph discretization)
        eucl = np.linalg.norm(targets - position, axis=1)
        # Allow for discretization differences (within 20% or 10 units)
        assert np.all(np.abs(distances[0] - eucl) < np.maximum(0.2 * eucl, 10))

    def test_invalid_metric_raises(self):
        """Invalid metric raises ValueError."""
        from neurospatial.reference_frames import compute_egocentric_distance

        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        with pytest.raises(ValueError, match="metric"):
            compute_egocentric_distance(targets, position, heading, metric="invalid")

    def test_geodesic_without_env_raises(self):
        """Geodesic metric without environment raises error."""
        from neurospatial.reference_frames import compute_egocentric_distance

        targets = np.array([[10.0, 0.0]])
        position = np.array([[0.0, 0.0]])
        heading = np.array([0.0])

        with pytest.raises(ValueError, match="env"):
            compute_egocentric_distance(targets, position, heading, metric="geodesic")


class TestHeadingFromVelocity:
    """Tests for heading_from_velocity function."""

    def test_smooth_trajectory(self):
        """Heading from smooth trajectory matches expected direction."""
        from neurospatial.reference_frames import heading_from_velocity

        # Straight line moving East
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([t * 10, np.zeros_like(t)])  # x = 0 to 100

        headings = heading_from_velocity(positions, dt=t[1] - t[0])

        # All headings should be 0 (East), except possibly at boundaries
        assert_allclose(headings[10:-10], 0.0, atol=0.1)

    def test_heading_northward(self):
        """Heading is π/2 for northward motion."""
        from neurospatial.reference_frames import heading_from_velocity

        # Straight line moving North
        t = np.linspace(0, 10, 100)
        positions = np.column_stack([np.zeros_like(t), t * 10])

        headings = heading_from_velocity(positions, dt=t[1] - t[0])

        # All headings should be π/2 (North)
        assert_allclose(headings[10:-10], np.pi / 2, atol=0.1)

    def test_stationary_periods_interpolated(self):
        """Stationary periods produce smoothly interpolated headings."""
        from neurospatial.reference_frames import heading_from_velocity

        # Moving, then stationary, then moving again
        n = 100
        positions = np.zeros((n, 2))

        # First 30 points: moving East
        positions[:30, 0] = np.arange(30)
        # Points 30-70: stationary at x=30
        positions[30:70, 0] = 30
        # Points 70-100: moving East again
        positions[70:, 0] = 30 + np.arange(30)

        headings = heading_from_velocity(
            positions, dt=0.1, min_speed=0.5, smoothing_sigma=2.0
        )

        # Stationary period should have smoothly interpolated headings
        # (not NaN or sudden jumps)
        assert not np.any(np.isnan(headings))

        # Check no discontinuities (large jumps)
        diff = np.abs(np.diff(headings))
        # Wrap around for circular comparison
        diff = np.minimum(diff, 2 * np.pi - diff)
        assert np.all(diff < 0.5)  # No large jumps

    def test_all_speeds_low_warning_and_nan(self):
        """Warning when all speeds below threshold, returns NaN."""
        from neurospatial.reference_frames import heading_from_velocity

        # Barely moving trajectory
        positions = np.random.RandomState(42).randn(100, 2) * 0.001

        with pytest.warns(UserWarning, match="speed"):
            headings = heading_from_velocity(positions, dt=0.1, min_speed=10.0)

        assert np.all(np.isnan(headings))

    def test_n_time_less_than_2_raises(self):
        """n_time < 2 raises ValueError."""
        from neurospatial.reference_frames import heading_from_velocity

        positions = np.array([[0.0, 0.0]])  # Only 1 point

        with pytest.raises(ValueError, match="at least 2"):
            heading_from_velocity(positions, dt=0.1)

    def test_smoothing_reduces_noise(self):
        """Larger smoothing sigma reduces heading noise."""
        from neurospatial.reference_frames import heading_from_velocity

        # Noisy trajectory moving mostly East
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 100)
        noise = rng.normal(0, 0.5, size=(100, 2))
        positions = np.column_stack([t * 10, np.zeros_like(t)]) + noise

        headings_no_smooth = heading_from_velocity(positions, dt=0.1, smoothing_sigma=0)
        headings_smooth = heading_from_velocity(positions, dt=0.1, smoothing_sigma=5)

        # Smoothed version should have less variance
        var_no_smooth = np.nanvar(headings_no_smooth[10:-10])
        var_smooth = np.nanvar(headings_smooth[10:-10])
        assert var_smooth < var_no_smooth


class TestHeadingFromBodyOrientation:
    """Tests for heading_from_body_orientation function."""

    def test_with_valid_keypoints(self):
        """Heading computed correctly from valid nose/tail keypoints."""
        from neurospatial.reference_frames import heading_from_body_orientation

        n = 50
        # Nose at (10, 0), tail at (0, 0) → heading = 0 (East)
        nose = np.tile([10.0, 0.0], (n, 1))
        tail = np.tile([0.0, 0.0], (n, 1))

        headings = heading_from_body_orientation(nose, tail)

        assert headings.shape == (n,)
        assert_allclose(headings, 0.0, atol=1e-10)

    def test_heading_northward(self):
        """Heading is π/2 for nose north of tail."""
        from neurospatial.reference_frames import heading_from_body_orientation

        n = 50
        # Nose at (0, 10), tail at (0, 0) → heading = π/2 (North)
        nose = np.tile([0.0, 10.0], (n, 1))
        tail = np.tile([0.0, 0.0], (n, 1))

        headings = heading_from_body_orientation(nose, tail)

        assert_allclose(headings, np.pi / 2, atol=1e-10)

    def test_nan_keypoints_interpolated(self):
        """NaN keypoints are interpolated without discontinuities."""
        from neurospatial.reference_frames import heading_from_body_orientation

        n = 100
        # Steady heading = 0 (East) with some NaN gaps
        nose = np.tile([10.0, 0.0], (n, 1))
        tail = np.tile([0.0, 0.0], (n, 1))

        # Add NaN gaps
        nose[30:40] = np.nan
        nose[60:65] = np.nan

        headings = heading_from_body_orientation(nose, tail)

        # No NaNs in output
        assert not np.any(np.isnan(headings))

        # All headings should still be ~0
        assert_allclose(headings, 0.0, atol=0.1)

    def test_nan_interpolation_near_pi_boundary(self):
        """NaN interpolation handles ±π boundary correctly."""
        from neurospatial.reference_frames import heading_from_body_orientation

        n = 100
        # Heading smoothly crosses from +π to -π
        angles = np.linspace(0.9 * np.pi, 1.1 * np.pi, n)
        # Wrap to (-π, π]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        nose = np.column_stack([np.cos(angles), np.sin(angles)])
        tail = np.zeros((n, 2))

        # Add NaN gap right at the boundary crossing
        nose[45:55] = np.nan

        headings = heading_from_body_orientation(nose, tail)

        # No NaNs in output
        assert not np.any(np.isnan(headings))

        # Check smooth transition (no large jumps)
        diff = np.abs(np.diff(headings))
        diff = np.minimum(diff, 2 * np.pi - diff)  # Circular distance
        assert np.all(diff < 0.2)

    def test_all_nan_raises(self):
        """All-NaN keypoints raises ValueError."""
        from neurospatial.reference_frames import heading_from_body_orientation

        nose = np.full((50, 2), np.nan)
        tail = np.full((50, 2), np.nan)

        with pytest.raises(ValueError, match="NaN"):
            heading_from_body_orientation(nose, tail)


class TestComputeEgocentricDistanceFunctions:
    """Tests for distance utility functions."""

    def test_euclidean_multiple_timepoints(self):
        """Euclidean distance across multiple timepoints."""
        from neurospatial.reference_frames import compute_egocentric_distance

        # Single target, multiple animal positions
        targets = np.array([[50.0, 50.0]])
        positions = np.array([[0.0, 0.0], [25.0, 25.0], [50.0, 50.0]])
        headings = np.zeros(3)

        distances = compute_egocentric_distance(
            targets, positions, headings, metric="euclidean"
        )

        assert distances.shape == (3, 1)
        expected = [
            np.sqrt(50**2 + 50**2),  # Distance from origin
            np.sqrt(25**2 + 25**2),  # Distance from (25, 25)
            0.0,  # At the target
        ]
        assert_allclose(distances[:, 0], expected, atol=1e-10)
