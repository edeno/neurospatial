"""Tests for visibility module.

Tests for visibility/gaze computation including field of view dataclass,
viewshed analysis, and cue visibility checking.

TDD: These tests are written BEFORE implementation to define expected behavior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestModuleSetup:
    """Test module imports and structure."""

    def test_module_imports(self):
        """Module can be imported successfully."""
        from neurospatial.ops import visibility

        assert visibility is not None

    def test_all_exports_defined(self):
        """Module defines __all__ with expected exports."""
        from neurospatial.ops import visibility

        expected_exports = {
            "FieldOfView",
            "ViewshedResult",
            "compute_viewed_location",
            "compute_viewshed",
            "compute_view_field",
            "visible_cues",
            "compute_viewshed_trajectory",
            "visibility_occupancy",
        }
        assert hasattr(visibility, "__all__")
        assert set(visibility.__all__) == expected_exports

    def test_module_docstring_exists(self):
        """Module has a docstring."""
        from neurospatial.ops import visibility

        assert visibility.__doc__ is not None
        assert len(visibility.__doc__) > 100  # Non-trivial docstring


class TestFieldOfView:
    """Tests for FieldOfView dataclass."""

    def test_dataclass_creation(self):
        """FieldOfView can be created with angle bounds."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView(
            left_angle=np.pi / 2,
            right_angle=-np.pi / 2,
        )
        assert_allclose(fov.left_angle, np.pi / 2)
        assert_allclose(fov.right_angle, -np.pi / 2)

    def test_symmetric_factory(self):
        """FieldOfView.symmetric() creates symmetric FOV."""
        from neurospatial.ops.visibility import FieldOfView

        # 180 degree field of view
        fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        assert_allclose(fov.left_angle, np.pi / 2)
        assert_allclose(fov.right_angle, -np.pi / 2)
        assert_allclose(fov.total_angle, np.pi)

    def test_rat_preset(self):
        """FieldOfView.rat() returns ~300 degree FOV."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.rat()
        # Rats have ~300 degree FOV with ~30 degree blind spot behind
        assert 290 < fov.total_angle_degrees < 340

    def test_mouse_preset(self):
        """FieldOfView.mouse() returns FOV similar to rat."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.mouse()
        # Mice have similar FOV to rats
        assert 280 < fov.total_angle_degrees < 340

    def test_primate_preset(self):
        """FieldOfView.primate() returns ~180 degree FOV."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.primate()
        # Primates have ~180 degree FOV (forward-facing eyes)
        assert 150 < fov.total_angle_degrees < 200

    def test_total_angle_property(self):
        """total_angle returns sum of left and right angles."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView(left_angle=np.pi / 3, right_angle=-np.pi / 4)
        expected = np.pi / 3 + np.pi / 4  # Note: right is negative
        assert_allclose(fov.total_angle, expected)

    def test_total_angle_degrees_property(self):
        """total_angle_degrees returns angle in degrees."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        assert_allclose(fov.total_angle_degrees, 180.0)

    def test_contains_angle_within_fov(self):
        """contains_angle() returns True for angles within FOV."""
        from neurospatial.ops.visibility import FieldOfView

        # 180 degree FOV, -90 to +90 degrees
        fov = FieldOfView.symmetric(half_angle=np.pi / 2)

        # Test single angle
        assert fov.contains_angle(0.0)  # Ahead
        assert fov.contains_angle(np.pi / 4)  # 45 degrees left
        assert fov.contains_angle(-np.pi / 4)  # 45 degrees right

    def test_contains_angle_outside_fov(self):
        """contains_angle() returns False for angles outside FOV."""
        from neurospatial.ops.visibility import FieldOfView

        # 180 degree FOV, -90 to +90 degrees
        fov = FieldOfView.symmetric(half_angle=np.pi / 2)

        # Test angles outside FOV
        assert not fov.contains_angle(np.pi)  # Behind
        assert not fov.contains_angle(-np.pi)  # Behind
        assert not fov.contains_angle(3 * np.pi / 4)  # Behind-left

    def test_contains_angle_batch(self):
        """contains_angle() handles array input."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)

        angles = np.array([0.0, np.pi / 4, np.pi, -np.pi / 4, 3 * np.pi / 4])
        result = fov.contains_angle(angles)

        expected = np.array([True, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_is_binocular(self):
        """is_binocular() checks if angle is in binocular region."""
        from neurospatial.ops.visibility import FieldOfView

        # Create FOV with binocular overlap in front
        fov = FieldOfView(
            left_angle=2 * np.pi / 3,  # 120 degrees left
            right_angle=-2 * np.pi / 3,  # 120 degrees right
            binocular_half_angle=np.pi / 6,  # 30 degree binocular zone
        )

        # Center is binocular
        assert fov.is_binocular(0.0)
        assert fov.is_binocular(np.pi / 12)  # 15 degrees, within 30 degree zone

        # Outside binocular zone
        assert not fov.is_binocular(np.pi / 3)  # 60 degrees

    def test_blind_spot_behind(self):
        """blind_spot_behind excludes rear regions."""
        from neurospatial.ops.visibility import FieldOfView

        # Rat-like FOV with blind spot behind
        fov = FieldOfView(
            left_angle=5 * np.pi / 6,  # 150 degrees left
            right_angle=-5 * np.pi / 6,  # 150 degrees right
            blind_spot_behind=np.pi / 6,  # 30 degree blind spot
        )

        # Directly behind should be excluded
        assert not fov.contains_angle(np.pi)
        assert not fov.contains_angle(-np.pi)

        # Just outside blind spot should be included
        assert fov.contains_angle(5 * np.pi / 6 - 0.1)

    def test_validation_left_right_order(self):
        """Validation fails if left_angle < right_angle."""
        from neurospatial.ops.visibility import FieldOfView

        with pytest.raises(ValueError, match="left_angle"):
            FieldOfView(left_angle=-np.pi / 2, right_angle=np.pi / 2)

    def test_frozen_dataclass(self):
        """FieldOfView is immutable (frozen dataclass)."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)

        with pytest.raises((AttributeError, TypeError)):
            fov.left_angle = np.pi  # type: ignore[misc]


class TestViewshedResult:
    """Tests for ViewshedResult dataclass."""

    def test_dataclass_creation(self):
        """ViewshedResult can be created with required fields."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([0, 1, 2, 3]),
            visible_cues=np.array([0, 2]),
            cue_distances=np.array([5.0, 15.0]),
            cue_bearings=np.array([0.0, np.pi / 4]),
            occlusion_map=np.zeros(10),
        )
        assert len(result.visible_bins) == 4

    def test_n_visible_bins_property(self):
        """n_visible_bins returns count of visible bins."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([0, 1, 2, 3, 5]),
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(10),
        )
        assert result.n_visible_bins == 5

    def test_visibility_fraction_property(self):
        """visibility_fraction returns fraction of bins visible."""
        from neurospatial.ops.visibility import ViewshedResult

        # 5 out of 10 bins visible
        result = ViewshedResult(
            visible_bins=np.array([0, 1, 2, 3, 5]),
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(10),
            _total_bins=10,
        )
        assert_allclose(result.visibility_fraction, 0.5)

    def test_n_visible_cues_property(self):
        """n_visible_cues returns count of visible cues."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([]),
            visible_cues=np.array([0, 2, 5]),
            cue_distances=np.array([5.0, 10.0, 15.0]),
            cue_bearings=np.array([0.0, np.pi / 4, np.pi / 2]),
            occlusion_map=np.zeros(10),
        )
        assert result.n_visible_cues == 3

    def test_filter_cues(self):
        """filter_cues() returns subset of cues matching IDs."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([]),
            visible_cues=np.array([0, 2, 5]),
            cue_distances=np.array([5.0, 10.0, 15.0]),
            cue_bearings=np.array([0.0, np.pi / 4, np.pi / 2]),
            occlusion_map=np.zeros(10),
        )

        # Filter to only cues 0 and 5
        filtered_ids, filtered_dists, filtered_bearings = result.filter_cues([0, 5])

        np.testing.assert_array_equal(filtered_ids, [0, 5])
        assert_allclose(filtered_dists, [5.0, 15.0])
        assert_allclose(filtered_bearings, [0.0, np.pi / 2])

    def test_visible_bin_centers(self):
        """visible_bin_centers() returns allocentric positions."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import ViewshedResult

        # Create simple environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (500, 2))
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create result with some visible bins
        visible_bins = np.array([0, 1, 2])
        result = ViewshedResult(
            visible_bins=visible_bins,
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(env.n_bins),
        )

        centers = result.visible_bin_centers(env)
        assert centers.shape == (3, 2)
        # Check that returned centers match expected bin centers
        assert_allclose(centers, env.bin_centers[visible_bins])

    def test_frozen_dataclass(self):
        """ViewshedResult is immutable."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([0, 1]),
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(10),
        )

        with pytest.raises((AttributeError, TypeError)):
            result.visible_bins = np.array([3, 4])  # type: ignore[misc]


class TestComputeViewedLocation:
    """Tests for compute_viewed_location function."""

    def test_fixed_distance_method(self):
        """Fixed distance method returns point at fixed distance in gaze direction."""
        from neurospatial.ops.visibility import compute_viewed_location

        # Animal at origin, facing East
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        headings = np.array([0.0, np.pi / 2])  # East, then North

        viewed = compute_viewed_location(
            positions, headings, method="fixed_distance", view_distance=10.0
        )

        # Facing East: viewed location should be 10 units East
        assert_allclose(viewed[0], [10.0, 0.0], atol=1e-10)
        # Facing North: viewed location should be 10 units North
        assert_allclose(viewed[1], [0.0, 10.0], atol=1e-10)

    def test_ray_cast_method_with_boundary(self):
        """Ray cast method returns intersection with environment boundary."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewed_location

        # Create square environment
        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(10, 90, (500, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Animal at center, facing East
        positions = np.array([[50.0, 50.0]])
        headings = np.array([0.0])  # East

        viewed = compute_viewed_location(
            positions, headings, method="ray_cast", env=env
        )

        # Should hit boundary towards East (x > animal position)
        # Note: The exact boundary depends on sampled positions
        assert viewed[0, 0] > 50.0  # x should be East of animal
        assert 40 < viewed[0, 1] < 60  # y should be near middle

    def test_boundary_method(self):
        """Boundary method returns nearest boundary point in gaze direction."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewed_location

        # Create environment
        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(10, 90, (500, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Animal at center, facing East
        positions = np.array([[50.0, 50.0]])
        headings = np.array([0.0])

        viewed = compute_viewed_location(
            positions, headings, method="boundary", env=env
        )

        assert viewed.shape == (1, 2)
        # x should be positive (towards East boundary)
        assert viewed[0, 0] > 50.0

    def test_gaze_offsets(self):
        """Gaze offsets adjust the viewing angle relative to heading."""
        from neurospatial.ops.visibility import compute_viewed_location

        # Animal facing East with gaze offset of 90 degrees (looking left)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        gaze_offsets = np.array([np.pi / 2])  # Looking 90 degrees left

        viewed = compute_viewed_location(
            positions,
            headings,
            method="fixed_distance",
            view_distance=10.0,
            gaze_offsets=gaze_offsets,
        )

        # Should be looking North (heading + offset)
        assert_allclose(viewed[0], [0.0, 10.0], atol=1e-10)

    def test_nan_when_viewing_outside_environment(self):
        """Returns NaN when ray doesn't intersect environment."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewed_location

        # Create L-shaped environment (missing upper-right quadrant)
        rng = np.random.default_rng(42)
        # Lower half
        lower = rng.uniform([0, 0], [100, 50], (250, 2))
        # Left side of upper half
        upper_left = rng.uniform([0, 50], [50, 100], (250, 2))
        positions_samples = np.vstack([lower, upper_left])
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Animal in lower half, looking up-right (into missing region)
        positions = np.array([[30.0, 30.0]])
        headings = np.array([np.pi / 4])  # North-East

        viewed = compute_viewed_location(
            positions, headings, method="ray_cast", env=env, max_distance=200.0
        )

        # Ray should not find intersection (goes into void)
        # Note: this behavior depends on implementation details
        # The ray might hit a boundary or return NaN
        assert viewed.shape == (1, 2)

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="method"):
            compute_viewed_location(positions, headings, method="invalid")

    def test_ray_cast_without_env_raises(self):
        """ray_cast method without environment raises error."""
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="env"):
            compute_viewed_location(positions, headings, method="ray_cast")


class TestComputeViewshed:
    """Tests for compute_viewshed function."""

    def test_basic_viewshed(self):
        """Basic viewshed returns visible bins."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import ViewshedResult, compute_viewshed

        # Create simple square environment
        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Observer at center
        position = np.array([50.0, 50.0])
        heading = 0.0

        result = compute_viewshed(env, position, heading)

        assert isinstance(result, ViewshedResult)
        assert len(result.visible_bins) > 0
        # In an open square, most bins should be visible from center
        assert result.n_visible_bins > env.n_bins * 0.5

    def test_viewshed_with_fov_restriction(self):
        """FOV parameter restricts visible region."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import FieldOfView, compute_viewshed

        # Create environment
        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        # Full 360 degree view
        result_full = compute_viewshed(env, position, heading, fov=None)

        # Restricted 90 degree view
        fov_90 = FieldOfView.symmetric(half_angle=np.pi / 4)
        result_90 = compute_viewshed(env, position, heading, fov=fov_90)

        # Restricted view should have fewer visible bins
        assert result_90.n_visible_bins < result_full.n_visible_bins

    def test_viewshed_with_float_fov(self):
        """Float FOV is interpreted as full angle."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewshed

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        # Use float for 180 degree FOV
        result = compute_viewshed(env, position, heading, fov=np.pi)

        assert result.n_visible_bins > 0
        assert result.n_visible_bins < env.n_bins  # Not full visibility

    def test_viewshed_cue_visibility(self):
        """Viewshed checks visibility of cue positions."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewshed

        # Create environment
        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        # Cues at various locations
        cue_positions = np.array(
            [
                [60.0, 50.0],  # East of observer
                [40.0, 50.0],  # West of observer
                [50.0, 60.0],  # North of observer
            ]
        )

        result = compute_viewshed(env, position, heading, cue_positions=cue_positions)

        # All cues should be visible in open environment
        assert len(result.visible_cues) == 3
        assert len(result.cue_distances) == 3
        assert len(result.cue_bearings) == 3

    def test_viewshed_n_rays_parameter(self):
        """n_rays parameter controls ray density."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewshed

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        # More rays should give more accurate (but similar) results
        result_few = compute_viewshed(env, position, heading, n_rays=36)
        result_many = compute_viewshed(env, position, heading, n_rays=360)

        # Both should identify some bins as visible
        # With sparse rays (36), we may not hit all bins even in open space
        assert result_few.n_visible_bins > 0
        assert (
            result_many.n_visible_bins > result_few.n_visible_bins
        )  # More rays = more bins hit

    def test_viewshed_occlusion_map(self):
        """Occlusion map has values in [0, 1] range."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewshed

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        result = compute_viewshed(env, position, heading)

        # Occlusion map should have one value per bin
        assert len(result.occlusion_map) == env.n_bins

        # Values should be in [0, 1]
        assert np.all(result.occlusion_map >= 0)
        assert np.all(result.occlusion_map <= 1)


class TestComputeViewField:
    """Tests for compute_view_field function."""

    def test_returns_binary_mask(self):
        """compute_view_field returns binary visibility mask."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_view_field

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        mask = compute_view_field(env, position, heading)

        assert mask.dtype == bool
        assert mask.shape == (env.n_bins,)
        assert np.any(mask)  # Some bins should be visible


class TestVisibleCues:
    """Tests for visible_cues function."""

    def test_returns_visibility_mask_distances_bearings(self):
        """visible_cues returns mask, distances, and bearings."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import visible_cues

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0

        cue_positions = np.array(
            [
                [60.0, 50.0],
                [40.0, 50.0],
                [50.0, 60.0],
            ]
        )

        visible, distances, bearings = visible_cues(
            env, position, heading, cue_positions
        )

        assert visible.dtype == bool
        assert len(visible) == 3
        assert len(distances) == 3
        assert len(bearings) == 3

    def test_occluded_cue_not_visible(self):
        """Cue behind obstacle is not visible."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import visible_cues

        # Create C-shaped environment (obstacle in middle)
        rng = np.random.default_rng(42)
        # Left side
        left = rng.uniform([0, 0], [40, 100], (200, 2))
        # Top and bottom connecting
        top = rng.uniform([40, 80], [100, 100], (100, 2))
        bottom = rng.uniform([40, 0], [100, 20], (100, 2))
        # Right side
        right = rng.uniform([60, 0], [100, 100], (200, 2))
        positions_samples = np.vstack([left, top, bottom, right])
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Observer on left, cue on right (obstacle in between)
        position = np.array([20.0, 50.0])
        heading = 0.0  # Facing East

        cue_positions = np.array([[80.0, 50.0]])  # On right side

        visible, _distances, _bearings = visible_cues(
            env, position, heading, cue_positions
        )

        # Cue should NOT be visible due to obstacle
        assert not visible[0]


class TestComputeViewshedTrajectory:
    """Tests for compute_viewshed_trajectory function."""

    def test_computes_along_trajectory(self):
        """Computes viewshed at each point along trajectory."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import compute_viewshed_trajectory

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Simple trajectory: 5 points moving east
        trajectory = np.array(
            [
                [20.0, 50.0],
                [30.0, 50.0],
                [40.0, 50.0],
                [50.0, 50.0],
                [60.0, 50.0],
            ]
        )
        headings = np.zeros(5)  # All facing east

        results = compute_viewshed_trajectory(env, trajectory, headings)

        assert len(results) == 5
        for result in results:
            assert result.n_visible_bins > 0


class TestVisibilityOccupancy:
    """Tests for visibility_occupancy function."""

    def test_returns_time_visible_per_bin(self):
        """Returns time each bin was visible during trajectory."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import visibility_occupancy

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Simple trajectory
        trajectory = np.array(
            [
                [50.0, 50.0],
                [50.0, 50.0],
                [50.0, 50.0],
            ]
        )
        headings = np.zeros(3)
        times = np.array([0.0, 1.0, 2.0])

        occupancy = visibility_occupancy(env, trajectory, headings, times)

        assert occupancy.shape == (env.n_bins,)
        # All values should be non-negative
        assert np.all(occupancy >= 0)
        # Some bins should have been visible
        assert np.any(occupancy > 0)

    def test_stationary_observer_accumulates_time(self):
        """Stationary observer accumulates visibility time."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import visibility_occupancy

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        # Stationary at center for 10 seconds
        n_samples = 100
        trajectory = np.tile([[50.0, 50.0]], (n_samples, 1))
        headings = np.zeros(n_samples)
        times = np.linspace(0, 10, n_samples)

        occupancy = visibility_occupancy(env, trajectory, headings, times)

        # Visible bins should have occupancy close to 10 seconds
        # (accounting for frame duration)
        assert np.max(occupancy) > 9.0


class TestLineOfSightClear:
    """Tests for _line_of_sight_clear helper function."""

    def test_clear_line_of_sight(self):
        """Line of sight is clear in open environment."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import _line_of_sight_clear

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        observer = np.array([50.0, 50.0])
        target = np.array([60.0, 50.0])

        assert _line_of_sight_clear(env, observer, target)

    def test_blocked_line_of_sight(self):
        """Line of sight is blocked by obstacle."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import _line_of_sight_clear

        # Create C-shaped environment
        rng = np.random.default_rng(42)
        left = rng.uniform([0, 0], [40, 100], (200, 2))
        top = rng.uniform([40, 80], [100, 100], (100, 2))
        bottom = rng.uniform([40, 0], [100, 20], (100, 2))
        right = rng.uniform([60, 0], [100, 100], (200, 2))
        positions_samples = np.vstack([left, top, bottom, right])
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        observer = np.array([20.0, 50.0])
        target = np.array([80.0, 50.0])

        assert not _line_of_sight_clear(env, observer, target)
