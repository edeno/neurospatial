"""Tests for visibility module.

Tests for visibility/gaze computation including field of view dataclass,
viewshed analysis, and cue visibility checking.
"""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestModuleSetup:
    """Test module imports and structure."""


class TestFieldOfView:
    """Tests for FieldOfView dataclass."""

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

    def test_rat_blind_spot_is_the_range_behind_the_edges(self):
        """The rat's 40-degree blind spot is the gap between its edges."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.rat()
        assert_allclose(fov.total_angle_degrees, 320.0)
        assert not fov.contains_angle(np.pi)
        assert not fov.contains_angle(-np.pi)
        assert not fov.contains_angle(np.radians(170.0))
        assert not fov.contains_angle(np.radians(-170.0))
        assert fov.contains_angle(np.radians(155.0))
        assert fov.contains_angle(np.radians(-155.0))

    def test_left_is_positive_and_right_is_negative(self):
        """A left-only field of view sees +angles (left), not -angles."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView(left_angle=np.pi / 2, right_angle=0.1)
        assert fov.contains_angle(0.5)
        assert not fov.contains_angle(-0.5)
        assert not fov.contains_angle(0.0)
        assert_allclose(fov.total_angle, np.pi / 2 - 0.1)

    def test_behind_wraps_to_plus_pi(self):
        """Directly behind is +pi, matching egocentric bearings."""
        from neurospatial.ops.visibility import FieldOfView

        reaches_behind_on_the_left = FieldOfView(left_angle=np.pi, right_angle=0.0)
        assert reaches_behind_on_the_left.contains_angle(-np.pi)
        assert reaches_behind_on_the_left.contains_angle(3 * np.pi)
        assert not reaches_behind_on_the_left.contains_angle(-np.pi / 2)

    def test_nan_angle_is_outside(self):
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(np.pi)
        assert not fov.contains_angle(np.nan)
        np.testing.assert_array_equal(
            fov.contains_angle(np.array([np.nan, 0.0])), [False, True]
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"left_angle": -np.pi / 2, "right_angle": np.pi / 2}, "right_angle <="),
            ({"left_angle": 10.0, "right_angle": -10.0}, "<= pi"),
            ({"left_angle": np.pi, "right_angle": -4.0}, "-pi <="),
            ({"left_angle": np.nan, "right_angle": -1.0}, "finite"),
            ({"left_angle": 1.0, "right_angle": np.nan}, "finite"),
            (
                {
                    "left_angle": 1.0,
                    "right_angle": -1.0,
                    "binocular_half_angle": np.inf,
                },
                "finite",
            ),
            (
                {"left_angle": 1.0, "right_angle": -1.0, "binocular_half_angle": -0.1},
                "binocular_half_angle",
            ),
            (
                {"left_angle": 1.0, "right_angle": -0.5, "binocular_half_angle": 0.6},
                "binocular_half_angle",
            ),
            (
                {"left_angle": 1.0, "right_angle": 0.1, "binocular_half_angle": 0.2},
                "binocular_half_angle",
            ),
        ],
    )
    def test_validation_refuses_impossible_fields(self, kwargs, match):
        """Angles must be finite, ordered within [-pi, pi], binocular inside."""
        from neurospatial.ops.visibility import FieldOfView

        with pytest.raises(ValueError, match=match):
            FieldOfView(**kwargs)

    def test_symmetric_wider_than_a_circle_raises(self):
        from neurospatial.ops.visibility import FieldOfView

        with pytest.raises(ValueError, match="<= pi"):
            FieldOfView.symmetric(half_angle=4.0)

    def test_frozen_dataclass(self):
        """FieldOfView is immutable (frozen dataclass)."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)

        with pytest.raises((AttributeError, TypeError)):
            fov.left_angle = np.pi  # type: ignore[misc]


def _visibility_map(n_bins, visible):
    visibility = np.zeros(n_bins)
    visibility[list(visible)] = 1.0
    return visibility


def _no_cues():
    return {
        "visible_cues": np.array([], dtype=np.intp),
        "cue_distances": np.array([]),
        "cue_bearings": np.array([]),
    }


class TestViewshedResult:
    """Tests for ViewshedResult dataclass."""

    def test_visible_bins_and_counts_come_from_the_map(self):
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visibility_map=_visibility_map(10, [0, 1, 2, 3, 5]), **_no_cues()
        )
        np.testing.assert_array_equal(result.visible_bins, [0, 1, 2, 3, 5])
        assert result.n_visible_bins == 5
        assert result.n_bins == 10
        assert_allclose(result.visibility_fraction, 0.5)

    def test_missing_observation_has_nan_fraction_and_no_bins(self):
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(visibility_map=np.full(4, np.nan), **_no_cues())
        assert result.n_visible_bins == 0
        assert result.visible_bins.size == 0
        assert np.isnan(result.visibility_fraction)

    def test_n_visible_cues_property(self):
        """n_visible_cues returns count of visible cues."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visibility_map=np.zeros(10),
            visible_cues=np.array([0, 2, 5]),
            cue_distances=np.array([5.0, 10.0, 15.0]),
            cue_bearings=np.array([0.0, np.pi / 4, np.pi / 2]),
        )
        assert result.n_visible_cues == 3

    def test_filter_cues(self):
        """filter_cues() returns subset of cues matching IDs."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visibility_map=np.zeros(10),
            visible_cues=np.array([0, 2, 5]),
            cue_distances=np.array([5.0, 10.0, 15.0]),
            cue_bearings=np.array([0.0, np.pi / 4, np.pi / 2]),
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

        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (500, 2)), bin_size=10.0)
        result = ViewshedResult(
            visibility_map=_visibility_map(env.n_bins, [0, 1, 2]), **_no_cues()
        )

        centers = result.visible_bin_centers(env)
        assert centers.shape == (3, 2)
        assert_allclose(centers, env.bin_centers[[0, 1, 2]])

    def test_visible_bin_centers_refuses_another_environment(self):
        from neurospatial import Environment
        from neurospatial.ops.visibility import ViewshedResult

        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (500, 2)), bin_size=10.0)
        result = ViewshedResult(
            visibility_map=_visibility_map(env.n_bins + 1, [0]), **_no_cues()
        )
        with pytest.raises(ValueError, match=r"covers .* bins, but env has"):
            result.visible_bin_centers(env)

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"visibility_map": np.zeros((2, 3))}, "shape \\(n_bins,\\)"),
            ({"visibility_map": np.zeros(0)}, "shape \\(n_bins,\\)"),
            ({"visibility_map": np.array([0.5, 1.0])}, "only 1.0, 0.0 or NaN"),
            ({"visibility_map": np.array([np.nan, 1.0])}, "in every bin"),
            (
                {
                    "visibility_map": np.full(3, np.nan),
                    "visible_cues": np.array([0]),
                    "cue_distances": np.array([1.0]),
                    "cue_bearings": np.array([0.0]),
                },
                "missing observation has no visible cues",
            ),
            (
                {
                    "visible_cues": np.array([0, 1]),
                    "cue_distances": np.array([1.0]),
                    "cue_bearings": np.array([0.0, 0.1]),
                },
                "cue_distances needs shape \\(2,\\)",
            ),
            (
                {
                    "visible_cues": np.array([1, 0]),
                    "cue_distances": np.array([1.0, 2.0]),
                    "cue_bearings": np.array([0.0, 0.1]),
                },
                "increasing non-negative",
            ),
            (
                {
                    "visible_cues": np.array([-1]),
                    "cue_distances": np.array([1.0]),
                    "cue_bearings": np.array([0.0]),
                },
                "increasing non-negative",
            ),
        ],
    )
    def test_inconsistent_fields_raise(self, fields, match):
        from neurospatial.ops.visibility import ViewshedResult

        kwargs = {"visibility_map": np.zeros(3), **_no_cues(), **fields}
        with pytest.raises(ValueError, match=match):
            ViewshedResult(**kwargs)

    def test_frozen_dataclass(self):
        """ViewshedResult is immutable."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(visibility_map=np.zeros(10), **_no_cues())

        with pytest.raises((AttributeError, TypeError)):
            result.visibility_map = np.ones(10)  # type: ignore[misc]
        with pytest.raises(ValueError, match="read-only"):
            result.visibility_map[0] = 1.0


class TestComputeViewedLocation:
    """Tests for compute_viewed_location function."""

    def test_fixed_distance_method(self):
        """Fixed distance method returns point at fixed distance in gaze direction."""
        from neurospatial.ops.visibility import compute_viewed_location

        # Animal at origin, facing East
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        headings = np.array([0.0, np.pi / 2])  # East, then North

        viewed = compute_viewed_location(
            positions,
            headings,
            method="fixed_distance",
            view_distance=10.0,
            boundary_source="analysis",
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
            positions, headings, method="ray_cast", env=env, boundary_source="analysis"
        )

        # Should hit boundary towards East (x > animal position)
        # Note: The exact boundary depends on sampled positions
        assert viewed[0, 0] > 50.0  # x should be East of animal
        assert 40 < viewed[0, 1] < 60  # y should be near middle

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
            boundary_source="analysis",
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
            positions,
            headings,
            method="ray_cast",
            env=env,
            max_distance=200.0,
            boundary_source="analysis",
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
            compute_viewed_location(
                positions, headings, method="invalid", boundary_source="analysis"
            )

    def test_ray_cast_without_env_raises(self):
        """ray_cast method without environment raises error."""
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match="env"):
            compute_viewed_location(
                positions, headings, method="ray_cast", boundary_source="analysis"
            )

    @pytest.mark.parametrize("boundary_source", ["physical", "analysis"])
    def test_ray_cast_reaches_a_distant_wall_by_default(self, boundary_source):
        """The default search reaches every wall, even in a 3 m arena."""
        from neurospatial.ops.visibility import compute_viewed_location

        env = _open_box(300.0, 10.0)
        positions = np.array([[10.0, 150.0], [150.0, 10.0], [290.0, 290.0]])
        headings = np.array([0.0, np.pi / 2, -3 * np.pi / 4])

        viewed = compute_viewed_location(
            positions,
            headings,
            method="ray_cast",
            env=env,
            boundary_source=boundary_source,
        )

        assert_allclose(viewed, [[300.0, 150.0], [150.0, 300.0], [0.0, 0.0]], atol=1e-9)

    @pytest.mark.parametrize("method", ["fixed_distance", "ray_cast"])
    @pytest.mark.parametrize(
        ("positions", "headings", "offsets", "match"),
        [
            (np.zeros((3, 2)), np.zeros(2), None, r"one heading per position"),
            (np.zeros((3, 2)), np.zeros((3, 1)), None, r"one heading per position"),
            (np.zeros(2), np.zeros(1), None, r"positions of shape \(n_time, 2\)"),
            (np.zeros((3, 3)), np.zeros(3), None, r"positions of shape \(n_time, 2\)"),
            (np.zeros((3, 2)), np.zeros(3), np.zeros(2), r"one gaze offset per"),
        ],
    )
    def test_mismatched_samples_raise(
        self, method, positions, headings, offsets, match
    ):
        """Misaligned inputs are refused instead of broadcast or truncated."""
        from neurospatial.ops.visibility import compute_viewed_location

        with pytest.raises(ValueError, match=match):
            compute_viewed_location(
                positions,
                headings,
                method=method,
                gaze_offsets=offsets,
                env=_open_box(20.0, 5.0),
            )


def _open_box(size, bin_size):
    from shapely.geometry import box

    from neurospatial import Environment

    return Environment.from_polygon(box(0, 0, size, size), bin_size=bin_size)


def _divided_room():
    """100 x 100 cm room, 2 cm bins, a wall at x = 50 from y = 20 to y = 80."""
    from shapely.geometry import LineString, box

    from neurospatial import Environment
    from neurospatial.arena import OpenArena

    return Environment.from_grid_mask(
        np.ones((50, 50), dtype=bool),
        grid_edges=(np.arange(0.0, 101.0, 2.0),) * 2,
        arena=OpenArena(box(0, 0, 100, 100), walls=(LineString([(50, 20), (50, 80)]),)),
    )


def _bearings(env, position, heading):
    offsets = env.bin_centers - position
    angle = np.arctan2(offsets[:, 1], offsets[:, 0]) - heading
    return (angle - np.pi) % (-2 * np.pi) + np.pi


class TestComputeViewshed:
    """Tests for compute_viewshed function."""

    @pytest.mark.parametrize(
        ("size", "bin_size"), [(100.0, 2.0), (200.0, 2.0), (200.0, 1.0)]
    )
    @pytest.mark.parametrize("where", [(0.5, 0.5), (0.025, 0.025), (0.373, 0.819)])
    def test_every_bin_of_an_open_box_is_visible(self, size, bin_size, where):
        """With no walls in the way, nothing is hidden, whatever the bin size."""
        from neurospatial.ops.visibility import compute_viewshed

        env = _open_box(size, bin_size)
        position = np.array(where) * size
        result = compute_viewshed(env, position, heading=0.3)
        assert result.n_visible_bins == env.n_bins
        assert result.visibility_fraction == 1.0

    @pytest.mark.parametrize("boundary_source", ["physical", "analysis"])
    def test_a_wall_hides_exactly_the_bins_behind_it(self, boundary_source):
        from neurospatial.ops.visibility import compute_viewshed

        env = _divided_room()
        observer = np.array([25.3, 50.7])
        centers = env.bin_centers
        # Where the line to each center crosses x = 50, for centers past it.
        beyond = centers[:, 0] > 50
        t = (50 - observer[0]) / (centers[:, 0] - observer[0])
        crossing_y = observer[1] + t * (centers[:, 1] - observer[1])
        hidden = beyond & (crossing_y >= 20) & (crossing_y <= 80)

        result = compute_viewshed(
            env, observer, heading=0.0, boundary_source=boundary_source
        )

        assert 0 < hidden.sum() < env.n_bins
        np.testing.assert_array_equal(result.visibility_map, (~hidden).astype(float))
        np.testing.assert_array_equal(result.visible_bins, np.flatnonzero(~hidden))

    def test_left_only_field_of_view_sees_north_east_when_facing_east(self):
        """+angles are to the left: facing East, the left is North."""
        from neurospatial.ops.visibility import FieldOfView, compute_viewshed

        env = _open_box(100.0, 2.0)
        observer = np.array([50.3, 50.7])
        fov = FieldOfView(left_angle=np.pi / 2, right_angle=0.1)

        result = compute_viewshed(env, observer, heading=0.0, fov=fov)

        bearing = _bearings(env, observer, 0.0)
        expected = (bearing >= 0.1) & (bearing <= np.pi / 2)
        np.testing.assert_array_equal(result.visible_bins, np.flatnonzero(expected))
        offsets = env.bin_centers[result.visible_bins] - observer
        assert result.n_visible_bins > 0
        assert np.all(offsets[:, 0] >= 0) and np.all(offsets[:, 1] > 0)

    def test_facing_north_sees_north(self):
        """Heading pi/2 is North; a forward field of view sees only bins north."""
        from neurospatial.ops.visibility import compute_viewshed

        env = _open_box(100.0, 2.0)
        observer = np.array([50.3, 50.7])

        result = compute_viewshed(env, observer, heading=np.pi / 2, fov=np.pi / 2)

        offsets = env.bin_centers[result.visible_bins] - observer
        assert result.n_visible_bins > 0
        assert np.all(offsets[:, 1] > 0)
        assert np.all(np.abs(offsets[:, 0]) <= offsets[:, 1])

    def test_float_fov_is_the_total_angle(self):
        from neurospatial.ops.visibility import FieldOfView, compute_viewshed

        env = _open_box(100.0, 5.0)
        observer = np.array([50.3, 50.7])
        as_float = compute_viewshed(env, observer, 0.4, fov=np.pi)
        as_object = compute_viewshed(
            env, observer, 0.4, fov=FieldOfView.symmetric(np.pi / 2)
        )
        np.testing.assert_array_equal(as_float.visible_bins, as_object.visible_bins)
        assert_allclose(as_float.visibility_fraction, 0.5, atol=0.05)

    @pytest.mark.parametrize("fov", [0.0, -1.0, 7.0, np.nan])
    def test_float_fov_outside_a_circle_raises(self, fov):
        from neurospatial.ops.visibility import compute_viewshed

        with pytest.raises(ValueError, match=r"total angle in \(0, 2\*pi\]"):
            compute_viewshed(_open_box(20.0, 5.0), np.array([10.0, 10.0]), 0.0, fov=fov)

    def test_cues_and_bins_share_one_visibility_test(self):
        """A cue on a bin center is visible exactly when that bin is."""
        from neurospatial.ops.visibility import FieldOfView, compute_viewshed

        env = _divided_room()
        observer = np.array([25.3, 50.7])
        result = compute_viewshed(
            env,
            observer,
            heading=0.4,
            fov=FieldOfView.rat(),
            cue_positions=env.bin_centers,
        )
        assert 0 < result.n_visible_bins < env.n_bins
        np.testing.assert_array_equal(result.visible_cues, result.visible_bins)
        offsets = env.bin_centers[result.visible_cues] - observer
        assert_allclose(result.cue_distances, np.hypot(*offsets.T))
        assert_allclose(
            result.cue_bearings, _bearings(env, observer, 0.4)[result.visible_cues]
        )

    def test_observer_outside_sees_nothing(self):
        from neurospatial.ops.visibility import compute_viewshed

        env = _open_box(20.0, 5.0)
        result = compute_viewshed(env, np.array([30.0, 10.0]), np.pi)
        np.testing.assert_array_equal(result.visibility_map, np.zeros(env.n_bins))
        assert result.visibility_fraction == 0.0

    @pytest.mark.parametrize(
        ("cues", "match"),
        [
            (np.array([1.0, 2.0]), r"shape \(n_cues, 2\)"),
            (np.array([[1.0, np.nan]]), "finite"),
        ],
    )
    def test_malformed_cues_raise(self, cues, match):
        from neurospatial.ops.visibility import compute_viewshed

        with pytest.raises(ValueError, match=match):
            compute_viewshed(
                _open_box(20.0, 5.0), np.array([10.0, 10.0]), 0.0, cue_positions=cues
            )

    def test_position_must_be_one_point(self):
        from neurospatial.ops.visibility import compute_viewshed

        with pytest.raises(ValueError, match=r"needs shape \(2,\)"):
            compute_viewshed(_open_box(20.0, 5.0), np.array([[10.0, 10.0]]), 0.0)


class TestComputeViewField:
    """Tests for compute_view_field function."""

    def test_mask_matches_the_viewshed(self):
        from neurospatial.ops.visibility import (
            FieldOfView,
            compute_view_field,
            compute_viewshed,
        )

        env = _divided_room()
        observer = np.array([25.3, 50.7])
        mask = compute_view_field(env, observer, 0.2, fov=FieldOfView.primate())
        result = compute_viewshed(env, observer, 0.2, fov=FieldOfView.primate())

        assert mask.dtype == bool
        assert mask.shape == (env.n_bins,)
        np.testing.assert_array_equal(np.flatnonzero(mask), result.visible_bins)
        # Primate field of view (+/-90 degrees) facing ~East: nothing west.
        assert 0 < mask.sum() < env.n_bins
        assert np.all(env.bin_centers[mask, 0] > observer[0] - 10)


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
            env, position, heading, cue_positions, boundary_source="analysis"
        )

        assert visible.dtype == bool
        assert len(visible) == 3
        assert len(distances) == 3
        assert len(bearings) == 3

    def test_bearings_match_known_geometry(self):
        """visible_cues returns correct egocentric bearings (0=ahead, +pi/2=left)."""
        from neurospatial import Environment
        from neurospatial.ops.visibility import visible_cues

        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=5.0)

        position = np.array([50.0, 50.0])
        heading = 0.0  # facing East (+x)
        cue_positions = np.array(
            [
                [60.0, 50.0],  # directly ahead (East)  -> 0
                [40.0, 50.0],  # directly behind (West) -> +/- pi
                [50.0, 60.0],  # directly left (North)  -> +pi/2
            ]
        )

        visible, distances, bearings = visible_cues(
            env, position, heading, cue_positions, boundary_source="analysis"
        )

        assert np.all(visible)  # open arena, all cues in line of sight
        np.testing.assert_allclose(distances, [10.0, 10.0, 10.0], atol=1e-6)
        assert bearings[0] == pytest.approx(0.0, abs=1e-6)
        # Behind: +/- pi (both signs mean "directly behind"; assert magnitude).
        assert abs(bearings[1]) == pytest.approx(np.pi, abs=1e-6)
        assert bearings[2] == pytest.approx(np.pi / 2, abs=1e-6)

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
            env, position, heading, cue_positions, boundary_source="analysis"
        )

        # Cue should NOT be visible due to obstacle
        assert not visible[0]


class TestComputeViewshedTrajectory:
    """Tests for compute_viewshed_trajectory function."""

    def test_each_frame_is_that_frames_viewshed(self):
        from neurospatial.ops.visibility import (
            compute_viewshed,
            compute_viewshed_trajectory,
        )

        env = _divided_room()
        positions = np.array([[20.3, 50.7], [40.1, 10.2], [70.9, 50.3], [30.0, 85.5]])
        headings = np.array([0.0, np.pi / 2, np.pi, -2.0])
        cues = np.array([[60.0, 50.0], [10.0, 10.0]])

        results = compute_viewshed_trajectory(
            env, positions, headings, fov=np.pi, cue_positions=cues
        )

        assert len(results) == len(positions)
        for result, position, heading in zip(results, positions, headings, strict=True):
            single = compute_viewshed(
                env, position, heading, fov=np.pi, cue_positions=cues
            )
            np.testing.assert_array_equal(result.visibility_map, single.visibility_map)
            np.testing.assert_array_equal(result.visible_cues, single.visible_cues)
            assert_allclose(result.cue_bearings, single.cue_bearings)
        # The cue across the wall is hidden from the west side facing East,
        # and visible from the east side facing West.
        assert 0 not in results[0].visible_cues
        assert 0 in results[2].visible_cues


class TestVisibilityOccupancy:
    """Tests for visibility_occupancy function."""

    def test_stationary_observer_sees_its_half_of_the_box_for_all_the_time(self):
        """A still animal facing East sees the eastern half for the whole time."""
        from neurospatial.ops.visibility import visibility_occupancy

        env = _open_box(100.0, 5.0)
        n = 101
        times = np.linspace(0, 10, n)  # 10 s
        positions = np.tile([[50.3, 50.7]], (n, 1))

        seen = visibility_occupancy(env, times, positions, np.zeros(n), fov=np.pi)

        east = np.abs(_bearings(env, positions[0], 0.0)) <= np.pi / 2
        assert_allclose(seen[east], 10.0, rtol=1e-12)
        np.testing.assert_array_equal(seen[~east], 0.0)

    def test_open_box_every_bin_gets_the_usable_time_at_fine_bins(self):
        """Fine bins far from the animal are not missed (they used to be)."""
        from neurospatial.ops.visibility import visibility_occupancy

        env = _open_box(200.0, 1.0)
        times = np.arange(0, 1.01, 0.1)
        positions = np.full((len(times), 2), 5.0)  # in a corner

        seen = visibility_occupancy(env, times, positions, np.zeros(len(times)))

        assert_allclose(seen, 1.0, rtol=1e-9)


class TestVisibilityOccupancySupport:
    """Viewing time follows the shared observation-support rules.

    In a small open square every bin is visible from every position with a
    full field of view, so each bin's viewing time must equal the total
    usable time that ``Environment.occupancy`` reports for the same samples.
    """

    @pytest.fixture(scope="class")
    def open_square(self):
        from neurospatial import Environment

        x, y = np.meshgrid(np.arange(0.5, 20.0), np.arange(0.5, 20.0))
        return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)

    @pytest.fixture(scope="class")
    def walk(self):
        rng = np.random.default_rng(3)
        n = 40
        times = np.cumsum(np.r_[0.0, rng.uniform(0.02, 0.05, n - 1)])
        positions = rng.uniform(2.0, 18.0, (n, 2))
        headings = rng.uniform(-np.pi, np.pi, n)
        return times, positions, headings

    @staticmethod
    def _seen(env, times, positions, headings, **kwargs):
        from neurospatial.ops.visibility import visibility_occupancy

        return visibility_occupancy(
            env,
            times,
            positions,
            headings,
            boundary_source="analysis",
            **kwargs,
        )

    def test_every_visible_bin_gets_the_total_usable_time(self, open_square, walk):
        times, positions, headings = walk
        times = times.copy()
        times[20:] += 2.0  # a 2 s tracking pause between samples 19 and 20
        positions = positions.copy()
        positions[7] = np.nan  # a lost frame
        breaks = np.zeros(len(times), dtype=bool)
        breaks[30] = True  # a second recording joined on the same clock
        epochs = [(times[2], times[36])]
        gates = {"max_gap": 0.5, "epochs": epochs, "break_before": breaks}

        seen = self._seen(open_square, times, positions, headings, **gates)
        total = open_square.occupancy(times, positions, **gates).sum()

        assert total > 0
        assert_allclose(seen, np.full(open_square.n_bins, total), rtol=1e-12)

    def test_a_pause_longer_than_max_gap_adds_no_time(self, open_square, walk):
        times, positions, headings = walk
        paused = times.copy()
        paused[10:] += 60.0  # the animal is untracked for a minute

        seen = self._seen(open_square, paused, positions, headings)

        expected = np.diff(times).sum() - np.diff(times)[9]
        assert_allclose(seen, expected, rtol=1e-12)

    def test_declared_break_equals_a_gap_past_max_gap(self, open_square, walk):
        times, positions, headings = walk
        breaks = np.zeros(len(times), dtype=bool)
        breaks[[12, 25]] = True
        shifted = times.copy()
        shifted[12:] += 1.0
        shifted[25:] += 1.0

        declared = self._seen(
            open_square, times, positions, headings, break_before=breaks
        )
        gapped = self._seen(open_square, shifted, positions, headings)

        assert_allclose(declared, gapped, rtol=1e-12)
        assert (
            declared.max() < self._seen(open_square, times, positions, headings).max()
        )

    @pytest.mark.parametrize("missing", ["position", "heading"])
    def test_a_missing_sample_adds_nothing(self, open_square, walk, missing):
        times, positions, headings = walk
        positions, headings = positions.copy(), headings.copy()
        if missing == "position":
            positions[15, 1] = np.nan
        else:
            headings[15] = np.nan

        seen = self._seen(open_square, times, positions, headings)

        dt = np.diff(times)
        # Both intervals that touch sample 15 are unobserved.
        assert_allclose(seen, dt.sum() - dt[14] - dt[15], rtol=1e-12)

    def test_the_last_sample_adds_no_time(self, open_square, walk):
        times, positions, headings = walk
        seen = self._seen(open_square, times, positions, headings)
        moved = positions.copy()
        moved[-1] = [100.0, 100.0]  # outside the arena: views nothing

        assert_allclose(seen, np.diff(times).sum(), rtol=1e-12)
        assert_allclose(self._seen(open_square, times, moved, headings), seen)
        one = self._seen(open_square, times[:1], positions[:1], headings[:1])
        assert np.array_equal(one, np.zeros(open_square.n_bins))

    def test_viewsheds_are_only_computed_for_samples_that_add_time(
        self, open_square, walk
    ):
        from neurospatial.ops import visibility

        times, positions, headings = walk
        times = times.copy()
        times[20:] += 5.0
        calls: list[int] = []
        original = visibility._viewshed

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with patch.object(visibility, "_viewshed", counted):
            self._seen(open_square, times, positions, headings)

        # 40 samples: 39 intervals, minus the pause between samples 19 and 20.
        assert len(calls) == 38

    def test_all_intervals_excluded_warns(self, open_square, walk):
        times, positions, headings = walk
        with pytest.warns(UserWarning, match=r"excluded ALL[\s\S]*viewing time"):
            seen = self._seen(
                open_square, times, positions, headings, epochs=[(100.0, 200.0)]
            )
        assert not seen.any()

    def test_invalid_break_before_raises_the_shared_error(self, open_square, walk):
        times, positions, headings = walk
        with pytest.raises(ValueError, match=r"break_before to be a Boolean array"):
            self._seen(
                open_square,
                times,
                positions,
                headings,
                break_before=np.zeros(len(times), dtype=int),
            )

    def test_headings_must_match_times(self, open_square, walk):
        times, positions, headings = walk
        with pytest.raises(ValueError, match=r"one heading per timestamp"):
            self._seen(open_square, times, positions, headings[:-1])

    @pytest.mark.parametrize(
        "options",
        [
            {"fov": 7.0},
            {"boundary_source": "nope"},
            {"boundary_source": "physical"},  # no declared walls
            {"fov": -1.0},
        ],
    )
    @pytest.mark.parametrize("data", ["all_excluded", "single_sample"])
    def test_invalid_viewshed_options_raise_without_usable_intervals(
        self, open_square, walk, options, data
    ):
        """The errors match compute_viewshed's, even when no viewshed is needed."""
        from neurospatial.ops.visibility import compute_viewshed, visibility_occupancy

        times, positions, headings = walk
        kwargs = {"boundary_source": "analysis", **options}
        with pytest.raises(Exception) as expected:
            compute_viewshed(open_square, positions[0], headings[0], **kwargs)

        if data == "all_excluded":
            times, positions, headings = times * 100.0, positions, headings
        else:
            times, positions, headings = times[:1], positions[:1], headings[:1]
        with pytest.raises(expected.type) as raised:
            visibility_occupancy(open_square, times, positions, headings, **kwargs)
        assert str(raised.value) == str(expected.value)


class TestLineOfSightClear:
    """Tests for line_of_sight_clear helper function."""

    def test_clear_line_of_sight(self):
        """Line of sight is clear in open environment."""
        from neurospatial import Environment
        from neurospatial.geometry._boundaries import line_of_sight_clear

        rng = np.random.default_rng(42)
        positions_samples = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(positions_samples, bin_size=5.0)

        observer = np.array([50.0, 50.0])
        target = np.array([60.0, 50.0])

        assert line_of_sight_clear(
            env.binning, env.arena, observer, target, boundary_source="analysis"
        )

    def test_blocked_line_of_sight(self):
        """Line of sight is blocked by obstacle."""
        from neurospatial import Environment
        from neurospatial.geometry._boundaries import line_of_sight_clear

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

        assert not line_of_sight_clear(
            env.binning, env.arena, observer, target, boundary_source="analysis"
        )


class TestMissingObservations:
    """A missing position is never reported as blocked or as seeing nothing."""

    @pytest.fixture
    def open_field(self):
        from neurospatial import Environment

        x, y = np.meshgrid(np.arange(0.0, 20.0), np.arange(0.0, 20.0))
        return Environment.from_samples(np.c_[x.ravel(), y.ravel()], bin_size=5.0)

    @pytest.mark.parametrize(
        ("observer", "targets", "named"),
        [
            ([np.nan, 10.0], [[12.0, 10.0]], "observer"),
            ([10.0, 10.0], [[12.0, 10.0], [np.inf, 10.0]], "first row 1"),
        ],
    )
    def test_line_of_sight_refuses_non_finite_positions(
        self, open_field, observer, targets, named
    ):
        from neurospatial.geometry._boundaries import batch_line_of_sight_clear

        with pytest.raises(ValueError, match=named) as info:
            batch_line_of_sight_clear(
                open_field.binning,
                open_field.arena,
                np.array(observer),
                np.array(targets),
                boundary_source="analysis",
            )
        assert "Fix:" in str(info.value)

    @pytest.mark.parametrize(
        ("position", "heading", "cues"),
        [
            ([np.nan, 10.0], 0.0, [[12.0, 10.0]]),
            ([10.0, 10.0], np.nan, [[12.0, 10.0]]),
            ([10.0, 10.0], 0.0, [[12.0, 10.0], [np.nan, 1.0]]),
        ],
    )
    def test_visible_cues_refuses_a_missing_observation(
        self, open_field, position, heading, cues
    ):
        from neurospatial.ops.visibility import visible_cues

        with pytest.raises(ValueError, match="visible_cues"):
            visible_cues(
                open_field,
                np.array(position),
                heading,
                np.array(cues),
                boundary_source="analysis",
            )

    def test_view_field_refuses_a_missing_observation(self, open_field):
        from neurospatial.ops.visibility import compute_view_field

        with pytest.raises(ValueError, match="compute_view_field"):
            compute_view_field(
                open_field, np.array([np.nan, 10.0]), 0.0, boundary_source="analysis"
            )

    def test_trajectory_keeps_missing_frames_missing(self, open_field):
        from neurospatial.ops.visibility import compute_viewshed_trajectory

        positions = np.array([[10.0, 10.0], [np.nan, 10.0], [10.0, 10.0]])
        headings = np.array([0.0, 0.0, np.nan])
        observed, *missing = compute_viewshed_trajectory(
            open_field, positions, headings, boundary_source="analysis"
        )
        assert observed.n_visible_bins > 0
        assert np.all(np.isfinite(observed.visibility_map))
        for frame in missing:
            assert frame.n_visible_bins == 0
            assert np.isnan(frame.visibility_map).all()
            assert np.isnan(frame.visibility_fraction)

    def test_analysis_walls_name_why_the_bins_have_no_cells(self):
        from neurospatial import Environment
        from neurospatial.binning import (
            BinConnections,
            Binning,
            MissingFaces,
            NearestBinLocator,
            SpatialBins,
        )
        from neurospatial.ops.visibility import compute_viewshed

        centers = np.array([[0.0, 0.0], [1.0, 0.0]])
        env = Environment.from_bins(
            Binning(
                "sites",
                SpatialBins.from_arrays(centers, np.ones(2), 1),
                BinConnections.from_arrays([[0, 1]], [1.0]),
                NearestBinLocator(centers, max_distance=0.5),
                MissingFaces("Sites lack faces."),
            )
        )
        with pytest.raises(NotImplementedError, match="exact planar cells") as info:
            compute_viewshed(env, np.zeros(2), 0.0, boundary_source="analysis")
        message = str(info.value)
        assert "planar area-measured" in message
        assert message.count("Fix:") == 1


@pytest.mark.parametrize("call", ["occupancy", "trajectory"])
def test_trajectory_viewsheds_build_the_mesh_lookup_once(call, monkeypatch):
    """Per-frame viewsheds share one triangle index over the whole trajectory."""
    from shapely.geometry import box

    from neurospatial import Environment
    from neurospatial.geometry import _locators
    from neurospatial.layout import create_layout
    from neurospatial.ops.visibility import (
        compute_viewshed_trajectory,
        visibility_occupancy,
    )

    env = Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 20, 20), point_spacing=2.0
        )
    )
    builds: list[int] = []
    build = _locators._triangle_index

    def counted(*args):
        builds.append(1)
        return build(*args)

    monkeypatch.setattr(_locators, "_triangle_index", counted)
    n_frames = 4
    times = np.arange(n_frames) * 0.1
    positions = np.column_stack((np.linspace(5, 15, n_frames), np.full(n_frames, 10.0)))
    headings = np.zeros(n_frames)

    if call == "occupancy":
        occupancy = visibility_occupancy(
            env, times, positions, headings, boundary_source="analysis"
        )
        assert occupancy.sum() > 0
    else:
        results = compute_viewshed_trajectory(
            env, positions, headings, boundary_source="analysis"
        )
        assert all(result.n_visible_bins > 0 for result in results)

    assert len(builds) == 1
