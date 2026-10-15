"""Ray hits and line of sight use retained physical walls, not analysis cuts."""

import numpy as np
import pytest
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.ops.visibility import (
    compute_viewed_location,
    compute_viewshed,
    visible_cues,
)


def room(*, wall=True):
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(
            box(0, 0, 10, 10), walls=(LineString([(5, 0), (5, 10)]),) if wall else ()
        ),
    )


def test_physical_wall_hit_is_exact_and_independent_of_analysis_crop():
    env = room()
    crop = env.subset(bins=env.bin_centers[:, 0] < 4)
    for e in (env, crop, Environment.from_dict(crop.to_dict())):
        np.testing.assert_allclose(
            compute_viewed_location(
                [[2.5, 4.5]], [0.0], env=e, method="ray_cast", max_distance=30
            ),
            [[5, 4.5]],
        )
    np.testing.assert_allclose(
        compute_viewed_location(
            [[2.5, 4.5]],
            [0.0],
            env=crop,
            method="ray_cast",
            max_distance=30,
            boundary_source="analysis",
        ),
        [[4, 4.5]],
    )


def test_unknown_footprint_requires_explicit_analysis_mode():
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2
    )
    with pytest.raises(NotImplementedError, match="Physical walls"):
        compute_viewed_location([[2.5, 4.5]], [0.0], env=env, method="ray_cast")
    np.testing.assert_allclose(
        compute_viewed_location(
            [[2.5, 4.5]], [0.0], env=env, method="ray_cast", boundary_source="analysis"
        ),
        [[10, 4.5]],
    )


def test_no_hit_outside_nonfinite_and_origin_on_wall():
    env = room()
    assert np.isnan(
        compute_viewed_location(
            [[2.5, 4.5]], [0.0], env=env, method="ray_cast", max_distance=1
        )
    ).all()
    assert np.isnan(
        compute_viewed_location([[-1, 4.5]], [0.0], env=env, method="ray_cast")
    ).all()
    assert np.isnan(
        compute_viewed_location(
            [[np.nan, 4.5], [2.5, 4.5]], [0.0, np.inf], env=env, method="ray_cast"
        )
    ).all()
    np.testing.assert_allclose(
        compute_viewed_location([[5, 4.5]], [0.0], env=env, method="ray_cast"),
        [[5, 4.5]],
    )


def test_cues_and_viewed_bins_are_occluded_by_internal_wall():
    env = room()
    observer = np.array([2.5, 4.5])
    cues = np.array([[3.5, 4.5], [5.0, 4.5], [7.5, 4.5]])
    visible, _, _ = visible_cues(env, observer, 0.0, cues)
    np.testing.assert_array_equal(visible, [True, True, False])
    result = compute_viewshed(env, observer, 0.0, n_rays=72)
    assert np.all(env.bin_centers[result.visible_bins, 0] < 5)


def test_missing_analysis_bins_do_not_occlude_physical_cues():
    full = room(wall=False)
    crop = full.subset(bins=full.bin_centers[:, 0] < 4)
    cues = np.array([[7.5, 4.5]])
    visible, _, _ = visible_cues(crop, np.array([2.5, 4.5]), 0.0, cues)
    assert visible[0]
    approximate, _, _ = visible_cues(
        crop, np.array([2.5, 4.5]), 0.0, cues, boundary_source="analysis"
    )
    assert not approximate[0]


def test_view_encoding_uses_observer_side_of_wall_and_records_source():
    from neurospatial.encoding.view import compute_view_rate

    env = room()
    positions = np.tile([2.5, 4.5], (3, 1))
    result = compute_view_rate(
        env,
        np.array([0.25, 1.25]),
        np.array([0.0, 1.0, 2.0]),
        positions,
        np.zeros(3),
        gaze_model="ray_cast",
        method="binned",
        bandwidth=0.0,
        max_gap=None,
    )
    expected = 4 * 10 + 4
    assert result.occupancy.sum() == 2.0
    assert result.occupancy[expected] == 2.0
    assert result.firing_rate[expected] == 1.0
    assert result.analysis_settings["boundary_source"] == "physical"


@pytest.mark.parametrize("offset", [0.0, 100.0, 1e9])
def test_rotated_wall_encoding_retains_observer_side(offset):
    from neurospatial.encoding.view import compute_view_rate
    from neurospatial.ops.transforms import Affine2D

    a = 0.3
    matrix = np.array(
        [
            [1.3 * np.cos(a), -1.3 * np.sin(a), offset],
            [1.3 * np.sin(a), 1.3 * np.cos(a), -offset],
            [0, 0, 1],
        ]
    )
    env = room().apply_transform(Affine2D(matrix))
    observed = np.array([2.5, 4.5]) @ matrix[:2, :2].T + matrix[:2, 2]
    result = compute_view_rate(
        env,
        np.array([0.25, 1.25]),
        np.array([0.0, 1.0, 2.0]),
        np.tile(observed, (3, 1)),
        np.full(3, a),
        gaze_model="ray_cast",
        method="binned",
        bandwidth=0.0,
        max_gap=None,
    )
    assert result.occupancy[44] == 2.0
    assert result.firing_rate[44] == 1.0


def test_first_hole_and_doorway_hits_and_analysis_mode_propagation():
    mask = np.ones((10, 10), dtype=bool)
    mask[4:6, 4:6] = False
    domain = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    env = Environment.from_grid_mask(
        mask, grid_edges=(np.arange(11.0),) * 2, arena=OpenArena(domain)
    )
    np.testing.assert_allclose(
        compute_viewed_location([[2.5, 4.5]], [0.0], env=env, method="ray_cast"),
        [[4, 4.5]],
    )
    doorway = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(
            box(0, 0, 10, 10),
            walls=(LineString([(5, 0), (5, 4)]), LineString([(5, 5), (5, 10)])),
        ),
    )
    np.testing.assert_allclose(
        compute_viewed_location([[2.5, 4.5]], [0.0], env=doorway, method="ray_cast"),
        [[10, 4.5]],
    )
    from neurospatial.encoding.view import compute_view_rate

    inferred = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(11.0),) * 2
    )
    result = compute_view_rate(
        inferred,
        np.array([0.25]),
        np.array([0.0, 1.0]),
        np.tile([2.5, 4.5], (2, 1)),
        np.zeros(2),
        gaze_model="ray_cast",
        boundary_source="analysis",
        method="binned",
        bandwidth=0.0,
        max_gap=None,
    )
    assert result.occupancy[94] == 1.0
    assert result.analysis_settings["boundary_source"] == "analysis"


@pytest.mark.parametrize("offset", [0.0, 100.0, 1e9])
def test_wall_cue_visibility_survives_frame_rounding(offset):
    from neurospatial.ops.transforms import Affine2D

    a = 0.3
    matrix = np.array(
        [
            [1.3 * np.cos(a), -1.3 * np.sin(a), offset],
            [1.3 * np.sin(a), 1.3 * np.cos(a), -offset],
            [0, 0, 1],
        ]
    )
    env = room().apply_transform(Affine2D(matrix))
    points = (
        np.array([[2.5, 4.5], [5, 4.5], [5.00001, 4.5]]) @ matrix[:2, :2].T
        + matrix[:2, 2]
    )
    visible, _, _ = visible_cues(env, points[0], a, points[1:])
    np.testing.assert_array_equal(visible, [True, False])
    np.testing.assert_allclose(
        compute_viewed_location(points[1:2], np.array([a]), env=env, method="ray_cast"),
        points[1:2],
    )


def test_rotated_angular_sweep_cannot_encode_or_see_through_wall():
    from neurospatial.encoding._view_binning import _precompute_view_bins
    from neurospatial.ops.transforms import Affine2D

    a = 0.3
    matrix = np.array(
        [
            [1.3 * np.cos(a), -1.3 * np.sin(a), 100],
            [1.3 * np.sin(a), 1.3 * np.cos(a), -100],
            [0, 0, 1.0],
        ]
    )
    env = room().apply_transform(Affine2D(matrix))
    point = np.array([2.5, 4.5]) @ matrix[:2, :2].T + matrix[:2, 2]
    headings = np.linspace(-np.pi, np.pi, 72, endpoint=False) + a
    bins = _precompute_view_bins(
        env, np.tile(point, (72, 1)), headings, gaze_model="ray_cast"
    )
    assert np.all((bins < 50) & (bins >= 0))
    assert np.all(compute_viewshed(env, point, a, n_rays=72).visible_bins < 50)


def test_wall_bin_lookup_keeps_transformed_subset_and_clipped_owner_maps():
    from neurospatial.encoding._view_binning import _precompute_view_bins
    from neurospatial.ops.transforms import Affine2D

    a = 0.3
    matrix = np.array(
        [
            [1.3 * np.cos(a), -1.3 * np.sin(a), 100],
            [1.3 * np.sin(a), 1.3 * np.cos(a), -100],
            [0, 0, 1.0],
        ]
    )
    parent = room()
    cropped = parent.subset(bins=parent.bin_centers[:, 0] < 4).apply_transform(
        Affine2D(matrix)
    )
    clipped = parent.clip(box(4.2, 4.2, 5, 4.8)).apply_transform(Affine2D(matrix))
    point = np.array([2.5, 4.5]) @ matrix[:2, :2].T + matrix[:2, 2]
    for env, expected in [(cropped, -1), (clipped, 0)]:
        for restored in (env, Environment.from_dict(env.to_dict())):
            np.testing.assert_array_equal(
                _precompute_view_bins(
                    restored, point.reshape(1, 2), np.array([a]), gaze_model="ray_cast"
                ),
                [expected],
            )


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("transform", [False, True])
def test_slanted_wall_hits_keep_all_supported_view_bins(transform, restore):
    from shapely.geometry import Polygon

    from neurospatial.encoding._view_binning import _precompute_view_bins

    polygon = Polygon([(0.2, 0.1), (9.8, 1.1), (7.3, 9.4), (1.5, 7.8)])
    grid = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
    )
    clipped = grid.clip(polygon)
    env = Environment.from_bins(clipped._binning, arena=OpenArena(polygon))
    headings = np.linspace(-np.pi, np.pi, 500, endpoint=False)
    origins = np.tile([2.5, 2.5], (len(headings), 1))
    if transform:
        angle = 0.31
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        matrix = np.eye(3)
        matrix[:2, :2] = rotation
        matrix[:2, 2] = [100, 200]
        from neurospatial.ops.transforms import Affine2D

        env = env.apply_transform(Affine2D(matrix))
        origins = origins @ rotation.T + [100, 200]
        headings += angle
    if restore:
        env = Environment.from_dict(env.to_dict())
    bins = _precompute_view_bins(env, origins, headings, gaze_model="ray_cast")
    assert np.all(bins >= 0)
    # Removing support at the hit must not assign it to a nearby retained cell.
    one = _precompute_view_bins(env, origins[:1], headings[:1], gaze_model="ray_cast")[
        0
    ]
    selected = env.subset(bins=np.arange(env.n_bins) != one)
    np.testing.assert_array_equal(
        _precompute_view_bins(
            selected, origins[:1], headings[:1], gaze_model="ray_cast"
        ),
        [-1],
    )


@pytest.mark.parametrize("gap", [1e-7, 1e-12])
def test_wall_ownership_does_not_cross_a_missing_analysis_strip(gap):
    from shapely.geometry import Polygon

    from neurospatial.encoding._view_binning import _precompute_view_bins

    polygon = Polygon([(0.2, 0.1), (9.8, 1.1), (7.3, 9.4), (1.5, 7.8)])
    grid = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
    )
    clipped = grid.clip(polygon.buffer(-gap))
    env = Environment.from_bins(clipped._binning, arena=OpenArena(polygon))
    headings = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    origins = np.tile([2.5, 2.5], (len(headings), 1))
    np.testing.assert_array_equal(
        _precompute_view_bins(env, origins, headings, gaze_model="ray_cast"), -1
    )


@pytest.mark.parametrize("offset", [1e-14, 1e-13, 1e-12, 1e-7])
def test_wall_contact_cannot_change_a_resolved_dropped_parent(offset):
    from neurospatial.encoding._view_binning import _precompute_view_bins

    full = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool),
        grid_edges=(np.arange(11.0),) * 2,
        arena=OpenArena(box(0, 0, 10, 10)),
    )
    selected = full.subset(bins=np.arange(full.n_bins) != 94)
    observer = np.array([[2.5, 2.5]])
    target = np.array([10.0, 4.0 + offset])
    delta = target - observer[0]
    heading = np.array([np.arctan2(delta[1], delta[0])])
    np.testing.assert_array_equal(
        _precompute_view_bins(selected, observer, heading, gaze_model="ray_cast"), [-1]
    )


@pytest.mark.parametrize("restore", [False, True])
def test_joined_rooms_keep_their_separating_wall_for_rays_and_cues(restore):
    rooms = [
        Environment.from_grid_mask(
            np.ones((5, 5), dtype=bool),
            grid_edges=(np.arange(start, start + 6.0), np.arange(6.0)),
            arena=OpenArena(box(start, 0, start + 5, 5)),
        )
        for start in [0, 5]
    ]
    env = Environment.combine(
        dict(zip(["left", "right"], rooms, strict=True)), auto_bridge=False
    )
    if restore:
        env = Environment.from_dict(env.to_dict())
    np.testing.assert_allclose(
        compute_viewed_location([[2.5, 2.5]], [0.0], env=env, method="ray_cast"),
        [[5.0, 2.5]],
    )
    visible, _, _ = visible_cues(env, np.array([2.5, 2.5]), 0.0, np.array([[7.5, 2.5]]))
    assert not visible[0]


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_slanted_wall_cues_remain_visible_at_interpolated_contacts(scale, offset):
    from neurospatial.ops._boundaries import clear_segments

    wall = LineString(np.array([[0.0, 0.0], [3.0, 2.0]]) * scale + offset)
    observer = np.array([0.2, -1.0]) * scale + offset
    targets = np.array(
        [
            wall.interpolate(t, normalized=True).coords[0]
            for t in np.linspace(0.1, 0.9, 9)
        ]
    )
    np.testing.assert_array_equal(clear_segments(wall, observer, targets), True)


def test_wall_cue_rounding_does_not_hide_an_earlier_occluder():
    import shapely

    from neurospatial.ops._boundaries import clear_segments

    wall = LineString([(0.0, 0.0), (3.0, 2.0)])
    observer = np.array([0.2, -1.0])
    target = np.array(wall.interpolate(0.6, normalized=True).coords[0])
    # Both the distant blocker and a blocker close to the cue remain resolved.
    for fraction in (0.5, 1.0 - 1e-12):
        middle = observer + fraction * (target - observer)
        direction = target - observer
        normal = np.array([-direction[1], direction[0]])
        blocker = LineString([middle - 0.1 * normal, middle + 0.1 * normal])
        assert not clear_segments(
            shapely.union_all([wall, blocker]), observer, target[None]
        )[0]


def test_wall_cue_rounding_does_not_allow_collinear_wall_overlap():
    from neurospatial.ops._boundaries import clear_segments

    wall = LineString([(0.0, 0.0), (3.0, 2.0)])
    assert not clear_segments(wall, np.array([0.0, 0.0]), np.array([[3.0, 2.0]]))[0]


def test_visible_cues_accepts_cues_on_a_slanted_measured_arena_wall():
    from shapely.geometry import Polygon

    polygon = Polygon([(0.0, 0.0), (3.0, 2.0), (3.0, -2.0), (0.0, -2.0)])
    env = Environment.from_polygon(polygon, bin_size=0.2)
    wall = LineString([(0.0, 0.0), (3.0, 2.0)])
    targets = np.array(
        [
            wall.interpolate(t, normalized=True).coords[0]
            for t in np.linspace(0.1, 0.9, 9)
        ]
    )
    visible, distances, bearings = visible_cues(
        env, np.array([0.2, -1.0]), 0.0, targets
    )
    np.testing.assert_array_equal(visible, True)
    assert np.all(np.isfinite(distances))
    assert np.all(np.isfinite(bearings))


def test_unowned_physical_wall_hit_refuses_view_encoding_without_fabricating_rates():
    from neurospatial.encoding.view import compute_view_rate

    x, y = np.meshgrid(np.arange(3.0, 98.0), np.arange(3.0, 98.0))
    env = Environment.from_samples(
        np.column_stack((x.ravel(), y.ravel())),
        bin_size=2.0,
        arena=OpenArena(box(0, 0, 100, 100)),
    )
    positions = np.tile([50.0, 50.0], (3, 1))
    # The physical hit is known even though the rate map has no cells there.
    np.testing.assert_allclose(
        compute_viewed_location(positions, np.zeros(3), env=env, method="ray_cast"),
        np.tile([100.0, 50.0], (3, 1)),
    )
    args = (
        env,
        np.array([0.25, 1.25]),
        np.array([0.0, 1.0, 2.0]),
        positions,
        np.zeros(3),
    )
    with pytest.raises(NotImplementedError, match=r"Physical wall hit.*retained bin"):
        compute_view_rate(
            *args, gaze_model="ray_cast", method="binned", bandwidth=0.0, max_gap=None
        )
    approximate = compute_view_rate(
        *args,
        gaze_model="ray_cast",
        method="binned",
        bandwidth=0.0,
        max_gap=None,
        boundary_source="analysis",
    )
    assert approximate.occupancy.sum() == 2.0
    assert np.any(np.isfinite(approximate.firing_rate))


def test_one_unowned_physical_hit_refuses_instead_of_silently_losing_exposure():
    from neurospatial.encoding._view_binning import prepare_view_binning

    full = room(wall=False)
    cropped = full.subset(bins=full.bin_centers[:, 0] < 4)
    positions = np.tile([2.5, 4.5], (3, 1))
    with pytest.raises(NotImplementedError, match=r"Physical wall hit.*retained bin"):
        prepare_view_binning(
            cropped,
            np.arange(3.0),
            positions,
            np.array([np.pi, 0.0, 0.0]),
            gaze_model="ray_cast",
            max_gap=None,
        )


@pytest.mark.parametrize("population", [False, True])
@pytest.mark.parametrize(
    "excluded", ["final", "epoch", "spike_window", "gap", "zero_duration"]
)
def test_unowned_wall_hits_outside_selected_time_do_not_refuse_encoding(
    population, excluded
):
    from neurospatial.encoding.view import compute_view_rate, compute_view_rates

    full = room(wall=False)
    env = full.subset(bins=full.bin_centers[:, 0] < 4)
    times = np.arange(3.0)
    headings = np.array([np.pi, 0.0, 0.0])
    options = {"max_gap": None}
    expected_occupancy = 1.0
    if excluded == "final":
        headings[1] = np.pi
        expected_occupancy = 2.0
    elif excluded == "epoch":
        options["epochs"] = np.array([[0.0, 1.0]])
    elif excluded == "spike_window":
        options["spike_window"] = np.array([[0.0, 1.0]])
    elif excluded == "gap":
        times[-1] = 10.0
        options["max_gap"] = 2.0
    else:
        times = np.array([0.0, 1.0, 1.0, 2.0])
        headings = np.array([np.pi, 0.0, np.pi, 0.0])
        expected_occupancy = 2.0
    positions = np.tile([2.5, 4.5], (len(times), 1))
    spikes = np.array([0.25])
    encode = compute_view_rates if population else compute_view_rate
    result = encode(
        env,
        [spikes] if population else spikes,
        times,
        positions,
        headings,
        gaze_model="ray_cast",
        method="binned",
        bandwidth=0.0,
        **options,
    )
    assert result.occupancy.sum() == expected_occupancy
    rates = result.firing_rates[0] if population else result.firing_rate
    assert np.nansum(rates * result.occupancy) == 1.0


@pytest.mark.parametrize(
    "endpoints,fraction",
    [([[-3.0, -2.0], [1.0, 0.2]], 0.75), ([[-1.0, -0.3], [1.0, 0.7]], 0.5)],
)
def test_slanted_wall_cues_with_cancelled_coordinates_remain_visible(
    endpoints, fraction
):
    from neurospatial.ops._boundaries import clear_segments

    wall = LineString(endpoints)
    target = np.array(wall.interpolate(fraction, normalized=True).coords[0])
    assert target[0] == 0.0
    assert clear_segments(wall, np.array([0.2, -1.0]), target[None])[0]


def test_cue_rounding_does_not_borrow_precision_from_an_unrelated_wall_axis():
    from neurospatial.ops._boundaries import clear_segments

    wall = LineString([(1.0, -1e12), (1.0, 1e12)])
    assert not clear_segments(
        wall, np.array([0.0, 0.0]), np.array([[1.0 + 1e-12, 0.0]])
    )[0]
