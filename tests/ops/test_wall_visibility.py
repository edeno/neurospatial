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
