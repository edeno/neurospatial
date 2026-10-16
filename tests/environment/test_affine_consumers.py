"""World queries, source-grid calculations and truthful affine rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding._binning import bin_spike_trains
from neurospatial.io import from_dict, to_dict
from neurospatial.ops.transforms import AffineND

MAPS = [
    np.array([[0.0, -1.0], [1.0, 0.0]]),
    np.diag([-2.0, 3.0]),
    np.array([[1.0, 0.3], [0.0, 1.0]]),
]


def transform(linear, offset=0.0):
    matrix = np.eye(len(linear) + 1)
    matrix[:-1, :-1] = linear
    matrix[:-1, -1] = offset
    return AffineND(matrix)


def grid(dims=2, hole=False):
    mask = np.ones((3,) * dims, dtype=bool)
    if hole:
        mask[(1,) * dims] = False
    return Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * dims)


@pytest.mark.parametrize("linear", MAPS)
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("hole", [False, True])
def test_linear_interpolation_uses_source_axes_and_preserves_missing_support(
    linear, dims, hole
):
    env = grid(dims, hole)
    matrix = np.eye(dims)
    matrix[:2, :2] = linear
    mapping = transform(matrix, 17.0)
    child = env.apply_transform(mapping)
    field = env.bin_centers @ np.arange(2.0, dims + 2)
    points = np.vstack([env.bin_centers, np.full(dims, 1.25), np.full(dims, 4.0)])
    expected = env.interpolate(field, points, mode="linear")
    np.testing.assert_allclose(
        child.interpolate(field, mapping(points), mode="linear"), expected, atol=1e-12
    )
    loaded = from_dict(to_dict(child))
    np.testing.assert_allclose(
        loaded.interpolate(field, mapping(points), mode="linear"), expected, atol=1e-12
    )


@pytest.mark.parametrize("linear", MAPS)
def test_quadmesh_corners_values_masks_and_animation_updates(linear):
    from neurospatial.animation._parallel import _update_field_artist

    env = grid(hole=True)
    mapping = transform(linear, 17.0)
    child = env.apply_transform(mapping)
    values = np.arange(env.n_bins, dtype=float)
    fig, ax = plt.subplots()
    try:
        child.plot_field(values, ax=ax, colorbar=False)
        artist = next(item for item in ax.collections if isinstance(item, QuadMesh))
        x, y = np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="xy")
        expected = mapping(np.column_stack([x.ravel(), y.ravel()])).reshape(4, 4, 2)
        np.testing.assert_allclose(artist.get_coordinates(), expected)
        _update_field_artist(artist, child.binning, values + 10)
        actual = artist.get_array()
        assert actual.mask[1, 1]
        np.testing.assert_array_equal(
            actual.compressed(), (values + 10).reshape(-1)[[0, 3, 5, 1, 6, 2, 4, 7]]
        )
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        child.plot(ax=ax, show_connectivity=False)
        assert any(isinstance(item, QuadMesh) for item in ax.collections)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("linear", MAPS)
@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_linear_counts_and_dwell_share_affine_holes_entry_exit_and_gaps(linear, offset):
    env = grid(hole=True)
    mapping = transform(linear, offset)
    child = env.apply_transform(mapping)
    times = np.array([0.0, 4.0, 8.0, 12.0])
    points = np.array([[-0.5, 1.5], [3.5, 1.5], [3.5, 0.5], [-0.5, 0.5]])
    spikes = np.arange(0.125, 12.0, 0.25)
    options = {"time_allocation": "linear", "max_gap": None, "warn_on_drop": False}
    old_counts, old_dwell = bin_spike_trains(env, [spikes], times, points, **options)
    counts, dwell = bin_spike_trains(child, [spikes], times, mapping(points), **options)
    np.testing.assert_array_equal(counts, old_counts)
    np.testing.assert_allclose(dwell, old_dwell, atol=2e-6)
    np.testing.assert_allclose(
        child.occupancy(times, mapping(points), time_allocation="linear", max_gap=None),
        dwell,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        child.occupancy(times, mapping(points), time_allocation="linear", max_gap=1.0),
        0.0,
    )


@pytest.mark.parametrize("linear", MAPS)
def test_transformed_3d_linear_allocation_matches_native_gates_and_fractions(linear):
    env = grid(3, hole=True)
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    mapping = transform(matrix, 17.0)
    child = env.apply_transform(mapping)
    times = np.arange(4.0)
    points = np.array(
        [[-0.5, 0.5, 0.5], [2.75, 0.5, 0.5], [0.25, 2.75, 2.75], [2.75, 0.25, 0.25]]
    )
    for seconds in (False, True):
        options = {
            "time_allocation": "linear",
            "max_gap": None,
            "return_seconds": seconds,
        }
        np.testing.assert_allclose(
            child.occupancy(times, mapping(points), **options),
            env.occupancy(times, points, **options),
            atol=1e-12,
        )


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_transformed_linear_motion_does_not_cross_physical_walls(offset):
    env = Environment.from_grid_mask(
        np.ones((3, 3), dtype=bool),
        grid_edges=(np.arange(4.0),) * 2,
        arena=OpenArena(box(0, 0, 3, 3), walls=(LineString([(1.0, 0.0), (1.0, 3.0)]),)),
    )
    mapping = transform(np.array([[1.0, 0.3], [0.2, 1.0]]), offset)
    child = env.apply_transform(mapping)
    with pytest.raises(ValueError, match="crosses a physical wall"):
        child.occupancy(
            np.array([0.0, 1.0]),
            mapping(np.array([[0.5, 1.5], [1.5, 1.5]])),
            time_allocation="linear",
            max_gap=None,
        )


def test_napari_raster_helpers_refuse_framed_grids_explicitly():
    from neurospatial.animation.rendering import field_to_rgb_for_napari
    from neurospatial.animation.transforms import EnvScale

    env = grid().apply_transform(transform(MAPS[0]))
    with pytest.raises(NotImplementedError, match=r"framed.*Matplotlib"):
        EnvScale.from_env(env)
    with pytest.raises(NotImplementedError, match=r"framed.*Matplotlib"):
        field_to_rgb_for_napari(
            env, np.ones(env.n_bins), np.zeros((256, 3), dtype=np.uint8), 0.0, 1.0
        )


def test_nearest_interpolation_still_uses_world_euclidean_distance_under_shear():
    from neurospatial.ops.binning import map_points_to_bins

    env = grid().apply_transform(transform(np.array([[1.0, 2.0], [0.0, 1.0]])))
    points = env._binning.grid_frame.to_world(np.array([[1.1, 1.1], [1.9, 1.9]]))
    field = np.arange(env.n_bins, dtype=float)
    owners = map_points_to_bins(env, points)
    np.testing.assert_array_equal(env.interpolate(field, points), field[owners])


def test_interpolation_does_not_snap_nearby_queries_over_a_hole():
    env = grid(hole=True)
    mapping = transform(np.array([[1.0, 0.3], [0.2, 1.0]]), 1e9)
    child = env.apply_transform(mapping)
    field = np.arange(env.n_bins, dtype=float)
    np.testing.assert_array_equal(
        child.interpolate(field, child.bin_centers, mode="linear"), field
    )
    point = mapping(np.array([[0.5 + 1e-4, 1.5]]))
    assert np.isnan(child.interpolate(field, point, mode="linear")[0])


@pytest.mark.parametrize("offset", [0.0, 1e9])
def test_nested_subset_3d_fractions_preserve_native_outside_stop_normalization(offset):
    env = grid(3, hole=True).subset(bins=np.arange(26) % 5 != 1)
    rotation = np.eye(3)
    rotation[:2, :2] = [[0.8, -0.6], [0.6, 0.8]]
    first = transform(rotation, offset)
    second = transform(np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]))
    child = from_dict(to_dict(env.apply_transform(first).apply_transform(second)))
    points = np.array([[0.25, 0.5, 0.5], [3.75, 0.5, 0.5], [1.75, 2.5, 2.5]])
    times = np.array([0.0, 7.0, 9.0])
    for seconds in (True, False):
        options = {
            "time_allocation": "linear",
            "max_gap": None,
            "return_seconds": seconds,
        }
        np.testing.assert_allclose(
            child.occupancy(times, second(first(points)), **options),
            env.occupancy(times, points, **options),
            atol=2e-6,
        )


def test_widget_reuses_transformed_quadmesh_with_updated_values():
    from neurospatial.animation.backends.widget_backend import PersistentFigureRenderer

    env = grid(hole=True).apply_transform(transform(MAPS[2]))
    renderer = PersistentFigureRenderer(env, cmap="viridis", vmin=0.0, vmax=20.0)
    try:
        first = renderer.render(np.zeros(env.n_bins), frame_idx=0)
        second = renderer.render(np.ones(env.n_bins) * 10, frame_idx=1)
        assert first != second
    finally:
        renderer.close()


def test_raster_shuffle_refuses_source_axes_as_world_axes():
    from neurospatial.stats.shuffle import shuffle_place_fields_circular_2d

    env = grid().apply_transform(transform(MAPS[0]))
    with pytest.raises(ValueError, match="world-aligned 2D grid"):
        next(
            shuffle_place_fields_circular_2d(
                env, np.ones((2, env.n_bins)), n_shuffles=1, rng=417
            )
        )


@pytest.mark.parametrize("offset", [0.0, 17.0, 100.0, 1e9])
def test_inverse_roundoff_preserves_interpolation_center_planes_without_filling_holes(
    offset,
):
    env = grid(hole=True)
    angle = 0.3
    mapping = transform(
        np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]),
        offset,
    )
    child = env.apply_transform(mapping)
    field = env.bin_centers @ np.array([2.0, 3.0])
    points = np.array([[0.5, 1.25], [0.5, 1.1], [1.25, 0.5], [1.1, 0.5], [0.5, 1.75]])
    np.testing.assert_allclose(
        child.interpolate(field, mapping(points), mode="linear"),
        env.interpolate(field, points, mode="linear"),
        atol=2e-6,
    )
    points[:, 0] += 1e-4
    points[:, 1] += 1e-4
    assert np.isnan(child.interpolate(field, mapping(points), mode="linear")).all()


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("offset", [0.0, 17.0, 1e9])
def test_boundary_following_paths_keep_half_open_source_ownership(dims, offset):
    env = grid(dims, hole=True)
    angle = 0.3
    linear = np.eye(dims)
    linear[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    mapping = transform(linear, offset)
    child = env.apply_transform(mapping)
    points = np.array(
        [[1.0, 0.5] + [1.5] * (dims - 2), [1.0, 2.5] + [1.5] * (dims - 2)]
    )
    times = np.array([0.0, 1.0])
    options = {"time_allocation": "linear", "max_gap": None}
    expected = env.occupancy(times, points, **options)
    assert expected.sum() == pytest.approx(0.5)
    np.testing.assert_allclose(
        child.occupancy(times, mapping(points), **options), expected, atol=2e-6
    )
    if dims == 2:
        spikes = [np.array([0.125, 0.375, 0.625, 0.875])]
        counts, dwell = bin_spike_trains(
            child, spikes, times, mapping(points), **options, warn_on_drop=False
        )
        assert counts.sum() == 2
        np.testing.assert_allclose(dwell, expected, atol=2e-6)
