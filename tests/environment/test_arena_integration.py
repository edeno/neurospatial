"""Scientific contracts for supplied and inferred arena geometry."""

import pickle
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from neurospatial import Environment, IncompatibleEnvironmentError
from neurospatial.arena import MaskArena, OpenArena
from neurospatial.decoding.posterior import _validate_encoding_environment
from neurospatial.geometry._arena import encode_arena
from neurospatial.ops.transforms import translate


def _arena_record(arena):
    """Compare domain values independently of detached public export identity."""
    metadata, arrays = encode_arena(arena)
    return metadata, {name: array.tolist() for name, array in arrays.items()}


def grid(shape=(4, 4)):
    """Unit cells with a declared computational mask, not measured walls."""
    return Environment.from_grid_mask(
        np.ones(shape, dtype=bool),
        grid_edges=tuple(np.arange(n + 1, dtype=float) for n in shape),
    )


def test_samples_and_image_masks_remain_first_class():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    env = Environment.from_samples(points, bin_size=1.0)
    assert isinstance(env.arena, MaskArena)
    assert env.arena.domain_known and not env.arena.physical_footprint_known
    assert not env.arena.walls
    mask = np.array([[True, False, True], [False, True, False]])
    image = Environment.from_pixel_mask(mask, pixel_size=(2.0, 3.0))
    assert isinstance(image.arena, MaskArena)
    np.testing.assert_array_equal(image.contains(image.bin_centers), True)
    assert image.n_bins == mask.sum()
    assert np.all(image.bin_sizes == 6)


def test_polygon_gate_and_inactive_support_are_distinct():
    env = Environment.from_polygon(Point(0, 0).buffer(5), bin_size=2)
    probes = np.array([[4.9, 1.8], [4.5, 2.5], [0.0, 0.0]])
    np.testing.assert_array_equal(env.bin_at(probes), [-1, -1, 10])
    np.testing.assert_array_equal(env.contains(probes), env.bin_at(probes) >= 0)
    child = env.subset(mask=np.arange(env.n_bins) != 10)
    assert _arena_record(child.arena) == _arena_record(env.arena)
    assert child.arena.contains(np.array([[0.0, 0.0]]))[0]
    assert not child.contains(np.array([[0.0, 0.0]]))[0]


def test_explicit_sample_domain_selects_whole_bins_by_center():
    env = Environment.from_samples(
        np.array([[0.0, 0.0], [3.0, 3.0]]),
        bin_size=1,
        infer_active_bins=False,
        arena=OpenArena(box(0, 0, 2, 2)),
    )
    assert np.all(env.bin_centers > 0) and np.all(env.bin_centers < 2)
    assert np.all(env.bin_sizes == 1)


def test_known_track_width_rejects_distant_points():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0)
    known = Environment.from_graph(
        graph, [(0, 1)], edge_spacing=0.0, bin_size=2.0, track_width=4.0
    )
    abstract = Environment.from_graph(graph, [(0, 1)], edge_spacing=0.0, bin_size=2.0)
    probes = np.array([[50.0, 2.0], [50.0, 2.1], [1000.0, 1000.0]])
    np.testing.assert_array_equal(known.contains(probes), [True, False, False])
    assert not abstract.arena.domain_known
    assert (abstract.bin_at(probes) >= 0).all()
    with pytest.raises(NotImplementedError, match="track_width"):
        abstract.contains(probes)


def test_closed_wall_blocks_navigation_and_diffusion_and_survives_subset():
    arena = OpenArena(box(0, 0, 4, 4), walls=(LineString([(2, 0), (2, 4)]),))
    env = Environment.from_grid_mask(
        np.ones((4, 4), bool),
        grid_edges=(np.arange(5.0), np.arange(5.0)),
        arena=arena,
    )
    left = env.bin_centers[:, 0] < 2
    assert all(left[u] == left[v] for u, v in env.connectivity.edges)
    kernel = env.compute_kernel(bandwidth=1.0)
    assert np.all(kernel[np.ix_(left, ~left)] == 0)
    child = env.subset(mask=env.bin_centers[:, 1] < 2)
    assert _arena_record(child.arena) == _arena_record(arena)
    assert child.arena.walls[0].equals(arena.walls[0])


@pytest.mark.parametrize(
    "wall", [LineString([(1.5, 0), (1.5, 4)]), LineString([(2, 0), (2, 1.5)])]
)
def test_unrepresentable_walls_refuse(wall):
    with pytest.raises(ValueError, match=r"wall|Wall"):
        Environment.from_grid_mask(
            np.ones((4, 4), bool),
            grid_edges=(np.arange(5.0), np.arange(5.0)),
            arena=OpenArena(box(0, 0, 4, 4), walls=(wall,)),
        )


def test_clip_splits_bins_and_preserves_mapping_through_derivation():
    parent = grid((1, 1)).apply_transform(translate(0, 0))
    child = parent.clip(MultiPolygon([box(0, 0, 0.2, 1), box(0.8, 0, 1, 1)]))
    assert child.n_bins == 2
    np.testing.assert_allclose(child.bin_sizes, [0.2, 0.2])
    np.testing.assert_array_equal(child.clip_mapping.parent_bin_indices, [0, 0])
    assert child.clip_mapping is not child._binning.clip_mapping
    assert child.clip_mapping.n_parent_bins == child._binning.clip_mapping.n_parent_bins
    np.testing.assert_array_equal(
        child.clip_mapping.parent_bin_indices,
        child._binning.clip_mapping.parent_bin_indices,
    )
    assert _arena_record(child.arena) == _arena_record(parent.arena)
    assert not child.connectivity.edges
    for variant in (
        child.with_metadata(),
        pickle.loads(pickle.dumps(child)),
        child.apply_transform(translate(10, 0)),
    ):
        np.testing.assert_array_equal(variant.clip_mapping.parent_bin_indices, [0, 0])
        assert not variant.clip_mapping.parent_bin_indices.flags.writeable
    subset = child.subset(mask=np.array([False, True]))
    np.testing.assert_array_equal(subset.clip_mapping.parent_bin_indices, [0])
    assert not subset.contains(np.array([[0.1, 0.5]]))[0]
    assert parent.n_bins == 1


def test_general_clip_uses_known_domain_and_conditional_numerics():
    parent = Environment.from_polygon(Point(0, 0).buffer(3), bin_size=1)
    clipped = parent.clip(box(-10, -10, 10, 10))
    assert clipped.bin_sizes.sum() <= parent.arena.polygon.area + 1e-12
    assert not clipped._binning.faces.available
    with pytest.raises(NotImplementedError, match=r"geometry|orthogonal|convex"):
        clipped.smooth(np.ones(clipped.n_bins), bandwidth=1)
    rectangle = grid((2, 2)).clip(box(0.25, 0, 2, 2))
    assert rectangle._binning.faces.available
    np.testing.assert_allclose(rectangle.smooth(np.ones(4), bandwidth=1), 1)


def test_polygon_hole_is_not_filled_in_field_plot():
    parent = grid((1, 1))
    polygon = Polygon(
        box(0.1, 0.1, 0.9, 0.9).exterior.coords,
        holes=[box(0.3, 0.3, 0.7, 0.7).exterior.coords],
    )
    child = parent.clip(polygon)
    values = child.interpolate(np.ones(1), np.array([[0.5, 0.5], [0.2, 0.5], [2, 2]]))
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert values[1] == 1
    ax = child.plot_field(np.ones(1), colorbar=False)
    ax.figure.canvas.draw()
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())
    x, y = np.rint(ax.transData.transform((0.5, 0.5))).astype(int)
    np.testing.assert_array_equal(pixels[pixels.shape[0] - y, x, :3], 255)
    plt.close(ax.figure)


def test_decoder_checks_actual_polygon_cells_not_only_summaries():
    parent = grid((1, 1))
    square = parent.clip(box(0, 0.25, 1, 0.75))
    diamond = parent.clip(Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]))
    np.testing.assert_array_equal(square.bin_centers, diamond.bin_centers)
    np.testing.assert_array_equal(square.bin_sizes, diamond.bin_sizes)
    with pytest.raises(IncompatibleEnvironmentError, match="polygon cell geometry"):
        _validate_encoding_environment(
            square, SimpleNamespace(env=diamond), context="probe"
        )


def test_seeded_clipped_arena_occupancy_rates_decoding_workflow():
    from neurospatial import compute_spatial_rates
    from neurospatial.decoding import BayesianDecoder

    env = grid((2, 2)).clip(box(0.25, 0, 2, 2))
    env = env.with_metadata(units="cm")
    rng = np.random.default_rng(20261014)
    bins = rng.integers(0, env.n_bins, 401)
    times = np.arange(len(bins)) * 0.05
    positions = env.bin_centers[bins].copy()
    positions[::11] = (-1, -1)
    assigned = env.bin_at(positions)
    spike_rng = np.random.default_rng(20261015)
    accepted = spike_rng.random((2, len(times) - 1)) < 0.4
    spikes = [times[:-1][unit] + 0.01 for unit in accepted]
    rates = compute_spatial_rates(
        env,
        spikes,
        times,
        positions,
        method="binned",
        bandwidth=0,
        min_occupancy=0.01,
        warn_on_drop=False,
    )
    # Encoding requires both endpoints to be observed inside. Use the same
    # intervals for standalone occupancy; its default only gates the start.
    valid = (assigned[:-1] >= 0) & (assigned[1:] >= 0)
    windows = np.column_stack((times[:-1][valid], times[1:][valid]))
    occupancy = env.occupancy(times, positions, epochs=windows)
    np.testing.assert_allclose(rates.occupancy, occupancy)
    expected = np.bincount(assigned[:-1][valid], minlength=env.n_bins) * 0.05
    np.testing.assert_allclose(occupancy, expected)
    spike_bins = env.bin_at(0.8 * positions[:-1] + 0.2 * positions[1:])
    for unit, mask in enumerate(accepted):
        counts = np.bincount(
            spike_bins[mask & valid & (spike_bins >= 0)],
            minlength=env.n_bins,
        )
        np.testing.assert_allclose(rates.firing_rates[unit], counts / occupancy)
    decoder = BayesianDecoder.from_rates(rates, dt=0.05)
    decoded = decoder.predict(spikes, times)
    np.testing.assert_allclose(decoded.posterior.sum(axis=1), 1)
    assert env.contains(decoded.map_position).all()
    restored = Environment.from_dict(env.to_dict())
    np.testing.assert_array_equal(restored.bin_at(positions), assigned)


def test_analytic_domain_needs_materialization_for_exact_clipping():
    class Disk:
        n_dims = 2
        coordinate_kind = "cartesian"
        domain_known = True
        physical_footprint_known = True
        walls = ()

        def contains(self, positions):
            return np.linalg.norm(positions, axis=1) <= 1

        def polygonal_domain(self):
            return None

    with pytest.raises(
        TypeError,
        match=r"arena= takes an arena from neurospatial.arena",
    ):
        Environment.from_grid_mask(
            np.ones((1, 1), bool),
            grid_edges=(np.array([-1.0, 1.0]),) * 2,
            arena=Disk(),
        )
    from shapely.geometry import Point

    env = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([-1.0, 1.0]),) * 2,
        arena=OpenArena(Point(0, 0).buffer(1)),
    )
    np.testing.assert_array_equal(
        env.contains(np.array([[0, 0], [0.9, 0.9]])), [True, False]
    )
    clipped = env.clip(box(-2, -2, 2, 2))
    assert clipped.arena.polygonal_domain().equals(env.arena.polygonal_domain())
