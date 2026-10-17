"""NaN marks missing data on every layout, never an error and never zero.

Per-sample queries give a missing sample bin ``-1``, ``False`` or a NaN value,
and the remaining samples get exactly the answers they get on their own.
Smoothing averages over observed bins and leaves unsupported bins NaN.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.layout import create_layout

pytestmark = pytest.mark.filterwarnings(
    "ignore:Triangular mesh is non-K-orthogonal:UserWarning"
)


def _grid() -> Environment:
    positions = np.random.default_rng(0).uniform(0, 30, (2000, 2))
    return Environment.from_samples(positions, bin_size=3.0)


def _holed() -> Environment:
    mask = np.ones((10, 10), dtype=bool)
    mask[3:6, 4:7] = False
    edges = (np.linspace(0, 30, 11), np.linspace(0, 30, 11))
    return Environment.from_grid_mask(mask, grid_edges=edges)


def _hexagonal() -> Environment:
    return Environment.from_layout(
        create_layout(
            "Hexagonal", hexagon_width=4.0, dimension_ranges=[(0, 30), (0, 30)]
        )
    )


def _track() -> Environment:
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)]):
        graph.add_node(node, pos=pos)
    graph.add_edge(0, 1, distance=20.0)
    graph.add_edge(1, 2, distance=20.0)
    return Environment.from_graph(
        graph, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )


def _mesh() -> Environment:
    return Environment.from_layout(
        create_layout(
            "TriangularMesh",
            boundary_polygon=Polygon([(0, 0), (30, 0), (30, 30), (0, 30)]),
            point_spacing=4.0,
        )
    )


def _polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 6,
    )


_CARTESIAN = {
    "grid": _grid,
    "holed": _holed,
    "hexagonal": _hexagonal,
    "track": _track,
    "mesh": _mesh,
}


@pytest.fixture(scope="module", params=sorted(_CARTESIAN))
def cartesian_env(request) -> Environment:
    return _CARTESIAN[request.param]()


@pytest.fixture(scope="module", params=sorted([*_CARTESIAN, "polar"]))
def any_env(request) -> Environment:
    return _polar() if request.param == "polar" else _CARTESIAN[request.param]()


def _with_missing_rows(env: Environment) -> tuple[np.ndarray, np.ndarray]:
    """Points at bin centers with NaN rows interleaved, and the finite rows."""
    finite = np.asarray(env.bin_centers[[0, env.n_bins // 2, env.n_bins - 1]])
    points = np.array(
        [finite[0], [np.nan, finite[0, 1]], finite[1], [np.nan, np.nan], finite[2]]
    )
    return points, finite


_FINITE_ROWS = [0, 2, 4]
_MISSING_ROWS = [1, 3]


def test_bin_at_gives_missing_samples_minus_one(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    bins = cartesian_env.bin_at(points)
    assert (bins[_MISSING_ROWS] == -1).all()
    np.testing.assert_array_equal(bins[_FINITE_ROWS], cartesian_env.bin_at(finite))


def test_contains_gives_missing_samples_false(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    if cartesian_env.is_linearized_track:
        # A track without a declared width has no footprint to test against.
        with pytest.raises(NotImplementedError, match="track footprint"):
            cartesian_env.contains(points)
        return
    inside = cartesian_env.contains(points)
    assert not inside[_MISSING_ROWS].any()
    np.testing.assert_array_equal(inside[_FINITE_ROWS], cartesian_env.contains(finite))


def test_nearest_bin_gives_missing_samples_minus_one_and_nan_distance(any_env):
    points, finite = _with_missing_rows(any_env)
    bins, dist = any_env.nearest_bin(points, return_dist=True)
    expected_bins, expected_dist = any_env.nearest_bin(finite, return_dist=True)
    assert (bins[_MISSING_ROWS] == -1).all()
    assert np.isnan(dist[_MISSING_ROWS]).all()
    np.testing.assert_array_equal(bins[_FINITE_ROWS], expected_bins)
    np.testing.assert_array_equal(dist[_FINITE_ROWS], expected_dist)
    np.testing.assert_array_equal(any_env.nearest_bin(points), bins)


def test_nearest_bin_treats_an_infinite_coordinate_as_outside(any_env):
    point = np.array([[np.inf, 1.0]])
    bins, dist = any_env.nearest_bin(point, return_dist=True)
    assert bins.tolist() == [-1]
    assert dist.tolist() == [np.inf]


def test_interpolate_gives_missing_samples_nan(cartesian_env):
    points, finite = _with_missing_rows(cartesian_env)
    field = np.sin(np.arange(cartesian_env.n_bins) * 0.37)
    values = cartesian_env.interpolate(field, points)
    assert np.isnan(values[_MISSING_ROWS]).all()
    np.testing.assert_array_equal(
        values[_FINITE_ROWS], cartesian_env.interpolate(field, finite)
    )


def test_interpolate_reads_a_missing_bin_value_as_nan(cartesian_env):
    field = np.sin(np.arange(cartesian_env.n_bins) * 0.37)
    field[1] = np.nan
    values = cartesian_env.interpolate(field, cartesian_env.bin_centers)
    np.testing.assert_array_equal(values, field)


def _trajectory(env: Environment) -> tuple[np.ndarray, np.ndarray]:
    """Two samples per bin center along a path, repeated, 30 Hz, no gaps."""
    path = env.path_between(bins=[0, env.n_bins - 1])
    walk = np.resize(np.repeat(np.asarray(path), 2), 60)
    times = np.arange(len(walk)) / 30.0
    return times, np.asarray(env.bin_centers[walk], dtype=float)


def test_occupancy_gives_a_missing_sample_no_time(cartesian_env):
    times, positions = _trajectory(cartesian_env)
    with_missing = positions.copy()
    with_missing[[10, 30]] = np.nan
    occupancy = cartesian_env.occupancy(times, with_missing)

    # The intervals that start at the missing samples are the only time lost.
    dt = np.diff(times)
    np.testing.assert_allclose(occupancy.sum(), dt.sum() - dt[10] - dt[30])
    expected = np.zeros(cartesian_env.n_bins)
    keep = np.setdiff1d(np.arange(len(dt)), [10, 30])
    np.add.at(expected, cartesian_env.bin_at(positions)[keep], dt[keep])
    np.testing.assert_allclose(occupancy, expected)


def test_bin_sequence_marks_or_drops_a_missing_sample(cartesian_env):
    times, positions = _trajectory(cartesian_env)
    positions[10] = np.nan
    bins = cartesian_env.bin_at(positions)
    marked = cartesian_env.bin_sequence(times, positions, dedup=False)
    dropped = cartesian_env.bin_sequence(
        times, positions, dedup=False, outside_value=None
    )
    np.testing.assert_array_equal(marked, bins)
    assert marked[10] == -1
    np.testing.assert_array_equal(dropped, np.delete(bins, 10))


@pytest.mark.parametrize("lag", [1, 2])
def test_transitions_skip_pairs_with_a_missing_or_outside_sample(cartesian_env, lag):
    times, positions = _trajectory(cartesian_env)
    positions[10] = np.nan
    positions[20] = cartesian_env.bin_centers.max(axis=0) + 100.0
    counts = cartesian_env.transitions(
        times=times, positions=positions, lag=lag, normalize=False, allow_teleports=True
    )

    # Count the pairs whose two samples are both inside, by hand.
    bins = cartesian_env.bin_at(positions)
    expected = np.zeros((cartesian_env.n_bins, cartesian_env.n_bins))
    for source, target in zip(bins[:-lag], bins[lag:], strict=True):
        if source >= 0 and target >= 0:
            expected[source, target] += 1
    np.testing.assert_array_equal(counts.toarray(), expected)
    # The missing sample removes the pair it starts and the one it ends.
    assert bins[10] == -1
    assert counts.sum() <= len(bins) - lag - 2


def test_a_precomputed_sequence_with_minus_one_is_still_refused(cartesian_env):
    with pytest.raises(ValueError, match="Invalid bin indices"):
        cartesian_env.transitions(np.array([0, -1, 0], dtype=np.int32))


# --- Smoothing over observed bins -------------------------------------------

_MODES = ["average", "transition", "density"]


def _holes(env: Environment) -> np.ndarray:
    return np.random.default_rng(1).random(env.n_bins) < 0.3


@pytest.mark.parametrize("mode", _MODES)
def test_smoothing_without_nan_is_the_plain_operator(any_env, mode):
    from neurospatial.ops.diffusion import (
        DiffusionEigenbasis,
        _resolve_basis,
        diffusion_apply,
    )

    field = np.sin(np.arange(any_env.n_bins) * 0.37)
    geometry = any_env.get_diffusion_geometry()
    _, basis = _resolve_basis(
        DiffusionEigenbasis(),
        geometry.W,
        geometry.volumes,
        geometry.n_components,
        3.0,
    )
    plain = diffusion_apply(basis, geometry.volumes, field[:, None], 3.0, mode)[:, 0]
    np.testing.assert_array_equal(
        any_env.smooth(field, bandwidth=3.0, mode=mode), plain
    )
    np.testing.assert_array_equal(
        any_env.diffuse(field, bandwidth=3.0, mode=mode), plain
    )


def test_a_constant_field_with_holes_stays_constant(any_env):
    holed = np.full(any_env.n_bins, 2.5)
    holed[_holes(any_env)] = np.nan
    smoothed = any_env.smooth(holed, bandwidth=3.0)
    assert np.isfinite(smoothed).all()
    np.testing.assert_allclose(smoothed, 2.5, rtol=1e-11)


@pytest.mark.parametrize("mode", ["transition", "density"])
@pytest.mark.parametrize("call", ["smooth", "diffuse"])
def test_extensive_modes_refuse_missing_bins_with_a_choice(any_env, mode, call):
    counts = np.ones(any_env.n_bins)
    counts[[0, 1]] = np.nan
    with pytest.raises(ValueError) as error:
        getattr(any_env, call)(counts, bandwidth=3.0, mode=mode)
    message = str(error.value)
    assert f"2 NaN (missing) bins in mode='{mode}'" in message
    assert "no defensible mass under a conserving operator" in message
    assert "np.nan_to_num" in message and "mode='average'" in message


def test_concrete_jax_input_with_nan_is_refused(any_env):
    jnp = pytest.importorskip("jax.numpy")
    field = np.ones(any_env.n_bins)
    field[0] = np.nan
    with pytest.raises(ValueError, match="backend='numpy'"):
        any_env.diffuse(
            jnp.asarray(field), bandwidth=3.0, mode="average", backend="jax"
        )
    # A NaN-free JAX field is the plain operator.
    np.testing.assert_allclose(
        np.asarray(
            any_env.diffuse(
                jnp.asarray(np.ones(any_env.n_bins)), bandwidth=3.0, backend="jax"
            )
        ),
        any_env.diffuse(np.ones(any_env.n_bins), bandwidth=3.0),
        rtol=1e-5,
    )


def test_traced_jax_input_is_not_inspected(any_env):
    jax = pytest.importorskip("jax")
    field = np.ones(any_env.n_bins)
    field[0] = np.nan
    smoothed = jax.jit(
        lambda f: any_env.diffuse(f, bandwidth=2.0, mode="average", backend="jax")
    )(jax.numpy.asarray(field))
    # The documented limitation: a traced apply is plain linear algebra, so
    # the NaN reaches every bin.
    assert np.isnan(np.asarray(smoothed)).all()


def test_an_interior_hole_is_filled_from_its_neighbors(any_env):
    field = np.sin(np.arange(any_env.n_bins) * 0.37)
    holed = field.copy()
    holed[any_env.n_bins // 2] = np.nan
    smoothed = any_env.smooth(holed, bandwidth=3.0)
    assert np.isfinite(smoothed).all()
    assert field.min() <= smoothed[any_env.n_bins // 2] <= field.max()


def test_a_batch_keeps_nan_free_fields_unchanged(any_env):
    field = np.sin(np.arange(any_env.n_bins) * 0.37)
    holed = np.cos(np.arange(any_env.n_bins) * 0.11)
    holed[_holes(any_env)] = np.nan
    batch = any_env.diffuse(
        np.column_stack([field, holed]), bandwidth=3.0, mode="average"
    )
    filled = any_env.diffuse(
        np.column_stack([field, np.nan_to_num(holed)]), bandwidth=3.0, mode="average"
    )
    np.testing.assert_array_equal(batch[:, 0], filled[:, 0])
    # Each field has its own observed mask; batching changes only roundoff.
    np.testing.assert_allclose(
        batch[:, 1], any_env.smooth(holed, bandwidth=3.0), rtol=1e-12, atol=1e-14
    )


@pytest.fixture(scope="module")
def two_rooms() -> Environment:
    """Two 5 x 4 rooms on one grid, separated by an inactive column."""
    mask = np.ones((11, 4), dtype=bool)
    mask[5] = False
    edges = (np.linspace(0, 22, 12), np.linspace(0, 8, 5))
    return Environment.from_grid_mask(mask, grid_edges=edges)


def test_a_room_with_no_observed_bin_stays_nan(two_rooms):
    labels = two_rooms.get_diffusion_geometry().labels
    assert two_rooms.get_diffusion_geometry().n_components == 2
    field = np.sin(np.arange(two_rooms.n_bins) * 0.37)
    holed = np.where(labels == 1, np.nan, field)
    smoothed = two_rooms.smooth(holed, bandwidth=2.0)

    assert np.isnan(smoothed[labels == 1]).all()
    # The observed room is fully observed, so it smooths as if alone.
    reference = two_rooms.smooth(np.where(labels == 1, 0.0, field), bandwidth=2.0)
    np.testing.assert_allclose(
        smoothed[labels == 0], reference[labels == 0], rtol=1e-12
    )


def test_bins_stay_nan_exactly_where_the_support_is_below_the_floor():
    from neurospatial.environment._fields import MISSING_SUPPORT_FLOOR

    track = Environment.from_samples(np.linspace(0.0, 60.0, 400)[:, None], bin_size=1.0)
    observed = track.bin_centers[:, 0] < 15.0
    field = np.where(observed, np.cos(track.bin_centers[:, 0] / 4.0), np.nan)
    smoothed = track.smooth(field, bandwidth=2.0)

    # The share of each bin's smoothing weight that falls on observed bins.
    support = (
        track.compute_kernel(bandwidth=2.0, mode="average", cache=False) @ observed
    )
    relative = support / support.max()
    clear = np.abs(relative / MISSING_SUPPORT_FLOOR - 1.0) > 0.01
    np.testing.assert_array_equal(
        np.isnan(smoothed)[clear], (relative <= MISSING_SUPPORT_FLOOR)[clear]
    )
    # Holes next to the observed bins are filled; far ones stay NaN.
    assert np.isfinite(smoothed[~observed]).any()
    assert np.isnan(smoothed).any()


def test_sparse_observations_with_a_wide_bandwidth_are_not_all_nan():
    track = Environment.from_samples(
        np.linspace(0.0, 200.0, 800)[:, None], bin_size=1.0
    )
    field = np.full(track.n_bins, np.nan)
    field[::25] = 1.0
    smoothed = track.smooth(field, bandwidth=30.0)
    assert np.isfinite(smoothed).all()
    np.testing.assert_allclose(smoothed, 1.0, rtol=1e-6)


def test_smooth_still_refuses_infinite_values(any_env):
    field = np.zeros(any_env.n_bins)
    field[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        any_env.smooth(field, bandwidth=3.0)


def _polygon() -> Environment:
    from shapely.geometry import Point

    return Environment.from_polygon(Point(15, 15).buffer(15), bin_size=3.0)


_OCCUPANCY_ENVS = {**_CARTESIAN, "polygon": _polygon}

# A track without a declared width maps every finite point to its nearest
# bin, so it has no "outside" case. Start allocation keeps intervals that
# start inside, so only linear allocation has the inside-then-NaN case.
_NO_USABLE_SAMPLE_CASES = [
    (name, samples, allocation)
    for name in sorted(_OCCUPANCY_ENVS)
    for samples in ("all_nan", "all_outside", "inside_then_nan")
    for allocation in ("start", "linear")
    if not (name == "track" and samples == "all_outside")
    and not (samples == "inside_then_nan" and allocation == "start")
]


@pytest.fixture(scope="module")
def occupancy_envs() -> dict[str, Environment]:
    return {name: build() for name, build in _OCCUPANCY_ENVS.items()}


@pytest.mark.parametrize(("name", "samples", "allocation"), _NO_USABLE_SAMPLE_CASES)
def test_occupancy_with_no_usable_sample_warns(
    occupancy_envs, name, samples, allocation
):
    env = occupancy_envs[name]
    times = np.arange(8) / 30.0
    inside = np.tile(np.asarray(env.bin_centers[0], dtype=float), (8, 1))
    try:
        env.occupancy(times, inside, time_allocation=allocation)
    except NotImplementedError as error:
        # Linear allocation needs rectangular or exact polygon cells.
        assert allocation == "linear"
        assert "time_allocation='linear'" in str(error)
        return

    if samples == "all_nan":
        positions = np.full((8, 2), np.nan)
    elif samples == "all_outside":
        positions = np.tile(env.bin_centers.max(axis=0) + 100.0, (8, 1))
    else:
        # Every interval has a NaN end, so a straight path has no length.
        positions = inside.copy()
        positions[1::2] = np.nan
    with pytest.warns(
        UserWarning, match="excluded ALL.*so occupancy is zero in every bin"
    ):
        occupancy = env.occupancy(times, positions, time_allocation=allocation)
    np.testing.assert_array_equal(occupancy, 0.0)
