"""Tests for the radial (graph-distance) autocorrelation profile."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.encoding.grid import _spatial_autocorrelation_graph
from neurospatial.simulation import GridCellModel


@pytest.fixture
def grid_cell_setup():
    """Create environment and grid cell rate map for testing."""
    from shapely.geometry import box

    # Create small environment for fast tests
    square = box(0, 0, 50, 50)
    env = Environment.from_polygon(square, bin_size=2.0)

    # Create grid cell
    grid_cell = GridCellModel(env, grid_spacing=20.0, max_rate=20.0)

    # Get rate map directly from firing rates at bin centers
    rate_map = grid_cell.firing_rate(np.zeros(len(env.bin_centers)), env.bin_centers)

    return env, rate_map


def test_radial_profile_has_one_value_per_distance_bin(grid_cell_setup):
    """The profile has one sorted center and correlation per distance bin."""
    env, rate_map = grid_cell_setup

    distances, correlations = _spatial_autocorrelation_graph(
        env, rate_map, n_distance_bins=30
    )

    # Basic checks
    assert len(distances) == 30
    assert len(correlations) == 30
    assert distances[0] < distances[-1]  # Sorted
    assert np.any(np.isfinite(correlations))  # Some valid correlations


def test_radial_profile_is_deterministic(grid_cell_setup):
    """Test production implementation produces consistent results across runs."""
    env, rate_map = grid_cell_setup

    # Run twice and verify results are identical
    distances1, correlations1 = _spatial_autocorrelation_graph(
        env, rate_map, n_distance_bins=30
    )
    distances2, correlations2 = _spatial_autocorrelation_graph(
        env, rate_map, n_distance_bins=30
    )

    assert_allclose(distances1, distances2, rtol=1e-10)
    # Use nan-aware comparison for correlations
    assert_allclose(
        np.nan_to_num(correlations1, nan=-999),
        np.nan_to_num(correlations2, nan=-999),
        rtol=1e-10,
    )


def test_radial_profile_matches_pairwise_pearson_by_distance():
    """Each distance bin is the Pearson correlation of the bin pairs in it."""
    from shapely.geometry import box

    env = Environment.from_polygon(box(0, 0, 30, 30), bin_size=3.0)
    rate_map = np.random.default_rng(0).gamma(2.0, 1.0, env.n_bins)
    rate_map[[4, 17]] = np.nan
    n_distance_bins = 12

    distances, correlations = _spatial_autocorrelation_graph(
        env, rate_map, n_distance_bins=n_distance_bins
    )

    valid = np.flatnonzero(np.isfinite(rate_map))
    # Paths may cross unvisited bins: they are still part of the arena. In
    # this convex room the shortest path is the straight line.
    centers = env.bin_centers
    pairs = [
        (np.linalg.norm(centers[a] - centers[b]), rate_map[a], rate_map[b])
        for i, a in enumerate(valid)
        for b in valid[i + 1 :]
    ]
    pair_distance, first, second = np.array(pairs).T
    edges = np.linspace(0.0, pair_distance.max(), n_distance_bins + 1)
    expected = np.full(n_distance_bins, np.nan)
    for k in range(n_distance_bins):
        inside = (pair_distance >= edges[k]) & (pair_distance < edges[k + 1])
        if inside.sum() >= 2:
            # Each unordered pair counts in both orientations.
            expected[k] = np.corrcoef(
                np.r_[first[inside], second[inside]],
                np.r_[second[inside], first[inside]],
            )[0, 1]

    assert_allclose(distances, (edges[:-1] + edges[1:]) / 2)
    assert_allclose(correlations, expected, atol=1e-12)


def test_radial_profile_does_not_depend_on_bin_numbering():
    """Renumbering the nodes of a path graph leaves the profile unchanged."""
    from neurospatial.binning import (
        BinConnections,
        Binning,
        MissingFaces,
        NearestBinLocator,
        SpatialBins,
    )
    from neurospatial.encoding.grid import spatial_autocorrelation_radial

    centers = np.column_stack([np.arange(5.0), np.zeros(5)])
    profiles = []
    for order in (np.arange(5), np.array([0, 2, 4, 1, 3])):
        # New bin i is old bin order[i]; the path still runs 0-1-2-3-4.
        pairs = np.argsort(order)[np.column_stack([np.arange(4), np.arange(1, 5)])]
        env = Environment.from_bins(
            Binning(
                "graph",
                SpatialBins.from_arrays(centers[order], np.ones(5), 1),
                BinConnections.from_arrays(pairs, np.ones(4)),
                NearestBinLocator(centers[order], max_distance=0.5),
                MissingFaces("Graph only."),
            )
        )
        profiles.append(
            spatial_autocorrelation_radial(
                env, np.arange(1.0, 6.0)[order], n_distance_bins=4, max_distance=4.1
            )[1]
        )

    assert_allclose(profiles[0], profiles[1], atol=1e-12)
    # Neighbors on a monotonic ramp are positively correlated.
    assert profiles[0][0] > 0


def _pearson_by_distance(pair_distances, rates, n_distance_bins):
    """Pearson correlation of the bin pairs in each distance band, by loops.

    ``pair_distances`` is the (n, n) distance between the bins of ``rates``.
    """
    upper = np.triu_indices(len(rates), k=1)
    pair_distance = pair_distances[upper]
    first, second = rates[upper[0]], rates[upper[1]]
    edges = np.linspace(0.0, pair_distance.max(), n_distance_bins + 1)
    expected = np.full(n_distance_bins, np.nan)
    for k in range(n_distance_bins):
        inside = (pair_distance >= edges[k]) & (pair_distance < edges[k + 1])
        if inside.sum() >= 2:
            # Each unordered pair counts in both orientations.
            expected[k] = np.corrcoef(
                np.r_[first[inside], second[inside]],
                np.r_[second[inside], first[inside]],
            )[0, 1]
    return expected


def _maps_with_different_missing_bins(env, n_maps):
    """Noisy maps of one environment, each with its own unvisited bins."""
    rng = np.random.default_rng(4)
    maps = rng.gamma(2.0, 1.0, (n_maps, env.n_bins))
    for k, rate_map in enumerate(maps):
        rate_map[rng.choice(env.n_bins, size=3 * k, replace=False)] = np.nan
    return maps


def test_radial_profiles_of_a_population_measure_the_environment_once(monkeypatch):
    """The pair distances belong to the environment, not to a unit's map.

    Every unit of a population reads the same distances, whichever bins its
    map is missing, so they are measured once. In a convex room the distance
    is the straight line between bin centers, which gives each profile
    without the code under test.
    """
    from shapely.geometry import box

    from neurospatial.encoding import spatial_autocorrelation_radial
    from neurospatial.ops import distance

    env = Environment.from_polygon(box(0, 0, 30, 30), bin_size=3.0)
    maps = _maps_with_different_missing_bins(env, n_maps=5)
    measured = []
    measure = distance.pairwise_distances

    def counting(env, bins, **kwargs):
        measured.append(len(bins))
        return measure(env, bins, **kwargs)

    monkeypatch.setattr(distance, "pairwise_distances", counting)

    centers = env.bin_centers
    for rate_map in maps:
        _, correlations = spatial_autocorrelation_radial(
            env, rate_map, n_distance_bins=12
        )
        valid = np.flatnonzero(np.isfinite(rate_map))
        straight = np.linalg.norm(centers[valid, None] - centers[valid], axis=2)
        assert_allclose(
            correlations,
            _pearson_by_distance(straight, rate_map[valid], 12),
            atol=1e-12,
        )
    assert measured == [env.n_bins]


def test_radial_profile_reads_the_same_distances_behind_a_wall():
    """Kept distances are the ones measured for a map's own bins: round walls."""
    from shapely.geometry import LineString, box

    from neurospatial.arena import OpenArena
    from neurospatial.encoding import spatial_autocorrelation_radial
    from neurospatial.ops.distance import pairwise_distances

    outline = box(0, 0, 30, 30)
    wall = LineString([(15, 0), (15, 21)])
    env = Environment.from_polygon(
        outline, bin_size=3.0, arena=OpenArena(outline, walls={"wall": wall})
    )
    for rate_map in _maps_with_different_missing_bins(env, n_maps=3):
        valid = np.flatnonzero(np.isfinite(rate_map))
        around_the_wall = pairwise_distances(env, valid)
        # Two bins facing each other across the wall are farther than the line.
        assert around_the_wall.max() > np.hypot(30, 30)

        _, correlations = spatial_autocorrelation_radial(
            env, rate_map, n_distance_bins=12
        )

        assert_allclose(
            correlations,
            _pearson_by_distance(around_the_wall, rate_map[valid], 12),
            atol=1e-12,
        )


def test_radial_profile_of_a_large_environment_keeps_no_distance_matrix(monkeypatch):
    """Above the memory the path engine keeps, each map measures its own pairs."""
    from shapely.geometry import box

    from neurospatial.encoding import spatial_autocorrelation_radial
    from neurospatial.ops import _paths, distance

    env = Environment.from_polygon(box(0, 0, 30, 30), bin_size=3.0)
    maps = _maps_with_different_missing_bins(env, n_maps=3)
    kept = [spatial_autocorrelation_radial(env, m, n_distance_bins=12) for m in maps]
    measured = []
    measure = distance.pairwise_distances

    def counting(env, bins, **kwargs):
        measured.append(len(bins))
        return measure(env, bins, **kwargs)

    monkeypatch.setattr(distance, "pairwise_distances", counting)
    # The 100 x 100 matrix of this environment no longer fits.
    monkeypatch.setattr(_paths, "_FIELD_CACHE_BYTES", 8 * env.n_bins**2 - 1)

    for rate_map, (distances, correlations) in zip(maps, kept, strict=True):
        again = spatial_autocorrelation_radial(env, rate_map, n_distance_bins=12)
        np.testing.assert_array_equal(again[0], distances)
        np.testing.assert_array_equal(again[1], correlations)
    assert measured == [int(np.isfinite(m).sum()) for m in maps]
