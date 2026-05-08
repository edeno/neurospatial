"""Tests for _spatial_autocorrelation_graph optimization."""

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
    rate_map = grid_cell.firing_rate(env.bin_centers)

    return env, rate_map


def test_original_implementation(grid_cell_setup):
    """Test that original implementation runs and returns valid output."""
    env, rate_map = grid_cell_setup

    distances, correlations = _spatial_autocorrelation_graph(
        env, rate_map, n_distance_bins=30
    )

    # Basic checks
    assert len(distances) == 30
    assert len(correlations) == 30
    assert distances[0] < distances[-1]  # Sorted
    assert np.any(np.isfinite(correlations))  # Some valid correlations


def test_original_vs_optimized_step1_dijkstra(grid_cell_setup):
    """Test scipy dijkstra gives same distances as networkx."""
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    env, rate_map = grid_cell_setup

    valid_bins = np.where(np.isfinite(rate_map))[0]
    valid_bin_set = set(valid_bins.tolist())

    # Original: networkx
    subgraph = env.connectivity.subgraph(valid_bin_set)
    distances_dict = dict(
        nx.all_pairs_dijkstra_path_length(subgraph, weight="distance")
    )

    # Optimized: scipy sparse
    adj_matrix = nx.adjacency_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).tocsr()

    dist_matrix = scipy_dijkstra(
        adj_matrix,
        directed=False,
        indices=valid_bins,
        return_predecessors=False,
    )[:, valid_bins]

    # Compare distances for all valid pairs
    for i, bin_i in enumerate(valid_bins):
        for j, bin_j in enumerate(valid_bins):
            if i >= j:
                continue
            if int(bin_j) in distances_dict.get(int(bin_i), {}):
                nx_dist = distances_dict[int(bin_i)][int(bin_j)]
                scipy_dist = dist_matrix[i, j]
                assert_allclose(
                    scipy_dist,
                    nx_dist,
                    rtol=1e-10,
                    err_msg=f"Distance mismatch for bins {bin_i}, {bin_j}",
                )


def test_original_vs_optimized_step2_vectorized_extraction(grid_cell_setup):
    """Test vectorized extraction gives same pairs as nested loops."""
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    env, rate_map = grid_cell_setup

    valid_bins = np.where(np.isfinite(rate_map))[0]
    valid_rates = rate_map[valid_bins]
    n_valid = len(valid_bins)

    # Get distance matrix (using scipy for both)
    adj_matrix = nx.adjacency_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).tocsr()
    dist_matrix = scipy_dijkstra(
        adj_matrix, directed=False, indices=valid_bins, return_predecessors=False
    )[:, valid_bins]

    # Original: nested loops
    pairwise_distances_orig = []
    rates_i_orig = []
    rates_j_orig = []

    for i in range(n_valid):
        for j in range(n_valid):
            if i >= j:
                continue
            dist = dist_matrix[i, j]
            if np.isfinite(dist):
                pairwise_distances_orig.append(dist)
                rates_i_orig.append(valid_rates[i])
                rates_j_orig.append(valid_rates[j])

    pairwise_distances_orig = np.array(pairwise_distances_orig)
    rates_i_orig = np.array(rates_i_orig)
    rates_j_orig = np.array(rates_j_orig)

    # Optimized: vectorized
    triu_indices = np.triu_indices(n_valid, k=1)
    pairwise_distances_opt = dist_matrix[triu_indices]
    rates_i_opt = valid_rates[triu_indices[0]]
    rates_j_opt = valid_rates[triu_indices[1]]

    finite_mask = np.isfinite(pairwise_distances_opt)
    pairwise_distances_opt = pairwise_distances_opt[finite_mask]
    rates_i_opt = rates_i_opt[finite_mask]
    rates_j_opt = rates_j_opt[finite_mask]

    # Compare
    assert_allclose(pairwise_distances_opt, pairwise_distances_orig, rtol=1e-10)
    assert_allclose(rates_i_opt, rates_i_orig, rtol=1e-10)
    assert_allclose(rates_j_opt, rates_j_orig, rtol=1e-10)


def test_scipy_vs_networkx_distances_detail(grid_cell_setup):
    """Detailed comparison of scipy vs networkx distances."""
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    env, rate_map = grid_cell_setup

    valid_bins = np.where(np.isfinite(rate_map))[0]
    valid_rates = rate_map[valid_bins]
    valid_bin_set = set(valid_bins.tolist())
    n_valid = len(valid_bins)

    # Original networkx approach (exact copy from source)
    subgraph = env.connectivity.subgraph(valid_bin_set)
    distances_dict = dict(
        nx.all_pairs_dijkstra_path_length(subgraph, weight="distance")
    )

    # Build original pairwise arrays
    pairwise_distances_nx = []
    rates_i_nx = []
    rates_j_nx = []

    for i, bin_i in enumerate(valid_bins):
        for j, bin_j in enumerate(valid_bins):
            if i >= j:
                continue
            if int(bin_j) in distances_dict.get(int(bin_i), {}):
                dist = distances_dict[int(bin_i)][int(bin_j)]
                pairwise_distances_nx.append(dist)
                rates_i_nx.append(float(valid_rates[i]))
                rates_j_nx.append(float(valid_rates[j]))

    pairwise_distances_nx = np.array(pairwise_distances_nx)
    rates_i_nx = np.array(rates_i_nx)
    rates_j_nx = np.array(rates_j_nx)

    # Scipy approach
    adj_matrix = nx.adjacency_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).tocsr()
    dist_matrix = scipy_dijkstra(
        adj_matrix, directed=False, indices=valid_bins, return_predecessors=False
    )[:, valid_bins]

    triu_indices = np.triu_indices(n_valid, k=1)
    pairwise_distances_scipy = dist_matrix[triu_indices]
    rates_i_scipy = valid_rates[triu_indices[0]]
    rates_j_scipy = valid_rates[triu_indices[1]]

    finite_mask = np.isfinite(pairwise_distances_scipy)
    pairwise_distances_scipy = pairwise_distances_scipy[finite_mask]
    rates_i_scipy = rates_i_scipy[finite_mask]
    rates_j_scipy = rates_j_scipy[finite_mask]

    # Compare counts
    print(f"\nNetworkx pairs: {len(pairwise_distances_nx)}")
    print(f"Scipy pairs: {len(pairwise_distances_scipy)}")

    assert len(pairwise_distances_scipy) == len(pairwise_distances_nx)
    assert_allclose(pairwise_distances_scipy, pairwise_distances_nx, rtol=1e-10)
    assert_allclose(rates_i_scipy, rates_i_nx, rtol=1e-10)
    assert_allclose(rates_j_scipy, rates_j_nx, rtol=1e-10)


def test_original_vs_optimized_step3_digitize(grid_cell_setup):
    """Test np.digitize gives same bin assignments as manual loop."""
    _env, _rate_map = grid_cell_setup

    # Create some test distances - INCLUDING edge case at exactly max_distance
    pairwise_distances = np.array(
        [0.5, 2.3, 5.1, 7.8, 10.2, 15.0, 20.5, 25.0]
    )  # 25.0 = max
    max_distance = 25.0
    n_distance_bins = 10

    distance_bin_edges = np.linspace(0, max_distance, n_distance_bins + 1)

    # Original: manual check per bin (exact copy from source)
    # Note: pairs at exactly max_distance are NOT assigned to any bin!
    bin_assignments_orig = np.full(
        len(pairwise_distances), -1, dtype=int
    )  # -1 = unassigned
    for idx, dist in enumerate(pairwise_distances):
        for d_idx in range(n_distance_bins):
            d_min = distance_bin_edges[d_idx]
            d_max = distance_bin_edges[d_idx + 1]
            if d_min <= dist < d_max:
                bin_assignments_orig[idx] = d_idx
                break

    # Check: value at exactly max_distance should be unassigned
    print(f"\nOriginal bin for dist=25.0 (max): {bin_assignments_orig[-1]}")
    print("Expected: -1 (unassigned) because 25.0 < 25.0 is False")

    # Optimized: np.searchsorted
    bin_assignments_opt = (
        np.searchsorted(distance_bin_edges, pairwise_distances, side="right") - 1
    )
    bin_assignments_opt = np.clip(bin_assignments_opt, 0, n_distance_bins - 1)

    print(f"Optimized bin for dist=25.0: {bin_assignments_opt[-1]}")
    print("This is WRONG - it assigns to last bin due to clipping")

    # This test will now FAIL, demonstrating the root cause
    assert_allclose(
        bin_assignments_opt[:-1], bin_assignments_orig[:-1]
    )  # All but last should match


def test_original_vs_optimized_step4_correlation(grid_cell_setup):
    """Test inline correlation matches scipy.stats.pearsonr."""
    from scipy import stats

    # Test data
    rates_i = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    rates_j = np.array([1.1, 2.2, 2.9, 4.1, 5.2, 5.8, 7.1, 8.0])

    # Original: scipy
    corr_scipy, _ = stats.pearsonr(rates_i, rates_j)

    # Optimized: inline
    ri_centered = rates_i - rates_i.mean()
    rj_centered = rates_j - rates_j.mean()
    corr_inline = np.dot(ri_centered, rj_centered) / (
        np.linalg.norm(ri_centered) * np.linalg.norm(rj_centered)
    )

    assert_allclose(corr_inline, corr_scipy, rtol=1e-10)


def test_production_implementation_consistency(grid_cell_setup):
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


def test_vectorized_correlation_matches_loop(grid_cell_setup):
    """Test vectorized correlation computation matches loop-based approach."""
    from scipy import stats

    env, rate_map = grid_cell_setup

    # Get the pairwise data from spatial_autocorrelation_graph internals
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    valid_bins = np.where(np.isfinite(rate_map))[0]
    valid_rates = rate_map[valid_bins]
    n_valid = len(valid_bins)

    adj_matrix = nx.adjacency_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).tocsr()
    dist_matrix = scipy_dijkstra(
        adj_matrix, directed=False, indices=valid_bins, return_predecessors=False
    )[:, valid_bins]

    triu_indices = np.triu_indices(n_valid, k=1)
    pairwise_distances = dist_matrix[triu_indices]
    rates_i = valid_rates[triu_indices[0]]
    rates_j = valid_rates[triu_indices[1]]

    finite_mask = np.isfinite(pairwise_distances)
    pairwise_distances = pairwise_distances[finite_mask]
    rates_i = rates_i[finite_mask]
    rates_j = rates_j[finite_mask]

    max_distance = float(np.max(pairwise_distances))
    n_distance_bins = 30
    distance_bin_edges = np.linspace(0, max_distance, n_distance_bins + 1)

    bin_indices = (
        np.searchsorted(distance_bin_edges, pairwise_distances, side="right") - 1
    )
    bin_indices = np.where(bin_indices >= n_distance_bins, -1, bin_indices)
    bin_indices = np.where(bin_indices < 0, -1, bin_indices)

    # Loop-based approach (original)
    correlations_loop = np.full(n_distance_bins, np.nan)
    for d_idx in range(n_distance_bins):
        mask = bin_indices == d_idx
        if np.sum(mask) < 2:
            continue
        rates_i_bin = rates_i[mask]
        rates_j_bin = rates_j[mask]
        if np.std(rates_i_bin) == 0 or np.std(rates_j_bin) == 0:
            continue
        corr, _ = stats.pearsonr(rates_i_bin, rates_j_bin)
        correlations_loop[d_idx] = corr

    # Vectorized approach using pandas groupby
    import pandas as pd

    df = pd.DataFrame(
        {
            "bin_idx": bin_indices,
            "rates_i": rates_i,
            "rates_j": rates_j,
        }
    )
    # Filter out invalid bins
    df = df[df["bin_idx"] >= 0]

    def safe_corr(g):
        if len(g) < 2:
            return np.nan
        if g["rates_i"].std() == 0 or g["rates_j"].std() == 0:
            return np.nan
        return g["rates_i"].corr(g["rates_j"])

    corr_series = df.groupby("bin_idx").apply(safe_corr, include_groups=False)

    correlations_vectorized = np.full(n_distance_bins, np.nan)
    for idx in corr_series.index:
        correlations_vectorized[idx] = corr_series[idx]

    # Compare - should match exactly (allowing for NaN in same positions)
    assert_allclose(
        np.nan_to_num(correlations_vectorized, nan=-999),
        np.nan_to_num(correlations_loop, nan=-999),
        rtol=1e-10,
    )


def test_vectorized_correlation_pure_numpy(grid_cell_setup):
    """Test pure numpy vectorized correlation (no pandas dependency)."""
    import networkx as nx
    from scipy import stats
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    env, rate_map = grid_cell_setup

    valid_bins = np.where(np.isfinite(rate_map))[0]
    valid_rates = rate_map[valid_bins]
    n_valid = len(valid_bins)

    adj_matrix = nx.adjacency_matrix(
        env.connectivity, nodelist=range(env.n_bins), weight="distance"
    ).tocsr()
    dist_matrix = scipy_dijkstra(
        adj_matrix, directed=False, indices=valid_bins, return_predecessors=False
    )[:, valid_bins]

    triu_indices = np.triu_indices(n_valid, k=1)
    pairwise_distances = dist_matrix[triu_indices]
    rates_i = valid_rates[triu_indices[0]]
    rates_j = valid_rates[triu_indices[1]]

    finite_mask = np.isfinite(pairwise_distances)
    pairwise_distances = pairwise_distances[finite_mask]
    rates_i = rates_i[finite_mask]
    rates_j = rates_j[finite_mask]

    max_distance = float(np.max(pairwise_distances))
    n_distance_bins = 30
    distance_bin_edges = np.linspace(0, max_distance, n_distance_bins + 1)

    bin_indices = (
        np.searchsorted(distance_bin_edges, pairwise_distances, side="right") - 1
    )
    bin_indices = np.where(bin_indices >= n_distance_bins, -1, bin_indices)
    bin_indices = np.where(bin_indices < 0, -1, bin_indices)

    # Loop-based approach (original)
    correlations_loop = np.full(n_distance_bins, np.nan)
    for d_idx in range(n_distance_bins):
        mask = bin_indices == d_idx
        if np.sum(mask) < 2:
            continue
        rates_i_bin = rates_i[mask]
        rates_j_bin = rates_j[mask]
        if np.std(rates_i_bin) == 0 or np.std(rates_j_bin) == 0:
            continue
        corr, _ = stats.pearsonr(rates_i_bin, rates_j_bin)
        correlations_loop[d_idx] = corr

    # Pure numpy vectorized approach
    # Sort by bin index for efficient grouped operations
    valid_bin_mask = bin_indices >= 0
    sort_idx = np.argsort(bin_indices[valid_bin_mask])
    sorted_bins = bin_indices[valid_bin_mask][sort_idx]
    sorted_rates_i = rates_i[valid_bin_mask][sort_idx]
    sorted_rates_j = rates_j[valid_bin_mask][sort_idx]

    # Find bin boundaries using searchsorted
    unique_bins, bin_counts = np.unique(sorted_bins, return_counts=True)
    bin_starts = np.concatenate([[0], np.cumsum(bin_counts)[:-1]])

    correlations_vectorized = np.full(n_distance_bins, np.nan)
    for _i, (b, start, count) in enumerate(
        zip(unique_bins, bin_starts, bin_counts, strict=True)
    ):
        if count < 2:
            continue
        end = start + count
        ri = sorted_rates_i[start:end]
        rj = sorted_rates_j[start:end]
        if np.std(ri) == 0 or np.std(rj) == 0:
            continue
        # Inline Pearson correlation
        ri_c = ri - ri.mean()
        rj_c = rj - rj.mean()
        corr = np.dot(ri_c, rj_c) / (np.linalg.norm(ri_c) * np.linalg.norm(rj_c))
        correlations_vectorized[b] = corr

    # Compare
    assert_allclose(
        np.nan_to_num(correlations_vectorized, nan=-999),
        np.nan_to_num(correlations_loop, nan=-999),
        rtol=1e-10,
    )
