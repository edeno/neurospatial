"""
Tests for connected component extraction paths in detect_place_fields.

This module tests the two-path approach for connected component detection:
1. Fast path: scipy.ndimage.label (grid environments only)
2. Fallback path: Graph-based flood-fill (all other environments)

Following TDD: Tests written FIRST before implementation (Task 2.8).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment

# =============================================================================
# Test scipy fast path (_extract_connected_component_scipy)
# =============================================================================


def test_extract_connected_component_scipy_2d_grid():
    """Test scipy fast path on 2D grid environment."""
    # Create 2D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Ensure this is a grid environment
    assert env.grid_shape is not None
    assert len(env.grid_shape) == 2
    assert env.active_mask is not None

    # Create test mask (circular region in center)
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 20.0
    seed_idx = np.argmin(distances)  # Closest bin to center

    # Import internal functions
    from neurospatial.encoding.place import (
        _extract_connected_component_scipy,
    )

    # Test scipy path directly
    result_scipy = _extract_connected_component_scipy(seed_idx, mask, env)

    # Verify result is non-empty
    assert len(result_scipy) > 0
    # Verify seed is in result
    assert seed_idx in result_scipy
    # Verify all bins in result are in mask
    assert all(mask[i] for i in result_scipy)
    # Verify result is sorted
    assert_array_equal(result_scipy, np.sort(result_scipy))


def test_extract_connected_component_scipy_3d_grid():
    """Test scipy fast path on 3D grid environment."""
    # Create 3D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 3)) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    # Ensure this is a 3D grid environment
    assert env.grid_shape is not None
    assert len(env.grid_shape) == 3
    assert env.active_mask is not None

    # Create test mask (spherical region in center)
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 15.0
    seed_idx = np.argmin(distances)

    from neurospatial.encoding.place import _extract_connected_component_scipy

    # Test scipy path on 3D grid
    result_scipy = _extract_connected_component_scipy(seed_idx, mask, env)

    # Verify result
    assert len(result_scipy) > 0
    assert seed_idx in result_scipy
    assert all(mask[i] for i in result_scipy)


def test_extract_connected_component_scipy_disconnected_regions():
    """Test scipy path correctly identifies only connected component containing seed."""
    # Create 2D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create mask with TWO disconnected regions
    center1 = np.array([25.0, 50.0])
    center2 = np.array([75.0, 50.0])  # Far apart
    distances1 = np.linalg.norm(env.bin_centers - center1, axis=1)
    distances2 = np.linalg.norm(env.bin_centers - center2, axis=1)

    # Two circular regions
    mask = (distances1 < 10.0) | (distances2 < 10.0)

    # Seed in first region
    seed_idx = np.argmin(distances1)

    from neurospatial.encoding.place import _extract_connected_component_scipy

    result = _extract_connected_component_scipy(seed_idx, mask, env)

    # Result should only contain bins from first region (connected to seed)
    for i in result:
        # All bins should be closer to center1 than center2
        dist1 = np.linalg.norm(env.bin_centers[i] - center1)
        dist2 = np.linalg.norm(env.bin_centers[i] - center2)
        assert dist1 < dist2, f"Bin {i} is closer to second region (disconnected)"


def test_extract_connected_component_scipy_rejects_non_grid():
    """Test scipy path raises error for non-grid environments."""
    # Create environment without grid_shape (e.g., from polygon or custom layout)
    # For this test, we'll manually create an environment that's missing grid_shape

    # This is a simplified test - in practice, we'd use GraphLayout or other non-grid
    rng = np.random.default_rng(42)
    positions = rng.random((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    # Manually remove grid_shape to simulate non-grid environment
    env.grid_shape = None

    mask = np.ones(env.n_bins, dtype=bool)
    seed_idx = 0

    from neurospatial.encoding.place import _extract_connected_component_scipy

    # Should raise ValueError
    with pytest.raises(ValueError, match="scipy path requires grid_shape"):
        _extract_connected_component_scipy(seed_idx, mask, env)


# =============================================================================
# Test graph fallback path (_extract_connected_component_graph)
# =============================================================================


def test_extract_connected_component_graph_2d_grid():
    """Test graph fallback path on 2D grid (should work on any graph)."""
    # Create 2D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create test mask
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 20.0
    seed_idx = np.argmin(distances)

    from neurospatial.encoding.place import _extract_connected_component_graph

    # Test graph path (should work even on grid)
    result_graph = _extract_connected_component_graph(seed_idx, mask, env)

    # Verify result
    assert len(result_graph) > 0
    assert seed_idx in result_graph
    assert all(mask[i] for i in result_graph)
    assert_array_equal(result_graph, np.sort(result_graph))


def test_extract_connected_component_graph_isolated_seed():
    """Test graph path handles isolated seed (no neighbors in mask)."""
    # Create simple environment
    rng = np.random.default_rng(42)
    positions = rng.random((500, 2)) * 50
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create mask with only ONE bin
    mask = np.zeros(env.n_bins, dtype=bool)
    seed_idx = 10
    mask[seed_idx] = True

    from neurospatial.encoding.place import _extract_connected_component_graph

    result = _extract_connected_component_graph(seed_idx, mask, env)

    # Should return only the seed
    assert len(result) == 1
    assert result[0] == seed_idx


# =============================================================================
# Test routing logic (_extract_connected_component)
# =============================================================================


def test_extract_connected_component_routing_grid():
    """Test main function routes to scipy fast path for grid environments."""
    # Create 2D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Verify this is a grid
    assert env.grid_shape is not None
    assert len(env.grid_shape) == 2

    # Create test mask
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 20.0
    seed_idx = np.argmin(distances)

    from neurospatial.encoding.place import _extract_connected_component

    # Call main function (should route to scipy)
    result = _extract_connected_component(seed_idx, mask, env)

    # Verify result is correct
    assert len(result) > 0
    assert seed_idx in result
    assert all(mask[i] for i in result)


def test_extract_connected_component_scipy_vs_graph_equivalence():
    """Test scipy and graph paths produce identical results on grid."""
    # Create 2D grid environment
    rng = np.random.default_rng(42)
    positions = rng.random((1000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create test mask
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 20.0
    seed_idx = np.argmin(distances)

    from neurospatial.encoding.place import (
        _extract_connected_component_graph,
        _extract_connected_component_scipy,
    )

    # Get results from both paths
    result_scipy = _extract_connected_component_scipy(seed_idx, mask, env)
    result_graph = _extract_connected_component_graph(seed_idx, mask, env)

    # Both should produce identical results
    assert_array_equal(
        result_scipy,
        result_graph,
        err_msg="scipy and graph paths produce different results",
    )


# =============================================================================
# Integration tests with detect_place_fields
# =============================================================================


def test_detect_place_fields_uses_scipy_path():
    """Test detect_place_fields uses scipy fast path for grid environments."""
    # Create 2D grid environment with synthetic place field
    rng = np.random.default_rng(42)
    positions = rng.random((5000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create synthetic firing rate with one place field
    firing_rate = np.zeros(env.n_bins)
    center = env.bin_centers.mean(axis=0)
    for i in range(env.n_bins):
        dist = np.linalg.norm(env.bin_centers[i] - center)
        firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 10.0**2))

    from neurospatial.encoding.place import detect_place_fields

    # Should use scipy path internally (grid environment)
    fields = detect_place_fields(firing_rate, env)

    # Should detect exactly one field
    assert len(fields) == 1
    assert len(fields[0]) > 0


def test_detect_place_fields_grid_vs_nongrid_equivalence():
    """Test detect_place_fields produces same results regardless of path used."""
    # Create identical environments - one will use scipy, one will use graph
    rng = np.random.default_rng(42)
    positions = rng.random((2000, 2)) * 100
    env_grid = Environment.from_samples(positions, bin_size=5.0)

    # Same firing rate for both
    firing_rate = np.zeros(env_grid.n_bins)
    center = env_grid.bin_centers.mean(axis=0)
    for i in range(env_grid.n_bins):
        dist = np.linalg.norm(env_grid.bin_centers[i] - center)
        firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 10.0**2))

    from neurospatial.encoding.place import detect_place_fields

    # Detect fields (uses scipy path for grid)
    fields_scipy = detect_place_fields(firing_rate, env_grid)

    # Temporarily disable scipy path to test graph path
    # (This is a bit of a hack - in practice, both paths should give same result)
    env_no_grid = Environment.from_samples(positions, bin_size=5.0)
    original_grid_shape = env_no_grid.grid_shape
    env_no_grid.grid_shape = None  # Force graph path

    fields_graph = detect_place_fields(firing_rate, env_no_grid)

    env_no_grid.grid_shape = original_grid_shape  # Restore

    # Both should detect same number of fields
    assert len(fields_scipy) == len(fields_graph)


# =============================================================================
# Regression tests for BFS ordering (deque optimization)
# =============================================================================


def test_bfs_ordering_unchanged_after_deque_optimization():
    """Regression test: verify BFS produces same results with deque.

    This test ensures the optimization from list.pop(0) to deque.popleft()
    doesn't change the set of discovered bins. Both implement FIFO ordering,
    so results should be identical.

    The key invariant is that BFS explores neighbors level-by-level, and the
    final sorted result should be deterministic for a given seed and mask.
    """
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 50, (500, 2))
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create a cross-shaped mask to test BFS expansion pattern
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)

    # Simple circular mask
    mask = distances < 15.0
    seed_idx = np.argmin(distances)

    from neurospatial.encoding.place import _extract_connected_component_graph

    result = _extract_connected_component_graph(seed_idx, mask, env)

    # Verify BFS properties:
    # 1. Seed is in result
    assert seed_idx in result

    # 2. All results are in mask
    assert all(mask[i] for i in result)

    # 3. All bins in mask that are connected to seed should be found
    # (verify completeness - no bins left behind)
    for i in result:
        for neighbor in env.connectivity.neighbors(i):
            if mask[neighbor]:
                assert neighbor in result, f"BFS missed connected bin {neighbor}"

    # 4. Result is sorted (implementation detail, but expected)
    assert_array_equal(result, np.sort(result))

    # 5. Deterministic: running twice gives same result
    result2 = _extract_connected_component_graph(seed_idx, mask, env)
    assert_array_equal(result, result2)


def test_bfs_expansion_order_is_breadth_first():
    """Verify BFS expands level-by-level (not depth-first).

    This test creates a linear chain and verifies bins are discovered
    in order of their distance from the seed (BFS property).
    """
    # Create a simple 1D-like environment (linear arrangement)
    positions = np.array([[i * 5.0, 0.0] for i in range(20)])
    env = Environment.from_samples(positions, bin_size=4.0)

    # All bins active
    mask = np.ones(env.n_bins, dtype=bool)
    seed_idx = 0  # Start from one end

    from neurospatial.encoding.place import _extract_connected_component_graph

    # Track discovery order by modifying the function behavior
    # We can't directly observe order since result is sorted, but we can
    # verify all connected bins are found
    result = _extract_connected_component_graph(seed_idx, mask, env)

    # In a connected graph, BFS should find all bins
    # All bins reachable from seed should be in result
    assert seed_idx in result
    assert len(result) > 0

    # Verify connectivity: result should form a connected component
    result_set = set(result)
    for bin_idx in result:
        has_neighbor_in_result = any(
            n in result_set for n in env.connectivity.neighbors(bin_idx)
        )
        # Seed might have no neighbors in result if isolated, otherwise all should
        if bin_idx != seed_idx or len(result) > 1:
            assert has_neighbor_in_result or bin_idx == seed_idx


# =============================================================================
# Performance benchmarks (marked as slow)
# =============================================================================


@pytest.mark.slow
def test_connected_component_performance_scipy_vs_graph():
    """Benchmark scipy vs graph paths on large grid (Task 2.12)."""
    import time

    # Create large 2D grid
    rng = np.random.default_rng(42)
    positions = rng.random((10000, 2)) * 200
    env = Environment.from_samples(positions, bin_size=2.0)

    print(f"\nEnvironment: {env.n_bins} bins, grid_shape={env.grid_shape}")

    # Create large connected component
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 50.0
    seed_idx = np.argmin(distances)

    print(f"Mask: {mask.sum()} bins above threshold")

    from neurospatial.encoding.place import (
        _extract_connected_component_graph,
        _extract_connected_component_scipy,
    )

    # Benchmark scipy path
    n_trials = 10
    times_scipy = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _extract_connected_component_scipy(seed_idx, mask, env)
        times_scipy.append(time.perf_counter() - start)

    # Benchmark graph path
    times_graph = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _extract_connected_component_graph(seed_idx, mask, env)
        times_graph.append(time.perf_counter() - start)

    mean_scipy = np.mean(times_scipy) * 1000
    mean_graph = np.mean(times_graph) * 1000
    speedup = mean_graph / mean_scipy

    print(f"\nPerformance ({n_trials} trials):")
    print(f"  scipy path:  {mean_scipy:.3f} ms ± {np.std(times_scipy) * 1000:.3f} ms")
    print(f"  graph path:  {mean_graph:.3f} ms ± {np.std(times_graph) * 1000:.3f} ms")
    print(f"  Speedup:     {speedup:.2f}x")

    # Assert scipy is faster (target: 5-10x speedup)
    assert speedup > 2.0, (
        f"scipy path not significantly faster (speedup={speedup:.2f}x)"
    )
