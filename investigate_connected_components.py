"""
Investigation: scipy.ndimage.label vs Graph-based Connected Components

This script compares two approaches for finding connected components in place
field detection:

1. **Fast path (scipy.ndimage.label)**: For grid-based environments
   - Uses scipy's optimized N-D connected component labeling
   - Only applicable when environment has grid_shape attribute
   - Expected ~5× speedup

2. **Fallback path (NetworkX or custom flood-fill)**: For non-grid environments
   - Uses graph traversal for arbitrary connectivity
   - Works for any environment type (1D tracks, irregular grids, etc.)

Goal: Determine strategy for Task 2.7 (Connected Components)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy import ndimage

from neurospatial import Environment

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Current Implementation (from place_fields.py)
# ============================================================================


def _extract_connected_component_current(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Current flood-fill implementation using graph connectivity."""
    component_set = {seed_idx}
    frontier = [seed_idx]

    while frontier:
        current = frontier.pop(0)
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


# ============================================================================
# Proposed Fast Path: scipy.ndimage.label (for grid environments)
# ============================================================================


def _extract_connected_component_scipy(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Fast path using scipy.ndimage.label for grid environments.

    Only applicable when env has grid_shape and active_mask.
    """
    if env.grid_shape is None or env.active_mask is None:
        raise ValueError("scipy path requires grid_shape and active_mask")

    # Reshape flat mask to N-D grid
    grid_mask = np.zeros(env.grid_shape, dtype=bool)
    grid_mask[env.active_mask] = mask

    # Label connected components in N-D grid
    # structure=None uses default connectivity (no diagonal neighbors for 2D)
    labeled, _n_components = ndimage.label(grid_mask, structure=None)

    # Find which component contains the seed
    # Convert seed_idx (flat) to grid coordinates
    seed_grid_coords = np.unravel_index(
        np.where(env.active_mask.ravel())[0][seed_idx], env.grid_shape
    )
    seed_label = labeled[seed_grid_coords]

    if seed_label == 0:
        # Seed not in any component (shouldn't happen if mask[seed_idx] is True)
        return np.array([seed_idx], dtype=np.int64)

    # Extract all bins in this component
    component_grid_mask = labeled == seed_label

    # Convert back to flat bin indices
    # Only include active bins
    component_flat_indices = np.where(
        component_grid_mask.ravel() & env.active_mask.ravel()
    )[0]

    # Map from grid flat indices to active bin indices
    active_to_grid = np.where(env.active_mask.ravel())[0]
    component_bins = np.searchsorted(active_to_grid, component_flat_indices)

    return np.array(sorted(component_bins), dtype=np.int64)


# ============================================================================
# Proposed Fallback Path: NetworkX connected_components (for non-grid)
# ============================================================================


def _extract_connected_component_networkx(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Fallback path using NetworkX for non-grid environments.

    Uses NetworkX's optimized connected components algorithm.
    """
    # Create subgraph with only masked nodes
    masked_nodes = np.where(mask)[0]
    subgraph = env.connectivity.subgraph(masked_nodes)

    # Find connected components
    for component in nx.connected_components(subgraph):
        if seed_idx in component:
            return np.array(sorted(component), dtype=np.int64)

    # Seed not in any component (isolated node)
    return np.array([seed_idx], dtype=np.int64)


# ============================================================================
# Testing Infrastructure
# ============================================================================


def test_correctness_grid():
    """Test that all three methods produce identical results on grid."""
    print("\n" + "=" * 70)
    print("TEST 1: Correctness on 2D Grid Environment")
    print("=" * 70)

    # Create 2D grid environment
    positions = np.random.rand(1000, 2) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    print(f"\nEnvironment: {env.n_bins} bins, grid_shape={env.grid_shape}")
    print(f"Active mask: {env.active_mask.sum()} active bins")

    # Create test mask (circular region in center)
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 20.0
    seed_idx = np.argmin(distances)  # Closest bin to center

    print(f"\nMask: {mask.sum()} bins above threshold")
    print(f"Seed: bin {seed_idx}")

    # Test all three methods
    result_current = _extract_connected_component_current(seed_idx, mask, env)
    result_networkx = _extract_connected_component_networkx(seed_idx, mask, env)

    try:
        result_scipy = _extract_connected_component_scipy(seed_idx, mask, env)
    except Exception as e:
        print(f"\n❌ scipy path failed: {e}")
        result_scipy = None

    # Compare results
    print("\nResults:")
    print(f"  Current (flood-fill):  {len(result_current)} bins")
    print(f"  NetworkX:              {len(result_networkx)} bins")
    if result_scipy is not None:
        print(f"  scipy.ndimage.label:   {len(result_scipy)} bins")

    # Check equality
    if np.array_equal(result_current, result_networkx):
        print("\n✅ Current == NetworkX")
    else:
        print("\n❌ Current != NetworkX")
        print(f"   Difference: {set(result_current) ^ set(result_networkx)}")

    if result_scipy is not None:
        if np.array_equal(result_current, result_scipy):
            print("✅ Current == scipy")
        else:
            print("❌ Current != scipy")
            print(f"   Difference: {set(result_current) ^ set(result_scipy)}")


def test_performance_grid():
    """Benchmark performance on large grid environment."""
    print("\n" + "=" * 70)
    print("TEST 2: Performance on Large 2D Grid")
    print("=" * 70)

    # Create large grid environment
    positions = np.random.rand(10000, 2) * 200
    env = Environment.from_samples(positions, bin_size=2.0)

    print(f"\nEnvironment: {env.n_bins} bins, grid_shape={env.grid_shape}")

    # Create test mask (large circular region)
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    mask = distances < 50.0
    seed_idx = np.argmin(distances)

    print(f"Mask: {mask.sum()} bins above threshold")

    # Benchmark current implementation
    n_trials = 10
    times_current = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _extract_connected_component_current(seed_idx, mask, env)
        times_current.append(time.perf_counter() - start)

    # Benchmark NetworkX
    times_networkx = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _extract_connected_component_networkx(seed_idx, mask, env)
        times_networkx.append(time.perf_counter() - start)

    # Benchmark scipy (if applicable)
    try:
        times_scipy = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _extract_connected_component_scipy(seed_idx, mask, env)
            times_scipy.append(time.perf_counter() - start)
    except Exception as e:
        print(f"\n❌ scipy path not applicable: {e}")
        times_scipy = None

    # Report results
    mean_current = np.mean(times_current) * 1000
    mean_networkx = np.mean(times_networkx) * 1000
    print(f"\nPerformance ({n_trials} trials):")
    print(
        f"  Current (flood-fill):  {mean_current:.3f} ms ± {np.std(times_current) * 1000:.3f} ms"
    )
    print(
        f"  NetworkX:              {mean_networkx:.3f} ms ± {np.std(times_networkx) * 1000:.3f} ms"
    )
    print(f"  Speedup (NetworkX):    {mean_current / mean_networkx:.2f}x")

    if times_scipy is not None:
        mean_scipy = np.mean(times_scipy) * 1000
        print(
            f"  scipy.ndimage.label:   {mean_scipy:.3f} ms ± {np.std(times_scipy) * 1000:.3f} ms"
        )
        print(f"  Speedup (scipy):       {mean_current / mean_scipy:.2f}x")


def test_non_grid_environment():
    """Test fallback path on non-grid environment (1D track)."""
    print("\n" + "=" * 70)
    print("TEST 3: Non-Grid Environment (1D Track)")
    print("=" * 70)

    try:
        from track_linearization import make_track_graph

        # Create 1D track graph
        position = np.linspace(0, 100, 100)[:, np.newaxis]
        track_graph = make_track_graph(position, 1.0)  # edge_spacing as positional arg

        env = Environment.from_graph(track_graph, edge_spacing=1.0)

        print(f"\nEnvironment: {env.n_bins} bins, 1D track")
        print(f"grid_shape: {env.grid_shape} (not a regular grid)")

        # Create test mask (middle section of track)
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[40:60] = True
        seed_idx = 50

        print(f"\nMask: {mask.sum()} bins above threshold")

        # Test methods
        result_current = _extract_connected_component_current(seed_idx, mask, env)
        result_networkx = _extract_connected_component_networkx(seed_idx, mask, env)

        print("\nResults:")
        print(f"  Current (flood-fill):  {len(result_current)} bins")
        print(f"  NetworkX:              {len(result_networkx)} bins")

        if np.array_equal(result_current, result_networkx):
            print("\n✅ Current == NetworkX")
        else:
            print("\n❌ Current != NetworkX")

        # scipy should NOT be applicable
        try:
            _extract_connected_component_scipy(seed_idx, mask, env)
            print("\n⚠️  scipy worked on 1D track (unexpected)")
        except (ValueError, Exception) as e:
            print(f"\n✅ scipy correctly rejected 1D track: {type(e).__name__}")

    except ImportError:
        print("\n⚠️  Skipping 1D track test (track-linearization not available)")


def test_detection_strategy():
    """Test strategy for detecting when to use scipy vs fallback."""
    print("\n" + "=" * 70)
    print("TEST 4: Detection Strategy")
    print("=" * 70)

    # Create different environment types
    envs = []

    # 2D grid
    positions_2d = np.random.rand(1000, 2) * 100
    env_2d = Environment.from_samples(positions_2d, bin_size=5.0)
    envs.append(("2D Grid", env_2d))

    # 3D grid
    positions_3d = np.random.rand(1000, 3) * 100
    env_3d = Environment.from_samples(positions_3d, bin_size=5.0)
    envs.append(("3D Grid", env_3d))

    print("\nDetection Strategy:")
    print("-" * 70)

    for name, env in envs:
        has_grid_shape = env.grid_shape is not None
        is_multidim_grid = (
            has_grid_shape and len(env.grid_shape) >= 2 and len(env.grid_shape) <= 3
        )
        has_active_mask = env.active_mask is not None

        print(f"\n{name}:")
        print(f"  grid_shape: {env.grid_shape}")
        print(f"  active_mask: {'Yes' if has_active_mask else 'No'}")
        print(f"  Use scipy.ndimage.label? {is_multidim_grid and has_active_mask}")

        if is_multidim_grid and has_active_mask:
            print("  → FAST PATH (scipy)")
        else:
            print("  → FALLBACK PATH (NetworkX or flood-fill)")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Connected Components Investigation (Task 2.7)")
    print("=" * 70)
    print("\nComparing:")
    print("  1. Current: Flood-fill with graph.neighbors()")
    print("  2. NetworkX: nx.connected_components()")
    print("  3. scipy: scipy.ndimage.label() (grid only)")

    # Run all tests
    test_detection_strategy()
    test_correctness_grid()
    test_performance_grid()
    # test_non_grid_environment()  # Skip due to track_linearization API incompatibility

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
Strategy for detect_place_fields() optimization:

1. **Check if scipy.ndimage.label is applicable**:
   - Condition: `env.grid_shape is not None and len(env.grid_shape) >= 2`
   - AND: `env.active_mask is not None`

2. **If applicable, use scipy.ndimage.label (FAST PATH)**:
   - Reshape flat mask to N-D grid using grid_shape
   - Use scipy.ndimage.label() for connected components
   - Convert labeled components back to flat bin indices
   - Expected ~2-10x speedup depending on grid size

3. **Otherwise, use NetworkX (FALLBACK PATH)**:
   - Replace current flood-fill with nx.connected_components()
   - Works for any graph structure (1D tracks, irregular grids, etc.)
   - Expected ~1.5-3x speedup over current flood-fill

4. **Implementation location**:
   - Add helper `_extract_connected_component_scipy()` to place_fields.py
   - Add helper `_extract_connected_component_networkx()` to place_fields.py
   - Modify `_extract_connected_component()` to route to appropriate helper

5. **Testing**:
   - Existing tests should pass unchanged (same results)
   - Add performance benchmarks with @pytest.mark.slow
   - Test on RegularGridLayout, GraphLayout, and irregular environments
""")
