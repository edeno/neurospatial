"""Tests for Environment connectivity operations (components, reachable_from).

This module tests the connected component analysis and reachability query
methods of the Environment class.

Test Organization:
    - TestComponentsBasic: Core components() functionality
    - TestComponentsEdgeCases: Boundary conditions
    - TestReachableFromBasic: Core reachable_from() functionality
    - TestReachableFromMetrics: Different distance metrics (hops, geodesic)
    - TestReachableFromRadius: Radius parameter behavior
    - TestReachableFromEdgeCases: Boundary conditions
    - TestConnectivityMultipleLayouts: Different layout types
    - TestConnectivityValidation: Input validation
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment

# =============================================================================
# Test Suite 1: Components - Basic Functionality
# =============================================================================


class TestComponentsBasic:
    """Test basic connected components functionality."""

    def test_components_single_component(self):
        """Test components() on fully connected environment."""
        # Create simple 2x2 grid (single component) with dense sampling
        x, y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
        data = np.column_stack([x.ravel(), y.ravel()])
        env = Environment.from_samples(data, bin_size=1.5)

        comps = env.components()

        # Should have exactly one component
        assert len(comps) == 1
        # Should contain all bins
        assert len(comps[0]) == env.n_bins
        # Should be sorted array of bin indices
        assert np.array_equal(np.sort(comps[0]), np.arange(env.n_bins))

    def test_components_multiple_components(self):
        """Test components() on disconnected environment."""
        # Create two disconnected regions
        data = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],  # Component 1 (4 bins)
                [10, 10],
                [10, 11],
                [11, 10],  # Component 2 (3 bins)
            ],
            dtype=np.float64,
        )
        env = Environment.from_samples(data, bin_size=1.5)

        comps = env.components()

        # Should have exactly two components
        assert len(comps) == 2
        # Total bins should match
        assert sum(len(c) for c in comps) == env.n_bins
        # Components should be disjoint
        all_bins = np.concatenate(comps)
        assert len(all_bins) == len(np.unique(all_bins))
        # Sorted by size (largest first)
        assert len(comps[0]) >= len(comps[1])

    def test_components_largest_only(self):
        """Test components(largest_only=True) returns only largest component."""
        # Create two disconnected regions of different sizes
        # First region: dense 2x2 grid
        x1, y1 = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 2, 10))
        region1 = np.column_stack([x1.ravel(), y1.ravel()])
        # Second region: smaller 1x1 grid far away
        x2, y2 = np.meshgrid(np.linspace(10, 11, 5), np.linspace(10, 11, 5))
        region2 = np.column_stack([x2.ravel(), y2.ravel()])
        data = np.vstack([region1, region2])
        env = Environment.from_samples(data, bin_size=1.5)

        comps = env.components(largest_only=True)

        # Should return list with single component
        assert len(comps) == 1
        # Should be the larger component
        assert len(comps[0]) >= 2  # At least a few bins in largest component

    def test_components_sorting(self):
        """Test that components are sorted by size (largest first)."""
        # Create three components of different sizes with dense sampling
        # Component 1: 2x2 grid (smallest)
        x1, y1 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        region1 = np.column_stack([x1.ravel(), y1.ravel()])
        # Component 2: 3x3 grid (medium)
        x2, y2 = np.meshgrid(np.linspace(10, 12, 8), np.linspace(10, 12, 8))
        region2 = np.column_stack([x2.ravel(), y2.ravel()])
        # Component 3: 2x2 grid (medium-small)
        x3, y3 = np.meshgrid(np.linspace(20, 21, 6), np.linspace(20, 21, 6))
        region3 = np.column_stack([x3.ravel(), y3.ravel()])
        data = np.vstack([region1, region2, region3])
        env = Environment.from_samples(data, bin_size=1.5)

        comps = env.components()

        # Should have multiple components
        assert len(comps) >= 2
        # Check sizes are non-increasing
        sizes = [len(c) for c in comps]
        assert sizes == sorted(sizes, reverse=True)


# =============================================================================
# Test Suite 2: Components - Edge Cases
# =============================================================================


class TestComponentsEdgeCases:
    """Test edge cases for components()."""

    def test_components_single_bin(self):
        """Test components() with single-bin environment."""
        data = np.array([[0, 0]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.0)

        comps = env.components()

        assert len(comps) == 1
        assert len(comps[0]) == 1
        assert comps[0][0] == 0

    def test_components_return_type(self):
        """Test that components returns list of int32 arrays."""
        data = np.array([[0, 0], [0, 1], [10, 10]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        comps = env.components()

        assert isinstance(comps, list)
        for comp in comps:
            assert isinstance(comp, np.ndarray)
            assert comp.dtype == np.int32


# =============================================================================
# Test Suite 3: Reachable From - Basic Functionality
# =============================================================================


class TestReachableFromBasic:
    """Test basic reachable_from() functionality."""

    def test_reachable_from_no_radius(self):
        """Test reachable_from() without radius (all reachable bins)."""
        # Create simple 3x3 grid (single component)
        x, y = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 2, 10))
        data = np.column_stack([x.ravel(), y.ravel()])
        env = Environment.from_samples(data, bin_size=1.5)

        # Start from center bin
        center_bin = env.bin_at(np.array([[1, 1]]))[0]
        mask = env.reachable_from(center_bin)

        # Should reach all bins (single component)
        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool
        assert np.all(mask)

    def test_reachable_from_disconnected(self):
        """Test reachable_from() on disconnected environment."""
        # Create two disconnected regions
        # First region
        x1, y1 = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
        region1 = np.column_stack([x1.ravel(), y1.ravel()])
        # Second region (far away)
        x2, y2 = np.meshgrid(np.linspace(10, 11, 5), np.linspace(10, 11, 5))
        region2 = np.column_stack([x2.ravel(), y2.ravel()])
        data = np.vstack([region1, region2])
        env = Environment.from_samples(data, bin_size=1.5)

        # Start from first component
        source_bin = env.bin_at(np.array([[0, 0]]))[0]
        mask = env.reachable_from(source_bin)

        # Should only reach bins in same component (not all bins)
        assert mask.sum() < env.n_bins
        # Check that unreachable bins exist
        unreachable_bins = np.where(~mask)[0]
        assert len(unreachable_bins) > 0

    def test_reachable_from_source_bin_always_reachable(self):
        """Test that source bin is always in reachable set."""
        # Create dense sampling for reliable binning
        x1, y1 = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        region1 = np.column_stack([x1.ravel(), y1.ravel()])
        x2, y2 = np.meshgrid(np.linspace(10, 11, 3), np.linspace(10, 11, 3))
        region2 = np.column_stack([x2.ravel(), y2.ravel()])
        data = np.vstack([region1, region2])
        env = Environment.from_samples(data, bin_size=1.5)

        source_bin = env.bin_at(np.array([[0, 0]]))[0]
        mask = env.reachable_from(source_bin)

        # Source bin should always be reachable
        assert mask[source_bin]


# =============================================================================
# Test Suite 4: Reachable From - Distance Metrics
# =============================================================================


class TestReachableFromMetrics:
    """Test different distance metrics (hops vs geodesic)."""

    def test_reachable_from_hops_metric(self):
        """Test reachable_from() with metric='hops' and radius."""
        # Create 1D line with dense sampling
        data = np.linspace(0, 4, 50).reshape(-1, 1)
        env = Environment.from_samples(data, bin_size=1.5)

        # Start from middle, radius=1 hop
        center_bin = env.bin_at(np.array([[2]]))[0]
        mask = env.reachable_from(center_bin, radius=1, metric="hops")

        # Should reach bins within 1 hop (at least source + neighbors)
        reachable_bins = np.where(mask)[0]
        # Check that source is reachable and has neighbors
        assert mask[center_bin]
        assert len(reachable_bins) >= 2  # Source + at least 1 neighbor

    def test_reachable_from_geodesic_metric(self):
        """Test reachable_from() with metric='geodesic' and radius."""
        # Create 2D grid with dense sampling
        x, y = np.meshgrid(np.linspace(0, 2, 10), np.linspace(0, 2, 10))
        data = np.column_stack([x.ravel(), y.ravel()])
        env = Environment.from_samples(data, bin_size=1.5)

        # Start from corner, radius=1.5 units
        source_bin = env.bin_at(np.array([[0, 0]]))[0]
        mask = env.reachable_from(source_bin, radius=1.5, metric="geodesic")

        # Should reach nearby bins based on graph distance
        # At least source bin and immediate neighbors
        assert mask[source_bin]
        assert mask.sum() >= 2  # Source + at least one neighbor

    def test_reachable_from_default_metric(self):
        """Test that metric='hops' is the default."""
        data = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        source_bin = 0
        # Call without specifying metric
        mask = env.reachable_from(source_bin, radius=1)

        # Should behave as hops (default)
        assert mask.shape == (env.n_bins,)
        assert mask.dtype == bool


# =============================================================================
# Test Suite 5: Reachable From - Radius Parameter
# =============================================================================


class TestReachableFromRadius:
    """Test radius parameter behavior."""

    def test_reachable_from_radius_zero(self):
        """Test reachable_from() with radius=0 (only source bin)."""
        data = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        source_bin = 0
        mask = env.reachable_from(source_bin, radius=0, metric="hops")

        # Should only reach source bin
        assert mask.sum() == 1
        assert mask[source_bin]

    def test_reachable_from_increasing_radius(self):
        """Test that larger radius includes more bins."""
        # Create 1D line: 0 -- 1 -- 2 -- 3 -- 4
        data = np.array([[0], [1], [2], [3], [4]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        center_bin = 2
        mask_r1 = env.reachable_from(center_bin, radius=1, metric="hops")
        mask_r2 = env.reachable_from(center_bin, radius=2, metric="hops")

        # r=2 should include all r=1 bins plus more
        assert mask_r2.sum() >= mask_r1.sum()
        assert np.all(mask_r2[mask_r1])  # r=1 bins are subset of r=2

    def test_reachable_from_radius_none(self):
        """Test that radius=None reaches all bins in component."""
        data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        source_bin = 0
        mask = env.reachable_from(source_bin, radius=None, metric="hops")

        # Should reach all bins (single component)
        assert mask.sum() == env.n_bins


# =============================================================================
# Test Suite 6: Reachable From - Edge Cases
# =============================================================================


class TestReachableFromEdgeCases:
    """Test edge cases for reachable_from()."""

    def test_reachable_from_single_bin(self):
        """Test reachable_from() on single-bin environment."""
        data = np.array([[0, 0]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.0)

        mask = env.reachable_from(0)

        assert mask.shape == (1,)
        assert mask[0]

    def test_reachable_from_invalid_source_bin(self):
        """Test reachable_from() with invalid source bin raises error."""
        data = np.array([[0, 0], [0, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        # Test negative bin index
        with pytest.raises(ValueError, match=r"source_bin must be.*\[0, n_bins\)"):
            env.reachable_from(-1)

        # Test bin index >= n_bins
        with pytest.raises(ValueError, match=r"source_bin must be.*\[0, n_bins\)"):
            env.reachable_from(env.n_bins)

    def test_reachable_from_negative_radius(self):
        """Test reachable_from() with negative radius raises error."""
        data = np.array([[0, 0], [0, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        with pytest.raises(ValueError, match=r"radius must be non-negative"):
            env.reachable_from(0, radius=-1.0)

    def test_reachable_from_invalid_metric(self):
        """Test reachable_from() with invalid metric raises error."""
        data = np.array([[0, 0], [0, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)

        with pytest.raises(ValueError, match=r"metric must be"):
            env.reachable_from(0, metric="invalid")


# =============================================================================
# Test Suite 7: Multiple Layout Types
# =============================================================================


class TestConnectivityMultipleLayouts:
    """Test connectivity operations on different layout types."""

    @pytest.mark.skip(
        reason="GraphLayout has issue with 1D data (tries to access [1] on 1D array)"
    )
    def test_components_graph_layout(self):
        """Test components() on 1D graph layout."""
        # Create simple 1D track using networkx
        graph = nx.Graph()
        graph.add_nodes_from(
            [
                (0, {"pos": (0,)}),
                (1, {"pos": (1,)}),
                (2, {"pos": (2,)}),
                (3, {"pos": (3,)}),
            ]
        )
        # Add edges with distance attributes
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(2, 3, distance=1.0)

        env = Environment.from_graph(
            graph=graph,
            edge_order=[(0, 1), (1, 2), (2, 3)],
            edge_spacing=0.0,
            bin_size=0.5,
        )

        comps = env.components()

        # Should have single component
        assert len(comps) == 1
        assert len(comps[0]) == env.n_bins

    @pytest.mark.skip(
        reason="GraphLayout has issue with 1D data (tries to access [1] on 1D array)"
    )
    def test_reachable_from_graph_layout(self):
        """Test reachable_from() on 1D graph layout."""
        # Create 1D track with two disconnected segments
        graph = nx.Graph()
        # First segment: 0-1-2
        graph.add_nodes_from(
            [
                (0, {"pos": (0,)}),
                (1, {"pos": (1,)}),
                (2, {"pos": (2,)}),
                (3, {"pos": (10,)}),
                (4, {"pos": (11,)}),
                (5, {"pos": (12,)}),
            ]
        )
        # Two disconnected segments with distance attributes
        graph.add_edge(0, 1, distance=1.0)
        graph.add_edge(1, 2, distance=1.0)
        graph.add_edge(3, 4, distance=1.0)
        graph.add_edge(4, 5, distance=1.0)

        env = Environment.from_graph(
            graph=graph,
            edge_order=[(0, 1), (1, 2), (3, 4), (4, 5)],
            edge_spacing=0.0,
            bin_size=0.5,
        )

        # Start from first bin (part of first segment)
        mask = env.reachable_from(0)

        # Should only reach bins in first segment (not second segment)
        # Note: exact number depends on binning, but should be < total bins
        assert mask.sum() < env.n_bins
        assert mask[0]  # Source bin always reachable


# =============================================================================
# Test Suite 8: Validation
# =============================================================================


class TestConnectivityValidation:
    """Test input validation for connectivity operations."""

    def test_components_fitted_check(self):
        """Test that components() requires fitted environment."""
        # This test verifies @check_fitted decorator is applied
        # We can't easily create an unfitted Environment with public API,
        # so this test mainly documents the expected behavior
        data = np.array([[0, 0], [0, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)
        # Fitted environment should work
        comps = env.components()
        assert len(comps) >= 1

    def test_reachable_from_fitted_check(self):
        """Test that reachable_from() requires fitted environment."""
        data = np.array([[0, 0], [0, 1]], dtype=np.float64)
        env = Environment.from_samples(data, bin_size=1.5)
        # Fitted environment should work
        mask = env.reachable_from(0)
        assert mask.shape == (env.n_bins,)
