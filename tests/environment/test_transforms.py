"""Tests for environment transform operations.

Covers:
- Rebinning environments with different bin sizes
- Subsetting environments by bin selection
- Edge cases and validation
- Property-based tests with Hypothesis

Test Plan Reference: TEST_PLAN2.md Priority 1.1
Target Coverage: 4% → 90% for src/neurospatial/environment/transforms.py
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.regions import Regions


class TestRebin:
    """Tests for Environment.rebin() method.

    Tests geometry-only coarsening operations that create new environments
    with reduced resolution.
    """

    def test_rebin_double_size(self, medium_2d_env: Environment) -> None:
        """Test rebinning with doubled bin size (factor=2).

        Note: rebin() marks all bins as active in the coarsened grid,
        so if the original environment has sparse active bins, the
        rebinned environment may have MORE bins than the original.
        """
        # Get original grid shape
        original_shape = medium_2d_env.layout.grid_shape
        original_grid_total = original_shape[0] * original_shape[1]

        # Rebin by factor 2
        rebinned = medium_2d_env.rebin(factor=2)

        # Should preserve dimensionality
        assert rebinned.n_dims == medium_2d_env.n_dims

        # Should preserve extent (approximately)
        for dim in range(medium_2d_env.n_dims):
            assert_allclose(
                rebinned.dimension_ranges[dim],
                medium_2d_env.dimension_ranges[dim],
                rtol=0.15,  # Allow up to 15% difference due to truncation
            )

        # Grid shape should be approximately halved
        rebinned_shape = rebinned.layout.grid_shape
        assert rebinned_shape[0] <= (original_shape[0] + 1) // 2
        assert rebinned_shape[1] <= (original_shape[1] + 1) // 2

        # Total grid size should be roughly 1/4 of original (for 2D with factor=2)
        rebinned_grid_total = rebinned_shape[0] * rebinned_shape[1]
        assert rebinned_grid_total <= original_grid_total // 4 + 50  # Allow some slack

    def test_rebin_half_size_not_supported(self, small_2d_env: Environment) -> None:
        """Test that rebinning to finer resolution is not supported.

        rebin() only supports coarsening (factor >= 1), not refinement.
        To get finer resolution, create a new environment with smaller bin_size.
        """
        # Attempting factor < 1 should raise ValueError
        with pytest.raises((ValueError, TypeError)):
            small_2d_env.rebin(factor=0.5)  # type: ignore

    def test_rebin_preserves_connectivity_structure(
        self, small_2d_env: Environment
    ) -> None:
        """Test that rebinning preserves graph connectivity structure."""
        rebinned = small_2d_env.rebin(factor=2)

        # All bins should be connected (no isolated components)
        assert nx.is_connected(rebinned.connectivity)

        # All nodes should have at least one neighbor (except isolated nodes)
        for node in rebinned.connectivity.nodes():
            degree = rebinned.connectivity.degree(node)
            assert degree > 0, f"Node {node} has degree 0 (isolated)"

    def test_rebin_invalid_bin_size_negative(self, medium_2d_env: Environment) -> None:
        """Test that negative factor raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            medium_2d_env.rebin(factor=-1)

    def test_rebin_invalid_bin_size_zero(self, medium_2d_env: Environment) -> None:
        """Test that zero factor raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            medium_2d_env.rebin(factor=0)

    def test_rebin_invalid_bin_size_too_large(self, small_2d_env: Environment) -> None:
        """Test that factor larger than grid shape raises ValueError."""
        grid_shape = small_2d_env.layout.grid_shape

        # Factor larger than smallest dimension
        too_large_factor = max(grid_shape) + 10

        with pytest.raises(ValueError, match="too large for grid shape"):
            small_2d_env.rebin(factor=too_large_factor)

    @pytest.mark.parametrize("factor", [1, 2, 3, 5])
    def test_rebin_various_factors(
        self, medium_2d_env: Environment, factor: int
    ) -> None:
        """Test rebinning with various integer factors."""
        rebinned = medium_2d_env.rebin(factor=factor)

        assert rebinned.n_bins > 0
        assert rebinned.n_dims == medium_2d_env.n_dims
        assert nx.is_connected(rebinned.connectivity)

        # Grid shape should be approximately divided by factor
        original_shape = medium_2d_env.layout.grid_shape
        rebinned_shape = rebinned.layout.grid_shape

        for orig_size, new_size in zip(original_shape, rebinned_shape, strict=False):
            # New size should be original // factor (or close due to truncation)
            expected_size = orig_size // factor
            assert new_size <= expected_size + 1
            assert new_size >= expected_size - 1

    def test_rebin_anisotropic_factor_tuple(self, medium_2d_env: Environment) -> None:
        """Test rebinning with different factors per dimension."""
        # Get original shape
        original_shape = medium_2d_env.layout.grid_shape

        # Use different factors for each dimension
        factor = (2, 3)

        rebinned = medium_2d_env.rebin(factor=factor)

        assert rebinned.n_dims == 2
        assert rebinned.n_bins > 0

        # Grid shape should reflect anisotropic coarsening
        rebinned_shape = rebinned.layout.grid_shape
        assert rebinned_shape[0] <= (original_shape[0] // 2) + 1
        assert rebinned_shape[1] <= (original_shape[1] // 3) + 1

    def test_rebin_preserves_metadata(self, medium_2d_env: Environment) -> None:
        """Test that rebinning preserves units and frame metadata."""
        # Create isolated copy to avoid mutating session-scoped fixture
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.units = "cm"
        env_copy.frame = "arena1"

        rebinned = env_copy.rebin(factor=2)

        assert rebinned.units == "cm"
        assert rebinned.frame == "arena1"

    def test_rebin_all_bins_active(self, medium_2d_env: Environment) -> None:
        """Test that all bins in rebinned environment are active."""
        rebinned = medium_2d_env.rebin(factor=2)

        # All bins should be active (even if original had inactive regions)
        assert rebinned.active_mask is not None
        assert np.all(rebinned.active_mask)

    def test_rebin_non_divisible_warns(self, small_2d_env: Environment) -> None:
        """Test that non-divisible grid dimensions trigger warning."""
        grid_shape = small_2d_env.layout.grid_shape

        # Choose factor that doesn't evenly divide grid shape
        # Find a factor that creates remainder
        factor = 3
        if grid_shape[0] % factor == 0 and grid_shape[1] % factor == 0:
            factor = 4  # Try different factor

        # Should warn about truncation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_2d_env.rebin(factor=factor)

            # Check that warning was issued (if grid not evenly divisible)
            if grid_shape[0] % factor != 0 or grid_shape[1] % factor != 0:
                assert len(w) > 0
                assert "not evenly divisible" in str(w[0].message)

    def test_rebin_only_regular_grid(self, graph_env: Environment) -> None:
        """Test that rebin only works for RegularGridLayout."""
        # graph_env is a GraphLayout, not RegularGridLayout
        with pytest.raises(
            NotImplementedError, match="only supported for RegularGridLayout"
        ):
            graph_env.rebin(factor=2)

    def test_rebin_factor_wrong_dimensions(self, medium_2d_env: Environment) -> None:
        """Test that factor tuple with wrong number of dimensions raises error."""
        # 2D environment, but provide 3D factor
        with pytest.raises(
            ValueError, match=r"factor has .* elements but environment has"
        ):
            medium_2d_env.rebin(factor=(2, 2, 2))

    def test_rebin_factor_non_integer(self, medium_2d_env: Environment) -> None:
        """Test that non-integer factor raises error."""
        with pytest.raises((ValueError, TypeError)):
            medium_2d_env.rebin(factor=2.5)  # type: ignore

    def test_rebin_updates_name(self, medium_2d_env: Environment) -> None:
        """Test that rebinned environment has updated name."""
        # Set original name
        medium_2d_env.name = "OriginalEnv"

        rebinned = medium_2d_env.rebin(factor=2)

        assert "rebinned" in rebinned.name.lower()
        assert "OriginalEnv" in rebinned.name


class TestSubset:
    """Tests for Environment.subset() method.

    Tests spatial subsetting operations that extract regions of environments.
    """

    def test_subset_by_bin_ids(self, medium_2d_env: Environment) -> None:
        """Test subsetting by explicit bin IDs."""
        # Select first 10 bins
        bin_ids = np.array([True] * 10 + [False] * (medium_2d_env.n_bins - 10))

        subset_env = medium_2d_env.subset(bins=bin_ids)

        assert subset_env.n_bins == 10
        assert subset_env.n_dims == medium_2d_env.n_dims

        # Bin centers should match (in new order)
        assert_allclose(subset_env.bin_centers, medium_2d_env.bin_centers[:10])

    def test_subset_preserves_connectivity(self, medium_2d_env: Environment) -> None:
        """Test that subset preserves connectivity between selected bins."""
        # Select a spatially contiguous region (center bins)
        center = medium_2d_env.bin_centers.mean(axis=0)
        distances = np.linalg.norm(medium_2d_env.bin_centers - center, axis=1)
        bin_ids = distances < 10.0

        subset_env = medium_2d_env.subset(bins=bin_ids)

        # Should have connectivity
        assert subset_env.connectivity.number_of_edges() > 0

        # Check that connectivity matches original for selected bins
        original_bin_indices = np.where(bin_ids)[0]
        for new_id, old_id in enumerate(original_bin_indices):
            # Get neighbors in original graph
            orig_neighbors = set(medium_2d_env.connectivity.neighbors(old_id))

            # Get neighbors in subset (should be subset of original neighbors)
            subset_neighbors = set(subset_env.connectivity.neighbors(new_id))

            # Map subset neighbors back to original indices
            subset_neighbors_old = {original_bin_indices[n] for n in subset_neighbors}

            # Subset neighbors should be a subset of original neighbors
            assert subset_neighbors_old.issubset(orig_neighbors)

    def test_subset_empty_raises_error(self, medium_2d_env: Environment) -> None:
        """Test that empty subset raises ValueError."""
        # All False mask
        bin_ids = np.zeros(medium_2d_env.n_bins, dtype=bool)

        with pytest.raises(ValueError, match="No bins selected"):
            medium_2d_env.subset(bins=bin_ids)

    def test_subset_wrong_shape_raises_error(self, medium_2d_env: Environment) -> None:
        """Test that mask with wrong shape raises error."""
        # Wrong length
        bin_ids = np.ones(10, dtype=bool)

        with pytest.raises(ValueError, match="must have shape"):
            medium_2d_env.subset(bins=bin_ids)

    def test_subset_wrong_dtype_raises_error(self, medium_2d_env: Environment) -> None:
        """Test that mask with wrong dtype raises error."""
        # Integer array instead of bool
        bin_ids = np.arange(medium_2d_env.n_bins)

        with pytest.raises(ValueError, match="must be boolean array"):
            medium_2d_env.subset(bins=bin_ids)  # type: ignore

    def test_subset_by_polygon(self, medium_2d_env: Environment) -> None:
        """Test subsetting by Shapely polygon."""
        # Create a box polygon in center of environment
        center = medium_2d_env.bin_centers.mean(axis=0)
        half_size = 10.0
        poly = box(
            center[0] - half_size,
            center[1] - half_size,
            center[0] + half_size,
            center[1] + half_size,
        )

        subset_env = medium_2d_env.subset(polygon=poly)

        assert subset_env.n_bins > 0
        assert subset_env.n_bins < medium_2d_env.n_bins

        # All bin centers should be inside or near polygon
        from shapely.geometry import Point

        for i in range(subset_env.n_bins):
            point = subset_env.bin_centers[i]
            # Check containment (allow boundary points)
            pt = Point(point[0], point[1])
            assert (
                poly.contains(pt)
                or poly.touches(pt)
                or poly.boundary.distance(pt) < 0.1
            )

    def test_subset_by_region_names(self, medium_2d_env: Environment) -> None:
        """Test subsetting by named regions."""
        # Make a copy to avoid mutating session-scoped fixture
        env = medium_2d_env.copy()

        # Add a region to the copy
        center = env.bin_centers.mean(axis=0)
        half_size = 10.0
        region_poly = box(
            center[0] - half_size,
            center[1] - half_size,
            center[0] + half_size,
            center[1] + half_size,
        )

        env.regions.add("center_region", polygon=region_poly)

        # Subset by region name
        subset_env = env.subset(region_names=["center_region"])

        assert subset_env.n_bins > 0
        assert subset_env.n_bins < medium_2d_env.n_bins

    def test_subset_region_not_found_raises_error(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that non-existent region name raises error."""
        with pytest.raises(ValueError, match=r"Region .* not found"):
            medium_2d_env.subset(region_names=["nonexistent_region"])

    def test_subset_point_region_raises_error(self, medium_2d_env: Environment) -> None:
        """Test that point-type regions raise error."""
        # Make a copy to avoid mutating session-scoped fixture
        env = medium_2d_env.copy()

        # Add a point region to the copy
        env.regions.add("goal", point=np.array([0.0, 0.0]))

        with pytest.raises(ValueError, match="point-type region"):
            env.subset(region_names=["goal"])

    def test_subset_inverted_selection(self, medium_2d_env: Environment) -> None:
        """Test inverted selection (complement)."""
        # Select center bins
        center = medium_2d_env.bin_centers.mean(axis=0)
        distances = np.linalg.norm(medium_2d_env.bin_centers - center, axis=1)
        bin_ids = distances < 5.0

        # Count bins in non-inverted selection
        n_selected = np.sum(bin_ids)

        # Get inverted selection (everything except center)
        subset_env = medium_2d_env.subset(bins=bin_ids, invert=True)

        # Should have complementary number of bins
        assert subset_env.n_bins == medium_2d_env.n_bins - n_selected

    def test_subset_no_params_raises_error(self, medium_2d_env: Environment) -> None:
        """Test that no selection parameters raises error."""
        with pytest.raises(ValueError, match="Exactly one of"):
            medium_2d_env.subset()

    def test_subset_multiple_params_raises_error(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that multiple selection parameters raises error."""
        bin_ids = np.ones(medium_2d_env.n_bins, dtype=bool)
        poly = box(0, 0, 10, 10)

        with pytest.raises(ValueError, match="Exactly one of"):
            medium_2d_env.subset(bins=bin_ids, polygon=poly)

    def test_subset_preserves_metadata(self, medium_2d_env: Environment) -> None:
        """Test that subsetting preserves units and frame metadata."""
        # Create isolated copy to avoid mutating session-scoped fixture
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.units = "cm"
        env_copy.frame = "arena1"

        # Select subset
        bin_ids = np.array([True] * 10 + [False] * (env_copy.n_bins - 10))
        subset_env = env_copy.subset(bins=bin_ids)

        assert subset_env.units == "cm"
        assert subset_env.frame == "arena1"

    def test_subset_drops_regions(self, medium_2d_env: Environment) -> None:
        """Test that subsetting drops all regions."""
        # Create a fresh copy to avoid region conflicts
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.regions = Regions()  # Fresh regions
        env_copy.regions.add("goal", point=np.array([0.0, 0.0]))

        # Select subset
        bin_ids = np.array([True] * 10 + [False] * (env_copy.n_bins - 10))
        subset_env = env_copy.subset(bins=bin_ids)

        # Regions should be empty
        assert len(subset_env.regions) == 0

    def test_subset_renumbers_nodes(self, small_2d_env: Environment) -> None:
        """Test that subset renumbers node IDs to be contiguous."""
        # Select every other bin
        bin_ids = np.zeros(small_2d_env.n_bins, dtype=bool)
        bin_ids[::2] = True

        subset_env = small_2d_env.subset(bins=bin_ids)

        # Node IDs should be 0, 1, 2, ..., n-1
        node_ids = sorted(subset_env.connectivity.nodes())
        assert node_ids == list(range(subset_env.n_bins))

    def test_subset_polygon_3d_raises_error(self, simple_3d_env: Environment) -> None:
        """Test that polygon subsetting on 3D environment raises error."""
        poly = box(0, 0, 10, 10)

        with pytest.raises(ValueError, match="only works for 2D"):
            simple_3d_env.subset(polygon=poly)

    def test_subset_region_names_empty_raises_error(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that empty region_names list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            medium_2d_env.subset(region_names=[])

    def test_subset_region_names_wrong_type_raises_error(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that region_names with wrong type raises error."""
        with pytest.raises(ValueError, match="must be a list or tuple"):
            medium_2d_env.subset(region_names="not_a_list")  # type: ignore


class TestRebinProperties:
    """Property-based tests for rebin() using Hypothesis.

    Tests mathematical invariants that should hold for all valid inputs.
    """

    @given(factor=st.integers(min_value=1, max_value=5))
    @pytest.mark.hypothesis
    def test_rebin_preserves_dimension_count(
        self, small_2d_env: Environment, factor: int
    ) -> None:
        """Property: rebin preserves number of dimensions."""
        original_n_dims = small_2d_env.n_dims

        rebinned = small_2d_env.rebin(factor=factor)

        assert rebinned.n_dims == original_n_dims

    @given(factor=st.integers(min_value=1, max_value=5))
    @pytest.mark.hypothesis
    def test_rebin_reduces_grid_size(
        self, small_2d_env: Environment, factor: int
    ) -> None:
        """Property: rebin reduces grid shape by factor.

        Note: rebin() may increase bin count for sparse environments
        (where original has many inactive bins), because all bins in
        the rebinned grid are marked as active.
        """
        original_shape = small_2d_env.layout.grid_shape
        original_grid_total = original_shape[0] * original_shape[1]

        rebinned = small_2d_env.rebin(factor=factor)

        # Grid shape should be reduced by factor
        rebinned_shape = rebinned.layout.grid_shape
        rebinned_grid_total = rebinned_shape[0] * rebinned_shape[1]

        # Total grid cells should be reduced (roughly by factor^n_dims)
        expected_reduction_factor = factor**small_2d_env.n_dims
        assert (
            rebinned_grid_total <= original_grid_total // expected_reduction_factor + 10
        )

    @given(factor=st.integers(min_value=1, max_value=3))
    @pytest.mark.hypothesis
    def test_rebin_maintains_connectivity(
        self, small_2d_env: Environment, factor: int
    ) -> None:
        """Property: rebinned environment is always connected."""
        rebinned = small_2d_env.rebin(factor=factor)

        # Should be a connected graph
        assert nx.is_connected(rebinned.connectivity)

    @given(
        factor_x=st.integers(min_value=1, max_value=4),
        factor_y=st.integers(min_value=1, max_value=4),
    )
    @pytest.mark.hypothesis
    def test_rebin_anisotropic_preserves_dims(
        self, small_2d_env: Environment, factor_x: int, factor_y: int
    ) -> None:
        """Property: anisotropic rebin preserves dimensionality."""
        factor = (factor_x, factor_y)

        rebinned = small_2d_env.rebin(factor=factor)

        assert rebinned.n_dims == 2
        assert rebinned.n_bins > 0


class TestSubsetProperties:
    """Property-based tests for subset() using Hypothesis.

    Tests mathematical invariants that should hold for all valid inputs.
    """

    @given(n_select=st.integers(min_value=1, max_value=20))
    def test_subset_maintains_bin_count(
        self, medium_2d_env: Environment, n_select: int
    ) -> None:
        """Property: subset bin count matches selection count."""
        # Ensure we don't select more bins than available
        assume(n_select <= medium_2d_env.n_bins)

        # Select first n_select bins
        bin_ids = np.array(
            [True] * n_select + [False] * (medium_2d_env.n_bins - n_select)
        )

        subset_env = medium_2d_env.subset(bins=bin_ids)

        assert subset_env.n_bins == n_select

    @given(n_select=st.integers(min_value=1, max_value=20))
    def test_subset_preserves_dimensionality(
        self, medium_2d_env: Environment, n_select: int
    ) -> None:
        """Property: subset preserves number of dimensions."""
        assume(n_select <= medium_2d_env.n_bins)

        bin_ids = np.array(
            [True] * n_select + [False] * (medium_2d_env.n_bins - n_select)
        )

        subset_env = medium_2d_env.subset(bins=bin_ids)

        assert subset_env.n_dims == medium_2d_env.n_dims

    def test_subset_complement_covers_all_bins(self, small_2d_env: Environment) -> None:
        """Property: subset + inverted subset covers all bins (no overlap)."""
        # Select half of bins
        bin_ids = np.zeros(small_2d_env.n_bins, dtype=bool)
        bin_ids[: small_2d_env.n_bins // 2] = True

        # Get subset and its complement
        subset_env = small_2d_env.subset(bins=bin_ids)
        complement_env = small_2d_env.subset(bins=bin_ids, invert=True)

        # Together they should cover all bins
        assert subset_env.n_bins + complement_env.n_bins == small_2d_env.n_bins


class TestRebinIntegration:
    """Integration tests for rebin() with other Environment operations."""

    def test_rebin_then_smooth(self, medium_2d_env: Environment) -> None:
        """Test rebinning followed by field smoothing."""
        rng = np.random.default_rng(42)
        rebinned = medium_2d_env.rebin(factor=2)

        # Create a field on rebinned environment
        field = rng.random(rebinned.n_bins)

        # Smooth should work
        smoothed = rebinned.smooth(field, bandwidth=5.0)

        assert smoothed.shape == (rebinned.n_bins,)
        assert not np.any(np.isnan(smoothed))

    def test_rebin_preserves_bin_at_functionality(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that bin_at() works correctly after rebinning."""
        rebinned = medium_2d_env.rebin(factor=2)

        # Query bin_at with rebinned environment's own bin centers
        for i in range(min(5, rebinned.n_bins)):  # Test first 5 bins
            point = rebinned.bin_centers[i]
            bin_idx = rebinned.bin_at(point[np.newaxis, :])

            # Should map to itself (or very close due to discretization)
            assert 0 <= bin_idx[0] < rebinned.n_bins


class TestSubsetIntegration:
    """Integration tests for subset() with other Environment operations."""

    def test_subset_then_neighbors(self, medium_2d_env: Environment) -> None:
        """Test neighbor queries work correctly after subsetting."""
        # Select center region
        center = medium_2d_env.bin_centers.mean(axis=0)
        distances = np.linalg.norm(medium_2d_env.bin_centers - center, axis=1)
        bin_ids = distances < 15.0

        subset_env = medium_2d_env.subset(bins=bin_ids)

        # Query neighbors for each bin (should work without errors)
        for bin_id in range(min(10, subset_env.n_bins)):
            neighbors = subset_env.neighbors(bin_id)
            assert isinstance(neighbors, list)

            # All neighbor IDs should be valid
            for neighbor_id in neighbors:
                assert 0 <= neighbor_id < subset_env.n_bins

    def test_subset_preserves_shortest_path(self, small_2d_env: Environment) -> None:
        """Test shortest path queries work after subsetting."""
        # Select a connected region (center bins)
        center = small_2d_env.bin_centers.mean(axis=0)
        distances = np.linalg.norm(small_2d_env.bin_centers - center, axis=1)
        bin_ids = distances < 10.0

        subset_env = small_2d_env.subset(bins=bin_ids)

        # Should be able to compute paths (if connected)
        if nx.is_connected(subset_env.connectivity):
            # path_between expects bin indices (integers), not bin centers
            path = subset_env.path_between(0, subset_env.n_bins - 1)
            assert isinstance(path, list)
            assert len(path) > 0


class TestApplyTransform:
    """Tests for Environment.apply_transform() method.

    Tests affine transformation application to environments, supporting both
    2D (Affine2D) and N-D (AffineND) transforms.
    """

    def test_apply_transform_identity_2d(self, medium_2d_env: Environment) -> None:
        """Test applying identity transform leaves environment unchanged."""
        from neurospatial.transforms import Affine2D

        # Identity transform (no change)
        transform = Affine2D(np.eye(3))

        transformed = medium_2d_env.apply_transform(transform)

        # Bin centers should be identical
        assert_allclose(transformed.bin_centers, medium_2d_env.bin_centers, atol=1e-10)
        assert transformed.n_bins == medium_2d_env.n_bins
        assert transformed.n_dims == medium_2d_env.n_dims

    def test_apply_transform_translation_2d(self, medium_2d_env: Environment) -> None:
        """Test applying 2D translation transform."""
        from neurospatial.transforms import translate

        # Translate by (10, 20)
        transform = translate(10, 20)

        transformed = medium_2d_env.apply_transform(transform)

        # Bin centers should be translated
        expected_centers = medium_2d_env.bin_centers + np.array([10, 20])
        assert_allclose(transformed.bin_centers, expected_centers, atol=1e-10)

    def test_apply_transform_rotation_2d(self, medium_2d_env: Environment) -> None:
        """Test applying 2D rotation transform."""
        from neurospatial.transforms import Affine2D

        # 45-degree rotation
        angle = np.pi / 4
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        transform = Affine2D(R)

        transformed = medium_2d_env.apply_transform(transform)

        # Should preserve number of bins and connectivity
        assert transformed.n_bins == medium_2d_env.n_bins
        assert (
            transformed.connectivity.number_of_edges()
            == medium_2d_env.connectivity.number_of_edges()
        )

        # Distances should be preserved under rotation
        orig_dists = [
            medium_2d_env.connectivity.edges[u, v]["distance"]
            for u, v in medium_2d_env.connectivity.edges()
        ]
        new_dists = [
            transformed.connectivity.edges[u, v]["distance"]
            for u, v in transformed.connectivity.edges()
        ]
        assert_allclose(sorted(new_dists), sorted(orig_dists), atol=1e-10)

    def test_apply_transform_scaling_2d(self, small_2d_env: Environment) -> None:
        """Test applying 2D scaling transform."""
        from neurospatial.transforms import scale_2d

        # Scale by factor of 2
        transform = scale_2d(2.0)

        transformed = small_2d_env.apply_transform(transform)

        # Bin centers should be scaled
        expected_centers = small_2d_env.bin_centers * 2.0
        assert_allclose(transformed.bin_centers, expected_centers, atol=1e-10)

    def test_apply_transform_composition_2d(self, small_2d_env: Environment) -> None:
        """Test applying composed transforms (rotation @ translation)."""
        from neurospatial.transforms import Affine2D, translate

        # Compose: translate then rotate
        rotation_angle = np.pi / 6
        R = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        rotation = Affine2D(R)
        translation = translate(5, 10)

        # Apply composed transform (rotation @ translation applies translation first)
        composed = rotation @ translation
        transformed = small_2d_env.apply_transform(composed)

        assert transformed.n_bins == small_2d_env.n_bins
        assert transformed.n_dims == small_2d_env.n_dims

    def test_apply_transform_affine_nd_2d(self, medium_2d_env: Environment) -> None:
        """Test applying AffineND transform to 2D environment."""
        from neurospatial.transforms import translate_3d

        # Can't use 3D transform on 2D environment
        transform_3d = translate_3d(10, 20, 30)

        with pytest.raises(ValueError, match=r"dimensionality.*does not match"):
            medium_2d_env.apply_transform(transform_3d)

    def test_apply_transform_affine_nd_3d(self, simple_3d_env: Environment) -> None:
        """Test applying AffineND transform to 3D environment."""
        from neurospatial.transforms import translate_3d

        # 3D translation
        transform = translate_3d(10, 20, 30)

        transformed = simple_3d_env.apply_transform(transform)

        # Bin centers should be translated
        expected_centers = simple_3d_env.bin_centers + np.array([10, 20, 30])
        assert_allclose(transformed.bin_centers, expected_centers, atol=1e-10)
        assert transformed.n_dims == 3

    def test_apply_transform_preserves_connectivity(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that transformation preserves connectivity structure."""
        from neurospatial.transforms import translate

        transform = translate(5, 5)
        transformed = medium_2d_env.apply_transform(transform)

        # Should preserve number of nodes and edges
        assert transformed.connectivity.number_of_nodes() == medium_2d_env.n_bins
        assert (
            transformed.connectivity.number_of_edges()
            == medium_2d_env.connectivity.number_of_edges()
        )

        # Node IDs should remain the same
        assert set(transformed.connectivity.nodes()) == set(
            medium_2d_env.connectivity.nodes()
        )

    def test_apply_transform_updates_node_positions(
        self, small_2d_env: Environment
    ) -> None:
        """Test that node 'pos' attributes are updated."""
        from neurospatial.transforms import translate

        transform = translate(10, 20)
        transformed = small_2d_env.apply_transform(transform)

        # Check that node positions are updated
        for node_id in transformed.connectivity.nodes():
            old_pos = np.array(small_2d_env.connectivity.nodes[node_id]["pos"])
            new_pos = np.array(transformed.connectivity.nodes[node_id]["pos"])

            expected_pos = old_pos + np.array([10, 20])
            assert_allclose(new_pos, expected_pos, atol=1e-10)

    def test_apply_transform_updates_edge_attributes(
        self, small_2d_env: Environment
    ) -> None:
        """Test that edge attributes (distance, vector, angle_2d) are updated."""
        from neurospatial.transforms import translate

        transform = translate(5, 5)
        transformed = small_2d_env.apply_transform(transform)

        # Check one edge
        edges_list = list(transformed.connectivity.edges())
        if edges_list:
            u, v = edges_list[0]

            # Get edge data
            edge_data = transformed.connectivity.edges[u, v]

            # Should have required attributes
            assert "distance" in edge_data
            assert "vector" in edge_data

            # For 2D, should have angle_2d
            if transformed.n_dims == 2:
                assert "angle_2d" in edge_data

    def test_apply_transform_functional_not_inplace(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that apply_transform returns new environment (functional)."""
        from neurospatial.transforms import translate

        transform = translate(10, 20)

        # Store original bin centers
        original_centers = medium_2d_env.bin_centers.copy()

        # Apply transform
        transformed = medium_2d_env.apply_transform(transform)

        # Original should be unchanged
        assert_allclose(medium_2d_env.bin_centers, original_centers)

        # Transformed should be different
        assert not np.allclose(transformed.bin_centers, original_centers)

        # They should be different objects
        assert transformed is not medium_2d_env

    def test_apply_transform_with_custom_name(self, small_2d_env: Environment) -> None:
        """Test applying transform with custom name."""
        from neurospatial.transforms import translate

        transform = translate(5, 5)

        transformed = small_2d_env.apply_transform(transform, name="rotated_env")

        assert transformed.name == "rotated_env"

    def test_apply_transform_default_name(self, medium_2d_env: Environment) -> None:
        """Test that default name appends '_transformed'."""
        from neurospatial.transforms import translate

        # Set original name
        medium_2d_env.name = "OriginalEnv"

        transform = translate(5, 5)
        transformed = medium_2d_env.apply_transform(transform)

        assert "transformed" in transformed.name.lower()

    def test_apply_transform_preserves_units(self, medium_2d_env: Environment) -> None:
        """Test that units metadata is preserved."""
        # Create isolated copy to avoid mutating fixture
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.units = "cm"

        from neurospatial.transforms import translate

        transform = translate(5, 5)
        transformed = env_copy.apply_transform(transform)

        assert transformed.units == "cm"

    def test_apply_transform_updates_frame(self, medium_2d_env: Environment) -> None:
        """Test that frame is updated with '_transformed' suffix."""
        # Create isolated copy
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.frame = "session1"

        from neurospatial.transforms import translate

        transform = translate(5, 5)
        transformed = env_copy.apply_transform(transform)

        assert "transformed" in transformed.frame

    def test_apply_transform_with_regions(self, medium_2d_env: Environment) -> None:
        """Test that regions are transformed along with environment."""
        # Create isolated copy
        import copy

        env_copy = copy.copy(medium_2d_env)
        env_copy.regions = Regions()  # Fresh regions

        # Add a point region
        env_copy.regions.add("goal", point=np.array([10.0, 20.0]))

        from neurospatial.transforms import translate

        # Translate by (5, 10)
        transform = translate(5, 10)
        transformed = env_copy.apply_transform(transform)

        # Region should exist and be transformed
        assert "goal" in transformed.regions

        # Point should be translated
        region_data = transformed.regions["goal"].data
        expected_point = np.array([15.0, 30.0])
        assert_allclose(region_data, expected_point, atol=1e-10)

    def test_apply_transform_with_polygon_region(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that polygon regions are transformed."""
        # Create isolated copy
        import copy

        from shapely.geometry import box

        env_copy = copy.copy(medium_2d_env)
        env_copy.regions = Regions()  # Fresh regions

        # Add a polygon region
        poly = box(0, 0, 10, 10)
        env_copy.regions.add("area", polygon=poly)

        from neurospatial.transforms import translate

        # Translate by (5, 5)
        transform = translate(5, 5)
        transformed = env_copy.apply_transform(transform)

        # Region should exist
        assert "area" in transformed.regions

        # Polygon should be transformed
        transformed_poly = transformed.regions["area"].data
        assert transformed_poly is not None

        # Check that bounds are translated
        orig_bounds = poly.bounds  # (minx, miny, maxx, maxy)
        new_bounds = transformed_poly.bounds

        assert_allclose(
            new_bounds,
            [
                orig_bounds[0] + 5,
                orig_bounds[1] + 5,
                orig_bounds[2] + 5,
                orig_bounds[3] + 5,
            ],
            atol=1e-10,
        )

    def test_apply_transform_dimension_mismatch_raises_error(
        self, medium_2d_env: Environment
    ) -> None:
        """Test that dimension mismatch raises ValueError."""
        from neurospatial.transforms import translate_3d

        # Try to apply 3D transform to 2D environment
        transform_3d = translate_3d(10, 20, 30)

        with pytest.raises(ValueError, match=r"dimensionality.*does not match"):
            medium_2d_env.apply_transform(transform_3d)

    def test_apply_transform_unfitted_env_raises_error(
        self, small_2d_env: Environment
    ) -> None:
        """Test that applying transform to unfitted environment raises error."""
        from neurospatial.transforms import translate

        # Create a fitted environment and then mark it as unfitted
        # (simulates an environment that hasn't been properly initialized)
        env = small_2d_env
        env._is_fitted = False

        transform = translate(5, 5)

        with pytest.raises(RuntimeError, match="must be fitted"):
            env.apply_transform(transform)
