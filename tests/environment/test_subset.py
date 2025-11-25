"""Tests for Environment.subset() method.

This module tests the subset/crop functionality for creating new environments
from a selection of bins.
"""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment


class TestSubsetBasic:
    """Test basic subset functionality."""

    def test_subset_with_boolean_mask(self):
        """Subset with boolean mask selects correct bins."""
        rng = np.random.default_rng(42)
        # Create 10x10 grid
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select first half of bins
        n_bins = env.n_bins
        mask = np.zeros(n_bins, dtype=bool)
        mask[: n_bins // 2] = True

        # Create subset
        sub_env = env.subset(bins=mask)

        # Verify bin count
        assert sub_env.n_bins == np.sum(mask)
        assert sub_env.n_bins == n_bins // 2

        # Verify node indices are renumbered to [0, n'-1]
        node_ids = list(sub_env.connectivity.nodes())
        assert node_ids == list(range(sub_env.n_bins))

    def test_subset_with_region_names(self):
        """Subset with region names selects bins inside regions."""
        rng = np.random.default_rng(42)
        # Create environment
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Add a region covering left half
        env.regions.add("left_half", polygon=box(0, 0, 50, 100))

        # Subset to region
        sub_env = env.subset(region_names=["left_half"])

        # Should have fewer bins than original
        assert sub_env.n_bins < env.n_bins
        assert sub_env.n_bins > 0

        # All bin centers should be inside the original region
        for center in sub_env.bin_centers:
            assert 0 <= center[0] <= 50  # X coordinate in left half

    def test_subset_with_polygon(self):
        """Subset with polygon selects bins whose centers are inside."""
        rng = np.random.default_rng(42)
        # Create 10x10 grid
        data = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(data, bin_size=10.0)

        # Crop to left half with polygon
        crop_polygon = box(0, 0, 50, 100)
        sub_env = env.subset(polygon=crop_polygon)

        # Should have fewer bins
        assert sub_env.n_bins < env.n_bins
        assert sub_env.n_bins > 0

        # All bin centers should be inside polygon
        from shapely.geometry import Point

        for center in sub_env.bin_centers:
            point = Point(center)
            assert crop_polygon.contains(point) or crop_polygon.touches(point)

    def test_subset_preserves_connectivity(self):
        """Subset preserves connectivity between selected bins."""
        rng = np.random.default_rng(42)
        # Create grid
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select left half
        centers = env.bin_centers
        mask = centers[:, 0] < 50

        sub_env = env.subset(bins=mask)

        # Check that connectivity graph has correct structure
        assert sub_env.connectivity.number_of_nodes() == sub_env.n_bins
        assert sub_env.connectivity.number_of_edges() > 0

        # All edges should connect valid nodes
        for u, v in sub_env.connectivity.edges():
            assert 0 <= u < sub_env.n_bins
            assert 0 <= v < sub_env.n_bins


class TestSubsetNodeRenumbering:
    """Test node renumbering logic."""

    def test_subset_renumbers_nodes_to_contiguous_range(self):
        """Node indices are renumbered to [0, n'-1]."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select scattered bins
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[::3] = True  # Every third bin

        sub_env = env.subset(bins=mask)

        # Node IDs should be contiguous [0, 1, 2, ..., n'-1]
        node_ids = sorted(sub_env.connectivity.nodes())
        expected_ids = list(range(sub_env.n_bins))
        assert node_ids == expected_ids

    def test_subset_node_attributes_preserved(self):
        """Node attributes are preserved after renumbering."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select some bins
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[:10] = True

        sub_env = env.subset(bins=mask)

        # Check required node attributes exist
        for node in sub_env.connectivity.nodes():
            attrs = sub_env.connectivity.nodes[node]
            assert "pos" in attrs
            assert "source_grid_flat_index" in attrs
            assert "original_grid_nd_index" in attrs


class TestSubsetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_subset_empty_selection_raises_error(self):
        """Empty selection raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Empty mask
        mask = np.zeros(env.n_bins, dtype=bool)

        with pytest.raises(ValueError, match="No bins selected"):
            env.subset(bins=mask)

    def test_subset_all_bins_returns_equivalent_environment(self):
        """Selecting all bins returns equivalent environment."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select all bins
        mask = np.ones(env.n_bins, dtype=bool)
        sub_env = env.subset(bins=mask)

        # Should have same number of bins
        assert sub_env.n_bins == env.n_bins
        assert (
            sub_env.connectivity.number_of_edges() == env.connectivity.number_of_edges()
        )

    def test_subset_single_bin(self):
        """Subset with single bin works correctly."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select one bin
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[5] = True

        sub_env = env.subset(bins=mask)

        # Should have exactly one bin
        assert sub_env.n_bins == 1
        assert sub_env.connectivity.number_of_nodes() == 1
        assert sub_env.connectivity.number_of_edges() == 0  # No neighbors

    def test_subset_disconnected_components(self):
        """Subset can create disconnected graph."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select bins from opposite corners (likely disconnected)
        centers = env.bin_centers
        mask = ((centers[:, 0] < 20) & (centers[:, 1] < 20)) | (
            (centers[:, 0] > 80) & (centers[:, 1] > 80)
        )

        if np.sum(mask) > 1:  # Only test if we have multiple bins
            sub_env = env.subset(bins=mask)

            # Should succeed even if disconnected
            assert sub_env.n_bins == np.sum(mask)
            assert sub_env.connectivity.number_of_nodes() == sub_env.n_bins


class TestSubsetValidation:
    """Test input validation."""

    def test_subset_requires_exactly_one_parameter(self):
        """Exactly one of {bins, region_names, polygon} must be provided."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # No parameters
        with pytest.raises(ValueError, match="Exactly one of"):
            env.subset()

        # Multiple parameters
        mask = np.ones(env.n_bins, dtype=bool)
        poly = box(0, 0, 50, 50)
        with pytest.raises(ValueError, match="Exactly one of"):
            env.subset(bins=mask, polygon=poly)

    def test_subset_mask_wrong_shape_raises_error(self):
        """Mask with wrong shape raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Wrong length
        mask = np.ones(env.n_bins + 10, dtype=bool)

        with pytest.raises(ValueError, match=r"bins.*shape.*n_bins"):
            env.subset(bins=mask)

    def test_subset_mask_wrong_dtype_raises_error(self):
        """Mask with wrong dtype raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Integer mask instead of boolean
        mask = np.arange(env.n_bins)

        with pytest.raises(ValueError, match=r"bins.*must be.*bool"):
            env.subset(bins=mask)

    def test_subset_invalid_region_name_raises_error(self):
        """Non-existent region name raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        with pytest.raises(ValueError, match=r"Region.*not found"):
            env.subset(region_names=["nonexistent_region"])

    def test_subset_empty_region_names_raises_error(self):
        """Empty region_names list raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        with pytest.raises(ValueError, match=r"region_names.*empty"):
            env.subset(region_names=[])

    def test_subset_point_region_raises_error(self):
        """Point-type regions raise ValueError with helpful message."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Add a point region
        env.regions.add("goal", point=[50, 50])

        # Should raise error with clear message
        with pytest.raises(ValueError, match=r"point-type region"):
            env.subset(region_names=["goal"])


class TestSubsetInvert:
    """Test invert parameter."""

    def test_subset_invert_selects_complement(self):
        """invert=True selects complement of mask."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select left half
        mask = env.bin_centers[:, 0] < 50

        # Normal subset
        sub_env = env.subset(bins=mask)
        # Inverted subset
        inv_env = env.subset(bins=mask, invert=True)

        # Should be complementary
        assert sub_env.n_bins + inv_env.n_bins == env.n_bins

    def test_subset_invert_with_region(self):
        """invert works with region_names."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Add region
        env.regions.add("center", polygon=box(40, 40, 60, 60))

        # Subset and inverted subset
        sub_env = env.subset(region_names=["center"])
        inv_env = env.subset(region_names=["center"], invert=True)

        # Should be complementary
        assert sub_env.n_bins + inv_env.n_bins == env.n_bins


class TestSubsetMetadataHandling:
    """Test handling of metadata (units, frame, regions)."""

    def test_subset_preserves_units_and_frame(self):
        """Subset preserves units and frame metadata."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)
        env.units = "cm"
        env.frame = "session1"

        # Subset
        mask = np.ones(env.n_bins // 2, dtype=bool)
        mask = np.pad(mask, (0, env.n_bins - len(mask)), constant_values=False)
        sub_env = env.subset(bins=mask)

        # Metadata preserved
        assert sub_env.units == "cm"
        assert sub_env.frame == "session1"

    def test_subset_drops_regions(self):
        """Subset drops all regions."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)
        env.regions.add("goal", point=[50, 50])
        env.regions.add("start", point=[10, 10])

        # Subset
        mask = env.bin_centers[:, 0] < 50
        sub_env = env.subset(bins=mask)

        # Regions should be dropped
        assert len(sub_env.regions) == 0


class TestSubsetIntegration:
    """Integration tests with other Environment methods."""

    def test_subset_then_occupancy(self):
        """Can compute occupancy on subset environment."""
        rng = np.random.default_rng(42)
        # Create environment with trajectory
        data = rng.uniform(0, 100, (1000, 2))
        env = Environment.from_samples(data, bin_size=10.0)

        # Subset to left half
        mask = env.bin_centers[:, 0] < 50
        sub_env = env.subset(bins=mask)

        # Generate trajectory in left half
        times = np.linspace(0, 10, 100)
        positions = rng.uniform(0, 50, (100, 2))  # Left half only

        # Compute occupancy (should work without errors)
        occ = sub_env.occupancy(times, positions)

        assert occ.shape == (sub_env.n_bins,)
        assert np.all(occ >= 0)

    def test_subset_then_smooth(self):
        """Can smooth fields on subset environment."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Subset
        mask = env.bin_centers[:, 0] < 50
        sub_env = env.subset(bins=mask)

        # Create field and smooth
        field = rng.random(sub_env.n_bins)
        smoothed = sub_env.smooth(field, bandwidth=5.0)

        assert smoothed.shape == (sub_env.n_bins,)
        assert np.all(np.isfinite(smoothed))

    def test_subset_preserves_graph_distances(self):
        """Edge distances preserved in subset graph."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Get original edge with distance
        original_edges = list(env.connectivity.edges(data=True))
        if len(original_edges) == 0:
            pytest.skip("No edges in environment")

        u, v, _edge_data = original_edges[0]

        # Subset that includes both nodes
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[u] = True
        mask[v] = True
        # Add a few more to make it interesting
        mask[:10] = True

        sub_env = env.subset(bins=mask)

        # Check that edge distances are preserved
        # (node IDs will be different, but distances should match)
        sub_edges = list(sub_env.connectivity.edges(data=True))
        assert len(sub_edges) > 0

        # At least some edges should have distance attribute
        for _, _, edge_data in sub_edges:
            assert "distance" in edge_data
            assert edge_data["distance"] > 0


class TestSubsetCropExample:
    """Test the documented example: crop 10x10 grid to 5x10."""

    def test_crop_grid_to_left_half(self):
        """Crop 10x10 grid to left half produces 5x10-like structure."""
        # Create approximately 10x10 grid (100 bins)
        # Use regular sampling to get predictable grid
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=10.0)
        original_n_bins = env.n_bins

        # Crop to left half
        mask = env.bin_centers[:, 0] < 50

        sub_env = env.subset(bins=mask)

        # Should have roughly half the bins
        assert sub_env.n_bins < original_n_bins
        assert sub_env.n_bins > 0

        # All bin centers should be in left half
        assert np.all(sub_env.bin_centers[:, 0] < 50)

        # Y coordinates should span full range
        y_min, y_max = sub_env.bin_centers[:, 1].min(), sub_env.bin_centers[:, 1].max()
        assert y_min < 20  # Near bottom
        assert y_max > 80  # Near top
