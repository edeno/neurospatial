"""Tests for Environment.subset() method.

This module tests the subset/crop functionality for creating new environments
from a selection of bins.
"""

import numpy as np
import pytest
from environment_helpers import publish_environment
from shapely.geometry import box

from neurospatial import Environment, Regions


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
        sub_env = env.subset(mask=mask)

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
        region_builder = Regions(env.regions.values())
        region_builder.add("left_half", polygon=box(0, 0, 50, 100))
        env = env.with_regions(region_builder)

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

        sub_env = env.subset(mask=mask)

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

        sub_env = env.subset(mask=mask)

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

        sub_env = env.subset(mask=mask)

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
            env.subset(mask=mask)

    def test_subset_all_bins_returns_equivalent_environment(self):
        """Selecting all bins returns equivalent environment."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select all bins
        mask = np.ones(env.n_bins, dtype=bool)
        sub_env = env.subset(mask=mask)

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

        sub_env = env.subset(mask=mask)

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
            sub_env = env.subset(mask=mask)

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
            env.subset(mask=mask, polygon=poly)

    def test_subset_mask_wrong_shape_raises_error(self):
        """Mask with wrong shape raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Wrong length
        mask = np.ones(env.n_bins + 10, dtype=bool)

        with pytest.raises(ValueError, match=r"mask.*shape.*n_bins"):
            env.subset(mask=mask)

    def test_subset_mask_wrong_dtype_raises_error(self):
        """Mask with wrong dtype raises ValueError."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Integer mask instead of boolean
        mask = np.arange(env.n_bins)

        with pytest.raises(ValueError, match=r"mask= is a boolean array[\s\S]*Fix:"):
            env.subset(mask=mask)

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

    def test_subset_point_region_keeps_the_bin_containing_it(self):
        """A point region selects the bin that contains the point."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)
        region_builder = Regions(env.regions.values())
        region_builder.add("goal", point=[50, 50])
        env = env.with_regions(region_builder)

        sub = env.subset(region_names=["goal"])

        (goal_bin,) = env.bins_in_region("goal")
        assert sub.n_bins == 1
        np.testing.assert_array_equal(sub.bin_centers, env.bin_centers[[goal_bin]])


class TestSubsetInvert:
    """Test invert parameter."""

    def test_subset_invert_selects_complement(self):
        """invert=True selects complement of mask."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Select left half
        mask = env.bin_centers[:, 0] < 50

        # Normal subset
        sub_env = env.subset(mask=mask)
        # Inverted subset
        inv_env = env.subset(mask=mask, invert=True)

        # Should be complementary
        assert sub_env.n_bins + inv_env.n_bins == env.n_bins

    def test_subset_invert_with_region(self):
        """invert works with region_names."""
        rng = np.random.default_rng(42)
        env = Environment.from_samples(rng.uniform(0, 100, (1000, 2)), bin_size=10.0)

        # Add region
        region_builder = Regions(env.regions.values())
        region_builder.add("center", polygon=box(40, 40, 60, 60))
        env = env.with_regions(region_builder)

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
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="session1")

        # Subset
        mask = np.ones(env.n_bins // 2, dtype=bool)
        mask = np.pad(mask, (0, env.n_bins - len(mask)), constant_values=False)
        sub_env = env.subset(mask=mask)

        # Metadata preserved
        assert sub_env.units == "cm"
        assert sub_env.frame == "session1"


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
        sub_env = env.subset(mask=mask)

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
        sub_env = env.subset(mask=mask)

        # Create field and smooth
        field = rng.random(sub_env.n_bins)
        smoothed = sub_env.smooth(field, bandwidth=5.0, mode="average")

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

        sub_env = env.subset(mask=mask)

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

        sub_env = env.subset(mask=mask)

        # Should have roughly half the bins
        assert sub_env.n_bins < original_n_bins
        assert sub_env.n_bins > 0

        # All bin centers should be in left half
        assert np.all(sub_env.bin_centers[:, 0] < 50)

        # Y coordinates should span full range
        y_min, y_max = sub_env.bin_centers[:, 1].min(), sub_env.bin_centers[:, 1].max()
        assert y_min < 20  # Near bottom
        assert y_max > 80  # Near top


class TestSubsetRoundTripsThroughToFileFromFile:
    """A subset environment survives the to_file/from_file round trip.

    Saved files store the bins directly, so a subset loads without
    rebuilding a layout, and keeps its parent's layout label.
    """

    def _build_grid_env(self):
        from neurospatial import Environment

        x = np.linspace(0.0, 20.0, 11)
        xx, yy = np.meshgrid(x, x)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        return Environment.from_samples(positions, bin_size=2.0, name="parent")

    def test_subset_to_file_from_file_round_trip(self, tmp_path):
        """Acceptance test: subset -> to_file -> from_file."""
        from neurospatial.io.files import from_file, to_file

        env = self._build_grid_env()
        # Boolean-mask subset selecting three specific active bins.
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[[0, 5, 10]] = True
        env_sub = env.subset(mask=mask)

        path = tmp_path / "sub"
        to_file(env_sub, path)
        loaded = from_file(path)

        assert loaded.n_bins == env_sub.n_bins
        # bin_centers may differ in row order if the subset's row-major
        # order differs from the original active mask's flattened order;
        # compare as sets of points instead.
        loaded_centers_sorted = np.array(sorted(loaded.bin_centers.tolist()))
        sub_centers_sorted = np.array(sorted(env_sub.bin_centers.tolist()))
        np.testing.assert_array_equal(loaded_centers_sorted, sub_centers_sorted)

    def test_subset_polygon_round_trip(self, tmp_path):
        """A polygon-cropped subset also round-trips."""
        from shapely.geometry import box

        from neurospatial.io.files import from_file, to_file

        env = self._build_grid_env()
        crop = box(2.0, 2.0, 12.0, 12.0)
        env_sub = env.subset(polygon=crop)

        path = tmp_path / "sub_poly"
        to_file(env_sub, path)
        loaded = from_file(path)

        assert loaded.n_bins == env_sub.n_bins

    def test_subset_metadata_survives_round_trip(self, tmp_path):
        """units / frame must persist through subset + to_file + from_file."""
        from neurospatial.io.files import from_file, to_file

        env = self._build_grid_env()
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="session1")

        mask = np.zeros(env.n_bins, dtype=bool)
        mask[[0, 5, 10]] = True
        env_sub = env.subset(mask=mask)
        # Subset preserves units/frame on the env it returns.
        assert env_sub.units == "cm"
        assert env_sub.frame == "session1"

        path = tmp_path / "sub_meta"
        to_file(env_sub, path)
        loaded = from_file(path)
        assert loaded.units == "cm"
        assert loaded.frame == "session1"

    def test_subset_keeps_the_grid_parent_layout_label(self):
        """A grid subset reports its parent's label and records the subset."""
        env = self._build_grid_env()
        mask = np.zeros(env.n_bins, dtype=bool)
        mask[[0, 5, 10]] = True
        env_sub = env.subset(mask=mask)
        assert env_sub.layout_type == env.layout_type == "RegularGrid"
        assert env_sub.derivations == ("subset",)


class TestSubsetKeepsParentConnectivity:
    """A grid subset keeps exactly the parent's edges among the selected bins."""

    @pytest.mark.parametrize("connect_diagonal", [True, False])
    def test_built_grid_subset_matches_a_fresh_mask_grid(self, connect_diagonal):
        """For a factory-built grid, mask adjacency is the parent's graph."""
        env = Environment.from_samples(
            np.random.default_rng(0).uniform(0, 30, (2000, 2)),
            bin_size=3.0,
            connect_diagonal_neighbors=connect_diagonal,
        )
        sub = env.subset(mask=np.arange(env.n_bins) % 3 != 0)
        fresh = Environment.from_grid_mask(
            sub.active_mask,
            grid_edges=sub.grid_edges,
            connect_diagonal_neighbors=connect_diagonal,
        )
        assert list(sub.connectivity.edges(data=True)) == list(
            fresh.connectivity.edges(data=True)
        )

    def test_wall_in_saved_graph_survives_subset_and_round_trip(self):
        """Edges a saved graph omits are not restored from mask adjacency."""
        from neurospatial.io import from_dict, to_dict
        from neurospatial.layout.engines.masked_grid import MaskedGridLayout

        layout = MaskedGridLayout()
        layout.build(
            active_mask=np.ones((6, 6), dtype=bool),
            grid_edges=(np.arange(7.0), np.arange(7.0)),
        )
        centers = np.asarray(layout.bin_centers)

        def crosses_wall(u: int, v: int) -> bool:
            (x0, y0), (x1, y1) = centers[u], centers[v]
            return (x0 - 3.0) * (x1 - 3.0) < 0 and min(y0, y1) > 2.0

        native = layout.binning().connections
        kept = [not crosses_wall(u, v) for u, v in native.pairs.tolist()]
        layout._set_neighbor_pairs(native.pairs[kept], native.pair_distances[kept])
        walled = publish_environment(layout=layout, layout_type_used="MaskedGrid")
        walled = from_dict(to_dict(walled))
        assert not any(crosses_wall(u, v) for u, v in walled.connectivity.edges())

        sub = walled.subset(mask=np.ones(walled.n_bins, dtype=bool))
        reloaded = from_dict(to_dict(sub))

        for result in (sub, reloaded):
            assert sorted(result.connectivity.edges()) == sorted(
                walled.connectivity.edges()
            )
            np.testing.assert_allclose(
                result.distance_to(bins=[0]), walled.distance_to(bins=[0])
            )
