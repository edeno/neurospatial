"""Tests for new CompositeEnvironment methods added in P1.1.

Tests for:
- bins_in_region()
- mask_for_region()
- shortest_path()
- info()
- save()/load()
"""

import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment


# Fixtures for test environments
@pytest.fixture
def two_simple_2d_envs():
    """Create two simple 2D environments for testing."""
    # Environment 1: left side
    data1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    env1 = Environment.from_samples(data1, bin_size=1.5)
    env1.name = "env1"

    # Environment 2: right side
    data2 = np.array([[10, 0], [11, 0], [10, 1], [11, 1]])
    env2 = Environment.from_samples(data2, bin_size=1.5)
    env2.name = "env2"

    return env1, env2


@pytest.fixture
def composite_with_regions(two_simple_2d_envs):
    """Create composite with regions defined."""
    env1, env2 = two_simple_2d_envs
    comp = CompositeEnvironment([env1, env2], auto_bridge=True)

    # Add regions using bin centers that definitely exist
    comp.regions.add("left_point", point=comp.bin_centers[0].tolist())
    comp.regions.add("right_point", point=comp.bin_centers[-1].tolist())

    return comp


class TestBinsInRegion:
    """Tests for bins_in_region() method."""

    def test_bins_in_region_point_region(self, composite_with_regions):
        """Test bins_in_region with point region."""
        comp = composite_with_regions

        # Get bins in left point region
        bins = comp.bins_in_region("left_point")

        assert isinstance(bins, np.ndarray)
        assert bins.dtype == np.int_
        assert len(bins) >= 1
        # Should include the first bin
        assert 0 in bins

    def test_bins_in_region_nonexistent(self, two_simple_2d_envs):
        """Test bins_in_region with nonexistent region raises KeyError."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        with pytest.raises(KeyError):
            comp.bins_in_region("nonexistent")

    def test_bins_in_region_polygon_2d(self, two_simple_2d_envs):
        """Test bins_in_region with polygon region."""
        from shapely.geometry import Polygon as ShapelyPolygon

        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        # Create polygon covering left environment
        polygon = ShapelyPolygon([(-1, -1), (5, -1), (5, 5), (-1, 5)])
        comp.regions.add("left_area", polygon=polygon)

        bins = comp.bins_in_region("left_area")

        assert isinstance(bins, np.ndarray)
        assert len(bins) > 0
        # Should contain bins from first environment
        assert bins.max() < env1.n_bins  # All from first env

    def test_bins_in_region_wrong_dimensionality(self, two_simple_2d_envs):
        """Test bins_in_region with wrong dimension point raises ValueError."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        # Add 1D point to 2D environment
        comp.regions.add("bad_point", point=[5.0])

        with pytest.raises(ValueError, match=r"dimension.*does not match"):
            comp.bins_in_region("bad_point")


class TestMaskForRegion:
    """Tests for mask_for_region() method."""

    def test_mask_for_region_basic(self, composite_with_regions):
        """Test mask_for_region returns correct boolean mask."""
        comp = composite_with_regions

        mask = comp.mask_for_region("left_point")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (comp.n_bins,)
        assert np.sum(mask) >= 1  # At least one bin selected

    def test_mask_for_region_matches_bins(self, composite_with_regions):
        """Test mask_for_region matches bins_in_region."""
        comp = composite_with_regions

        bins = comp.bins_in_region("left_point")
        mask = comp.mask_for_region("left_point")

        # Mask should be True exactly where bins indicate
        assert np.array_equal(np.where(mask)[0], bins)

    def test_mask_for_region_nonexistent(self, two_simple_2d_envs):
        """Test mask_for_region with nonexistent region raises KeyError."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        with pytest.raises(KeyError):
            comp.mask_for_region("nonexistent")


class TestShortestPath:
    """Tests for shortest_path() method."""

    def test_shortest_path_within_environment(self, two_simple_2d_envs):
        """Test shortest_path between bins in same sub-environment."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        # Path within first environment
        if env1.n_bins >= 2:
            path = comp.shortest_path(0, 1)

            assert isinstance(path, list)
            assert len(path) >= 2
            assert path[0] == 0
            assert path[-1] == 1

    def test_shortest_path_across_bridge(self, two_simple_2d_envs):
        """Test shortest_path crosses bridge between environments."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        # Path from first to second environment
        source_bin = 0  # First bin of env1
        target_bin = comp.n_bins - 1  # Last bin of env2

        path = comp.shortest_path(source_bin, target_bin)

        assert isinstance(path, list)
        # Path should exist if bridge was created
        if len(path) > 0:
            assert path[0] == source_bin
            assert path[-1] == target_bin
            # Path should span both environments
            assert any(b < env1.n_bins for b in path)
            assert any(b >= env1.n_bins for b in path)

    def test_shortest_path_same_node(self, two_simple_2d_envs):
        """Test shortest_path from node to itself."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        path = comp.shortest_path(0, 0)

        assert path == [0]

    def test_shortest_path_no_path_warns(self, two_simple_2d_envs):
        """Test shortest_path warns when no path exists."""
        env1, env2 = two_simple_2d_envs
        # No auto-bridge, so environments are disconnected
        comp = CompositeEnvironment([env1, env2], auto_bridge=False)

        source_bin = 0  # In env1
        target_bin = env1.n_bins  # In env2 (disconnected)

        with pytest.warns(UserWarning, match="No path found"):
            path = comp.shortest_path(source_bin, target_bin)

        assert path == []

    def test_shortest_path_invalid_nodes(self, two_simple_2d_envs):
        """Test shortest_path with invalid bin indices raises."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        with pytest.raises(nx.NodeNotFound):
            comp.shortest_path(comp.n_bins + 10, 0)


class TestInfo:
    """Tests for info() method."""

    def test_info_prints_without_error(self, two_simple_2d_envs):
        """Test info() prints without raising errors."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        # Should not raise
        comp.info()

    def test_info_return_string(self, two_simple_2d_envs):
        """Test info() returns string when requested."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        result = comp.info(return_string=True)

        assert isinstance(result, str)
        assert "Composite Environment Information" in result
        assert "sub-environments" in result.lower()
        assert str(comp.n_bins) in result
        assert str(comp.n_dims) in result

    def test_info_contains_bridge_stats(self, two_simple_2d_envs):
        """Test info() includes bridge statistics."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        info_str = comp.info(return_string=True)

        assert "Bridge" in info_str
        # Should show either bridge stats or no bridges message
        assert ("Count:" in info_str) or ("No bridges" in info_str)

    def test_info_shows_subenv_details(self, two_simple_2d_envs):
        """Test info() shows sub-environment details."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=False)

        info_str = comp.info(return_string=True)

        assert "env1" in info_str or "env2" in info_str
        assert "RegularGrid" in info_str  # Layout type
        assert "Bins:" in info_str


class TestSaveLoad:
    """Tests for save() and load() methods."""

    def test_save_and_load_basic(self, two_simple_2d_envs):
        """Test saving and loading composite environment."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_composite.pkl"

            # Save
            comp.save(str(filepath))
            assert filepath.exists()

            # Load
            loaded = CompositeEnvironment.load(str(filepath))

            assert loaded.n_bins == comp.n_bins
            assert loaded.n_dims == comp.n_dims
            assert len(loaded._subenvs_info) == len(comp._subenvs_info)

    def test_load_preserves_structure(self, two_simple_2d_envs):
        """Test loaded composite has same structure as original."""
        env1, env2 = two_simple_2d_envs

        # Add regions to sub-environments (will be preserved through save/load)
        env1.regions.add("env1_region", point=env1.bin_centers[0].tolist())
        env2.regions.add("env2_region", point=env2.bin_centers[0].tolist())

        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_composite.pkl"
            comp.save(str(filepath))
            loaded = CompositeEnvironment.load(str(filepath))

            # Check bin centers match
            np.testing.assert_array_almost_equal(loaded.bin_centers, comp.bin_centers)

            # Check connectivity preserved
            assert (
                loaded.connectivity.number_of_nodes()
                == comp.connectivity.number_of_nodes()
            )
            assert (
                loaded.connectivity.number_of_edges()
                == comp.connectivity.number_of_edges()
            )

            # Check regions preserved (regions from sub-environments)
            assert "env1_region" in loaded.regions or "env2_region" in loaded.regions

    def test_load_preserves_bridges(self, two_simple_2d_envs):
        """Test loaded composite preserves bridge edges."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        n_bridges_original = len(comp._bridge_list)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_composite.pkl"
            comp.save(str(filepath))
            loaded = CompositeEnvironment.load(str(filepath))

            assert len(loaded._bridge_list) == n_bridges_original

    def test_save_creates_file(self, two_simple_2d_envs):
        """Test save() creates the file."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pkl"
            comp.save(str(filepath))

            assert filepath.exists()
            assert filepath.stat().st_size > 0


class TestIntegration:
    """Integration tests for multiple methods working together."""

    def test_region_query_after_load(self, two_simple_2d_envs):
        """Test region queries work after loading."""
        env1, env2 = two_simple_2d_envs

        # Add regions to sub-environments before creating composite
        env1.regions.add("test_point", point=env1.bin_centers[0].tolist())

        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_composite.pkl"
            comp.save(str(filepath))
            loaded = CompositeEnvironment.load(str(filepath))

            # Query should work on regions from sub-environments
            bins = loaded.bins_in_region("test_point")
            assert len(bins) >= 1

            mask = loaded.mask_for_region("test_point")
            assert np.sum(mask) >= 1

    def test_path_finding_after_load(self, two_simple_2d_envs):
        """Test pathfinding works after loading."""
        env1, env2 = two_simple_2d_envs
        comp = CompositeEnvironment([env1, env2], auto_bridge=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_composite.pkl"
            comp.save(str(filepath))
            loaded = CompositeEnvironment.load(str(filepath))

            # Pathfinding should work
            path = loaded.shortest_path(0, 0)
            assert path == [0]
