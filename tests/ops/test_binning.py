"""Tests for neurospatial.ops.binning module."""

import numpy as np
import pytest


class TestBinningImports:
    """Test that binning module exports are available from new location."""

    def test_import_map_points_to_bins(self):
        """Test map_points_to_bins can be imported from ops.binning."""
        from neurospatial.ops.binning import map_points_to_bins

        assert callable(map_points_to_bins)

    def test_import_tie_break_strategy(self):
        """Test TieBreakStrategy can be imported from ops.binning."""
        from neurospatial.ops.binning import TieBreakStrategy

        assert hasattr(TieBreakStrategy, "LOWEST_INDEX")
        assert hasattr(TieBreakStrategy, "CLOSEST_CENTER")

    def test_import_regions_to_mask(self):
        """Test regions_to_mask can be imported from ops.binning."""
        from neurospatial.ops.binning import regions_to_mask

        assert callable(regions_to_mask)

    def test_import_resample_field(self):
        """Test resample_field can be imported from ops.binning."""
        from neurospatial.ops.binning import resample_field

        assert callable(resample_field)

    def test_import_clear_kdtree_cache(self):
        """Test clear_kdtree_cache can be imported from ops.binning."""
        from neurospatial.ops.binning import clear_kdtree_cache

        assert callable(clear_kdtree_cache)


class TestMapPointsToBins:
    """Test map_points_to_bins function from new location."""

    @pytest.fixture
    def grid_env(self):
        """Create a simple grid environment."""
        from neurospatial import Environment

        # Create data on a regular grid (deterministic, no RNG needed)
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=2.0, name="grid")
        return env

    def test_map_points_basic(self, grid_env):
        """Test basic point mapping from new import location."""
        from neurospatial.ops.binning import map_points_to_bins

        points = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (3,)
        assert bins.dtype == np.int64
        # All points should map to valid bins (>= 0)
        assert np.all(bins >= 0)

    def test_tie_break_strategy_enum(self, grid_env):
        """Test using TieBreakStrategy enum from new location."""
        from neurospatial.ops.binning import TieBreakStrategy, map_points_to_bins

        points = np.array([[5.0, 5.0]])
        bins = map_points_to_bins(
            points, grid_env, tie_break=TieBreakStrategy.LOWEST_INDEX
        )

        assert bins.shape == (1,)
        assert bins[0] >= 0

    def test_lowest_index_tie_break_handles_more_than_ten_ties(self):
        """lowest_index should consider all equidistant bins, not only KDTree k."""
        from neurospatial.ops.binning import map_points_to_bins

        class MinimalEnvironment:
            def __init__(self, bin_centers):
                self.bin_centers = bin_centers
                self._kdtree_cache = None

        n_bins = 20
        angles = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
        bin_centers = np.column_stack([np.cos(angles), np.sin(angles)])
        env = MinimalEnvironment(bin_centers)
        points = np.array([[0.0, 0.0]])

        bins = map_points_to_bins(
            points,
            env,  # type: ignore[arg-type]
            tie_break="lowest_index",
            max_distance=2.0,
        )

        assert bins[0] == 0
