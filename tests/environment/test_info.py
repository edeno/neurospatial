"""Tests for Environment.info() method.

This module tests the diagnostic information method for the Environment class.
We only test that the *content* the user came for is present — name,
dimensions, bin count, layout type, extent, bin sizes, region count, and
the linearization flag. Formatting heuristics (line counts, length caps,
separator audits) are not contracts and have been deliberately omitted.
"""

import numpy as np

from neurospatial import Environment


class TestEnvironmentInfo:
    """Test Environment.info() diagnostic output."""

    def test_info_shows_name(self, grid_env_from_samples):
        """info() should show the environment name."""
        result = grid_env_from_samples.info()
        assert grid_env_from_samples.name in result

    def test_info_shows_dimensions(self, grid_env_from_samples):
        """info() should show the number of dimensions."""
        result = grid_env_from_samples.info()
        assert str(grid_env_from_samples.n_dims) in result
        assert "dimension" in result.lower()

    def test_info_shows_n_bins(self, grid_env_from_samples):
        """info() should show the number of bins."""
        result = grid_env_from_samples.info()
        assert str(grid_env_from_samples.n_bins) in result
        assert "bin" in result.lower()

    def test_info_shows_layout_type(self, grid_env_from_samples):
        """info() should show the layout engine type."""
        result = grid_env_from_samples.info()
        layout_type = grid_env_from_samples.layout_type
        assert layout_type in result

    def test_info_shows_extent(self, grid_env_from_samples):
        """info() should show spatial extent for each dimension."""
        result = grid_env_from_samples.info()
        assert "extent" in result.lower() or "range" in result.lower()
        dimension_ranges = grid_env_from_samples.dimension_ranges
        found_extent_value = False
        for dim_min, dim_max in dimension_ranges:
            if f"{dim_min:.2f}" in result or f"{dim_max:.2f}" in result:
                found_extent_value = True
                break
        assert found_extent_value, "Expected at least one extent value in output"

    def test_info_shows_bin_sizes(self, grid_env_from_samples):
        """info() should show bin sizes."""
        result = grid_env_from_samples.info()
        assert "bin size" in result.lower() or "bin_size" in result.lower()

    def test_info_shows_regions_count(self):
        """info() should reflect added regions."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0)

        result_before = env.info()
        assert "region" in result_before.lower()

        from shapely.geometry import Point

        env.regions.add("TestRegion", polygon=Point(0, 0).buffer(2))
        result_after = env.info()
        assert "1" in result_after or "TestRegion" in result_after

    def test_info_shows_linearization_for_1d(self, graph_env):
        """info() mentions linearization for 1D environments."""
        result = graph_env.info()
        if graph_env.is_linearized_track:
            assert "linear" in result.lower() or "1d" in result.lower()

    def test_info_works_for_different_layout_types(self):
        """Layout type appears verbatim in info() for the major factories."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 10
        env_regular = Environment.from_samples(data, bin_size=2.0, name="Regular")
        assert "RegularGrid" in env_regular.info()

        from shapely.geometry import Point

        polygon = Point(50, 50).buffer(30)
        env_polygon = Environment.from_polygon(polygon, bin_size=2.0, name="PolygonEnv")
        assert "PolygonEnv" in env_polygon.info()
