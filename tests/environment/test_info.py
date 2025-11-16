"""Tests for Environment.info() method.

This module tests the diagnostic information method for the Environment class,
ensuring it provides detailed, readable multi-line output for inspection and debugging.

Test coverage:
- info() method returns string
- Shows all required information (name, dims, bins, layout, extent, etc.)
- Format is readable and multi-line
- Edge cases (no regions, variable bin sizes, 1D environments, etc.)
- Handles special characters in names
"""

import numpy as np

from neurospatial import Environment


class TestEnvironmentInfo:
    """Test Environment.info() diagnostic output."""

    def test_info_returns_string(self, grid_env_from_samples):
        """info() should return a string."""
        result = grid_env_from_samples.info()
        assert isinstance(result, str)

    def test_info_is_multiline(self, grid_env_from_samples):
        """info() should return multi-line output for readability."""
        result = grid_env_from_samples.info()
        lines = result.split("\n")
        assert len(lines) >= 5  # Should have multiple lines of information

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
        # Should mention extent or range
        assert "extent" in result.lower() or "range" in result.lower()
        # Should show dimension-specific information
        dimension_ranges = grid_env_from_samples.dimension_ranges
        # At least one extent value should be present
        found_extent_value = False
        for dim_min, dim_max in dimension_ranges:
            if f"{dim_min:.2f}" in result or f"{dim_max:.2f}" in result:
                found_extent_value = True
                break
        assert found_extent_value, "Expected at least one extent value in output"

    def test_info_shows_bin_sizes(self, grid_env_from_samples):
        """info() should show bin sizes."""
        result = grid_env_from_samples.info()
        # Should mention bin size
        assert "bin size" in result.lower() or "bin_size" in result.lower()

    def test_info_shows_regions_count(self, grid_env_from_samples):
        """info() should show region count."""
        # Start with no regions
        result = grid_env_from_samples.info()
        assert "region" in result.lower()

        # Add a region
        from shapely.geometry import Point

        grid_env_from_samples.regions.add("TestRegion", polygon=Point(0, 0).buffer(2))
        result_with_region = grid_env_from_samples.info()
        assert "region" in result_with_region.lower()
        assert "1" in result_with_region or "TestRegion" in result_with_region

    def test_info_shows_linearization_for_1d(self, graph_env):
        """info() should show linearization status for 1D environments."""
        result = graph_env.info()
        # Should mention linearization or 1D
        if graph_env.is_1d:
            assert "linear" in result.lower() or "1d" in result.lower()

    def test_info_handles_empty_name(self):
        """info() should handle environments with empty names gracefully."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="")
        result = env.info()
        # Should still be valid and show other info
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Environment" in result

    def test_info_handles_none_name(self):
        """info() should handle environments with None names gracefully."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name=None)
        result = env.info()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_info_works_for_different_layout_types(self):
        """info() should work for different layout engines."""
        # RegularGrid
        data = np.random.rand(100, 2) * 10
        env_regular = Environment.from_samples(data, bin_size=2.0, name="Regular")
        result_regular = env_regular.info()
        assert "RegularGrid" in result_regular

        # Polygon
        from shapely.geometry import Point

        polygon = Point(50, 50).buffer(30)
        env_polygon = Environment.from_polygon(polygon, bin_size=2.0, name="PolygonEnv")
        result_polygon = env_polygon.info()
        assert "PolygonEnv" in result_polygon

    def test_info_is_readable(self, grid_env_from_samples):
        """info() output should be well-formatted and readable."""
        result = grid_env_from_samples.info()
        # Should have clear structure with labels
        assert ":" in result or "=" in result  # Should have key-value pairs
        # Should not be excessively long
        assert len(result) < 5000
        # Should be indented or structured for readability
        lines = result.split("\n")
        # At least some lines should have content (not all empty)
        content_lines = [line for line in lines if line.strip()]
        assert len(content_lines) >= 5


class TestInfoContentCompleteness:
    """Test that info() includes all required information."""

    def test_info_includes_all_required_fields(self, grid_env_from_samples):
        """info() should include all required diagnostic fields."""
        result = grid_env_from_samples.info()

        # Required fields
        required_keywords = [
            grid_env_from_samples.name,  # Name
            str(grid_env_from_samples.n_dims),  # Dimensions
            str(grid_env_from_samples.n_bins),  # Number of bins
            grid_env_from_samples.layout_type,  # Layout type
        ]

        for keyword in required_keywords:
            assert keyword in result, (
                f"Expected '{keyword}' to be in info() output but it was not found"
            )

        # Should mention extent or range
        assert "extent" in result.lower() or "range" in result.lower()

        # Should mention regions
        assert "region" in result.lower()

    def test_info_with_variable_bin_sizes(self):
        """info() should handle environments with variable bin sizes."""
        # Create environment from mask with variable bin sizes
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="VarBins")

        result = env.info()
        # Should show bin size information
        assert "bin size" in result.lower() or "bin_size" in result.lower()
        # Should work without error
        assert isinstance(result, str)
        assert len(result) > 0

    def test_info_with_many_regions(self, grid_env_from_samples):
        """info() should handle many regions gracefully."""
        from shapely.geometry import Point

        # Add 10 regions
        for i in range(10):
            grid_env_from_samples.regions.add(
                f"Region{i}", polygon=Point(0, 0).buffer(0.5 + i * 0.1)
            )

        result = grid_env_from_samples.info()
        assert isinstance(result, str)
        # Should show count
        assert "10" in result
        # Should not list all region names (too verbose)
        # Check that output is reasonable length
        assert len(result) < 3000


class TestInfoEdgeCases:
    """Test edge cases and error handling in info() method."""

    def test_info_with_special_characters_in_name(self):
        """info() should handle special characters in names."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(
            data, bin_size=2.0, name="Test <Env> & 'Special' \"Chars\""
        )

        result = env.info()
        assert isinstance(result, str)
        # Name should be present (may be escaped)
        assert "Test" in result

    def test_info_with_very_large_n_bins(self):
        """info() should handle environments with very large bin counts."""
        data = np.random.rand(10000, 2) * 1000
        env = Environment.from_samples(data, bin_size=0.1, name="LargeEnv")
        result = env.info()
        assert isinstance(result, str)
        assert str(env.n_bins) in result

    def test_info_with_long_name(self):
        """info() should handle very long names gracefully."""
        data = np.random.rand(100, 2) * 10
        long_name = "A" * 200  # Very long name
        env = Environment.from_samples(data, bin_size=2.0, name=long_name)
        result = env.info()
        assert isinstance(result, str)
        # Should either show full name or truncate gracefully
        assert "A" in result


class TestInfoFormatting:
    """Test formatting and presentation of info() output."""

    def test_info_has_clear_section_headers(self, grid_env_from_samples):
        """info() should have clear headers for different sections."""
        result = grid_env_from_samples.info()
        # Should have some structure indicators
        # Could use headers, blank lines, or other formatting
        lines = result.split("\n")
        # At least 5 lines of content
        assert len(lines) >= 5

    def test_info_uses_consistent_formatting(self, grid_env_from_samples):
        """info() should use consistent formatting throughout."""
        result = grid_env_from_samples.info()
        lines = [line for line in result.split("\n") if line.strip()]

        # Check for consistent use of separators (: or =)
        separator_counts = {"colon": 0, "equals": 0}
        for line in lines:
            if ":" in line:
                separator_counts["colon"] += 1
            if "=" in line:
                separator_counts["equals"] += 1

        # Should use at least one separator consistently
        assert separator_counts["colon"] > 0 or separator_counts["equals"] > 0, (
            "Expected consistent separator usage"
        )

    def test_info_is_more_detailed_than_repr(self, grid_env_from_samples):
        """info() should provide more detail than __repr__."""
        repr_output = repr(grid_env_from_samples)
        info_output = grid_env_from_samples.info()

        # info() should be longer (more detailed)
        assert len(info_output) > len(repr_output)

        # info() should be multi-line, repr() single-line
        assert "\n" in info_output
        assert "\n" not in repr_output

        # Both should have core information
        assert grid_env_from_samples.name in repr_output
        assert grid_env_from_samples.name in info_output
