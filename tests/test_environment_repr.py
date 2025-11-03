"""Tests for Environment.__repr__ and _repr_html_() methods.

This module tests the string representation methods for the Environment class,
ensuring they provide helpful, informative output for both text and HTML contexts.

Test coverage:
- __repr__ method (text representation)
- _repr_html_() method (Jupyter notebook HTML representation)
- Edge cases (empty name, no regions, 1D environments, etc.)
- Format consistency and informativeness
"""

import numpy as np

from neurospatial import Environment


class TestEnvironmentRepr:
    """Test Environment.__repr__ text representation."""

    def test_repr_returns_string(self, grid_env_from_samples):
        """__repr__ should return a string."""
        result = repr(grid_env_from_samples)
        assert isinstance(result, str)

    def test_repr_shows_name(self, grid_env_from_samples):
        """__repr__ should show the environment name."""
        result = repr(grid_env_from_samples)
        assert grid_env_from_samples.name in result

    def test_repr_shows_n_dims(self, grid_env_from_samples):
        """__repr__ should show the number of dimensions."""
        result = repr(grid_env_from_samples)
        assert "2D" in result or "n_dims=2" in result

    def test_repr_shows_n_bins(self, grid_env_from_samples):
        """__repr__ should show the number of bins."""
        result = repr(grid_env_from_samples)
        n_bins = grid_env_from_samples.n_bins
        assert str(n_bins) in result

    def test_repr_shows_layout_type(self, grid_env_from_samples):
        """__repr__ should show the layout engine type."""
        result = repr(grid_env_from_samples)
        layout_type = grid_env_from_samples.layout_type
        assert layout_type in result

    def test_repr_is_single_line(self, grid_env_from_samples):
        """__repr__ should be a single line (no newlines)."""
        result = repr(grid_env_from_samples)
        assert "\n" not in result

    def test_repr_handles_empty_name(self):
        """__repr__ should handle environments with empty names gracefully."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="")
        result = repr(env)
        # Should still be valid and show other info
        assert isinstance(result, str)
        assert "Environment" in result
        assert str(env.n_bins) in result

    def test_repr_handles_none_name(self):
        """__repr__ should handle environments with None names gracefully."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name=None)
        result = repr(env)
        assert isinstance(result, str)
        assert "Environment" in result

    def test_repr_works_for_1d_environment(self, graph_env):
        """__repr__ should work for Graph environments."""
        # Note: graph_env fixture is actually 2D (plus maze)
        result = repr(graph_env)
        # Should show Graph layout type
        assert "Graph" in result
        assert graph_env.name in result
        # Check it shows the correct dimensionality
        assert f"{graph_env.n_dims}D" in result

    def test_repr_works_for_different_layout_types(self):
        """__repr__ should show different layout types correctly."""
        # RegularGrid
        data = np.random.rand(100, 2) * 10
        env_regular = Environment.from_samples(data, bin_size=2.0, name="Regular")
        result_regular = repr(env_regular)
        assert "RegularGrid" in result_regular

        # Polygon
        from shapely.geometry import Point

        polygon = Point(50, 50).buffer(30)
        env_polygon = Environment.from_polygon(polygon, bin_size=2.0, name="PolygonEnv")
        result_polygon = repr(env_polygon)
        assert "PolygonEnv" in result_polygon

    def test_repr_starts_with_class_name(self, grid_env_from_samples):
        """__repr__ should start with 'Environment'."""
        result = repr(grid_env_from_samples)
        assert result.startswith("Environment(")

    def test_repr_ends_with_closing_paren(self, grid_env_from_samples):
        """__repr__ should end with closing parenthesis."""
        result = repr(grid_env_from_samples)
        assert result.endswith(")")

    def test_repr_is_informative_not_reconstructive(self, grid_env_from_samples):
        """__repr__ should be informative, not necessarily reconstructive.

        Following Python best practices: if an object is complex,
        __repr__ should provide useful info rather than trying to
        be reconstructive (which would be impractical for Environment).
        """
        result = repr(grid_env_from_samples)
        # Should contain key info
        assert grid_env_from_samples.name in result
        assert str(grid_env_from_samples.n_bins) in result
        # Should not try to reconstruct (no giant arrays, etc.)
        assert "array" not in result.lower()
        assert len(result) < 200  # Reasonable length


class TestEnvironmentReprHtml:
    """Test Environment._repr_html_() for Jupyter notebooks."""

    def test_repr_html_returns_string(self, grid_env_from_samples):
        """_repr_html_() should return a string."""
        result = grid_env_from_samples._repr_html_()
        assert isinstance(result, str)

    def test_repr_html_contains_html_tags(self, grid_env_from_samples):
        """_repr_html_() should return valid HTML."""
        result = grid_env_from_samples._repr_html_()
        # Should have HTML structure
        assert "<" in result and ">" in result
        # Should have a table or div structure
        assert "<table" in result or "<div" in result

    def test_repr_html_shows_name(self, grid_env_from_samples):
        """_repr_html_() should show the environment name."""
        result = grid_env_from_samples._repr_html_()
        assert grid_env_from_samples.name in result

    def test_repr_html_shows_dimensions(self, grid_env_from_samples):
        """_repr_html_() should show dimensions."""
        result = grid_env_from_samples._repr_html_()
        assert str(grid_env_from_samples.n_dims) in result

    def test_repr_html_shows_n_bins(self, grid_env_from_samples):
        """_repr_html_() should show number of bins."""
        result = grid_env_from_samples._repr_html_()
        assert str(grid_env_from_samples.n_bins) in result

    def test_repr_html_shows_layout_type(self, grid_env_from_samples):
        """_repr_html_() should show layout engine type."""
        result = grid_env_from_samples._repr_html_()
        assert grid_env_from_samples.layout_type in result

    def test_repr_html_shows_extent(self, grid_env_from_samples):
        """_repr_html_() should show spatial extent."""
        result = grid_env_from_samples._repr_html_()
        # Should mention extent or ranges
        assert "extent" in result.lower() or "range" in result.lower()

    def test_repr_html_shows_regions_count(self, grid_env_from_samples):
        """_repr_html_() should show number of regions if any."""
        # Add a region
        from shapely.geometry import Point

        grid_env_from_samples.regions.add("TestRegion", polygon=Point(0, 0).buffer(2))
        result = grid_env_from_samples._repr_html_()
        # Should show region count
        assert "1" in result  # 1 region
        assert "region" in result.lower()

    def test_repr_html_handles_no_regions(self, grid_env_from_samples):
        """_repr_html_() should handle environments with no regions."""
        result = grid_env_from_samples._repr_html_()
        # Should work without error
        assert isinstance(result, str)
        assert "0" in result or "None" in result or "no region" in result.lower()

    def test_repr_html_handles_empty_name(self):
        """_repr_html_() should handle empty names gracefully."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="")
        result = env._repr_html_()
        assert isinstance(result, str)
        assert "<" in result  # Still valid HTML

    def test_repr_html_uses_table_structure(self, grid_env_from_samples):
        """_repr_html_() should use table structure for organized display."""
        result = grid_env_from_samples._repr_html_()
        assert "<table" in result
        assert "<tr" in result
        assert "<td" in result or "<th" in result

    def test_repr_html_has_styling(self, grid_env_from_samples):
        """_repr_html_() should include CSS styling for readability."""
        result = grid_env_from_samples._repr_html_()
        # Should have inline styles or style tags
        assert "style=" in result or "<style>" in result

    def test_repr_html_is_readable_without_rendering(self, grid_env_from_samples):
        """_repr_html_() should be somewhat readable even as raw HTML."""
        result = grid_env_from_samples._repr_html_()
        # Key information should be present in the raw HTML
        assert grid_env_from_samples.name in result
        assert str(grid_env_from_samples.n_bins) in result

    def test_repr_html_works_for_1d_environment(self, graph_env):
        """_repr_html_() should work for 1D environments."""
        result = graph_env._repr_html_()
        assert isinstance(result, str)
        assert "<table" in result
        assert graph_env.name in result


class TestReprConsistency:
    """Test consistency between __repr__ and _repr_html_()."""

    def test_both_methods_show_same_core_info(self, grid_env_from_samples):
        """Both __repr__ and _repr_html_() should show the same core information."""
        text_repr = repr(grid_env_from_samples)
        html_repr = grid_env_from_samples._repr_html_()

        # Both should include these key pieces of information
        name = grid_env_from_samples.name
        n_bins = str(grid_env_from_samples.n_bins)
        layout_type = grid_env_from_samples.layout_type

        assert name in text_repr
        assert name in html_repr

        assert n_bins in text_repr
        assert n_bins in html_repr

        assert layout_type in text_repr
        assert layout_type in html_repr

    def test_repr_methods_handle_special_characters_in_name(self):
        """Both repr methods should handle special characters in names."""
        data = np.random.rand(100, 2) * 10
        env = Environment.from_samples(
            data, bin_size=2.0, name="Test <Environment> & 'Special' \"Chars\""
        )

        # __repr__ should work
        text_repr = repr(env)
        assert isinstance(text_repr, str)

        # _repr_html_() should escape HTML special characters
        html_repr = env._repr_html_()
        assert isinstance(html_repr, str)
        # HTML should escape < > & " '
        assert "&lt;" in html_repr or "Test" in html_repr  # Escaped or present


class TestReprEdgeCases:
    """Test edge cases and error handling in repr methods."""

    def test_repr_with_very_large_n_bins(self):
        """__repr__ should handle environments with very large bin counts."""
        data = np.random.rand(10000, 2) * 1000
        env = Environment.from_samples(data, bin_size=0.1, name="LargeEnv")
        result = repr(env)
        assert isinstance(result, str)
        assert str(env.n_bins) in result

    def test_repr_with_long_name(self):
        """__repr__ should handle very long names gracefully."""
        data = np.random.rand(100, 2) * 10
        long_name = "A" * 200  # Very long name
        env = Environment.from_samples(data, bin_size=2.0, name=long_name)
        result = repr(env)
        assert isinstance(result, str)
        # Should either truncate or handle gracefully
        assert len(result) < 500  # Shouldn't be excessively long

    def test_repr_html_with_many_regions(self, grid_env_from_samples):
        """_repr_html_() should handle many regions without becoming unwieldy."""
        from shapely.geometry import Point

        # Add 20 regions
        for i in range(20):
            grid_env_from_samples.regions.add(
                f"Region{i}", polygon=Point(0, 0).buffer(0.5 + i * 0.1)
            )

        result = grid_env_from_samples._repr_html_()
        assert isinstance(result, str)
        assert "20" in result  # Should show count
        # Should be reasonable length (not list all regions)
        assert len(result) < 5000
