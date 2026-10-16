"""Tests for Environment.__repr__ and _repr_html_() methods.

We assert only that the content the user came for is in the output:
name, dimensions, bin count, layout type. Cosmetic formatting (single-line,
must-start-with-class-name, length caps, exact HTML tags) is not a contract
and has been deliberately left untested.

Empty names remain explicit, and invalid replacement names cannot change the
representation of an existing environment.
"""

import numpy as np
import pytest

from neurospatial import Environment, Regions


class TestEnvironmentRepr:
    """Test Environment.__repr__ text representation."""

    def test_repr_shows_name(self, grid_env_from_samples):
        """__repr__ should show the environment name."""
        assert grid_env_from_samples.name in repr(grid_env_from_samples)

    def test_repr_shows_n_dims(self, grid_env_from_samples):
        """__repr__ should show the number of dimensions."""
        result = repr(grid_env_from_samples)
        assert "2D" in result or "n_dims=2" in result

    def test_repr_shows_n_bins(self, grid_env_from_samples):
        """__repr__ should show the number of bins."""
        assert str(grid_env_from_samples.n_bins) in repr(grid_env_from_samples)

    def test_repr_shows_layout_type(self, grid_env_from_samples):
        """__repr__ should show the layout engine type."""
        assert grid_env_from_samples.layout_type in repr(grid_env_from_samples)

    def test_repr_handles_empty_name(self):
        """__repr__ should distinguish empty-string from None."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="")
        result = repr(env)
        # The empty string must be visibly distinct from the None case.
        # repr(env) for an empty name now contains ``name=''``.
        assert "name=''" in result
        assert "name=None" not in result

    def test_invalid_name_replacement_preserves_repr(self):
        """A name must be a string; failed replacement preserves the source."""
        rng = np.random.default_rng(42)
        data = rng.random((100, 2)) * 10
        env = Environment.from_samples(data, bin_size=2.0, name="x")
        with pytest.raises(TypeError, match="name"):
            env.with_metadata(name=None)  # type: ignore[arg-type]
        assert "name='x'" in repr(env)

    def test_repr_works_for_1d_environment(self, graph_env):
        """__repr__ should work for Graph environments."""
        result = repr(graph_env)
        assert "Graph" in result
        assert graph_env.name in result
        assert f"{graph_env.n_dims}D" in result

    def test_str_returns_info_summary(self, grid_env_from_samples):
        """``str(env)`` returns the same multi-line summary as ``env.info()``."""
        assert str(grid_env_from_samples) == grid_env_from_samples.info()


class TestEnvironmentReprHtml:
    """Test Environment._repr_html_() for Jupyter notebooks."""

    def test_repr_html_shows_name(self, grid_env_from_samples):
        """_repr_html_() should show the environment name."""
        assert grid_env_from_samples.name in grid_env_from_samples._repr_html_()

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
        assert "extent" in result.lower() or "range" in result.lower()

    def test_repr_html_shows_regions_count(self, grid_env_from_samples):
        """_repr_html_() reflects the region count when a region is added."""
        from shapely.geometry import Point

        env = grid_env_from_samples
        region_builder = Regions(env.regions.values())
        region_builder.add("TestRegion", polygon=Point(0, 0).buffer(2))
        env = env.with_regions(region_builder)
        result = env._repr_html_()
        assert "1" in result
        assert "region" in result.lower()

    def test_repr_html_works_for_1d_environment(self, graph_env):
        """_repr_html_() should work for 1D environments."""
        result = graph_env._repr_html_()
        assert graph_env.name in result
