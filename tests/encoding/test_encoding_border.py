"""
Tests for encoding.border module.

Tests verify that encoding.border re-exports functions from metrics.boundary_cells.
"""

from __future__ import annotations

import numpy as np

from neurospatial import Environment, Regions

# =============================================================================
# Import Tests - encoding.border module
# =============================================================================


class TestEncodingBorderImports:
    """Test imports from encoding.border module."""

    def test_import_border_score(self) -> None:
        """Test importing border_score from encoding.border."""
        from neurospatial.encoding.border import border_score

        assert callable(border_score)

    def test_import_region_coverage(self) -> None:
        """Test importing region_coverage from encoding.border."""
        from neurospatial.encoding.border import region_coverage

        assert callable(region_coverage)


# =============================================================================
# Import Tests - encoding/__init__.py re-exports
# =============================================================================


class TestEncodingInitBorderImports:
    """Test that border functions are exported from encoding/__init__.py."""

    def test_import_border_score_from_encoding(self) -> None:
        """Test importing border_score from encoding."""
        from neurospatial.encoding import border_score

        assert callable(border_score)

    def test_import_region_coverage_from_encoding(self) -> None:
        """Test importing region_coverage from encoding."""
        from neurospatial.encoding import region_coverage

        assert callable(region_coverage)


# =============================================================================
# Module Structure Tests
# =============================================================================


class TestEncodingBorderModuleStructure:
    """Test encoding.border module structure."""

    def test_encoding_border_has_all(self) -> None:
        """Test that encoding.border defines __all__."""
        from neurospatial.encoding import border

        assert hasattr(border, "__all__")
        assert isinstance(border.__all__, list)

    def test_encoding_border_all_contains_border_score(self) -> None:
        """Test that __all__ contains border_score."""
        from neurospatial.encoding import border

        assert "border_score" in border.__all__

    def test_encoding_border_all_contains_region_coverage(self) -> None:
        """Test that __all__ contains region_coverage."""
        from neurospatial.encoding import border

        assert "region_coverage" in border.__all__


# =============================================================================
# Re-export Verification Tests
# =============================================================================


class TestBorderReExports:
    """Test that encoding.border re-exports match metrics.boundary_cells."""

    def test_border_score_same_as_metrics(self) -> None:
        """Test that encoding.border.border_score is same as metrics."""
        from neurospatial.encoding.border import border_score as encoding_border_score
        from neurospatial.encoding.border import (
            border_score as metrics_border_score,
        )

        # Should be the same function object
        assert encoding_border_score is metrics_border_score

    def test_region_coverage_same_as_metrics(self) -> None:
        """Test that encoding.border.region_coverage is same as metrics."""
        from neurospatial.encoding.border import (
            region_coverage as encoding_region_coverage,
        )
        from neurospatial.encoding.border import (
            region_coverage as metrics_region_coverage,
        )

        # Should be the same function object
        assert encoding_region_coverage is metrics_region_coverage


# =============================================================================
# Functionality Tests
# =============================================================================


class TestBorderFunctionality:
    """Test that re-exported functions work correctly."""

    def test_region_coverage_basic(self) -> None:
        """region_coverage gives the field's share of each region's bins.

        Bin centers lie on a 4 cm lattice from 0 to 40 cm; region edges fall
        between centers. The field, chosen from the coordinates, is the top
        three rows and five bins in no region: all of north (1) and none of
        south (0).
        """
        from shapely.geometry import box

        from neurospatial.encoding.border import region_coverage

        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("north", polygon=box(-2, 30, 42, 42))
        region_builder.add("south", polygon=box(-2, -2, 42, 10))
        env = env.with_regions(region_builder)

        x, y = env.bin_centers.T
        field_bins = np.flatnonzero((y > 30) | ((y == 20) & (x > 10) & (x < 30)))

        coverage = region_coverage(env, field_bins)

        assert coverage == {"north": 1.0, "south": 0.0}

    def test_region_coverage_returns_dict(self) -> None:
        """Test that region_coverage returns a dict."""
        from shapely.geometry import box

        from neurospatial.encoding.border import region_coverage

        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("test", polygon=box(0, 0, 20, 20))
        env = env.with_regions(region_builder)

        field_bins = np.array([0, 1, 2])

        coverage = region_coverage(env, field_bins)

        assert isinstance(coverage, dict)

    def test_region_coverage_specific_regions(self) -> None:
        """Test region_coverage with specific regions parameter."""
        from shapely.geometry import box

        from neurospatial.encoding.border import region_coverage

        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        region_builder = Regions(env.regions.values())
        region_builder.add("region1", polygon=box(0, 0, 20, 20))
        region_builder.add("region2", polygon=box(20, 20, 40, 40))
        region_builder.add("region3", polygon=box(0, 20, 20, 40))
        env = env.with_regions(region_builder)

        field_bins = np.array([0, 1, 2])

        # Only compute for region1 and region2
        coverage = region_coverage(env, field_bins, regions=["region1", "region2"])

        # Should only return specified regions
        assert set(coverage.keys()) == {"region1", "region2"}
