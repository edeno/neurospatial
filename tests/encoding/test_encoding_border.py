"""
Tests for encoding.border module.

Following TDD: Tests written FIRST before implementation.
Tests verify that encoding.border re-exports functions from metrics.boundary_cells.
"""

from __future__ import annotations

import numpy as np

from neurospatial import Environment

# =============================================================================
# Import Tests - encoding.border module
# =============================================================================


class TestEncodingBorderImports:
    """Test imports from encoding.border module."""

    def test_import_border_score(self) -> None:
        """Test importing border_score from encoding.border."""
        from neurospatial.encoding.border import border_score

        assert callable(border_score)

    def test_import_compute_region_coverage(self) -> None:
        """Test importing compute_region_coverage from encoding.border."""
        from neurospatial.encoding.border import compute_region_coverage

        assert callable(compute_region_coverage)


# =============================================================================
# Import Tests - encoding/__init__.py re-exports
# =============================================================================


class TestEncodingInitBorderImports:
    """Test that border functions are exported from encoding/__init__.py."""

    def test_import_border_score_from_encoding(self) -> None:
        """Test importing border_score from encoding."""
        from neurospatial.encoding import border_score

        assert callable(border_score)

    def test_import_compute_region_coverage_from_encoding(self) -> None:
        """Test importing compute_region_coverage from encoding."""
        from neurospatial.encoding import compute_region_coverage

        assert callable(compute_region_coverage)


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

    def test_encoding_border_all_contains_compute_region_coverage(self) -> None:
        """Test that __all__ contains compute_region_coverage."""
        from neurospatial.encoding import border

        assert "compute_region_coverage" in border.__all__


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

    def test_compute_region_coverage_same_as_metrics(self) -> None:
        """Test that encoding.border.compute_region_coverage is same as metrics."""
        from neurospatial.encoding.border import (
            compute_region_coverage as encoding_compute_region_coverage,
        )
        from neurospatial.encoding.border import (
            compute_region_coverage as metrics_compute_region_coverage,
        )

        # Should be the same function object
        assert encoding_compute_region_coverage is metrics_compute_region_coverage


# =============================================================================
# Functionality Tests
# =============================================================================


class TestBorderFunctionality:
    """Test that re-exported functions work correctly."""

    def test_border_score_boundary_field(self) -> None:
        """Test border_score with field on boundary."""
        from neurospatial.encoding.border import border_score

        # Create a small environment
        x = np.linspace(0, 30, 300)
        y = np.linspace(0, 30, 300)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=10.0)

        # Create field on boundary
        firing_rate = np.zeros(env.n_bins)
        firing_rate[env.boundary_bins] = 5.0

        score = border_score(firing_rate, env)

        # Boundary field should have high score
        assert score > 0.5, f"Expected score > 0.5 for boundary field, got {score}"

    def test_border_score_central_field(self) -> None:
        """Test border_score with field in center."""
        from neurospatial.encoding.border import border_score

        # Create environment
        x = np.linspace(0, 50, 500)
        y = np.linspace(0, 50, 500)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=10.0)

        # Find center bin
        center_point = np.array([25.0, 25.0])
        distances = np.linalg.norm(env.bin_centers - center_point, axis=1)
        center_bin = np.argmin(distances)

        # Create field in center only
        firing_rate = np.zeros(env.n_bins)
        firing_rate[center_bin] = 5.0

        score = border_score(firing_rate, env)

        # Central field should have low/negative score
        assert score < 0.0, f"Expected score < 0 for central field, got {score}"

    def test_border_score_returns_float(self) -> None:
        """Test that border_score returns a float."""
        from neurospatial.encoding.border import border_score

        x = np.linspace(0, 30, 300)
        y = np.linspace(0, 30, 300)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=10.0)

        firing_rate = np.zeros(env.n_bins)
        firing_rate[env.boundary_bins] = 5.0

        score = border_score(firing_rate, env)

        assert isinstance(score, (float, np.floating))

    def test_compute_region_coverage_basic(self) -> None:
        """Test compute_region_coverage with wall regions."""
        from shapely.geometry import box

        from neurospatial.encoding.border import compute_region_coverage

        # Create environment
        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        # Add wall regions
        env.regions.add("north", polygon=box(0, 30, 40, 40))
        env.regions.add("south", polygon=box(0, 0, 40, 10))

        # Create field along north wall
        firing_rate = np.zeros(env.n_bins)
        north_bins = np.where(env.mask_for_region("north"))[0]
        firing_rate[north_bins] = 5.0
        field_bins = np.where(firing_rate > 0)[0]

        coverage = compute_region_coverage(field_bins, env)

        # North should have high coverage, south should be low
        assert coverage["north"] > 0.8
        assert coverage["south"] < 0.1

    def test_compute_region_coverage_returns_dict(self) -> None:
        """Test that compute_region_coverage returns a dict."""
        from shapely.geometry import box

        from neurospatial.encoding.border import compute_region_coverage

        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        env.regions.add("test", polygon=box(0, 0, 20, 20))

        field_bins = np.array([0, 1, 2])

        coverage = compute_region_coverage(field_bins, env)

        assert isinstance(coverage, dict)

    def test_compute_region_coverage_specific_regions(self) -> None:
        """Test compute_region_coverage with specific regions parameter."""
        from shapely.geometry import box

        from neurospatial.encoding.border import compute_region_coverage

        x = np.linspace(0, 40, 400)
        y = np.linspace(0, 40, 400)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        env = Environment.from_samples(positions, bin_size=4.0)

        env.regions.add("region1", polygon=box(0, 0, 20, 20))
        env.regions.add("region2", polygon=box(20, 20, 40, 40))
        env.regions.add("region3", polygon=box(0, 20, 20, 40))

        field_bins = np.array([0, 1, 2])

        # Only compute for region1 and region2
        coverage = compute_region_coverage(
            field_bins, env, regions=["region1", "region2"]
        )

        # Should only return specified regions
        assert set(coverage.keys()) == {"region1", "region2"}
