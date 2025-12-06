"""Tests for encoding/spatial_view.py module.

Tests verify:
1. All symbols are importable from encoding.spatial_view
2. All symbols are importable from encoding package
3. Module structure is correct
4. Re-exports match original implementations
5. Basic functionality works
"""

from __future__ import annotations

import numpy as np
import pytest

# =============================================================================
# Import Tests - encoding.spatial_view
# =============================================================================


class TestSpatialViewImports:
    """Test imports from encoding.spatial_view module."""

    def test_import_SpatialViewFieldResult(self):
        """SpatialViewFieldResult importable from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult

        assert SpatialViewFieldResult is not None

    def test_import_compute_spatial_view_field(self):
        """compute_spatial_view_field importable from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import compute_spatial_view_field

        assert callable(compute_spatial_view_field)

    def test_import_SpatialViewMetrics(self):
        """SpatialViewMetrics importable from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import SpatialViewMetrics

        assert SpatialViewMetrics is not None

    def test_import_spatial_view_cell_metrics(self):
        """spatial_view_cell_metrics importable from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import spatial_view_cell_metrics

        assert callable(spatial_view_cell_metrics)

    def test_import_is_spatial_view_cell(self):
        """is_spatial_view_cell importable from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import is_spatial_view_cell

        assert callable(is_spatial_view_cell)

    def test_import_compute_viewed_location(self):
        """compute_viewed_location re-exported from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import compute_viewed_location

        assert callable(compute_viewed_location)

    def test_import_compute_viewshed(self):
        """compute_viewshed re-exported from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import compute_viewshed

        assert callable(compute_viewshed)

    def test_import_visibility_occupancy(self):
        """visibility_occupancy re-exported from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import visibility_occupancy

        assert callable(visibility_occupancy)

    def test_import_FieldOfView(self):
        """FieldOfView re-exported from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import FieldOfView

        assert FieldOfView is not None


# =============================================================================
# Import Tests - encoding package
# =============================================================================


class TestEncodingPackageImports:
    """Test imports from encoding/__init__.py."""

    def test_import_SpatialViewFieldResult_from_encoding(self):
        """SpatialViewFieldResult importable from encoding."""
        from neurospatial.encoding import SpatialViewFieldResult

        assert SpatialViewFieldResult is not None

    def test_import_compute_spatial_view_field_from_encoding(self):
        """compute_spatial_view_field importable from encoding."""
        from neurospatial.encoding import compute_spatial_view_field

        assert callable(compute_spatial_view_field)

    def test_import_SpatialViewMetrics_from_encoding(self):
        """SpatialViewMetrics importable from encoding."""
        from neurospatial.encoding import SpatialViewMetrics

        assert SpatialViewMetrics is not None

    def test_import_spatial_view_cell_metrics_from_encoding(self):
        """spatial_view_cell_metrics importable from encoding."""
        from neurospatial.encoding import spatial_view_cell_metrics

        assert callable(spatial_view_cell_metrics)

    def test_import_is_spatial_view_cell_from_encoding(self):
        """is_spatial_view_cell importable from encoding."""
        from neurospatial.encoding import is_spatial_view_cell

        assert callable(is_spatial_view_cell)

    def test_import_compute_viewed_location_from_encoding(self):
        """compute_viewed_location importable from encoding."""
        from neurospatial.encoding import compute_viewed_location

        assert callable(compute_viewed_location)

    def test_import_compute_viewshed_from_encoding(self):
        """compute_viewshed importable from encoding."""
        from neurospatial.encoding import compute_viewshed

        assert callable(compute_viewshed)

    def test_import_visibility_occupancy_from_encoding(self):
        """visibility_occupancy importable from encoding."""
        from neurospatial.encoding import visibility_occupancy

        assert callable(visibility_occupancy)

    def test_import_FieldOfView_from_encoding(self):
        """FieldOfView importable from encoding."""
        from neurospatial.encoding import FieldOfView

        assert FieldOfView is not None


# =============================================================================
# Module Structure Tests
# =============================================================================


class TestModuleStructure:
    """Test module structure and __all__ exports."""

    def test_module_has_all(self):
        """encoding.spatial_view has __all__ defined."""
        from neurospatial.encoding import spatial_view

        assert hasattr(spatial_view, "__all__")
        assert isinstance(spatial_view.__all__, list)

    def test_all_contains_expected_exports(self):
        """encoding.spatial_view.__all__ contains expected exports."""
        from neurospatial.encoding import spatial_view

        expected = [
            # From spatial_view_field.py
            "SpatialViewFieldResult",
            "compute_spatial_view_field",
            # From metrics/spatial_view_cells.py
            "SpatialViewMetrics",
            "spatial_view_cell_metrics",
            "is_spatial_view_cell",
            # Re-exports from ops.visibility
            "compute_viewed_location",
            "compute_viewshed",
            "visibility_occupancy",
            "FieldOfView",
        ]
        for name in expected:
            assert name in spatial_view.__all__, f"{name} not in __all__"

    def test_all_entries_are_importable(self):
        """All entries in __all__ are importable."""
        from neurospatial.encoding import spatial_view

        for name in spatial_view.__all__:
            assert hasattr(spatial_view, name), f"{name} not accessible"


# =============================================================================
# Re-export Identity Tests
# =============================================================================


class TestReexportIdentity:
    """Test that re-exports are identical to original implementations."""

    def test_SpatialViewFieldResult_same_as_original(self):
        """SpatialViewFieldResult is same class as original."""
        from neurospatial.encoding.spatial_view import SpatialViewFieldResult
        from neurospatial.spatial_view_field import (
            SpatialViewFieldResult as Original,
        )

        assert SpatialViewFieldResult is Original

    def test_compute_spatial_view_field_same_as_original(self):
        """compute_spatial_view_field is same function as original."""
        from neurospatial.encoding.spatial_view import compute_spatial_view_field
        from neurospatial.spatial_view_field import (
            compute_spatial_view_field as original,
        )

        assert compute_spatial_view_field is original

    def test_SpatialViewMetrics_same_as_original(self):
        """SpatialViewMetrics is same class as original."""
        from neurospatial.encoding.spatial_view import SpatialViewMetrics
        from neurospatial.metrics.spatial_view_cells import (
            SpatialViewMetrics as Original,
        )

        assert SpatialViewMetrics is Original

    def test_spatial_view_cell_metrics_same_as_original(self):
        """spatial_view_cell_metrics is same function as original."""
        from neurospatial.encoding.spatial_view import spatial_view_cell_metrics
        from neurospatial.metrics.spatial_view_cells import (
            spatial_view_cell_metrics as original,
        )

        assert spatial_view_cell_metrics is original

    def test_is_spatial_view_cell_same_as_original(self):
        """is_spatial_view_cell is same function as original."""
        from neurospatial.encoding.spatial_view import is_spatial_view_cell
        from neurospatial.metrics.spatial_view_cells import (
            is_spatial_view_cell as original,
        )

        assert is_spatial_view_cell is original

    def test_compute_viewed_location_same_as_ops(self):
        """compute_viewed_location is same as ops.visibility."""
        from neurospatial.encoding.spatial_view import compute_viewed_location
        from neurospatial.ops.visibility import compute_viewed_location as original

        assert compute_viewed_location is original

    def test_compute_viewshed_same_as_ops(self):
        """compute_viewshed is same as ops.visibility."""
        from neurospatial.encoding.spatial_view import compute_viewshed
        from neurospatial.ops.visibility import compute_viewshed as original

        assert compute_viewshed is original

    def test_visibility_occupancy_same_as_ops(self):
        """visibility_occupancy is same as ops.visibility."""
        from neurospatial.encoding.spatial_view import visibility_occupancy
        from neurospatial.ops.visibility import visibility_occupancy as original

        assert visibility_occupancy is original

    def test_FieldOfView_same_as_ops(self):
        """FieldOfView is same as ops.visibility."""
        from neurospatial.encoding.spatial_view import FieldOfView
        from neurospatial.ops.visibility import FieldOfView as Original

        assert FieldOfView is Original


# =============================================================================
# Functionality Tests
# =============================================================================


@pytest.fixture
def spatial_view_env():
    """Create environment for spatial view tests."""
    from neurospatial import Environment

    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (500, 2))
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def spatial_view_data(spatial_view_env):
    """Create test data for spatial view field computation."""
    rng = np.random.default_rng(42)
    n_time = 500
    times = np.linspace(0, 50, n_time)
    trajectory = rng.uniform(20, 80, (n_time, 2))
    headings = rng.uniform(-np.pi, np.pi, n_time)
    spike_times = rng.choice(times, size=50, replace=False)
    return {
        "env": spatial_view_env,
        "spike_times": spike_times,
        "times": times,
        "positions": trajectory,
        "headings": headings,
    }


class TestSpatialViewFieldFunctionality:
    """Test spatial view field computation functionality."""

    def test_compute_spatial_view_field_basic(self, spatial_view_data):
        """compute_spatial_view_field produces valid result."""
        from neurospatial.encoding.spatial_view import (
            SpatialViewFieldResult,
            compute_spatial_view_field,
        )

        result = compute_spatial_view_field(
            spatial_view_data["env"],
            spatial_view_data["spike_times"],
            spatial_view_data["times"],
            spatial_view_data["positions"],
            spatial_view_data["headings"],
            view_distance=10.0,
        )

        assert isinstance(result, SpatialViewFieldResult)
        assert len(result.field) == spatial_view_data["env"].n_bins
        assert len(result.view_occupancy) == spatial_view_data["env"].n_bins

    def test_spatial_view_cell_metrics_basic(self, spatial_view_data):
        """spatial_view_cell_metrics produces valid metrics."""
        from neurospatial.encoding.spatial_view import (
            SpatialViewMetrics,
            spatial_view_cell_metrics,
        )

        metrics = spatial_view_cell_metrics(
            spatial_view_data["env"],
            spatial_view_data["spike_times"],
            spatial_view_data["times"],
            spatial_view_data["positions"],
            spatial_view_data["headings"],
        )

        assert isinstance(metrics, SpatialViewMetrics)
        assert isinstance(metrics.view_field_skaggs_info, float)
        assert isinstance(metrics.place_field_skaggs_info, float)
        assert isinstance(metrics.is_spatial_view_cell, bool)

    def test_is_spatial_view_cell_returns_bool(self, spatial_view_data):
        """is_spatial_view_cell returns boolean."""
        from neurospatial.encoding.spatial_view import is_spatial_view_cell

        result = is_spatial_view_cell(
            spatial_view_data["env"],
            spatial_view_data["spike_times"],
            spatial_view_data["times"],
            spatial_view_data["positions"],
            spatial_view_data["headings"],
        )

        assert isinstance(result, bool)


class TestVisibilityReexportFunctionality:
    """Test re-exported visibility functions work correctly."""

    def test_compute_viewed_location_basic(self):
        """compute_viewed_location works from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import compute_viewed_location

        positions = np.array([[50.0, 50.0], [60.0, 60.0]])
        headings = np.array([0.0, np.pi / 4])

        result = compute_viewed_location(
            positions,
            headings,
            method="fixed_distance",
            view_distance=10.0,
        )

        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))

    def test_FieldOfView_preset(self):
        """FieldOfView presets work from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import FieldOfView

        rat_fov = FieldOfView.rat()
        assert rat_fov is not None
        assert hasattr(rat_fov, "left_angle")
        assert hasattr(rat_fov, "right_angle")

    def test_compute_viewshed_basic(self, spatial_view_env):
        """compute_viewshed works from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import compute_viewshed

        position = np.array([50.0, 50.0])
        heading = 0.0

        result = compute_viewshed(
            spatial_view_env,
            position,
            heading,
        )

        assert hasattr(result, "visible_bins")
        assert len(result.visible_bins) <= spatial_view_env.n_bins

    def test_visibility_occupancy_basic(self, spatial_view_env):
        """visibility_occupancy works from encoding.spatial_view."""
        from neurospatial.encoding.spatial_view import visibility_occupancy

        rng = np.random.default_rng(42)
        n_time = 100
        positions = rng.uniform(20, 80, (n_time, 2))
        headings = rng.uniform(-np.pi, np.pi, n_time)
        times = np.linspace(0, 10, n_time)

        result = visibility_occupancy(
            spatial_view_env,
            positions,
            headings,
            times=times,
        )

        assert len(result) == spatial_view_env.n_bins
        assert np.all(result >= 0)
