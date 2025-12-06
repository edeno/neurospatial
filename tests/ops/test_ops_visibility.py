"""Tests for neurospatial.ops.visibility module.

These tests verify that the visibility module is correctly exported from the
new ops/ location after the package reorganization.
"""

from __future__ import annotations

import numpy as np


class TestVisibilityImportsFromOps:
    """Test that visibility classes and functions are importable from ops.visibility."""

    def test_import_field_of_view_from_ops(self) -> None:
        """FieldOfView should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import FieldOfView

        # Verify it's a class
        assert isinstance(FieldOfView, type)

    def test_import_viewshed_result_from_ops(self) -> None:
        """ViewshedResult should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import ViewshedResult

        # Verify it's a class
        assert isinstance(ViewshedResult, type)

    def test_import_compute_viewed_location_from_ops(self) -> None:
        """compute_viewed_location should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import compute_viewed_location

        assert callable(compute_viewed_location)

    def test_import_compute_viewshed_from_ops(self) -> None:
        """compute_viewshed should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import compute_viewshed

        assert callable(compute_viewshed)

    def test_import_compute_view_field_from_ops(self) -> None:
        """compute_view_field should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import compute_view_field

        assert callable(compute_view_field)

    def test_import_compute_viewshed_trajectory_from_ops(self) -> None:
        """compute_viewshed_trajectory should be importable from ops.visibility."""
        from neurospatial.ops.visibility import compute_viewshed_trajectory

        assert callable(compute_viewshed_trajectory)

    def test_import_visibility_occupancy_from_ops(self) -> None:
        """visibility_occupancy should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import visibility_occupancy

        assert callable(visibility_occupancy)

    def test_import_visible_cues_from_ops(self) -> None:
        """visible_cues should be importable from neurospatial.ops.visibility."""
        from neurospatial.ops.visibility import visible_cues

        assert callable(visible_cues)


class TestVisibilityExportsFromOpsPackage:
    """Test that visibility exports are available from neurospatial.ops."""

    def test_field_of_view_from_ops_package(self) -> None:
        """FieldOfView should be importable from neurospatial.ops."""
        from neurospatial.ops import FieldOfView

        assert isinstance(FieldOfView, type)

    def test_viewshed_result_from_ops_package(self) -> None:
        """ViewshedResult should be importable from neurospatial.ops."""
        from neurospatial.ops import ViewshedResult

        assert isinstance(ViewshedResult, type)

    def test_compute_viewed_location_from_ops_package(self) -> None:
        """compute_viewed_location should be importable from neurospatial.ops."""
        from neurospatial.ops import compute_viewed_location

        assert callable(compute_viewed_location)

    def test_compute_viewshed_from_ops_package(self) -> None:
        """compute_viewshed should be importable from neurospatial.ops."""
        from neurospatial.ops import compute_viewshed

        assert callable(compute_viewshed)

    def test_compute_view_field_from_ops_package(self) -> None:
        """compute_view_field should be importable from neurospatial.ops."""
        from neurospatial.ops import compute_view_field

        assert callable(compute_view_field)

    def test_visibility_occupancy_from_ops_package(self) -> None:
        """visibility_occupancy should be importable from neurospatial.ops."""
        from neurospatial.ops import visibility_occupancy

        assert callable(visibility_occupancy)

    def test_visible_cues_from_ops_package(self) -> None:
        """visible_cues should be importable from neurospatial.ops."""
        from neurospatial.ops import visible_cues

        assert callable(visible_cues)

    def test_compute_viewshed_trajectory_from_ops_package(self) -> None:
        """compute_viewshed_trajectory should be importable from neurospatial.ops."""
        from neurospatial.ops import compute_viewshed_trajectory

        assert callable(compute_viewshed_trajectory)


class TestFieldOfViewFromOps:
    """Test FieldOfView class functionality from ops location."""

    def test_symmetric_fov(self) -> None:
        """Test symmetric field of view creation."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        assert np.isclose(fov.total_angle_degrees, 180.0)

    def test_rat_fov(self) -> None:
        """Test rat field of view preset."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.rat()
        assert 290 < fov.total_angle_degrees < 340

    def test_primate_fov(self) -> None:
        """Test primate field of view preset."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.primate()
        assert 150 < fov.total_angle_degrees < 200

    def test_contains_angle(self) -> None:
        """Test FOV angle containment check."""
        from neurospatial.ops.visibility import FieldOfView

        fov = FieldOfView.symmetric(half_angle=np.pi / 2)
        assert fov.contains_angle(0.0)  # Ahead
        assert not fov.contains_angle(np.pi)  # Behind


class TestComputeViewedLocationFromOps:
    """Test compute_viewed_location function from ops location."""

    def test_fixed_distance_method(self) -> None:
        """Test fixed distance viewed location computation."""
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])  # Facing East

        viewed = compute_viewed_location(
            positions, headings, method="fixed_distance", view_distance=10.0
        )
        assert np.allclose(viewed[0], [10.0, 0.0])

    def test_fixed_distance_multiple_positions(self) -> None:
        """Test fixed distance with multiple positions."""
        from neurospatial.ops.visibility import compute_viewed_location

        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        headings = np.array([0.0, np.pi / 2])  # East, North

        viewed = compute_viewed_location(
            positions, headings, method="fixed_distance", view_distance=5.0
        )
        assert viewed.shape == (2, 2)
        assert np.allclose(viewed[0], [5.0, 0.0])
        assert np.allclose(viewed[1], [10.0, 15.0], atol=1e-10)


class TestViewshedResultFromOps:
    """Test ViewshedResult class from ops location."""

    def test_n_visible_bins(self) -> None:
        """Test n_visible_bins property."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([0, 1, 2]),
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(10),
        )
        assert result.n_visible_bins == 3

    def test_visibility_fraction(self) -> None:
        """Test visibility_fraction property."""
        from neurospatial.ops.visibility import ViewshedResult

        result = ViewshedResult(
            visible_bins=np.array([0, 1, 2, 3, 4]),
            visible_cues=np.array([]),
            cue_distances=np.array([]),
            cue_bearings=np.array([]),
            occlusion_map=np.zeros(10),
            _total_bins=10,
        )
        assert result.visibility_fraction == 0.5
