"""Tests for regions operations module."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon

from neurospatial.regions import Region, Regions
from neurospatial.regions.ops import (
    _get_points_in_single_region_mask,
    _prepare_points,
    points_in_any_region,
    regions_containing_points,
)


class TestPreparePoints:
    """Tests for _prepare_points helper function."""

    def test_basic_2d_array(self):
        """Test with basic 2D numpy array."""
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = _prepare_points(pts)

        assert_array_equal(result, pts)
        assert result.dtype == np.float64

    def test_single_point_1d(self):
        """Test single point as 1D array."""
        pts = np.array([5.0, 10.0])

        result = _prepare_points(pts)

        assert result.shape == (1, 2)
        assert_array_equal(result, [[5.0, 10.0]])

    def test_single_point_2d(self):
        """Test single point as 2D array."""
        pts = np.array([[5.0, 10.0]])

        result = _prepare_points(pts)

        assert result.shape == (1, 2)
        assert_array_equal(result, [[5.0, 10.0]])

    def test_list_of_lists(self):
        """Test with list of lists."""
        pts = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        result = _prepare_points(pts)

        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert_array_equal(result, expected)

    def test_empty_array(self):
        """Test with empty array."""
        pts = np.empty((0, 2))

        result = _prepare_points(pts)

        assert result.shape == (0, 2)

    def test_invalid_shape_1d_wrong_size(self):
        """Test error on 1D array with wrong size."""
        pts = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Single point must have 2 coordinates"):
            _prepare_points(pts)

    def test_invalid_shape_3d(self):
        """Test error on 3D array."""
        pts = np.array([[[1, 2], [3, 4]]])

        with pytest.raises(ValueError, match="must be of shape \\(N, 2\\)"):
            _prepare_points(pts)

    def test_invalid_shape_wrong_columns(self):
        """Test error on wrong number of columns."""
        pts = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="must be of shape \\(N, 2\\)"):
            _prepare_points(pts)

    def test_with_transform(self):
        """Test with transform applied."""
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])

        # Simple transform: divide by 10
        class SimpleTransform:
            def __call__(self, pts):
                return pts / 10

        transform = SimpleTransform()
        result = _prepare_points(pts, transform=transform)

        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_array_equal(result, expected)

    def test_unconvertible_input(self):
        """Test error on input that can't be converted."""
        pts = "not valid input"

        with pytest.raises(ValueError, match="Could not convert points"):
            _prepare_points(pts)


class TestGetPointsInSingleRegionMask:
    """Tests for _get_points_in_single_region_mask helper function."""

    def test_point_region_inside(self):
        """Test points inside tolerance of point region."""
        region = Region(name="center", data=np.array([5.0, 5.0]), kind="point")
        pts = np.array([[5.0, 5.0], [5.0 + 1e-9, 5.0], [10.0, 10.0]])

        mask = _get_points_in_single_region_mask(
            pts, region, point_tolerance=1e-8, include_boundary=True
        )

        assert mask[0]  # Exact match
        assert mask[1]  # Within tolerance
        assert not mask[2]  # Outside tolerance

    def test_point_region_outside(self):
        """Test points outside tolerance of point region."""
        region = Region(name="center", data=np.array([5.0, 5.0]), kind="point")
        pts = np.array([[5.1, 5.1], [6.0, 6.0]])

        mask = _get_points_in_single_region_mask(
            pts, region, point_tolerance=1e-8, include_boundary=True
        )

        assert np.all(~mask)

    def test_polygon_region_inside(self):
        """Test points inside polygon region."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=poly, kind="polygon")
        pts = np.array([[5.0, 5.0], [15.0, 15.0], [0.0, 0.0]])

        mask = _get_points_in_single_region_mask(
            pts, region, point_tolerance=1e-8, include_boundary=True
        )

        assert mask[0]  # Inside
        assert not mask[1]  # Outside
        assert mask[2]  # On boundary (with include_boundary=True)

    def test_polygon_region_exclude_boundary(self):
        """Test boundary points excluded when include_boundary=False."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=poly, kind="polygon")
        pts = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 5.0]])

        mask = _get_points_in_single_region_mask(
            pts, region, point_tolerance=1e-8, include_boundary=False
        )

        assert mask[0]  # Inside
        assert not mask[1]  # On boundary (excluded)
        assert not mask[2]  # On boundary (excluded)

    def test_empty_points_array(self):
        """Test with empty points array."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        region = Region(name="box", data=poly, kind="polygon")
        pts = np.empty((0, 2))

        mask = _get_points_in_single_region_mask(
            pts, region, point_tolerance=1e-8, include_boundary=True
        )

        assert mask.shape == (0,)
        assert mask.dtype == bool


class TestPointsInAnyRegion:
    """Tests for points_in_any_region function."""

    def test_single_region_polygon(self):
        """Test with single polygon region."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        pts = np.array([[5.0, 5.0], [15.0, 15.0], [0.0, 0.0]])

        mask = points_in_any_region(pts, regions)

        assert mask[0]  # Inside
        assert not mask[1]  # Outside
        assert mask[2]  # On boundary

    def test_multiple_regions(self):
        """Test with multiple overlapping regions."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        pts = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])

        mask = points_in_any_region(pts, regions)

        assert mask[0]  # In box1 only
        assert mask[1]  # In both boxes (overlap)
        assert mask[2]  # In box2 only
        assert not mask[3]  # Outside both

    def test_empty_points(self):
        """Test with empty points array."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        pts = np.empty((0, 2))

        mask = points_in_any_region(pts, regions)

        assert mask.shape == (0,)

    def test_empty_regions(self):
        """Test with empty regions collection."""
        regions = Regions([])

        pts = np.array([[5.0, 5.0], [10.0, 10.0]])

        mask = points_in_any_region(pts, regions)

        assert np.all(~mask)  # All False

    def test_point_region(self):
        """Test with point region."""
        regions = Regions([Region(name="pt", data=np.array([5.0, 5.0]), kind="point")])

        pts = np.array([[5.0, 5.0], [5.0 + 1e-9, 5.0], [10.0, 10.0]])

        mask = points_in_any_region(pts, regions, point_tolerance=1e-8)

        assert mask[0]  # Exact match
        assert mask[1]  # Within tolerance
        assert not mask[2]  # Outside tolerance

    def test_with_transform(self):
        """Test with coordinate transform."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        # Points in pixel space (need to divide by 10)
        pts_px = np.array([[5.0, 5.0], [15.0, 15.0]])

        class PixelToWorld:
            def __call__(self, pts):
                return pts / 10

        mask = points_in_any_region(pts_px, regions, transform=PixelToWorld())

        assert mask[0]  # (0.5, 0.5) inside unit box
        assert not mask[1]  # (1.5, 1.5) outside unit box

    def test_early_exit_optimization(self):
        """Test early exit when all points covered."""
        # Create a large region that covers everything
        large_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        small_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        regions = Regions(
            [
                Region(name="large", data=large_poly, kind="polygon"),
                Region(name="small", data=small_poly, kind="polygon"),
            ]
        )

        pts = np.array([[5.0, 5.0], [20.0, 20.0], [50.0, 50.0]])

        mask = points_in_any_region(pts, regions)

        # All points should be inside the large region
        assert np.all(mask)


class TestRegionsContainingPoints:
    """Tests for regions_containing_points function."""

    def test_return_list_format(self):
        """Test returning list of lists of region names."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        pts = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])

        result = regions_containing_points(pts, regions, return_dataframe=False)

        assert len(result) == 4
        assert result[0] == ["box1"]  # In box1 only
        assert set(result[1]) == {"box1", "box2"}  # In both
        assert result[2] == ["box2"]  # In box2 only
        assert result[3] == []  # Outside both

    def test_return_dataframe_format(self):
        """Test returning DataFrame format."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        pts = np.array([[2.0, 2.0], [7.0, 7.0], [12.0, 12.0], [20.0, 20.0]])

        df = regions_containing_points(pts, regions, return_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 2)
        assert list(df.columns) == ["box1", "box2"]
        assert df.loc[0, "box1"]
        assert not df.loc[0, "box2"]
        assert df.loc[1, "box1"]
        assert df.loc[1, "box2"]
        assert not df.loc[2, "box1"]
        assert df.loc[2, "box2"]
        assert not df.loc[3, "box1"]
        assert not df.loc[3, "box2"]

    def test_filter_by_region_names(self):
        """Test filtering to specific region names."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        poly3 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
                Region(name="box3", data=poly3, kind="polygon"),
            ]
        )

        pts = np.array([[7.0, 7.0], [12.0, 12.0]])

        df = regions_containing_points(
            pts, regions, region_names=["box1", "box2"], return_dataframe=True
        )

        assert df.shape == (2, 2)
        assert list(df.columns) == ["box1", "box2"]
        # box3 should not be included

    def test_empty_points_dataframe(self):
        """Test with empty points returns empty DataFrame."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        pts = np.empty((0, 2))

        df = regions_containing_points(pts, regions, return_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 0
        assert "box" in df.columns

    def test_empty_points_list(self):
        """Test with empty points returns empty list."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        pts = np.empty((0, 2))

        result = regions_containing_points(pts, regions, return_dataframe=False)

        assert result == []

    def test_no_regions_dataframe(self):
        """Test with no regions returns empty columns."""
        regions = Regions([])
        pts = np.array([[5.0, 5.0]])

        df = regions_containing_points(pts, regions, return_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 0)

    def test_region_names_not_found(self):
        """Test that nonexistent region names are silently ignored."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        regions = Regions([Region(name="box1", data=poly, kind="polygon")])

        pts = np.array([[5.0, 5.0]])

        df = regions_containing_points(
            pts,
            regions,
            region_names=["box1", "nonexistent", "also_missing"],
            return_dataframe=True,
        )

        assert df.shape == (1, 1)
        assert list(df.columns) == ["box1"]

    def test_with_point_regions(self):
        """Test with point regions."""
        regions = Regions(
            [
                Region(name="pt1", data=np.array([5.0, 5.0]), kind="point"),
                Region(name="pt2", data=np.array([10.0, 10.0]), kind="point"),
            ]
        )

        pts = np.array([[5.0, 5.0], [10.0 + 1e-9, 10.0]])

        df = regions_containing_points(
            pts, regions, return_dataframe=True, point_tolerance=1e-8
        )

        assert df.loc[0, "pt1"]
        assert not df.loc[0, "pt2"]
        assert not df.loc[1, "pt1"]
        assert df.loc[1, "pt2"]

    def test_with_transform(self):
        """Test with coordinate transform."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        regions = Regions([Region(name="box", data=poly, kind="polygon")])

        pts_px = np.array([[5.0, 5.0], [15.0, 15.0]])

        class PixelToWorld:
            def __call__(self, pts):
                return pts / 10

        df = regions_containing_points(
            pts_px, regions, transform=PixelToWorld(), return_dataframe=True
        )

        assert df.loc[0, "box"]  # (0.5, 0.5) inside
        assert not df.loc[1, "box"]  # (1.5, 1.5) outside

    def test_maintains_region_name_order(self):
        """Test that region_names order is preserved in DataFrame."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        poly3 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
                Region(name="box3", data=poly3, kind="polygon"),
            ]
        )

        pts = np.array([[5.0, 5.0]])

        # Request in specific order
        df = regions_containing_points(
            pts,
            regions,
            region_names=["box3", "box1", "box2"],
            return_dataframe=True,
        )

        assert list(df.columns) == ["box3", "box1", "box2"]

    def test_list_format_with_multiple_regions_per_point(self):
        """Test list format returns region objects in correct format."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        regions = Regions(
            [
                Region(name="box1", data=poly1, kind="polygon"),
                Region(name="box2", data=poly2, kind="polygon"),
            ]
        )

        pts = np.array([[7.0, 7.0]])  # In both regions

        result = regions_containing_points(pts, regions, return_dataframe=False)

        assert len(result) == 1
        assert len(result[0]) == 2
        # Results are region names (strings), not Region objects
        assert isinstance(result[0][0], str)
        assert set(result[0]) == {"box1", "box2"}
