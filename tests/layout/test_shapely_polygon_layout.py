"""Behavioral tests for ShapelyPolygonLayout driven directly through the engine."""

import numpy as np
import shapely
from shapely.geometry import Polygon

from neurospatial.layout.engines.shapely_polygon import (
    ShapelyPolygonLayout,
)


def test_point_inside_polygon_maps_to_its_containing_cell():
    """A point in a convex polygon maps to the unit cell that holds it."""
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    layout = ShapelyPolygonLayout()
    layout.build(polygon=poly, bin_size=1.0)

    points = np.random.default_rng(0).uniform(0.0, 10.0, (200, 2))
    result = layout.binning().locator.locate(points)
    assert np.all(result >= 0)
    np.testing.assert_array_equal(layout.bin_centers[result], np.floor(points) + 0.5)


def test_point_outside_polygon_returns_negative_one():
    """A point outside the polygon bounds maps to -1."""
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    layout = ShapelyPolygonLayout()
    layout.build(polygon=poly, bin_size=1.0)

    result = layout.binning().locator.locate(np.array([[20.0, 20.0]]))
    assert result[0] == -1


def test_point_in_polygon_hole_returns_negative_one(polygon_with_hole):
    """A point inside an interior hole maps to -1."""
    layout = ShapelyPolygonLayout()
    layout.build(polygon=polygon_with_hole, bin_size=1.0)

    # Hole spans (4, 4)-(6, 6); its center (5, 5) is not in any active bin.
    result = layout.binning().locator.locate(np.array([[5.0, 5.0]]))
    assert result[0] == -1


def test_point_on_polygon_edge():
    """A point exactly on the exterior edge.

    Pinned to current behavior: the active mask uses ``shapely.contains``,
    which excludes boundary points, but the grid locator uses grid-index
    containment. A point at (0, 5) on the left edge lands in the leftmost
    column of active cells, so it maps to a valid bin rather than -1.
    """
    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    layout = ShapelyPolygonLayout()
    layout.build(polygon=poly, bin_size=1.0)

    result = layout.binning().locator.locate(np.array([[0.0, 5.0]]))
    assert result[0] >= 0


def test_irregular_polygon_active_bins_inside(l_shaped_polygon):
    """All active bin centers of an L-shaped polygon lie inside the polygon.

    No active bin should fall into the excised upper-right quadrant.
    """
    layout = ShapelyPolygonLayout()
    layout.build(polygon=l_shaped_polygon, bin_size=1.0)

    assert layout.bin_centers.shape[0] > 0
    inside = shapely.contains(
        l_shaped_polygon,
        shapely.points(layout.bin_centers[:, 0], layout.bin_centers[:, 1]),
    )
    assert bool(np.all(inside))

    # Explicitly: no active center sits in the missing upper-right block.
    upper_right = (layout.bin_centers[:, 0] > 4.0) & (layout.bin_centers[:, 1] > 4.0)
    assert not bool(np.any(upper_right))
