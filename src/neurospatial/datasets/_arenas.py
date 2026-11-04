"""Stated open arenas placed on the tracked samples, for datasets whose files state no placement.

A paper states an arena's shape and size; the file often states no origin or
rotation. These helpers place the stated shape where the samples are: a
rectangle on their minimum rotated rectangle, a circle on their minimum
bounding circle, after leaving out the few samples farthest from their
median, so a glitch does not move the arena. The positions are not changed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial.datasets._recording import tracked_mask

__all__ = [
    "PLACEMENT_TRIM",
    "fitted_circle",
    "fitted_rectangle",
    "shape_fill",
    "trimmed",
]

#: Share of samples, per tail and axis, left out before placing an arena.
PLACEMENT_TRIM = 0.001


def trimmed(positions: NDArray[np.float64], trim: float) -> NDArray[np.float64]:
    """Tracked samples less the ``2 * n_dims * trim`` share farthest from their median.

    Parameters
    ----------
    positions : ndarray, shape (n_time, n_dims)
    trim : float
        Per tail and axis.

    Returns
    -------
    ndarray, shape (n_kept, n_dims)

    Examples
    --------
    >>> points = np.vstack([np.zeros((999, 2)), [[100.0, 100.0]]])
    >>> trimmed(points, 0.0001).shape
    (999, 2)
    """
    tracked = positions[tracked_mask(positions)]
    radius = np.linalg.norm(tracked - np.median(tracked, axis=0), axis=1)
    share = 2 * tracked.shape[1] * trim
    kept: NDArray[np.float64] = tracked[radius <= np.quantile(radius, 1 - share)]
    return kept


def fitted_rectangle(
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """The minimum rotated rectangle around the tracked samples, less far outliers.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)

    Returns
    -------
    centre : ndarray, shape (2,)
    angle : float
        Rotation from the positions' x axis, radians in [-π/4, π/4).
    sides : ndarray, shape (2,)
        The rectangle's two side lengths, the one along ``angle`` first.

    Examples
    --------
    >>> x, y = np.meshgrid(np.linspace(0, 4, 41), np.linspace(0, 2, 21))
    >>> centre, angle, sides = fitted_rectangle(np.column_stack([x.ravel(), y.ravel()]))
    >>> np.round(centre, 6).tolist(), round(angle, 6), np.round(sides, 6).tolist()
    ([2.0, 1.0], 0.0, [4.0, 2.0])
    """
    from shapely.geometry import MultiPoint

    points = MultiPoint(trimmed(positions, PLACEMENT_TRIM))
    # GEOS divides by an edge's zero extent for axis-aligned hull edges; the
    # rectangle it returns is still the minimum one.
    with np.errstate(divide="ignore", invalid="ignore"):
        fitted = points.minimum_rotated_rectangle
    corners = np.asarray(fitted.exterior.coords)[:4]
    sides = np.diff(np.vstack([corners, corners[:1]]), axis=0)[:2]
    lengths = np.linalg.norm(sides, axis=1)
    # The side nearer the x axis sets the rotation.
    first = int(np.argmax(np.abs(sides[:, 0]) / lengths))
    angle = float(np.arctan2(sides[first, 1], sides[first, 0]))
    angle = float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    centre = np.asarray(fitted.centroid.coords[0], dtype=np.float64)
    return centre, angle, lengths[[first, 1 - first]]


def fitted_circle(positions: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """The minimum bounding circle of the tracked samples, less far outliers.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)

    Returns
    -------
    centre : ndarray, shape (2,)
    diameter : float

    Examples
    --------
    >>> angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    >>> ring = np.column_stack([3 + 5 * np.cos(angles), -1 + 5 * np.sin(angles)])
    >>> centre, diameter = fitted_circle(ring)
    >>> np.round(centre, 3).tolist(), round(diameter, 3)
    ([3.0, -1.0], 10.0)
    """
    import shapely
    from shapely.geometry import MultiPoint

    points = MultiPoint(trimmed(positions, PLACEMENT_TRIM))
    circle = shapely.minimum_bounding_circle(points)
    centre = np.asarray(circle.centroid.coords[0], dtype=np.float64)
    return centre, 2 * float(shapely.minimum_bounding_radius(points))


def shape_fill(positions: NDArray[np.float64]) -> tuple[float, float]:
    """How much of their minimum rectangle and of their minimum circle the samples fill.

    The convex hull's area over each shape's area: samples that fill a
    square give a rectangle share near 1 and a circle share near 2/π (0.64);
    samples that fill a circle give π/4 (0.79) and about 1.

    Parameters
    ----------
    positions : ndarray, shape (n_time, 2)

    Returns
    -------
    rectangle, circle : float
        The hull's share of the minimum rotated rectangle and of the minimum
        bounding circle, after leaving out far outliers.
    """
    import shapely
    from shapely.geometry import MultiPoint

    points = MultiPoint(trimmed(positions, PLACEMENT_TRIM))
    hull = points.convex_hull.area
    with np.errstate(divide="ignore", invalid="ignore"):  # as in fitted_rectangle
        rectangle = points.minimum_rotated_rectangle.area
    return (
        float(hull / rectangle),
        float(hull / shapely.minimum_bounding_circle(points).area),
    )
