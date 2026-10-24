"""Owned bin shapes and immutable outlines for spatial drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.geometry._arrays import ReadOnlyState, axis_ranges, boolean_selection


@dataclass(frozen=True, eq=False)
class HexagonShapes(ReadOnlyState):
    """Regular hexagons with centers ``(n_bins, 2)``, radius and orientation.

    Parameters
    ----------
    centers : ndarray, shape (n_bins, 2)
        Hexagon centers.
    radius : float
        Distance from a center to each corner, in length units.
    orientation : float
        Rotation in radians, counterclockwise. At 0 the hexagons are
        pointy-top: one corner points straight along +y.
    """

    centers: NDArray[np.float64]
    radius: float
    orientation: float

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        if (
            centers.ndim != 2
            or centers.shape[1] != 2
            or not np.all(np.isfinite(centers))
        ):
            raise ValueError(
                "HexagonShapes centers must be finite with shape (n_bins, 2), "
                f"got shape {centers.shape}."
            )
        radius, orientation = float(self.radius), float(self.orientation)
        if not (np.isfinite(radius) and radius > 0) or not np.isfinite(orientation):
            raise ValueError(
                "HexagonShapes radius must be finite and positive and its "
                f"orientation finite, got radius={self.radius!r}, "
                f"orientation={self.orientation!r}."
            )
        centers.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "orientation", orientation)

    def corner_offsets(self) -> NDArray[np.float64]:
        """Corners relative to a hexagon's center, counterclockwise, shape (6, 2).

        The first corner points along ``orientation + π/2`` radians, as
        Matplotlib's ``RegularPolygon`` draws it.
        """
        angles = np.arange(6) * np.pi / 3 + np.pi / 2 + self.orientation
        offsets: NDArray[np.float64] = self.radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        return offsets

    def extent(self) -> tuple[tuple[float, float], ...]:
        """Bound the hexagons, corners included, one ``(low, high)`` per axis.

        Returns
        -------
        tuple of (float, float)
            The lowest and highest coordinate of the cells on each axis.
        """
        offsets = self.corner_offsets()
        return axis_ranges(
            self.centers.min(axis=0) + offsets.min(axis=0),
            self.centers.max(axis=0) + offsets.max(axis=0),
        )

    def subset(self, keep: ArrayLike) -> HexagonShapes:
        """Keep the selected hexagons in their original order.

        Parameters
        ----------
        keep : array_like of bool, shape (n_bins,)
            True for each hexagon to keep.

        Raises
        ------
        ValueError
            If ``keep`` is not a boolean mask with one entry per hexagon.
        """
        mask = boolean_selection(keep, len(self.centers))
        return HexagonShapes(self.centers[mask], self.radius, self.orientation)


@dataclass(frozen=True, eq=False)
class TriangleShapes(ReadOnlyState):
    """Mesh points ``(n_points, 2)`` and active triangles ``(n_bins, 3)``."""

    points: NDArray[np.float64]
    triangles: NDArray[np.intp]

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=np.float64, copy=True)
        raw = np.asarray(self.triangles)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError(
                "TriangleShapes points must be finite with shape (n_points, 2), "
                f"got shape {points.shape}."
            )
        if (
            raw.ndim != 2
            or raw.shape[1] != 3
            or (raw.size and raw.dtype.kind not in "iu")
            or np.any((raw < 0) | (raw >= len(points)))
        ):
            raise ValueError(
                "TriangleShapes triangles must be integer indices into the "
                f"{len(points)} points with shape (n_bins, 3)."
            )
        triangles = np.array(raw, dtype=np.intp, copy=True)
        points.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangles", triangles)

    def extent(self) -> tuple[tuple[float, float], ...]:
        """Bound the triangles, one ``(low, high)`` per axis.

        Only the vertices of the retained triangles count: a subset keeps
        every mesh point.

        Returns
        -------
        tuple of (float, float)
            The lowest and highest coordinate of the cells on each axis.
        """
        vertices = self.points[np.unique(self.triangles)]
        return axis_ranges(vertices.min(axis=0), vertices.max(axis=0))

    def subset(self, keep: ArrayLike) -> TriangleShapes:
        """Keep the selected triangles in their original order.

        Parameters
        ----------
        keep : array_like of bool, shape (n_bins,)
            True for each triangle to keep.

        Raises
        ------
        ValueError
            If ``keep`` is not a boolean mask with one entry per triangle.
        """
        mask = boolean_selection(keep, len(self.triangles))
        return TriangleShapes(self.points, self.triangles[mask])


@dataclass(frozen=True, eq=False)
class Outline:
    """An immutable polygon or multipolygon outline."""

    geometry: Polygon | MultiPolygon
