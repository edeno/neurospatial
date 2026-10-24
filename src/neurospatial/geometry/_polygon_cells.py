"""Owned planar cell records and the locator that clips its parent's bins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from neurospatial._exceptions import _format_error
from neurospatial.geometry._affine import _check_homogeneous
from neurospatial.geometry._arena import mapped_geometry
from neurospatial.geometry._arrays import (
    ReadOnlyState,
    active_order,
    axis_ranges,
    boolean_selection,
)
from neurospatial.geometry._locators import (
    Locator,
    LookupIndexes,
    SubsetLocator,
    TransformedLocator,
    _lookup_index,
    locate_within,
)


def _error(what: str, fix: str) -> ValueError:
    return ValueError(
        _format_error(
            what,
            why="Why: exact cell geometry must remain consistent with point lookup and physical measures.",
            fix=fix,
        )
    )


def _validate_polygons(polygons: Sequence[Polygon]) -> None:
    """``_validate_polygon`` for every polygon, screened in vectorized calls.

    The first polygon that fails is checked again one at a time, so the error
    is the one ``_validate_polygon`` gives.
    """
    geometries = _geometry_array(polygons)
    coordinates, owner = shapely.get_coordinates(geometries, return_index=True)
    nonfinite = np.bincount(
        owner,
        weights=~np.all(np.isfinite(coordinates), axis=1),
        minlength=len(geometries),
    )
    area = shapely.area(geometries)
    bad = (
        shapely.is_empty(geometries)
        | shapely.has_z(geometries)
        | (nonfinite > 0)
        | ~shapely.is_valid(geometries)
        | ~np.isfinite(area)
        | (area <= 0)
    )
    for index in np.flatnonzero(bad)[:1]:
        _validate_polygon(polygons[index], f"cells[{index}]")


def _validate_polygon(geometry: Any, name: str) -> None:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise _error(
            f"{name} must be a Polygon or MultiPolygon.",
            "provide valid finite Cartesian 2-D polygon geometry",
        )
    if (
        geometry.is_empty
        or geometry.has_z
        or not np.all(np.isfinite(shapely.get_coordinates(geometry)))
    ):
        raise _error(
            f"{name} must be nonempty, finite and two-dimensional.",
            "remove non-finite or extra coordinates and provide a nonempty polygon",
        )
    if not geometry.is_valid or not np.isfinite(geometry.area) or geometry.area <= 0:
        raise _error(
            f"{name} is invalid: {shapely.is_valid_reason(geometry)}.",
            "correct the polygon without implicit geometry repair",
        )


@dataclass(frozen=True, eq=False)
class PolygonCells:
    """Ordered immutable connected polygons, including their interior rings."""

    polygons: tuple[Polygon, ...]

    def __post_init__(self) -> None:
        polygons = tuple(self.polygons)
        for index, polygon in enumerate(polygons):
            if not isinstance(polygon, Polygon):
                raise _error(
                    f"cells[{index}] must be a connected Polygon.",
                    "split disconnected positive-area pieces into separate bins",
                )
        _validate_polygons(polygons)
        _check_disjoint(polygons)
        object.__setattr__(self, "polygons", polygons)

    def extent(self) -> tuple[tuple[float, float], ...]:
        """Bound the polygons, one ``(low, high)`` per axis.

        Returns
        -------
        tuple of (float, float)
            The lowest and highest coordinate of the cells on each axis.
        """
        bounds = np.array([polygon.bounds for polygon in self.polygons])
        return axis_ranges(bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0))

    def subset(self, keep: ArrayLike) -> PolygonCells:
        """Keep the selected polygons in their original order.

        Parameters
        ----------
        keep : array_like of bool, shape (n_bins,)
            True for each polygon to keep.

        Raises
        ------
        ValueError
            If ``keep`` is not a boolean mask with one entry per polygon.
        """
        mask = boolean_selection(keep, len(self.polygons))
        return PolygonCells(
            tuple(
                polygon
                for polygon, selected in zip(self.polygons, mask, strict=True)
                if selected
            )
        )


@dataclass(frozen=True, eq=False)
class ClipMapping(ReadOnlyState):
    """Owned child-to-immediate-parent indices, shape ``(n_child_bins,)``."""

    parent_bin_indices: NDArray[np.intp]
    n_parent_bins: int

    def __post_init__(self) -> None:
        raw = np.asarray(self.parent_bin_indices)
        if (
            not isinstance(self.n_parent_bins, (int, np.integer))
            or isinstance(self.n_parent_bins, (bool, np.bool_))
            or self.n_parent_bins < 0
            or self.n_parent_bins > np.iinfo(np.intp).max
        ):
            raise ValueError(
                "n_parent_bins must be a representable nonnegative integer."
            )
        if (
            raw.ndim != 1
            or (raw.size and raw.dtype.kind not in "iu")
            or np.any(raw < 0)
            or np.any(raw >= self.n_parent_bins)
        ):
            raise ValueError(
                "parent_bin_indices must be one-dimensional integer indices in [0, n_parent_bins)."
            )
        indices = np.array(raw, dtype=np.intp, copy=True)
        indices.setflags(write=False)
        object.__setattr__(self, "parent_bin_indices", indices)
        object.__setattr__(self, "n_parent_bins", int(self.n_parent_bins))


@dataclass(frozen=True, eq=False)
class ClippedLocator:
    """Apply immediate-parent ownership, then closed child-polygon membership."""

    base: Locator
    cells: PolygonCells
    mapping: ClipMapping

    def __post_init__(self) -> None:
        if len(self.cells.polygons) != len(self.mapping.parent_bin_indices):
            raise ValueError("cells and mapping must contain the same number of bins.")

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Locate ``(n_points, 2)`` coordinates without resurrecting dropped owners."""
        query = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return self.locate_within(query, np.zeros(query.shape), indexes)

    def locate_within(
        self,
        points: NDArray[np.float64],
        error: NDArray[np.float64],
        indexes: LookupIndexes | None = None,
        *,
        half_open: bool = False,
    ) -> NDArray[np.intp]:
        """Locate points known to within ``error``.

        The parent's owner is found within the error, and a clipped cell of
        that parent within the error of the point holds it; of several, the
        lowest-numbered.
        """
        query = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError(f"Expected points of shape (M, 2), got {query.shape}.")
        result = np.full(len(query), -1, dtype=np.intp)
        rows = np.flatnonzero(np.all(np.isfinite(query), axis=1))
        if not len(rows) or not self.cells.polygons:
            return result
        parents = locate_within(
            self.base,
            query[rows],
            np.asarray(error)[rows],
            indexes,
            half_open=half_open,
        )
        if np.any(parents >= self.mapping.n_parent_bins):
            raise ValueError("base locator returned a parent outside n_parent_bins.")
        tree = _lookup_index(self, indexes, lambda: STRtree(self.cells.polygons))
        radius = np.hypot.reduce(np.asarray(error)[rows], axis=1)
        if np.any(radius > 0):
            point_ids, child_ids = tree.query(
                shapely.points(query[rows]), predicate="dwithin", distance=radius
            )
        else:
            point_ids, child_ids = tree.query(
                shapely.points(query[rows]), predicate="covered_by"
            )
        matching = self.mapping.parent_bin_indices[child_ids] == parents[point_ids]
        local = np.full(len(rows), len(self.cells.polygons), dtype=np.intp)
        np.minimum.at(local, point_ids[matching], child_ids[matching])
        local[local == len(self.cells.polygons)] = -1
        result[rows] = local
        return result


def clip_mapping_of(locator: Locator) -> ClipMapping | None:
    """The latest clip's mapping in the current bin order, or None before clipping.

    The clipped locator owns the mapping because lookup needs it; later
    transforms keep the bin order and later subsets select from it.
    """
    if isinstance(locator, ClippedLocator):
        return locator.mapping
    if isinstance(locator, TransformedLocator):
        return clip_mapping_of(locator.base)
    if isinstance(locator, SubsetLocator):
        mapping = clip_mapping_of(locator.base)
        if mapping is None:
            return None
        child = locator.parent_to_child
        if child.shape != mapping.parent_bin_indices.shape:
            raise ValueError(
                "A subset's parent count disagrees with the clip it selects from."
            )
        kept = active_order(child)
        return ClipMapping(mapping.parent_bin_indices[kept], mapping.n_parent_bins)
    return None


def _geometry_array(polygons: Sequence[Polygon]) -> NDArray[np.object_]:
    geometries = np.empty(len(polygons), dtype=object)
    geometries[:] = polygons
    return geometries


def _check_disjoint(polygons: tuple[Polygon, ...]) -> None:
    """Refuse cells whose interiors overlap, naming the first such pair.

    Every candidate pair from the spatial index is intersected in one
    vectorized call; a pair overlaps when its intersection has positive area.
    Two cells meet only inside the overlap of their bounding boxes, so a pair
    whose boxes overlap in no area (cells sharing a grid edge or corner)
    cannot overlap and is not intersected.
    """
    geometries = _geometry_array(polygons)
    first, second = STRtree(geometries).query(geometries)
    bounds = shapely.bounds(geometries)
    low = np.maximum(bounds[first, :2], bounds[second, :2])
    high = np.minimum(bounds[first, 2:], bounds[second, 2:])
    candidate = (second > first) & np.all(high > low, axis=1)
    first, second = first[candidate], second[candidate]
    overlap = np.flatnonzero(
        shapely.area(shapely.intersection(geometries[first], geometries[second])) > 0
    )
    if overlap.size:
        pair = overlap[np.argmin(first[overlap])]
        raise _error(
            f"cells[{first[pair]}] and cells[{second[pair]}] overlap in their "
            "interiors.",
            "provide a nonoverlapping cell partition",
        )


def transform_polygon_cells(
    cells: PolygonCells, matrix: NDArray[np.float64]
) -> PolygonCells:
    """Map shared vertex coordinates with one homogeneous Cartesian affine map."""
    matrix = np.asarray(matrix, dtype=np.float64)
    _check_homogeneous(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("polygon transform must have shape (3, 3).")
    # Node both sides before mapping: rounding an extra vertex in only one
    # ring can bend a formerly shared straight edge on only that side.
    tree = STRtree(cells.polygons)
    noded: dict[int, Polygon] = {}
    for atom in polygonize(unary_union([p.boundary for p in cells.polygons])):
        owners = tree.query(atom.representative_point(), predicate="within")
        if len(owners):
            owner = int(owners[0])
            if owner in noded or not atom.equals(cells.polygons[owner]):
                raise _error(
                    "Shared cell boundaries could not be noded without changing support.",
                    "provide a resolvable nonoverlapping polygon partition",
                )
            noded[owner] = atom
    if len(noded) != len(cells.polygons):
        raise _error(
            "Noding lost a cell's interior.",
            "rescale coordinates or provide resolvable polygon geometry",
        )
    geometries = np.asarray([noded[i] for i in range(len(noded))], dtype=object)
    return PolygonCells(tuple(mapped_geometry(geometries, matrix)))


def _polygon_coordinate_errors(polygon: Polygon | MultiPolygon) -> NDArray[np.float64]:
    """Bound each coordinate axis without hiding a thin, well-resolved axis."""
    coordinates = shapely.get_coordinates(polygon)
    # As for native grid edges, a feature needs at least 10,000 times the
    # coordinate uncertainty. Normalize each axis independently: a long y
    # edge does not make an exactly stored tiny x width unresolvable.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        uncertainty = 4 * np.max(
            np.spacing(np.abs(coordinates)), axis=0
        ) + 64 * np.finfo(float).eps * np.ptp(coordinates, axis=0)
        normalized = (coordinates - coordinates[0]) / uncertainty
    # Overlay can insert consecutive vertices separated by less than the
    # coordinate uncertainty at an otherwise well-resolved corner. Those
    # redundant vertices are not the cell's width. Simplify only this temporary
    # normalized clearance probe; retain the original geometry and every cell.
    if (
        not np.all(np.isfinite(normalized))
        or shapely.set_coordinates(polygon, normalized)
        .simplify(1.0, preserve_topology=True)
        .minimum_clearance
        < 1e4
    ):
        raise ValueError(
            "Polygon features are not resolvable at the stored coordinate precision. "
            "Rescale or recenter coordinates before clipping or transforming."
        )
    return np.asarray(uncertainty, dtype=np.float64)
