"""Owned geometry for deterministic point-to-bin lookup."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, cast

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from neurospatial._constants import KDTREE_LEAF_SIZE
from neurospatial.geometry._arrays import ReadOnlyState, get_centers
from neurospatial.geometry._hex_lookup import points_to_hex_bin_ind
from neurospatial.geometry._regular_lookup import (
    outline_cells,
    points_to_regular_grid_bin_ind,
)

if TYPE_CHECKING:
    from neurospatial.geometry._grid_frame import FramedGrid

_T = TypeVar("_T")


class LookupIndexes(Protocol):
    """Where locators keep the search structures they build on first use.

    A KD-tree, triangle index or polygon tree depends only on its locator's
    own immutable data. An environment passes its analysis cache, so each
    structure is built once per environment, shared by its relabelled copies
    and released by ``clear_cache()``.
    """

    def lookup_index(self, locator: object, build: Callable[[], _T]) -> _T:
        """Return the structure ``build`` makes for ``locator``, building it once."""
        ...


class Locator(Protocol):
    """Find the bin that contains each query point."""

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return bin indices for ``(n_points, n_dims)`` coordinates, or -1.

        Non-finite rows return -1; a wrong number of columns raises ValueError.
        The input is not modified. ``indexes`` keeps the search structures a
        locator builds (an environment passes its analysis cache); without
        it they are built for this call only.
        """
        ...


class CallIndexes:
    """Search structures kept for one computation over a binning.

    Repeated ``locate`` calls on a binning that belongs to no environment,
    such as the copy ``env.binning`` returns, share one of these so that each
    structure is built once for the computation.
    """

    __slots__ = ("_built",)

    def __init__(self) -> None:
        self._built: dict[object, Any] = {}

    def lookup_index(self, locator: object, build: Callable[[], _T]) -> _T:
        """Return the structure ``build`` makes for ``locator``, building it once."""
        if locator not in self._built:
            self._built[locator] = build()
        return cast("_T", self._built[locator])


def _lookup_index(
    locator: object, indexes: LookupIndexes | None, build: Callable[[], _T]
) -> _T:
    """Fetch ``locator``'s structure from ``indexes``, or build it for one call."""
    return build() if indexes is None else indexes.lookup_index(locator, build)


class AffineMatrix(Protocol):
    """Structural homogeneous affine matrix, without an ops dependency."""

    @property
    def A(self) -> NDArray[np.float64]: ...  # noqa: N802


def _check_homogeneous(matrix: NDArray[np.float64]) -> None:
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix[-1], np.r_[np.zeros(len(matrix) - 1), 1.0])
    ):
        raise ValueError(
            "inverse/affine matrix must be finite, square and homogeneous."
        )
    if np.linalg.slogdet(matrix[:-1, :-1])[0] == 0:
        raise ValueError("inverse/affine matrix must be invertible.")


def _owned_array(array: Any, dtype: Any) -> NDArray[Any]:
    """Own an array without freezing the caller's buffer."""
    owned = np.array(array, dtype=dtype, copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, eq=False)
class RectilinearLocator(ReadOnlyState):
    """Locate points in the rectangular cells of an active grid.

    Cells are half-open, ``[low, high)`` on every axis, so a face shared by
    two active cells has one owner. The outline of the active cells is
    closed: a point on a face with no active cell beyond it (the arena's
    edge, or an arm tip) belongs to the active cell it bounds, as the last
    bin of ``np.histogram`` includes its right edge. At an outline corner
    the cell whose upper faces the point touches on the fewest axes wins,
    the lowest axes first.

    Parameters
    ----------
    edges : tuple of ndarray
        Increasing cell edges along each coordinate axis.
    shape : tuple of int
        Number of full-grid cells along each coordinate axis.
    active_mask : ndarray of bool, shape ``shape``, or None
        Active cells. None treats the entire grid as active.
    flat_to_active : ndarray of intp, shape (prod(shape),), or None
        Active-bin index of each full-grid cell, or -1 when inactive.
    """

    edges: tuple[NDArray[np.float64], ...]
    shape: tuple[int, ...]
    active_mask: NDArray[np.bool_] | None
    flat_to_active: NDArray[np.intp] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "edges", tuple(_owned_array(edge, np.float64) for edge in self.edges)
        )
        if self.active_mask is not None:
            object.__setattr__(
                self, "active_mask", _owned_array(self.active_mask, np.bool_)
            )
        if self.flat_to_active is not None:
            object.__setattr__(
                self, "flat_to_active", _owned_array(self.flat_to_active, np.intp)
            )

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return active-bin indices, or -1 outside the grid or active cells."""
        query_points = np.atleast_2d(points)
        if query_points.ndim != 2:
            raise ValueError(
                f"Expected a 2D points array, got shape {query_points.shape}."
            )
        n_dims = query_points.shape[1]
        if n_dims != len(self.edges) or n_dims != len(self.shape):
            raise ValueError(
                f"Dimensionality mismatch: points have {n_dims} dimension(s), but "
                f"grid_edges has {len(self.edges)} and grid_shape has "
                f"{len(self.shape)}. Points must match the grid's dimensionality."
            )
        located = self.locate_half_open(query_points)
        unowned = np.flatnonzero(
            (located < 0) & np.all(np.isfinite(query_points), axis=1)
        )
        if unowned.size:
            located[unowned] = self._outline_owner(query_points[unowned])
        return located

    def locate_half_open(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return active-bin indices by half-open cells alone, or -1.

        A combined environment asks every child for this owner before any
        child claims a point on its outline, so a face shared by two children
        keeps the half-open owner.
        """
        return points_to_regular_grid_bin_ind(
            points=np.atleast_2d(points),
            grid_edges=self.edges,
            grid_shape=self.shape,
            active_mask=self.active_mask,
            inverse_map=self.flat_to_active,
        ).astype(np.intp, copy=False)

    def _outline_owner(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Active bin each finite, unowned point bounds on the outline, or -1."""
        flat = outline_cells(points, self.edges, self.shape, self.active_mask)
        owner = np.full(len(points), -1, dtype=np.intp)
        found = flat >= 0
        owner[found] = self._active_index(flat[found])
        return owner

    def _active_index(self, flat: NDArray[np.intp]) -> NDArray[np.intp]:
        """Active-bin index of full-grid cells, or -1 for inactive cells."""
        if self.flat_to_active is not None:
            return self.flat_to_active[flat]
        if self.active_mask is None:
            return flat
        mapping = np.full(self.active_mask.size, -1, dtype=np.intp)
        mapping[self.active_mask.ravel()] = np.arange(
            np.count_nonzero(self.active_mask)
        )
        return mapping[flat]

    def active_centers(self) -> NDArray[np.float64]:
        """Midpoints of the active cells, in active-bin (row-major) order.

        Returns
        -------
        ndarray of float64, shape (n_active, n_dims)
            The cell centers implied by ``edges``. Stored bin centers of a
            rectilinear binning are these values, so centers and edges cannot
            disagree.
        """
        mask = (
            np.ones(self.shape, dtype=bool)
            if self.active_mask is None
            else self.active_mask
        )
        coordinates = np.unravel_index(np.flatnonzero(mask), self.shape)
        return np.column_stack(
            [
                get_centers(edge)[index]
                for edge, index in zip(self.edges, coordinates, strict=True)
            ]
        )

    def subset(self, keep: NDArray[np.bool_]) -> RectilinearLocator:
        """Keep whole active cells in their original row-major order."""
        mask = (
            np.ones(self.shape, dtype=bool)
            if self.active_mask is None
            else self.active_mask.copy()
        )
        keep = np.asarray(keep)
        active = np.flatnonzero(mask)
        if keep.dtype.kind != "b" or keep.shape != (len(active),):
            raise ValueError(f"keep must be a boolean array of shape ({len(active)},).")
        mask.ravel()[active] = keep
        mapping = np.full(mask.size, -1, dtype=np.intp)
        mapping[mask.ravel()] = np.arange(np.count_nonzero(mask))
        return RectilinearLocator(self.edges, self.shape, mask, mapping)

    def transformed(self, transform: AffineMatrix) -> RectilinearLocator | None:
        """Map positive diagonal scales and translations without changing owners."""
        matrix = np.asarray(transform.A, dtype=np.float64)
        _check_homogeneous(matrix)
        if matrix.shape != (len(self.edges) + 1, len(self.edges) + 1):
            raise ValueError("affine matrix dimensionality must match grid edges.")
        linear = matrix[:-1, :-1]
        diagonal = np.diag(linear)
        if not np.array_equal(linear, np.diag(diagonal)) or np.any(diagonal <= 0):
            return None
        edges = tuple(
            diagonal[axis] * edge + matrix[axis, -1]
            for axis, edge in enumerate(self.edges)
        )
        return RectilinearLocator(
            edges, self.shape, self.active_mask, self.flat_to_active
        )


@dataclass(frozen=True, eq=False)
class SubsetLocator(ReadOnlyState):
    """Map parent bins to child bins, keeping dropped and outside points outside."""

    base: Locator
    parent_to_child: NDArray[np.intp]

    def __post_init__(self) -> None:
        raw = np.asarray(self.parent_to_child)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or np.any(raw < -1):
            raise ValueError(
                "parent_to_child must be one-dimensional integer indices >= -1."
            )
        selected = raw[raw >= 0]
        if not np.array_equal(np.sort(selected), np.arange(len(selected))):
            raise ValueError(
                "parent_to_child must number child bins exactly once from zero."
            )
        object.__setattr__(self, "parent_to_child", _owned_array(raw, np.intp))

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return child indices after checking the parent's outside sentinel."""
        parent = self.base.locate(points, indexes)
        out = np.full(parent.shape, -1, dtype=np.intp)
        inside = parent >= 0
        out[inside] = self.parent_to_child[parent[inside]]
        return out


@dataclass(frozen=True, eq=False)
class TransformedLocator(ReadOnlyState):
    """Find bins by applying an owned inverse homogeneous affine matrix.

    In a binning with a ``grid_frame`` (a rotated or sheared grid), the
    binning gives this locator its frame. Points are then assigned by the
    frame's half-open source cells, with points on a cell face within the
    map's roundoff snapped onto it, the rule linear time allocation uses, so
    a point on a face shared by two cells belongs to the cell above it.
    Without a frame, the parent locator owns the inverse-mapped point.

    Parameters
    ----------
    base : Locator
        Locator in the source coordinates.
    inverse : ndarray of float64, shape (n_dims + 1, n_dims + 1)
        Homogeneous map from current to source coordinates.
    frame : FramedGrid or None, optional
        The grid frame of the binning this locator belongs to; a ``Binning``
        with a ``grid_frame`` sets it, so it is rarely passed directly.
    """

    base: Locator
    inverse: NDArray[np.float64]
    frame: FramedGrid | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        matrix = _owned_array(self.inverse, np.float64)
        _check_homogeneous(matrix)
        object.__setattr__(self, "inverse", matrix)

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Undo the affine map before using the parent's boundary ownership."""
        query = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n_dims = len(self.inverse) - 1
        if query.ndim != 2 or query.shape[1] != n_dims:
            raise ValueError(
                f"Expected points of shape (M, {n_dims}), got {query.shape}."
            )
        if self.frame is not None:
            return self.frame._locate(
                query, operation="Bin lookup on a rotated or sheared grid"
            )
        original = np.full(query.shape, np.nan, dtype=np.float64)
        finite = np.all(np.isfinite(query), axis=1)
        original[finite] = (
            query[finite] @ self.inverse[:-1, :-1].T + self.inverse[:-1, -1]
        )
        return self.base.locate(original, indexes)


@dataclass(frozen=True, eq=False)
class HexLocator(ReadOnlyState):
    """Locate points in a pointy-top hexagonal lattice.

    Parameters
    ----------
    offset_x, offset_y : float
        Center coordinates of the conceptual lattice's first hexagon.
    radius : float
        Distance from a hexagon's center to a vertex.
    centers_shape : tuple of int
        Number of hexagon rows and columns in the conceptual lattice.
    flat_to_active : ndarray of intp, shape (prod(centers_shape),)
        Active-bin index of each full-grid cell, or -1 when inactive.
    """

    offset_x: float
    offset_y: float
    radius: float
    centers_shape: tuple[int, int]
    flat_to_active: NDArray[np.intp]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "flat_to_active", _owned_array(self.flat_to_active, np.intp)
        )

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return active-bin indices, or -1 outside active hexagonal cells."""
        query_points = np.asarray(points)
        if query_points.ndim != 2 or query_points.shape[1] != 2:
            raise ValueError(
                f"Expected points of shape (M, 2), got {query_points.shape}."
            )
        original_flat_indices = points_to_hex_bin_ind(
            points=query_points,
            grid_offset_x=self.offset_x,
            grid_offset_y=self.offset_y,
            hex_radius=self.radius,
            centers_shape=self.centers_shape,
        )
        indices = np.full(len(original_flat_indices), -1, dtype=np.intp)
        valid = original_flat_indices != -1
        indices[valid] = self.flat_to_active[original_flat_indices[valid]]
        return indices


# Barycentric tolerance of scipy's Delaunay.find_simplex: a point belongs to a
# triangle when every barycentric coordinate lies in [-eps, 1 + eps].
_BARYCENTRIC_EPS = 100 * np.finfo(np.float64).eps
# scipy treats a simplex whose transform has reciprocal condition number below
# this as degenerate and never returns it; Qhull emits such zero-area slivers
# between nearly collinear boundary samples.
_DEGENERATE_RCOND = 1000 * np.finfo(np.float64).eps
# Overlap depth, relative to triangle size, accepted as coordinate rounding.
_OVERLAP_RTOL = 1e-9
# Query points per batch of candidate-triangle tests.
_QUERY_CHUNK = 65_536


class _TriangleIndex(NamedTuple):
    """Bounding-box tree and barycentric transforms of the locatable triangles."""

    tree: shapely.STRtree
    triangles: NDArray[np.intp]
    inverse: NDArray[np.float64]
    origin: NDArray[np.float64]


def _separated(
    triangles: NDArray[np.float64],
    others: NDArray[np.float64],
    depth: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Test whether an edge line of each triangle separates it from another.

    Parameters
    ----------
    triangles, others : ndarray of float64, shape (n_pairs, 3, 2)
        Counterclockwise triangle vertices, paired row by row.
    depth : ndarray of float64, shape (n_pairs,)
        Distance ``others`` may reach inside an edge and still count as outside.

    Returns
    -------
    ndarray of bool, shape (n_pairs,)
        True where every vertex of ``others`` lies on or outside one edge of
        ``triangles``. Two convex polygons have disjoint interiors exactly when
        an edge of one of them separates them in this way.
    """
    separated = np.zeros(len(triangles), dtype=bool)
    for start in range(3):
        origin = triangles[:, start]
        edge = triangles[:, (start + 1) % 3] - origin
        offset = others - origin[:, None, :]
        # Twice the signed area: |edge| times the distance left of the edge.
        cross = edge[:, None, 0] * offset[..., 1] - edge[:, None, 1] * offset[..., 0]
        limit = depth * np.hypot(edge[:, 0], edge[:, 1])
        separated |= np.all(cross <= limit[:, None], axis=1)
    return separated


def _triangle_index(
    points: NDArray[np.float64], simplices: NDArray[np.intp]
) -> _TriangleIndex:
    """Validate stored triangles and index them for point lookup.

    Parameters
    ----------
    points : ndarray of float64, shape (n_points, 2)
        Triangle vertex coordinates.
    simplices : ndarray of intp, shape (n_simplices, 3)
        Vertex indices of each stored triangle.

    Returns
    -------
    _TriangleIndex
        Lookup structures for the non-degenerate triangles, keyed by their
        stored triangle numbers.

    Raises
    ------
    ValueError
        If the triangles do not form a valid triangulation of the points:
        malformed arrays, out-of-range or repeated vertex indices, or
        overlapping triangles.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Mesh points must have shape (n_points, 2), got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Mesh points must be finite.")
    if simplices.ndim != 2 or simplices.shape[1] != 3:
        raise ValueError(
            f"Mesh simplices must have shape (n_simplices, 3), got {simplices.shape}."
        )
    out_of_range = np.flatnonzero(
        np.any((simplices < 0) | (simplices >= len(points)), axis=1)
    )
    if out_of_range.size:
        raise ValueError(
            f"Mesh triangles {out_of_range[:5].tolist()} use a vertex index "
            f"outside 0..{len(points) - 1}; the stored mesh is corrupt."
        )
    ordered = np.sort(simplices, axis=1)
    repeated = np.flatnonzero(np.any(ordered[:, 1:] == ordered[:, :-1], axis=1))
    if repeated.size:
        raise ValueError(
            f"Mesh triangles {repeated[:5].tolist()} repeat a vertex; the stored "
            "mesh is corrupt."
        )

    vertices = points[simplices]
    # Columns of the 2x2 transform scipy inverts: v0 - v2 and v1 - v2.
    first = vertices[:, 0] - vertices[:, 2]
    second = vertices[:, 1] - vertices[:, 2]
    det = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    norm_1 = np.maximum(np.abs(first).sum(axis=1), np.abs(second).sum(axis=1))
    norm_inf = np.maximum(
        np.abs(first[:, 0]) + np.abs(second[:, 0]),
        np.abs(first[:, 1]) + np.abs(second[:, 1]),
    )
    # For a 2x2 matrix, |det| / (||A||_1 ||A||_inf) is the 1-norm reciprocal
    # condition number that scipy estimates.
    locatable = np.abs(det) >= _DEGENERATE_RCOND * norm_1 * norm_inf
    triangles = np.flatnonzero(locatable).astype(np.intp)

    det = det[triangles]
    inverse = (
        np.stack(
            (
                np.stack((second[triangles, 1], -second[triangles, 0]), axis=-1),
                np.stack((-first[triangles, 1], first[triangles, 0]), axis=-1),
            ),
            axis=1,
        )
        / det[:, None, None]
    )
    lower = vertices[triangles].min(axis=1)
    upper = vertices[triangles].max(axis=1)
    span = np.sum(upper - lower, axis=1)
    # Barycentric eps admits points up to eps * altitude outside a triangle.
    pad = 2 * _BARYCENTRIC_EPS * span[:, None]
    boxes = shapely.box(*(lower - pad).T, *(upper + pad).T)
    tree = shapely.STRtree(boxes)

    # Only triangles with intersecting bounding boxes can overlap.
    left, right = tree.query(boxes)
    distinct = left < right
    left, right = left[distinct], right[distinct]
    counterclockwise = vertices[triangles]
    clockwise = det < 0
    counterclockwise[clockwise] = counterclockwise[clockwise][:, ::-1]
    depth = _OVERLAP_RTOL * np.maximum(span[left], span[right])
    overlap = ~(
        _separated(counterclockwise[left], counterclockwise[right], depth)
        | _separated(counterclockwise[right], counterclockwise[left], depth)
    )
    if np.any(overlap):
        pairs = np.column_stack((triangles[left], triangles[right]))[overlap]
        raise ValueError(
            f"Mesh triangles overlap (stored triangle pairs {pairs[:5].tolist()}); "
            "a point could belong to more than one bin, so the stored mesh is "
            "corrupt."
        )
    return _TriangleIndex(tree, triangles, inverse, vertices[triangles, 2])


@dataclass(frozen=True, eq=False)
class MeshLocator(ReadOnlyState):
    """Locate points in a fixed triangulation clipped to a polygon.

    The stored triangles define the bins, so lookup never depends on how this
    platform's Qhull would triangulate the points. A point on an edge or vertex
    shared by several triangles belongs to the lowest-numbered one, and
    zero-area triangles contain no points.

    Parameters
    ----------
    points : ndarray of float64, shape (n_points, 2)
        Triangle vertex coordinates.
    simplices : ndarray of intp, shape (n_simplices, 3)
        Stored triangulation, including inactive triangles.
    simplex_to_active : ndarray of intp, shape (n_simplices,)
        Active-bin index of each original triangle, or -1 when inactive.
    boundary : shapely.geometry.Polygon or None
        Boundary applied to query points, including holes and concave notches.
    """

    points: NDArray[np.float64]
    simplices: NDArray[np.intp]
    simplex_to_active: NDArray[np.intp]
    boundary: Polygon | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _owned_array(self.points, np.float64))
        object.__setattr__(self, "simplices", _owned_array(self.simplices, np.intp))
        object.__setattr__(
            self, "simplex_to_active", _owned_array(self.simplex_to_active, np.intp)
        )

    def _find_triangles(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None
    ) -> NDArray[np.intp]:
        """Return the lowest-numbered stored triangle containing each point, or -1."""
        # Building the index also validates the stored triangles.
        index = _lookup_index(
            self, indexes, lambda: _triangle_index(self.points, self.simplices)
        )
        found = np.full(len(points), len(self.simplices), dtype=np.intp)
        finite = np.flatnonzero(np.all(np.isfinite(points), axis=1))
        # Chunks bound the memory of the (point, candidate triangle) pairs.
        for start in range(0, len(finite), _QUERY_CHUNK):
            rows = finite[start : start + _QUERY_CHUNK]
            query, entry = index.tree.query(shapely.points(points[rows]))
            query = rows[query]
            offset = points[query] - index.origin[entry]
            coords = np.einsum("kij,kj->ki", index.inverse[entry], offset)
            coords = np.column_stack((coords, 1.0 - coords[:, 0] - coords[:, 1]))
            inside = np.all(
                (coords >= -_BARYCENTRIC_EPS) & (coords <= 1.0 + _BARYCENTRIC_EPS),
                axis=1,
            )
            np.minimum.at(found, query[inside], index.triangles[entry[inside]])
        found[found == len(self.simplices)] = -1
        return found

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return active triangle indices for points inside the boundary."""
        pts2d = np.atleast_2d(points).astype(np.float64, copy=False)
        if pts2d.ndim != 2 or pts2d.shape[1] != 2:
            raise ValueError(f"Expected points of shape (M, 2), got {pts2d.shape}.")

        orig_simplices = self._find_triangles(pts2d, indexes)
        active_triangle_idxs = np.full(orig_simplices.shape, -1, dtype=np.intp)
        valid_mask = orig_simplices != -1
        if np.any(valid_mask):
            found_orig = orig_simplices[valid_mask]
            active_triangle_idxs[valid_mask] = self.simplex_to_active[found_orig]
            if self.boundary is not None:
                xcoords = pts2d[valid_mask, 0]
                ycoords = pts2d[valid_mask, 1]
                on_or_inside = shapely.contains_xy(self.boundary, xcoords, ycoords)
                active_triangle_idxs[np.flatnonzero(valid_mask)[~on_or_inside]] = -1
        return active_triangle_idxs.astype(np.intp, copy=False)


@dataclass(frozen=True, eq=False)
class NearestBinLocator(ReadOnlyState):
    """Find the nearest bin center, optionally within a maximum distance.

    Parameters
    ----------
    centers : ndarray of float64, shape (n_centers, n_dims)
        Coordinates used for nearest-neighbor lookup.
    max_distance : float or None, default=None
        Maximum accepted distance. None preserves unlimited nearest lookup.
    bin_indices : ndarray of intp, shape (n_centers,), or None
        Bin index for each stored center. None numbers centers consecutively.
    """

    centers: NDArray[np.float64]
    max_distance: float | None = None
    bin_indices: NDArray[np.intp] | None = None

    def __post_init__(self) -> None:
        centers = _owned_array(self.centers, np.float64)
        if centers.ndim != 2:
            raise ValueError(
                "centers must be a 2D array with shape (n_centers, n_dims)."
            )
        bin_indices = (
            np.arange(len(centers), dtype=np.intp)
            if self.bin_indices is None
            else _owned_array(self.bin_indices, np.intp)
        )
        if bin_indices.shape != (len(centers),):
            raise ValueError(
                f"bin_indices must have shape ({len(centers)},), "
                f"got {bin_indices.shape}."
            )
        bin_indices.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "bin_indices", bin_indices)

    def kdtree(self, indexes: LookupIndexes | None = None) -> cKDTree:
        """Return the KD-tree over ``centers``, kept in ``indexes`` when given."""
        return _lookup_index(
            self, indexes, lambda: cKDTree(self.centers, leafsize=KDTREE_LEAF_SIZE)
        )

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return nearest bin indices, mapping non-finite rows to -1."""
        query_points = np.atleast_2d(points)
        n_query_points = query_points.shape[0]
        if query_points.ndim != 2:
            raise ValueError(
                f"Expected a 2D points array, got shape {query_points.shape}."
            )
        expected_dims = int(self.centers.shape[1])
        if query_points.shape[1] != expected_dims:
            raise ValueError(
                f"Query points have {query_points.shape[1]} dimension(s) but this "
                f"environment bins in {expected_dims}-D; pass points of shape "
                f"(n_points, {expected_dims}). Note: ``to_linear()`` output and "
                f"other 1-D linearized positions are NOT bin coordinates -- bin "
                f"with the original {expected_dims}-D positions instead."
            )
        final_bin_indices = np.full(n_query_points, -1, dtype=np.intp)
        if len(self.centers) == 0:
            return final_bin_indices
        finite_rows = np.all(np.isfinite(query_points), axis=1)
        if not np.any(finite_rows):
            return final_bin_indices
        tree = self.kdtree(indexes)
        try:
            distances, kdtree_internal_indices = tree.query(query_points[finite_rows])
        except Exception as exc:
            warnings.warn(
                f"KDTree query failed: {exc}. Returning -1 for all points.",
                RuntimeWarning,
                stacklevel=2,
            )
            return final_bin_indices
        max_valid_kdtree_idx = int(tree.data.shape[0] - 1)
        kdtree_internal_indices = np.clip(
            kdtree_internal_indices, 0, max_valid_kdtree_idx
        )
        assert self.bin_indices is not None
        mapped = self.bin_indices[kdtree_internal_indices]
        if self.max_distance is not None:
            mapped = np.where(distances <= self.max_distance, mapped, -1)
        final_bin_indices[finite_rows] = mapped
        return np.asarray(final_bin_indices, dtype=np.intp)
