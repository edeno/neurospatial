"""Immutable continuous domains, separate from bins and analysis selections.

Arena containment includes physical boundaries. A bin locator can retain its
own half-open ownership rules. An inferred computational domain does not assert
that its boundary is a measured physical wall.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arrays import ReadOnlyState, owned_grid

__all__ = [
    "Arena",
    "EgocentricSurround",
    "MaskArena",
    "OpenArena",
    "TrackArena",
    "UnknownArena",
]

_CoordinateKind = Literal["cartesian", "polar"]


@runtime_checkable
class Arena(Protocol):
    """Continuous coordinates and known domain geometry, without a binning.

    Implementations must own immutable geometry and keep containment queries
    independent of mutable caller data.
    """

    @property
    def n_dims(self) -> int: ...

    @property
    def coordinate_kind(self) -> _CoordinateKind: ...

    @property
    def domain_known(self) -> bool: ...

    @property
    def physical_footprint_known(self) -> bool: ...

    @property
    def walls(self) -> tuple[LineString, ...]: ...

    @property
    def wall_names(self) -> tuple[str, ...]:
        """The name of the wall each segment of ``walls`` belongs to."""
        ...

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test ``(n_positions, n_dims)`` native coordinates, including boundaries."""
        ...

    def as_track(self) -> TrackArena | None:
        """This arena as a track in its current coordinates, or None.

        A transformed track answers containment in its stored frame; this
        gives its centerline and width mapped into the current coordinates.
        """
        ...

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Apply a supported homogeneous affine map to Cartesian coordinates."""
        ...

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        """Return exact planar polygonal support when available."""
        ...

    def domain_bounds(self) -> NDArray[np.float64] | None:
        """Return a box enclosing the known Cartesian domain.

        Returns
        -------
        ndarray of float64, shape (2, n_dims), or None
            Lower and upper corners, or None when the domain is unknown or
            has no Cartesian box (for example a track without a width).
        """
        ...


def _error(what: str, fix: str) -> ValueError:
    return ValueError(_format_error(what, fix=fix))


def _real_array(value: Any, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise _error(
            f"{name} must contain real numeric values.", "pass real-valued arrays"
        )
    return np.array(raw, dtype=np.float64, copy=True)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise _error(f"{name} must be boolean.", "pass True or False")
    return bool(value)


def _query(
    positions: Any, n_dims: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    query = np.atleast_2d(_real_array(positions, "positions"))
    if query.ndim != 2 or query.shape[1] != n_dims:
        raise _error(
            f"Expected positions of shape (n_positions, {n_dims}), got {query.shape}.",
            "pass one coordinate column per arena dimension",
        )
    return query, np.all(np.isfinite(query), axis=1)


def _unknown() -> NotImplementedError:
    return NotImplementedError(
        _format_error(
            "Arena containment is unknown because no physical footprint was supplied.",
            fix="supply explicit arena geometry or a positive full track width",
        )
    )


def _polygon_bounds(polygon: Polygon | MultiPolygon) -> NDArray[np.float64]:
    """Lower and upper corners of a planar polygon, shape (2, 2)."""
    return np.array([polygon.bounds[:2], polygon.bounds[2:]])


def _polygon(value: Any, name: str = "polygon") -> Polygon | MultiPolygon:
    if (
        not isinstance(value, (Polygon, MultiPolygon))
        or value.is_empty
        or value.has_z
        or not value.is_valid
        or not np.all(np.isfinite(shapely.get_coordinates(value)))
        or not np.isfinite(value.area)
        or value.area <= 0
    ):
        reason = (
            shapely.is_valid_reason(value)
            if isinstance(value, shapely.Geometry)
            else type(value).__name__
        )
        raise _error(
            f"{name} must be a valid finite nonempty 2-D polygon: {reason}.",
            "provide a valid Polygon or MultiPolygon; do not repair geometry implicitly",
        )
    return value


def _affine(matrix: Any, n_dims: int) -> NDArray[np.float64]:
    owned = _real_array(matrix, "matrix")
    if (
        owned.shape != (n_dims + 1, n_dims + 1)
        or not np.all(np.isfinite(owned))
        or not np.array_equal(owned[-1], np.r_[np.zeros(n_dims), 1.0])
    ):
        raise _error(
            "matrix must be finite, affine and match the arena dimensions.",
            "pass a homogeneous matrix with last row [0, ..., 0, 1]",
        )
    with np.errstate(under="ignore"):
        sign, _ = np.linalg.slogdet(owned[:-1, :-1])
    if sign == 0:
        raise _error(
            "matrix must be invertible.",
            "use nonzero scales and an invertible linear map",
        )
    return owned


def _mapped(
    points: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(
            points @ matrix[:-1, :-1].T + matrix[:-1, -1], dtype=np.float64
        )
    if not np.all(np.isfinite(result)):
        raise _error(
            "Transformed arena coordinates exceed floating-point range.",
            "rescale the coordinate units before transforming",
        )
    return result


def _mapped_vertices(
    points: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Map ``(..., n_dims)`` vertices, each distinct vertex exactly once.

    Matrix multiplication can round equal rows differently in separate
    vectorized/tail paths, so mapping every row would let a shared vertex
    (a closed ring, a track junction) come apart even though the inputs
    are identical.
    """
    flat = points.reshape(-1, points.shape[-1])
    unique, indices = np.unique(flat, axis=0, return_inverse=True)
    return _mapped(unique, matrix)[indices.ravel()].reshape(points.shape)


def mapped_geometry(geometry: Any, matrix: NDArray[np.float64]) -> Any:
    # Cell vertices use this same array arithmetic. A different affine
    # implementation can round their common vertices differently, inventing
    # slivers where a transformed arena boundary meets its cells.
    coordinates = shapely.get_coordinates(geometry)
    return shapely.set_coordinates(geometry, _mapped_vertices(coordinates, matrix))


@dataclass(frozen=True, eq=False)
class UnknownArena:
    """Explicitly unknown continuous geometry.

    Parameters
    ----------
    n_dims : int
        Positive number of native coordinate columns.
    coordinate_kind : {'cartesian', 'polar'}, default='cartesian'
        Coordinate interpretation; polar coordinates have two columns.
    """

    n_dims: int
    coordinate_kind: _CoordinateKind = "cartesian"
    domain_known: ClassVar[bool] = False
    physical_footprint_known: ClassVar[bool] = False
    walls: ClassVar[tuple[LineString, ...]] = ()
    wall_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_dims, (bool, np.bool_))
            or not isinstance(self.n_dims, (int, np.integer))
            or self.n_dims < 1
        ):
            raise _error(
                "n_dims must be a positive integer.",
                "pass the number of coordinate columns",
            )
        if self.coordinate_kind not in ("cartesian", "polar") or (
            self.coordinate_kind == "polar" and self.n_dims != 2
        ):
            raise _error(
                "coordinate_kind must be cartesian or two-dimensional polar.",
                "use the coordinate kind and dimension of the stored bins",
            )
        object.__setattr__(self, "n_dims", int(self.n_dims))

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Refuse to assert containment without continuous geometry."""
        _query(positions, self.n_dims)
        raise _unknown()

    def as_track(self) -> TrackArena | None:
        """None: this arena is not a track."""
        return None

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Retain unknown Cartesian geometry after a valid affine map."""
        if self.coordinate_kind != "cartesian":
            raise _error(
                "Cartesian affine maps do not apply to polar coordinates.",
                "transform the original Cartesian observations instead",
            )
        _affine(matrix, self.n_dims)
        return UnknownArena(self.n_dims, self.coordinate_kind)

    def polygonal_domain(self) -> None:
        """Return no invented polygon."""
        return None

    def domain_bounds(self) -> None:
        """An unknown domain has no bounds."""
        return None


@dataclass(frozen=True, eq=False, init=False)
class OpenArena:
    """A known planar polygonal domain with separately asserted internal walls.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Valid finite support, including any holes.
    walls : mapping of str to LineString, or sequence of LineString, optional
        Explicit internal physical walls; analysis boundaries are not inferred.
        A mapping names each wall, for example ``{"divider": line}``; walls
        given as a sequence are named ``"wall_0"``, ``"wall_1"``, ... unless
        ``wall_names`` is given.
    wall_names : sequence of str, optional
        One name per wall in a sequence of walls. Several segments of one wall
        may share its name. Not allowed with a mapping of walls.
    physical_footprint_known : bool, default=True
        False for a computational domain inferred from observations.

    Attributes
    ----------
    walls : tuple of LineString
        The wall segments, in the order given.
    wall_names : tuple of str, shape (n_walls,)
        The name of the wall each segment belongs to. Errors and warnings
        report walls by these names.
    """

    polygon: Polygon | MultiPolygon
    walls: tuple[LineString, ...]
    wall_names: tuple[str, ...]
    physical_footprint_known: bool
    n_dims: ClassVar[int] = 2
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"
    domain_known: ClassVar[bool] = True

    def __init__(
        self,
        polygon: Polygon | MultiPolygon,
        *,
        walls: Mapping[str, LineString] | Sequence[LineString] = (),
        wall_names: Sequence[str] | None = None,
        physical_footprint_known: bool = True,
    ) -> None:
        if isinstance(walls, Mapping):
            if wall_names is not None:
                raise _error(
                    "wall_names cannot be combined with a mapping of walls.",
                    "name the walls with the mapping keys, or pass walls as a "
                    "sequence with wall_names",
                )
            names = tuple(walls)
            segments = tuple(walls.values())
        else:
            segments = tuple(walls)
            names = (
                tuple(f"wall_{index}" for index in range(len(segments)))
                if wall_names is None
                else tuple(wall_names)
            )
        if len(names) != len(segments):
            raise _error(
                f"wall_names has {len(names)} names for {len(segments)} walls.",
                "give one name per wall",
            )
        for name in names:
            if not isinstance(name, str) or not name:
                raise _error(
                    f"Wall names must be nonempty strings, got {name!r}.",
                    'name each wall with a string such as "divider"',
                )
        object.__setattr__(self, "polygon", polygon)
        object.__setattr__(self, "walls", segments)
        object.__setattr__(self, "wall_names", names)
        object.__setattr__(self, "physical_footprint_known", physical_footprint_known)
        self.__post_init__()

    def __post_init__(self) -> None:
        _polygon(self.polygon)
        walls = self.walls
        for index, wall in enumerate(walls):
            if (
                not isinstance(wall, LineString)
                or wall.is_empty
                or wall.has_z
                or not wall.is_valid
                or not np.all(np.isfinite(shapely.get_coordinates(wall)))
                or not np.isfinite(wall.length)
                or wall.length <= 0
            ):
                raise _error(
                    f"Wall {self.wall_names[index]!r} must be a finite nonempty "
                    "2-D LineString.",
                    "supply explicit physical wall segments",
                )
            if not self.polygon.covers(wall):
                raise _error(
                    f"Wall {self.wall_names[index]!r} lies outside the arena domain.",
                    "provide walls within the polygonal domain",
                )
        object.__setattr__(
            self,
            "physical_footprint_known",
            _boolean(self.physical_footprint_known, "physical_footprint_known"),
        )

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test closed polygon support, excluding hole interiors."""
        query, finite = _query(positions, 2)
        inside = np.zeros(len(query), dtype=np.bool_)
        inside[finite] = shapely.covers(self.polygon, shapely.points(query[finite]))
        return inside

    def as_track(self) -> TrackArena | None:
        """None: this arena is not a track."""
        return None

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Map polygon rings and explicit walls together."""
        return _TransformedArena(self, _affine(matrix, 2))

    def polygonal_domain(self) -> Polygon | MultiPolygon:
        """Return the immutable stored polygonal domain."""
        return self.polygon

    def domain_bounds(self) -> NDArray[np.float64]:
        """Bound the stored polygon."""
        return _polygon_bounds(self.polygon)


@dataclass(frozen=True, eq=False)
class MaskArena(ReadOnlyState):
    """Closed union of active Cartesian cells, without inferred physical walls.

    Parameters
    ----------
    edges : tuple of ndarray
        Finite strictly increasing per-axis edges; spacing may vary.
    active_mask : ndarray of bool
        Active cells, with shape ``tuple(len(edge) - 1 for edge in edges)``.
    physical_footprint_known : bool, default=False
        Whether this mask was explicitly asserted to be the physical footprint.
    """

    edges: tuple[NDArray[np.float64], ...]
    active_mask: NDArray[np.bool_]
    physical_footprint_known: bool = field(default=False, kw_only=True)
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"
    domain_known: ClassVar[bool] = True
    walls: ClassVar[tuple[LineString, ...]] = ()
    wall_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        edges, _, mask = owned_grid(
            self.edges, None, self.active_mask, name="MaskArena"
        )
        if not np.any(mask):
            raise _error(
                "MaskArena active_mask has no active cell.",
                "pass a boolean cell mask with at least one active cell",
            )
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(
            self,
            "physical_footprint_known",
            _boolean(self.physical_footprint_known, "physical_footprint_known"),
        )

    @property
    def n_dims(self) -> int:
        """Number of Cartesian coordinate axes."""
        return len(self.edges)

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test the closed union, including a boundary beside any active cell."""
        query, finite = _query(positions, self.n_dims)
        inside = np.zeros(len(query), dtype=np.bool_)
        selected = query[finite]
        right = np.column_stack(
            [
                np.searchsorted(edge, selected[:, axis], side="right") - 1
                for axis, edge in enumerate(self.edges)
            ]
        )
        left = np.column_stack(
            [
                np.searchsorted(edge, selected[:, axis], side="left") - 1
                for axis, edge in enumerate(self.edges)
            ]
        )
        bounds = np.asarray(self.active_mask.shape)
        valid = np.all((right >= 0) & (right < bounds), axis=1)
        accepted = np.zeros(len(selected), dtype=np.bool_)
        accepted[valid] = self.active_mask[tuple(right[valid].T)]
        for row in np.flatnonzero(~accepted & np.any(left != right, axis=1)):
            choices = [
                tuple({int(lo), int(hi)})
                for lo, hi in zip(left[row], right[row], strict=True)
            ]
            accepted[row] = any(
                all(0 <= cell[axis] < bounds[axis] for axis in range(self.n_dims))
                and self.active_mask[cell]
                for cell in itertools.product(*choices)
            )
        inside[finite] = accepted
        return inside

    def as_track(self) -> TrackArena | None:
        """None: this arena is not a track."""
        return None

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Retain masks under positive axis scaling, or store the affine map.

        A reflection keeps the map, as a grid's bins do (``FramedGrid``), so
        the arena and its cells stay in one source frame.
        """
        matrix = _affine(matrix, self.n_dims)
        diagonal = np.diag(matrix[:-1, :-1])
        if np.array_equal(matrix[:-1, :-1], np.diag(diagonal)) and np.all(diagonal > 0):
            edges = []
            for axis, edge in enumerate(self.edges):
                with np.errstate(over="ignore", invalid="ignore"):
                    edges.append(edge * diagonal[axis] + matrix[axis, -1])
            return MaskArena(
                tuple(edges),
                self.active_mask,
                physical_footprint_known=self.physical_footprint_known,
            )
        return _TransformedArena(self, matrix)

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        """Union the exact active rectangles for two-dimensional masks only."""
        if self.n_dims != 2:
            return None
        cells = [
            box(
                self.edges[0][i],
                self.edges[1][j],
                self.edges[0][i + 1],
                self.edges[1][j + 1],
            )
            for i, j in np.argwhere(self.active_mask)
        ]
        return cast("Polygon | MultiPolygon", shapely.union_all(cells))

    def domain_bounds(self) -> NDArray[np.float64]:
        """Bound the active cells (their exact union in two dimensions)."""
        polygon = self.polygonal_domain()
        if polygon is not None:
            return _polygon_bounds(polygon)
        cells = np.argwhere(self.active_mask)
        return np.array(
            [
                [
                    edge[int(low)]
                    for edge, low in zip(self.edges, cells.min(axis=0), strict=True)
                ],
                [
                    edge[int(high) + 1]
                    for edge, high in zip(self.edges, cells.max(axis=0), strict=True)
                ],
            ]
        )


@dataclass(frozen=True, eq=False)
class TrackArena(ReadOnlyState):
    """Union of exact constant-width capsules around centerline segments.

    Parameters
    ----------
    segments : ndarray, shape (n_segments, 2, n_dims)
        Finite nondegenerate centerline segments in Cartesian coordinates.
    width : float or None
        Positive full physical width. None preserves an unknown footprint,
        except on a one-dimensional track: positions there have no off-track
        direction, so the footprint is the centerline itself, ends included.
        End caps are round and junctions are unions of segment capsules.
    """

    segments: NDArray[np.float64]
    width: float | None
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"
    walls: ClassVar[tuple[LineString, ...]] = ()
    wall_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        segments = _real_array(self.segments, "segments")
        if (
            segments.ndim != 3
            or segments.shape[0] < 1
            or segments.shape[1] != 2
            or segments.shape[2] < 1
            or not np.all(np.isfinite(segments))
        ):
            raise _error(
                "segments must have finite shape (n_segments, 2, n_dims).",
                "provide at least one centerline segment",
            )
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            lengths = np.hypot.reduce(segments[:, 1] - segments[:, 0], axis=1)
        if np.any(~np.isfinite(lengths) | (lengths <= 0)):
            raise _error(
                "segments must have positive representable lengths.",
                "remove coincident endpoints or rescale coordinates",
            )
        width = self.width
        if width is not None:
            if (
                isinstance(width, (bool, np.bool_))
                or not isinstance(width, (int, float, np.integer, np.floating))
                or not np.isfinite(width)
                or width / 2 <= 0
            ):
                raise _error(
                    "width must be a finite positive full track width or None.",
                    "supply a measured full width, or None for unknown geometry",
                )
            object.__setattr__(self, "width", float(width))
        segments.setflags(write=False)
        object.__setattr__(self, "segments", segments)

    @property
    def n_dims(self) -> int:
        """Number of centerline coordinate columns."""
        return int(self.segments.shape[2])

    @property
    def domain_known(self) -> bool:
        """Whether a width was supplied or the track is one-dimensional."""
        return self.width is not None or self.n_dims == 1

    @property
    def physical_footprint_known(self) -> bool:
        """Whether the physical capsule footprint is known."""
        return self.width is not None

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Use point-to-segment distance rather than nearest bin centers.

        A one-dimensional track without a width contains the closed
        intervals its segments cover.
        """
        query, finite = _query(positions, self.n_dims)
        inside = np.zeros(len(query), dtype=np.bool_)
        if self.width is None:
            if self.n_dims != 1:
                raise _unknown()
            low = self.segments[:, :, 0].min(axis=1)
            high = self.segments[:, :, 0].max(axis=1)
            values = query[finite, :1]
            inside[finite] = np.any((values >= low) & (values <= high), axis=1)
            return inside
        inside[finite] = _track_distances(self, query[finite]) <= self.width / 2
        return inside

    def as_track(self) -> TrackArena:
        """The track itself, already in its current coordinates."""
        return self

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Map centerlines and width only under a Cartesian similarity.

        Containment is still answered in the stored frame, with bounded
        rounding, so a point on the mapped boundary stays on the track.
        :meth:`as_track` gives the mapped centerline and width.
        """
        matrix = _affine(matrix, self.n_dims)
        scale = self._similarity_scale(matrix)
        _mapped(self.segments, matrix)
        if self.width is not None and not np.isfinite(self.width * scale):
            raise _error(
                "Transformed track width exceeds floating-point range.",
                "rescale the coordinate units before transforming",
            )
        return _TransformedArena(self, matrix)

    def _similarity_scale(self, matrix: NDArray[np.float64]) -> float:
        """Uniform length scale of an affine ``matrix`` that is a similarity."""
        linear = matrix[:-1, :-1]
        with np.errstate(under="ignore"):
            _, logdet = np.linalg.slogdet(linear)
        scale = math.exp(float(logdet) / self.n_dims)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            normalized = linear / scale
            residual = np.max(np.abs(normalized @ normalized.T - np.eye(self.n_dims)))
        if residual > 1e-12 or not np.isfinite(residual):
            raise _error(
                "TrackArena transforms must be a similarity.",
                "use translation, rotation, reflection and uniform scale, or rebuild the measured footprint",
            )
        return scale

    def polygonal_domain(self) -> None:
        """Do not approximate round caps with a polygonal buffer."""
        return None

    def domain_bounds(self) -> NDArray[np.float64] | None:
        """Bound the capsules of a track with a known width, or a 1-D track."""
        if not self.domain_known:
            return None
        half_width = 0.0 if self.width is None else self.width / 2
        return np.array(
            [
                self.segments.min(axis=(0, 1)) - half_width,
                self.segments.max(axis=(0, 1)) + half_width,
            ]
        )


@dataclass(frozen=True, eq=False)
class EgocentricSurround:
    """Computational radial/angular support, without inferred physical walls.

    Parameters
    ----------
    distance_range : tuple of float
        Nonnegative finite increasing radial limits in length units.
    angle_range : tuple of float
        Finite increasing angular limits in radians.
    circular : bool
        Whether the angular axis wraps across the seam, so the bins at the
        two ends of the angle range are neighbors. Containment does not
        depend on it: an angle names a direction, so every window compares
        angles modulo two pi (see :meth:`wrap_angles`), and a partial window
        keeps its configured angular exclusions.
    """

    distance_range: tuple[float, float]
    angle_range: tuple[float, float]
    circular: bool
    n_dims: ClassVar[int] = 2
    coordinate_kind: ClassVar[_CoordinateKind] = "polar"
    domain_known: ClassVar[bool] = True
    physical_footprint_known: ClassVar[bool] = False
    walls: ClassVar[tuple[LineString, ...]] = ()
    wall_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        for name in ("distance_range", "angle_range"):
            values = _real_array(getattr(self, name), name)
            if (
                values.shape != (2,)
                or not np.all(np.isfinite(values))
                or values[0] >= values[1]
                or (name == "distance_range" and values[0] < 0)
            ):
                raise _error(
                    f"{name} must contain finite increasing limits.",
                    "pass two ordered limits with nonnegative radial distances",
                )
            object.__setattr__(self, name, (float(values[0]), float(values[1])))
        low, high = self.angle_range
        if high - low > 2 * np.pi and not np.isclose(
            high - low, 2 * np.pi, rtol=8 * np.finfo(float).eps, atol=0.0
        ):
            raise _error(
                "angle_range spans more than one full turn (2π).",
                "pass at most one turn; angles a whole turn apart name one direction",
            )
        object.__setattr__(self, "circular", _boolean(self.circular, "circular"))

    def wrap_angles(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Express each angle in ``[low, low + 2π)`` of the angle range.

        An angle and the same angle plus a whole number of turns name one
        direction. Distances are unchanged, and a NaN or infinite angle stays
        NaN.

        Parameters
        ----------
        positions : ndarray, shape (n_positions, 2)
            ``(distance, angle)`` rows, angles in radians.

        Returns
        -------
        ndarray, shape (n_positions, 2)
            A copy with each angle wrapped to start at the range's low limit.
            On a full circle that is the angle range itself; on a partial
            window, an angle inside the window is unchanged.
        """
        query = np.array(np.atleast_2d(positions), dtype=np.float64, copy=True)
        low = self.angle_range[0]
        with np.errstate(invalid="ignore"):
            query[:, 1] = low + np.remainder(query[:, 1] - low, 2 * np.pi)
        return query

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test native ``(distance, angle)`` positions against the closed window.

        The window includes its edges, while the bins are half-open: a row
        exactly at the outer distance limit, or at the upper angle of a
        partial window, is inside the window but in no bin, so
        ``Environment.bin_at`` gives it ``-1``.
        """
        query, finite = _query(positions, 2)
        selected = self.wrap_angles(query[finite])
        angles = selected[:, 1]
        inside = np.zeros(len(query), dtype=np.bool_)
        inside[finite] = (
            (selected[:, 0] >= self.distance_range[0])
            & (selected[:, 0] <= self.distance_range[1])
            & (angles >= self.angle_range[0])
            & (angles <= self.angle_range[1])
        )
        return inside

    def as_track(self) -> TrackArena | None:
        """None: this arena is not a track."""
        return None

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Refuse Cartesian affine maps of radial/angular coordinates."""
        raise _error(
            "Cartesian affine maps do not apply to EgocentricSurround.",
            "transform the Cartesian observations before computing egocentric coordinates",
        )

    def polygonal_domain(self) -> None:
        """A polar window is not a Cartesian polygon."""
        return None

    def domain_bounds(self) -> None:
        """A polar window has no Cartesian box."""
        return None


@dataclass(frozen=True, eq=False)
class _TransformedArena(ReadOnlyState):
    """Owned affine frame with locally bounded inverse arithmetic uncertainty."""

    base: Arena
    forward: NDArray[np.float64]
    inverse: NDArray[np.float64] = field(init=False, repr=False)
    _inverse_sensitivity: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.base.coordinate_kind != "cartesian":
            raise _error(
                "Only Cartesian arenas have affine frames.",
                "retain native polar coordinates",
            )
        forward = _affine(self.forward, self.base.n_dims)
        if isinstance(self.base, OpenArena):
            _mapped(shapely.get_coordinates(self.base.polygon), forward)
        with np.errstate(under="ignore"):
            inverse = np.asarray(np.linalg.inv(forward), dtype=np.float64)
        if not np.all(np.isfinite(inverse)):
            raise _error(
                "The affine inverse exceeds floating-point range.",
                "rescale coordinates before transforming",
            )
        linear, inverse_linear = forward[:-1, :-1], inverse[:-1, :-1]
        # A componentwise Neumann bound accounts for the numerical inverse.
        # The factor two leaves a margin for rounding in the small bound solve.
        eps = np.finfo(np.float64).eps
        gamma = (self.n_dims + 2) * eps / (1 - (self.n_dims + 2) * eps)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            residual = np.abs(np.eye(self.n_dims) - inverse_linear @ linear)
            residual += 2 * gamma * (np.abs(inverse_linear) @ np.abs(linear))
        if not np.all(np.isfinite(residual)) or np.max(residual.sum(axis=1)) >= 0.5:
            raise _error(
                "The affine inverse has no useful floating-point error bound.",
                "rescale coordinates or use a better-conditioned affine map",
            )
        sensitivity = 2 * np.linalg.solve(
            np.eye(self.n_dims) - residual, np.abs(inverse_linear)
        )
        if not np.all(np.isfinite(sensitivity)):
            raise _error(
                "The affine inverse sensitivity exceeds floating-point range.",
                "rescale the coordinate units before transforming",
            )
        for array in (forward, inverse, sensitivity):
            array.setflags(write=False)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "inverse", inverse)
        object.__setattr__(self, "_inverse_sensitivity", sensitivity)

    @property
    def n_dims(self) -> int:
        return self.base.n_dims

    @property
    def coordinate_kind(self) -> _CoordinateKind:
        return self.base.coordinate_kind

    @property
    def domain_known(self) -> bool:
        return self.base.domain_known

    @property
    def physical_footprint_known(self) -> bool:
        return self.base.physical_footprint_known

    @property
    def walls(self) -> tuple[LineString, ...]:
        return tuple(mapped_geometry(wall, self.forward) for wall in self.base.walls)

    @property
    def wall_names(self) -> tuple[str, ...]:
        return self.base.wall_names

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        query, finite = _query(positions, self.n_dims)
        inside = np.zeros(len(query), dtype=np.bool_)
        original, error = self._project(query[finite])
        representable = np.all(np.isfinite(original), axis=1)
        selected = np.zeros(len(original), dtype=np.bool_)
        selected[representable] = contains_with_error(
            self.base, original[representable], error[representable]
        )
        inside[finite] = selected
        return inside

    def _project(
        self, query: NDArray[np.float64], incoming: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bound inverse projection through a forward residual in target units."""
        eps = np.finfo(np.float64).eps
        gamma = (self.n_dims + 4) * eps / (1 - (self.n_dims + 4) * eps)
        linear, shift = self.forward[:-1, :-1], self.forward[:-1, -1]
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            original = (query - shift) @ self.inverse[:-1, :-1].T
            remapped = original @ linear.T + shift
            difference = np.abs(query - remapped)
            # Translation is one rounded addition, not a dot product of
            # origin-sized terms. Half-ULPs bound that addition and the
            # caller's forward-mapped query without growing an origin buffer.
            residual = (
                difference
                + 0.5
                * (
                    np.spacing(difference)
                    + np.spacing(np.abs(remapped))
                    + np.spacing(np.abs(query))
                )
                + 2 * gamma * (np.abs(original) @ np.abs(linear).T)
            )
            if incoming is not None:
                residual += incoming
            error = residual @ self._inverse_sensitivity.T
        return original, error

    def as_track(self) -> TrackArena | None:
        """The base track mapped into the current coordinates, or None.

        Vertices go through the same arithmetic as cell vertices, and the
        width is scaled by the similarity's uniform scale.
        """
        track = self.base.as_track()
        if track is None:
            return None
        return TrackArena(
            _mapped_vertices(track.segments, self.forward),
            None
            if track.width is None
            else track.width * track._similarity_scale(self.forward),
        )

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        return _TransformedArena(self, _affine(matrix, self.n_dims))

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        domain = self.base.polygonal_domain()
        return (
            None
            if domain is None
            else cast("Polygon | MultiPolygon", mapped_geometry(domain, self.forward))
        )

    def domain_bounds(self) -> NDArray[np.float64] | None:
        """Bound the mapped corners of the base domain's box."""
        limits = self.base.domain_bounds()
        if limits is None:
            return None
        corners = np.array(list(itertools.product(*limits.T)))
        mapped = corners @ self.forward[:-1, :-1].T + self.forward[:-1, -1]
        return np.array([mapped.min(axis=0), mapped.max(axis=0)])


def _track_distances(
    arena: TrackArena, query: NDArray[np.float64]
) -> NDArray[np.float64]:
    distance = np.full(len(query), np.inf)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for start, end in arena.segments:
            vector = end - start
            length = np.hypot.reduce(vector)
            direction = vector / length
            offset = query - start
            projection = np.clip(offset @ direction, 0, length)
            distance = np.minimum(
                distance,
                np.hypot.reduce(offset - projection[:, None] * direction, axis=1),
            )
    return distance


def _feature_cap(arena: Arena) -> NDArray[np.float64]:
    """Keep arithmetic allowances below a quarter of local domain features."""
    if isinstance(arena, _CombinedArena):
        return np.asarray(
            np.min([_feature_cap(child) for child in arena.children], axis=0),
            dtype=np.float64,
        )
    if isinstance(arena, MaskArena):
        return np.array([np.min(np.diff(edge)) / 4 for edge in arena.edges])
    if isinstance(arena, OpenArena):
        return np.full(2, arena.polygon.minimum_clearance / (4 * np.sqrt(2)))
    if isinstance(arena, TrackArena) and arena.width is not None:
        lengths = np.hypot.reduce(arena.segments[:, 1] - arena.segments[:, 0], axis=1)
        feature = min(arena.width / 2, float(np.min(lengths)))
        # Separate capsule components can leave a narrower meaningful gap.
        for index, segment in enumerate(arena.segments):
            for other in arena.segments[index + 1 :]:
                gap = _segment_distance(segment, other) - arena.width
                if gap > 0:
                    feature = min(feature, gap)
        return np.full(arena.n_dims, feature / (4 * np.sqrt(arena.n_dims)))
    return np.zeros(arena.n_dims)


def _segment_distance(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Distance between finite Cartesian segments in any coordinate dimension."""
    u, v = first[1] - first[0], second[1] - second[0]
    lu, lv = float(np.hypot.reduce(u)), float(np.hypot.reduce(v))
    u, v = u / lu, v / lv
    w = first[0] - second[0]
    b, d, e = float(u @ v), float(u @ w), float(v @ w)
    distances = []
    if 1 - b * b > 0:
        s, t = (b * e - d) / (1 - b * b), (e - b * d) / (1 - b * b)
        if 0 <= s <= lu and 0 <= t <= lv:
            distances.append(float(np.hypot.reduce(w + s * u - t * v)))
    for point in first:
        offset = point - second[0]
        distances.append(
            float(np.hypot.reduce(offset - np.clip(offset @ v, 0, lv) * v))
        )
    for point in second:
        offset = point - first[0]
        distances.append(
            float(np.hypot.reduce(offset - np.clip(offset @ u, 0, lu) * u))
        )
    return min(distances)


def contains_with_error(
    arena: Arena, query: NDArray[np.float64], error: NDArray[np.float64]
) -> NDArray[np.bool_]:
    if isinstance(arena, _CombinedArena):
        return np.asarray(
            np.logical_or.reduce(
                [contains_with_error(child, query, error) for child in arena.children]
            ),
            dtype=np.bool_,
        )
    if isinstance(arena, _TransformedArena):
        original, propagated = arena._project(query, error)
        return contains_with_error(arena.base, original, propagated)
    inside = arena.contains(query)
    rows = np.flatnonzero(~inside)
    if not len(rows):
        return inside
    allowance = np.minimum(error[rows], _feature_cap(arena))
    if isinstance(arena, OpenArena):
        distance = shapely.distance(arena.polygon, shapely.points(query[rows]))
        inside[rows] = distance <= np.hypot.reduce(allowance, axis=1)
    elif isinstance(arena, MaskArena):
        snapped = query[rows].copy()
        for axis, edge in enumerate(arena.edges):
            right = np.clip(np.searchsorted(edge, snapped[:, axis]), 0, len(edge) - 1)
            left = np.maximum(right - 1, 0)
            nearest = np.where(
                np.abs(snapped[:, axis] - edge[left])
                < np.abs(snapped[:, axis] - edge[right]),
                left,
                right,
            )
            close = np.abs(snapped[:, axis] - edge[nearest]) <= allowance[:, axis]
            snapped[close, axis] = edge[nearest[close]]
        inside[rows] = arena.contains(snapped)
    elif isinstance(arena, TrackArena) and arena.width is not None:
        inside[rows] = _track_distances(
            arena, query[rows]
        ) <= arena.width / 2 + np.hypot.reduce(allowance, axis=1)
    return inside


@dataclass(frozen=True, eq=False)
class _CombinedArena:
    """Union of owned component domains without inventing physical walls."""

    children: tuple[Arena, ...]
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if not children or any(
            child.n_dims != children[0].n_dims or child.coordinate_kind != "cartesian"
            for child in children
        ):
            raise ValueError("Combined arenas require matching Cartesian dimensions.")
        object.__setattr__(self, "children", children)

    @property
    def n_dims(self) -> int:
        return self.children[0].n_dims

    @property
    def domain_known(self) -> bool:
        return all(child.domain_known for child in self.children)

    @property
    def physical_footprint_known(self) -> bool:
        return all(child.physical_footprint_known for child in self.children)

    @property
    def walls(self) -> tuple[LineString, ...]:
        return tuple(wall for child in self.children for wall in child.walls)

    @property
    def wall_names(self) -> tuple[str, ...]:
        return tuple(name for child in self.children for name in child.wall_names)

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        query, finite = _query(positions, self.n_dims)
        return np.asarray(
            finite
            & np.logical_or.reduce([child.contains(query) for child in self.children]),
            dtype=np.bool_,
        )

    def as_track(self) -> TrackArena | None:
        """None: this arena is not a track."""
        return None

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        return _TransformedArena(self, matrix)

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        polygons = [child.polygonal_domain() for child in self.children]
        if any(polygon is None for polygon in polygons):
            return None
        return cast("Polygon | MultiPolygon", shapely.union_all(polygons))

    def domain_bounds(self) -> NDArray[np.float64] | None:
        """Bound the union, or every component box when it is not polygonal."""
        polygon = self.polygonal_domain()
        if polygon is not None:
            return _polygon_bounds(polygon)
        limits = [child.domain_bounds() for child in self.children]
        if any(limit is None for limit in limits):
            return None
        children = np.array(limits)
        return np.array([children[:, 0].min(axis=0), children[:, 1].max(axis=0)])
