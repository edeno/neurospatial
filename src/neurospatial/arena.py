"""Immutable continuous domains, separate from bins and analysis selections.

Arena containment includes physical boundaries. A bin locator can retain its
own half-open ownership rules. An inferred computational domain does not assert
that its boundary is a measured physical wall.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.affinity import affine_transform
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from neurospatial._exceptions import _format_error

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

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test ``(n_positions, n_dims)`` native coordinates, including boundaries."""
        ...

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Apply a supported homogeneous affine map to Cartesian coordinates."""
        ...

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        """Return exact planar polygonal support when available."""
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


def _mapped_geometry(geometry: Any, matrix: NDArray[np.float64]) -> Any:
    _mapped(shapely.get_coordinates(geometry), matrix)
    return affine_transform(
        geometry,
        [
            matrix[0, 0],
            matrix[0, 1],
            matrix[1, 0],
            matrix[1, 1],
            matrix[0, -1],
            matrix[1, -1],
        ],
    )


class _ArrayState:
    """Restore owned-array protection after pickle and deepcopy."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, np.ndarray):
                        item.setflags(write=False)
            object.__setattr__(self, name, value)


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


@dataclass(frozen=True, eq=False)
class OpenArena:
    """A known planar polygonal domain with separately asserted internal walls.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Valid finite support, including any holes.
    walls : tuple of LineString, optional
        Explicit internal physical walls; analysis boundaries are not inferred.
    physical_footprint_known : bool, default=True
        False for a computational domain inferred from observations.
    """

    polygon: Polygon | MultiPolygon
    walls: tuple[LineString, ...] = field(default=(), kw_only=True)
    physical_footprint_known: bool = field(default=True, kw_only=True)
    n_dims: ClassVar[int] = 2
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"
    domain_known: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _polygon(self.polygon)
        walls = tuple(self.walls)
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
                    f"walls[{index}] must be a finite nonempty 2-D LineString.",
                    "supply explicit physical wall segments",
                )
            if not self.polygon.covers(wall):
                raise _error(
                    f"walls[{index}] lies outside the arena domain.",
                    "provide walls within the polygonal domain",
                )
        object.__setattr__(self, "walls", walls)
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

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Map polygon rings and explicit walls together."""
        return _TransformedArena(self, _affine(matrix, 2))

    def polygonal_domain(self) -> Polygon | MultiPolygon:
        """Return the immutable stored polygonal domain."""
        return self.polygon


@dataclass(frozen=True, eq=False)
class MaskArena(_ArrayState):
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

    def __post_init__(self) -> None:
        edges = tuple(
            _real_array(edge, f"edges[{axis}]") for axis, edge in enumerate(self.edges)
        )
        mask = np.asarray(self.active_mask)
        if not edges or any(
            edge.ndim != 1
            or len(edge) < 2
            or not np.all(np.isfinite(edge))
            or np.any(edge[1:] <= edge[:-1])
            for edge in edges
        ):
            raise _error(
                "edges must be finite strictly increasing 1-D arrays.",
                "supply at least two edges per coordinate axis",
            )
        if (
            mask.dtype.kind != "b"
            or mask.shape != tuple(len(edge) - 1 for edge in edges)
            or not np.any(mask)
        ):
            raise _error(
                "active_mask must be boolean, nonempty, and match the edges.",
                "pass a boolean cell mask with at least one active cell",
            )
        mask = np.array(mask, dtype=np.bool_, copy=True)
        for array in (*edges, mask):
            array.setflags(write=False)
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

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Retain axis-aligned masks, or store an inverse coordinate map."""
        matrix = _affine(matrix, self.n_dims)
        diagonal = np.diag(matrix[:-1, :-1])
        if np.array_equal(matrix[:-1, :-1], np.diag(diagonal)):
            edges = []
            mask = self.active_mask
            for axis, edge in enumerate(self.edges):
                with np.errstate(over="ignore", invalid="ignore"):
                    mapped = edge * diagonal[axis] + matrix[axis, -1]
                if diagonal[axis] < 0:
                    mapped = mapped[::-1]
                    mask = np.flip(mask, axis=axis)
                edges.append(mapped)
            return MaskArena(
                tuple(edges),
                mask,
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


@dataclass(frozen=True, eq=False)
class TrackArena(_ArrayState):
    """Union of exact constant-width capsules around centerline segments.

    Parameters
    ----------
    segments : ndarray, shape (n_segments, 2, n_dims)
        Finite nondegenerate centerline segments in Cartesian coordinates.
    width : float or None
        Positive full physical width. None preserves an unknown footprint.
        End caps are round and junctions are unions of segment capsules.
    """

    segments: NDArray[np.float64]
    width: float | None
    coordinate_kind: ClassVar[_CoordinateKind] = "cartesian"
    walls: ClassVar[tuple[LineString, ...]] = ()

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
        """Whether a positive physical width was supplied."""
        return self.width is not None

    @property
    def physical_footprint_known(self) -> bool:
        """Whether the physical capsule footprint is known."""
        return self.domain_known

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Use point-to-segment distance rather than nearest bin centers."""
        query, finite = _query(positions, self.n_dims)
        if self.width is None:
            raise _unknown()
        inside = np.zeros(len(query), dtype=np.bool_)
        inside[finite] = _track_distances(self, query[finite]) <= self.width / 2
        return inside

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Map centerlines and width only under a Cartesian similarity."""
        matrix = _affine(matrix, self.n_dims)
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
        _mapped(self.segments, matrix)
        if self.width is not None and not np.isfinite(self.width * scale):
            raise _error(
                "Transformed track width exceeds floating-point range.",
                "rescale the coordinate units before transforming",
            )
        return _TransformedArena(self, matrix)

    def polygonal_domain(self) -> None:
        """Do not approximate round caps with a polygonal buffer."""
        return None


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
        Normalize equivalent angles modulo two pi before testing the window.
        Partial windows retain their configured angular exclusions.
    """

    distance_range: tuple[float, float]
    angle_range: tuple[float, float]
    circular: bool
    n_dims: ClassVar[int] = 2
    coordinate_kind: ClassVar[_CoordinateKind] = "polar"
    domain_known: ClassVar[bool] = True
    physical_footprint_known: ClassVar[bool] = False
    walls: ClassVar[tuple[LineString, ...]] = ()

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
        object.__setattr__(self, "circular", _boolean(self.circular, "circular"))

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Test native ``(distance, angle)`` positions, including window edges."""
        query, finite = _query(positions, 2)
        selected = query[finite]
        angles = selected[:, 1]
        if self.circular:
            angles = self.angle_range[0] + np.remainder(
                angles - self.angle_range[0], 2 * np.pi
            )
        inside = np.zeros(len(query), dtype=np.bool_)
        inside[finite] = (
            (selected[:, 0] >= self.distance_range[0])
            & (selected[:, 0] <= self.distance_range[1])
            & (angles >= self.angle_range[0])
            & (angles <= self.angle_range[1])
        )
        return inside

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        """Refuse Cartesian affine maps of radial/angular coordinates."""
        raise _error(
            "Cartesian affine maps do not apply to EgocentricSurround.",
            "transform the Cartesian observations before computing egocentric coordinates",
        )

    def polygonal_domain(self) -> None:
        """A polar window is not a Cartesian polygon."""
        return None


@dataclass(frozen=True, eq=False)
class _TransformedArena(_ArrayState):
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
        return tuple(_mapped_geometry(wall, self.forward) for wall in self.base.walls)

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        query, finite = _query(positions, self.n_dims)
        inside = np.zeros(len(query), dtype=np.bool_)
        original, error = self._project(query[finite])
        representable = np.all(np.isfinite(original), axis=1)
        selected = np.zeros(len(original), dtype=np.bool_)
        selected[representable] = _contains_with_error(
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

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        return _TransformedArena(self, _affine(matrix, self.n_dims))

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        domain = self.base.polygonal_domain()
        return (
            None
            if domain is None
            else cast("Polygon | MultiPolygon", _mapped_geometry(domain, self.forward))
        )


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


def _contains_with_error(
    arena: Arena, query: NDArray[np.float64], error: NDArray[np.float64]
) -> NDArray[np.bool_]:
    if isinstance(arena, _CombinedArena):
        return np.asarray(
            np.logical_or.reduce(
                [_contains_with_error(child, query, error) for child in arena.children]
            ),
            dtype=np.bool_,
        )
    if isinstance(arena, _TransformedArena):
        original, propagated = arena._project(query, error)
        return _contains_with_error(arena.base, original, propagated)
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


def _wkb(geometry: shapely.Geometry) -> NDArray[np.uint8]:
    result: NDArray[np.uint8] = np.frombuffer(
        shapely.to_wkb(geometry), dtype=np.uint8
    ).copy()
    return result


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

    def contains(self, positions: NDArray[np.float64]) -> NDArray[np.bool_]:
        query, finite = _query(positions, self.n_dims)
        return np.asarray(
            finite
            & np.logical_or.reduce([child.contains(query) for child in self.children]),
            dtype=np.bool_,
        )

    def transformed(self, matrix: NDArray[np.float64]) -> Arena:
        return _TransformedArena(self, matrix)

    def polygonal_domain(self) -> Polygon | MultiPolygon | None:
        polygons = [child.polygonal_domain() for child in self.children]
        if any(polygon is None for polygon in polygons):
            return None
        return cast("Polygon | MultiPolygon", shapely.union_all(polygons))


def encode_arena(arena: Arena) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Encode built-in arena data with local array keys, without I/O.

    Parameters
    ----------
    arena : Arena
        A supported immutable arena value.

    Returns
    -------
    metadata : dict
        JSON-compatible tagged data with codec version one.
    arrays : dict of ndarray
        Independent writable numerical arrays using local field names.
    """
    meta: dict[str, Any] = {"codec": type(arena).__name__, "codec_version": 1}
    arrays: dict[str, NDArray[Any]] = {}
    if type(arena) is _CombinedArena:
        entries = []
        for index, child in enumerate(arena.children):
            entry, values = encode_arena(child)
            entries.append(entry)
            arrays.update(
                {f"children/{index}/{key}": value for key, value in values.items()}
            )
        meta["children"] = entries
    elif type(arena) is UnknownArena:
        meta.update(n_dims=arena.n_dims, coordinate_kind=arena.coordinate_kind)
    elif type(arena) is OpenArena:
        meta.update(
            n_walls=len(arena.walls),
            physical_footprint_known=arena.physical_footprint_known,
        )
        arrays["polygon_wkb"] = _wkb(arena.polygon)
        arrays.update(
            {f"walls_wkb_{i}": _wkb(wall) for i, wall in enumerate(arena.walls)}
        )
    elif type(arena) is MaskArena:
        meta.update(
            n_dims=arena.n_dims, physical_footprint_known=arena.physical_footprint_known
        )
        arrays.update(
            {f"edges_{axis}": edge.copy() for axis, edge in enumerate(arena.edges)}
        )
        arrays["active_mask"] = arena.active_mask.copy()
    elif type(arena) is TrackArena:
        meta["width"] = arena.width
        arrays["segments"] = arena.segments.copy()
    elif type(arena) is EgocentricSurround:
        meta.update(
            distance_range=list(arena.distance_range),
            angle_range=list(arena.angle_range),
            circular=arena.circular,
        )
    elif type(arena) is _TransformedArena:
        base_meta, base_arrays = encode_arena(arena.base)
        meta["base"] = base_meta
        arrays["forward"] = arena.forward.copy()
        arrays.update({f"base/{key}": value for key, value in base_arrays.items()})
    else:
        raise _error(
            f"Cannot encode unregistered arena type {type(arena).__name__!r}.",
            "use a built-in arena type with a supported codec",
        )
    return meta, arrays


def _count(meta: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = meta[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(
            f"{name} must be an integer >= {minimum}.", "restore valid arena metadata"
        )
    return int(value)


def _read_wkb(arrays: Mapping[str, NDArray[Any]], name: str) -> shapely.Geometry:
    data = np.asarray(arrays[name])
    if data.ndim != 1 or data.dtype != np.uint8:
        raise _error(
            f"{name} must be a one-dimensional uint8 WKB array.",
            "restore valid arena geometry bytes",
        )
    return cast("shapely.Geometry", shapely.from_wkb(data.tobytes()))


def _check_arena_nesting(meta: Mapping[str, Any]) -> None:
    """Bound recursive domain records before entering the recursive decoder."""
    stack: list[tuple[object, int]] = [(meta, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 32:
            raise ValueError("Arena nesting exceeds 32 levels; flatten the domains.")
        if not isinstance(value, Mapping):
            continue
        if value.get("codec") == "_CombinedArena":
            children = value.get("children", ())
            if isinstance(children, (list, tuple)):
                stack.extend((child, depth + 1) for child in children)
        elif value.get("codec") == "_TransformedArena":
            stack.append((value.get("base", {}), depth + 1))


def decode_arena(meta: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]) -> Arena:
    """Validate and own an arena payload, without builders or I/O.

    Parameters
    ----------
    meta : mapping
        Tagged arena metadata produced by :func:`encode_arena`.
    arrays : mapping of ndarray
        Local arrays; all published array buffers are copied.

    Returns
    -------
    Arena
        Validated immutable geometry with protected owned arrays.

    Raises
    ------
    ValueError
        If a codec/version is unknown or stored geometry is malformed.
    """
    _check_arena_nesting(meta)
    try:
        kind = meta.get("codec")
        if isinstance(kind, str) and kind not in {
            "UnknownArena",
            "OpenArena",
            "MaskArena",
            "TrackArena",
            "EgocentricSurround",
            "_TransformedArena",
            "_CombinedArena",
        }:
            from neurospatial.extensions import _decode_snapshot

            snapshot = _decode_snapshot(meta, arrays, "")
            if not isinstance(snapshot, Arena):
                raise ValueError("The installed adapter must decode an arena snapshot.")
            return snapshot
        if _count(meta, "codec_version", minimum=1) != 1:
            raise _error(
                "Unsupported arena codec version.",
                "read this record with a version that supports its arena codec",
            )
        kind = meta["codec"]
        if kind == "_CombinedArena":
            return _CombinedArena(
                tuple(
                    decode_arena(
                        entry,
                        {
                            key.removeprefix(f"children/{index}/"): value
                            for key, value in arrays.items()
                            if key.startswith(f"children/{index}/")
                        },
                    )
                    for index, entry in enumerate(meta["children"])
                )
            )
        if kind == "UnknownArena":
            return UnknownArena(
                _count(meta, "n_dims", minimum=1),
                coordinate_kind=meta["coordinate_kind"],
            )
        if kind == "OpenArena":
            return OpenArena(
                cast("Polygon | MultiPolygon", _read_wkb(arrays, "polygon_wkb")),
                walls=tuple(
                    cast("LineString", _read_wkb(arrays, f"walls_wkb_{i}"))
                    for i in range(_count(meta, "n_walls"))
                ),
                physical_footprint_known=_boolean(
                    meta["physical_footprint_known"], "physical_footprint_known"
                ),
            )
        if kind == "MaskArena":
            return MaskArena(
                tuple(
                    arrays[f"edges_{axis}"]
                    for axis in range(_count(meta, "n_dims", minimum=1))
                ),
                arrays["active_mask"],
                physical_footprint_known=_boolean(
                    meta["physical_footprint_known"], "physical_footprint_known"
                ),
            )
        if kind == "TrackArena":
            return TrackArena(arrays["segments"], meta["width"])
        if kind == "EgocentricSurround":
            return EgocentricSurround(
                meta["distance_range"],
                meta["angle_range"],
                _boolean(meta["circular"], "circular"),
            )
        if kind == "_TransformedArena":
            base_arrays = {
                key.removeprefix("base/"): value
                for key, value in arrays.items()
                if key.startswith("base/")
            }
            return _TransformedArena(
                decode_arena(meta["base"], base_arrays), arrays["forward"]
            )
        raise _error(
            f"Unknown arena codec {kind!r}.",
            "install a version supporting this required arena codec",
        )
    except (KeyError, TypeError, shapely.errors.GEOSException) as exc:
        raise _error(
            f"Malformed arena payload: {exc}.", "restore a complete valid arena record"
        ) from exc
