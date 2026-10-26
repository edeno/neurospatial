"""Certify disjoint components and join them with inferred bridges."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise, product
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from neurospatial._validation import validate_finite_scalar
from neurospatial.geometry._arena import (
    Arena,
    MaskArena,
    TrackArena,
    _CombinedArena,
    _TransformedArena,
    affine_geometry,
    segment_distance,
)
from neurospatial.geometry._arrays import active_order
from neurospatial.geometry._binning import (
    Binning,
    KnownFaces,
    TrackLinearization,
)
from neurospatial.geometry._composition import (
    CombinedLocator,
    combined_faces,
    combined_normals,
)
from neurospatial.geometry._explicit import prepare_bins
from neurospatial.geometry._locators import (
    Locator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._native_faces import NativeFaces
from neurospatial.geometry._polygon import cells_from_locator, materialize_cells
from neurospatial.geometry._polygon_cells import PolygonCells


@dataclass
class _Piece:
    kind: str
    data: Any
    low: NDArray[np.float64]
    high: NDArray[np.float64]

    @classmethod
    def polygon(cls, polygon: Polygon) -> _Piece:
        return cls(
            "polygon",
            polygon,
            np.array(polygon.bounds[:2]),
            np.array(polygon.bounds[2:]),
        )

    @classmethod
    def box(cls, vertices: NDArray[np.float64]) -> _Piece:
        return cls("box", vertices, vertices.min(axis=0), vertices.max(axis=0))


def cell_boxes(locator: Locator) -> list[NDArray[np.float64]]:
    """Return parallelotope vertices in binary corner order, in active-bin order."""
    if isinstance(locator, CombinedLocator):
        return [box for child in locator.children for box in cell_boxes(child.locator)]
    if isinstance(locator, SubsetLocator):
        boxes = cell_boxes(locator.base)
        selected = active_order(locator.parent_to_child)
        return [boxes[index] for index in selected]
    if isinstance(locator, TransformedLocator):
        matrix = locator.forward
        return [
            vertices @ matrix[:-1, :-1].T + matrix[:-1, -1]
            for vertices in cell_boxes(locator.base)
        ]
    if not isinstance(locator, RectilinearLocator):
        raise ValueError("No stored rectilinear cells.")
    indices = np.flatnonzero(locator.active_mask.ravel())
    corners = np.array(list(product((0, 1), repeat=len(locator.shape))), dtype=np.intp)
    return [
        np.column_stack(
            [
                edge[cell[axis] + corners[:, axis]]
                for axis, edge in enumerate(locator.edges)
            ]
        )
        for cell in zip(*np.unravel_index(indices, locator.shape), strict=True)
    ]


def _nearest_partition(
    locator: Locator,
) -> (
    tuple[NearestBinLocator, NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]
    | None
):
    """Identify retained cells of a common nearest query, including wrappers.

    Returns the nearest query, the bin each of its sites maps to (or -1),
    and the maps from current coordinates to the query's (inverse) and back
    (forward).
    """
    if isinstance(locator, NearestBinLocator):
        assert locator.bin_indices is not None
        identity = np.eye(locator.centers.shape[1] + 1)
        return locator, locator.bin_indices.copy(), identity, identity
    if isinstance(locator, (SubsetLocator, TransformedLocator)):
        value = _nearest_partition(locator.base)
        if value is None:
            return None
        base, mapping, inverse, forward = value
        if isinstance(locator, SubsetLocator):
            active = mapping >= 0
            mapping[active] = locator.parent_to_child[mapping[active]]
        else:
            inverse = inverse @ locator.inverse
            forward = locator.forward @ forward
        return base, mapping, inverse, forward
    return None


def _partitioned(first: Binning, second: Binning) -> bool:
    a, b = _nearest_partition(first.locator), _nearest_partition(second.locator)
    if a is None or b is None:
        return False
    left, lm, li, _ = a
    right, rm, ri, _ = b
    return bool(
        left.max_distance == right.max_distance
        and np.array_equal(left.centers, right.centers)
        and np.array_equal(li, ri)
        and not np.any((lm >= 0) & (rm >= 0))
    )


def arena_pieces(arena: Arena) -> list[_Piece]:
    """Use exact domains or conservative affine enclosures of balls/capsules."""
    if isinstance(arena, _CombinedArena):
        return [piece for child in arena.children for piece in arena_pieces(child)]
    if isinstance(arena, _TransformedArena):
        matrix = arena.forward
        linear, shift = matrix[:-1, :-1], matrix[:-1, -1]
        pieces = []
        for piece in arena_pieces(arena.base):
            if piece.kind == "polygon":
                polygon = affine_geometry(piece.data, matrix)
                pieces.append(_Piece.polygon(polygon))
            elif piece.kind == "box":
                pieces.append(_Piece.box(piece.data @ linear.T + shift))
            else:
                points, radius = piece.data
                points = points @ linear.T + shift
                radius = float(radius * np.linalg.norm(linear, ord=2))
                low, high = (
                    (points.min(axis=0), points.max(axis=0))
                    if piece.kind == "capsule"
                    else (points, points)
                )
                pieces.append(
                    _Piece(piece.kind, (points, radius), low - radius, high + radius)
                )
        return pieces
    polygon = arena.polygonal_domain()
    if polygon is not None:
        polygons = (polygon,) if isinstance(polygon, Polygon) else tuple(polygon.geoms)
        return [_Piece.polygon(poly) for poly in polygons]
    if isinstance(arena, MaskArena):
        boxes = cell_boxes(
            RectilinearLocator(arena.edges, arena.active_mask.shape, arena.active_mask)
        )
        return [_Piece.box(vertices) for vertices in boxes]
    if isinstance(arena, TrackArena) and arena.width is not None:
        radius = arena.width / 2
        return [
            _Piece(
                "capsule",
                (segment, radius),
                segment.min(axis=0) - radius,
                segment.max(axis=0) + radius,
            )
            for segment in arena.segments
        ]
    raise ValueError(
        "Cannot certify component lookup support: unrestricted nearest lookup or unknown track footprint. Supply a finite lookup radius, a known domain, or disjoint subsets of the same parent locator."
    )


def _pieces(binning: Binning, arena: Arena) -> list[_Piece]:
    if not binning.bins.n_bins:
        return []
    if binning.bins.n_dims == 2 and binning.bins.has_measure_dim(2):
        try:
            polygons = materialize_cells(binning)
        except ValueError:
            pass
        else:
            return [_Piece.polygon(poly) for poly in polygons]
    try:
        boxes = cell_boxes(binning.locator)
    except ValueError:
        pass
    else:
        return [_Piece.box(vertices) for vertices in boxes]
    nearest = _nearest_partition(binning.locator)
    if nearest is not None:
        base, mapping, _, forward = nearest
        if base.max_distance is not None and np.isfinite(base.max_distance):
            linear = forward[:-1, :-1]
            # A transformed ball is enclosed by this sphere. Separation proves
            # disjointness; overlap of outer bounds is only inconclusive.
            radius = float(base.max_distance * np.linalg.norm(linear, ord=2))
            centers = base.centers[mapping >= 0] @ linear.T + forward[:-1, -1]
            return [
                _Piece("ball", (center, radius), center - radius, center + radius)
                for center in centers
            ]
    return arena_pieces(arena)


def _boxes_overlap(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    dims = a.shape[1]
    a_low, a_high, b_low, b_high = (
        a.min(axis=0),
        a.max(axis=0),
        b.min(axis=0),
        b.max(axis=0),
    )
    if np.any(np.minimum(a_high, b_high) <= np.maximum(a_low, b_low)):
        return False
    # Axis-aligned boxes need no approximate geometry predicate.
    if all(
        np.all((v == low) | (v == high))
        for v, low, high in ((a, a_low, a_high), (b, b_low, b_high))
    ):
        return True
    if dims != 3:
        raise ValueError(
            "Cannot certify transformed box overlap outside 3D; use an axis-aligned frame."
        )
    ae = np.column_stack([a[1 << (dims - axis - 1)] - a[0] for axis in range(dims)])
    be = np.column_stack([b[1 << (dims - axis - 1)] - b[0] for axis in range(dims)])
    axes = [
        *np.linalg.inv(ae),
        *np.linalg.inv(be),
        *[np.cross(x, y) for x in ae.T for y in be.T],
    ]
    origin = a[0]
    uncertain = False
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm == 0:
            continue
        axis = axis / norm
        ap, bp = (a - origin) @ axis, (b - origin) @ axis
        overlap = min(ap.max(), bp.max()) - max(ap.min(), bp.min())
        if overlap <= 0:
            return False
        error = (
            64 * np.finfo(float).eps * max(np.ptp(ap), np.ptp(bp), np.max(np.abs(bp)))
        )
        uncertain |= overlap <= error
    if uncertain:
        raise ValueError(
            "Cannot resolve transformed-cell contact at this coordinate precision; rescale the geometry."
        )
    return True


def _overlap(a: _Piece, b: _Piece) -> bool:
    if np.any(np.minimum(a.high, b.high) <= np.maximum(a.low, b.low)):
        return False
    if a.kind == b.kind == "polygon":
        return bool(a.data.intersection(b.data).area > 0)
    if a.kind == b.kind == "box":
        return _boxes_overlap(a.data, b.data)
    if a.kind == b.kind == "ball":
        ac, ar = a.data
        bc, br = b.data
        return bool(np.linalg.norm(ac - bc) < ar + br)
    if a.kind == b.kind == "capsule":
        ac, ar = a.data
        bc, br = b.data
        u, v = ac[1] - ac[0], bc[1] - bc[0]
        lu, lv = float(np.hypot.reduce(u)), float(np.hypot.reduce(v))
        u, v = u / lu, v / lv
        parallel = np.array_equal(u, v) or np.array_equal(u, -v)
        sine = float(np.hypot.reduce(v - (u @ v) * u))
        if not parallel and sine * sine <= 64 * np.finfo(float).eps:
            raise ValueError(
                "Cannot certify nearly parallel track clearance; restrict the domains or use a well-resolved geometry."
            )
        distance = segment_distance(ac, bc)
        scale = max(lu, lv, float(np.max(np.abs(ac))), float(np.max(np.abs(bc))))
        error = 256 * np.finfo(float).eps * scale / (1.0 if parallel else sine * sine)
        if distance > ar + br + error:
            return False
        if distance < ar + br - error:
            return True
        raise ValueError(
            "Cannot resolve track contact at coordinate precision; rescale the geometry."
        )
    if a.kind == "ball" and b.kind == "polygon":
        center, radius = a.data
        return bool(b.data.distance(Point(center)) < radius)
    if b.kind == "ball" and a.kind == "polygon":
        return _overlap(b, a)
    if {a.kind, b.kind} == {"capsule", "polygon"}:
        # A track of half-width r around its centerline shares area with a
        # cell exactly when the cell comes closer than r to the centerline.
        capsule, cell = (a, b) if a.kind == "capsule" else (b, a)
        segment, radius = capsule.data
        return bool(cell.data.distance(LineString(segment)) < radius)
    raise ValueError(
        "Cannot certify disjoint mixed geometry with intersecting bounds; restrict the domains or use a shared supported representation."
    )


def _grid_segment_intervals(
    locator: Locator,
    segment: NDArray[np.float64],
    selected: NDArray[np.bool_] | None = None,
    error: NDArray[np.float64] | None = None,
) -> list[tuple[float, float]] | None:
    """Intersect a segment with grid-support interiors, including shared faces."""
    if isinstance(locator, SubsetLocator):
        mapping = locator.parent_to_child
        retained = mapping >= 0
        if selected is not None:
            retained[retained] = selected[mapping[retained]]
        return _grid_segment_intervals(locator.base, segment, retained, error)
    if isinstance(locator, TransformedLocator):
        local, propagated = locator.affine.project(segment, error)
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(propagated)):
            raise ValueError(
                "Cannot represent automatic bridge coordinates in the grid frame; "
                "rescale the inputs or disable auto_bridge."
            )
        return _grid_segment_intervals(locator.base, local, selected, propagated)
    if not isinstance(locator, RectilinearLocator):
        return None
    if error is not None:
        segment = segment.copy()
        for axis, edges in enumerate(locator.edges):
            if np.any(error[:, axis] >= np.min(np.diff(edges)) / 4):
                raise ValueError(
                    "Cannot resolve automatic bridge clearance at this grid's "
                    "coordinate precision; rescale the inputs or disable auto_bridge."
                )
            for row in range(2):
                nearest = int(np.argmin(np.abs(edges - segment[row, axis])))
                if abs(edges[nearest] - segment[row, axis]) <= error[row, axis]:
                    segment[row, axis] = edges[nearest]
    mask = locator.active_mask.copy()
    if selected is not None:
        mask.ravel()[np.flatnonzero(mask)] = selected
    direction = segment[1] - segment[0]
    cuts = [0.0, 1.0]
    for axis, edges in enumerate(locator.edges):
        if direction[axis] != 0:
            values = (edges - segment[0, axis]) / direction[axis]
            interior = values[(values > 0) & (values < 1)]
            if len(np.unique(interior)) != len(interior):
                raise ValueError(
                    "Cannot resolve distinct grid crossings along an automatic "
                    "bridge; rescale the inputs or disable auto_bridge."
                )
            cuts.extend(interior.tolist())
    ordered = sorted(set(cuts))
    intervals = []
    for low, high in pairwise(ordered):
        middle = (low + high) / 2
        if not low < middle < high:
            raise ValueError(
                "Cannot resolve grid-interior bridge intervals at this precision; "
                "rescale the inputs or disable auto_bridge."
            )
        point = segment[0] + middle * direction
        # On an internal grid face every adjacent cell must be active for
        # this point to be interior to the union. An exterior face is excluded.
        choices = [
            {
                int(np.searchsorted(edges, point[axis], side="left")) - 1,
                int(np.searchsorted(edges, point[axis], side="right")) - 1,
            }
            for axis, edges in enumerate(locator.edges)
        ]
        if all(
            all(
                0 <= index < size
                for index, size in zip(cell, locator.shape, strict=True)
            )
            and mask[cell]
            for cell in product(*choices)
        ):
            intervals.append((low, high))
    return intervals


def _ball_segment_interval(
    segment: NDArray[np.float64], center: NDArray[np.float64], radius: float
) -> tuple[float, float] | None:
    """Intersect an open ball with a nondegenerate segment, without sampling."""
    direction = segment[1] - segment[0]
    length = float(np.hypot.reduce(direction))
    unit = direction / length
    offset = segment[0] - center
    along = -float(offset @ unit)
    distance = float(np.hypot.reduce(offset + along * unit))
    uncertainty = (
        32
        * (len(offset) + 4)
        * np.finfo(float).eps
        * (np.hypot.reduce(offset) + length)
    )
    if distance <= uncertainty or abs(distance - radius) <= uncertainty:
        # Normalized projection can leave an epsilon-sized residual even on
        # an exactly collinear diagonal. Resolve ambiguous geometry using the
        # exact dyadic input values, rather than treating that residual as a
        # miss when the footprint is smaller than arithmetic round-off.
        exact_offset = [
            Fraction(float(point)) - Fraction(float(origin))
            for point, origin in zip(segment[0], center, strict=True)
        ]
        exact_direction = [
            Fraction(float(stop)) - Fraction(float(start))
            for start, stop in zip(segment[0], segment[1], strict=True)
        ]
        square_length = sum((value * value for value in exact_direction), Fraction(0))
        dot = sum(
            (a * b for a, b in zip(exact_offset, exact_direction, strict=True)),
            Fraction(0),
        )
        square_offset = sum((value * value for value in exact_offset), Fraction(0))
        square_radius = Fraction(radius) ** 2
        projection = -dot / square_length
        closest = max(Fraction(0), min(Fraction(1), projection))
        if (
            square_offset + 2 * closest * dot + closest**2 * square_length
            >= square_radius
        ):
            return None
        perpendicular = square_offset - dot**2 / square_length
        margin = float((square_radius - perpendicular) / square_radius)
        span = (radius / length) * np.sqrt(margin)
        low, high = (
            max(0.0, float(projection) - span),
            min(1.0, float(projection) + span),
        )
        if not low < high:
            raise ValueError(
                "Cannot resolve a bounded nearest footprint along an automatic bridge; "
                "rescale the inputs or disable auto_bridge."
            )
        return low, high
    if distance >= radius:
        return None
    span = radius * np.sqrt((1 - distance / radius) * (1 + distance / radius))
    with np.errstate(over="ignore"):
        low, high = (along - span) / length, (along + span) / length
    low, high = max(0.0, low), min(1.0, high)
    if low == high and 0 < along / length < 1:
        raise ValueError(
            "Cannot resolve a bounded nearest footprint along an automatic bridge; "
            "rescale the inputs or disable auto_bridge."
        )
    return (low, high) if low < high else None


def _nearest_segment_intervals(
    locator: Locator, segment: NDArray[np.float64]
) -> list[tuple[float, float]] | None:
    """Intersect actual retained Voronoi support and its optional radius bound."""
    partition = _nearest_partition(locator)
    if partition is None:
        return None
    base, mapping, inverse, _ = partition
    local = segment @ inverse[:-1, :-1].T + inverse[:-1, -1]
    if not np.all(np.isfinite(local)) or np.array_equal(local[0], local[1]):
        raise ValueError(
            "Cannot represent automatic bridge coordinates in the nearest frame; "
            "rescale the inputs or disable auto_bridge."
        )
    direction = local[1] - local[0]
    intervals = []
    for center in base.centers[mapping >= 0]:
        low, high = 0.0, 1.0
        # Other retained sites do not create an external support boundary.
        # A dropped site's bisector does; a line lying on it is boundary-only.
        for dropped in base.centers[mapping < 0]:
            difference = dropped - center
            with np.errstate(over="ignore", invalid="ignore"):
                intercept = float(
                    2 * ((local[0] - center) @ difference) - difference @ difference
                )
                slope = float(2 * (direction @ difference))
            if not np.isfinite(intercept) or not np.isfinite(slope):
                raise ValueError(
                    "Cannot certify automatic bridge Voronoi geometry at this "
                    "coordinate scale; rescale the inputs or disable auto_bridge."
                )
            if slope == 0:
                if intercept >= 0:
                    high = low
                    break
            elif slope > 0:
                high = min(high, -intercept / slope)
            else:
                low = max(low, -intercept / slope)
        if base.max_distance is not None and np.isfinite(base.max_distance):
            ball = _ball_segment_interval(local, center, base.max_distance)
            if ball is None:
                continue
            low, high = max(low, ball[0]), min(high, ball[1])
        if low < high:
            intervals.append((low, high))
    return intervals


def _arena_segment_interior(
    arena: Arena,
    segment: NDArray[np.float64],
    error: NDArray[np.float64] | None = None,
) -> bool:
    """Test known domain interiors using their exact stored geometry."""
    if not arena.domain_known:
        return True
    if isinstance(arena, _TransformedArena):
        local, propagated = arena.affine.project(segment, error)
        return _arena_segment_interior(arena.base, local, propagated)
    if isinstance(arena, MaskArena):
        return bool(
            _grid_segment_intervals(
                RectilinearLocator(
                    arena.edges, arena.active_mask.shape, arena.active_mask
                ),
                segment,
                error=error,
            )
        )
    polygon = arena.polygonal_domain()
    if polygon is not None:
        return bool(LineString(segment).relate_pattern(polygon, "1********"))
    if isinstance(arena, _CombinedArena):
        raise ValueError(
            "Cannot certify automatic bridge clearance through a nonplanar union "
            "domain; disable auto_bridge."
        )
    if isinstance(arena, TrackArena) and arena.width is not None:
        return any(
            segment_distance(segment, track) < arena.width / 2
            for track in arena.segments
        )
    raise ValueError(
        "Cannot certify automatic bridge clearance through this domain; disable auto_bridge."
    )


def segment_enters_component(
    segment: NDArray[np.float64], binning: Binning, arena: Arena
) -> bool:
    """Detect positive-length traversal of a third component's support interior."""
    if not binning.bins.n_bins or np.array_equal(segment[0], segment[1]):
        return False
    intervals = _grid_segment_intervals(binning.locator, segment)
    if (
        intervals is None
        and binning.bins.n_dims == 2
        and binning.bins.has_measure_dim(2)
    ):
        try:
            polygons = materialize_cells(binning)
        except ValueError:
            pass
        else:
            support = unary_union(polygons)
            domain = arena.polygonal_domain() if arena.domain_known else None
            if domain is not None:
                support = support.intersection(domain)
            if domain is not None or not arena.domain_known:
                return bool(LineString(segment).relate_pattern(support, "1********"))
            raise ValueError(
                "Cannot certify bridge clearance for polygon cells with a nonpolygonal domain; disable auto_bridge."
            )
    if intervals is None:
        intervals = _nearest_segment_intervals(binning.locator, segment)
    if intervals is None:
        track = binning.linearization
        if (
            isinstance(track, TrackLinearization)
            and arena.domain_known
            and np.all(track.active_mask)
        ):
            return _arena_segment_interior(arena, segment)
        raise ValueError(
            "Cannot certify automatic bridge clearance through a third component; "
            "provide exact cells or a supported bounded footprint, or disable auto_bridge."
        )
    direction = segment[1] - segment[0]
    for low, high in intervals:
        portion = segment[0] + np.array([low, high])[:, None] * direction
        if np.array_equal(portion[0], portion[1]):
            raise ValueError(
                "Cannot resolve bridge clearance within the component domain; "
                "rescale the inputs or disable auto_bridge."
            )
        if _arena_segment_interior(arena, portion):
            return True
    return False


def _bound_candidates(
    left: list[_Piece], right: list[_Piece]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return piece pairs whose bounds are not separated along any axis.

    Parameters
    ----------
    left, right : list of _Piece
        Pieces of two components with bounds of shape ``(n_dims,)``.

    Returns
    -------
    left_index, right_index : NDArray[np.intp], shape (n_candidates,)
        Pairs sorted by left index, then right index. A pair is excluded
        exactly when ``_overlap`` would reject it by its bounds alone.
    """
    if not left or not right:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    left_low = np.array([piece.low for piece in left], dtype=np.float64)
    left_high = np.array([piece.high for piece in left], dtype=np.float64)
    right_low = np.array([piece.low for piece in right], dtype=np.float64)
    right_high = np.array([piece.high for piece in right], dtype=np.float64)

    def envelopes(low: NDArray[np.float64], high: NDArray[np.float64]) -> Any:
        # Closed rectangles over the first two axes enclose every pair whose
        # bounds intersect strictly on all axes.
        if low.shape[1] == 1:
            low = np.column_stack([low, np.zeros(len(low))])
            high = np.column_stack([high, np.ones(len(high))])
        return shapely.box(low[:, 0], low[:, 1], high[:, 0], high[:, 1])

    # The tree indexes finite rectangles only. Pieces with nonfinite bounds
    # are paired with every other piece so the exact bound test decides them.
    left_finite = np.isfinite(left_low).all(axis=1) & np.isfinite(left_high).all(axis=1)
    right_finite = np.isfinite(right_low).all(axis=1) & np.isfinite(right_high).all(
        axis=1
    )
    left_rows, right_rows = np.flatnonzero(left_finite), np.flatnonzero(right_finite)
    query, tree_index = shapely.STRtree(
        envelopes(right_low[right_rows], right_high[right_rows])
    ).query(envelopes(left_low[left_rows], left_high[left_rows]))
    every_right = np.arange(len(right))
    unbounded_left = np.flatnonzero(~left_finite)
    unbounded_right = np.flatnonzero(~right_finite)
    left_index = np.concatenate(
        [
            left_rows[query],
            np.repeat(unbounded_left, len(right)),
            np.repeat(left_rows, len(unbounded_right)),
        ]
    )
    right_index = np.concatenate(
        [
            right_rows[tree_index],
            np.tile(every_right, len(unbounded_left)),
            np.tile(unbounded_right, len(left_rows)),
        ]
    )
    flat = np.unique(left_index * len(right) + right_index)
    left_index, right_index = np.divmod(flat, len(right))
    separated = np.any(
        np.minimum(left_high[left_index], right_high[right_index])
        <= np.maximum(left_low[left_index], right_low[right_index]),
        axis=1,
    )
    return (
        left_index[~separated].astype(np.intp),
        right_index[~separated].astype(np.intp),
    )


# Set in a CombinedLocator's __dict__ once its components are certified. A
# locator is immutable, so the certificate holds for it and for its copies and
# unpickled versions (which keep __dict__); a decoded locator is certified anew.
_CERTIFIED = "_components_certified"


def certify_locator(locator: CombinedLocator) -> None:
    """Certify a combined locator's components once per record.

    Raises
    ------
    ValueError
        If two components overlap or their disjoint support cannot be
        certified.
    """
    if locator.__dict__.get(_CERTIFIED, False):
        return
    certify_components(locator.children, locator.arenas, locator.keys)
    object.__setattr__(locator, _CERTIFIED, True)


def certify_components(
    children: tuple[Binning, ...], arenas: tuple[Arena, ...], keys: tuple[str, ...]
) -> None:
    """Refuse positive overlap or uncertified support between components."""
    for i, first in enumerate(children):
        for j in range(i + 1, len(children)):
            second = children[j]
            if (
                not first.bins.n_bins
                or not second.bins.n_bins
                or _partitioned(first, second)
            ):
                continue
            try:
                left, right = _pieces(first, arenas[i]), _pieces(second, arenas[j])
                # Candidate pairs keep full-scan order, so the overlap or
                # uncertain contact reported is the first one a full scan of
                # every piece pair would find.
                if any(
                    _overlap(left[a], right[b])
                    for a, b in zip(*_bound_candidates(left, right), strict=True)
                ):
                    raise ValueError(
                        "Overlapping cells or intersecting support bounds; disjoint lookup support cannot be certified. Restrict or separate the components."
                    )
            except ValueError as error:
                raise ValueError(
                    f"Cannot combine {keys[i]!r} and {keys[j]!r}: {error}"
                ) from error


BridgeEndpoint = tuple[str, int]
BridgeKey = tuple[BridgeEndpoint, BridgeEndpoint]


def _nearest_indices(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Query nearest centers, resolving exact ties by the lowest local bin ID."""
    tree = cKDTree(target)
    count = min(2, len(target))
    distances, indices = tree.query(source, k=count)
    if count == 1:
        return np.asarray(indices, dtype=np.intp), np.asarray(
            distances, dtype=np.float64
        )
    nearest = np.asarray(indices[:, 0], dtype=np.intp)
    lengths = np.asarray(distances[:, 0], dtype=np.float64)
    for row in np.flatnonzero(distances[:, 0] == distances[:, 1]):
        candidates = np.asarray(
            tree.query_ball_point(source[row], np.nextafter(lengths[row], np.inf)),
            dtype=np.intp,
        )
        actual = np.linalg.norm(target[candidates] - source[row], axis=1)
        nearest[row] = np.min(candidates[actual == actual.min()])
        lengths[row] = actual.min()
    return nearest, lengths


def combine_binnings(
    keys: tuple[str, ...],
    binnings: tuple[Binning, ...],
    arenas: tuple[Arena, ...],
    *,
    auto_bridge: bool,
    max_mnn_distance: float | None,
    bridge_conductances: Mapping[BridgeKey, float] | None,
    kind: str,
) -> tuple[Binning, int]:
    """Join disjoint Cartesian component binnings with inferred bridges.

    Bridges join mutual nearest-neighbor centers of two components when the
    straight segment between them crosses no wall and enters no third
    component.

    Parameters
    ----------
    keys : tuple of str
        Component keys, in output bin order.
    binnings : tuple of Binning
        Cartesian component binnings with the same number of coordinate
        dimensions. Their bins may be measured differently (for example a
        track's length bins next to an arena's area bins); a bridge between
        such bins is then navigation-only.
    arenas : tuple of Arena
        Each component's arena.
    auto_bridge : bool
        Infer mutual-nearest-neighbor bridges.
    max_mnn_distance : float or None
        Inclusive maximum bridge length, in coordinate units.
    bridge_conductances : mapping or None
        Conductance of inferred bridges, keyed by pairs of
        ``(component_key, local_bin_id)`` endpoints.
    kind : str
        Layout label of the combined binning.

    Returns
    -------
    binning : Binning
        The combined, validated binning, labelled ``kind``.
    n_bridges : int
        Number of inferred bridges.

    Raises
    ------
    TypeError
        If ``auto_bridge`` is not a bool.
    ValueError
        If components overlap, support cannot be certified, a bridge option
        is invalid, or a bridge with positive conductance joins bins measured
        in different dimensions.
    NotImplementedError
        If ``auto_bridge`` is True outside 2-D while an arena has walls,
        because a bridge's wall crossings cannot be certified there.
    """
    if type(auto_bridge) is not bool:
        raise TypeError("auto_bridge must be boolean.")
    validate_finite_scalar(
        max_mnn_distance, name="max_mnn_distance", positive=False, none_allowed=True
    )
    # Records are immutable, so the components are shared; a native face
    # recipe is built once here, as every combined face and diffusion reads it.
    children = tuple(
        replace(child, faces=KnownFaces(child.build_faces()))
        if isinstance(child.faces, NativeFaces)
        else child
        for child in binnings
    )
    certify_components(children, arenas, keys)
    n_dims = children[0].bins.n_dims
    offsets = np.cumsum([0, *[child.bins.n_bins for child in children]])
    pairs, distances = [], []
    if auto_bridge:
        for i, child in enumerate(children):
            for j in range(i + 1, len(children)):
                other = children[j]
                if not child.bins.n_bins or not other.bins.n_bins:
                    continue
                right, lengths = _nearest_indices(
                    child.bins.centers, other.bins.centers
                )
                left, _reverse_lengths = _nearest_indices(
                    other.bins.centers, child.bins.centers
                )
                if not np.all(np.isfinite(lengths)):
                    raise ValueError(
                        "Bridge distances exceed coordinate precision; rescale the inputs."
                    )
                for u, v in enumerate(right):
                    if left[v] != u:
                        continue
                    distance = float(lengths[u])
                    if max_mnn_distance is not None and distance > max_mnn_distance:
                        continue
                    if n_dims == 2:
                        link = LineString(
                            [child.bins.centers[u], other.bins.centers[v]]
                        )
                        if any(
                            link.intersects(wall)
                            for arena in arenas
                            for wall in arena.walls
                        ):
                            continue
                    elif any(arena.walls for arena in arenas):
                        raise NotImplementedError(
                            "Automatic bridges cannot certify wall crossings outside 2D; disable auto_bridge."
                        )
                    segment = np.array([child.bins.centers[u], other.bins.centers[v]])
                    if any(
                        segment_enters_component(segment, third, arenas[k])
                        for k, third in enumerate(children)
                        if k not in (i, j)
                    ):
                        continue
                    pairs.append((int(offsets[i] + u), int(offsets[j] + v)))
                    distances.append(distance)
    bridge_pairs = np.array(pairs, dtype=np.intp).reshape(-1, 2)
    conductances = np.zeros(len(pairs), dtype=np.float64)
    by_pair = {frozenset(pair): index for index, pair in enumerate(pairs)}
    assigned: set[frozenset[int]] = set()
    for endpoints, value in (bridge_conductances or {}).items():
        try:
            (source_key, source_bin), (target_key, target_bin) = endpoints
            i, j = keys.index(source_key), keys.index(target_key)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Conductance keys must be pairs of (component_key, local_bin_id) endpoints."
            ) from error
        if any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < children[owner].bins.n_bins
            for index, owner in ((source_bin, i), (target_bin, j))
        ):
            raise ValueError(
                "Conductance endpoint bin IDs must be valid local integers."
            )
        edge = frozenset((int(offsets[i] + source_bin), int(offsets[j] + target_bin)))
        if edge not in by_pair:
            raise ValueError(
                "Conductance endpoints must identify an inferred bridge; inspect get_bridge_attributes()."
            )
        validate_finite_scalar(value, name="Bridge conductance", positive=False)
        if edge in assigned:
            raise ValueError(
                "Supply one conductance per unordered bridge; reversed endpoints duplicate the same connection."
            )
        assigned.add(edge)
        conductances[by_pair[edge]] = value
    locator = CombinedLocator(
        keys, children, arenas, bridge_pairs, np.array(distances), conductances
    )
    # The locator holds the components certified above.
    object.__setattr__(locator, _CERTIFIED, True)
    bins = locator.spatial_bins()
    connections = locator.bin_connections()
    faces = combined_faces(locator, connections, bins.sizes)
    shapes = None
    if n_dims == 2 and bins.has_measure_dim(2):
        with suppress(ValueError):
            shapes = PolygonCells(cells_from_locator(locator))
    normals = combined_normals(locator) if not len(pairs) else None
    combined = prepare_bins(
        Binning(
            kind,
            bins,
            connections,
            locator,
            faces,
            shapes=shapes,
            face_normals=normals,
        )
    )
    return combined, len(pairs)
