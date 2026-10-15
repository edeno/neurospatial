"""Certify disjoint component support without substituting sampled lookup."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon

from neurospatial.arena import (
    Arena,
    MaskArena,
    TrackArena,
    _CombinedArena,
    _segment_distance,
    _TransformedArena,
)
from neurospatial.layout._binning import Binning, Locator
from neurospatial.layout._locators import (
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import materialize_cells


@dataclass
class _Piece:
    kind: str
    data: Any
    low: NDArray[np.float64]
    high: NDArray[np.float64]


def _boxes(locator: Locator) -> list[NDArray[np.float64]]:
    """Return parallelotope vertices in binary corner order, in active-bin order."""
    from neurospatial.layout._composition import CombinedLocator

    if isinstance(locator, CombinedLocator):
        return [box for child in locator.children for box in _boxes(child.locator)]
    if isinstance(locator, SubsetLocator):
        boxes = _boxes(locator.base)
        selected = np.flatnonzero(locator.parent_to_child >= 0)
        selected = selected[np.argsort(locator.parent_to_child[selected])]
        return [boxes[index] for index in selected]
    if isinstance(locator, TransformedLocator):
        matrix = np.linalg.inv(locator.inverse)
        return [
            vertices @ matrix[:-1, :-1].T + matrix[:-1, -1]
            for vertices in _boxes(locator.base)
        ]
    if not isinstance(locator, RectilinearLocator):
        raise ValueError("No stored rectilinear cells.")
    mask = (
        np.ones(locator.shape, dtype=bool)
        if locator.active_mask is None
        else locator.active_mask
    )
    indices = np.flatnonzero(mask.ravel())
    if locator.flat_to_active is not None:
        indices = indices[np.argsort(locator.flat_to_active[indices])]
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
) -> tuple[NearestBinLocator, NDArray[np.intp], NDArray[np.float64]] | None:
    """Identify retained cells of a common nearest query, including wrappers."""
    if isinstance(locator, NearestBinLocator):
        assert locator.bin_indices is not None
        return locator, locator.bin_indices.copy(), np.eye(locator.centers.shape[1] + 1)
    if isinstance(locator, (SubsetLocator, TransformedLocator)):
        value = _nearest_partition(locator.base)
        if value is None:
            return None
        base, mapping, inverse = value
        if isinstance(locator, SubsetLocator):
            active = mapping >= 0
            mapping[active] = locator.parent_to_child[mapping[active]]
        else:
            inverse = inverse @ locator.inverse
        return base, mapping, inverse
    return None


def _partitioned(first: Binning, second: Binning) -> bool:
    a, b = _nearest_partition(first.locator), _nearest_partition(second.locator)
    if a is None or b is None:
        return False
    left, lm, li = a
    right, rm, ri = b
    return bool(
        left.max_distance == right.max_distance
        and np.array_equal(left.centers, right.centers)
        and np.array_equal(li, ri)
        and not np.any((lm >= 0) & (rm >= 0))
    )


def _arena_pieces(arena: Arena) -> list[_Piece]:
    """Use exact domains or conservative affine enclosures of balls/capsules."""
    if isinstance(arena, _CombinedArena):
        return [piece for child in arena.children for piece in _arena_pieces(child)]
    if isinstance(arena, _TransformedArena):
        matrix = arena.forward
        linear, shift = matrix[:-1, :-1], matrix[:-1, -1]
        pieces = []
        from shapely.affinity import affine_transform

        for piece in _arena_pieces(arena.base):
            if piece.kind == "polygon":
                polygon = affine_transform(
                    piece.data,
                    [
                        linear[0, 0],
                        linear[0, 1],
                        linear[1, 0],
                        linear[1, 1],
                        shift[0],
                        shift[1],
                    ],
                )
                pieces.append(
                    _Piece(
                        "polygon",
                        polygon,
                        np.array(polygon.bounds[:2]),
                        np.array(polygon.bounds[2:]),
                    )
                )
            elif piece.kind == "box":
                vertices = piece.data @ linear.T + shift
                pieces.append(
                    _Piece("box", vertices, vertices.min(axis=0), vertices.max(axis=0))
                )
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
        return [
            _Piece(
                "polygon", poly, np.array(poly.bounds[:2]), np.array(poly.bounds[2:])
            )
            for poly in polygons
        ]
    if isinstance(arena, MaskArena):
        boxes = _boxes(
            RectilinearLocator(
                arena.edges, arena.active_mask.shape, arena.active_mask, None
            )
        )
        return [
            _Piece("box", vertices, vertices.min(axis=0), vertices.max(axis=0))
            for vertices in boxes
        ]
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
    if binning.bins.n_dims == 2 and binning.bins.measure_dim == 2:
        try:
            polygons = materialize_cells(binning)
        except ValueError:
            pass
        else:
            return [
                _Piece(
                    "polygon",
                    poly,
                    np.array(poly.bounds[:2]),
                    np.array(poly.bounds[2:]),
                )
                for poly in polygons
            ]
    try:
        boxes = _boxes(binning.locator)
    except ValueError:
        pass
    else:
        return [
            _Piece("box", vertices, vertices.min(axis=0), vertices.max(axis=0))
            for vertices in boxes
        ]
    nearest = _nearest_partition(binning.locator)
    if nearest is not None:
        base, mapping, inverse = nearest
        if base.max_distance is not None and np.isfinite(base.max_distance):
            forward = np.linalg.inv(inverse)
            linear = forward[:-1, :-1]
            # A transformed ball is enclosed by this sphere. Separation proves
            # disjointness; overlap of outer bounds is only inconclusive.
            radius = float(base.max_distance * np.linalg.norm(linear, ord=2))
            centers = base.centers[mapping >= 0] @ linear.T + forward[:-1, -1]
            return [
                _Piece("ball", (center, radius), center - radius, center + radius)
                for center in centers
            ]
    return _arena_pieces(arena)


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
        distance = _segment_distance(ac, bc)
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
    raise ValueError(
        "Cannot certify disjoint mixed geometry with intersecting bounds; restrict the domains or use a shared supported representation."
    )


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
                if any(_overlap(a, b) for a in left for b in right):
                    raise ValueError(
                        "Overlapping cells or intersecting support bounds; disjoint lookup support cannot be certified. Restrict or separate the components."
                    )
            except ValueError as error:
                raise ValueError(
                    f"Cannot combine {keys[i]!r} and {keys[j]!r}: {error}"
                ) from error
