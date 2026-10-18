"""Infer, check and apply an arena for a binning without inventing walls."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
import shapely.ops
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, unary_union

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.geometry._arena import (
    Arena,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _CombinedArena,
    _TransformedArena,
)
from neurospatial.geometry._binning import KnownFaces
from neurospatial.geometry._codecs import encode_arena
from neurospatial.geometry._composition import CombinedLocator
from neurospatial.geometry._coordinates import CartesianSystem, CoordinateSystem
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._locators import SubsetLocator, TransformedLocator
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.geometry._polygon_cells import ClippedLocator, PolygonCells
from neurospatial.geometry._validation import _polygon_area_allowance
from neurospatial.geometry._wall_frame import wall_source_frame

if TYPE_CHECKING:
    from neurospatial.geometry._binning import Binning


def infer_arena(binning: Binning) -> Arena:
    """Infer a Cartesian domain from declared shapes or cells, without walls.

    A polar environment's arena is never inferred: its
    :class:`EgocentricSurround` declares whether the angle wraps, which the
    bins do not record.
    """
    combined = _composition_arena(binning.locator)
    if combined is not None:
        return combined
    n_dims = binning.bins.n_dims
    grid = binning.rectilinear
    if binning.bins.n_bins == 0:
        return UnknownArena(n_dims)
    track = binning.linearization
    if track is not None:
        graph = getattr(track, "graph", None)
        if graph is not None:
            segments = np.array(
                [
                    [graph.nodes[u]["pos"], graph.nodes[v]["pos"]]
                    for u, v in graph.edges
                ],
                dtype=np.float64,
            )
            return TrackArena(segments, width=None)
        return UnknownArena(n_dims)
    if binning.outline is not None:
        return OpenArena(binning.outline.geometry)
    if binning.grid_frame is not None:
        frame = binning.grid_frame
        return MaskArena(frame.source.edges, frame.source.active_mask).transformed(
            frame.forward
        )
    if grid is not None and len(grid.shape) == n_dims:
        return MaskArena(grid.edges, grid.active_mask)
    # Exact area cells are sufficient to infer an analysis domain, but not
    # measured physical walls. Keep unsupported custom layouts usable.
    if binning.shapes is not None and n_dims == 2:
        return OpenArena(
            unary_union(materialize_cells(binning)), physical_footprint_known=False
        )
    return UnknownArena(n_dims)


def _composition_arena(locator: object) -> Arena | None:
    if isinstance(locator, CombinedLocator):
        return _CombinedArena(locator.arenas)
    if isinstance(locator, (SubsetLocator, TransformedLocator, ClippedLocator)):
        arena = _composition_arena(locator.base)
        if arena is not None and isinstance(locator, TransformedLocator):
            arena = arena.transformed(
                np.asarray(np.linalg.inv(locator.inverse), dtype=np.float64)
            )
        return arena
    return None


def validate_arena(
    arena: Arena, binning: Binning, coordinates: CoordinateSystem
) -> None:
    """Reject a domain whose coordinate semantics disagree with its bins."""
    combined_arena = _composition_arena(binning.locator)
    if combined_arena is not None:

        def matches(locator: object, candidate: Arena) -> bool:
            if isinstance(locator, (SubsetLocator, ClippedLocator)):
                return matches(locator.base, candidate)
            if isinstance(locator, TransformedLocator):
                return (
                    isinstance(candidate, _TransformedArena)
                    and np.array_equal(locator.inverse, candidate.inverse)
                    and matches(locator.base, candidate.base)
                )
            if not isinstance(locator, CombinedLocator):
                return False
            for child, child_arena in zip(
                locator.children, locator.arenas, strict=True
            ):
                validate_arena(child_arena, child, CartesianSystem())
                if child_arena.walls:
                    checked = apply_walls(child, child_arena)
                    if not np.array_equal(
                        checked.connections.pairs, child.connections.pairs
                    ):
                        raise ValueError(
                            "Component connectivity crosses its declared physical walls."
                        )
            if locator.children[0].bins.n_dims == 2:
                centers = locator.spatial_bins().centers
                for pair in locator.bridge_pairs:
                    link = LineString(centers[pair])
                    if any(
                        link.intersects(wall)
                        for domain in locator.arenas
                        for wall in domain.walls
                    ):
                        raise ValueError(
                            "A stored navigation bridge crosses a known physical wall."
                        )
            actual_meta, actual_arrays = encode_arena(candidate)
            expected_meta, expected_arrays = encode_arena(
                _CombinedArena(locator.arenas)
            )
            return (
                actual_meta == expected_meta
                and actual_arrays.keys() == expected_arrays.keys()
                and all(
                    np.array_equal(actual_arrays[key], expected_arrays[key])
                    for key in actual_arrays
                )
            )

        if not matches(binning.locator, arena):
            raise ValueError("Combined arena disagrees with stored component domains.")
    if arena.n_dims != binning.bins.n_dims:
        raise ValueError("Arena coordinate dimensions must match the binning.")
    if arena.coordinate_kind != coordinates.kind:
        raise ValueError(
            f"Arena coordinate_kind must be {coordinates.kind!r} for this environment."
        )
    if arena.domain_known and not np.all(arena.contains(binning.bins.centers)):
        raise ValueError(
            "Arena excludes a bin center; select bins by center before constructing the environment."
        )

    if isinstance(binning.shapes, PolygonCells) and combined_arena is None:
        domain = arena.polygonal_domain()
        if domain is not None:
            domain_clearance = domain.minimum_clearance
            for index, cell in enumerate(binning.shapes.polygons):
                outside = cell.difference(domain).area
                if outside == 0:
                    continue
                allowance = _polygon_area_allowance(
                    cell, reference_clearance=domain_clearance
                )
                if outside > allowance:
                    raise ValueError(
                        f"Polygon cell {index} extends outside the saved arena domain."
                    )


def _wall_cut_message(
    binning: Binning,
    arena: Arena,
    wall_segments: Sequence[LineString],
    cell: Polygon,
    index: int,
) -> str:
    """Name the wall that crosses bin ``index`` and how to align the grid."""
    crossing = next(
        number
        for number, segment in enumerate(wall_segments)
        if segment.intersection(cell).difference(cell.boundary).length > 0
    )
    ends = " to ".join(
        "(" + ", ".join(f"{value:g}" for value in point) + ")"
        for point in np.asarray(arena.walls[crossing].coords)[[0, -1]]
    )
    center = ", ".join(f"{value:g}" for value in binning.bins.centers[index])
    return _format_error(
        f"Physical wall {arena.wall_names[crossing]!r} cuts through bin {index} "
        f"(center ({center})), from {ends}.",
        why="Why: a wall blocks movement between the bins on either side, so "
        "it must run along cell edges; a wall through the middle of a cell "
        "would put one bin on both sides.",
        fix="let the factory fit the bins to the walls: from_samples, "
        "open_field, from_polygon, from_grid_mask, from_pixel_mask and "
        "from_layout move the wall onto bin edges by default "
        "(wall_fit='snap') or split the bins it crosses with "
        "wall_fit='split'; from_bins, rebin and saved environments keep their "
        "bins, so build those bins along the walls",
    )


def _divides(cell: Polygon, walls: shapely.Geometry) -> bool:
    """Whether the walls cut ``cell`` into two or more pieces.

    A wall that only enters a cell and ends inside it does not divide it.
    """
    if not walls.intersects(cell):
        return False
    pieces = shapely.ops.split(cell, walls)
    return bool(sum(piece.area > 0 for piece in pieces.geoms) > 1)


def fit_walls(
    binning: Binning, arena: Arena, mode: Literal["split", "snap"]
) -> tuple[Binning, Arena]:
    """Fit the bins and walls together before publication.

    ``"split"`` cuts each cell a wall divides into one exact polygon cell per
    side, so the walls stay where they were measured. ``"snap"`` keeps the
    cells and moves each wall onto the cell edges that separate the bins it
    divides, warning how far it moved. Bins that no wall divides are
    returned unchanged either way.
    """
    if not arena.walls:
        return binning, arena
    cells = materialize_cells(binning)
    walls = unary_union(arena.walls)
    if not any(_divides(cell, walls) for cell in cells):
        return binning, arena
    if mode == "split":
        from neurospatial.geometry._polygon import split_binning

        return split_binning(binning, arena.walls)[0], arena
    if not isinstance(arena, OpenArena):
        raise ValueError(
            _format_error(
                f"wall_fit='snap' moves the walls of an OpenArena; this arena is "
                f"a {type(arena).__name__}.",
                why="Why: only an OpenArena records walls that can be replaced.",
                fix="use wall_fit='split', or rebuild the arena as "
                "OpenArena(outline, walls=...)",
            )
        )
    snapped = _snapped_walls(binning, cells, arena.walls, arena.wall_names)
    moves = [
        wall.hausdorff_distance(MultiLineString(parts))
        for wall, parts in zip(arena.walls, snapped, strict=True)
    ]
    farthest = int(np.argmax(moves))
    warnings.warn(
        _format_error(
            f"wall_fit='snap' moved the walls onto bin edges by up to "
            f"{moves[farthest]:.3g} (in the units of the positions), most for "
            f"wall {arena.wall_names[farthest]!r}.",
            why="Why: snapping keeps every bin whole, so a wall that ran "
            "through bins now runs along the edges between them; paths, "
            "occupancy and distances to walls all use the moved walls.",
            fix="use wall_fit='split' to keep the measured walls and split the "
            "bins they cross, or build the bins along the walls",
        ),
        UserWarning,
        stacklevel=_external_stacklevel(),
    )
    return binning, replace(
        arena,
        walls=tuple(part for parts in snapped for part in parts),
        wall_names=tuple(
            name
            for name, parts in zip(arena.wall_names, snapped, strict=True)
            for _ in parts
        ),
    )


def _wall_sides(wall: LineString, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Which side of ``wall`` each point lies on, by its nearest wall segment.

    True is the left of the wall's direction; a point on the wall counts as
    left, so a wall through bin centers puts those bins on one side.
    """
    coords = np.asarray(wall.coords)
    starts, ends = coords[:-1], coords[1:]
    direction = ends - starts  # (n_segments, 2)
    offset = points[:, None, :] - starts[None, :, :]  # (n_points, n_segments, 2)
    along = np.clip(
        np.einsum("psk,sk->ps", offset, direction)
        / np.einsum("sk,sk->s", direction, direction),
        0.0,
        1.0,
    )
    nearest = starts[None] + along[..., None] * direction[None]
    segment = np.argmin(np.hypot.reduce(points[:, None, :] - nearest, axis=2), axis=1)
    rows = np.arange(len(points))
    cross = (
        direction[segment, 0] * offset[rows, segment, 1]
        - direction[segment, 1] * offset[rows, segment, 0]
    )
    return np.asarray(cross >= 0.0)


def _snapped_walls(
    binning: Binning,
    cells: Sequence[Polygon],
    walls: Sequence[LineString],
    names: Sequence[str],
) -> list[tuple[LineString, ...]]:
    """Each wall as the cell edges between the bins whose centers it separates.

    Returns one tuple of lines per wall: its edges joined where they meet.
    """
    centers = binning.bins.centers
    pairs = binning.connections.pairs
    links = shapely.linestrings(
        np.stack((centers[pairs[:, 0]], centers[pairs[:, 1]]), 1)
    )
    snapped = []
    for wall, name in zip(walls, names, strict=True):
        left = _wall_sides(wall, centers)
        crossing = np.flatnonzero(
            shapely.intersects(links, wall) & (left[pairs[:, 0]] != left[pairs[:, 1]])
        )
        edges = [
            cells[u].boundary.intersection(cells[v].boundary)
            for u, v in pairs[crossing]
        ]
        edges = [edge for edge in edges if edge.length > 0]
        if not edges:
            raise ValueError(
                _format_error(
                    f"wall_fit='snap' cannot place wall {name!r} "
                    f"{list(wall.coords)} on bin edges: it separates no two bin "
                    "centers.",
                    why="Why: the wall is shorter than a bin, so the nearest bin "
                    "edges would remove it.",
                    fix="use wall_fit='split', or a smaller bin_size",
                )
            )
        merged = linemerge(unary_union(edges))
        snapped.append(
            (merged,) if isinstance(merged, LineString) else tuple(merged.geoms)
        )
    return snapped


def apply_walls(binning: Binning, arena: Arena) -> Binning:
    """Close the neighbor pairs whose shared edge a wall blocks completely.

    A closed pair stays listed in ``binning.connections`` but leaves paths and
    finite-volume faces. A partly covered edge stays open with the open share
    of its face. Pairs already closed stay closed.
    """
    if _composition_arena(binning.locator) is not None:
        # Each owned child already carries its certified wall geometry. Rechecking
        # its independently rounded global polygons loses the native source frame.
        # New bridge crossings are checked when constructing/validating the root.
        return binning
    if not arena.walls:
        return binning

    wall_frame = wall_source_frame(binning.locator, arena, binning.bins.centers)
    if wall_frame is None:
        cells, wall_segments, centers = (
            materialize_cells(binning),
            arena.walls,
            binning.bins.centers,
        )
    else:
        # Shared snapshots avoid independently rounded collinear vertices.
        # Retained source-frame cells preserve current clipping and ownership.
        cells, wall_segments, centers, _, _ = wall_frame
    walls = unary_union(wall_segments)
    for index, cell in enumerate(cells):
        if _divides(cell, walls):
            raise ValueError(
                _wall_cut_message(binning, arena, wall_segments, cell, index)
            )
    pairs = binning.connections.pairs
    keep = np.ones(len(pairs), dtype=bool)
    # A face partly covered by a wall keeps its open share of the conductance.
    open_share = np.ones(len(pairs))
    current_cells = None
    for index, (u, v) in enumerate(pairs):
        contact = cells[u].boundary.intersection(cells[v].boundary)
        blocked = contact.intersection(walls)
        if contact.length > 0 and blocked.length > 0:
            if blocked.length < contact.length:
                open_share[index] = 1.0 - blocked.length / contact.length
            else:
                keep[index] = False
        elif contact.length == 0:
            link = LineString(centers[[u, v]])
            crosses = link.crosses(walls)
            if wall_frame is not None and crosses:
                if current_cells is None:
                    current_cells = materialize_cells(binning)
                if (
                    current_cells[u]
                    .boundary.intersection(current_cells[v].boundary)
                    .length
                    > 0
                ):
                    raise ValueError(
                        f"Physical wall face alignment for bins ({u}, {v}) is unresolved in the source frame. "
                        "Declare walls before clipping or transforming, or recenter coordinates."
                    )
            keep[index] = contact.intersection(walls).is_empty and not crosses
    if np.all(keep) and np.all(open_share == 1.0):
        return binning
    connections = binning.connections.closed(keep)
    faces = binning.faces
    if faces.available:
        source = faces.build()
        area = source.A.copy()
        partial = np.flatnonzero(open_share < 1.0)
        if len(partial):
            # The open length in the published coordinates; taking the
            # minimum keeps a face that was already narrowed unchanged when
            # the environment is published again.
            if current_cells is None:
                current_cells = materialize_cells(binning)
            for index in partial:
                u, v = pairs[index]
                full = (
                    current_cells[u].boundary.intersection(current_cells[v].boundary)
                ).length
                area[index] = min(area[index], open_share[index] * full)
        faces = KnownFaces(
            FaceGeometry(
                connections.pairs,
                area[keep],
                source.distance[keep],
                binning.bins.sizes,
            )
        )
    normals = (
        FaceNormals(binning.face_normals.values[keep])
        if binning.face_normals is not None
        else None
    )
    return replace(binning, connections=connections, faces=faces, face_normals=normals)


def blocking_walls(binning: Binning, arena: Arena) -> list[str | None]:
    """Name the wall that closes each closed pair, in ``all_pairs`` order.

    The named wall is the one covering most of the edge the pair's cells
    share, or for cells that meet only at a corner, the first wall the link
    between their centers crosses. None marks a pair that no wall of
    ``arena`` closes, such as one supplied closed to ``from_bins``.
    """
    links = binning.connections
    closed = links.all_pairs[~links.is_open]
    if not len(closed) or not arena.walls:
        return [None] * len(closed)
    frame = wall_source_frame(binning.locator, arena, binning.bins.centers)
    if frame is None:
        cells, segments, centers = (
            materialize_cells(binning),
            arena.walls,
            binning.bins.centers,
        )
    else:
        # The frame apply_walls used, so attribution matches what it closed.
        cells, segments, centers, _, _ = frame
    names: list[str | None] = []
    for u, v in closed:
        contact = cells[u].boundary.intersection(cells[v].boundary)
        if contact.length > 0:
            cover = [contact.intersection(wall).length for wall in segments]
        else:
            link = LineString(centers[[u, v]])
            cover = [float(link.intersects(wall)) for wall in segments]
        best = int(np.argmax(cover))
        names.append(arena.wall_names[best] if cover[best] > 0 else None)
    return names
