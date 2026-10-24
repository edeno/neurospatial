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
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
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
from neurospatial.geometry._binning import BinConnections, KnownFaces
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


def _divides(
    cell: Polygon | MultiPolygon,
    walls: shapely.Geometry,
    domain: Polygon | MultiPolygon | None,
) -> bool:
    """Whether the walls cut the part of ``cell`` inside ``domain`` apart.

    A wall that only enters a cell and ends inside it does not divide it. A
    cell at a slanted outline sticks out past it, and a wall ending on the
    outline divides only the part inside, so pass ``walls`` through
    :func:`_reaching` first.
    """
    if not walls.intersects(cell):
        return False
    if domain is not None and not domain.covers(cell):
        cell = cell.intersection(domain)
    parts = [
        part
        for part in shapely.get_parts(cell)
        if isinstance(part, Polygon) and part.area > 0
    ]
    # A piece within the rounding of the part's area is a sliver from
    # splitting along an edge the part already has, not a side of a wall.
    pieces = sum(
        piece.area > _polygon_area_allowance(part)
        for part in parts
        for piece in shapely.ops.split(part, walls).geoms
    )
    return bool(pieces > len(parts))


def _reaching(
    walls: Sequence[LineString], domain: Polygon | MultiPolygon | None
) -> shapely.Geometry:
    """The walls, each end within :func:`_wall_end_slack` of ``domain``'s edge
    extended past it.

    :func:`wall_inside` may have pulled such an end in from the arena's edge;
    extending it again lets the wall divide a cell clipped to the arena.
    """
    if domain is None:
        return unary_union(walls)
    slack = _wall_end_slack(domain)
    boundary = domain.boundary
    reaching = []
    for wall in walls:
        coords = np.asarray(wall.coords, dtype=np.float64)
        for end, previous in ((0, 1), (-1, -2)):
            if shapely.distance(shapely.points(coords[end]), boundary) <= slack:
                outward = coords[end] - coords[previous]
                coords[end] += 2 * slack * outward / np.hypot.reduce(outward)
        reaching.append(LineString(coords))
    return unary_union(reaching)


def fit_walls(
    binning: Binning, arena: Arena, mode: Literal["split", "snap"]
) -> tuple[Binning, Arena]:
    """Fit the bins and walls together before publication.

    ``"split"`` cuts each cell a wall divides into one exact polygon cell per
    side, so the walls stay where they were measured. Exact polygon cells
    cover only the arena, so every cell is also clipped to the arena outline,
    as :meth:`Environment.clip` does; this changes only cells that stick out
    past a slanted or curved outline. ``"snap"`` keeps the cells and moves
    each wall onto the cell edges that separate the bins it divides, warning
    how far it moved. Otherwise, bins that no wall divides are returned
    unchanged.
    """
    if not arena.walls:
        return binning, arena
    cells = materialize_cells(binning)
    domain = arena.polygonal_domain()
    walls = _reaching(arena.walls, domain)
    if not any(_divides(cell, walls, domain) for cell in cells):
        return binning, arena
    if mode == "split":
        from neurospatial.geometry._polygon import split_binning

        return split_binning(binning, (walls,), domain=domain)[0], arena
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
    # A bin whose center is inside a slanted outline can have an edge outside
    # it; keep each snapped wall to the arena, which refuses walls beyond it.
    snapped = [
        tuple(inside for part in parts for inside in clip_wall(part, arena.polygon))
        for parts in snapped
    ]
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


def _wall_end_slack(polygon: Polygon | MultiPolygon) -> float:
    """How far :func:`wall_inside` may pull a wall end in from ``polygon``'s edge.

    A millionth of the polygon's size.
    """
    x0, y0, x1, y1 = polygon.bounds
    return 1e-6 * float(max(x1 - x0, y1 - y0))


def wall_inside(piece: LineString, polygon: Polygon) -> LineString | None:
    """``piece``, shortened at its ends if needed so that ``polygon`` covers it.

    A wall clipped at a slanted edge ends at a computed intersection point that
    can fall a rounding error outside the polygon, which ``OpenArena`` then
    refuses. Pull each end that lies outside in along the wall, by at most
    :func:`_wall_end_slack`, until the arena's own containment test accepts the
    piece; an end already inside stays where it is. None if the piece is never
    accepted (a piece lying along a slanted edge).
    """
    if polygon.covers(piece):
        return piece
    ends = shapely.points(np.asarray(piece.coords)[[0, -1]])
    outside = ~shapely.covers(polygon, ends)
    slack = _wall_end_slack(polygon)
    for step in slack * np.array([1e-6, 1e-4, 1e-2, 1.0]):
        start, stop = step * outside
        if piece.length <= start + stop:
            break
        shorter = shapely.ops.substring(piece, start, piece.length - stop)
        if isinstance(shorter, LineString) and polygon.covers(shorter):
            return shorter
    return None


def clip_wall(wall: LineString, polygon: Polygon) -> list[LineString]:
    """The pieces of ``wall`` inside ``polygon``, each accepted by ``OpenArena``.

    A wall already inside is returned as it is.
    """
    if polygon.covers(wall):
        return [wall]
    return [
        inside
        for piece in shapely.get_parts(wall.intersection(polygon))
        if isinstance(piece, LineString) and piece.length > 0
        if (inside := wall_inside(piece, polygon)) is not None
    ]


def _wall_sides(wall: LineString, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Which side of ``wall`` each point lies on, by its nearest wall point.

    True is the left of the wall's direction; a point on the wall counts as
    left, so a wall through bin centers puts those bins on one side. A point
    nearest a bend of the wall lies in the wedge outside the bend, and is
    sided by the bend's bisecting normal (the sum of the unit left normals of
    the two segments meeting there); the side of either segment's line alone
    is wrong there for bends sharper than a right angle. Beyond the wall's
    ends, the end segment's line decides.
    """
    coords = np.asarray(wall.coords, dtype=np.float64)
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
    # The interior vertex nearest each point, if its nearest wall point is one.
    t = along[rows, segment]
    vertex = np.where(t == 0.0, segment, np.where(t == 1.0, segment + 1, -1))
    at_bend = (vertex > 0) & (vertex < len(coords) - 1)
    if np.any(at_bend):
        unit = direction / np.hypot.reduce(direction, axis=1)[:, None]
        left_normal = np.stack((-unit[:, 1], unit[:, 0]), axis=1)
        bend = vertex[at_bend]
        bisector = left_normal[bend - 1] + left_normal[bend]
        cross[at_bend] = np.einsum("pk,pk->p", points[at_bend] - coords[bend], bisector)
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
                    why="Why: every bin center lies on one side of it (it is "
                    "shorter than a bin, or cuts off a part of the arena "
                    "narrower than a bin), so the nearest bin edges would "
                    "remove it.",
                    fix="use wall_fit='split', or a smaller bin_size",
                )
            )
        # linemerge refuses a single line: a wall on one bin edge is one.
        union = unary_union(edges)
        merged = union if isinstance(union, LineString) else linemerge(union)
        snapped.append(
            (merged,) if isinstance(merged, LineString) else tuple(merged.geoms)
        )
    return snapped


def _wall_geometry(
    binning: Binning, arena: Arena
) -> tuple[
    Sequence[Polygon | MultiPolygon],
    Sequence[LineString],
    NDArray[np.float64],
    Polygon | MultiPolygon | None,
    NDArray[np.float64] | None,
]:
    """Cells, wall segments, centers and arena domain in the frame walls are checked in.

    The last value is None for the published cells and walls. For a stored
    source frame recovered through transforms, it bounds how far each cell's
    boundary there can be from the exact image of the published cell.
    """
    frame = wall_source_frame(binning.locator, arena, binning.bins.centers)
    if frame is None:
        return (
            materialize_cells(binning),
            arena.walls,
            binning.bins.centers,
            arena.polygonal_domain(),
            None,
        )
    # Shared snapshots avoid independently rounded collinear vertices.
    # Retained source-frame cells preserve current clipping and ownership.
    cells, segments, centers, _, frames, errors = frame
    # The innermost transformed arena wraps the arena the walls belong to.
    return cells, segments, centers, frames[-1].base.polygonal_domain(), errors


def _face_band(
    cell_u: Polygon | MultiPolygon,
    cell_v: Polygon | MultiPolygon,
    error_u: float,
    error_v: float,
) -> shapely.Geometry:
    """Where the face of two cells known to within ``error_u``/``error_v`` can be."""
    return cell_u.buffer(error_u).intersection(cell_v.buffer(error_v))


def _face_cover(
    wall_segments: Sequence[LineString], band: shapely.Geometry, width: float
) -> tuple[Literal["none", "part", "whole"], int | None]:
    """How much of a face known only to within ``band`` the walls run along.

    ``band`` is the thin strip where the face can lie and ``width`` how far
    its position is uncertain across it. A wall piece that enters the strip
    through one long side and leaves through the other crosses the face at a
    single point, at any angle, and runs along none of it. Any other piece
    that extends more than ``width`` along the strip could lie on the face.
    Returns ``"whole"`` when such pieces cover the strip's length to within
    ``width`` at each end, ``"part"`` when they cover less, and ``"none"``
    when there are none, with the index of the first wall that runs along
    (None when none does).
    """
    outline = shapely.get_coordinates(band)
    center = outline.mean(axis=0)
    # The strip's long axis is the face direction; its normal points across.
    axes = np.linalg.svd(outline - center)[2]
    along_band = (outline - center) @ axes[0]
    across_band = (outline - center) @ axes[1]
    start, stop = along_band.min(), along_band.max()
    middle = (across_band.min() + across_band.max()) / 2
    spans = []
    first: int | None = None
    candidates = np.flatnonzero(
        shapely.intersects(np.asarray(wall_segments, dtype=object), band)
    )
    for number in candidates:
        wall = wall_segments[number]
        wall_ends = shapely.get_coordinates(wall)[[0, -1]]
        for piece in shapely.get_parts(wall.intersection(band)):
            points = shapely.get_coordinates(piece)
            if len(points) < 2:
                continue
            ends = points[[0, -1]]
            along = (points - center) @ axes[0]
            sides = (ends - center) @ axes[1] - middle
            crosses = (
                # Neither end is the wall's own end inside the strip, and the
                # piece leaves through the long sides, not the strip's ends.
                not np.any(np.all(ends[:, None, :] == wall_ends[None], axis=2))
                and along[[0, -1]].min() > start + width
                and along[[0, -1]].max() < stop - width
                and sides[0] * sides[1] < 0
            )
            if not crosses and np.ptp(along) > width:
                spans.append((along.min(), along.max()))
                if first is None:
                    first = int(number)
    if not spans:
        return "none", first
    covered, reach = 0.0, start
    for low, high in sorted(spans):
        covered += max(0.0, high - max(low, reach))
        reach = max(reach, high)
    return ("whole" if stop - start - covered <= 2 * width else "part"), first


def _unresolved_face_message(
    arena: Arena,
    wall: int,
    pair: tuple[int, int],
    bound: float,
    kind: Literal["along", "crosses"],
) -> str:
    """Say which wall may block the face of ``pair`` and why that is unknown.

    ``kind`` is whether the wall runs along part of the face or crosses the
    line between the two centers.
    """
    if kind == "along":
        how = "runs along part of the face between"
        fix = (
            "clip the environment before transforming it. A wall meant to lie "
            f"off the bin edge can instead be moved more than {bound:.3g} from "
            "it, or the coordinates recentered near the origin, where rounding "
            "is smaller"
        )
    else:
        how = "crosses the line between the centers of"
        fix = "clip the environment before transforming it"
    return _format_error(
        f"Physical wall {arena.wall_names[wall]!r} {how} bins {pair[0]} and "
        f"{pair[1]}, so whether it blocks their face is unresolved.",
        why="Why: the bins were clipped after being rotated, scaled or moved, "
        "so in the coordinates the walls were declared in, the face between "
        f"them is only known to within {bound:.3g}, too coarse to tell whether "
        "this wall lies on the face or inside one of the bins.",
        fix=fix,
    )


def _uncertain_face(
    arena: Arena,
    cells: Sequence[Polygon | MultiPolygon],
    wall_segments: Sequence[LineString],
    walls: shapely.Geometry,
    centers: NDArray[np.float64],
    errors: NDArray[np.float64],
    pair: tuple[int, int],
    contact: shapely.Geometry,
) -> Literal["closed", "open"]:
    """Decide a published face whose source-frame cells are only approximate.

    The face lies somewhere in the strip the two cells' error bounds allow.
    Returns ``"closed"`` when walls run along the whole strip and ``"open"``
    when no wall runs along it: cells that meet only at a rounded point are
    not closed through it.

    Raises
    ------
    ValueError
        If walls run along only part of the strip, or if the source cells meet
        only at a point and a wall crosses the line between their centers.
    """
    u, v = pair
    bound = errors[u] + errors[v]
    band = _face_band(cells[u], cells[v], errors[u], errors[v])
    cover, wall = _face_cover(wall_segments, band, bound)
    if cover == "part":
        assert wall is not None  # a wall runs along part of the strip
        raise ValueError(_unresolved_face_message(arena, wall, pair, bound, "along"))
    if cover == "whole":
        return "closed"
    link = LineString(centers[[u, v]])
    if contact.length == 0 and link.crosses(walls):
        crossing = next(
            number
            for number, segment in enumerate(wall_segments)
            if segment.intersects(link)
        )
        raise ValueError(
            _unresolved_face_message(arena, crossing, pair, bound, "crosses")
        )
    return "open"


def _open_length(
    face: shapely.Geometry,
    walls: shapely.Geometry,
    domain: Polygon | MultiPolygon | None,
) -> float:
    """Length of ``face`` that no wall covers.

    A gap no longer than :func:`_wall_end_slack` against the arena's edge is
    where :func:`wall_inside` pulled a clipped wall end in to keep it inside
    the arena; it counts as covered.
    """
    gaps = shapely.get_parts(face.difference(walls))
    gaps = gaps[shapely.length(gaps) > 0]
    if domain is not None and len(gaps):
        slack = _wall_end_slack(domain)
        rounding = (shapely.length(gaps) <= slack) & (
            shapely.distance(gaps, domain.boundary) <= slack
        )
        gaps = gaps[~rounding]
    return float(np.sum(shapely.length(gaps)))


def _closing_walls(
    connections: BinConnections,
    cells: Sequence[Polygon | MultiPolygon],
    segments: Sequence[LineString],
    centers: NDArray[np.float64],
    errors: NDArray[np.float64] | None,
) -> NDArray[np.intp]:
    """Index of the wall closing each closed pair; see ``BinConnections.closed_by``.

    Follows the rules :func:`apply_walls` closes pairs by: the wall covering
    most of the edge the cells share; for cells that meet only at a corner,
    the first wall the corner touches, else the first wall the line between
    their centers crosses; and for cells known only to within ``errors``,
    the wall running furthest along the strip where their face can lie.
    """
    walls = np.asarray(segments, dtype=object)
    closed_by = np.full(len(connections.all_pairs), -1, dtype=np.intp)
    for row in np.flatnonzero(~connections.is_open):
        u, v = connections.all_pairs[row]
        contact = cells[u].boundary.intersection(cells[v].boundary)
        if contact.length > 0:
            cover = shapely.length(shapely.intersection(contact, walls))
        else:
            link = LineString(centers[[u, v]])
            cover = 2.0 * shapely.intersects(contact, walls) + shapely.crosses(
                link, walls
            )
        if not np.any(cover > 0) and errors is not None and errors[u] + errors[v] > 0:
            band = _face_band(cells[u], cells[v], errors[u], errors[v])
            cover = shapely.length(shapely.intersection(walls, band))
        best = int(np.argmax(cover))
        if cover[best] > 0:
            closed_by[row] = best
    return closed_by


def apply_walls(binning: Binning, arena: Arena) -> Binning:
    """Close the neighbor pairs whose shared edge walls block completely.

    Only the part of a shared edge inside the arena counts: a bin at a slanted
    outline sticks out past it, where no wall can be. Cells that meet only at
    a corner close when the corner touches a wall or the line between their
    centers crosses one. A closed pair stays listed in ``binning.connections``
    but leaves paths and finite-volume faces. A partly covered edge stays open
    with the open share of its face. Pairs already closed stay closed. Every
    closed pair a wall of ``arena`` would close records that wall in
    ``connections.closed_by``; a pair closed for another reason records -1.
    """
    if _composition_arena(binning.locator) is not None:
        # Each owned child already carries its certified wall geometry. Rechecking
        # its independently rounded global polygons loses the native source frame.
        # New bridge crossings are checked when constructing/validating the root.
        return binning
    links = binning.connections
    if not arena.walls:
        if np.all(links.closed_by < 0):
            return binning
        # No wall of this arena closes a pair; drop indices into other walls.
        unattributed = np.full(len(links.closed_by), -1, dtype=np.intp)
        return replace(binning, connections=replace(links, closed_by=unattributed))

    cells, wall_segments, centers, domain, errors = _wall_geometry(binning, arena)
    walls = unary_union(wall_segments)
    if errors is not None:
        # Cells whose uncertain boundary can reach a wall.
        near_wall = shapely.dwithin(np.asarray(cells, dtype=object), walls, errors)
    reaching = _reaching(wall_segments, domain)
    for index, cell in enumerate(cells):
        if _divides(cell, reaching, domain):
            raise ValueError(
                _wall_cut_message(binning, arena, wall_segments, cell, index)
            )
    pairs = binning.connections.pairs
    keep = np.ones(len(pairs), dtype=bool)
    # A face partly covered by a wall keeps its open share of the conductance.
    open_share = np.ones(len(pairs))
    current_cells = None

    def published_contact(u: int, v: int) -> float:
        """Length of the face of ``u`` and ``v`` in the published cells."""
        nonlocal current_cells
        if current_cells is None:
            current_cells = materialize_cells(binning)
        return float(
            current_cells[u].boundary.intersection(current_cells[v].boundary).length
        )

    for index, (u, v) in enumerate(pairs):
        contact = cells[u].boundary.intersection(cells[v].boundary)
        if (
            errors is not None
            and errors[u] + errors[v] > 0
            and near_wall[u]
            and near_wall[v]
            and published_contact(u, v) > 0
        ):
            # The source cells are known only to within their error bounds, so
            # whether they touch is rounding. Decide from the strip where the
            # face of the published cells can lie instead.
            decision = _uncertain_face(
                arena,
                cells,
                wall_segments,
                walls,
                centers,
                errors,
                (int(u), int(v)),
                contact,
            )
            keep[index] = decision == "open"
            continue
        blocked = contact.intersection(walls)
        if contact.length > 0 and blocked.length > 0:
            # Only the part of the edge inside the arena is a face an animal
            # can cross: a bin at a slanted outline sticks out past it.
            face = contact if domain is None else contact.intersection(domain)
            opening = _open_length(face, walls, domain)
            if opening > 0:
                open_share[index] = opening / face.length
            else:
                keep[index] = False
        elif contact.length == 0:
            link = LineString(centers[[u, v]])
            crosses = link.crosses(walls)
            if errors is not None and crosses and published_contact(u, v) > 0:
                raise ValueError(
                    _unresolved_face_message(
                        arena,
                        next(
                            number
                            for number, segment in enumerate(wall_segments)
                            if segment.intersects(link)
                        ),
                        (int(u), int(v)),
                        errors[u] + errors[v],
                        "crosses",
                    )
                )
            keep[index] = contact.intersection(walls).is_empty and not crosses
    connections = links.closed(keep)
    connections = replace(
        connections,
        closed_by=_closing_walls(connections, cells, wall_segments, centers, errors),
    )
    if np.all(keep) and np.all(open_share == 1.0):
        return replace(binning, connections=connections)
    faces = binning.faces
    if faces.available:
        source = faces.build()
        area = source.A.copy()
        partial = np.flatnonzero(open_share < 1.0)
        if len(partial):
            # The open length in the published coordinates; taking the
            # minimum keeps a face that was already narrowed unchanged when
            # the environment is published again.
            for index in partial:
                u, v = pairs[index]
                full = published_contact(u, v)
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
