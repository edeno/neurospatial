"""Assign positions to bins inside a known continuous domain."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arena import EgocentricSurround

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._locators import LookupIndexes, RectilinearLocator


def locate(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    indexes: LookupIndexes | None = None,
    *,
    call: str,
) -> NDArray[np.intp]:
    """Assign positions to active bins, applying known arena containment first.

    Parameters
    ----------
    binning : Binning
        Published bins and their point locator.
    arena : Arena
        Continuous domain; positions outside a known domain map to ``-1``.
    positions : ndarray of float64, shape (n_positions, n_dims)
        Query positions in the environment's coordinates. Egocentric polar
        angles are first wrapped into the angle range, so ``angle + 2π``
        finds the same bin as ``angle``.
    indexes : LookupIndexes, optional
        Keeps the locator's search structures, usually the environment's
        analysis cache. None builds them for this call only.
    call : str
        The operation the scientist called, such as ``"Environment.bin_at"``,
        named in the error for positions with the wrong number of columns.

    Returns
    -------
    ndarray of intp, shape (n_positions,)
        Active-bin index of each position, or ``-1`` outside.

    Raises
    ------
    ValueError
        If ``positions`` is not one row of ``n_dims`` coordinates per position
        (see :func:`require_position_columns`).
    NotImplementedError
        If the bins lie along a track in two or more dimensions whose width
        is unknown (see :func:`require_track_width`), or the domain is unknown
        and nearest-bin lookup has no radius (see
        :func:`require_lookup_radius`).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_dims = binning.bins.n_dims
    if positions.shape == (n_dims,):
        # One point given as a flat array is one row.
        positions = positions[np.newaxis]
    require_position_columns(positions, n_dims, call=call)
    if not arena.domain_known:
        require_track_width(binning, arena)
        require_lookup_radius(binning)
        return binning.locator.locate(positions, indexes).astype(np.intp, copy=False)
    if isinstance(arena, EgocentricSurround):
        # An angle a whole turn outside the range names a direction inside it.
        positions = arena.wrap_angles(positions)
    inside = arena.contains(positions)
    indices = np.full(len(positions), -1, dtype=np.intp)
    if np.any(inside):
        indices[inside] = binning.locator.locate(positions[inside], indexes)
    _snap_outline_slivers(binning, arena, positions, inside, indices)
    return indices


def _snap_outline_slivers(
    binning: Binning,
    arena: Arena,
    positions: NDArray[np.float64],
    inside: NDArray[np.bool_],
    indices: NDArray[np.intp],
) -> None:
    """Give in-arena positions in a grid cell dropped at the outline a bin.

    A grid keeps the cells whose centers lie strictly inside the arena's
    outline, so slivers of a curved or diagonal wall, and the cells whose
    centers lie exactly on a wall, are inside the arena but in no bin. A
    position inside such a cell (one of the grid's cells whose center is not
    strictly inside) belongs to the nearest neighboring cell whose center is
    strictly inside, less than one cell away, when that cell is a bin. The
    same strict test as bin retention decides both, so a grid whose cell
    centers fall on an outline wall (a 10 cm corridor on 2 cm bins) gives
    those positions a bin. A cell removed on purpose (by ``subset``) keeps its
    sliver positions outside, as do positions beyond the grid. A rotated,
    reflected or sheared grid applies the rule to its source cells
    (``binning.grid_frame``), measuring distance there, so a transform moves
    positions and their bins together. Updates ``indices`` in place.
    """
    orphan = inside & (indices < 0)
    if not np.any(orphan):
        return
    grid = _sliver_grid(binning)
    if grid is None:
        return
    cells_locator, to_source, to_world = grid
    edges = cells_locator.edges
    shape = np.asarray(cells_locator.shape)
    rows = np.flatnonzero(orphan)
    source = to_source(positions[rows])
    # A point on the grid's last edge lies in the last cell (closed outline).
    cells = np.column_stack(
        [
            np.where(
                source[:, axis] == edge[-1],
                len(edge) - 2,
                np.searchsorted(edge, source[:, axis], side="right") - 1,
            )
            for axis, edge in enumerate(edges)
        ]
    )
    in_grid = np.all((cells >= 0) & (cells < shape), axis=1)
    rows, cells, source = rows[in_grid], cells[in_grid], source[in_grid]
    if rows.size == 0:
        return
    in_arena = _kept_center_test(arena, edges, to_world)
    sliver = ~in_arena(_cell_centers(edges, cells))
    rows, cells, source = rows[sliver], cells[sliver], source[sliver]
    if rows.size == 0:
        return
    neighbors, valid = _sliver_owners(cells, edges, in_arena)
    centers = _cell_centers(edges, neighbors)
    # Distances in the grid's own frame break ties between equidistant
    # neighbors the same way before and after a rotation or reflection.
    distance = np.linalg.norm(centers - source[:, None, :], axis=2)
    distance[~valid] = np.inf
    best = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(len(rows)), best])
    owner = neighbors[np.arange(len(rows)), best][found]
    flat = np.ravel_multi_index(tuple(owner.T), cells_locator.shape)
    indices[rows[found]] = cells_locator._active_index(flat)


def outline_sliver_regions(binning: Binning, arena: Arena) -> list[tuple[int, Polygon]]:
    """The arena inside planar grid cells dropped at the outline, by owning bin.

    :func:`locate` gives a position in such a sliver the nearest neighboring
    bin whose cell center is strictly inside (see
    :func:`_snap_outline_slivers`). These are the same assignments as
    polygons: each dropped cell is cut along the perpendicular bisectors
    between its owners' centers, in the grid's own frame. Exact polygon cells
    that take these parts, kept to the arena, cover every position the grid's
    lookup gives a bin.

    Returns
    -------
    list of (int, Polygon)
        Active-bin index and one part of a dropped cell that reaches into the
        arena, in world coordinates and not yet kept to it: intersect each
        part with the arena once, as the cells are, so that the parts and
        cells meet at identical vertices. Empty when the bins are not planar
        grid cells or the arena has no polygonal domain.
    """
    grid = _sliver_grid(binning)
    domain = arena.polygonal_domain()
    if grid is None or domain is None or len(grid[0].edges) != 2:
        return []
    cells_locator, _, to_world = grid
    edges = cells_locator.edges
    shape = np.asarray(cells_locator.shape)
    in_arena = _kept_center_test(arena, edges, to_world)
    # Only a cell next to a bin can have an owner.
    bins = np.argwhere(cells_locator.active_mask)
    near = (bins[:, None, :] + _neighbor_offsets(2)[None]).reshape(-1, 2)
    near = np.unique(near[np.all((near >= 0) & (near < shape), axis=1)], axis=0)
    # A bin whose center rounds onto the outline holds its own cell.
    dropped = (
        ~in_arena(_cell_centers(edges, near))
        & ~cells_locator.active_mask[near[:, 0], near[:, 1]]
    )
    slivers = near[dropped]
    if slivers.size == 0:
        return []
    neighbors, valid = _sliver_owners(slivers, edges, in_arena)
    owner_bins = cells_locator._active_index(
        np.ravel_multi_index((neighbors[..., 0], neighbors[..., 1]), tuple(shape))
    )
    owner_centers = _cell_centers(edges, neighbors)
    parts: list[tuple[int, Polygon]] = []
    for cell, owned, owner_bin, centers in zip(
        slivers, valid, owner_bins, owner_centers, strict=True
    ):
        (x0, x1), (y0, y1) = (
            edge[i : i + 2] for edge, i in zip(edges, cell, strict=True)
        )
        rect = shapely.box(x0, y0, x1, y1)
        reach = 4.0 * float(np.hypot(x1 - x0, y1 - y0))
        owners = np.flatnonzero(owned)
        # Each part is cut from what earlier owners left, so the parts share
        # their computed boundaries: no rounding overlap or gap between them.
        remaining = rect
        for position, j in enumerate(owners):
            nearest = remaining
            if position < len(owners) - 1:
                for k in owners[owners != j]:
                    nearest = nearest.intersection(
                        _closer_half_plane(centers[j], centers[k], reach)
                    )
                remaining = remaining.difference(nearest)
            if owner_bin[j] < 0 or nearest.is_empty:
                continue
            world = shapely.transform(nearest, to_world)
            if isinstance(world, Polygon) and world.intersection(domain).area > 0:
                parts.append((int(owner_bin[j]), world))
    return parts


def _closer_half_plane(
    center: NDArray[np.float64], other: NDArray[np.float64], reach: float
) -> Polygon:
    """The points within ``reach`` of the bisector that are nearer ``center``."""
    toward = (other - center) / np.hypot(*(other - center))
    along = np.array([-toward[1], toward[0]])
    middle = 0.5 * (center + other)
    return Polygon(
        [
            middle + reach * along,
            middle - reach * along,
            middle - reach * (along + toward),
            middle + reach * (along - toward),
        ]
    )


def _neighbor_offsets(n_dims: int) -> NDArray[np.intp]:
    """Offsets to every neighboring grid cell, shape (3**n_dims - 1, n_dims)."""
    offsets = np.array(list(np.ndindex(*(3,) * n_dims))) - 1
    return offsets[np.any(offsets != 0, axis=1)]


def _cell_centers(
    edges: tuple[NDArray[np.float64], ...], grid_cells: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Grid-frame coordinates of grid cell centers, shape (..., n_dims)."""
    return np.stack(
        [
            0.5 * (edge[grid_cells[..., axis]] + edge[grid_cells[..., axis] + 1])
            for axis, edge in enumerate(edges)
        ],
        axis=-1,
    )


def _kept_center_test(
    arena: Arena,
    edges: tuple[NDArray[np.float64], ...],
    to_world: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> Callable[[NDArray[np.float64]], NDArray[np.bool_]]:
    """Whether grid-frame cell centers lie strictly inside the arena.

    The rule that kept the grid's bins: a cell whose center lies on the
    outline was not kept, so it is a sliver too, not a missing bin.
    """
    domain = arena.polygonal_domain() if len(edges) == 2 else None

    def in_arena(grid_centers: NDArray[np.float64]) -> NDArray[np.bool_]:
        flat = to_world(grid_centers.reshape(-1, len(edges)))
        inside = (
            shapely.contains_xy(domain, flat[:, 0], flat[:, 1])
            if domain is not None
            else arena.contains(flat)
        )
        return np.asarray(inside).reshape(grid_centers.shape[:-1])

    return in_arena


def _sliver_owners(
    grid_cells: NDArray[np.intp],
    edges: tuple[NDArray[np.float64], ...],
    in_arena: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """The neighbors that may own positions in each dropped grid cell.

    Returns
    -------
    neighbors : ndarray of intp, shape (n_cells, 3**n_dims - 1, n_dims)
        Every neighboring cell, clipped to the grid.
    valid : ndarray of bool, shape (n_cells, 3**n_dims - 1)
        Whether the neighbor lies in the grid and its center is strictly
        inside the arena. A valid neighbor removed by ``subset`` is no bin.
    """
    shape = np.array([len(edge) - 1 for edge in edges])
    neighbors = grid_cells[:, None, :] + _neighbor_offsets(len(edges))[None, :, :]
    valid = np.all((neighbors >= 0) & (neighbors < shape), axis=2)
    neighbors = np.clip(neighbors, 0, shape - 1)
    valid &= in_arena(_cell_centers(edges, neighbors))
    return neighbors, valid


def _identity(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return points


def _sliver_grid(
    binning: Binning,
) -> (
    tuple[
        RectilinearLocator,
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ]
    | None
):
    """The grid cells of ``binning`` with maps between world and grid frame.

    Returns
    -------
    tuple or None
        ``(cells, to_source, to_world)``: the cells as a locator whose
        ``_active_index`` numbers the current bins, a map from ``(n_points,
        n_dims)`` world points into the cells' frame (snapping onto a face
        within the map's roundoff, as lookup does) and its inverse. None when
        the bins are not grid cells.
    """
    from neurospatial.geometry._locators import RectilinearLocator

    if isinstance(binning.locator, RectilinearLocator):
        return binning.locator, _identity, _identity
    frame = binning.grid_frame
    if frame is None:
        return None

    def to_source(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return frame._project_to_planes(
            points,
            frame.source.edges,
            operation="Bin lookup on a rotated or sheared grid",
        )

    return frame._source_locator(), to_source, frame.to_world


def require_position_columns(
    positions: NDArray[np.float64],
    n_dims: int,
    *,
    call: str,
    name: str = "positions",
) -> None:
    """Refuse positions that are not one row of ``n_dims`` coordinates each.

    Every lookup of positions in an environment's bins applies this rule, so
    the same mistake gets the same message from every operation.

    Parameters
    ----------
    positions : ndarray of float64
        The array as given, expected shape (n_positions, n_dims).
    n_dims : int
        Number of coordinates of each bin center.
    call : str
        The operation the scientist called, named in the message.
    name : str, default="positions"
        The argument that held ``positions``, such as ``"points"``.

    Raises
    ------
    ValueError
        If ``positions`` is not two-dimensional with ``n_dims`` columns.
    """
    if positions.ndim == 2 and positions.shape[1] == n_dims:
        return
    one = name.removesuffix("s")
    why = (
        f"Why: each row is one {one}, with one column per coordinate of "
        "env.bin_centers."
    )
    fix = f"pass an array of shape (n_{name}, {n_dims}), one row per {one}"
    if n_dims == 1:
        fix += f"; make a 1-D array one column with np.reshape({name}, (-1, 1))"
    elif positions.shape == (n_dims,):
        fix += f"; one {one} is one row, np.reshape({name}, (1, -1))"
    elif positions.ndim == 1 or (positions.ndim == 2 and positions.shape[1] == 1):
        why += (
            " Linearized positions from to_linear() are distances along a "
            "track, not bin coordinates."
        )
        fix += f"; use the original {n_dims}-D coordinates, not linearized ones"
    raise ValueError(
        _format_error(
            f"{call} got {wrong_position_columns(positions.shape, n_dims, name)}.",
            why=why,
            fix=fix,
        )
    )


def wrong_position_columns(
    shape: tuple[int, ...], n_dims: int, name: str = "positions"
) -> str:
    """Describe positions of ``shape`` that do not match ``n_dims``-D bins.

    The clause that :func:`require_position_columns` and checks that report
    several input problems together use for this mistake.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array as given.
    n_dims : int
        Number of coordinates of each bin center.
    name : str, default="positions"
        The argument that held the positions.

    Returns
    -------
    str
        A clause such as ``"positions with shape (5, 3), but this
        environment's bins are 2-D, so positions needs shape
        (n_positions, 2)"``.
    """
    return (
        f"{name} with shape {shape}, but this environment's bins are "
        f"{n_dims}-D, so {name} needs shape (n_{name}, {n_dims})"
    )


TRACK_WIDTH_FIX = (
    "give the track its measured full width, in the units of the positions, "
    "with env.with_track_width(width); it keeps the bins and works on a saved, "
    "subset or transformed track too. A new track takes track_width= in "
    "Environment.linear_track, Environment.maze or Environment.from_graph"
)


def require_lookup_radius(binning: Binning) -> None:
    """Refuse nearest-bin lookup that has no radius and no known domain.

    Called only when the environment's domain is unknown. Bins located by
    their nearest center, without ``max_distance``, would then put every
    position, however far away, in some bin. Unlimited lookup must be chosen
    explicitly with ``max_distance=math.inf``.

    Parameters
    ----------
    binning : Binning
        The bins and their locator, possibly wrapped by subsets or affine
        maps. A combined environment's components whose domain is unknown
        are checked in turn, as if each were looked up on its own.

    Raises
    ------
    NotImplementedError
        If the locator, or that of a component with an unknown domain, finds
        the nearest bin center with ``max_distance`` None, or such a
        component is a track without a width (see
        :func:`require_track_width`).
    """
    from neurospatial.geometry._composition import CombinedLocator
    from neurospatial.geometry._locators import (
        NearestBinLocator,
        SubsetLocator,
        TransformedLocator,
    )

    locator = binning.locator
    while isinstance(locator, (SubsetLocator, TransformedLocator)):
        locator = locator.base
    if isinstance(locator, CombinedLocator):
        for child, arena in zip(locator.children, locator.arenas, strict=True):
            if not arena.domain_known:
                require_track_width(child, arena)
                require_lookup_radius(child)
        return
    if (
        not isinstance(locator, NearestBinLocator)
        or locator.max_distance is not None
        # With no bins, no position can be put in a wrong one.
        or binning.bins.n_bins == 0
    ):
        return
    raise NotImplementedError(
        _format_error(
            "Positions cannot be assigned to these bins: they are found by "
            "nearest bin center with no max_distance, and the environment has "
            "no known domain.",
            why="Why: without a radius or a domain every position, however far "
            "from the bins (a tracking jump or a position from another "
            "arena), would be put in its nearest bin.",
            fix="build the locator with NearestBinLocator(centers, "
            "max_distance=radius), a radius in the bin centers' length units; "
            "pass max_distance=math.inf to deliberately give every position its "
            "nearest bin; or pass the arena that bounds the bins to "
            "Environment.from_bins(..., arena=...)",
        )
    )


def require_track_width(binning: Binning, arena: Arena) -> None:
    """Refuse to map positions to a track's bins when its width is unknown.

    A centerline and a bin size do not say how far from the centerline a
    position is still on the track, so without a width every position, however
    far off the track, would be put in its nearest bin. A track whose
    positions have one coordinate has no off-track direction: its arena knows
    its domain (the centerline, ends included) and never reaches this check.

    Parameters
    ----------
    binning : Binning
        The bins; a track carries a linearization.
    arena : Arena
        The continuous domain; a track built with a width knows it.

    Raises
    ------
    NotImplementedError
        If the bins lie along a track in two or more dimensions and the arena
        does not know the track's footprint.
    """
    if (
        binning.linearization is not None
        and binning.bins.n_dims >= 2
        and not arena.domain_known
    ):
        raise NotImplementedError(
            _format_error(
                "Positions cannot be assigned to the bins of a track built "
                "without track_width.",
                why="Why: a centerline and bin size do not say how far from the "
                "centerline a position is still on the track, so a position "
                "far off the track (a reflection or a tracking jump) would be "
                "put in its nearest bin.",
                fix=TRACK_WIDTH_FIX,
            )
        )


def locate_outline_points(
    binning: Binning,
    arena: Arena,
    points: NDArray[np.float64],
    indexes: LookupIndexes | None = None,
) -> NDArray[np.intp]:
    """Assign points of the arena's own outline, such as wall pieces, to bins.

    Like :func:`locate` for a position inside the arena, without testing the
    containment that floating-point roundoff makes unreliable on the outline
    itself: the bin whose cell holds the point, else (in a sliver whose grid
    cell was not kept) the neighboring bin that owns positions there, else -1.

    Parameters
    ----------
    binning : Binning
        Published bins and their point locator.
    arena : Arena
        The arena whose outline the points lie on.
    points : ndarray of float64, shape (n_points, n_dims)
        Points on the outline, in the environment's coordinates.
    indexes : LookupIndexes, optional
        The locator's search structures, as for :func:`locate`.

    Returns
    -------
    ndarray of intp, shape (n_points,)
        Bin of each point, or -1 when no bin owns it.
    """
    points = np.asarray(points, dtype=np.float64)
    indices = binning.locator.locate(points, indexes).astype(np.intp, copy=False)
    _snap_outline_slivers(
        binning, arena, points, np.ones(len(points), dtype=bool), indices
    )
    return indices
