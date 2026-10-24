"""Assign positions to bins inside a known continuous domain."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
        is unknown (see :func:`require_track_width`).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_dims = binning.bins.n_dims
    if positions.shape == (n_dims,):
        # One point given as a flat array is one row.
        positions = positions[np.newaxis]
    require_position_columns(positions, n_dims, call=call)
    if not arena.domain_known:
        require_track_width(binning, arena)
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

    A grid keeps the cells whose centers lie inside the arena's outline, so
    slivers of a curved or diagonal wall are inside the arena but in no bin.
    A position inside such a cell (one of the grid's cells, whose center lies
    outside the arena) belongs to the nearest neighboring cell whose center
    lies inside, less than one cell away, when that cell is a bin. A cell
    removed on purpose (by ``subset``) keeps its sliver positions outside, as
    do positions beyond the grid. A rotated, reflected or sheared grid
    applies the rule to its source cells (``binning.grid_frame``), measuring
    distance there, so a transform moves positions and their bins together.
    Updates ``indices`` in place.
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

    def centers_of(grid_cells: NDArray[np.intp]) -> NDArray[np.float64]:
        """Grid-frame coordinates of grid cell centers, shape (..., n_dims)."""
        return np.stack(
            [
                0.5 * (edge[grid_cells[..., axis]] + edge[grid_cells[..., axis] + 1])
                for axis, edge in enumerate(edges)
            ],
            axis=-1,
        )

    def in_arena(grid_centers: NDArray[np.float64]) -> NDArray[np.bool_]:
        flat = grid_centers.reshape(-1, len(edges))
        return np.asarray(arena.contains(to_world(flat))).reshape(
            grid_centers.shape[:-1]
        )

    sliver = ~in_arena(centers_of(cells))
    rows, cells, source = rows[sliver], cells[sliver], source[sliver]
    if rows.size == 0:
        return
    # Every neighboring cell, (n_rows, n_offsets, n_dims).
    offsets = np.array(list(np.ndindex(*(3,) * len(shape)))) - 1
    offsets = offsets[np.any(offsets != 0, axis=1)]
    neighbors = cells[:, None, :] + offsets[None, :, :]
    valid = np.all((neighbors >= 0) & (neighbors < shape), axis=2)
    neighbors = np.clip(neighbors, 0, shape - 1)
    centers = centers_of(neighbors)
    valid &= in_arena(centers)
    # Distances in the grid's own frame break ties between equidistant
    # neighbors the same way before and after a rotation or reflection.
    distance = np.linalg.norm(centers - source[:, None, :], axis=2)
    distance[~valid] = np.inf
    best = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(len(rows)), best])
    owner = neighbors[np.arange(len(rows)), best][found]
    flat = np.ravel_multi_index(tuple(owner.T), cells_locator.shape)
    indices[rows[found]] = cells_locator._active_index(flat)


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


def require_track_width(binning: Binning, arena: Arena) -> None:
    """Refuse to map positions to a track's bins when its width is unknown.

    A centerline and a bin size do not say how far from the centerline a
    position is still on the track, so without a width every position, however
    far off the track, would be put in its nearest bin. A track whose
    positions have one coordinate has no off-track direction and is not
    refused.

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
