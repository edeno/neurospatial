"""Assign positions to bins inside a known continuous domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arena import EgocentricSurround

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._locators import LookupIndexes


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
    do positions beyond the grid. Updates ``indices`` in place.
    """
    from neurospatial.geometry._locators import RectilinearLocator

    locator = binning.locator
    orphan = inside & (indices < 0)
    if not np.any(orphan) or not isinstance(locator, RectilinearLocator):
        return
    shape = np.asarray(locator.shape)
    rows = np.flatnonzero(orphan)
    cells = np.column_stack(
        [
            np.searchsorted(edge, positions[rows, axis], side="right") - 1
            for axis, edge in enumerate(locator.edges)
        ]
    )
    in_grid = np.all((cells >= 0) & (cells < shape), axis=1)
    rows, cells = rows[in_grid], cells[in_grid]
    if rows.size == 0:
        return

    def centers_of(grid_cells: NDArray[np.intp]) -> NDArray[np.float64]:
        return np.stack(
            [
                0.5 * (edge[grid_cells[..., axis]] + edge[grid_cells[..., axis] + 1])
                for axis, edge in enumerate(locator.edges)
            ],
            axis=-1,
        )

    sliver = ~arena.contains(centers_of(cells))
    rows, cells = rows[sliver], cells[sliver]
    if rows.size == 0:
        return
    # Every neighboring cell, (n_rows, n_offsets, n_dims).
    offsets = np.array(list(np.ndindex(*(3,) * len(shape)))) - 1
    offsets = offsets[np.any(offsets != 0, axis=1)]
    neighbors = cells[:, None, :] + offsets[None, :, :]
    valid = np.all((neighbors >= 0) & (neighbors < shape), axis=2)
    neighbors = np.clip(neighbors, 0, shape - 1)
    centers = centers_of(neighbors)
    flat_centers = centers.reshape(-1, len(shape))
    valid &= arena.contains(flat_centers).reshape(valid.shape)
    distance = np.linalg.norm(centers - positions[rows, None, :], axis=2)
    distance[~valid] = np.inf
    best = np.argmin(distance, axis=1)
    found = np.isfinite(distance[np.arange(len(rows)), best])
    owner = neighbors[np.arange(len(rows)), best][found]
    flat = np.ravel_multi_index(tuple(owner.T), locator.shape)
    indices[rows[found]] = locator._active_index(flat)


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
