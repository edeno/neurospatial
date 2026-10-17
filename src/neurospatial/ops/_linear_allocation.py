"""Split straight trajectory intervals across the rectangular cells they cross.

The ray-grid traversal (a digital differential analyzer) finds where each
straight segment between consecutive samples crosses cell edges. Each piece
between crossings belongs to the cell at its midpoint, and receives the
interval's duration in proportion to its length.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.geometry import RectilinearGrid

# Ray direction components and segment lengths below this are treated as zero:
# a segment shorter than this is a stationary sample, and a direction component
# smaller than this is parallel to that axis's cell edges (no crossings).
RAY_EPSILON = 1e-12

# Candidate edge crossings handled per block. Each crossing holds about 200
# bytes of intermediate arrays, so a block stays near 50 MB however far the
# trajectory jumps between samples.
BLOCK_CROSSINGS = 1 << 18


def _grid_cells(
    points: NDArray[np.float64],
    edges: Sequence[NDArray[np.float64]],
    shape: tuple[int, ...],
) -> NDArray[np.intp]:
    """Row-major flat cell index of each point, or ``-1`` outside the grid.

    Cells are half-open, ``[edges[i], edges[i + 1])``, along every axis.

    Parameters
    ----------
    points : ndarray of float64, shape (n_points, n_dims)
    edges : sequence of ndarray
        Cell edges per axis.
    shape : tuple of int
        Number of cells per axis.

    Returns
    -------
    ndarray of intp, shape (n_points,)
    """
    flat = np.zeros(len(points), dtype=np.intp)
    inside = np.ones(len(points), dtype=bool)
    stride = 1
    for axis in reversed(range(len(shape))):
        index = np.searchsorted(edges[axis], points[:, axis], side="right") - 1
        inside &= (index >= 0) & (index < shape[axis])
        flat += index * stride
        stride *= shape[axis]
    flat[~inside] = -1
    return flat


def crossing_blocks(
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    edges: Sequence[NDArray[np.float64]],
    *,
    max_crossings: int = BLOCK_CROSSINGS,
) -> list[slice]:
    """Split intervals into consecutive blocks with a bounded number of crossings.

    Parameters
    ----------
    starts, stops : ndarray of float64, shape (n_intervals, n_dims)
        Segment endpoints of each interval.
    edges : sequence of ndarray
        Cell edges per axis.
    max_crossings : int
        Budget of candidate edge crossings per block. A single interval with
        more candidates than the budget forms a block of its own.

    Returns
    -------
    list of slice
        Consecutive blocks covering ``range(n_intervals)`` in order.
    """
    # Upper bound on the crossings ray_grid_pieces examines for each interval,
    # plus one for the interval's own boundaries.
    counts = np.ones(len(starts), dtype=np.int64)
    for axis, axis_edges in enumerate(edges):
        low = np.minimum(starts[:, axis], stops[:, axis])
        high = np.maximum(starts[:, axis], stops[:, axis])
        first = np.searchsorted(axis_edges, low, side="left") - 1
        last = np.searchsorted(axis_edges, high, side="right") + 1
        counts += np.clip(last - first, 0, len(axis_edges))
    total = np.cumsum(counts)
    blocks = []
    start = 0
    while start < len(starts):
        done = total[start - 1] if start else 0
        stop = int(np.searchsorted(total, done + max_crossings, side="right"))
        stop = max(stop, start + 1)
        blocks.append(slice(start, stop))
        start = stop
    return blocks


def ray_grid_pieces(
    starts: NDArray[np.float64],
    stops: NDArray[np.float64],
    edges: Sequence[NDArray[np.float64]],
    shape: tuple[int, ...],
    durations: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Split straight intervals at the cell edges they cross.

    Parameters
    ----------
    starts, stops : ndarray of float64, shape (n_intervals, n_dims)
        Segment endpoints of each interval.
    edges : sequence of ndarray
        Cell edges per axis; ``edges[d]`` has shape ``(shape[d] + 1,)``.
    shape : tuple of int
        Number of cells per axis, active or not.
    durations : ndarray of float64, shape (n_intervals,)
        Time to divide along each interval.

    Returns
    -------
    interval : ndarray of intp, shape (n_pieces,)
        Interval each piece belongs to.
    cell : ndarray of intp, shape (n_pieces,)
        Row-major full-grid cell index of each piece.
    time : ndarray of float64, shape (n_pieces,)
        Duration given to each piece, proportional to its length.

    Notes
    -----
    Pieces are in interval order and, within an interval, in order along the
    segment, so sequential accumulation reproduces a per-interval traversal
    exactly. A segment shorter than ``RAY_EPSILON`` gives its whole duration to
    its start cell. Pieces whose midpoint lies outside the grid, and intervals
    with a non-finite endpoint, contribute nothing.
    """
    n_dims = starts.shape[1]
    delta = stops - starts
    lengths = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(lengths)
    still = finite & (lengths < RAY_EPSILON)
    moving = np.flatnonzero(finite & ~still)

    # Edge crossings: start + t * direction = edge, strictly inside the segment.
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = delta[moving] / lengths[moving, None]
    crossing_interval: list[NDArray[np.intp]] = []
    crossing_t: list[NDArray[np.float64]] = []
    for axis in range(n_dims):
        component = direction[:, axis]
        rows = np.flatnonzero(np.abs(component) >= RAY_EPSILON)
        axis_edges = edges[axis]
        start = starts[moving[rows], axis]
        stop = stops[moving[rows], axis]
        # One extra edge on each side covers crossings that rounding of the
        # parametric distance places just inside the segment.
        first = np.maximum(
            np.searchsorted(axis_edges, np.minimum(start, stop), side="left") - 1, 0
        )
        last = np.minimum(
            np.searchsorted(axis_edges, np.maximum(start, stop), side="right") + 1,
            len(axis_edges),
        )
        counts = last - first
        owner = np.repeat(rows, counts)
        offsets = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        edge = axis_edges[np.repeat(first, counts) + offsets]
        t = (edge - starts[moving[owner], axis]) / component[owner]
        inside = (t > 0) & (t < lengths[moving[owner]])
        crossing_interval.append(owner[inside])
        crossing_t.append(t[inside])

    # Segment boundaries per interval: 0, sorted crossings, then the length.
    n_moving = len(moving)
    local = np.concatenate(
        [np.arange(n_moving), *crossing_interval, np.arange(n_moving)]
    )
    value = np.concatenate([np.zeros(n_moving), *crossing_t, lengths[moving]])
    position = np.concatenate(
        [
            np.zeros(n_moving, dtype=np.int8),
            np.ones(len(local) - 2 * n_moving, dtype=np.int8),
            np.full(n_moving, 2, dtype=np.int8),
        ]
    )
    order = np.lexsort((value, position, local))
    local, value = local[order], value[order]
    same = local[:-1] == local[1:]
    owner = local[:-1][same]
    low, high = value[:-1][same], value[1:][same]
    intervals = moving[owner]
    midpoint = starts[intervals] + ((low + high) / 2)[:, None] * direction[owner]
    piece_cells = _grid_cells(midpoint, edges, shape)
    piece_time = durations[intervals] * ((high - low) / lengths[intervals])

    still_rows = np.flatnonzero(still)
    still_cells = _grid_cells(starts[still_rows], edges, shape)

    interval = np.concatenate([still_rows, intervals])
    cell = np.concatenate([still_cells, piece_cells])
    time = np.concatenate([durations[still_rows], piece_time])
    keep = cell >= 0
    order = np.argsort(interval[keep], kind="stable")
    return interval[keep][order], cell[keep][order], time[keep][order]


def linear_grid_occupancy(
    n_bins: int,
    grid: RectilinearGrid,
    positions: NDArray[np.float64],
    dt: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    bin_indices: NDArray[np.intp],
    *,
    return_seconds: bool,
) -> NDArray[np.float64]:
    """Allocate each valid interval across the grid cells its straight path crosses.

    Parameters
    ----------
    n_bins : int
        Number of active bins.
    grid : RectilinearGrid
        Per-axis cell edges, full-grid shape and active mask.
    positions : ndarray of float64, shape (n_samples, n_dims)
        Position samples.
    dt : ndarray of float64, shape (n_samples - 1,)
        Time between consecutive samples.
    valid_mask : ndarray of bool, shape (n_samples - 1,)
        Intervals that pass the observation filters.
    bin_indices : ndarray of intp, shape (n_samples,)
        Active bin of each sample, or ``-1`` outside the environment.
    return_seconds : bool
        If True, allocate seconds. If False, each interval contributes one
        count, split in proportion to path length; the share that falls on
        inactive cells is dropped, as start allocation drops outside samples.

    Returns
    -------
    ndarray of float64, shape (n_bins,)
        Seconds (or proportional counts) in each active bin.
    """
    full_to_active = np.full(grid.active_mask.size, -1, dtype=np.intp)
    active = np.flatnonzero(grid.active_mask.ravel())
    full_to_active[active] = np.arange(active.size, dtype=np.intp)

    occupancy = np.zeros(n_bins, dtype=np.float64)
    valid = np.flatnonzero(valid_mask)
    # Blocks follow interval order, so accumulating block by block adds every
    # piece in the same order as a sequential traversal.
    for block in crossing_blocks(positions[valid], positions[valid + 1], grid.edges):
        _accumulate_block(
            occupancy,
            full_to_active,
            grid,
            positions,
            dt,
            valid[block],
            bin_indices,
            return_seconds=return_seconds,
        )
    return occupancy


def _accumulate_block(
    occupancy: NDArray[np.float64],
    full_to_active: NDArray[np.intp],
    grid: RectilinearGrid,
    positions: NDArray[np.float64],
    dt: NDArray[np.float64],
    valid: NDArray[np.intp],
    bin_indices: NDArray[np.intp],
    *,
    return_seconds: bool,
) -> None:
    """Add one block of valid intervals to ``occupancy``, in interval order.

    Parameters
    ----------
    occupancy : ndarray of float64, shape (n_bins,)
        Accumulated seconds (or counts), updated in place.
    full_to_active : ndarray of intp, shape (n_cells,)
        Active-bin index of each full-grid cell, or ``-1``.
    valid : ndarray of intp, shape (n_block,)
        Indices of the block's valid intervals, increasing.
    grid, positions, dt, bin_indices, return_seconds
        As for :func:`linear_grid_occupancy`.
    """
    start_bin, end_bin = bin_indices[valid], bin_indices[valid + 1]
    # Both endpoints in one active bin: the whole interval belongs to it.
    within = (start_bin == end_bin) & (start_bin >= 0)
    weights = dt[valid] if return_seconds else np.ones(len(valid))

    crossing = valid[~within]
    interval, cell, time = ray_grid_pieces(
        positions[crossing],
        positions[crossing + 1],
        grid.edges,
        grid.shape,
        dt[crossing],
    )
    if not return_seconds:
        # Normalize over every crossed cell, active or not, so mass on
        # inactive cells is dropped rather than redistributed.
        totals = np.zeros(len(crossing))
        np.add.at(totals, interval, time)
        positive = totals[interval] > 0
        interval, cell = interval[positive], cell[positive]
        time = (time[positive] / totals[interval]) * weights[~within][interval]
    target = full_to_active[cell]
    retained = target >= 0

    # Accumulate in interval order, as a sequential traversal would.
    order_key = np.concatenate(
        [np.flatnonzero(within), np.flatnonzero(~within)[interval[retained]]]
    )
    bins = np.concatenate([start_bin[within], target[retained]])
    values = np.concatenate([weights[within], time[retained]])
    order = np.argsort(order_key, kind="stable")
    np.add.at(occupancy, bins[order], values[order])
