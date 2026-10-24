"""Pure point assignment on rectilinear cells.

Cells are half-open inside the active cells; their outline is closed
(:func:`outline_cells`).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import mask_active_map


def points_to_regular_grid_bin_ind(
    points: NDArray[np.float64],
    grid_edges: tuple[NDArray[np.float64], ...],
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_] | None = None,
    inverse_map: NDArray[np.intp] | None = None,
) -> NDArray[np.int_]:
    """Map N-D points to their corresponding bin indices in a regular grid.

    If `active_mask` is provided, maps to indices relative to active bins
    (0 to `n_active_bins - 1`). Otherwise, maps to flat indices of the
    full conceptual grid.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        N-dimensional points to map. NaNs are filtered out.
    grid_edges : Tuple[NDArray[np.float64], ...]
        Tuple where each element is a 1D array of bin edge positions for one
        dimension of the full grid.
    grid_shape : Tuple[int, ...]
        N-D shape (number of bins in each dimension) of the full grid.
    active_mask : Optional[NDArray[np.bool_]], shape `grid_shape`, optional
        If provided, an N-D boolean mask indicating active bins in the full
        grid. Output indices will be relative to these active bins.
        If None (default), output indices are flat indices of the full grid.
    inverse_map : Optional[NDArray[np.intp]], shape (active_mask.size,), optional
        Pre-computed inverse mapping from flat grid index to active bin index.
        If provided with active_mask, avoids recomputing the inverse map on
        each call. Should satisfy: inverse_map[flat_idx] = active_bin_idx
        for active bins, -1 otherwise.

    Returns
    -------
    bin_indices : NDArray[np.int_], shape (n_points,)
        Bin index of each input point, in input order.
        - If `active_mask` is provided: Indices are `0` to `n_active_bins - 1`.
          Points in inactive cells or outside the grid get -1.
        - If `active_mask` is None: Flat indices of the full grid. Points
          outside the grid get -1.
        Cells are half-open, ``[edge[i], edge[i + 1])``, so a point exactly on
        the last edge of an axis is outside the grid here and gets -1 (see
        :func:`outline_cells`). NaN points get -1.

    """
    points_atleast_2d = np.atleast_2d(points)
    valid_input_mask = ~np.any(np.isnan(points_atleast_2d), axis=1)

    # Initialize output assuming all points are invalid or unmapped
    # Output shape should match original number of points (before NaN filter)
    output_indices = np.full(points_atleast_2d.shape[0], -1, dtype=int)

    if not np.any(valid_input_mask):  # No valid points after NaN filter
        return output_indices

    valid_points = points_atleast_2d[valid_input_mask]
    n_dims = valid_points.shape[1]
    if n_dims != len(grid_edges) or n_dims != len(grid_shape):
        raise ValueError(
            f"Dimensionality mismatch: points have {n_dims} dimension(s), but "
            f"grid_edges has {len(grid_edges)} and grid_shape has "
            f"{len(grid_shape)}. Points must match the grid's dimensionality."
        )

    multi_bin_idx_list = []
    point_is_within_grid_bounds = np.ones(valid_points.shape[0], dtype=bool)

    for i in range(n_dims):
        # np.digitize counts edges <= x, so subtracting 1 gives the half-open
        # cell index: -1 below the grid and grid_shape[i] at or above its end.
        dim_indices = np.digitize(valid_points[:, i], grid_edges[i]) - 1
        point_is_within_grid_bounds &= (dim_indices >= 0) & (
            dim_indices < grid_shape[i]
        )
        multi_bin_idx_list.append(dim_indices)

    original_bin_flat_idx_for_valid_points = np.full(
        valid_points.shape[0],
        -1,
        dtype=int,
    )

    if np.any(point_is_within_grid_bounds):
        # np.ravel_multi_index refuses coordinates outside the grid.
        coords_for_ravel = tuple(
            idx[point_is_within_grid_bounds] for idx in multi_bin_idx_list
        )
        original_bin_flat_idx_for_valid_points[point_is_within_grid_bounds] = (
            np.ravel_multi_index(coords_for_ravel, grid_shape)
        )

    final_mapped_indices_for_valid_points = np.full(
        valid_points.shape[0],
        -1,
        dtype=int,
    )

    if active_mask is not None:
        active_mask_flat = active_mask.ravel()
        if inverse_map is None:
            inverse_map = mask_active_map(active_mask)

        valid_flat_mask = original_bin_flat_idx_for_valid_points != -1
        # Clipping only makes -1 indexable; valid_flat_mask excludes those rows.
        safe_indices = np.clip(
            original_bin_flat_idx_for_valid_points, 0, active_mask_flat.size - 1
        )
        in_active_mask = valid_flat_mask & active_mask_flat[safe_indices]
        final_mapped_indices_for_valid_points[in_active_mask] = inverse_map[
            original_bin_flat_idx_for_valid_points[in_active_mask]
        ]
    else:
        final_mapped_indices_for_valid_points = original_bin_flat_idx_for_valid_points

    output_indices[valid_input_mask] = final_mapped_indices_for_valid_points
    return output_indices


def outline_cells(
    points: NDArray[np.float64],
    grid_edges: tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]],
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_] | None,
) -> NDArray[np.intp]:
    """Full-grid cell each point bounds on the outline of the active cells.

    A point on a face with no active cell beyond it (the grid's edge, a hole
    or an arm tip) belongs to the active cell the face bounds from below, as
    the last bin of ``np.histogram`` includes its right edge. At an outline
    corner the cell whose upper faces the point touches on the fewest axes
    wins, the lowest axes first.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        Finite points whose half-open cell is inactive or outside the grid.
    grid_edges : tuple of NDArray[np.float64]
        Increasing cell edges per axis; ``grid_edges[d]`` has shape
        ``(grid_shape[d] + 1,)``.
    grid_shape : tuple of int
        Number of cells per axis.
    active_mask : NDArray[np.bool_] of shape ``grid_shape``, or None
        Active cells; None treats every cell as active.

    Returns
    -------
    NDArray[np.intp], shape (n_points,)
        Row-major full-grid index of the owning active cell, or -1 when the
        point is not on the outline.
    """
    shape = np.asarray(grid_shape)
    n_dims = len(grid_shape)
    # Half-open cell of each point per axis; -1 below and shape[d] above.
    cells = np.column_stack(
        [
            np.searchsorted(edge, points[:, axis], side="right") - 1
            for axis, edge in enumerate(grid_edges)
        ]
    ).astype(np.intp)
    # On axis d the point lies on the lower face of its half-open cell, which
    # is the upper face of the cell below it.
    on_face = np.column_stack(
        [
            (cells[:, axis] >= 1)
            & (cells[:, axis] <= shape[axis])
            & (points[:, axis] == edge[np.clip(cells[:, axis], 0, len(edge) - 1)])
            for axis, edge in enumerate(grid_edges)
        ]
    )
    active = None if active_mask is None else np.asarray(active_mask).ravel()
    owner = np.full(len(points), -1, dtype=np.intp)
    for size in range(1, n_dims + 1):
        for axes in itertools.combinations(range(n_dims), size):
            shift = np.zeros(n_dims, dtype=np.intp)
            shift[list(axes)] = 1
            candidate = cells - shift
            ok = (
                (owner < 0)
                & np.all(on_face[:, list(axes)], axis=1)
                & np.all((candidate >= 0) & (candidate < shape), axis=1)
            )
            if not ok.any():
                continue
            flat = np.ravel_multi_index(tuple(candidate[ok].T), grid_shape)
            if active is not None:
                ok_rows = np.flatnonzero(ok)[active[flat]]
                flat = flat[active[flat]]
            else:
                ok_rows = np.flatnonzero(ok)
            owner[ok_rows] = flat
    return owner
