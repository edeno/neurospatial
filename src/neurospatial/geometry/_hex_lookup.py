"""Pure coordinate conversion, point assignment and neighbors on hexagonal cells."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MIN_HEX_RADIUS = 1e-10


def _cartesian_to_fractional_cube(
    points_x: NDArray[np.float64],
    points_y: NDArray[np.float64],
    hex_radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian (x, y) coordinates to fractional axial coordinates of a hex grid.

    Axial coordinates represent positions in a hexagonal grid using two axes (q, r),
    simplifying neighbor and distance calculations compared to Cartesian coordinates.

    Parameters
    ----------
    points_x : NDArray[np.float64], shape (n_points,)
        X-coordinate(s) in Cartesian space, typically relative to the hexagonal
        grid's origin.
    points_y : NDArray[np.float64], shape (n_points,)
        Y-coordinate(s) in Cartesian space, relative to the grid's origin.
    hex_radius : float
        Width of each hexagon in the grid, measured as the distance between opposite sides.

    Returns
    -------
    q_frac : NDArray[np.float64], shape (n_points,)
        Fractional q cube coordinate.
    r_frac : NDArray[np.float64], shape (n_points,)
        Fractional r cube coordinate.
    s_frac : NDArray[np.float64], shape (n_points,)
        Fractional s cube coordinate, calculated as -q_frac - r_frac.

    Notes
    -----
    The conversion follows the standard formulas (see
    https://www.redblobgames.com/grids/hexagons/#coordinates):

        q = (sqrt(3)/3 * x - 1/3 * y) / hex_width
        r = (2/3 * y) / hex_width

    Fractional axial coordinates allow for smooth interpolation before rounding
    to discrete hex indices.

    Points outside the grid bounds or containing NaNs in inputs will produce NaN outputs.

    Raises
    ------
    ValueError
        If hex_radius is below MIN_HEX_RADIUS (1e-10), which would cause
        numerical instability in coordinate conversion.

    """
    # Validate hex_radius is not too small (prevents numerical instability)
    if abs(hex_radius) < MIN_HEX_RADIUS:
        raise ValueError(
            f"hex_radius too small ({hex_radius:.2e}). "
            f"Minimum supported radius: {MIN_HEX_RADIUS:.2e}. "
            "This prevents division by near-zero values which cause numerical instability."
        )

    # Handle near-zero case using tolerance-based comparison (more robust than hex_radius == 0)
    if np.isclose(hex_radius, 0.0, atol=1e-12):
        # Degenerate grid - all points map to origin
        zero_coords = np.zeros_like(points_x)
        return zero_coords, zero_coords, zero_coords

    q_frac = (np.sqrt(3.0) / 3.0 * points_x - 1.0 / 3.0 * points_y) / hex_radius
    r_frac = (2.0 / 3.0 * points_y) / hex_radius
    s_frac = -q_frac - r_frac
    return q_frac, r_frac, s_frac


def _round_fractional_cube_to_integer_axial(
    q_frac: NDArray[np.float64],
    r_frac: NDArray[np.float64],
    s_frac: NDArray[np.float64],
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Round fractional cube coordinates to integer axial coordinates (q, r).

    This rounding method ensures that the sum of the corresponding integer
    cube coordinates (q_int + r_int + s_int) would be zero by adjusting
    the coordinate with the largest fractional difference from its rounded integer.

    Parameters
    ----------
    q_frac : NDArray[np.float64], shape (n_points,)
        Fractional q cube coordinates.
    r_frac : NDArray[np.float64], shape (n_points,)
        Fractional r cube coordinates.
    s_frac : NDArray[np.float64], shape (n_points,)
        Fractional s cube coordinates (where s_frac = -q_frac - r_frac).

    Returns
    -------
    q_axial : NDArray[np.int_], shape (n_points,)
        Integer q axial coordinate.
    r_axial : NDArray[np.int_], shape (n_points,)
        Integer r axial coordinate.

    """
    # Initial rounding to nearest integer
    q_round_f: NDArray[np.float64] = np.round(q_frac)
    r_round_f: NDArray[np.float64] = np.round(r_frac)
    s_round_f: NDArray[np.float64] = np.round(
        s_frac,
    )  # s_round = np.round(-q_frac - r_frac)

    # Differences from original fractional values
    q_diff: NDArray[np.float64] = np.abs(q_round_f - q_frac)
    r_diff: NDArray[np.float64] = np.abs(r_round_f - r_frac)
    s_diff: NDArray[np.float64] = np.abs(s_round_f - s_frac)

    # Initialize axial coordinates with the basic rounded values
    q_axial: NDArray[np.int_] = q_round_f.astype(int)
    r_axial: NDArray[np.int_] = r_round_f.astype(int)
    # s_axial for sum check, not directly returned for offset conversion
    # s_axial: NDArray[np.int_] = s_round_f.astype(np.int_)

    # Correct coordinates to ensure q_axial + r_axial + s_axial = 0
    # Condition where q_diff is strictly the largest
    cond1: NDArray[np.bool_] = (q_diff > r_diff) & (q_diff > s_diff)
    q_axial[cond1] = (-r_round_f[cond1] - s_round_f[cond1]).astype(int)
    # r_axial[cond1] remains r_round_f[cond1]

    # Condition where r_diff is strictly the largest (and q_diff was not)
    cond2: NDArray[np.bool_] = (~cond1) & (r_diff > s_diff)
    r_axial[cond2] = (-q_round_f[cond2] - s_round_f[cond2]).astype(int)
    # q_axial[cond2] remains q_round_f[cond2]

    # Condition where s_diff is largest (or ties put it here), and others were not
    # In this case, s_axial would be adjusted:
    # s_axial[cond3] = (-q_round_f[cond3] - r_round_f[cond3]).astype(np.int_)
    # q_axial and r_axial remain as their initially rounded values, which is correct.
    # No explicit action needed for q_axial, r_axial under cond3 as they are already set.

    return q_axial, r_axial


def _axial_to_offset_bin_indices(
    q_axial: NDArray[np.int_],
    r_axial: NDArray[np.int_],
    n_hex_x: int,
    n_hex_y: int,
) -> NDArray[np.int_]:
    """Convert integer axial coordinates (q, r) to 1D bin indices.

    This conversion is for an "odd-r" offset grid arrangement where odd rows
    are shifted relative to even rows. The resulting 1D index corresponds
    to a flattened row-major order of the conceptual rectangular grid of hexagons.

    Parameters
    ----------
    q_axial : NDArray[np.int_], shape (n_points,)
        Integer q axial coordinates.
    r_axial : NDArray[np.int_], shape (n_points,)
        Integer r axial coordinates.
    n_hex_x : int
        Number of hexagons in the x-direction (columns) of the full conceptual grid.
    n_hex_y : int
        Number of hexagons in the y-direction (rows) of the full conceptual grid.

    Returns
    -------
    NDArray[np.int_], shape (n_points,)
        1D bin indices corresponding to the input axial coordinates.
        Points that fall outside the defined grid dimensions (`n_hex_x`, `n_hex_y`)
        are assigned an index of -1.

    """
    # Convert axial to "odd-r" offset coordinates
    # col = q + (r - (r & 1)) / 2
    # row = r
    # (r_axial & 1) is 1 if r_axial is odd, 0 if even. Works for negative r_axial.
    grid_col: NDArray[np.int_] = q_axial + (r_axial - (r_axial & 1)) // 2
    grid_row: NDArray[np.int_] = r_axial

    # Calculate 1D bin index (row-major flattening)
    # bin_idx = row * num_cols + col
    bin_idx: NDArray[np.int_] = grid_row * n_hex_x + grid_col

    # Mark out-of-bounds indices as -1
    valid_mask: NDArray[np.bool_] = (
        (grid_col >= 0) & (grid_col < n_hex_x) & (grid_row >= 0) & (grid_row < n_hex_y)
    )

    final_bin_idx: NDArray[np.int_] = np.full_like(bin_idx, -1, dtype=int)
    final_bin_idx[valid_mask] = bin_idx[valid_mask]

    return final_bin_idx


def _points_to_hex_bin_ind(
    points: NDArray[np.float64],
    grid_offset_x: float,
    grid_offset_y: float,
    hex_radius: float,
    centers_shape: tuple[int, int],
) -> NDArray[np.int_]:
    """Assign 2D Cartesian data points to hexagon bin indices in a predefined grid.

    This function converts Cartesian points to hexagonal grid coordinates and
    then to 1D flat indices relative to the full conceptual grid of hexagons.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, 2)
        The (x, y) coordinates of the points to assign to bins.
        NaN input points result in an index of -1.
    grid_offset_x : float
        The x-coordinate of the grid's effective origin (e.g., min_x from
        `_create_hex_grid`).
    grid_offset_y : float
        The y-coordinate of the grid's effective origin (e.g., min_y from
        `_create_hex_grid`).
    hex_radius : float
        The radius of the hexagons (distance from center to vertex).
    centers_shape : Tuple[int, int]
        The shape (n_hex_rows, n_hex_cols) of the full conceptual grid of hexagons.

    Returns
    -------
    NDArray[np.int_], shape (n_points,)
        The 1D bin index (relative to the full conceptual grid) for each
        data point. Points falling outside the defined grid area or NaN input
        points are assigned an index of -1.

    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.array([], dtype=int)

    points = np.atleast_2d(points)

    output_indices: NDArray[np.int_] = np.full(n_points, -1, dtype=int)

    # Identify valid (non-NaN) points
    valid_mask: NDArray[np.bool_] = np.asarray(
        ~np.isnan(points).any(axis=1), dtype=np.bool_
    )
    if not np.any(valid_mask):
        return output_indices  # All points are NaN or empty after all

    valid_points: NDArray[np.float64] = points[valid_mask]

    # Adjust points relative to the grid's effective origin
    # (grid_offset_x, grid_offset_y) is the center of hex (0,0) in grid indices
    adj_points_x: NDArray[np.float64] = valid_points[:, 0] - grid_offset_x
    adj_points_y: NDArray[np.float64] = valid_points[:, 1] - grid_offset_y

    # Convert Cartesian to fractional cube coordinates
    q_frac, r_frac, s_frac = _cartesian_to_fractional_cube(
        adj_points_x,
        adj_points_y,
        hex_radius,
    )

    # Round fractional cube coordinates to integer axial coordinates
    q_axial, r_axial = _round_fractional_cube_to_integer_axial(q_frac, r_frac, s_frac)

    # Convert axial coordinates to 1D bin indices
    n_hex_y, n_hex_x = centers_shape
    bin_indices_for_valid_points = _axial_to_offset_bin_indices(
        q_axial,
        r_axial,
        n_hex_x,
        n_hex_y,
    )

    output_indices[valid_mask] = bin_indices_for_valid_points

    return output_indices


def _get_hex_grid_neighbor_deltas(is_odd_row: bool) -> list[tuple[int, int]]:
    """Return (delta_col, delta_row) coordinate deltas for hexagon neighbors.

    This is for an "odd-r" pointy-top hex grid, where odd rows (1, 3, ...)
    are typically shifted right relative to even rows (0, 2, ...).
    The y-axis (row index) increases "upwards" in grid coordinates.

    Parameters
    ----------
    is_odd_row : bool
        True if the current hexagon's row index is odd, False otherwise.

    Returns
    -------
    List[Tuple[int, int]]
        A list of 6 (delta_col, delta_row) tuples, each representing the
        offset to one of the 6 neighbors.

    """
    if is_odd_row:
        # Neighbors for an odd row (shifted right): E, W, NW, NE, SW, SE
        # (dc, dr)
        return [(1, 0), (-1, 0), (0, 1), (1, 1), (0, -1), (1, -1)]
    # Even row
    # Neighbors for an even row: E, W, NW, NE, SW, SE
    # (dc, dr)
    return [(1, 0), (-1, 0), (-1, 1), (0, 1), (-1, -1), (0, -1)]
