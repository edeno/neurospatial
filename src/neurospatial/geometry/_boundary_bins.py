"""Boundary bins: bins at the edge of an environment's cells.

One implementation serves every layout, and it reads only cell geometry.
Lattice locators (rectangular and hexagonal, through subsets and affine maps)
supply their logical cell grid, and a bin is on the boundary when a neighbor
cell is missing: an orthogonal neighbor on a rectangular grid, any of the six
neighbors on a hexagonal lattice. Exact cell shapes (polygon cells, triangle
meshes) use the outline of their union. Bins measured by length (tracks) have
no outline; their boundary is the ends of the chain, the bins with fewer than
two neighbors. A combined environment whose cells are exact 2-D shapes
uses the outline of their union, so bins along a seam where two components
touch are not boundary bins; other combined environments use each
component's own rule. Bins measured by area or volume without cell geometry
have no defined boundary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._composition import (
    CombinedLocator,
    component_selections,
    composition_view,
)
from neurospatial.geometry._context import cell_lattice
from neurospatial.geometry._derive import select_bins
from neurospatial.geometry._hex_lookup import get_hex_grid_neighbor_deltas
from neurospatial.geometry._polygon import _boundary_cell_indices, cell_polygons


def boundary_bins(
    binning: Binning, *, periodic_axes: tuple[int, ...]
) -> NDArray[np.intp]:
    """Find the bins on the boundary of an environment.

    Parameters
    ----------
    binning : Binning
        Published bins, cell shapes and locator.
    periodic_axes : tuple of int
        Grid axes that wrap around, such as a circular angle.

    Returns
    -------
    ndarray of intp, shape (n_boundary,)
        Sorted boundary bin indices.

    Raises
    ------
    NotImplementedError
        If bins measured by area or volume carry no cell geometry, so their
        outline is unknown.
    """
    if binning.bins.n_bins == 0:
        return np.empty(0, dtype=np.intp)
    lattice = cell_lattice(binning.locator)
    if lattice is not None and lattice.kind == "hex":
        return _hex_boundary_bins(lattice.shape, lattice.flat_to_active)
    if lattice is not None and lattice.kind == "grid":
        # Cell adjacency survives coordinate transforms and subsets.
        return _grid_boundary_bins(lattice.shape, lattice.flat_to_active, periodic_axes)
    if binning.shapes is not None:
        return _boundary_cell_indices(cell_polygons(binning))
    view = composition_view(binning.locator)
    if view is not None:
        return _combined_boundary_bins(*view[:2])
    if binning.bins.has_measure_dim(1):
        return _chain_ends(binning)
    raise NotImplementedError(
        _format_error(
            "Boundary bins need cell geometry, and these bins are points "
            "measured by area or volume without cells.",
            why="Why: a bin is on the boundary when its cell touches the "
            "outline of the cells; without cells the outline is unknown, and "
            "guessing it from neighbor counts marks the wrong bins.",
            fix="build the environment with a layout that has cells (grid, "
            "hexagonal, triangular mesh or polygon), or pass a Binning with "
            "shapes to Environment.from_bins",
        )
    )


def _combined_boundary_bins(
    root: CombinedLocator, indices: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Each component's boundary among the bins it keeps, in current bins.

    Parameters
    ----------
    root : CombinedLocator
        The combined binning the current bins come from.
    indices : ndarray of intp, shape (n_bins,)
        For each current bin, the index of the root bin it comes from.

    Returns
    -------
    ndarray of intp, shape (n_boundary,)
        Sorted current bins on the boundary of their own component. Affine
        maps do not change which bins these are; dropping bins can expose
        new ones.
    """
    parts = []
    for selection in component_selections(root, indices):
        child, keep, rows = selection
        kept = child if np.all(keep) else select_bins(child, keep)
        parts.append(rows[boundary_bins(kept, periodic_axes=())])
    return np.sort(np.concatenate(parts)).astype(np.intp, copy=False)


def _chain_ends(binning: Binning) -> NDArray[np.intp]:
    """Bins of a length-measured chain with fewer than two neighbors.

    Parameters
    ----------
    binning : Binning
        Bins measured by length, such as a track's.

    Returns
    -------
    ndarray of intp, shape (n_ends,)
        Sorted bins at the ends of the chain, including isolated bins.
    """
    pairs = binning.connections.all_pairs
    degree = np.bincount(pairs.ravel(), minlength=binning.bins.n_bins)
    return np.flatnonzero(degree < 2).astype(np.intp, copy=False)


def _grid_boundary_bins(
    shape: tuple[int, ...],
    flat_to_active: NDArray[np.intp],
    periodic_axes: tuple[int, ...],
) -> NDArray[np.intp]:
    """Find the bins of a rectangular grid with a missing orthogonal neighbor.

    Parameters
    ----------
    shape : tuple of int
        Logical grid shape.
    flat_to_active : ndarray of intp, shape (prod(shape),)
        Active-bin index of each grid cell, or -1 when inactive.
    periodic_axes : tuple of int
        Grid axes that wrap around, so their first and last cells are
        neighbors.

    Returns
    -------
    ndarray of intp, shape (n_boundary,)
        Sorted bins with an orthogonal neighbor cell inactive or outside the
        grid.
    """
    active = (flat_to_active >= 0).reshape(shape)
    cells = np.flatnonzero(flat_to_active >= 0)
    index = np.unravel_index(cells, shape)
    on_boundary = np.zeros(len(cells), dtype=bool)
    for axis, size in enumerate(shape):
        for step in (-1, 1):
            neighbor = index[axis] + step
            if axis in periodic_axes:
                neighbor %= size
            outside = (neighbor < 0) | (neighbor >= size)
            shifted = list(index)
            shifted[axis] = np.clip(neighbor, 0, size - 1)
            on_boundary |= outside | ~active[tuple(shifted)]
    return np.sort(flat_to_active[cells[on_boundary]]).astype(np.intp, copy=False)


def _hex_boundary_bins(
    centers_shape: tuple[int, ...], flat_to_active: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Find the bins of a hexagonal lattice with a missing neighbor cell.

    Parameters
    ----------
    centers_shape : tuple of int
        Hexagon rows and columns of the conceptual lattice.
    flat_to_active : ndarray of intp, shape (prod(centers_shape),)
        Active-bin index of each lattice cell, or -1 when inactive.

    Returns
    -------
    ndarray of intp, shape (n_boundary,)
        Sorted bins with at least one of their six neighbors inactive or
        outside the lattice.
    """
    n_rows, n_cols = centers_shape
    cells = np.flatnonzero(flat_to_active >= 0)
    rows, cols = np.divmod(cells, n_cols)
    # (parity, neighbor, (d_col, d_row)): odd rows are shifted right.
    deltas = np.array(
        [get_hex_grid_neighbor_deltas(False), get_hex_grid_neighbor_deltas(True)]
    )[rows % 2]
    neighbor_cols = cols[:, None] + deltas[..., 0]
    neighbor_rows = rows[:, None] + deltas[..., 1]
    inside = (
        (neighbor_rows >= 0)
        & (neighbor_rows < n_rows)
        & (neighbor_cols >= 0)
        & (neighbor_cols < n_cols)
    )
    neighbor_bins = np.full(inside.shape, -1, dtype=np.intp)
    neighbor_bins[inside] = flat_to_active[
        neighbor_rows[inside] * n_cols + neighbor_cols[inside]
    ]
    on_boundary = np.any(neighbor_bins < 0, axis=1)
    return np.sort(flat_to_active[cells[on_boundary]]).astype(np.intp, copy=False)
