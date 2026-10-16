"""Boundary bins: bins at the edge of an environment's cells or graph.

One implementation serves every layout. Exact polygon cells use their union's
outline. Lattice locators (rectangular and hexagonal, through subsets and
affine maps) supply their logical cell grid, and a bin is on the boundary when
a neighbor cell is missing: an orthogonal neighbor on a rectangular grid, any
of the six neighbors on a hexagonal lattice. Other bins use graph-degree rules.
"""

from __future__ import annotations

import warnings
from functools import singledispatch
from typing import NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._binning import Binning
from neurospatial.geometry._hex_lookup import _get_hex_grid_neighbor_deltas
from neurospatial.geometry._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._polygon import PolygonCells, _boundary_cell_indices


def boundary_bins(
    binning: Binning,
    graph: nx.Graph,
    grid_shape: tuple[int, ...] | None,
    active_mask: NDArray[np.bool_] | None,
    *,
    periodic_axes: tuple[int, ...],
) -> NDArray[np.intp]:
    """Find the bins on the boundary of an environment.

    Parameters
    ----------
    binning : Binning
        Published bins, cell shapes and locator.
    graph : networkx.Graph
        Navigation graph over the bins.
    grid_shape : tuple of int or None
        Logical grid shape used when the locator has no lattice of its own.
    active_mask : ndarray of bool, shape ``grid_shape``, or None
        Active cells of that logical grid.
    periodic_axes : tuple of int
        Grid axes that wrap around, such as a circular angle.

    Returns
    -------
    ndarray of intp, shape (n_boundary,)
        Sorted boundary bin indices.
    """
    shapes = binning.shapes
    if isinstance(shapes, PolygonCells):
        return _boundary_cell_indices(shapes.polygons)
    cells = _logical_cells(binning.locator)
    if cells is not None and cells.hexagonal:
        return _hex_boundary_bins(cells.shape, cells.mapping)
    if cells is not None:
        # Cell adjacency survives coordinate transforms and subsets.
        grid_shape = cells.shape
        active_mask = (cells.mapping >= 0).reshape(grid_shape)
    return find_boundary_nodes(
        graph=graph,
        grid_shape=grid_shape,
        active_mask=active_mask,
        periodic_axes=periodic_axes,
    ).astype(np.intp, copy=False)


class _LogicalCells(NamedTuple):
    """A locator's logical cell lattice.

    Attributes
    ----------
    shape : tuple of int
        Logical grid shape.
    mapping : ndarray of intp, shape (prod(shape),)
        Active-bin index of each flat cell, or ``-1``.
    hexagonal : bool
        Whether the cells form a hexagonal lattice with six neighbors per cell
        rather than a rectangular grid.
    """

    shape: tuple[int, ...]
    mapping: NDArray[np.intp]
    hexagonal: bool


@singledispatch
def _logical_cells(locator: object) -> _LogicalCells | None:
    """Return a locator's logical cell lattice, if it has one."""
    return None


@_logical_cells.register
def _(locator: RectilinearLocator) -> _LogicalCells:
    mask = locator.active_mask
    mask = np.ones(locator.shape, dtype=bool) if mask is None else mask
    mapping = np.full(mask.size, -1, dtype=np.intp)
    mapping[mask.ravel()] = np.arange(np.count_nonzero(mask))
    return _LogicalCells(locator.shape, mapping, hexagonal=False)


@_logical_cells.register
def _(locator: HexLocator) -> _LogicalCells:
    return _LogicalCells(
        locator.centers_shape, locator.flat_to_active.copy(), hexagonal=True
    )


@_logical_cells.register
def _(locator: SubsetLocator) -> _LogicalCells | None:
    cells = _logical_cells(locator.base)
    if cells is None:
        return None
    mapping = cells.mapping
    # Compose the subset map without indexing the -1 sentinel.
    inside = mapping >= 0
    mapping[inside] = locator.parent_to_child[mapping[inside]]
    return cells


@_logical_cells.register
def _(locator: TransformedLocator) -> _LogicalCells | None:
    return _logical_cells(locator.base)


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
        [_get_hex_grid_neighbor_deltas(False), _get_hex_grid_neighbor_deltas(True)]
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


def _is_grid_boundary_node(
    node_id: int,
    graph: nx.Graph,
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_],
    periodic_axes: tuple[int, ...] = (),
) -> bool:
    """Check if a node is on the boundary using grid topology.

    A node is boundary if any orthogonal neighbor is out-of-bounds
    or inactive. Along a periodic axis the neighbor index wraps, so the
    first and last cells of that axis are neighbors.

    Parameters
    ----------
    node_id : int
        Node ID to check.
    graph : nx.Graph
        Connectivity graph with node attributes.
    grid_shape : tuple[int, ...]
        Shape of the grid.
    active_mask : NDArray[np.bool_]
        Boolean mask indicating active bins.
    periodic_axes : tuple of int, default=()
        Grid axes that wrap around (e.g. a circular angle).

    Returns
    -------
    bool
        True if node is on boundary, False otherwise.

    """
    node_data = graph.nodes[node_id]
    original_nd_idx = node_data.get("original_grid_nd_index")

    if original_nd_idx is None:
        warnings.warn(
            f"Node {node_id} missing 'original_grid_nd_index'. "
            "Cannot use N-D grid boundary logic.",
            UserWarning,
            stacklevel=2,
        )
        return False

    n_dims = len(grid_shape)
    for dim_idx in range(n_dims):
        for offset_val in [-1, 1]:
            neighbor_nd_idx_list = list(original_nd_idx)
            neighbor_nd_idx_list[dim_idx] += offset_val
            if dim_idx in periodic_axes:
                neighbor_nd_idx_list[dim_idx] %= grid_shape[dim_idx]
            neighbor_nd_idx = tuple(neighbor_nd_idx_list)

            # Check if neighbor is out of bounds
            if not (0 <= neighbor_nd_idx[dim_idx] < grid_shape[dim_idx]):
                return True

            # Check if neighbor is inactive
            if not active_mask[neighbor_nd_idx]:
                return True

    return False


def _find_grid_boundary_nodes(
    graph: nx.Graph,
    grid_shape: tuple[int, ...],
    active_mask: NDArray[np.bool_],
    periodic_axes: tuple[int, ...] = (),
) -> list[int]:
    """Find boundary nodes using N-D grid topology.

    Parameters
    ----------
    graph : nx.Graph
        Connectivity graph.
    grid_shape : tuple[int, ...]
        Shape of the grid.
    active_mask : NDArray[np.bool_]
        Boolean mask of active bins.
    periodic_axes : tuple of int, default=()
        Grid axes that wrap around (e.g. a circular angle).

    Returns
    -------
    list[int]
        List of boundary node IDs.

    """
    boundary_nodes = []
    for node_id in graph.nodes():
        if _is_grid_boundary_node(
            node_id, graph, grid_shape, active_mask, periodic_axes
        ):
            boundary_nodes.append(node_id)
    return boundary_nodes


def _compute_degree_threshold(
    layout_kind: str | None,
    median_degree: float,
    max_degree: float,
    grid_shape: tuple[int, ...] | None,
) -> float:
    """Compute degree threshold for boundary detection.

    Parameters
    ----------
    layout_kind : str or None
        Type of layout (e.g., "Hexagonal", "Graph").
    median_degree : float
        Median degree of all nodes.
    max_degree : float
        Maximum degree of all nodes.
    grid_shape : tuple[int, ...] or None
        Shape of grid if applicable.

    Returns
    -------
    float
        Degree threshold below which nodes are considered boundary.

    """
    if layout_kind == "Hexagonal" and median_degree > 5:
        return 5.5  # Hexagonal grids: bins with < 6 neighbors

    if layout_kind == "Graph":
        return 1.5 if max_degree >= 2 else max_degree

    # 1D grids use the same logic as Graph layouts
    if grid_shape is not None and len(grid_shape) == 1:
        return 1.5 if max_degree >= 2 else max_degree

    # N-D grid fallback
    if (
        grid_shape is not None
        and len(grid_shape) > 1
        and median_degree > (2 * len(grid_shape) - 1)
    ):
        return 2 * len(grid_shape) - 0.5

    # General fallback
    return median_degree


def _find_degree_based_boundary_nodes(
    graph: nx.Graph,
    layout_kind: str | None,
    grid_shape: tuple[int, ...] | None,
) -> list[int]:
    """Find boundary nodes using degree-based heuristics.

    Parameters
    ----------
    graph : nx.Graph
        Connectivity graph.
    layout_kind : str or None
        Type of layout for layout-specific thresholds.
    grid_shape : tuple[int, ...] or None
        Shape of grid if applicable.

    Returns
    -------
    list[int]
        List of boundary node IDs.

    """
    degrees = dict(graph.degree())
    if not degrees:
        return []

    all_degree_values = list(degrees.values())
    median_degree = np.median(all_degree_values)
    max_degree = np.max(all_degree_values)

    # Compute threshold based on layout type
    threshold = _compute_degree_threshold(
        layout_kind,
        median_degree,
        max_degree,
        grid_shape,
    )

    # Find nodes below threshold
    boundary_nodes = [
        node_id for node_id, degree in degrees.items() if degree < threshold
    ]

    # Fallback: if no boundaries found and degrees vary, mark nodes < max_degree
    if (
        not boundary_nodes
        and max_degree > 0
        and median_degree != max_degree
        and grid_shape is None  # Only for non-grid layouts
    ):
        boundary_nodes = [
            node_id for node_id, degree in degrees.items() if degree < max_degree
        ]

    return boundary_nodes


def find_boundary_nodes(
    graph: nx.Graph,
    grid_shape: tuple[int, ...] | None = None,
    active_mask: NDArray[np.bool_] | None = None,
    layout_kind: str | None = None,
    periodic_axes: tuple[int, ...] = (),
) -> NDArray[np.int_]:
    """Identify boundary nodes in a connectivity graph G.

    A node is considered "boundary" if:
      1) For an N>1 grid layout (grid_shape and active_mask provided), it lies
         on the edge of the active region: at least one of its orthogonal N-D
         neighbors is out of bounds or inactive.
      2) Otherwise, its degree is below a threshold set from the node degrees:
         - Hexagonal (layout_kind=="Hexagonal") with median degree > 5: 5.5.
         - Graph layouts (layout_kind=="Graph") and 1-D grids: 1.5 when the
           maximum degree is at least 2, else the maximum degree.
         - Other grids with median degree > 2*N - 1: 2*N - 0.5.
         - Otherwise: the median degree.
      3) If rule 2 finds no node, the layout has no grid_shape and the degrees
         vary, nodes with degree below the maximum degree are boundary.

    Parameters
    ----------
    graph : nx.Graph
        The connectivity graph where each node ID is an active-bin ID,
        and node attributes include possibly 'original_grid_nd_index'.
    grid_shape : Tuple[int, ...], optional
        The shape of the full grid. Only used if len(grid_shape) > 1 and
        active_mask is provided.
    active_mask : np.ndarray, optional
        An N-D boolean mask of shape=grid_shape. True=active bin.
    layout_kind : str, optional
        A string identifier (e.g. "RegularGrid", "Hexagonal", "Graph").
        Used for special heuristics (e.g. hexagon degrees).
    periodic_axes : tuple of int, default=()
        Grid axes that wrap around, such as a circular angular axis; their
        first and last cells are neighbors. Used only by the grid method.


    Returns
    -------
    np.ndarray[np.int_]
        Sorted array of node IDs (active-bin IDs) identified as boundary.

    Raises
    ------
    ValueError
        If G is empty or invalid parameters are provided.

    """
    if graph.number_of_nodes() == 0:
        return np.array([], dtype=int)

    # Determine if we can use grid-based boundary detection
    use_grid_method = (
        active_mask is not None and grid_shape is not None and len(grid_shape) > 1
    )

    if use_grid_method:
        # These assertions are safe because use_grid_method checks they're not None
        assert grid_shape is not None
        assert active_mask is not None
        boundary_nodes = _find_grid_boundary_nodes(
            graph, grid_shape, active_mask, periodic_axes
        )
    else:
        boundary_nodes = _find_degree_based_boundary_nodes(
            graph,
            layout_kind,
            grid_shape,
        )

    return np.array(sorted(set(boundary_nodes)), dtype=int)
