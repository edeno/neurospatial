"""Whether two environments' bins mean the same places, connected the same way.

Analyses that pair arrays over bins from two environments (decoding with rates
estimated elsewhere, comparing rate maps between conditions) first ask whether
the bins mean the same places in the same coordinates, and then whether the
space is connected the same way. A door or barrier experiment keeps the bins
but changes connectivity or walls, so the two answers are reported separately.
Equal bin counts or array shapes never establish either.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.ops import unary_union

from neurospatial.geometry import (
    HexagonShapes,
    PolygonCells,
    TriangleShapes,
    encode,
    encode_arena,
    materialize_cells,
)

if TYPE_CHECKING:
    from scipy import sparse

    from neurospatial.environment.core import Environment
    from neurospatial.geometry import Arena, Binning, RectilinearGrid, SpatialBins


class _EnvironmentDifferences(NamedTuple):
    """The first difference in bin meaning and in topology, or None for each.

    Attributes
    ----------
    bins : str or None
        First difference in what the bins mean: ``"coordinate systems"``,
        ``"units or coordinate frames"``, ``"bin coordinates or ordering"``,
        ``"bin measures"``, ``"polygon cell geometry"``,
        ``"bin-boundary ownership"`` or ``"cell shapes"``.
    topology : str or None
        First difference in how the space is connected:
        ``"weighted connectivity"``, ``"continuous arena or physical walls"``
        or ``"smoothing conductances between neighboring bins"``.
    """

    bins: str | None
    topology: str | None


def _environment_differences(
    first: Environment, second: Environment
) -> _EnvironmentDifferences:
    """Compare two environments' bin meaning and topology.

    Parameters
    ----------
    first, second : Environment
        The environments to compare.

    Returns
    -------
    _EnvironmentDifferences
        ``(None, None)`` when the bins mean the same places in the same
        coordinates and the space is connected the same way.

    Notes
    -----
    Bin and topology checks run independently, so a topology difference never
    hides a bin difference. Environments that share published geometry, such
    as ``env.with_metadata(...)`` copies, differ at most in their coordinate
    labels.

    Equal centers and measures do not make equal cells: a 10 x 10 square and
    a 20 x 5 rectangle share a center and an area. Cells are compared
    whenever both environments record them (polygon cells, triangles,
    hexagons, or grid boxes, axis-aligned or in an affine frame). Bins
    without recorded cells, such as explicit point bins or track bins, are
    compared by center and measure only.

    Smoothing passes a field between neighbors through conductances (face
    area over distance), which the connectivity graph does not hold: two
    environments with the same bins, neighbors, distances and arena can
    smooth differently. Their conductances are compared too, and an
    environment that cannot smooth differs from one that can.
    """
    if first is second:
        return _EnvironmentDifferences(None, None)
    if first.coordinate_system != second.coordinate_system:
        labels = "coordinate systems"
    elif first.units != second.units or first.frame != second.frame:
        labels = "units or coordinate frames"
    else:
        labels = None
    if first._geometry is second._geometry:
        return _EnvironmentDifferences(labels, None)
    return _EnvironmentDifferences(
        labels or _bin_difference(first, second), _topology_difference(first, second)
    )


def _bin_difference(first: Environment, second: Environment) -> str | None:
    """Return the first difference between published bins with equal labels."""
    if not np.array_equal(first.bin_centers, second.bin_centers):
        return "bin coordinates or ordering"
    first_binning, second_binning = first._binning, second._binning
    if not _same_measures(first_binning.bins, second_binning.bins):
        return "bin measures"
    if isinstance(first_binning.shapes, PolygonCells) or isinstance(
        second_binning.shapes, PolygonCells
    ):
        return _polygon_cell_difference(first_binning, second_binning)
    first_corners = _cell_corners(first_binning)
    second_corners = _cell_corners(second_binning)
    if first_corners is None or second_corners is None:
        return None
    if not np.array_equal(first_corners, second_corners):
        return "cell shapes"
    return None


def _cell_corners(binning: Binning) -> NDArray[np.float64] | None:
    """Each bin's cell corners in a canonical order, or None without cells.

    Triangles, regular hexagons and grid boxes are convex, so their corners
    determine them. Sorting each cell's corners by coordinate makes the
    result independent of vertex order and of how mesh points are numbered.

    Returns
    -------
    ndarray of float64, shape (n_bins, n_corners, n_dims), or None
        None when the binning records no triangles, hexagons or grid.
    """
    shapes = binning.shapes
    if isinstance(shapes, TriangleShapes):
        corners = shapes.points[shapes.triangles]
    elif isinstance(shapes, HexagonShapes):
        corners = shapes.centers[:, np.newaxis, :] + shapes.corner_offsets()
    elif binning.rectilinear is not None:
        corners = _box_corners(binning.rectilinear)
    elif binning.grid_frame is not None:
        forward = binning.grid_frame.forward
        corners = (
            _box_corners(binning.grid_frame.source) @ forward[:-1, :-1].T
            + forward[:-1, -1]
        )
    else:
        return None
    # lexsort orders by its last key first: x, then y, then z.
    order = np.lexsort(np.moveaxis(corners, -1, 0)[::-1], axis=-1)
    return np.take_along_axis(corners, order[..., np.newaxis], axis=1)


def _box_corners(grid: RectilinearGrid) -> NDArray[np.float64]:
    """Corners of each active grid cell, shape (n_bins, 2**n_dims, n_dims).

    Active cells are bins in row-major order of the mask.
    """
    cells = np.nonzero(grid.active_mask)
    bounds = np.stack(
        [
            np.column_stack([edges[index], edges[index + 1]])
            for edges, index in zip(grid.edges, cells, strict=True)
        ],
        axis=1,
    )  # (n_bins, n_dims, 2): low and high edge on each axis
    n_dims = len(cells)
    choices = np.array(list(product((0, 1), repeat=n_dims)))
    corners: NDArray[np.float64] = bounds[:, np.arange(n_dims), choices]
    return corners


def _same_measures(first: SpatialBins, second: SpatialBins) -> bool:
    """Whether two sets of bins have equal measures, in equal dimensions."""
    return np.array_equal(first.sizes, second.sizes) and np.array_equal(
        first.measure_dims, second.measure_dims
    )


def _polygon_cell_difference(first: Binning, second: Binning) -> str | None:
    """Compare exact polygon cells, then which bin owns each cell boundary."""
    first_cells, second_cells = materialize_cells(first), materialize_cells(second)
    if len(first_cells) != len(second_cells) or any(
        not left.equals(right)
        for left, right in zip(first_cells, second_cells, strict=True)
    ):
        return "polygon cell geometry"
    try:
        first_meta, first_arrays = encode(first.locator, "locator")
        second_meta, second_arrays = encode(second.locator, "locator")
    except ValueError:
        return "bin-boundary ownership"
    if (
        first_meta != second_meta
        or first_arrays.keys() != second_arrays.keys()
        or any(
            not np.array_equal(value, second_arrays[key])
            for key, value in first_arrays.items()
        )
    ):
        return "bin-boundary ownership"
    return None


def _topology_difference(first: Environment, second: Environment) -> str | None:
    """Return the first difference in connectivity, arena or walls."""
    if not _same_weighted_edges(first.connectivity, second.connectivity):
        return "weighted connectivity"
    if not _same_arena(first._arena, second._arena):
        return "continuous arena or physical walls"
    if not _same_conductances(_conductances(first), _conductances(second)):
        return "smoothing conductances between neighboring bins"
    return None


def _conductances(env: Environment) -> sparse.csr_matrix | None:
    """Smoothing conductances between bins, or None when ``env`` cannot smooth."""
    try:
        return env._diffusion_geometry.W
    except NotImplementedError:
        return None


def _same_conductances(
    first: sparse.csr_matrix | None, second: sparse.csr_matrix | None
) -> bool:
    """Whether both conductance matrices are missing or exactly equal."""
    if first is None or second is None:
        return first is None and second is None
    if first.shape != second.shape:
        return False
    return bool((first != second).nnz == 0)


def _mapped_topology_difference(
    first: Environment, second: Environment, bin_pairs: NDArray[np.integer]
) -> str | None:
    """Compare topology through a one-to-one pairing of bins.

    Parameters
    ----------
    first, second : Environment
        The environments to compare.
    bin_pairs : ndarray of int, shape (n_pairs, 2)
        Rows of ``(bin of first, bin of second)`` that are the same place;
        each bin appears at most once in its column.

    Returns
    -------
    str or None
        The first topology difference, or None:

        - ``"weighted connectivity"``: two paired bins are neighbors in one
          environment and not the other, or neighbors at different distances.
        - ``"connections from mapped bins to bins outside the mapping"``: a
          paired bin has a neighbor outside the mapping, in either
          environment, with no counterpart in the other: no unpaired
          neighbor at the same center and the same distance.
        - ``"continuous arena or physical walls"``: the arenas or walls
          differ.
        - ``"smoothing conductances between neighboring bins"``: a paired bin
          exchanges a field with a neighbor through a different conductance
          (face area over distance), or only one environment can smooth.

    Notes
    -----
    The topology is the same when every paired bin's full set of neighbors,
    with edge distances, corresponds in the two environments: a paired
    neighbor through ``bin_pairs``, an unpaired neighbor by its center
    (compared exactly, as bin centers are in ``_environment_differences``).
    A smoothing over the graph then used the same neighbors for every paired
    bin, and its conductance to each neighbor must match too, so smoothing
    passed a field between them alike. Identical environments under any
    partial mapping, a renumbered copy,
    and a mapping that covers whole connected components keep the topology;
    a subset compared with its parent does not, because the parent's border
    bins have neighbors the subset lacks. The arena and walls are compared
    whole, as in ``_environment_differences``.
    """
    first_bins, second_bins = bin_pairs[:, 0].tolist(), bin_pairs[:, 1].tolist()
    to_first = dict(zip(second_bins, first_bins, strict=True))
    first_graph = first.connectivity.subgraph(first_bins)
    second_graph = nx.relabel_nodes(second.connectivity.subgraph(second_bins), to_first)
    if not _same_weighted_edges(first_graph, second_graph):
        return "weighted connectivity"
    first_weights, second_weights = _conductances(first), _conductances(second)
    own_labels = {bin_id: bin_id for bin_id in first_bins}
    first_outside = _outside_neighbors(first, own_labels, first_weights)
    second_outside = _outside_neighbors(second, to_first, second_weights)
    if Counter(neighbor[:3] for neighbor in first_outside) != Counter(
        neighbor[:3] for neighbor in second_outside
    ):
        return "connections from mapped bins to bins outside the mapping"
    if not _same_arena(first._arena, second._arena):
        return "continuous arena or physical walls"
    if not _same_conductances(
        _paired_conductances(first_weights, first_bins),
        _paired_conductances(second_weights, second_bins),
    ) or Counter(first_outside) != Counter(second_outside):
        return "smoothing conductances between neighboring bins"
    return None


def _paired_conductances(
    weights: sparse.csr_matrix | None, bins: list[int]
) -> sparse.csr_matrix | None:
    """Conductances among ``bins``, rows and columns in the order given."""
    return None if weights is None else weights[bins][:, bins]


def _outside_neighbors(
    env: Environment, labels: dict[int, int], weights: sparse.csr_matrix | None
) -> list[tuple[int, tuple[float, ...], float, float | None]]:
    """Mapped bins' unmapped neighbors as ``(label, center, distance, conductance)``.

    ``labels`` maps each mapped bin of ``env`` to its ID in the shared
    numbering, so the same neighbor of the same place counts alike in both
    environments. ``conductance`` is the smoothing conductance between the
    two bins, or None when ``env`` cannot smooth (``weights`` is None).
    """
    edges = list(nx.edge_boundary(env.connectivity, labels, data="distance"))
    if not edges:
        return []
    inside, outside, distances = zip(*edges, strict=True)
    conductances: list[float | None] = (
        [None] * len(edges)
        if weights is None
        else np.asarray(weights[list(inside), list(outside)]).ravel().tolist()
    )
    centers = env.bin_centers[list(outside)].tolist()
    return [
        (labels[u], tuple(center), distance, conductance)
        for u, center, distance, conductance in zip(
            inside, centers, distances, conductances, strict=True
        )
    ]


def _same_weighted_edges(first: nx.Graph, second: nx.Graph) -> bool:
    """Whether two graphs have the same edges with the same distances."""
    return first.number_of_edges() == second.number_of_edges() and all(
        second.has_edge(u, v) and data["distance"] == second.edges[u, v]["distance"]
        for u, v, data in first.edges(data=True)
    )


def _same_arena(first: Arena, second: Arena) -> bool:
    """Compare continuous meaning independently of polygon ring storage order."""
    if first is second:
        return True
    if any(
        getattr(first, name) != getattr(second, name)
        for name in (
            "n_dims",
            "coordinate_kind",
            "domain_known",
            "physical_footprint_known",
        )
    ):
        return False
    first_domain, second_domain = first.polygonal_domain(), second.polygonal_domain()
    if first_domain is not None or second_domain is not None:
        return (
            first_domain is not None
            and second_domain is not None
            and first_domain.equals(second_domain)
            and unary_union(first.walls).equals(unary_union(second.walls))
        )
    try:
        first_meta, first_arrays = encode_arena(first)
        second_meta, second_arrays = encode_arena(second)
    except ValueError:
        return False
    return (
        first_meta == second_meta
        and first_arrays.keys() == second_arrays.keys()
        and all(
            np.array_equal(value, second_arrays[key])
            for key, value in first_arrays.items()
        )
    )
