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
from typing import TYPE_CHECKING, Any, NamedTuple

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

from neurospatial.geometry import (
    ClippedLocator,
    HexagonShapes,
    MissingCellGeometryError,
    PolygonCells,
    SubsetLocator,
    TriangleShapes,
    encode,
    encode_arena,
    materialize_cells,
)

if TYPE_CHECKING:
    from scipy import sparse

    from neurospatial.environment.core import Environment
    from neurospatial.geometry import (
        Arena,
        Binning,
        Locator,
        RectilinearGrid,
        TrackLinearization,
    )


class _EnvironmentDifferences(NamedTuple):
    """The first difference in bin meaning and in topology, or None for each.

    Attributes
    ----------
    bins : str or None
        First difference in what the bins mean: ``"coordinate systems"``,
        ``"units or coordinate frames"``, ``"bin coordinates or ordering"``,
        ``"bin measures"``, ``"polygon cell geometry"``,
        ``"bin-boundary ownership"``, ``"cell shapes"`` or
        ``"track geometry"``.
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
    hexagons, or grid boxes, axis-aligned or in an affine frame). Two tracks
    are compared by their segments (ordered endpoint positions and lengths,
    and the spacing between segments) and by the linear edges of each bin:
    bins with equal centers can run along different segments. When either
    environment records neither cells nor a track, such as explicit point
    bins, the bins are compared by center and measure only. Each pair of
    bins is compared by ``_paired_cell_differences``, the one definition of
    the same cell that bin mappings use too.

    Smoothing passes a field between neighbors through conductances (face
    area over distance), which the connectivity graph does not hold: two
    environments with the same bins, neighbors, distances and arena can
    smooth differently. Their conductances are compared too, and an
    environment that cannot smooth differs from one that can.
    """
    if first is second:
        return _EnvironmentDifferences(None, None)
    labels = _label_difference(first, second)
    if first._geometry is second._geometry:
        return _EnvironmentDifferences(labels, None)
    return _EnvironmentDifferences(
        labels or _bin_difference(first, second), _topology_difference(first, second)
    )


def _label_difference(first: Environment, second: Environment) -> str | None:
    """The first difference in coordinate labels, or None.

    Returns
    -------
    str or None
        ``"coordinate systems"`` or ``"units or coordinate frames"``. No
        pairing of bins can bridge either: the same numbers mean different
        places.
    """
    if first.coordinate_system != second.coordinate_system:
        return "coordinate systems"
    if first.units != second.units or first.frame != second.frame:
        return "units or coordinate frames"
    return None


def _bin_difference(first: Environment, second: Environment) -> str | None:
    """Return the first difference between published bins with equal labels."""
    if first.n_bins != second.n_bins:
        return "bin coordinates or ordering"
    bins = np.arange(first.n_bins)
    reason = _mapped_bin_difference(first, second, np.column_stack([bins, bins]))
    return "bin coordinates or ordering" if reason == "bin coordinates" else reason


def _mapped_bin_difference(
    first: Environment, second: Environment, bin_pairs: NDArray[np.integer]
) -> str | None:
    """Compare paired bins as ``_environment_differences`` compares all bins.

    Parameters
    ----------
    first, second : Environment
        The environments to compare. Their coordinate system, units and
        frame are not compared here.
    bin_pairs : ndarray of int, shape (n_pairs, 2)
        Rows of ``(bin of first, bin of second)`` meant to be the same place.

    Returns
    -------
    str or None
        The first difference between paired bins, in the order of
        ``_CELL_DIFFERENCES``: ``"bin coordinates"``, ``"bin measures"``,
        ``"polygon cell geometry"``, ``"bin-boundary ownership"``,
        ``"cell shapes"`` or ``"track geometry"``; None when every pair is
        the same cell (see ``_paired_cell_differences``). The identity
        pairing of two environments with the same bins gives the answer of
        ``_environment_differences``.
    """
    codes = _paired_cell_differences(first, second, bin_pairs[:, 0], bin_pairs[:, 1])
    failed = codes[codes >= 0]
    return _CELL_DIFFERENCES[int(failed.min())] if failed.size else None


# What can differ between two bins meant to be the same cell, in the order
# they are checked and reported.
_CELL_DIFFERENCES = (
    "bin coordinates",
    "bin measures",
    "polygon cell geometry",
    "bin-boundary ownership",
    "cell shapes",
    "track geometry",
)
# A polygon cell's center and area are computed from its polygon, so the same
# polygon can give centers and areas that differ by this relative rounding.
_ROUNDING = 64 * np.finfo(np.float64).eps


def _compares_polygons(first: Binning, second: Binning) -> bool:
    """Whether paired bins are compared as exact polygon cells."""
    return (
        _records_cells(first)
        and _records_cells(second)
        and (
            isinstance(first.shapes, PolygonCells)
            or isinstance(second.shapes, PolygonCells)
        )
    )


def _paired_cell_differences(
    first: Environment,
    second: Environment,
    rows_first: NDArray[np.integer],
    rows_second: NDArray[np.integer],
) -> NDArray[np.intp]:
    """The first difference between each pair of bins, or -1 for the same cell.

    This is the package's one definition of "the same cell". Two bins are the
    same cell when they have the same center and measure, and, where both
    environments record cells, the same cell:

    - Grid boxes, triangles and hexagons: exactly equal centers, measures and
      corners.
    - Polygon cells (either environment clipped or built from polygons):
      exactly the same polygon, owning the same boundary points (see
      ``_ownership_keys``). Their centers and areas are computed from the
      polygon, so they may differ from the other bin's by rounding: within
      ``_ROUNDING`` of the coordinates' magnitude plus the cell's width for
      centers, and of the area for measures.
    - Track bins (both environments built on a track): exactly equal
      centers and measures, the same linear ``[low, high)`` edges along the
      track, and the same track segments and spacing (see
      ``_track_layout``).
    - Other bins without recorded cells, such as explicit point bins:
      exactly equal centers and measures.

    Parameters
    ----------
    first, second : Environment
        The environments of each pair's bins.
    rows_first, rows_second : ndarray of int, shape (n_pairs,)
        The paired bins of ``first`` and of ``second``.

    Returns
    -------
    ndarray of intp, shape (n_pairs,)
        Index into ``_CELL_DIFFERENCES`` of each pair's first difference, or
        -1 when the bins are the same cell.
    """
    first_binning, second_binning = first._binning, second._binning
    first_bins, second_bins = first_binning.bins, second_binning.bins
    codes = np.full(len(rows_first), -1, dtype=np.intp)

    def mark(code: int, differs: NDArray[np.bool_]) -> None:
        codes[(codes < 0) & differs] = code

    if first_bins.centers.shape[1] != second_bins.centers.shape[1]:
        # Coordinates of different widths are different places; comparing
        # them elementwise would broadcast a 1-D center against each axis.
        codes[:] = 0
        return codes
    first_centers = first_bins.centers[rows_first]
    second_centers = second_bins.centers[rows_second]
    first_sizes = first_bins.sizes[rows_first]
    second_sizes = second_bins.sizes[rows_second]
    exact_centers = np.all(first_centers == second_centers, axis=1)
    exact_sizes = first_sizes == second_sizes
    same_dims = (
        first_bins.measure_dims[rows_first] == second_bins.measure_dims[rows_second]
    )
    if _compares_polygons(first_binning, second_binning):
        first_cells = _cell_array(first_binning)[rows_first]
        second_cells = _cell_array(second_binning)[rows_second]
        same_polygon = np.asarray(shapely.equals(first_cells, second_cells), bool)
        allowance = _ROUNDING * (
            np.maximum(np.abs(first_centers), np.abs(second_centers))
            + np.sqrt(np.abs(first_sizes))[:, np.newaxis]
        )
        close_centers = np.all(
            np.abs(first_centers - second_centers) <= allowance, axis=1
        )
        close_sizes = np.abs(first_sizes - second_sizes) <= _ROUNDING * np.abs(
            first_sizes
        )
        mark(0, ~(exact_centers | (same_polygon & close_centers)))
        mark(1, ~same_dims | ~(exact_sizes | (same_polygon & close_sizes)))
        mark(2, ~same_polygon)
        rows = np.flatnonzero(codes < 0)
        owned = _same_ownership(
            first_binning.locator,
            second_binning.locator,
            np.asarray(rows_first)[rows],
            np.asarray(rows_second)[rows],
        )
        codes[rows[~owned]] = 3
        return codes
    mark(0, ~exact_centers)
    mark(1, ~(exact_sizes & same_dims))
    first_track, second_track = (
        first_binning.linearization,
        second_binning.linearization,
    )
    if first_track is not None and second_track is not None:
        # Exact equality: layouts and edges are stored numbers, not results
        # of arithmetic on the bins.
        same_layout = _track_layout(first_track) == _track_layout(second_track)
        first_bounds = _track_bounds(first_track)[rows_first]
        second_bounds = _track_bounds(second_track)[rows_second]
        same_bounds = np.all(first_bounds == second_bounds, axis=1)
        mark(5, ~(same_layout & same_bounds))
    if not (_records_cells(first_binning) and _records_cells(second_binning)):
        return codes
    first_corners = _cell_corners(first_binning)
    second_corners = _cell_corners(second_binning)
    if first_corners is None or second_corners is None:
        return codes
    if first_corners.shape[1:] != second_corners.shape[1:]:
        mark(4, np.ones(len(codes), dtype=bool))
        return codes
    mark(
        4,
        ~np.all(first_corners[rows_first] == second_corners[rows_second], axis=(1, 2)),
    )
    return codes


def _track_layout(track: TrackLinearization) -> tuple[Any, ...]:
    """Ordered segments (start, end positions, distance) and spacing.

    Node labels are not compared: two tracks with the same segments in the
    same order are the same track whatever their nodes are called. A single
    spacing is the same gap between every pair of successive segments.
    """
    graph = track.graph
    segments = tuple(
        (graph.nodes[u]["pos"], graph.nodes[v]["pos"], graph.edges[u, v]["distance"])
        for u, v in track.edge_order
    )
    spacing = track.edge_spacing
    gaps = (
        (spacing,) * max(len(segments) - 1, 0)
        if isinstance(spacing, float)
        else tuple(spacing)
    )
    return segments, gaps


def _track_bounds(track: TrackLinearization) -> NDArray[np.float64]:
    """Linear [low, high) of each bin, shape (n_bins, 2)."""
    cells = np.flatnonzero(track.active_mask)
    edges = track.grid_edges[0]
    return np.column_stack([edges[cells], edges[cells + 1]])


def _cell_array(binning: Binning) -> NDArray[np.object_]:
    """Each bin's polygon, as an object array for vectorized shapely calls."""
    cells = materialize_cells(binning)
    array = np.empty(len(cells), dtype=object)
    array[:] = cells
    return array


def _records_cells(binning: Binning) -> bool:
    """Whether the binning records each bin's cell, not only its center."""
    return (
        isinstance(binning.shapes, PolygonCells | TriangleShapes | HexagonShapes)
        or binning.rectilinear is not None
        or binning.grid_frame is not None
    )


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


def _ownership_keys(
    locator: Locator, rows: NDArray[np.integer]
) -> tuple[NDArray[np.object_], NDArray[np.intp]]:
    """The locator that decides each bin's boundary points, and its bin there.

    A subset gives a point to the kept bin whose parent owned it, and to no
    bin when that parent was dropped, so its bins own their parents' points.
    A clip gives a point to the piece of the parent bin that owned it, so a
    parent bin's only piece owns the parent's points inside that piece.
    Callers compare keys only for bins whose polygons are already known to be
    equal, so a piece and a whole cell with the same key own the same points.
    Pieces of a cell split in several are kept at their clip, whose lookup
    breaks ties between them.
    """
    keys: NDArray[np.intp] = np.array(rows, dtype=np.intp)
    roots = np.empty(len(keys), dtype=object)
    active = np.ones(len(keys), dtype=bool)
    while np.any(active):
        if isinstance(locator, SubsetLocator):
            parent_to_child = locator.parent_to_child
            kept = np.flatnonzero(parent_to_child >= 0)
            child_to_parent = np.empty(kept.size, dtype=np.intp)
            child_to_parent[parent_to_child[kept]] = kept
            keys[active] = child_to_parent[keys[active]]
            locator = locator.base
        elif isinstance(locator, ClippedLocator):
            sources = locator.mapping.parent_bin_indices
            pieces = np.bincount(sources, minlength=locator.mapping.n_parent_bins)
            only_piece = active.copy()
            only_piece[active] = pieces[sources[keys[active]]] == 1
            roots[active & ~only_piece] = locator
            active = only_piece
            keys[active] = sources[keys[active]]
            locator = locator.base
        else:
            roots[active] = locator
            break
    return roots, keys


def _same_ownership(
    first: Locator,
    second: Locator,
    rows_first: NDArray[np.integer],
    rows_second: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """Whether each pair's bins own the same boundary points."""
    first_roots, first_keys = _ownership_keys(first, rows_first)
    second_roots, second_keys = _ownership_keys(second, rows_second)
    same: NDArray[np.bool_] = np.asarray(first_keys == second_keys, dtype=bool)
    verdicts: dict[tuple[int, int], bool] = {}
    for row in np.flatnonzero(same):
        first_root, second_root = first_roots[row], second_roots[row]
        key = (id(first_root), id(second_root))
        if key not in verdicts:
            verdicts[key] = _same_locator(first_root, second_root)
        same[row] = verdicts[key]
    return same


def _same_locator(first: Locator, second: Locator) -> bool:
    """Whether two locators are the same, as their saved records compare."""
    if first is second:
        return True
    try:
        first_meta, first_arrays = encode(first, "locator")
        second_meta, second_arrays = encode(second, "locator")
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
    """Smoothing conductances between bins, or None when ``env`` cannot smooth.

    An environment cannot smooth when its bins have no finite-volume cell
    faces (including a combined environment with such a component), or when
    it is a combined environment moved by a transform that is not a
    similarity. Any other error propagates.
    """
    try:
        return env._diffusion_geometry.W
    except MissingCellGeometryError:
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
