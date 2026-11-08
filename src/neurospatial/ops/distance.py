"""Distance primitives for spatial environments.

Distances between bins, distance fields from one or more source bins, and
neighborhoods, in one of three metrics:

- ``"geodesic"`` (the default): the shortest path inside the environment,
  at any angle. It equals the straight line between two bin centers when
  nothing is in the way and goes around walls otherwise; it is the distance
  the behavior functions and ``decoding_error`` measure.
- ``"bin_graph"``: the walk along the environment's connections between
  bin centers. It is faster, especially for matrices and fine grids, but an
  oblique distance reads up to ``sqrt(4 - 2 sqrt(2))`` (about 1.082) times
  the straight line on an 8-connected square grid, ``2 / sqrt(3)`` (about
  1.155) on a hexagonal grid and about 1.128 on a 26-connected 3-D grid. It
  is exact between bin centers on a track, and it works on egocentric polar
  environments.
- ``"euclidean"``: the straight line between bin centers, through walls.

Hop counts (edges, not lengths) are ``Environment.reachable_from(...,
metric="hops")`` and ``Environment.rings``.

Functions
---------
euclidean_distance_matrix
    Pairwise Euclidean distance matrix between points.
geodesic_distance_matrix
    Distance between every pair of bins.
distance_field
    Distance from every bin to the nearest of one or more source bins.
pairwise_distances
    Distances between selected bins.
neighbors_within
    Bins within a given distance of each center bin.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, NoReturn

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from scipy.sparse import csr_array

    from neurospatial.environment import Environment
    from neurospatial.ops._types import DistanceMetric, PathMetric


def euclidean_distance_matrix(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pairwise Euclidean distance matrix between points.

    Parameters
    ----------
    centers : NDArray[np.float64], shape (N, n_dims)
        Array of N points in n_dims-dimensional space.

    Returns
    -------
    NDArray[np.float64], shape (N, N)
        Pairwise Euclidean distance matrix where element (i, j) is the
        distance between points i and j.

    """
    from scipy.spatial.distance import pdist, squareform

    if centers.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if centers.shape[0] == 1:
        return np.zeros((1, 1), dtype=np.float64)
    # scipy.spatial.distance functions return untyped arrays
    result: NDArray[np.float64] = squareform(pdist(centers, metric="euclidean"))
    return result


def geodesic_distance_matrix(
    env: Environment,
    *,
    metric: PathMetric = "geodesic",
) -> NDArray[np.float64]:
    """Distance between every pair of bins, along paths inside the environment.

    Parameters
    ----------
    env : Environment
        The environment whose bins are measured.
    metric : {"geodesic", "bin_graph"}, default="geodesic"
        ``"geodesic"`` is the shortest path between the two bin centers that
        stays inside the environment, at any angle: the straight line when
        nothing is in the way, around walls otherwise. ``"bin_graph"`` walks
        the connections between bin centers, each edge as long as the
        distance between its two centers. Choose it for speed (see Notes).

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_bins)
        Element ``(i, j)`` is the distance between bins ``i`` and ``j``;
        ``np.inf`` between disconnected parts. Symmetric.

    Raises
    ------
    CartesianOnlyError
        For ``metric="geodesic"`` on an egocentric polar environment, whose
        bins are (distance, angle) in the animal's frame. Use
        ``metric="bin_graph"`` there.
    ValueError
        If ``metric`` is not ``"geodesic"`` or ``"bin_graph"``.

    Notes
    -----
    **Geodesic.** A path grows from every bin (Theta*, as in the behavior
    functions); a path is not always the shortest one near walls, so entry
    ``(i, j)`` is the shorter of the paths grown from ``i`` and from ``j``.
    It costs one field per bin: about 3.5 s and 350 MB of working memory for
    2,500 bins in an open box on an Apple M5 Max, up to seven times more with
    walls (24 s with two walls), and on the order of a minute for 10,000
    bins. The result itself is ``n_bins**2 * 8`` bytes (800 MB for 10,000
    bins).

    **Bin graph.** ``scipy.sparse.csgraph.shortest_path`` on the bin graph,
    about 6-40 times faster (0.6 s for 2,500 bins). An oblique distance reads
    up to ``sqrt(4 - 2 * sqrt(2))`` (about 1.082) times the straight line on
    an 8-connected square grid, ``2 / sqrt(3)`` (about 1.155) on a hexagonal
    grid and about 1.128 on a 26-connected 3-D grid; between bin centers on
    a track it is exact. It works on egocentric polar environments.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.distance import geodesic_distance_matrix
    >>> edges = np.arange(0.0, 11.0, 2.0)
    >>> env = Environment.from_grid_mask(
    ...     np.ones((5, 5), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> # Corner to corner: the straight line, or a staircase on the graph.
    >>> distances = geodesic_distance_matrix(env)
    >>> round(float(distances[0, -1]), 3)  # sqrt(8**2 + 8**2)
    11.314
    >>> round(float(geodesic_distance_matrix(env, metric="bin_graph")[0, 2]), 3)
    4.0
    """
    from scipy.sparse.csgraph import shortest_path

    _require_path_metric(metric, "geodesic_distance_matrix()")
    if env.n_bins == 0:
        return np.empty((0, 0), dtype=np.float64)
    if metric == "bin_graph":
        adjacency = _bin_adjacency(
            env.connectivity, weight="distance", operation="geodesic_distance_matrix()"
        )
        dist_matrix: NDArray[np.float64] = shortest_path(
            csgraph=adjacency, method="auto", directed=False, return_predecessors=False
        )
        return dist_matrix
    return _symmetric_fields(env, np.arange(env.n_bins), "geodesic_distance_matrix")


def _symmetric_fields(
    env: Environment, bins: NDArray[np.intp], call: str
) -> NDArray[np.float64]:
    """Any-angle distances between ``bins``: the shorter of both directions."""
    return _between_bin_sets(env, bins, bins, call)


def _between_bin_sets(
    env: Environment, first: NDArray[np.intp], second: NDArray[np.intp], call: str
) -> NDArray[np.float64]:
    """Any-angle distance between each bin of ``first`` and each of ``second``.

    Entry ``(i, j)`` is the shorter of the paths grown from ``first[i]`` and
    from ``second[j]``. One field grows from each distinct bin of either set,
    a block at a time, so the working memory beyond the result is one block
    of ``(block, n_bins)`` rows however many pairs there are.

    Returns
    -------
    NDArray[np.float64], shape (len(first), len(second))
    """
    from neurospatial.ops._paths import _environment_paths

    paths = _environment_paths(env, call=call)
    first = np.asarray(first, dtype=np.intp)
    second = np.asarray(second, dtype=np.intp)
    roots, where = np.unique(np.concatenate([first, second]), return_inverse=True)
    root_of_first, root_of_second = where[: first.size], where[first.size :]
    out = np.full((first.size, second.size), np.inf)
    block = max(1, _DIJKSTRA_BLOCK_VALUES // max(env.n_bins, 1))
    for lo in range(0, roots.size, block):
        hi = min(lo + block, roots.size)
        # fields[k, b]: the path from bin b, grown from root lo + k.
        fields = paths.to_bins(roots[lo:hi])
        columns = np.flatnonzero((root_of_second >= lo) & (root_of_second < hi))
        if columns.size:
            grown = fields[root_of_second[columns] - lo][:, first].T
            out[:, columns] = np.minimum(out[:, columns], grown)
        rows = np.flatnonzero((root_of_first >= lo) & (root_of_first < hi))
        if rows.size:
            grown = fields[root_of_first[rows] - lo][:, second]
            out[rows] = np.minimum(out[rows], grown)
    return out


def _raise_missing_nodes(missing: list[int], operation: str) -> NoReturn:
    raise nx.NodeNotFound(
        _format_error(
            f"{operation}: bin(s) {missing} are not bins of the environment.",
            why="-1 marks a sample outside the environment, which has no "
            "distance; other values are not bins of this environment.",
            fix="drop the -1 samples first (bins[bins >= 0]), or pass bins "
            "from env.bin_at() of positions inside the environment",
        )
    )


def _require_bin_graph(graph: nx.Graph, operation: str) -> int:
    """Refuse a graph that is not an undirected simple graph of bins ``0 .. n - 1``.

    Distance arrays are indexed by bin, so row ``i`` must be node ``i``; each
    edge is one connection between two bins, with one length.

    Returns
    -------
    int
        The number of bins, ``graph.number_of_nodes()``.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError(
            _format_error(
                f"{operation}: the graph must be an undirected simple nx.Graph, "
                f"got {type(graph).__name__}.",
                why="bin distances are undirected and each pair of bins has one "
                "connection length, so reciprocal or parallel edges would be "
                "counted twice or read inconsistently.",
                fix="pass env.connectivity, or collapse the graph with "
                "nx.Graph(graph) after keeping one edge (the shortest) per pair of "
                "bins",
            )
        )
    n_bins: int = graph.number_of_nodes()
    if all(node in graph for node in range(n_bins)):
        return n_bins
    stray = [
        node
        for node in graph.nodes
        if not (isinstance(node, (int, np.integer)) and 0 <= node < n_bins)
    ]
    raise ValueError(
        _format_error(
            f"{operation}: the graph's nodes must be the bin indices 0 to "
            f"{n_bins - 1}; nodes {stray[:5]} are not.",
            why="Entry i of the result is the distance for bin i, so every "
            "node must be the index of the bin it stands for.",
            fix="pass env.connectivity, or relabel the graph with "
            "nx.convert_node_labels_to_integers(graph, ordering='sorted') and "
            "translate your bin labels the same way",
        )
    )


def _require_edge_weight(graph: nx.Graph, weight: str | None, operation: str) -> None:
    """Refuse a ``weight`` attribute that some edges lack.

    NetworkX and SciPy would otherwise count such an edge as length 1, so a
    misspelled attribute would silently give hop counts.
    """
    if weight is None:
        return
    measured = nx.get_edge_attributes(graph, weight)
    if len(measured) == graph.number_of_edges():
        return
    unmeasured = next((u, v) for u, v in graph.edges if (u, v) not in measured)
    present = sorted(graph.edges[unmeasured])
    raise ValueError(
        _format_error(
            f"{operation}: {graph.number_of_edges() - len(measured)} edge(s) have "
            f"no {weight!r} attribute, for example edge {unmeasured} "
            f"(its attributes: {present}).",
            why="an edge without a length would count as 1, so distances "
            "would silently become edge counts.",
            fix="pass weight= one of the edge attributes (env.connectivity "
            "edges carry 'distance'), or weight=None to count edges",
        )
    )


def _bin_adjacency(graph: nx.Graph, *, weight: str | None, operation: str) -> csr_array:
    """Sparse edge-length matrix whose row and column ``i`` are bin ``i``.

    Parameters
    ----------
    graph : nx.Graph
        Graph whose nodes are the bins ``0 .. n_bins - 1``.
    weight : str or None
        Edge attribute holding each edge's length; None counts edges.
    operation : str
        Calling function, named in errors.

    Returns
    -------
    scipy.sparse.csr_array of float64, shape (n_bins, n_bins)
        Symmetric edge lengths (1.0 per edge when ``weight`` is None), for
        ``scipy.sparse.csgraph`` with ``directed=False``.

    Raises
    ------
    ValueError
        If ``graph`` is directed or a multigraph, its nodes are not
        ``0 .. n_bins - 1``, or an edge lacks ``weight``.
    """
    from scipy.sparse import csr_array

    n_bins = _require_bin_graph(graph, operation)
    n_edges = graph.number_of_edges()
    if weight is None:
        pairs = np.fromiter(
            (node for edge in graph.edges for node in edge),
            dtype=np.intp,
            count=2 * n_edges,
        ).reshape(n_edges, 2)
        lengths = np.ones(n_edges)
    else:
        # One pass reads each edge as (u, v, length); a missing length reads
        # as NaN, told apart from a stored NaN only on that rare path.
        rows = np.fromiter(
            (
                value
                for edge in graph.edges(data=weight, default=np.nan)
                for value in edge
            ),
            dtype=np.float64,
            count=3 * n_edges,
        ).reshape(n_edges, 3)
        if np.isnan(rows[:, 2]).any():
            _require_edge_weight(graph, weight, operation)
        pairs, lengths = rows[:, :2].astype(np.intp), rows[:, 2]
    u, v = pairs[:, 0], pairs[:, 1]
    mirror = u != v  # a self-loop is one entry, as in nx.to_scipy_sparse_array
    return csr_array(
        (
            np.concatenate([lengths, lengths[mirror]]),
            (np.concatenate([u, v[mirror]]), np.concatenate([v, u[mirror]])),
        ),
        shape=(n_bins, n_bins),
    )


# Dijkstra runs per block of source bins: (block, n_bins) float64 rows stay
# near 160 MB however large the environment.
_DIJKSTRA_BLOCK_VALUES = 20_000_000


def _dijkstra_blocks(
    adjacency: csr_array, sources: NDArray[np.intp], *, limit: float = np.inf
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """Shortest-path rows from each source, in memory-bounded blocks.

    Yields
    ------
    start : int
        Position in ``sources`` of the block's first source.
    lengths : ndarray of float64, shape (n_block, n_bins)
        Path length from each source in the block to every bin; ``inf`` when
        unreachable or farther than ``limit``.
    """
    from scipy.sparse.csgraph import dijkstra

    block = max(1, _DIJKSTRA_BLOCK_VALUES // max(adjacency.shape[0], 1))
    for start in range(0, len(sources), block):
        lengths = dijkstra(
            adjacency,
            directed=False,
            indices=sources[start : start + block],
            limit=limit,
        )
        yield start, lengths


def _geodesic_pair_distances(
    graph: nx.Graph,
    sources: NDArray[np.intp],
    targets: NDArray[np.intp],
    *,
    operation: str,
    weight: str | None = "distance",
) -> NDArray[np.float64]:
    """Shortest-path length along ``graph`` for each ``(source, target)`` bin pair.

    Runs Dijkstra only from the distinct source bins, in blocks, instead of
    building the dense ``(n_bins, n_bins)`` all-pairs matrix.

    Parameters
    ----------
    graph : nx.Graph
        Graph whose nodes are the bins ``0 .. n_bins - 1``.
    sources, targets : NDArray[np.intp], shape (n_pairs,)
        Bin index pairs, each in ``0 .. n_bins - 1``.
    operation : str
        Calling function, named in errors.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length; None counts edges.

    Returns
    -------
    NDArray[np.float64], shape (n_pairs,)
        Path length per pair; ``inf`` between disconnected components.
    """
    adjacency = _bin_adjacency(graph, weight=weight, operation=operation)
    unique_sources, source_row = np.unique(sources, return_inverse=True)
    out = np.empty(len(sources), dtype=np.float64)
    for start, lengths in _dijkstra_blocks(adjacency, unique_sources):
        in_block = (source_row >= start) & (source_row < start + len(lengths))
        out[in_block] = lengths[source_row[in_block] - start, targets[in_block]]
    return out


def _require_metric(metric: str, operation: str) -> None:
    if metric not in ("geodesic", "bin_graph", "euclidean"):
        raise ValueError(
            _format_error(
                f"{operation}: metric must be 'geodesic', 'bin_graph' or "
                f"'euclidean', got {metric!r}.",
                why="distances are measured along the shortest path inside the "
                "environment, along the connections between bin centers, or "
                "along straight lines.",
                fix="use metric='geodesic' to go around walls, metric='bin_graph' "
                "for the faster walk between bin centers, or metric='euclidean' "
                "for straight lines",
            )
        )


def _require_path_metric(metric: str, operation: str) -> None:
    if metric not in ("geodesic", "bin_graph"):
        raise ValueError(
            _format_error(
                f"{operation}: metric must be 'geodesic' or 'bin_graph', got "
                f"{metric!r}.",
                why="this function measures paths inside the environment.",
                fix="use metric='geodesic' for the shortest path at any angle, or "
                "metric='bin_graph' for the faster walk between bin centers",
            )
        )


def _require_cartesian_centers(env: Environment, operation: str) -> None:
    from neurospatial.geometry import require_cartesian

    require_cartesian(
        env.coordinate_system,
        f"{operation}(metric='euclidean')",
        why="The straight-line distance between bin centers treats them as "
        "Cartesian positions.",
        fix="use metric='bin_graph', which walks the connections between bin centers",
    )


def _as_bin_indices(
    values: Iterable[int] | NDArray[np.int_],
    n_bins: int,
    operation: str,
    argument: str,
) -> NDArray[np.intp]:
    """Integer bin indices of a ``0 .. n_bins - 1`` graph, refusing others.

    A fractional or boolean value names no bin, and -1 marks a sample outside
    the environment; neither is truncated, dropped or read as the last bin.
    """
    array = np.asarray(values)
    if array.size and array.dtype.kind not in "iu":
        raise TypeError(
            _format_error(
                f"{operation}: {argument} must be integer bin indices, got "
                f"dtype {array.dtype}.",
                why="a fractional or boolean value names no bin, and rounding "
                "it would silently pick another bin.",
                fix="pass bin indices such as env.bin_at(positions), or "
                "np.flatnonzero(mask) for a boolean mask of bins",
            )
        )
    indices = array.astype(np.intp).ravel()
    outside = (indices < 0) | (indices >= n_bins)
    if outside.any():
        _raise_missing_nodes(np.unique(indices[outside]).tolist(), operation)
    return indices


def distance_field(
    env: Environment,
    sources: Iterable[int] | NDArray[np.int_],
    *,
    metric: DistanceMetric = "geodesic",
    cutoff: float | None = None,
) -> NDArray[np.float64]:
    """Distance from every bin to the nearest source bin.

    A common primitive for spatial analysis: the distance from every bin to
    the nearest of a set of bins, such as goal locations, reward sites or
    boundaries.

    Parameters
    ----------
    env : Environment
        The environment whose bins are measured.
    sources : sequence of int or NDArray[np.int_]
        Source bin indices. Each bin gets the distance to the nearest one.
    metric : {"geodesic", "bin_graph", "euclidean"}, default="geodesic"
        - ``"geodesic"``: the shortest path from each bin center to the
          nearest source center that stays inside the environment, at any
          angle. It goes around walls, and in an open arena it is the
          straight line.
        - ``"bin_graph"``: the walk along the connections between bin
          centers, each edge as long as the distance between its centers.
          It is faster, but an oblique distance reads up to about 1.082
          times the straight line on an 8-connected square grid (1.155 on a
          hexagonal grid, 1.128 on a 26-connected 3-D grid); exact between
          bin centers on a track. It works on egocentric polar environments.
        - ``"euclidean"``: the straight line between bin centers, through
          walls.
    cutoff : float, optional
        Bins farther than ``cutoff`` from every source are ``np.inf``. A
        short cutoff makes the geodesic and bin-graph fields much faster,
        since they are grown only as far as it.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Distance from each bin to its nearest source; ``np.inf`` where no
        source can be reached or beyond ``cutoff``.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the three, ``cutoff`` is negative, or
        ``sources`` is empty.
    TypeError
        If ``sources`` are not integers.
    networkx.NodeNotFound
        If a source is not a bin, such as the outside marker -1.
    CartesianOnlyError
        For ``metric="geodesic"`` or ``"euclidean"`` on an egocentric polar
        environment. Use ``metric="bin_graph"`` there.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.distance import distance_field
    >>> edges = np.arange(0.0, 11.0, 2.0)
    >>> env = Environment.from_grid_mask(
    ...     np.ones((5, 5), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> field = distance_field(env, sources=[0])
    >>> round(float(field[-1]), 3)  # the straight line to the far corner
    11.314
    >>> round(float(distance_field(env, sources=[0], metric="bin_graph")[7]), 3)
    4.828
    >>> distance_field(env, sources=[0], cutoff=3.0)[:3].tolist()
    [0.0, 2.0, inf]

    Notes
    -----
    **Geodesic.** One any-angle field (Theta*, as the behavior functions
    use) grown from the sources; each bin's value is the length of its path
    toward the nearest source. A field costs about 10 ms for 2,500 open bins
    (50 ms with walls) and 33 ms for 10,000, and it is kept, so asking again
    is a lookup.

    **Bin graph.** One Dijkstra search from all sources together,
    ``O((V + E) log V)``, a few milliseconds.

    **Euclidean.** A KD-tree query over the source centers.

    See Also
    --------
    geodesic_distance_matrix : Distance between every pair of bins.
    pairwise_distances : Distances between selected bins.
    neighbors_within : Bins within a radius.
    """
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    _require_metric(metric, "distance_field()")
    if cutoff is not None and not cutoff >= 0:  # NaN is no distance
        msg = (
            f"cutoff must be non-negative, got {cutoff}. "
            f"Use cutoff=None for no distance limit, or provide a positive value."
        )
        raise ValueError(msg)

    n_bins = env.n_bins
    if n_bins == 0:
        return np.empty(0, dtype=np.float64)
    sources_array = _as_bin_indices(sources, n_bins, "distance_field()", "sources")
    if len(sources_array) == 0:
        msg = (
            "sources must contain at least one node. "
            "Provide a non-empty list or array of source node indices."
        )
        raise ValueError(msg)
    limit = np.inf if cutoff is None else float(cutoff)

    if metric == "euclidean":
        _require_cartesian_centers(env, "distance_field()")
        coords = np.asarray(env.bin_centers, dtype=np.float64)
        # A KD-tree over the sources is O(N log M) time and O(N + M) memory,
        # however many sources there are.
        distances, _ = cKDTree(coords[sources_array]).query(coords, k=1)
        if cutoff is not None:
            distances = np.where(distances <= cutoff, distances, np.inf)
        return np.asarray(distances, dtype=np.float64)

    if metric == "geodesic":
        from neurospatial.ops._paths import _environment_paths

        paths = _environment_paths(env, call="distance_field")
        return paths.from_bins(np.unique(sources_array), limit=limit)

    adjacency = _bin_adjacency(
        env.connectivity, weight="distance", operation="distance_field()"
    )
    # One Dijkstra from all sources together gives each bin its nearest
    # source's distance directly, without an (n_sources, n_nodes) array.
    distances = dijkstra(
        csgraph=adjacency,
        directed=False,
        indices=sources_array,
        limit=limit,
        min_only=True,
    )
    return np.asarray(distances, dtype=np.float64)


def pairwise_distances(
    env: Environment,
    bins: Iterable[int] | NDArray[np.int_],
    *,
    metric: DistanceMetric = "geodesic",
) -> NDArray[np.float64]:
    """Distances between every pair of the given bins.

    Cheaper than :func:`geodesic_distance_matrix` when only some bins are
    needed.

    Parameters
    ----------
    env : Environment
        The environment whose bins are measured.
    bins : sequence of int or NDArray[np.int_]
        Bin indices to measure between.
    metric : {"geodesic", "bin_graph", "euclidean"}, default="geodesic"
        As in :func:`distance_field`: the shortest path inside the
        environment at any angle, the walk along the connections between
        bin centers (faster; up to about 1.082 times the straight line on a
        square grid, 1.155 hexagonal, 1.128 in 3-D; works on polar
        environments), or the straight line.

    Returns
    -------
    NDArray[np.float64], shape (len(bins), len(bins))
        Element ``(i, j)`` is the distance between ``bins[i]`` and
        ``bins[j]``; ``np.inf`` between disconnected parts. The geodesic
        matrix is symmetric: each entry is the shorter of the paths grown
        from either bin.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the three.
    TypeError
        If ``bins`` are not integers.
    networkx.NodeNotFound
        If a bin index is not a bin, such as the outside marker -1.
    CartesianOnlyError
        For ``metric="geodesic"`` or ``"euclidean"`` on an egocentric polar
        environment.

    Notes
    -----
    The geodesic matrix grows one any-angle field per bin: about 0.2 s for
    200 bins of a 2,500-bin open box and 1.3 s for 200 of 10,000 bins (1.5 s
    with walls), against 0.04 s and 0.2 s for the bin graph. The fields are
    kept, so asking again is fast.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.distance import pairwise_distances
    >>> edges = np.arange(0.0, 11.0, 2.0)
    >>> env = Environment.from_grid_mask(
    ...     np.ones((5, 5), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> distances = pairwise_distances(env, [0, 7, 24])
    >>> distances.shape
    (3, 3)
    >>> round(float(distances[0, 1]), 3)  # 2 cm across, 4 cm up: sqrt(20)
    4.472
    >>> round(float(pairwise_distances(env, [0, 7], metric="bin_graph")[0, 1]), 3)
    4.828

    See Also
    --------
    geodesic_distance_matrix : Distance between every pair of bins.
    distance_field : Distance to the nearest source.
    """
    from scipy.sparse.csgraph import dijkstra

    _require_metric(metric, "pairwise_distances()")
    bins_array = _as_bin_indices(bins, env.n_bins, "pairwise_distances()", "bins")
    if len(bins_array) == 0:
        return np.empty((0, 0), dtype=np.float64)
    if metric == "euclidean":
        _require_cartesian_centers(env, "pairwise_distances()")
        return euclidean_distance_matrix(
            np.asarray(env.bin_centers, dtype=np.float64)[bins_array]
        )
    if metric == "geodesic":
        return _symmetric_fields(env, bins_array, "pairwise_distances")

    adjacency = _bin_adjacency(
        env.connectivity, weight="distance", operation="pairwise_distances()"
    )
    # Run Dijkstra from all bins at once; shape (n_bins_given, n_bins).
    dist_from_sources = dijkstra(
        csgraph=adjacency,
        directed=False,
        indices=bins_array,
        return_predecessors=False,
    )
    dist_matrix: NDArray[np.float64] = dist_from_sources[:, bins_array]
    return dist_matrix


def neighbors_within(
    env: Environment,
    centers: Iterable[int] | NDArray[np.int_],
    *,
    radius: float,
    metric: DistanceMetric = "geodesic",
    include_center: bool = True,
) -> list[NDArray[np.int_]]:
    """Bins within ``radius`` of each center bin.

    Parameters
    ----------
    env : Environment
        The environment whose bins are measured.
    centers : sequence of int or NDArray[np.int_]
        Center bin indices to query around.
    radius : float
        Neighborhood radius, in the environment's units.
    metric : {"geodesic", "bin_graph", "euclidean"}, default="geodesic"
        As in :func:`distance_field`: the shortest path inside the
        environment at any angle (a round neighborhood in an open arena),
        the walk along the connections between bin centers (faster; an
        octagon on a square grid, since an oblique distance reads up to
        about 1.082 times the straight line, 1.155 on a hexagonal grid,
        1.128 in 3-D; works on polar environments), or the straight line
        through walls. For neighborhoods counted in edges, use
        ``Environment.reachable_from(..., metric="hops")``.
    include_center : bool, default=True
        Whether to include the center bin itself in each neighborhood.

    Returns
    -------
    list[NDArray[np.int_]]
        For each center, the unique bin indices within ``radius``, in no
        particular order.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the three, or ``radius`` is negative.
    TypeError
        If ``centers`` are not integers.
    networkx.NodeNotFound
        If a center is not a bin, such as the outside marker -1.
    CartesianOnlyError
        For ``metric="geodesic"`` or ``"euclidean"`` on an egocentric polar
        environment.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.distance import neighbors_within
    >>> edges = np.arange(0.0, 11.0, 2.0)
    >>> env = Environment.from_grid_mask(
    ...     np.ones((5, 5), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> # Within 4.5 cm of the middle bin: the 3 x 3 block and the 4 bins
    >>> # 4 cm away along the axes, but not the corners sqrt(20) cm away.
    >>> len(neighbors_within(env, [12], radius=4.0)[0])
    13

    Notes
    -----
    The geodesic and bin-graph searches stop at ``radius``: a geodesic
    neighborhood grows only the bins whose centers lie within ``radius`` of
    the center (no path is shorter than the straight line), about 25 ms for
    200 centers in a 2,500-bin box and 0.1 s in a 10,000-bin one.

    See Also
    --------
    distance_field : Distance to the nearest source.
    pairwise_distances : Distances between selected bins.
    """
    from scipy.spatial import cKDTree

    _require_metric(metric, "neighbors_within()")
    if not radius >= 0:  # NaN is no distance
        msg = (
            f"radius must be non-negative, got {radius}. "
            f"Provide a positive radius value or 0 for only center nodes."
        )
        raise ValueError(msg)

    centers_array = _as_bin_indices(
        centers, env.n_bins, "neighbors_within()", "centers"
    )
    if len(centers_array) == 0:
        return []

    if metric == "euclidean":
        _require_cartesian_centers(env, "neighbors_within()")
        coords = np.asarray(env.bin_centers, dtype=np.float64)
        neighborhoods = [
            np.asarray(neigh_list, dtype=np.int_)
            for neigh_list in cKDTree(coords).query_ball_point(
                coords[centers_array], r=radius
            )
        ]
    elif metric == "geodesic":
        from neurospatial.ops._paths import _environment_paths

        paths = _environment_paths(env, call="neighbors_within")
        block = max(1, _DIJKSTRA_BLOCK_VALUES // max(env.n_bins, 1))
        neighborhoods = [
            np.flatnonzero(np.isfinite(row)).astype(np.int_)
            for lo in range(0, centers_array.size, block)
            for row in paths.to_bins(
                centers_array[lo : lo + block], limit=float(radius)
            )
        ]
    else:
        adjacency = _bin_adjacency(
            env.connectivity, weight="distance", operation="neighbors_within()"
        )
        neighborhoods = [
            np.flatnonzero(np.isfinite(row)).astype(np.int_)
            for _, lengths in _dijkstra_blocks(adjacency, centers_array, limit=radius)
            for row in lengths
        ]
    if include_center:
        return neighborhoods
    return [
        neigh[neigh != c] for c, neigh in zip(centers_array, neighborhoods, strict=True)
    ]
