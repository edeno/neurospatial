"""Distance primitives for spatial environments.

Low-level helpers for computing Euclidean and graph-geodesic distances between
bins (nodes), distance fields from one or more source bins, and local
neighborhood lookups. These operate on arrays of bin centers (shape
``(n_bins, n_dims)``) or on the connectivity graph of an environment.

Functions
---------
euclidean_distance_matrix
    Pairwise Euclidean distance matrix between bin centers.
geodesic_distance_matrix
    Pairwise shortest-path (graph) distance matrix between bins.
geodesic_distance_between_points
    Geodesic distance between two specific bins.
distance_field
    Shortest-path distance from one or more source bins to every bin.
pairwise_distances
    Pairwise distances (Euclidean or geodesic) between selected bins.
neighbors_within
    Bins within a given distance of a source bin.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, NoReturn

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from scipy.sparse import csr_array


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
    G: nx.Graph,
    *,
    weight: str | None = "distance",
) -> NDArray[np.float64]:
    """Compute geodesic (shortest-path) distance matrix on a graph.

    Uses scipy's optimized shortest path implementation for improved performance.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity, such as
        ``env.connectivity``. Its nodes must be the bin indices
        ``0 .. n_bins - 1``.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length. None counts edges (hops).

    Returns
    -------
    NDArray[np.float64], shape (n_bins, n_bins)
        Geodesic distance matrix where element (i, j) is the shortest path
        length from bin i to bin j. Disconnected bins have distance np.inf.

    Raises
    ------
    ValueError
        If the nodes of ``G`` are not exactly ``0 .. n_bins - 1``, or an edge
        lacks the ``weight`` attribute.

    Notes
    -----
    This function uses scipy.sparse.csgraph.shortest_path for optimized
    performance (~10-100× faster than pure NetworkX implementation).
    The algorithm automatically selects between Dijkstra's algorithm
    (for sparse graphs) and Floyd-Warshall (for dense graphs).

    Examples
    --------
    >>> import networkx as nx
    >>> from neurospatial.ops.distance import geodesic_distance_matrix
    >>> G = nx.Graph()
    >>> # Rows follow the bin index, whatever order the edges were added in.
    >>> G.add_edge(2, 3, distance=1.0)
    >>> G.add_edge(0, 1, distance=1.0)
    >>> G.add_edge(1, 2, distance=2.0)
    >>> geodesic_distance_matrix(G)[0].tolist()
    [0.0, 1.0, 3.0, 4.0]

    """
    from scipy.sparse.csgraph import shortest_path

    adjacency = _bin_adjacency(G, weight=weight, operation="geodesic_distance_matrix()")
    if adjacency.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Compute shortest paths using scipy's optimized implementation
    dist_matrix: NDArray[np.float64] = shortest_path(
        csgraph=adjacency,
        method="auto",  # Automatically choose best algorithm
        directed=False,  # Undirected graphs
        return_predecessors=False,
    )

    return dist_matrix


def geodesic_distance_between_points(
    G: nx.Graph,
    bin_from: int,
    bin_to: int,
    *,
    default: float = np.inf,
    weight: str | None = "distance",
) -> float:
    """Compute geodesic distance between two specific nodes in a graph.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity.
    bin_from : int
        Source node/bin index.
    bin_to : int
        Target node/bin index.
    default : float, default=np.inf
        Value to return if no path exists.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length. None counts edges (hops).

    Returns
    -------
    float
        Shortest path length from bin_from to bin_to, summing ``weight``.
        Returns `default` if no path exists.

    Raises
    ------
    networkx.NodeNotFound
        If either bin is not a node of ``G``, such as the outside marker -1:
        a missing bin has no distance, which is not the same as no path.
    ValueError
        If an edge lacks the ``weight`` attribute.

    """
    _require_nodes(G, [bin_from, bin_to], "geodesic_distance_between_points()")
    _require_edge_weight(G, weight, "geodesic_distance_between_points()")
    try:
        length = nx.shortest_path_length(
            G,
            source=bin_from,
            target=bin_to,
            weight=weight,
        )
        return float(length)
    except nx.NetworkXNoPath:
        return default


def _require_nodes(G: nx.Graph, nodes: Iterable[int], operation: str) -> None:
    """Refuse bins that are not graph nodes, naming the outside marker -1."""
    missing = [int(node) for node in nodes if node not in G]
    if missing:
        _raise_missing_nodes(missing, operation)


def _raise_missing_nodes(missing: list[int], operation: str) -> NoReturn:
    raise nx.NodeNotFound(
        _format_error(
            f"{operation}: bin(s) {missing} are not nodes of the graph.",
            why="-1 marks a sample outside the environment, which has no "
            "graph distance; other values are not bins of this graph.",
            fix="drop the -1 samples first (bins[bins >= 0]), or pass bins "
            "from env.bin_at() of positions inside the environment",
        )
    )


def _require_bin_graph(G: nx.Graph, operation: str) -> int:
    """Refuse a graph whose nodes are not exactly the bins ``0 .. n - 1``.

    Distance arrays are indexed by bin, so row ``i`` must be node ``i``.

    Returns
    -------
    int
        The number of bins, ``G.number_of_nodes()``.
    """
    n_bins = G.number_of_nodes()
    if all(node in G for node in range(n_bins)):
        return n_bins
    stray = [
        node
        for node in G.nodes
        if not (isinstance(node, (int, np.integer)) and 0 <= node < n_bins)
    ]
    raise ValueError(
        _format_error(
            f"{operation}: the graph's nodes must be the bin indices 0 to "
            f"{n_bins - 1}; nodes {stray[:5]} are not.",
            why="Entry i of the result is the distance for bin i, so every "
            "node must be the index of the bin it stands for.",
            fix="pass env.connectivity, or relabel the graph with "
            "nx.convert_node_labels_to_integers(G, ordering='sorted') and "
            "translate your bin labels the same way",
        )
    )


def _require_edge_weight(G: nx.Graph, weight: str | None, operation: str) -> None:
    """Refuse a ``weight`` attribute that some edges lack.

    NetworkX and SciPy would otherwise count such an edge as length 1, so a
    misspelled attribute would silently give hop counts.
    """
    if weight is None:
        return
    measured = nx.get_edge_attributes(G, weight)
    if len(measured) == G.number_of_edges():
        return
    unmeasured = next((u, v) for u, v in G.edges if (u, v) not in measured)
    present = sorted(G.edges[unmeasured])
    raise ValueError(
        _format_error(
            f"{operation}: {G.number_of_edges() - len(measured)} edge(s) have "
            f"no {weight!r} attribute, for example edge {unmeasured} "
            f"(its attributes: {present}).",
            why="an edge without a length would count as 1, so distances "
            "would silently become edge counts.",
            fix="pass weight= one of the edge attributes (env.connectivity "
            "edges carry 'distance'), or weight=None to count edges",
        )
    )


def _bin_adjacency(G: nx.Graph, *, weight: str | None, operation: str) -> csr_array:
    """Sparse edge-length matrix whose row and column ``i`` are bin ``i``.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes are the bins ``0 .. n_bins - 1``.
    weight : str or None
        Edge attribute holding each edge's length; None counts edges.
    operation : str
        Calling function, named in errors.

    Returns
    -------
    scipy.sparse.csr_array, shape (n_bins, n_bins)
        Edge lengths, for ``scipy.sparse.csgraph`` with ``directed=False``.

    Raises
    ------
    ValueError
        If the nodes are not ``0 .. n_bins - 1`` or an edge lacks ``weight``.
    """
    from scipy.sparse import csr_array

    n_bins = _require_bin_graph(G, operation)
    if n_bins == 0:
        return csr_array((0, 0), dtype=np.float64)
    _require_edge_weight(G, weight, operation)
    adjacency: csr_array = nx.to_scipy_sparse_array(
        G, nodelist=range(n_bins), weight=weight, format="csr"
    )
    return adjacency


def _validate_source_nodes(
    sources: NDArray[np.int_],
    G: nx.Graph,
) -> list[int]:
    """Validate source nodes and return list of valid sources.

    Parameters
    ----------
    sources : NDArray[np.int_]
        Array of source node indices to validate.
    G : nx.Graph
        Graph to check nodes against.

    Returns
    -------
    list[int]
        List of valid source node indices that exist in the graph.

    Raises
    ------
    ValueError
        If no valid source nodes are found after validation. Error message
        includes the invalid indices and suggests checking node IDs.

    Warnings
    --------
    UserWarning
        Issued for each source node that is not found in the graph, with
        the invalid node ID.

    """
    import warnings

    # Vectorized membership check using numpy
    sources_arr = np.asarray(sources)
    graph_nodes = np.array(list(G.nodes))

    # Use np.isin for efficient membership testing
    valid_mask = np.isin(sources_arr, graph_nodes)
    valid_sources: list[int] = sources_arr[valid_mask].astype(int).tolist()
    invalid_sources: list[int] = sources_arr[~valid_mask].astype(int).tolist()

    # Issue warnings for invalid sources (preserves original behavior)
    if invalid_sources:
        min_node = min(G.nodes) if G.nodes else "none"
        max_node = max(G.nodes) if G.nodes else "none"
        for src in invalid_sources:
            warnings.warn(
                f"Source node {src} not in graph (valid node IDs: "
                f"{min_node} to {max_node}), skipping",
                stacklevel=3,
                category=UserWarning,
            )

    if len(valid_sources) == 0:
        raise ValueError(
            f"No valid source nodes found in graph. Invalid nodes: {invalid_sources}. "
            f"Valid node IDs range from {min(G.nodes) if G.nodes else 'none'} "
            f"to {max(G.nodes) if G.nodes else 'none'}. "
            f"Check that source indices match actual graph node IDs."
        )

    return valid_sources


def distance_field(
    G: nx.Graph,
    sources: list[int] | NDArray[np.int_],
    *,
    metric: str = "geodesic",
    bin_centers: NDArray[np.float64] | None = None,
    weight: str | None = "distance",
    cutoff: float | None = None,
) -> NDArray[np.float64]:
    """Compute distance field: distance from each node to nearest source node.

    This is a common primitive for spatial analysis - compute the distance
    from every bin to the nearest bin in a set of source bins (e.g., goal
    locations, reward sites, or boundaries).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity.
    sources : list[int] or NDArray[np.int_]
        List of source node indices. Distance field measures distance to
        nearest node in this set.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric to use:

        - 'geodesic': Shortest-path length on graph using edge weights.
          Respects graph connectivity.
        - 'euclidean': Straight-line L2 distance in coordinate space.
          Requires ``bin_centers`` parameter.
    bin_centers : NDArray[np.float64], shape (n_nodes, n_dims), optional
        Node coordinates in N-dimensional space. Required when
        ``metric='euclidean'``. Ignored for geodesic metric.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length; None counts edges (hops).
        Only used when ``metric='geodesic'``.
    cutoff : float, optional
        Maximum distance threshold. Nodes farther than ``cutoff`` from
        all sources will have distance ``np.inf``. Interpretation:

        - Geodesic: Path length limit (uses Dijkstra cutoff)
        - Euclidean: Coordinate distance limit (post-computed clipping)

    Returns
    -------
    NDArray[np.float64], shape (n_nodes,)
        For each node i, the distance to the nearest source node.
        Nodes unreachable from all sources (or beyond cutoff) have
        distance ``np.inf``.

    Raises
    ------
    ValueError
        If metric is not 'geodesic' or 'euclidean'. Valid options: 'geodesic', 'euclidean'.
    ValueError
        If cutoff is negative. Must be non-negative or None.
    ValueError
        If sources list is empty. Provide at least one source node.
    ValueError
        If metric='euclidean' but bin_centers is None. Provide bin_centers array.
    ValueError
        If bin_centers shape doesn't match number of graph nodes. Ensure
        bin_centers.shape[0] == G.number_of_nodes().
    ValueError
        If no valid source nodes found after validation. Check that source
        indices match actual graph node IDs.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from neurospatial.ops.distance import distance_field
    >>> # Create a simple graph
    >>> G = nx.Graph()  # doctest: +SKIP
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])  # doctest: +SKIP
    >>> for u, v in G.edges:  # doctest: +SKIP
    ...     G.edges[u, v]["distance"] = 1.0
    >>> # Geodesic distance field from node 2
    >>> dists = distance_field(G, sources=[2])  # doctest: +SKIP
    >>> dists  # doctest: +SKIP
    array([2., 1., 0., 1., 2.])
    >>> # With cutoff
    >>> dists = distance_field(G, sources=[2], cutoff=1.5)  # doctest: +SKIP
    >>> dists  # doctest: +SKIP
    array([inf, 1., 0., 1., inf])

    Notes
    -----
    **Geodesic mode**: Uses Dijkstra's algorithm with multiple sources,
    which is O((V + E) log V) where V is number of nodes and E is number
    of edges. The ``cutoff`` parameter is passed directly to Dijkstra
    for efficiency.

    **Euclidean mode**: Computes L2 distances from node coordinates with a
    KD-tree query over the source positions.

    For large graphs or repeated queries, consider caching the result.

    See Also
    --------
    geodesic_distance_matrix : Compute all-pairs geodesic distances
    pairwise_distances : Compute distances between specific node pairs
    neighbors_within : Find all neighbors within radius

    """
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    # Validate inputs
    if metric not in ("geodesic", "euclidean"):
        raise ValueError(
            f"metric must be 'geodesic' or 'euclidean', got '{metric}'. "
            f"Use metric='geodesic' for graph-based distances or "
            f"metric='euclidean' for coordinate-based distances."
        )

    if cutoff is not None and cutoff < 0:
        raise ValueError(
            f"cutoff must be non-negative, got {cutoff}. "
            f"Use cutoff=None for no distance limit, or provide a positive value."
        )

    sources_array = np.asarray(sources, dtype=int).ravel()

    n_nodes = _require_bin_graph(G, "distance_field()")
    if n_nodes == 0:
        return np.empty(0, dtype=np.float64)

    if len(sources_array) == 0:
        raise ValueError(
            "sources must contain at least one node. "
            "Provide a non-empty list or array of source node indices."
        )

    # Euclidean metric requires bin_centers
    if metric == "euclidean":
        if bin_centers is None:
            raise ValueError(
                "bin_centers parameter is required when metric='euclidean'. "
                "Provide an array of node coordinates with shape (n_nodes, n_dims)."
            )
        if bin_centers.shape[0] != n_nodes:
            raise ValueError(
                f"bin_centers size {bin_centers.shape[0]} must match "
                f"number of nodes in graph ({n_nodes}). "
                f"Ensure bin_centers.shape[0] == G.number_of_nodes()."
            )

        # Validate sources exist in graph (use consistent validation)
        valid_sources = _validate_source_nodes(sources_array, G)

        # A KD-tree over the sources is O(N log M) time and O(N + M) memory,
        # however many sources there are.
        tree = cKDTree(bin_centers[valid_sources])
        distances, _ = tree.query(bin_centers, k=1)

        # Apply cutoff
        if cutoff is not None:
            distances = np.where(distances <= cutoff, distances, np.inf)

        return np.asarray(distances, dtype=np.float64)

    # Geodesic metric - use consistent validation
    valid_sources = _validate_source_nodes(sources_array, G)

    adjacency = _bin_adjacency(G, weight=weight, operation="distance_field()")

    # Run Dijkstra from all sources at once (vectorized)
    # Returns shape (n_sources, n_nodes)
    dist_from_sources = dijkstra(
        csgraph=adjacency,
        directed=False,
        indices=valid_sources,
        limit=cutoff if cutoff is not None else np.inf,
        return_predecessors=False,
    )

    # Take minimum distance across all sources (vectorized)
    if len(valid_sources) == 1:
        distances = dist_from_sources.ravel()
    else:
        distances = np.min(dist_from_sources, axis=0)

    return np.asarray(distances, dtype=np.float64)


def pairwise_distances(
    G: nx.Graph,
    nodes: list[int] | NDArray[np.int_],
    *,
    weight: str | None = "distance",
) -> NDArray[np.float64]:
    """Compute pairwise geodesic distances between specified nodes.

    This is more efficient than computing the full distance matrix when you
    only need distances between a subset of nodes.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity.
    nodes : list[int] or NDArray[np.int_]
        List of node indices to compute distances between.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length; None counts edges (hops).

    Returns
    -------
    NDArray[np.float64], shape (n_nodes, n_nodes)
        Pairwise distance matrix where element (i, j) is the shortest path
        length between nodes[i] and nodes[j]. Disconnected nodes have distance np.inf.

    Raises
    ------
    networkx.NodeNotFound
        If a node is not in ``G``, such as the outside marker -1.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurospatial.ops.distance import pairwise_distances
    >>> G = nx.cycle_graph(10)  # doctest: +SKIP
    >>> for u, v in G.edges:  # doctest: +SKIP
    ...     G.edges[u, v]["distance"] = 1.0
    >>> # Compute distances between nodes 0, 3, 7
    >>> dists = pairwise_distances(G, [0, 3, 7])  # doctest: +SKIP
    >>> dists.shape  # doctest: +SKIP
    (3, 3)
    >>> dists[0, 1]  # Distance from node 0 to node 3  # doctest: +SKIP
    3.0

    See Also
    --------
    geodesic_distance_matrix : Compute all-pairs distances
    distance_field : Compute distance to nearest source

    """
    from scipy.sparse.csgraph import dijkstra

    nodes_array = np.asarray(nodes, dtype=int)
    n = len(nodes_array)

    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    _require_nodes(G, nodes_array, "pairwise_distances()")

    adjacency = _bin_adjacency(G, weight=weight, operation="pairwise_distances()")

    # Run Dijkstra from all nodes at once; shape (n_nodes, n_graph_nodes).
    dist_from_sources = dijkstra(
        csgraph=adjacency,
        directed=False,
        indices=nodes_array,
        return_predecessors=False,
    )
    dist_matrix: NDArray[np.float64] = dist_from_sources[:, nodes_array]
    return dist_matrix


def neighbors_within(
    G: nx.Graph,
    centers: list[int] | NDArray[np.int_],
    *,
    radius: float,
    metric: str = "geodesic",
    bin_centers: NDArray[np.float64] | None = None,
    weight: str = "distance",
    include_center: bool = True,
) -> list[NDArray[np.int_]]:
    """Return bins within radius of each center bin.

    For each center bin, find all bins whose distance (geodesic or Euclidean)
    is within the specified radius.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity.
    centers : list[int] or NDArray[np.int_]
        Center bin indices to query around.
    radius : float
        Neighborhood radius. Interpreted in:

        - Geodesic: Graph path length units (using edge ``weight`` attribute)
        - Euclidean: Coordinate units (L2 distance in ``bin_centers``)
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance metric to use:

        - 'geodesic': Shortest-path length on graph
        - 'euclidean': Straight-line L2 distance
    bin_centers : NDArray[np.float64], shape (n_nodes, n_dims), optional
        Node coordinates. Required when ``metric='euclidean'``.
    weight : str or None, default="distance"
        Edge attribute holding each edge's length, for geodesic distances;
        None counts edges (hops).
    include_center : bool, default=True
        Whether to include the center bin itself in each neighborhood.

    Returns
    -------
    list[NDArray[np.int_]]
        For each center, a 1-D array of bin indices within ``radius``.
        Order of bins within each array is unspecified. Arrays contain
        unique indices (no duplicates).

    Raises
    ------
    ValueError
        If metric is not 'geodesic' or 'euclidean'. Valid options: 'geodesic', 'euclidean'.
    ValueError
        If radius is negative. Must be non-negative.
    ValueError
        If metric='euclidean' but bin_centers is None. Provide bin_centers array.
    ValueError
        If bin_centers shape doesn't match number of graph nodes. Ensure
        bin_centers.shape[0] == G.number_of_nodes().
    nx.NodeNotFound
        If a center node is not in the graph. Raised with context about valid
        node ID range.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurospatial.ops.distance import neighbors_within
    >>> # Create line graph
    >>> G = nx.Graph()  # doctest: +SKIP
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])  # doctest: +SKIP
    >>> for u, v in G.edges:  # doctest: +SKIP
    ...     G.edges[u, v]["distance"] = 1.0
    >>> # Find neighbors within distance 1.5 of node 2
    >>> neighborhoods = neighbors_within(
    ...     G, centers=[2], radius=1.5, metric="geodesic"
    ... )  # doctest: +SKIP
    >>> sorted(neighborhoods[0])  # doctest: +SKIP
    [1, 2, 3]

    Notes
    -----
    **Geodesic mode**: Uses single-source Dijkstra per center with
    ``cutoff=radius`` for efficiency. Complexity per center is
    O((V + E) log V) but early termination typically makes it faster.

    **Euclidean mode**: Uses KD-tree ball query over ``bin_centers``.
    Complexity is O(log V) per center on average.

    See Also
    --------
    distance_field : Compute distance to nearest source
    pairwise_distances : Compute distances between specific pairs

    """
    from scipy.spatial import cKDTree

    # Validate inputs
    if metric not in ("geodesic", "euclidean"):
        raise ValueError(
            f"metric must be 'geodesic' or 'euclidean', got '{metric}'. "
            f"Use metric='geodesic' for graph-based distances or "
            f"metric='euclidean' for coordinate-based distances."
        )

    if radius < 0:
        raise ValueError(
            f"radius must be non-negative, got {radius}. "
            f"Provide a positive radius value or 0 for only center nodes."
        )

    n_nodes = _require_bin_graph(G, "neighbors_within()")
    centers_array = np.asarray(centers, dtype=int).ravel()

    if len(centers_array) == 0:
        return []

    # Euclidean metric
    if metric == "euclidean":
        if bin_centers is None:
            raise ValueError(
                "bin_centers parameter is required when metric='euclidean'. "
                "Provide an array of node coordinates with shape (n_nodes, n_dims)."
            )
        if bin_centers.shape[0] != n_nodes:
            raise ValueError(
                f"bin_centers size {bin_centers.shape[0]} must match "
                f"number of nodes in graph ({n_nodes}). "
                f"Ensure bin_centers.shape[0] == G.number_of_nodes()."
            )

        # Build KD-tree once
        tree = cKDTree(bin_centers)

        # Query neighborhoods
        center_coords = bin_centers[centers_array]
        neighborhoods_list = tree.query_ball_point(center_coords, r=radius)

        # Convert to numpy arrays and optionally exclude centers
        result = []
        for c, neigh_list in zip(centers_array, neighborhoods_list, strict=True):
            arr = np.asarray(neigh_list, dtype=np.int_)
            if not include_center:
                arr = arr[arr != c]
            result.append(arr)

        return result

    # Geodesic metric - validate centers and provide helpful errors
    _require_edge_weight(G, weight, "neighbors_within()")
    result = []
    for c in centers_array:
        if c not in G.nodes:
            node_range = (
                f"{min(G.nodes)} to {max(G.nodes)}" if G.nodes else "no nodes in graph"
            )
            raise nx.NodeNotFound(
                f"Center node {c} not in graph. Valid node IDs range from {node_range}. "
                f"Check that center index matches an actual graph node ID."
            )

        # Dijkstra from center with cutoff
        lengths = nx.single_source_dijkstra_path_length(
            G, source=int(c), cutoff=radius, weight=weight
        )

        neigh = np.fromiter(lengths.keys(), dtype=np.int_, count=len(lengths))

        if not include_center:
            neigh = neigh[neigh != c]

        result.append(neigh)

    return result
