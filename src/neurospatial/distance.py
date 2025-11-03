import networkx as nx
import numpy as np
from numpy.typing import NDArray


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
    n_states: int,
    weight: str = "distance",
) -> NDArray[np.float64]:
    """Compute geodesic (shortest-path) distance matrix on a graph.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing spatial connectivity.
    n_states : int
        Number of states/nodes in the graph.
    weight : str, default="distance"
        Edge attribute to use as weight for path length calculation.

    Returns
    -------
    NDArray[np.float64], shape (n_states, n_states)
        Geodesic distance matrix where element (i, j) is the shortest path
        length from node i to node j. Disconnected nodes have distance np.inf.

    """
    if G.number_of_nodes() == 0:
        return np.empty((0, 0), dtype=np.float64)
    dist_matrix = np.full((n_states, n_states), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_matrix, 0.0)
    for src, lengths in nx.shortest_path_length(G, weight=weight):
        for dst, L in lengths.items():
            dist_matrix[src, dst] = float(L)
    return dist_matrix


def geodesic_distance_between_points(
    G: nx.Graph,
    bin_from: int,
    bin_to: int,
    default: float = np.inf,
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
        Value to return if no path exists or nodes are invalid.

    Returns
    -------
    float
        Shortest path length from bin_from to bin_to using edge weight "distance".
        Returns `default` if either index is invalid or no path exists.

    """
    try:
        length = nx.shortest_path_length(
            G,
            source=bin_from,
            target=bin_to,
            weight="distance",
        )
        return float(length)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return default


def distance_field(
    G: nx.Graph,
    sources: list[int] | NDArray[np.int_],
    weight: str = "distance",
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
    weight : str, default="distance"
        Edge attribute to use as weight for path length calculation.

    Returns
    -------
    NDArray[np.float64], shape (n_nodes,)
        For each node i, the distance to the nearest source node.
        Nodes unreachable from all sources have distance np.inf.

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from neurospatial.distance import distance_field
    >>> # Create a simple graph
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    >>> for u, v in G.edges:
    ...     G.edges[u, v]["distance"] = 1.0
    >>> # Compute distance field from node 2
    >>> dists = distance_field(G, sources=[2])
    >>> dists
    array([2., 1., 0., 1., 2.])

    Notes
    -----
    This function uses Dijkstra's algorithm with multiple sources, which is
    O((V + E) log V) where V is number of nodes and E is number of edges.

    For large graphs or repeated queries, consider caching the result.

    See Also
    --------
    geodesic_distance_matrix : Compute all-pairs distances
    pairwise_distances : Compute distances between specific node pairs

    """
    sources_array = np.asarray(sources, dtype=int)

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return np.empty(0, dtype=np.float64)

    if len(sources_array) == 0:
        raise ValueError("sources must contain at least one node")

    # Initialize distance array
    distances = np.full(n_nodes, np.inf, dtype=np.float64)

    # Check that all source nodes are valid
    valid_sources = []
    for src in sources_array:
        if src in G.nodes:
            valid_sources.append(int(src))
        else:
            import warnings

            warnings.warn(f"Source node {src} not in graph, skipping", stacklevel=2)

    if len(valid_sources) == 0:
        raise ValueError("No valid source nodes found in graph")

    # Run Dijkstra from each source and keep minimum distance
    for src in valid_sources:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        for node, length in lengths.items():
            distances[node] = min(distances[node], float(length))

    return distances


def pairwise_distances(
    G: nx.Graph,
    nodes: list[int] | NDArray[np.int_],
    weight: str = "distance",
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
    weight : str, default="distance"
        Edge attribute to use as weight for path length calculation.

    Returns
    -------
    NDArray[np.float64], shape (n_nodes, n_nodes)
        Pairwise distance matrix where element (i, j) is the shortest path
        length between nodes[i] and nodes[j]. Disconnected nodes have distance np.inf.

    Examples
    --------
    >>> import networkx as nx
    >>> from neurospatial.distance import pairwise_distances
    >>> G = nx.cycle_graph(10)
    >>> for u, v in G.edges:
    ...     G.edges[u, v]["distance"] = 1.0
    >>> # Compute distances between nodes 0, 3, 7
    >>> dists = pairwise_distances(G, [0, 3, 7])
    >>> dists.shape
    (3, 3)
    >>> dists[0, 1]  # Distance from node 0 to node 3
    3.0

    See Also
    --------
    geodesic_distance_matrix : Compute all-pairs distances
    distance_field : Compute distance to nearest source

    """
    nodes_array = np.asarray(nodes, dtype=int)
    n = len(nodes_array)

    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    dist_matrix = np.full((n, n), np.inf, dtype=np.float64)

    # Compute distances
    for i, src in enumerate(nodes_array):
        if src not in G.nodes:
            continue

        # Set self-distance to 0 for valid nodes
        dist_matrix[i, i] = 0.0

        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        for j, dst in enumerate(nodes_array):
            if dst in lengths:
                dist_matrix[i, j] = float(lengths[dst])

    return dist_matrix
