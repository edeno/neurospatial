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
