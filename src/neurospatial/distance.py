import networkx as nx
import numpy as np


def euclidean_distance_matrix(centers: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between points.

    Parameters
    ----------
    centers : np.ndarray, shape (N, n_dims)
        Array of N points in n_dims-dimensional space.

    Returns
    -------
    np.ndarray, shape (N, N)
        Pairwise Euclidean distance matrix where element (i, j) is the
        distance between points i and j.

    """
    from scipy.spatial.distance import pdist, squareform

    if centers.shape[0] == 0:
        return np.empty((0, 0), float)
    if centers.shape[0] == 1:
        return np.zeros((1, 1), float)
    return squareform(pdist(centers, metric="euclidean"))


def geodesic_distance_matrix(
    G: nx.Graph,
    n_states: int,
    weight: str = "distance",
) -> np.ndarray:
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
    np.ndarray, shape (n_states, n_states)
        Geodesic distance matrix where element (i, j) is the shortest path
        length from node i to node j. Disconnected nodes have distance np.inf.

    """
    if G.number_of_nodes() == 0:
        return np.empty((0, 0), float)
    dist_matrix = np.full((n_states, n_states), np.inf, dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)
    for src, lengths in nx.shortest_path_length(G, weight=weight):
        for dst, L in lengths.items():
            dist_matrix[src, dst] = L
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
        return nx.shortest_path_length(
            G,
            source=bin_from,
            target=bin_to,
            weight="distance",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return default
