"""Differential operators for graph signal processing on spatial environments.

This module implements the differential operator framework for graph-discretized
spatial environments, enabling gradient, divergence, and Laplacian computations
on irregular spatial graphs.

The differential operator D is a sparse matrix (n_bins × n_edges) that encodes
the oriented edge structure of the connectivity graph. It satisfies the fundamental
relationship: L = D @ D.T, where L is the graph Laplacian.

References
----------
.. [1] PyGSP: Graph Signal Processing in Python
       https://pygsp.readthedocs.io/
.. [2] Shuman et al. (2013). "The emerging field of signal processing on graphs."
       IEEE Signal Processing Magazine, 30(3), 83-98.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol
    from neurospatial.environment.core import Environment


def compute_differential_operator(
    env: Environment | EnvironmentProtocol,
) -> sparse.csc_matrix:
    """Compute the differential operator matrix for graph signal processing.

    The differential operator D is a sparse matrix of shape (n_bins, n_edges)
    that encodes the oriented edge structure of the connectivity graph. For each
    edge e = (i, j) with weight w_e (distance):

    - D[i, e] = -sqrt(w_e)  (source node)
    - D[j, e] = +sqrt(w_e)  (destination node)

    This construction ensures the fundamental relationship: L = D @ D.T,
    where L is the graph Laplacian matrix.

    Parameters
    ----------
    env : EnvironmentProtocol
        Environment with a connectivity graph. Must have a `connectivity`
        attribute containing a NetworkX graph with 'distance' edge attributes.

    Returns
    -------
    D : scipy.sparse.csc_matrix
        Sparse differential operator matrix of shape (n_bins, n_edges).
        Stored in Compressed Sparse Column (CSC) format for efficient
        matrix-vector products.

    Notes
    -----
    The differential operator provides the foundation for gradient and divergence
    operations on graph-discretized spatial fields:

    - Gradient: grad(f) = D.T @ f  (scalar field → edge field)
    - Divergence: div(g) = D @ g   (edge field → scalar field)
    - Laplacian: lap(f) = D @ D.T @ f = div(grad(f))

    Edge weights (distances) are incorporated via their square root, following
    the weighted graph Laplacian convention in graph signal processing [1]_.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.differential import compute_differential_operator
    >>> # Create a simple 1D chain environment
    >>> data = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> env = Environment.from_samples(data, bin_size=1.0)
    >>> # Compute differential operator
    >>> D = compute_differential_operator(env)
    >>> D.shape
    (4, 3)
    >>> # Verify Laplacian relationship: L = D @ D.T
    >>> import networkx as nx
    >>> L_from_D = D @ D.T
    >>> L_nx = nx.laplacian_matrix(env.connectivity, weight="distance")
    >>> np.allclose(L_from_D.toarray(), L_nx.toarray())
    True

    See Also
    --------
    Environment.differential_operator : Cached property for accessing this operator

    References
    ----------
    .. [1] PyGSP: Graph Signal Processing in Python
           https://pygsp.readthedocs.io/
    """
    # Get number of bins and edges
    n_bins = env.n_bins
    n_edges = len(env.connectivity.edges)

    # Handle edge case: no edges (single node or disconnected graph)
    if n_edges == 0:
        return sparse.csc_matrix((n_bins, 0), dtype=np.float64)

    # Preallocate arrays for sparse matrix construction (COO format)
    # Each edge contributes 2 entries (source and destination)
    row_indices = np.zeros(2 * n_edges, dtype=np.int32)
    col_indices = np.zeros(2 * n_edges, dtype=np.int32)
    data_values = np.zeros(2 * n_edges, dtype=np.float64)

    # Build differential operator entries
    idx = 0
    for edge_id, (i, j, edge_data) in enumerate(env.connectivity.edges(data=True)):
        # Get edge weight (distance)
        distance = edge_data["distance"]

        # Compute sqrt of weight for differential operator
        sqrt_weight = np.sqrt(distance)

        # Source node gets negative weight
        row_indices[idx] = i
        col_indices[idx] = edge_id
        data_values[idx] = -sqrt_weight
        idx += 1

        # Destination node gets positive weight
        row_indices[idx] = j
        col_indices[idx] = edge_id
        data_values[idx] = +sqrt_weight
        idx += 1

    # Construct sparse matrix in COO format, then convert to CSC
    d_coo = sparse.coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_bins, n_edges),
        dtype=np.float64,
    )

    # Convert to CSC for efficient column-wise operations
    d_csc = d_coo.tocsc()

    return d_csc
