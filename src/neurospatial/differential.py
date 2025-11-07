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
from numpy.typing import NDArray
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


def gradient(
    field: NDArray[np.float64],
    env: Environment | EnvironmentProtocol,
) -> NDArray[np.float64]:
    """Compute the gradient of a scalar field on the graph.

    The gradient operator transforms a scalar field defined on bins (nodes) into
    an edge field that represents the directional derivative along each edge.
    Mathematically, for a scalar field f and differential operator D:

        gradient(f) = D.T @ f

    Each edge's gradient value represents the change in the field value from the
    source node to the destination node, weighted by the square root of the edge
    distance (following graph signal processing convention).

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Scalar field defined on the environment's bins. Each element corresponds
        to a field value at a bin center.
    env : EnvironmentProtocol
        Environment with connectivity graph and differential operator. Must be
        fitted (i.e., created via a factory method like `Environment.from_samples()`).

    Returns
    -------
    gradient_field : NDArray[np.float64], shape (n_edges,)
        Edge field representing the gradient. Each element corresponds to the
        directional derivative along one edge in the connectivity graph.

    Raises
    ------
    ValueError
        If field shape does not match the number of bins in the environment.

    Notes
    -----
    The gradient operation is the adjoint of the divergence operation:

    - Gradient: scalar field → edge field (D.T @ f)
    - Divergence: edge field → scalar field (D @ g)
    - Laplacian: scalar field → scalar field (D @ D.T @ f = div(grad(f)))

    For regular grids, the gradient approximates the continuous spatial gradient
    via finite differences. For irregular graphs, the gradient follows the graph
    connectivity structure.

    Examples
    --------
    Compute gradient of a distance field (useful for goal-directed navigation):

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.differential import gradient
    >>> # Create 1D chain environment
    >>> data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    >>> env = Environment.from_samples(data, bin_size=1.0)
    >>> # Create distance field (distance from left end)
    >>> field = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> # Compute gradient
    >>> grad = gradient(field, env)
    >>> grad.shape
    (4,)
    >>> # For uniform spacing and linear field, gradient should be constant
    >>> np.allclose(np.abs(grad), np.abs(grad[0]), rtol=0.1)
    True

    Gradient of a constant field is zero:

    >>> const_field = np.ones(env.n_bins) * 5.0
    >>> grad_const = gradient(const_field, env)
    >>> np.allclose(grad_const, 0.0, atol=1e-10)
    True

    See Also
    --------
    divergence : Compute divergence of an edge field
    compute_differential_operator : Construct the differential operator matrix
    Environment.differential_operator : Cached differential operator property

    References
    ----------
    .. [1] Shuman et al. (2013). "The emerging field of signal processing on graphs."
           IEEE Signal Processing Magazine, 30(3), 83-98.
    """
    # Validate input shape
    if field.shape != (env.n_bins,):
        msg = (
            f"field must have shape ({env.n_bins},) to match environment bins, "
            f"but got shape {field.shape}"
        )
        raise ValueError(msg)

    # Compute gradient using differential operator transpose
    # gradient(f) = D.T @ f
    diff_op = env.differential_operator  # Use cached property
    gradient_field = diff_op.T @ field

    # Convert sparse result to dense array and ensure proper dtype
    if sparse.issparse(gradient_field):
        result: np.ndarray = np.asarray(gradient_field, dtype=np.float64).ravel()
    else:
        result = np.asarray(gradient_field, dtype=np.float64).ravel()

    return result
