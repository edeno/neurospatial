"""Spatial signal processing primitives for graph-based operations.

This module provides graph-based operations for spatial signal processing
including neighborhood aggregation and convolution on irregular graphs.

Functions
---------
neighbor_reduce
    Aggregate field values over spatial neighborhoods.
graph_convolve
    Convolve a spatial field with a custom kernel on the graph.

Examples
--------
>>> from neurospatial.ops.graph import neighbor_reduce, graph_convolve

Import via ops package:

>>> from neurospatial.ops import neighbor_reduce, graph_convolve
"""

from __future__ import annotations

__all__ = ["graph_convolve", "neighbor_reduce"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.ops._protocols import GraphConsumer


def neighbor_reduce(
    env: GraphConsumer,
    field: NDArray[np.float64],
    *,
    op: Literal["sum", "mean", "max", "min", "std"] = "mean",
    weights: NDArray[np.float64] | None = None,
    include_self: bool = False,
) -> NDArray[np.float64]:
    """
    Aggregate field values over spatial neighborhoods.

    Applies a reduction operation (sum, mean, max, min, std) to the values
    of neighboring bins in the spatial graph. This is a fundamental primitive
    for spatial signal processing on irregular graphs.

    Parameters
    ----------
    env : Environment
        Spatial environment providing graph connectivity.
    field : array, shape (n_bins,)
        Scalar field values at each bin.
    op : {'sum', 'mean', 'max', 'min', 'std'}, default='mean'
        Reduction operation to apply over neighbors.
    weights : array, shape (n_bins,), optional
        Weights for each bin value. If provided, performs weighted aggregation
        (only valid for 'sum' and 'mean' operations). Default is None (unweighted).
    include_self : bool, default=False
        If True, include the bin itself in the neighborhood aggregation.
        If False, only consider graph neighbors.

    Returns
    -------
    result : array, shape (n_bins,)
        Field values after neighborhood aggregation. Bins with no neighbors
        return NaN, and so does every bin with a NaN (missing) value in its
        neighborhood, for every ``op``: a missing neighbor is neither zero
        nor ignored.

    Raises
    ------
    ValueError
        If field shape doesn't match environment, if weights are provided for
        incompatible operations, or if operation is invalid.

    Notes
    -----
    **Applications**:

    - **Coherence**: Spatial correlation between firing rate and neighbor average
      (Muller & Kubie 1989). Computed as correlation of field and
      ``neighbor_reduce(env, field, op='mean')``.
    - **Smoothness**: Measure how much field values differ from local neighborhood.
    - **Local statistics**: Compute local variability (op='std') or extrema
      (op='max'/'min').

    **Algorithm**:

    1. For each bin, retrieve neighbors from ``env.connectivity`` graph
    2. Optionally include self in neighborhood
    3. Apply reduction operation (sum, mean, max, min, std)
    4. Handle isolated nodes (no neighbors) → return NaN

    **Missing values (NaN)**: a NaN neighbor makes the result NaN for every
    ``op``, since the sum, mean, extremes and spread of a neighborhood with an
    unobserved bin are unknown. To aggregate only the observed neighbors, pass
    ``op="mean"`` with ``weights`` that are zero at missing bins and a finite
    placeholder value there, for example
    ``neighbor_reduce(env, np.where(observed, field, 0.0), weights=observed.astype(float))``;
    a bin whose neighbors are all missing then has zero total weight and
    stays NaN.

    **Performance**: ``sum`` and ``mean`` are vectorized as a sparse
    matrix-vector product against the CSR adjacency matrix (O(n_edges)).
    ``max``/``min``/``std`` are not linear and use a per-bin loop over
    neighbors (O(n_bins × avg_degree)).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.graph import neighbor_reduce
    >>> # Create 3x3 grid
    >>> positions = np.array(
    ...     [
    ...         [0, 0],
    ...         [1, 0],
    ...         [2, 0],
    ...         [0, 1],
    ...         [1, 1],
    ...         [2, 1],
    ...         [0, 2],
    ...         [1, 2],
    ...         [2, 2],
    ...     ]
    ... )
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> # Field with bin indices
    >>> field = np.arange(env.n_bins, dtype=float)
    >>> # Mean of neighbors
    >>> neighbor_mean = neighbor_reduce(env, field, op="mean")
    >>> # from_samples connects diagonal neighbors too, so the center bin (4)
    >>> # has 8 neighbors: every other bin. Their mean is (36 - 4) / 8 = 4.0.
    >>> sorted(env.connectivity.neighbors(4))
    [0, 1, 2, 3, 5, 6, 7, 8]
    >>> print(f"Center neighbor mean: {neighbor_mean[4]:.1f}")
    Center neighbor mean: 4.0

    See Also
    --------
    Environment.smooth : Boundary-aware finite-volume diffusion smoothing
    graph_convolve : General convolution with custom kernels

    References
    ----------
    .. [1] Muller & Kubie (1989). The firing of hippocampal place cells
           predicts the future position of freely moving rats. Journal of
           Neuroscience, 9(12).
    """
    # Validate inputs
    if field.shape[0] != env.n_bins:
        raise ValueError(
            f"field.shape must match environment bins (got field.shape={field.shape}, "
            f"expected ({env.n_bins},))"
        )

    valid_ops = {"sum", "mean", "max", "min", "std"}
    if op not in valid_ops:
        raise ValueError(f"op must be one of {valid_ops}, got '{op}'")

    if weights is not None:
        if weights.shape[0] != env.n_bins:
            raise ValueError(
                f"weights.shape must match environment bins (got weights.shape={weights.shape}, "
                f"expected ({env.n_bins},))"
            )
        if op not in {"sum", "mean"}:
            raise ValueError(
                f"weights are only supported for 'sum' and 'mean' operations, got '{op}'"
            )

    # ``sum`` and ``mean`` are linear over the neighborhood and can be expressed
    # as a sparse matrix-vector product against the (binary) adjacency matrix.
    # ``max``/``min``/``std`` are not linear, so they keep the loop fallback.
    if op == "sum" or op == "mean":
        return _neighbor_reduce_linear(
            env, field, op=op, weights=weights, include_self=include_self
        )

    result = np.full(env.n_bins, np.nan, dtype=np.float64)
    # Loop fallback for non-linear reductions (max, min, std).
    for bin_id in range(env.n_bins):
        # Get neighbors from connectivity graph
        neighbors = list(env.connectivity.neighbors(bin_id))

        # Optionally include self
        if include_self:
            neighbors = [bin_id, *neighbors]

        # Handle isolated nodes (no neighbors)
        if len(neighbors) == 0:
            continue  # result[bin_id] remains NaN

        # Get neighbor values
        neighbor_values = field[neighbors]

        # Apply reduction operation (non-linear, unweighted only)
        match op:
            case "max":
                result[bin_id] = np.max(neighbor_values)
            case "min":
                result[bin_id] = np.min(neighbor_values)
            case "std":
                result[bin_id] = np.std(neighbor_values)

    return result


def _neighbor_reduce_linear(
    env: GraphConsumer,
    field: NDArray[np.float64],
    *,
    op: Literal["sum", "mean"],
    weights: NDArray[np.float64] | None,
    include_self: bool,
) -> NDArray[np.float64]:
    """Vectorized ``sum``/``mean`` neighbor reduction via a CSR adjacency matrix.

    Parameters
    ----------
    env : Environment
        Spatial environment providing graph connectivity.
    field : NDArray[np.float64], shape (n_bins,)
        Scalar field values at each bin.
    op : {'sum', 'mean'}
        Linear reduction operation.
    weights : NDArray[np.float64], shape (n_bins,) or None
        Optional per-bin weights for weighted sum/mean.
    include_self : bool
        If True, include each bin itself in its neighborhood.

    Returns
    -------
    NDArray[np.float64], shape (n_bins,)
        Reduced field. Bins with empty neighborhoods (degree 0, or degree 0
        plus self when ``include_self`` is False) are NaN. Weighted means whose
        neighbor-weight sum is non-positive are NaN.

    Notes
    -----
    Equivalent to the per-bin loop ``sum``/``mean`` over neighbors, expressed as
    a sparse matrix-vector product ``A @ x`` where ``A`` is the binary adjacency
    matrix (with the diagonal added when ``include_self`` is True).
    """
    import scipy.sparse as sp

    from neurospatial.ops.distance import _bin_adjacency

    n_bins = env.n_bins
    adjacency = _bin_adjacency(
        env.connectivity, weight=None, operation="neighbor_reduce()"
    )
    if include_self:
        adjacency = adjacency + sp.eye(n_bins, format="csr", dtype=np.float64)

    # Unweighted is weighted by ones: ``adjacency @ ones`` is the neighbor
    # count, so ``weight_sum > 0`` marks exactly the non-empty neighborhoods.
    if weights is None:
        weights = np.ones(n_bins, dtype=np.float64)
    numerator = np.asarray(adjacency @ (field * weights), dtype=np.float64)
    if op == "sum":
        numerator[np.asarray(adjacency.sum(axis=1)).ravel() == 0] = np.nan
        return numerator
    # Mean: sum(w * v) / sum(w) over neighbors.
    weight_sum = adjacency @ weights
    result = np.full(n_bins, np.nan, dtype=np.float64)
    valid = weight_sum > 0
    result[valid] = numerator[valid] / weight_sum[valid]
    return result


def graph_convolve(
    env: GraphConsumer,
    field: NDArray[np.float64],
    kernel: Callable[[NDArray[np.float64]], NDArray[np.float64]] | NDArray[np.float64],
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """
    Convolve a spatial field with a custom kernel on the graph.

    Applies spatial convolution using either a callable kernel function
    (distance → weight) or a precomputed kernel matrix. Supports arbitrary
    kernels including box filters, Mexican hat (DoG), and custom designs.

    Parameters
    ----------
    env : Environment
        Spatial environment providing graph connectivity and distances.
    field : array, shape (n_bins,)
        Scalar field values at each bin.
    kernel : callable or array
        Kernel specification, either:
        - **Callable**: Function mapping distances → weights.
          Signature: ``kernel(distances: NDArray) -> NDArray``. It is called
          once with the full geodesic distance matrix, shape
          ``(n_bins, n_bins)`` (``inf`` between disconnected bins), and must
          return weights of the same shape, ``[i, j]`` weighting source bin
          ``j`` at bin ``i``.
        - **Array**: Precomputed kernel matrix of shape (n_bins, n_bins)
          where kernel[i, j] is the weight from bin j to bin i.
    normalize : bool, default=True
        If True, normalize kernel weights to sum to 1 per bin (preserves
        constant fields). If False, use raw kernel weights (useful for
        edge detection kernels like Mexican hat).

    Returns
    -------
    result : array, shape (n_bins,)
        Convolved field values. With ``normalize=True`` each bin is the
        weighted average of the observed (non-NaN) source bins its kernel
        reaches, and NaN where it reaches none. With ``normalize=False`` a
        bin is NaN wherever a nonzero kernel weight falls on a NaN source.

    Raises
    ------
    ValueError
        If field shape doesn't match environment, or if kernel matrix has
        incorrect shape.

    Notes
    -----
    **Kernel Types**:

    - **Box kernel**: Uniform weights within distance threshold
    - **Mexican hat**: Difference of Gaussians for edge detection
    - **Custom kernels**: Any distance-based or precomputed weights

    **Missing values (NaN)**: NaN marks a bin with no observation, which is
    not a value of zero.

    - ``normalize=True`` averages over the observed source bins only
      (Nadaraya-Watson), like ``Environment.smooth(mode="average")``: missing
      bins get no weight, so they neither pull their neighbors toward zero nor
      spread NaN. A missing bin that the kernel connects to observed bins
      receives their weighted average; this estimate does not make the bin
      observed, so keep the original support mask. A bin whose kernel reaches
      no observed bin is NaN.
    - ``normalize=False`` returns the raw weighted sum, which is unknown
      wherever a nonzero weight falls on a missing value: those bins are NaN.
      Weights of exactly zero ignore a missing source, so a precomputed
      kernel matrix can exclude missing bins explicitly. A kernel with full
      support, such as a Gaussian of graph distance, makes the whole
      connected part NaN; fill the field explicitly or normalize instead.

    **Algorithm**:

    1. Compute the all-pairs graph distance matrix (callable kernels)
    2. Apply the kernel function, or take the precomputed weights
    3. Optionally normalize each row over its observed source bins
    4. Compute the weighted sum

    **Performance**: Time complexity O(n_bins²) for callable kernels (computing
    distances), O(n_bins²) for matrix multiplication with precomputed kernels.

    Examples
    --------
    **Box kernel convolution**:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.graph import graph_convolve
    >>> # Create 3x3 grid
    >>> positions = np.array([[i, j] for i in range(3) for j in range(3)])
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> # Spike at center
    >>> field = np.zeros(env.n_bins)
    >>> field[4] = 1.0  # Center bin
    >>> # Box kernel: uniform within radius 1.5
    >>> def box_kernel(distances):
    ...     return np.where(distances <= 1.5, 1.0, 0.0)
    >>> result = graph_convolve(env, field, box_kernel, normalize=True)
    >>> # The corner bins reach 4 bins (themselves, two edges and the center),
    >>> # so the spike is a quarter of their average; the center reaches all 9.
    >>> print(f"Max value: {result.max():.3f}, center: {result[4]:.3f}")
    Max value: 0.250, center: 0.111

    **Balanced center-surround contrast**:

    Normalize the narrow and broad Gaussian averages separately on this finite
    graph. Their difference removes a constant field, including at boundaries;
    unit-height Gaussians alone would give zero at the center and only negative
    weights elsewhere. With missing observations, normalize each average over
    the observed source bins instead.

    >>> def mexican_hat(distances):
    ...     sigma1, sigma2 = 0.5, 1.5
    ...     g1 = np.exp(-(distances**2) / (2 * sigma1**2))
    ...     g2 = np.exp(-(distances**2) / (2 * sigma2**2))
    ...     return g1 / g1.sum(axis=-1, keepdims=True) - g2 / g2.sum(
    ...         axis=-1, keepdims=True
    ...     )
    >>> result = graph_convolve(env, field, mexican_hat, normalize=False)
    >>> bool(result[4] > 0)  # An impulse has positive center contrast.
    True
    >>> bool(np.any(result < 0))  # Surrounding bins have negative contrast.
    True
    >>> constant = graph_convolve(
    ...     env, np.ones(env.n_bins), mexican_hat, normalize=False
    ... )
    >>> np.allclose(constant, 0.0, atol=1e-12)
    True

    See Also
    --------
    Environment.smooth : Boundary-aware finite-volume diffusion smoothing
    neighbor_reduce : Local aggregation over graph neighborhoods

    References
    ----------
    .. [1] Shuman et al. (2013). The emerging field of signal processing
           on graphs. IEEE Signal Processing Magazine, 30(3).
    """
    # Validate field shape
    if field.shape[0] != env.n_bins:
        raise ValueError(
            f"field.shape must match environment bins (got field.shape={field.shape}, "
            f"expected ({env.n_bins},))"
        )

    # Determine kernel type and compute kernel matrix if needed
    if callable(kernel):
        # Callable kernel: compute distance-based weights for each bin
        # Use geodesic_distance_matrix for efficient all-pairs shortest paths
        from neurospatial.ops.distance import geodesic_distance_matrix

        # Compute all pairwise geodesic distances at once: O(n_bins * E log V)
        # This is much faster than O(n_bins^2 * E log V) from individual Dijkstra calls
        dist_matrix = geodesic_distance_matrix(env.connectivity, weight="distance")

        # Apply kernel function to entire distance matrix (vectorized)
        # Most kernel functions (e.g., Gaussian, exponential) are ufunc-compatible
        kernel_matrix = kernel(dist_matrix)
    else:
        # Precomputed kernel matrix
        kernel_matrix = kernel
        if kernel_matrix.shape != (env.n_bins, env.n_bins):
            raise ValueError(
                f"kernel matrix shape must be ({env.n_bins}, {env.n_bins}), "
                f"got {kernel_matrix.shape}"
            )

    # NaN marks a missing source value, not zero activity.
    missing = np.isnan(field)
    field_clean = np.where(missing, 0.0, field)

    if normalize:
        # Nadaraya-Watson: average over the observed source bins only. The
        # missing bins are zero in field_clean, so two matrix-vector products
        # give the observed numerator and weight without an n_bins² copy.
        weight_sums = kernel_matrix @ (~missing).astype(np.float64)
        numerator = kernel_matrix @ field_clean
        supported = weight_sums > 0
        result: NDArray[np.float64] = np.full(env.n_bins, np.nan, dtype=np.float64)
        result[supported] = numerator[supported] / weight_sums[supported]
        return result

    # A raw weighted sum is unknown wherever a nonzero weight meets a missing
    # value; a weight of exactly zero ignores it.
    result = np.asarray(kernel_matrix @ field_clean, dtype=np.float64)
    if missing.any():
        result[(kernel_matrix[:, missing] != 0).any(axis=1)] = np.nan
    return result
