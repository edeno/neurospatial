"""Spatial signal processing primitives for graph-based operations.

This module provides graph-based operations for spatial signal processing
including neighborhood aggregation and convolution on irregular graphs.

Functions
---------
neighbor_reduce
    Aggregate field values over spatial neighborhoods.
convolve
    Convolve a spatial field with a custom kernel on the graph.

Examples
--------
>>> from neurospatial.ops.graph import neighbor_reduce, convolve

Import via ops package:

>>> from neurospatial.ops import neighbor_reduce, convolve

Notes
-----
Moved from neurospatial.primitives in package reorganization.
"""

from __future__ import annotations

__all__ = ["convolve", "neighbor_reduce"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


def neighbor_reduce(
    env: Environment,
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
        return NaN.

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
      ``neighbor_reduce(field, env, op='mean')``.
    - **Smoothness**: Measure how much field values differ from local neighborhood.
    - **Local statistics**: Compute local variability (op='std') or extrema
      (op='max'/'min').

    **Algorithm**:

    1. For each bin, retrieve neighbors from ``env.connectivity`` graph
    2. Optionally include self in neighborhood
    3. Apply reduction operation (sum, mean, max, min, std)
    4. Handle isolated nodes (no neighbors) → return NaN

    **Performance**: Optimized for sparse graphs using NetworkX neighbor iteration.
    Time complexity O(n_bins × avg_degree).

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
    >>> # Center bin (4) has neighbors [1, 3, 5, 7]
    >>> # Mean = (1 + 3 + 5 + 7) / 4 = 4.0
    >>> print(f"Center neighbor mean: {neighbor_mean[4]:.1f}")  # doctest: +SKIP
    Center neighbor mean: 4.0

    See Also
    --------
    Environment.smooth : Gaussian smoothing using distance-based kernels
    convolve : General convolution with custom kernels

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

    # Initialize result array
    result = np.full(env.n_bins, np.nan, dtype=np.float64)

    # Iterate over all bins
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

        # Apply reduction operation
        if weights is None:
            # Unweighted operations
            match op:
                case "sum":
                    result[bin_id] = np.sum(neighbor_values)
                case "mean":
                    result[bin_id] = np.mean(neighbor_values)
                case "max":
                    result[bin_id] = np.max(neighbor_values)
                case "min":
                    result[bin_id] = np.min(neighbor_values)
                case "std":
                    result[bin_id] = np.std(neighbor_values)
        else:
            # Weighted operations (only sum and mean)
            neighbor_weights = weights[neighbors]
            match op:
                case "sum":
                    result[bin_id] = np.sum(neighbor_values * neighbor_weights)
                case "mean":
                    # Weighted mean: sum(w * v) / sum(w)
                    weight_sum = np.sum(neighbor_weights)
                    if weight_sum > 0:
                        result[bin_id] = (
                            np.sum(neighbor_values * neighbor_weights) / weight_sum
                        )
                    else:
                        result[bin_id] = np.nan

    return result


def convolve(
    env: Environment,
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
          Signature: ``kernel(distances: NDArray) -> NDArray``
          where distances has shape (n_bins,).
        - **Array**: Precomputed kernel matrix of shape (n_bins, n_bins)
          where kernel[i, j] is the weight from bin j to bin i.
    normalize : bool, default=True
        If True, normalize kernel weights to sum to 1 per bin (preserves
        constant fields). If False, use raw kernel weights (useful for
        edge detection kernels like Mexican hat).

    Returns
    -------
    result : array, shape (n_bins,)
        Convolved field values. NaN values in input field are handled by
        skipping them in the convolution and renormalizing weights.

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

    **NaN Handling**: NaN values in the field are excluded from the convolution
    by treating them as zero contribution and renormalizing the weights. This
    prevents NaN propagation while maintaining local normalization.

    **Algorithm**:

    1. For each bin, compute pairwise distances to all other bins
    2. Apply kernel function or lookup precomputed weights
    3. Optionally normalize weights to sum to 1
    4. Compute weighted sum, excluding NaN values in field

    **Performance**: Time complexity O(n_bins²) for callable kernels (computing
    distances), O(n_bins²) for matrix multiplication with precomputed kernels.

    Examples
    --------
    **Box kernel convolution**:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.graph import convolve
    >>> # Create 3x3 grid
    >>> positions = np.array([[i, j] for i in range(3) for j in range(3)])
    >>> env = Environment.from_samples(positions, bin_size=1.0)
    >>> # Spike at center
    >>> field = np.zeros(env.n_bins)
    >>> field[4] = 1.0  # Center bin
    >>> # Box kernel: uniform within radius 1.5
    >>> def box_kernel(distances):
    ...     return np.where(distances <= 1.5, 1.0, 0.0)
    >>> result = convolve(env, field, box_kernel, normalize=True)
    >>> print(f"Max value: {result.max():.3f}")  # doctest: +SKIP
    Max value: 0.111

    **Mexican hat for edge detection**:

    >>> def mexican_hat(distances):
    ...     sigma1, sigma2 = 0.5, 1.5
    ...     g1 = np.exp(-(distances**2) / (2 * sigma1**2))
    ...     g2 = np.exp(-(distances**2) / (2 * sigma2**2))
    ...     return g1 - g2
    >>> result = convolve(env, field, mexican_hat, normalize=False)
    >>> # Center positive, surrounding negative (edge enhancement)

    See Also
    --------
    Environment.smooth : Gaussian smoothing with distance-based kernels
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
        dist_matrix = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

        # Apply kernel function to each row of distances
        kernel_matrix = np.zeros((env.n_bins, env.n_bins), dtype=np.float64)
        for i in range(env.n_bins):
            kernel_matrix[i, :] = kernel(dist_matrix[i, :])
    else:
        # Precomputed kernel matrix
        kernel_matrix = kernel
        if kernel_matrix.shape != (env.n_bins, env.n_bins):
            raise ValueError(
                f"kernel matrix shape must be ({env.n_bins}, {env.n_bins}), "
                f"got {kernel_matrix.shape}"
            )

    # Handle NaN values in field
    valid_mask = ~np.isnan(field)
    field_clean = np.where(valid_mask, field, 0.0)

    # Normalize kernel weights per bin if requested
    if normalize:
        # For each bin (row), normalize weights to sum to 1
        # But only over valid (non-NaN) source bins
        kernel_normalized = np.zeros_like(kernel_matrix)
        for i in range(env.n_bins):
            # Get weights for bin i (row i)
            weights = kernel_matrix[i, :]
            # Only consider weights where field is valid
            valid_weights = weights * valid_mask
            weight_sum = valid_weights.sum()
            if weight_sum > 0:
                kernel_normalized[i, :] = valid_weights / weight_sum
            else:
                # No valid neighbors - result will be 0
                kernel_normalized[i, :] = 0.0
        kernel_matrix = kernel_normalized

    # Perform convolution: result[i] = sum_j kernel[i, j] * field[j]
    result: NDArray[np.float64] = kernel_matrix @ field_clean

    # For normalized case, bins with no valid neighbors get NaN
    # For unnormalized case, all bins should have values (kernel always applied)
    if normalize:
        # Check if any weight was applied
        has_valid_neighbor: NDArray[np.bool_] = kernel_matrix.sum(axis=1) > 0
        result = np.where(has_valid_neighbor, result, np.nan)

    return result
