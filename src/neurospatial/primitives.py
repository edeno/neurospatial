"""Spatial signal processing primitives for graph-based operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


def neighbor_reduce(
    field: NDArray[np.float64],
    env: Environment | EnvironmentProtocol,
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
    field : array, shape (n_bins,)
        Scalar field values at each bin.
    env : Environment
        Spatial environment providing graph connectivity.
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
    >>> from neurospatial.primitives import neighbor_reduce
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
    >>> neighbor_mean = neighbor_reduce(field, env, op="mean")
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
            if op == "sum":
                result[bin_id] = np.sum(neighbor_values)
            elif op == "mean":
                result[bin_id] = np.mean(neighbor_values)
            elif op == "max":
                result[bin_id] = np.max(neighbor_values)
            elif op == "min":
                result[bin_id] = np.min(neighbor_values)
            elif op == "std":
                result[bin_id] = np.std(neighbor_values)
        else:
            # Weighted operations (only sum and mean)
            neighbor_weights = weights[neighbors]
            if op == "sum":
                result[bin_id] = np.sum(neighbor_values * neighbor_weights)
            elif op == "mean":
                # Weighted mean: sum(w * v) / sum(w)
                weight_sum = np.sum(neighbor_weights)
                if weight_sum > 0:
                    result[bin_id] = (
                        np.sum(neighbor_values * neighbor_weights) / weight_sum
                    )
                else:
                    result[bin_id] = np.nan

    return result
