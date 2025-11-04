"""spatial.py - Spatial query utilities for neurospatial
========================================================

This module provides high-performance spatial query utilities including:
- Batch mapping of points to bins with KD-tree caching
- Deterministic tie-breaking on bin boundaries
- Distance calculations

These are core primitives used throughout neurospatial and by downstream packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def _estimate_typical_bin_spacing(
    kdtree: cKDTree, bin_centers: NDArray[np.float64]
) -> float:
    """Estimate typical bin spacing using deterministic quantile-based sampling.

    Parameters
    ----------
    kdtree : cKDTree
        KD-tree built from bin centers.
    bin_centers : NDArray[np.float64], shape (n_bins, n_dims)
        Coordinates of all bin centers.

    Returns
    -------
    typical_spacing : float
        Median nearest-neighbor distance, estimated from sample of bins.
        Returns np.inf if there is only one bin.

    Notes
    -----
    Uses deterministic quantile-based sampling for reproducibility, selecting
    up to 100 evenly-spaced bin indices.
    """
    if len(bin_centers) <= 1:
        return np.inf

    sample_size = min(100, len(bin_centers))
    # Deterministic quantile-based sampling (not random)
    sample_indices = np.linspace(0, len(bin_centers) - 1, sample_size, dtype=int)
    sample_centers = bin_centers[sample_indices]
    nn_dists, _ = kdtree.query(sample_centers, k=2, workers=-1)
    return float(np.median(nn_dists[:, 1]))


def map_points_to_bins(
    points: NDArray[np.float64],
    env: Environment,
    *,
    tie_break: Literal["lowest_index", "closest_center"] = "lowest_index",
    return_dist: bool = False,
    max_distance: float | None = None,
    max_distance_factor: float | None = None,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Map points to bin indices with deterministic tie-breaking.

    This function provides fast, batch mapping of continuous coordinates to
    discrete bin indices using KD-tree queries. It handles edge cases like
    boundary points consistently through configurable tie-breaking rules.

    Internally caches a KD-tree on first call for O(log N) lookups.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        Continuous coordinates to map to bins.
    env : Environment
        Environment containing the bin discretization.
    tie_break : {"lowest_index", "closest_center"}, default="lowest_index"
        Strategy for resolving ties when a point is equidistant from multiple
        bin centers:

        - "lowest_index": Choose the bin with smallest index (deterministic)
        - "closest_center": Return the actual closest (may be non-deterministic
          for exact ties, but faster)

    return_dist : bool, default=False
        If True, also return the distance from each point to its assigned bin center.
    max_distance : float, optional
        Absolute distance threshold in physical units. Points farther than this
        from the nearest bin center are marked as outside (-1). Cannot be used
        with max_distance_factor.
    max_distance_factor : float, optional
        Relative distance threshold as a multiple of typical bin spacing. Points
        farther than (max_distance_factor × typical_bin_spacing) from the nearest
        bin center are marked as outside (-1). Cannot be used with max_distance.

    Returns
    -------
    bin_indices : NDArray[np.int_], shape (n_points,)
        Bin index for each point. Value of -1 indicates point is outside all bins.
    distances : NDArray[np.float64], shape (n_points,), optional
        Distance from each point to its assigned bin center.
        Only returned if `return_dist=True`.

    Raises
    ------
    ValueError
        If both max_distance and max_distance_factor are specified, or if either
        is negative, or if invalid tie_break mode.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.spatial import map_points_to_bins
    >>> data = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> points = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 50.0]])
    >>> bins = map_points_to_bins(points, env)
    >>> bins
    array([ 42,  89,  -1])

    >>> # Get distances too
    >>> bins, dists = map_points_to_bins(points, env, return_dist=True)
    >>> dists
    array([0.23, 0.45, inf])

    >>> # Filter outliers with absolute threshold
    >>> bins = map_points_to_bins(points, env, max_distance=15.0)

    >>> # Filter outliers with relative threshold (adapts to bin size)
    >>> bins = map_points_to_bins(points, env, max_distance_factor=1.5)

    Notes
    -----
    This function builds and caches a KD-tree on the environment's bin_centers
    on first call. Subsequent calls reuse the cached tree for O(log N) performance.

    The cache is stored as a private attribute on the Environment object.

    **IMPORTANT**: Environment objects are designed to be immutable after creation.
    Modifying bin_centers or other spatial attributes after creation will cause
    the cache to become stale and produce incorrect results. If you need a modified
    environment, create a new Environment instance instead.

    The typical bin spacing is estimated using deterministic quantile-based
    sampling for reproducibility.

    See Also
    --------
    Environment.bin_at : Basic point-to-bin mapping (delegates to layout engine)
    Environment.contains : Check if points are within environment bounds

    """
    # Validate parameters
    if max_distance is not None and max_distance_factor is not None:
        raise ValueError(
            "Cannot specify both max_distance and max_distance_factor. "
            "Choose one distance threshold method."
        )

    if max_distance is not None and max_distance < 0:
        raise ValueError(f"max_distance must be non-negative, got {max_distance}")

    if max_distance_factor is not None and max_distance_factor <= 0:
        raise ValueError(
            f"max_distance_factor must be positive, got {max_distance_factor}"
        )
    # Build or retrieve cached KD-tree
    if not hasattr(env, "_kdtree_cache") or env._kdtree_cache is None:
        env._kdtree_cache = cKDTree(env.bin_centers)

    kdtree: cKDTree = env._kdtree_cache

    # Query KD-tree
    if tie_break == "closest_center":
        # Fast path: just use nearest neighbor
        distances, indices = kdtree.query(points, k=1, workers=-1)
        bin_indices: NDArray[np.int64] = indices.astype(np.int64)

    elif tie_break == "lowest_index":
        # Deterministic path: find all ties and pick lowest index
        # Query for top 2 neighbors to detect ties (avoid redundant k=1 query)
        distances_k2, indices_k2 = kdtree.query(points, k=2, workers=-1)
        distances = distances_k2[:, 0]
        indices = indices_k2[:, 0].copy()  # Copy to allow modification

        # Check where distance to 2nd nearest equals 1st nearest (within tolerance)
        has_tie = np.abs(distances_k2[:, 0] - distances_k2[:, 1]) < 1e-10

        # For any ties detected in k=2 query, pick the lowest index immediately
        if np.any(has_tie):
            # For simple 2-way ties, just pick minimum of the two
            simple_ties = has_tie.copy()

            # First handle simple ties where only 2 bins are equidistant
            for idx in np.where(simple_ties)[0]:
                # Get the two nearest indices from k=2 query
                two_nearest = indices_k2[idx, :2]
                indices[idx] = two_nearest.min()

            # Now check if we need to query more neighbors for complex ties
            # Re-query only points that might have more than 2 tied neighbors
            max_neighbors = min(10, len(env.bin_centers))
            if max_neighbors > 2:
                distances_kn, indices_kn = kdtree.query(
                    points[has_tie], k=max_neighbors, workers=-1
                )

                # For each tied point, find ALL neighbors at same distance
                for i, (dists, idxs) in enumerate(
                    zip(distances_kn, indices_kn, strict=False)
                ):
                    min_dist = dists[0]
                    # Find all indices within tolerance of minimum distance
                    tied_indices = idxs[np.abs(dists - min_dist) < 1e-10]
                    # Update only if we found more than the 2 we already considered
                    if len(tied_indices) > 2:
                        indices[has_tie][i] = tied_indices.min()

        bin_indices = indices.astype(np.int64)

    else:
        raise ValueError(
            f"Invalid tie_break mode: {tie_break!r}. "
            f"Must be 'lowest_index' or 'closest_center'."
        )

    # Check if any points are outside the environment based on distance threshold
    if max_distance is not None or max_distance_factor is not None:
        # Explicit distance threshold provided
        if max_distance is not None:
            threshold = max_distance
        else:
            # Estimate typical bin spacing and apply factor
            # assert: max_distance_factor is not None (checked above)
            assert max_distance_factor is not None  # for mypy
            typical_bin_spacing = _estimate_typical_bin_spacing(kdtree, env.bin_centers)
            threshold = max_distance_factor * typical_bin_spacing

        # Mark points beyond threshold as outside
        bin_indices[distances > threshold] = -1
    elif len(env.bin_centers) > 1:
        # Backward compatibility: use old heuristic (10× typical spacing)
        typical_bin_spacing = _estimate_typical_bin_spacing(kdtree, env.bin_centers)

        # Mark points that are suspiciously far as outside
        threshold = 10 * typical_bin_spacing
        bin_indices[distances > threshold] = -1

    if return_dist:
        # Set distance to inf for points outside environment
        distances_out = distances.copy()
        distances_out[bin_indices == -1] = np.inf
        return (bin_indices, distances_out)

    return bin_indices


def clear_kdtree_cache(env: Environment) -> None:
    """Clear the cached KD-tree for an environment.

    This is useful if bin_centers have been modified (not recommended) or
    to free memory.

    Parameters
    ----------
    env : Environment
        Environment whose KD-tree cache should be cleared.

    Examples
    --------
    >>> from neurospatial.spatial import clear_kdtree_cache
    >>> clear_kdtree_cache(env)

    """
    if hasattr(env, "_kdtree_cache"):
        env._kdtree_cache = None
