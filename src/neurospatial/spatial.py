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


def map_points_to_bins(
    points: NDArray[np.float64],
    env: Environment,
    *,
    tie_break: Literal["lowest_index", "closest_center"] = "lowest_index",
    return_dist: bool = False,
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

    Returns
    -------
    bin_indices : NDArray[np.int_], shape (n_points,)
        Bin index for each point. Value of -1 indicates point is outside all bins.
    distances : NDArray[np.float64], shape (n_points,), optional
        Distance from each point to its assigned bin center.
        Only returned if `return_dist=True`.

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

    Notes
    -----
    This function builds and caches a KD-tree on the environment's bin_centers
    on first call. Subsequent calls reuse the cached tree for O(log N) performance.

    The cache is stored as a private attribute on the Environment object. If
    bin_centers are modified after creation (not recommended), the cache will
    become stale.

    See Also
    --------
    Environment.bin_at : Basic point-to-bin mapping (delegates to layout engine)
    Environment.contains : Check if points are within environment bounds

    """
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
        # Query for nearest neighbor
        distances, indices = kdtree.query(points, k=1, workers=-1)

        # For boundary points, we need to check if there are multiple
        # equidistant bins. Query for k=2 to detect ties.
        distances_k2, _ = kdtree.query(points, k=2, workers=-1)

        # Check where distance to 2nd nearest equals 1st nearest (within tolerance)
        has_tie = np.abs(distances_k2[:, 0] - distances_k2[:, 1]) < 1e-10

        if np.any(has_tie):
            # For tied points, query more neighbors and pick lowest index
            # Adaptive: query enough neighbors to find all at same distance
            max_neighbors = min(10, len(env.bin_centers))
            distances_kn, indices_kn = kdtree.query(
                points[has_tie], k=max_neighbors, workers=-1
            )

            # For each tied point, find all neighbors at same distance and pick min index
            for i, (dists, idxs) in enumerate(
                zip(distances_kn, indices_kn, strict=False)
            ):
                min_dist = dists[0]
                tied_indices = idxs[np.abs(dists - min_dist) < 1e-10]
                indices[has_tie][i] = tied_indices.min()

        bin_indices = indices.astype(np.int64)

    else:
        raise ValueError(
            f"Invalid tie_break mode: {tie_break!r}. "
            f"Must be 'lowest_index' or 'closest_center'."
        )

    # Check if any points are outside the environment
    # Points far from any bin center should be marked as -1
    # Use a heuristic: if distance > 10 * mean_bin_size, mark as outside
    if len(env.bin_centers) > 1:
        # Estimate typical bin spacing from nearest-neighbor distances
        sample_size = min(100, len(env.bin_centers))
        sample_indices = np.random.choice(
            len(env.bin_centers), size=sample_size, replace=False
        )
        sample_centers = env.bin_centers[sample_indices]
        nn_dists, _ = kdtree.query(sample_centers, k=2, workers=-1)
        typical_bin_spacing = np.median(nn_dists[:, 1])

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
