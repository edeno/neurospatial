"""binning.py - Point-to-bin mapping utilities for neurospatial.
================================================================

This module provides high-performance spatial query utilities including:
- Batch mapping of points to bins with KD-tree caching
- Deterministic tie-breaking on bin boundaries
- Region rasterization to bin masks
- Field resampling between environments

These are core primitives used throughout neurospatial and by downstream packages.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


class TieBreakStrategy(Enum):
    """Strategy for resolving ties when a point is equidistant from multiple bins.

    Attributes
    ----------
    LOWEST_INDEX : str
        Choose the bin with the smallest index among equidistant candidates.
        This ensures deterministic, reproducible results even when points lie
        exactly on bin boundaries. Recommended for scientific reproducibility.

    CLOSEST_CENTER : str
        Choose the bin whose center is geometrically closest to the point.
        This breaks numerical ties but may still be non-deterministic for
        exact ties. Faster than LOWEST_INDEX as it skips secondary checks.

    Examples
    --------
    >>> from neurospatial.ops.binning import TieBreakStrategy, map_points_to_bins
    >>> import numpy as np
    >>> # Use enum for autocomplete and type safety
    >>> result = map_points_to_bins(
    ...     points, env, tie_break=TieBreakStrategy.CLOSEST_CENTER
    ... )  # doctest: +SKIP
    """

    LOWEST_INDEX = "lowest_index"
    CLOSEST_CENTER = "closest_center"


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
    tie_break: TieBreakStrategy
    | Literal["lowest_index", "closest_center"] = TieBreakStrategy.LOWEST_INDEX,
    return_dist: bool = False,
    max_distance: float | None = None,
    max_distance_factor: float | None = None,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Map points to bins using nearest-neighbor with KDTree caching.

    This function provides fast, batch mapping of continuous coordinates to
    discrete bin indices using KD-tree nearest-neighbor queries. Unlike
    `Environment.bin_at()` which uses geometric containment, this finds the
    bin whose center is closest to each point.

    Internally caches a KD-tree on first call for O(log N) lookups. Best for
    large batches, trajectory processing, and when approximate assignment is
    acceptable.

    Parameters
    ----------
    points : NDArray[np.float64], shape (n_points, n_dims)
        Continuous coordinates to map to bins.
    env : Environment
        Environment containing the bin discretization.
    tie_break : TieBreakStrategy or {"lowest_index", "closest_center"}, default=TieBreakStrategy.LOWEST_INDEX
        Strategy for resolving ties when a point is equidistant from multiple
        bin centers. Can pass either a TieBreakStrategy enum member or a string.

        - TieBreakStrategy.LOWEST_INDEX or "lowest_index": Choose the bin with
          smallest index (deterministic, recommended for reproducibility)
        - TieBreakStrategy.CLOSEST_CENTER or "closest_center": Return the actual
          closest (may be non-deterministic for exact ties, but faster). On an
          exact tie between equidistant centers, the bin returned is the one
          reported by the underlying KDTree query; that ordering is
          implementation-defined and may differ between platforms.

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
        Bin index for each point. The "outside" sentinel ``-1`` is
        returned only when ``max_distance`` (or ``max_distance_factor``)
        rejects the nearest-neighbor distance, or when the implicit
        ``10 × typical_bin_spacing`` heuristic does so. A point a
        couple of centimeters outside the active mask of a 2 cm-bin
        env will silently bind to the nearest edge bin under the
        default heuristic. For trajectory occupancy / bin-sequence
        purposes use :meth:`Environment.bin_at` (geometric containment,
        always returns ``-1`` for outside points). Use this function
        for nearest-neighbor *interpolation* queries where assigning
        points to the closest in-env bin is the goal.
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
    >>> from neurospatial.ops.binning import map_points_to_bins
    >>> data = np.random.randn(1000, 2) * 10  # doctest: +SKIP
    >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
    >>> points = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 50.0]])  # doctest: +SKIP
    >>> bins = map_points_to_bins(points, env)  # doctest: +SKIP
    >>> bins  # doctest: +SKIP
    array([ 42,  89,  -1])

    >>> # Get distances too
    >>> bins, dists = map_points_to_bins(
    ...     points, env, return_dist=True
    ... )  # doctest: +SKIP
    >>> dists  # doctest: +SKIP
    array([0.23, 0.45, inf])

    >>> # Filter outliers with absolute threshold
    >>> bins = map_points_to_bins(points, env, max_distance=15.0)  # doctest: +SKIP

    >>> # Filter outliers with relative threshold (adapts to bin size)
    >>> bins = map_points_to_bins(
    ...     points, env, max_distance_factor=1.5
    ... )  # doctest: +SKIP

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
    Environment.bin_at : Geometric containment mapping (exact bin assignment)
    Environment.contains : Check if points are within environment bounds

    Notes
    -----
    **Geometric Containment vs Nearest-Neighbor:**

    - `Environment.bin_at()`: Uses layout-specific geometric containment to
      determine which bin actually contains the point. This is exact but may
      be slower for large batches and doesn't support advanced features like
      tie-breaking or distance thresholds.

    - `map_points_to_bins()` (this function): Uses KDTree to find the bin whose
      center is closest to the point. Fast with caching, supports tie-breaking,
      and allows distance thresholds. Best for large batches and trajectory
      processing.

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

    # Convert string to enum if needed (backwards compatibility)
    if isinstance(tie_break, str):
        try:
            tie_break = TieBreakStrategy(tie_break)
        except ValueError:
            raise ValueError(
                f"Invalid tie_break value: '{tie_break}'. "
                f"Must be one of: {[e.value for e in TieBreakStrategy]}"
            ) from None

    # Build or retrieve cached KD-tree
    if not hasattr(env, "_kdtree_cache") or env._kdtree_cache is None:
        env._kdtree_cache = cKDTree(env.bin_centers)

    kdtree: cKDTree = env._kdtree_cache

    # Query KD-tree
    if tie_break == TieBreakStrategy.CLOSEST_CENTER:
        # Fast path: just use nearest neighbor
        distances, indices = kdtree.query(points, k=1, workers=-1)
        bin_indices: NDArray[np.int64] = indices.astype(np.int64)

    elif tie_break == TieBreakStrategy.LOWEST_INDEX:
        # Deterministic path: find all ties and pick lowest index.
        # We first query only the k=10 nearest candidates as a fast approximation:
        # the lowest-index winner almost always lies among the nearest few. If a
        # tie is *detected* among those candidates (see ``tied_rows`` below), we
        # fall back to an exact all-center distance computation for just those
        # rows, so >10 equidistant bins are still resolved correctly. The k=10
        # cap only bounds the cheap first pass; it is not an upper bound on the
        # number of ties that can be resolved exactly.
        max_neighbors = min(10, len(env.bin_centers))
        distances_kn, indices_kn = kdtree.query(points, k=max_neighbors, workers=-1)

        # Handle k=1 case (single bin environment)
        if max_neighbors == 1:
            distances = np.atleast_1d(distances_kn)
            bin_indices = np.atleast_1d(indices_kn).astype(np.int64)
        else:
            distances = distances_kn[:, 0]
            # Vectorized tie resolution: for each point, find minimum index among ties
            # Create mask of tied neighbors (within tolerance of minimum distance)
            min_dists = distances_kn[:, 0:1]  # Keep dims for broadcasting
            is_tied = np.abs(distances_kn - min_dists) < 1e-10  # (n_points, k)

            # For each point, find minimum index among tied neighbors
            # Set non-tied indices to large value so they're ignored in min
            tied_indices = np.where(is_tied, indices_kn, np.iinfo(np.intp).max)
            bin_indices = np.min(tied_indices, axis=1).astype(np.int64)

            # cKDTree only returns max_neighbors candidates above. For exact ties
            # involving more candidates, compute all-center distances only for
            # rows where a tie was detected.
            tied_rows = np.count_nonzero(is_tied, axis=1) > 1
            if np.any(tied_rows):
                all_distances = np.linalg.norm(
                    env.bin_centers[None, :, :] - points[tied_rows, None, :],
                    axis=2,
                )
                min_distances = np.min(all_distances, axis=1, keepdims=True)
                all_tied = np.abs(all_distances - min_distances) < 1e-10
                bin_indices[tied_rows] = np.argmax(all_tied, axis=1).astype(np.int64)

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
    env : EnvironmentProtocol
        Environment whose KD-tree cache should be cleared.

    See Also
    --------
    Environment.clear_cache : Clear all caches (KDTree + cached properties + kernels)

    Notes
    -----
    For most use cases, prefer `env.clear_cache()` which clears all caches
    including the KDTree cache and cached properties. This function is provided
    for backward compatibility and fine-grained control.

    Examples
    --------
    >>> from neurospatial.ops.binning import clear_kdtree_cache
    >>> clear_kdtree_cache(env)  # doctest: +SKIP

    """
    if hasattr(env, "_kdtree_cache"):
        env._kdtree_cache = None


def regions_to_mask(
    env: Environment,
    regions: str | list[str] | object,  # Will be Region | Regions after import
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Rasterize continuous regions onto discrete environment bins.

    This function converts continuous polygon regions into a discrete boolean
    mask over environment bins. It is the dual operation of mask_to_polygon,
    completing the round-trip between continuous and discrete representations.

    Parameters
    ----------
    env : EnvironmentProtocol
        Environment with bin centers defining the discrete grid.
    regions : str, list[str], Region, or Regions
        Region(s) to rasterize. Can be:
        - A single region name (str) from env.regions
        - A list of region names (list[str]) from env.regions
        - A single Region object
        - A Regions collection
    include_boundary : bool, default=True
        Whether to include bins whose centers lie on region boundaries:
        - True: Include boundary bins (uses shapely.covers)
        - False: Exclude boundary bins (uses shapely.contains)

    Returns
    -------
    mask : NDArray[np.bool_], shape (n_bins,)
        Boolean mask where True indicates bin center is inside region(s).
        For multiple regions, returns the union (logical OR).
        Point regions always return all False (points have no area).

    Raises
    ------
    KeyError
        If a region name is not found in env.regions.
    TypeError
        If regions parameter has invalid type or include_boundary is not bool.

    Notes
    -----
    **Region Types**:
    - Polygon regions: Rasterized using Shapely containment checks
    - Point regions: Always return all False (points have no area)

    **Multiple Regions**:

    When multiple regions are provided, the result is their **union** (logical OR).
    Bins are included if they fall inside ANY of the regions. Overlapping
    regions are counted only once.

    For intersection or difference, use boolean operations on separate masks::

        mask_a = regions_to_mask(env, "region_a")
        mask_b = regions_to_mask(env, "region_b")
        intersection = mask_a & mask_b
        difference = mask_a & ~mask_b

    **Boundary Semantics**:

    - include_boundary=True: Uses shapely.covers (includes boundary)
    - include_boundary=False: Uses shapely.contains (excludes boundary)

    **Duality with Continuous Representations**:

    This function is the approximate inverse of mask_to_region in regions.io.
    The round-trip mask → polygon → mask preserves spatial structure but may
    differ in fine details due to contour approximation.

    This function completes the "dual operations" framework for spatial discretization,
    pairing with apply_kernel (forward/adjoint diffusion) and resample_field
    (layout-to-layout resampling).

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.binning import regions_to_mask
    >>> from neurospatial.regions import Regions

    Create environment and regions:

    >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> regions = Regions()
    >>> _ = regions.add("center", polygon=box(3, 3, 7, 7))

    Rasterize onto bins:

    >>> mask = regions_to_mask(env, regions)
    >>> mask.shape
    (36,)
    >>> bool(np.any(mask))  # Some bins inside
    True

    Use region names from environment:

    >>> _ = env.regions.add("test", polygon=box(3, 3, 7, 7))
    >>> mask = regions_to_mask(env, "test")
    >>> mask.shape
    (36,)

    Multiple regions (union):

    >>> _ = env.regions.add("left", polygon=box(0, 0, 4, 10))
    >>> _ = env.regions.add("right", polygon=box(6, 0, 10, 10))
    >>> mask = regions_to_mask(env, ["left", "right"])
    >>> bool(np.any(mask))
    True

    See Also
    --------
    neurospatial.regions.io.mask_to_region : Convert mask to polygon (inverse operation)
    Environment.region_membership : 2D membership array for all regions
    apply_kernel : Apply forward/adjoint diffusion operations on fields
    resample_field : Resample fields between different discretizations

    """
    # Import here to avoid circular dependency
    from neurospatial.regions import Region, Regions

    # Input validation for include_boundary
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}"
        )

    # Normalize regions parameter to Regions object
    regions_obj: Regions
    if isinstance(regions, str):
        # Single region name from env.regions
        if regions not in env.regions:
            available = list(env.regions.keys())
            raise KeyError(
                f"Region '{regions}' not found in environment. "
                f"Available regions: {available}. "
                f"Use env.regions.add('{regions}', polygon=...) to create this region first."
            )
        regions_obj = Regions([env.regions[regions]])
    elif isinstance(regions, list):
        # List of region names from env.regions
        region_list = []
        for name in regions:
            if name not in env.regions:
                available = list(env.regions.keys())
                raise KeyError(
                    f"Region '{name}' not found in environment. "
                    f"Available regions: {available}. "
                    f"Use env.regions.add('{name}', polygon=...) to create this region first."
                )
            region_list.append(env.regions[name])
        regions_obj = Regions(region_list)
    elif isinstance(regions, Region):
        # Single Region object
        regions_obj = Regions([regions])
    elif isinstance(regions, Regions):
        # Regions collection
        regions_obj = regions
    else:
        raise TypeError(
            f"regions must be str, list[str], Region, or Regions, "
            f"got {type(regions).__name__}"
        )

    # Handle empty regions
    if len(regions_obj) == 0:
        return np.zeros(env.n_bins, dtype=bool)

    # Get bin centers
    bin_centers = env.bin_centers

    # Initialize mask (union across all regions)
    mask = np.zeros(env.n_bins, dtype=bool)

    # Import shapely functions for vectorized operations
    from shapely import contains, covers
    from shapely import points as shapely_points

    # Check if we have any polygon regions (to avoid unnecessary computation)
    has_polygon_regions = any(r.kind == "polygon" for r in regions_obj.values())

    # Pre-compute shapely Points array once for all polygon regions
    points = None
    if has_polygon_regions:
        # Currently only supports 2D
        if bin_centers.shape[1] != 2:
            raise NotImplementedError(
                f"regions_to_mask currently only supports 2D environments "
                f"for polygon regions. Environment has {bin_centers.shape[1]} dimensions."
            )
        # Create shapely Points array from bin centers (computed once)
        points = shapely_points(bin_centers[:, 0], bin_centers[:, 1])

    # Iterate over regions and update mask
    for region in regions_obj.values():
        # Point regions have no area, skip them with warning
        if region.kind == "point":
            import warnings

            warnings.warn(
                f"Region '{region.name}' is a point region and has no area. "
                f"Point regions always return empty masks (no bins selected). "
                f"Consider using env.regions.buffer('{region.name}', distance=...) "
                f"to create a small polygon around the point if you want to select nearby bins.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Handle polygon regions (points already computed above)
        if region.kind == "polygon":
            # Vectorized containment check using pre-computed points
            if include_boundary:
                # covers: True if point is inside or on boundary
                region_mask = covers(region.data, points)
            else:
                # contains: True only if strictly inside
                region_mask = contains(region.data, points)

            # Union with existing mask
            mask |= region_mask

    return mask


def resample_field(
    field: NDArray[np.float64],
    src_env: Environment,
    dst_env: Environment,
    *,
    method: Literal["nearest", "diffuse"] = "nearest",
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """
    Resample field from source to destination environment.

    This function enables moving fields between different discretizations,
    supporting multi-resolution analysis and cross-session alignment.

    Parameters
    ----------
    field : NDArray[np.float64], shape (src_env.n_bins,)
        Field defined on source environment
    src_env : Environment
        Source environment where field is defined
    dst_env : Environment
        Destination environment to resample onto
    method : {"nearest", "diffuse"}, optional
        Resampling method. Default is "nearest".

        - "nearest": KD-tree nearest-neighbor lookup (fast, preserves values)
        - "diffuse": nearest-neighbor pullback followed by a boundary-aware,
          valid-bin-normalized diffusion **average** of the pulled-back field.
          For **intensive** fields only (rate maps / probability densities); it
          does not conserve mass, so do not use it for spike counts, occupancy,
          or any extensive/mass quantity.
    bandwidth : float, optional
        Smoothing bandwidth for diffuse method (in spatial units).
        Required when method="diffuse", ignored for method="nearest".

    Returns
    -------
    resampled : NDArray[np.float64], shape (dst_env.n_bins,)
        Field resampled onto destination environment bins. Destination bins
        whose centers fall outside the source environment (no source bin within
        range) are set to ``np.nan``.

    Raises
    ------
    ValueError
        If field size doesn't match source environment,
        if method is invalid,
        if bandwidth is missing/invalid for diffuse method,
        or if environments have different dimensions.

    See Also
    --------
    map_points_to_bins : Low-level point-to-bin mapping
    apply_kernel : Apply diffusion kernels to fields
    regions_to_mask : Convert regions to bin masks (spatial discretization)

    Notes
    -----
    **Choosing a Method**:

    **Use "nearest" when:**

    - You want exact field values preserved (no smoothing)
    - You're resampling between similar resolutions
    - You need maximum speed
    - Example: Aligning same-session data with slightly different bin sizes

    **Use "diffuse" when:**

    - You're changing resolution significantly (>2× bin size ratio)
    - You want to reduce aliasing artifacts
    - You need spatially smooth results
    - Example: Comparing place fields across different grid resolutions

    **Rule of thumb**: Use "nearest" by default. Use "diffuse" when resampling
    across large resolution changes or when visual smoothness matters.

    **Method Details**:

    **Nearest method**: Uses KD-tree to find nearest source bin for each
    destination bin center. Fast and preserves exact field values from source.
    No interpolation or smoothing is applied.

    **Diffuse method**: Applies nearest-neighbor mapping, then a masked
    (valid-bin-normalized) diffusion average — Nadaraya-Watson with the
    row-stochastic heat operator ``H`` on ``dst_env``: ``smooth(value * valid) /
    smooth(valid)``. Only bins covered by the source contribute weight, so
    covered bins adjacent to an uncovered region are **not** biased toward zero,
    and a ``NaN`` in the source field is interpolated from valid neighbours
    (within the bandwidth) rather than propagating. Bins outside the source, and
    bins with no valid neighbour within the bandwidth, are ``NaN``.

    **Dimension requirements**: Both environments must have the same number
    of dimensions (e.g., both 2D).

    **Mass and Values**:

    - **Nearest method**: Preserves exact field values from source bins.
      Total integrated mass (sum(field * bin_sizes)) changes if bin sizes differ
      between source and destination—this is expected and correct.

    - **Diffuse method**: Averages an **intensive** field (rate map /
      probability density) after resampling — it is volume-unbiased but does
      **not** conserve mass, so it must not be used for spike counts, occupancy,
      or any extensive quantity. Use nearest if exact value preservation is
      critical, or smooth an extensive field with ``env.smooth(mode="transition")``.

    **Use cases**:

    - Align fields from different sessions with different bin sizes
    - Compare fields at multiple resolutions
    - Transfer learned representations between environments
    - Upsample/downsample spatial fields

    Examples
    --------
    Basic nearest-neighbor resampling:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.binning import resample_field
    >>> # Source: coarse resolution
    >>> src_data = np.array([[i, j] for i in range(21) for j in range(21)])
    >>> src_env = Environment.from_samples(src_data, bin_size=4.0)
    >>> # Destination: fine resolution
    >>> dst_data = np.array([[i, j] for i in range(21) for j in range(21)])
    >>> dst_env = Environment.from_samples(dst_data, bin_size=2.0)
    >>> # Create field on source
    >>> field = np.random.rand(src_env.n_bins)
    >>> # Resample to destination
    >>> resampled = resample_field(field, src_env, dst_env, method="nearest")
    >>> resampled.shape == (dst_env.n_bins,)
    True

    Diffuse method (intensive fields only — e.g. a rate map or density):

    >>> # An intensive field (rate map). Do NOT diffuse spike counts / mass.
    >>> rate_map = np.random.rand(src_env.n_bins)
    >>> resampled_smooth = resample_field(
    ...     rate_map, src_env, dst_env, method="diffuse", bandwidth=2.0
    ... )
    >>> resampled_smooth.shape == (dst_env.n_bins,)
    True

    Identity resampling (same environment):

    >>> resampled_identity = resample_field(field, src_env, src_env)
    >>> np.allclose(resampled_identity, field)
    True
    """
    # Validate inputs
    if field.shape[0] != src_env.n_bins:
        raise ValueError(
            f"Field size {field.shape[0]} does not match source environment "
            f"n_bins ({src_env.n_bins})"
        )

    if method not in ("nearest", "diffuse"):
        raise ValueError(f"method must be 'nearest' or 'diffuse' (got '{method}')")

    if method == "diffuse":
        if bandwidth is None:
            raise ValueError(
                "bandwidth must be provided when method='diffuse'. "
                "Specify bandwidth parameter (in spatial units)."
            )
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive (got {bandwidth})")

    # Check dimension compatibility
    if src_env.bin_centers.shape[1] != dst_env.bin_centers.shape[1]:
        raise ValueError(
            f"Source and destination environments must have same number of dimensions. "
            f"Source has {src_env.bin_centers.shape[1]}D, "
            f"destination has {dst_env.bin_centers.shape[1]}D."
        )

    # Step 1: Map each destination bin center to nearest source bin
    # Use map_points_to_bins for KD-tree cached nearest-neighbor lookup
    # (return_dist defaults to False, so the result is the index array).
    dst_to_src_indices = cast(
        "NDArray[np.int64]",
        map_points_to_bins(
            dst_env.bin_centers,
            src_env,
            tie_break=TieBreakStrategy.LOWEST_INDEX,
        ),
    )

    # Step 2: Pullback field values via nearest neighbor.
    # map_points_to_bins returns -1 for destination bin centers that fall
    # outside the source environment. Naive ``field[dst_to_src_indices]`` would
    # silently resolve -1 to ``field[-1]`` (Python negative indexing). Gather
    # safely with the negative indices clamped, then mask those positions NaN.
    outside_source = dst_to_src_indices < 0
    safe_indices = np.where(outside_source, 0, dst_to_src_indices)
    resampled: NDArray[np.float64] = field[safe_indices].astype(np.float64, copy=True)
    resampled[outside_source] = np.nan

    # Step 3: Optionally apply smoothing for diffuse method
    if method == "diffuse":
        # Type narrowing: bandwidth is guaranteed to be float at this point
        assert bandwidth is not None  # Already validated above

        # A pullback field is *intensive* (a rate map / probability density
        # sampled onto dst_env), so smooth it with the row-stochastic average
        # kernel (masked Nadaraya-Watson), not the mass-conserving transition
        # kernel. This removes the volume bias on non-uniform M and, crucially,
        # avoids the down-bias of the old zero-fill-then-single-smooth: uncovered
        # / NaN bins contribute *no weight* (rather than a real 0 that pulls
        # covered neighbours toward zero).
        kernel = cast("EnvironmentProtocol", dst_env).compute_kernel(
            bandwidth=bandwidth, mode="average"
        )

        # `valid` bins are covered by the source AND finite; only those
        # contribute. The value is zero-filled where invalid so an un-zeroed
        # NaN cannot poison every reachable bin through the matmul.
        valid = (~outside_source) & np.isfinite(resampled)
        values = np.where(valid, resampled, 0.0)
        num = kernel @ values
        den = kernel @ valid.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed = np.where(den > 0.0, num / den, np.nan)

        # Re-impose NaN only on structurally out-of-source bins. An interior
        # source-NaN bin with valid neighbours within the bandwidth is thereby
        # interpolated (like the binned gap-fill); a bin with no valid neighbour
        # (den == 0) is already NaN from the `where` above.
        smoothed[outside_source] = np.nan
        resampled = smoothed

    return resampled
