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

from neurospatial.geometry._coordinates import require_cartesian
from neurospatial.geometry._region_bins import region_bin_mask

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.ops._protocols import (
        BinSpacingConsumer,
        CartesianNearestBinConsumer,
        RegionMaskConsumer,
        ResampleSourceConsumer,
        ResampleTargetConsumer,
    )
    from neurospatial.regions import Region, Regions


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
    ...     env, points, tie_break=TieBreakStrategy.CLOSEST_CENTER
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


def _typical_bin_spacing(env: BinSpacingConsumer) -> float:
    """Typical distance between neighbouring bins, measured along the space.

    On a linearized track this is the median bin length along the track
    (``env.bin_sizes``): Euclidean nearest neighbours can lie on another arm
    of the track. Otherwise it is the median nearest-neighbour distance
    between bin centres, which equals ``bin_size`` on a regular grid.

    Parameters
    ----------
    env : Environment
        An environment.

    Returns
    -------
    float
        The spacing; ``np.inf`` for a single-bin environment that is not a
        track.
    """
    if env.is_linearized_track:
        return float(np.median(env.bin_sizes))
    return _estimate_typical_bin_spacing(cKDTree(env.bin_centers), env.bin_centers)


def map_points_to_bins(
    env: CartesianNearestBinConsumer,
    points: NDArray[np.float64],
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
    env : Environment
        Environment containing the bin discretization.
    points : NDArray[np.float64], shape (n_points, n_dims)
        Continuous coordinates to map to bins.
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
        points to the closest in-env bin is the goal. A point with a NaN
        coordinate is a missing sample and gets ``-1``, as does a point with
        an infinite coordinate.
    distances : NDArray[np.float64], shape (n_points,), optional
        Distance from each point to its assigned bin center: ``inf`` for a
        rejected point and NaN for a missing one.
        Only returned if `return_dist=True`.

    Raises
    ------
    ValueError
        If both max_distance and max_distance_factor are specified, or if either
        is negative, or if invalid tie_break mode.
    CartesianOnlyError
        If ``env`` is polar: nearest centers in ``(distance, angle)`` space
        mix units and ignore the angle seam.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.binning import map_points_to_bins
    >>> data = np.random.randn(1000, 2) * 10  # doctest: +SKIP
    >>> env = Environment.from_samples(data, bin_size=2.0)  # doctest: +SKIP
    >>> points = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 50.0]])  # doctest: +SKIP
    >>> bins = map_points_to_bins(env, points)  # doctest: +SKIP
    >>> bins  # doctest: +SKIP
    array([ 42,  89,  -1])

    >>> # Get distances too
    >>> bins, dists = map_points_to_bins(
    ...     env, points, return_dist=True
    ... )  # doctest: +SKIP
    >>> dists  # doctest: +SKIP
    array([0.23, 0.45, inf])

    >>> # Filter outliers with absolute threshold
    >>> bins = map_points_to_bins(env, points, max_distance=15.0)  # doctest: +SKIP

    >>> # Filter outliers with relative threshold (adapts to bin size)
    >>> bins = map_points_to_bins(
    ...     env, points, max_distance_factor=1.5
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
    require_cartesian(
        env.coordinate_system,
        "map_points_to_bins()",
        why="It finds the nearest bin center to each Cartesian point.",
        fix="find the bins of (distance, angle) rows with env.nearest_bin(points)",
    )

    return env.nearest_bin(
        points,
        tie_break=tie_break,
        return_dist=return_dist,
        max_distance=max_distance,
        max_distance_factor=max_distance_factor,
    )


def _validate_nearest_options(
    tie_break: TieBreakStrategy | Literal["lowest_index", "closest_center"],
    max_distance: float | None,
    max_distance_factor: float | None,
) -> TieBreakStrategy:
    """Validate distance controls and normalize the supported tie-break strings."""
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

    return tie_break


def _nearest_bin_query(
    bin_centers: NDArray[np.float64],
    kdtree: cKDTree,
    points: NDArray[np.float64],
    *,
    tie_break: TieBreakStrategy = TieBreakStrategy.LOWEST_INDEX,
    return_dist: bool = False,
    max_distance: float | None = None,
    max_distance_factor: float | None = None,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Query explicit centers and their tree using validated distance controls.

    Parameters
    ----------
    bin_centers : ndarray, shape (n_bins, n_dims)
        Coordinates indexed by the already constructed kdtree.
    kdtree : scipy.spatial.cKDTree
        Tree over bin_centers; its lifetime belongs to the caller.
    points : ndarray, shape (n_points, n_dims)
        Query coordinates in the same stored coordinate columns.
    """
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
        max_neighbors = min(10, len(bin_centers))
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
                    bin_centers[None, :, :] - points[tied_rows, None, :],
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
            typical_bin_spacing = _estimate_typical_bin_spacing(kdtree, bin_centers)
            threshold = max_distance_factor * typical_bin_spacing

        # Mark points beyond threshold as outside
        bin_indices[distances > threshold] = -1
    elif len(bin_centers) > 1:
        # Backward compatibility: use old heuristic (10× typical spacing)
        typical_bin_spacing = _estimate_typical_bin_spacing(kdtree, bin_centers)

        # Mark points that are suspiciously far as outside
        threshold = 10 * typical_bin_spacing
        bin_indices[distances > threshold] = -1

    if return_dist:
        # Set distance to inf for points outside environment
        distances_out = distances.copy()
        distances_out[bin_indices == -1] = np.inf
        return (bin_indices, distances_out)

    return bin_indices


def regions_to_mask(
    env: RegionMaskConsumer,
    regions: str | list[str | Region] | Region | Regions,
    *,
    include_boundary: bool = True,
) -> NDArray[np.bool_]:
    """Select the environment bins that one or more regions cover.

    A point region covers the one bin that contains the point. A polygon
    region covers the bins whose centers lie inside it. Every region query
    (``Environment.bins_in_region``, ``region_mask``, ``region_membership``,
    ``distance_to(region=...)`` and ``subset(region_names=...)``) uses this
    same rule, so they agree on every layout.

    Parameters
    ----------
    env : Environment
        Environment whose bins are selected.
    regions : str, list of str or Region, Region, or Regions
        Region(s) to select. A name refers to an entry in ``env.regions``;
        Region objects are used directly.
    include_boundary : bool, default=True
        For polygon regions, count a bin whose center lies exactly on the
        polygon's edge as inside (``shapely.covers``); with False only centers
        strictly inside count (``shapely.contains``). Point regions ignore it.

    Returns
    -------
    mask : NDArray[np.bool_], shape (n_bins,)
        True for each bin covered by any of the regions (their union).

    Raises
    ------
    RegionNotFoundError
        If a region name is not in ``env.regions`` (a ``KeyError``).
    TypeError
        If ``regions`` has an invalid type or ``include_boundary`` is not a bool.
    ValueError
        If a point region has a different number of coordinates than the
        bins, or a polygon region is used with bins that are not 2-D.

    Notes
    -----
    For intersection or difference, combine separate masks::

        mask_a = regions_to_mask(env, "region_a")
        mask_b = regions_to_mask(env, "region_b")
        intersection = mask_a & mask_b
        difference = mask_a & ~mask_b

    This function is the approximate inverse of
    :func:`neurospatial.regions.mask_to_region`; a round trip
    mask → polygon → mask keeps the spatial structure but may differ at the
    edges because the contour is approximated.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely.geometry import box
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.binning import regions_to_mask
    >>> from neurospatial.regions import Regions
    >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> regions = Regions()
    >>> _ = regions.add("center", polygon=box(3, 3, 7, 7))
    >>> _ = regions.add("reward", point=(9.0, 9.0))
    >>> env = env.with_regions(regions)
    >>> int(regions_to_mask(env, "center").sum())
    4
    >>> int(regions_to_mask(env, "reward").sum())  # the bin containing the point
    1
    >>> regions_to_mask(env, ["center", "reward"]).shape
    (36,)

    See Also
    --------
    neurospatial.regions.mask_to_region : Convert a bin mask to a polygon.
    Environment.region_membership : Bins-by-regions membership table.
    resample_field : Resample fields between different discretizations.
    """
    return _regions_mask(
        regions,
        named_regions=env.regions,
        binning=env.binning,
        arena=env.arena,
        include_boundary=include_boundary,
    )


def _regions_mask(
    regions: str | list[str | Region] | Region | Regions,
    *,
    named_regions: Regions,
    binning: Binning,
    arena: Arena,
    include_boundary: bool,
) -> NDArray[np.bool_]:
    """Union of the bins covered by regions or by names in ``named_regions``."""
    from neurospatial._exceptions import RegionNotFoundError
    from neurospatial.regions import Region, Regions

    def named(name: str) -> Region:
        if name not in named_regions:
            raise RegionNotFoundError(
                name, available=list(named_regions.keys()), argument="regions"
            )
        return named_regions[name]

    resolved: list[Region]
    if isinstance(regions, str):
        resolved = [named(regions)]
    elif isinstance(regions, list):
        resolved = [named(r) if isinstance(r, str) else r for r in regions]
    elif isinstance(regions, Region):
        resolved = [regions]
    elif isinstance(regions, Regions):
        resolved = list(regions.values())
    else:
        raise TypeError(
            "regions must be str, list[str], Region, or Regions; "
            f"got {type(regions).__name__}."
        )
    if not isinstance(include_boundary, bool):
        raise TypeError(
            f"include_boundary must be a bool, got {type(include_boundary).__name__}."
        )
    mask = np.zeros(binning.bins.n_bins, dtype=np.bool_)
    for region in resolved:
        mask |= region_bin_mask(
            region, binning=binning, arena=arena, include_boundary=include_boundary
        )
    return mask


def resample_field(
    field: NDArray[np.float64],
    src_env: ResampleSourceConsumer,
    dst_env: ResampleTargetConsumer,
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
    CartesianOnlyError
        If either environment is polar: bins are matched by nearest Cartesian
        bin center.

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
    and a ``NaN`` in the source field is interpolated rather than propagating.
    The heat kernel has full support across each connected diffusion component,
    so any bin in a component that contains at least one valid bin is filled
    (weighted by distance — near valid bins dominate). A bin is left ``NaN`` only
    if it is outside the source or lies in a component with no valid bin at all.

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
    for env in (src_env, dst_env):
        require_cartesian(
            env.coordinate_system,
            "resample_field()",
            why=(
                "It matches each target bin to the nearest Cartesian source bin center."
            ),
            fix=(
                "resample between allocentric environments; build polar maps with the "
                "same Environment.from_polar_egocentric(...) settings so their bins "
                "match"
            ),
        )
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
            src_env,
            dst_env.bin_centers,
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
        # operator (masked Nadaraya-Watson), not the mass-conserving transition
        # kernel: uncovered / NaN bins contribute *no weight* rather than a real
        # 0 that pulls covered neighbours toward zero. Only a bin outside the
        # source or in a component with no valid bin is left NaN (no far-field
        # floor), and the signed numerator is never clipped.
        from neurospatial.ops.diffusion import (
            diffusion_component_labels,
            masked_diffusion_average,
        )

        valid = (~outside_source) & np.isfinite(resampled)
        n_components, labels = diffusion_component_labels(dst_env)
        smoothed = masked_diffusion_average(
            lambda columns: np.asarray(
                dst_env.diffuse(columns, bandwidth, mode="average")
            ),
            resampled[:, None],
            valid[:, None],
            labels,
            n_components,
            floor=None,
        )[:, 0]

        smoothed[outside_source] = np.nan
        resampled = smoothed

    return resampled
