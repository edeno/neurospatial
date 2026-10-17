"""Point-to-bin mapping utilities for neurospatial.

This module provides high-performance spatial query utilities including:
- Batch mapping of points to bins with KD-tree caching
- Deterministic tie-breaking on bin boundaries
- Region rasterization to bin masks
- Field resampling between environments

These are core primitives used throughout neurospatial and by downstream packages.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial.geometry._coordinates import require_cartesian
from neurospatial.geometry._locators import CallIndexes
from neurospatial.geometry._region_bins import region_bin_mask

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.ops._protocols import (
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
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.mapping import TieBreakStrategy
    >>> env = Environment.from_samples(np.zeros((1, 2)), bin_size=1.0)
    >>> # Use enum for autocomplete and type safety
    >>> env.nearest_bin(
    ...     np.array([[0.1, 0.1]]), tie_break=TieBreakStrategy.CLOSEST_CENTER
    ... )
    array([0])
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


def _typical_bin_spacing(
    bin_centers: NDArray[np.float64],
    bin_sizes: NDArray[np.float64],
    *,
    is_linearized_track: bool,
    tree: cKDTree,
) -> float:
    """Typical distance between neighbouring bins, measured along the space.

    On a linearized track this is the median bin length along the track
    (``bin_sizes``): Euclidean nearest neighbours can lie on another arm
    of the track. Otherwise it is the median nearest-neighbour distance
    between bin centres, which equals ``bin_size`` on a regular grid.

    Parameters
    ----------
    bin_centers : ndarray of float64, shape (n_bins, n_dims)
        Bin centers, ``env.bin_centers``.
    bin_sizes : ndarray of float64, shape (n_bins,)
        Bin measures, ``env.bin_sizes``.
    is_linearized_track : bool
        Whether the bins lie along a linearized track,
        ``env.is_linearized_track``.
    tree : scipy.spatial.cKDTree
        KD-tree over ``bin_centers``, ``env.bin_center_tree()``.

    Returns
    -------
    float
        The spacing; ``np.inf`` for a single-bin environment that is not a
        track.
    """
    if is_linearized_track:
        return float(np.median(bin_sizes))
    return _estimate_typical_bin_spacing(tree, bin_centers)


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
    >>> from neurospatial.ops.mapping import regions_to_mask
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
    indexes = CallIndexes()
    for region in resolved:
        mask |= region_bin_mask(
            region,
            binning=binning,
            arena=arena,
            include_boundary=include_boundary,
            indexes=indexes,
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
    Environment.nearest_bin : Nearest-center point-to-bin mapping
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
    >>> from neurospatial.ops.mapping import resample_field
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
    for argument, env in (("src_env", src_env), ("dst_env", dst_env)):
        require_cartesian(
            env.coordinate_system,
            f"resample_field({argument}=...)",
            why=(
                "It matches each target bin to the nearest Cartesian source bin center."
            ),
            fix=(
                "resample between allocentric environments; build polar maps with the "
                "same Environment.from_polar_egocentric(...) settings so their bins "
                "match"
            ),
        )
    # Read every component once; the resampling below uses only these.
    src_centers = src_env.bin_centers
    dst_centers = dst_env.bin_centers
    if field.shape[0] != src_centers.shape[0]:
        raise ValueError(
            f"Field size {field.shape[0]} does not match source environment "
            f"n_bins ({src_centers.shape[0]})"
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
    if src_centers.shape[1] != dst_centers.shape[1]:
        raise ValueError(
            f"Source and destination environments must have same number of dimensions. "
            f"Source has {src_centers.shape[1]}D, "
            f"destination has {dst_centers.shape[1]}D."
        )

    # The source's nearest-center lookup is read here, at the public boundary:
    # the source environment caches its KD-tree, so repeated resampling from it
    # does not rebuild the tree.
    resampled, outside_source = _pull_back_nearest(
        field,
        dst_centers,
        lambda points: cast(
            "NDArray[np.int64]",
            src_env.nearest_bin(points, tie_break=TieBreakStrategy.LOWEST_INDEX),
        ),
    )
    if method == "nearest":
        return resampled
    assert bandwidth is not None  # Already validated above
    # The destination's diffusion operator is extracted here, at the public
    # boundary: its truncated eigenbasis is cached by the destination
    # environment, so repeated resampling onto it does not rebuild the basis.
    geometry = dst_env.get_diffusion_geometry()
    return _smooth_pulled_back(
        resampled,
        outside_source,
        lambda columns: np.asarray(
            dst_env.diffuse(columns, bandwidth=bandwidth, mode="average")
        ),
        n_components=geometry.n_components,
        labels=geometry.labels,
    )


def _pull_back_nearest(
    field: NDArray[np.float64],
    dst_centers: NDArray[np.float64],
    nearest_source_bin: Callable[[NDArray[np.float64]], NDArray[np.int64]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Give each destination bin the value of its nearest source bin center.

    Parameters
    ----------
    field : ndarray of float64, shape (n_src_bins,)
        Values on the source bins.
    dst_centers : ndarray of float64, shape (n_dst_bins, n_dims)
        Destination bin centers, in the source's coordinates.
    nearest_source_bin : callable
        Maps points of shape ``(n_points, n_dims)`` to the nearest source bin,
        ``-1`` for a point too far from every source center.

    Returns
    -------
    resampled : ndarray of float64, shape (n_dst_bins,)
        Pulled-back values; NaN where the source value is NaN or the
        destination bin is outside the source.
    outside_source : ndarray of bool, shape (n_dst_bins,)
        Destination bins that the nearest-center lookup rejects as outside the
        source.
    """
    dst_to_src_indices = nearest_source_bin(dst_centers)
    # The lookup returns -1 for destination bin centers that fall outside the
    # source environment. Naive ``field[dst_to_src_indices]`` would silently
    # resolve -1 to ``field[-1]`` (Python negative indexing). Gather safely with
    # the negative indices clamped, then mask those positions NaN.
    outside_source = dst_to_src_indices < 0
    safe_indices = np.where(outside_source, 0, dst_to_src_indices)
    resampled: NDArray[np.float64] = field[safe_indices].astype(np.float64, copy=True)
    resampled[outside_source] = np.nan
    return resampled, outside_source


def _smooth_pulled_back(
    resampled: NDArray[np.float64],
    outside_source: NDArray[np.bool_],
    smooth_average: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    n_components: int,
    labels: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Smooth a pulled-back intensive field over its valid destination bins.

    A pullback field is *intensive* (a rate map / probability density sampled
    onto the destination), so it is smoothed with the row-stochastic average
    operator (masked Nadaraya-Watson), not the mass-conserving transition
    kernel: uncovered / NaN bins contribute *no weight* rather than a real 0
    that pulls covered neighbours toward zero. Only a bin outside the source or
    in a component with no valid bin is left NaN (no far-field floor), and the
    signed numerator is never clipped.

    Parameters
    ----------
    resampled : ndarray of float64, shape (n_bins,)
        Pulled-back values, NaN outside the source or where the source value
        is missing.
    outside_source : ndarray of bool, shape (n_bins,)
        Destination bins outside the source; they stay NaN. Bins whose source
        value is NaN are filled from their valid neighbours.
    smooth_average : callable
        The destination's ``mode="average"`` diffusion of a
        ``(n_bins, n_fields)`` batch.
    n_components : int
        Number of connected components of the destination's diffusion graph.
    labels : ndarray of int, shape (n_bins,)
        Component of each destination bin.

    Returns
    -------
    ndarray of float64, shape (n_bins,)
        Smoothed values, NaN outside the source.
    """
    from neurospatial.ops.diffusion import masked_diffusion_average

    valid = (~outside_source) & np.isfinite(resampled)
    smoothed = masked_diffusion_average(
        smooth_average,
        resampled[:, None],
        valid[:, None],
        labels,
        n_components,
        floor=None,
    )[:, 0]
    smoothed[outside_source] = np.nan
    return smoothed
