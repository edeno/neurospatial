"""Point-to-bin mapping utilities for neurospatial.

This module provides high-performance spatial query utilities including:
- Nearest-bin-center queries over a caller's KD-tree
- Deterministic tie-breaking on bin boundaries
- Region rasterization to bin masks
- Field resampling between environments

These are core primitives used throughout neurospatial and by downstream packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from neurospatial.geometry import CallIndexes, region_bin_masks, require_cartesian

if TYPE_CHECKING:
    from neurospatial.geometry import Arena, Binning
    from neurospatial.ops._protocols import (
        NamedRegionConsumer,
        RegionMaskConsumer,
        ResampleSourceConsumer,
        ResampleTargetConsumer,
    )
    from neurospatial.regions import Region, Regions


TieBreakStrategy: TypeAlias = Literal["lowest_index", "closest_center"]
"""How ``Environment.nearest_bin`` resolves a point equidistant from several centers.

- ``"lowest_index"``: the tied bin with the smallest index, so a point on a
  bin boundary always gets the same bin. The default.
- ``"closest_center"``: whichever tied center the KD-tree finds first, which
  is not necessarily the lowest index. It skips the tie check, so it is a
  little faster.
"""

_TIE_BREAKS: tuple[TieBreakStrategy, ...] = ("lowest_index", "closest_center")


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
    tie_break: TieBreakStrategy,
    max_distance: float | None,
    max_distance_factor: float | None,
) -> TieBreakStrategy:
    """Validate the tie-break name and the distance controls."""
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

    if tie_break not in _TIE_BREAKS:
        raise ValueError(
            f"Invalid tie_break value: {tie_break!r}. Must be one of: "
            f"{list(_TIE_BREAKS)}"
        )

    return tie_break


def _lowest_tied_index(
    bin_centers: NDArray[np.float64],
    kdtree: cKDTree,
    points: NDArray[np.float64],
    nearest: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Lowest index among the centers equidistant (within 1e-10) to each point.

    Only the centers in a ball just wider than each point's nearest distance
    can tie, so their distances are computed instead of all centers'.

    Parameters
    ----------
    bin_centers : ndarray, shape (n_bins, n_dims)
    kdtree : scipy.spatial.cKDTree
        Tree over ``bin_centers``.
    points : ndarray, shape (n_points, n_dims)
    nearest : ndarray, shape (n_points,)
        Distance from each point to its nearest center, from ``kdtree``.

    Returns
    -------
    ndarray of int64, shape (n_points,)
    """
    # The margin covers the tie tolerance and rounding differences between
    # the tree's distances and np.linalg.norm.
    radius = nearest * (1.0 + 1e-12) + 2e-10
    candidates = kdtree.query_ball_point(
        points, radius, workers=-1, return_sorted=False
    )
    counts = np.fromiter(map(len, candidates), dtype=np.intp, count=len(points))
    index = np.concatenate(candidates).astype(np.int64)
    row = np.repeat(np.arange(len(points)), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    distance = np.linalg.norm(bin_centers[index] - points[row], axis=1)
    row_min = np.minimum.reduceat(distance, starts)
    tied = np.abs(distance - row_min[row]) < 1e-10
    return np.minimum.reduceat(np.where(tied, index, np.iinfo(np.int64).max), starts)


@overload
def _nearest_bin_query(
    bin_centers: NDArray[np.float64],
    kdtree: cKDTree,
    points: NDArray[np.float64],
    *,
    tie_break: TieBreakStrategy = ...,
    return_dist: Literal[False] = ...,
    max_distance: float | None = ...,
    max_distance_factor: float | None = ...,
) -> NDArray[np.int64]: ...
@overload
def _nearest_bin_query(
    bin_centers: NDArray[np.float64],
    kdtree: cKDTree,
    points: NDArray[np.float64],
    *,
    tie_break: TieBreakStrategy = ...,
    return_dist: bool,
    max_distance: float | None = ...,
    max_distance_factor: float | None = ...,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def _nearest_bin_query(
    bin_centers: NDArray[np.float64],
    kdtree: cKDTree,
    points: NDArray[np.float64],
    *,
    tie_break: TieBreakStrategy = "lowest_index",
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
    if tie_break == "closest_center":
        # Fast path: just use nearest neighbor
        distances, indices = kdtree.query(points, k=1, workers=-1)
        bin_indices: NDArray[np.int64] = indices.astype(np.int64)

    elif tie_break == "lowest_index":
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
            tied_rows = np.flatnonzero(np.count_nonzero(is_tied, axis=1) > 1)
            if tied_rows.size:
                bin_indices[tied_rows] = _lowest_tied_index(
                    bin_centers, kdtree, points[tied_rows], distances[tied_rows]
                )

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


def _require_region_mask(
    env: NamedRegionConsumer,
    name: str,
    *,
    operation: str,
    argument: str,
    in_list: bool = False,
) -> NDArray[np.bool_]:
    """Bins of the named region, refusing a missing region or one with no bins.

    Parameters
    ----------
    env : Environment
        Environment whose ``regions`` hold ``name``.
    name : str
        Region name.
    operation : str
        Calling function, named in the error, e.g. ``"segment_trials()"``.
    argument : str
        Calling function's argument that holds ``name``.
    in_list : bool, default False
        Whether ``argument`` takes a list of names.

    Returns
    -------
    mask : NDArray[np.bool_], shape (n_bins,)
        True for each bin the region covers (``regions_to_mask``).

    Raises
    ------
    RegionNotFoundError
        If ``name`` is not in ``env.regions``.
    ValueError
        If the region covers no bins, so the analysis would silently find no
        visits, trials or runs for it.
    """
    from neurospatial._exceptions import RegionNotFoundError, _format_error

    if name not in env.regions:
        raise RegionNotFoundError(
            name, available=list(env.regions.keys()), argument=argument, in_list=in_list
        )
    mask = regions_to_mask(env, [name])
    if mask.any():
        return mask
    region = env.regions[name]
    if region.kind == "point":
        point = np.asarray(region.data, dtype=np.float64)
        coords = ", ".join(f"{value:g}" for value in point)
        gap = float(np.min(np.linalg.norm(env.bin_centers - point, axis=1)))
        units = f" {env.units}" if env.units else ""
        why = (
            f"Why: a point region selects the bin that contains it, and "
            f"({coords}) lies outside every bin; the nearest bin center is "
            f"{gap:.3g}{units} away. A point on the arena's edge, such as a "
            "reward well at the end of an arm, is outside."
        )
        fix = (
            "move the point inside the environment, or give the region an "
            f"area, for example polygon=shapely.Point({coords}).buffer(radius)"
        )
    else:
        why = (
            "Why: a polygon region selects the bins whose centers it contains, "
            "and this one contains none."
        )
        fix = "enlarge the polygon or move it inside the environment"
    raise ValueError(
        _format_error(
            f"Region {name!r} covers no bins, so {operation} cannot find the "
            "animal in it.",
            why=why,
            fix=fix,
        )
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
    masks = region_bin_masks(
        resolved,
        binning=binning,
        arena=arena,
        include_boundary=include_boundary,
        indexes=CallIndexes(),
    )
    return np.asarray(masks.any(axis=1), dtype=np.bool_)


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

        - "nearest": value of the nearest source bin center (fast, preserves
          values), for destination bins inside the source
        - "diffuse": nearest-neighbor pullback followed by a boundary-aware,
          valid-bin-normalized diffusion **average** of the pulled-back field.
          For **intensive** fields only (rate maps / probability densities); it
          does not conserve mass, so do not use it for spike counts, occupancy,
          or any extensive/mass quantity.
    bandwidth : float, optional
        Smoothing bandwidth for the diffuse method, in the destination
        environment's length units (``dst_env.units``). Required when
        method="diffuse", ignored for method="nearest".

    Returns
    -------
    resampled : NDArray[np.float64], shape (dst_env.n_bins,)
        Field resampled onto destination environment bins. Destination bins
        whose centers lie in no source bin (``src_env.bin_at`` returns -1) are
        ``np.nan``. With "diffuse", missing (NaN) source values are averaged
        over as ``dst_env.smooth(..., mode="average")`` does: a missing bin is
        filled from observed neighbours, and a bin too far from every observed
        bin for the smoothing to resolve its value stays ``np.nan``.

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
    NotImplementedError
        If the source is a track built without ``track_width``: its centerline
        does not say which destination bins lie on it.

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

    **Nearest method**: A destination bin whose center lies in a source bin
    takes the value of the nearest source bin center (lowest index on ties),
    found with the source's KD-tree; one whose center lies in no source bin is
    NaN, however close it is to the source's edge. This is
    ``src_env.interpolate(field, dst_env.bin_centers, mode="nearest")``. No
    interpolation or smoothing is applied.

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

    # A destination bin is outside the source when its center lies in no
    # source bin, by the source's own containment rule (Environment.bin_at);
    # a distance-to-nearest-center threshold would let bins well outside the
    # source copy its edge values. Inside, a bin takes the value of the nearest
    # source bin center, as Environment.interpolate(mode="nearest") does. The
    # source caches its KD-tree, so repeated resampling does not rebuild it.
    outside_source = src_env.bin_at(dst_centers) < 0
    nearest_source_bins = _nearest_bin_query(
        src_centers,
        src_env.bin_center_tree(),
        dst_centers,
        tie_break="lowest_index",
    )
    resampled = _pull_back_nearest(field, nearest_source_bins, outside_source)
    if method == "nearest":
        return resampled
    assert bandwidth is not None  # Already validated above
    # The pulled-back field is intensive, so it is averaged over its valid
    # bins exactly as env.smooth(mode="average") averages a rate map with
    # missing bins: a NaN bin gets no weight and is filled from valid
    # neighbours, and a bin whose smoothed support is truncation noise stays
    # NaN rather than reporting noise / noise. The destination caches its
    # eigenbasis, so repeated resampling onto it does not rebuild it.
    smoothed = np.array(
        dst_env.diffuse(resampled, bandwidth=bandwidth, mode="average"),
        dtype=np.float64,
    )
    smoothed[outside_source] = np.nan
    return smoothed


def _pull_back_nearest(
    field: NDArray[np.float64],
    nearest_source_bins: NDArray[np.int64],
    outside_source: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Give each destination bin the value of its nearest source bin center.

    Parameters
    ----------
    field : ndarray of float64, shape (n_src_bins,)
        Values on the source bins.
    nearest_source_bins : ndarray of int64, shape (n_dst_bins,)
        Nearest source bin of each destination bin center.
    outside_source : ndarray of bool, shape (n_dst_bins,)
        Destination bins whose centers lie in no source bin.

    Returns
    -------
    ndarray of float64, shape (n_dst_bins,)
        Pulled-back values; NaN where the source value is NaN or the
        destination bin is outside the source.
    """
    resampled: NDArray[np.float64] = field[nearest_source_bins].astype(
        np.float64, copy=True
    )
    resampled[outside_source] = np.nan
    return resampled
