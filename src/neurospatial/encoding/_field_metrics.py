"""Field geometry and comparison metrics for spatial encoding.

This module provides utility functions for analyzing place field properties,
including geometry, stability, and comparisons between sessions.

Field Geometry Metrics
----------------------
field_size : Compute field size (area) in physical units.
rate_map_centroid : Compute firing-rate-weighted centroid.
field_shape_metrics : Compute geometric shape metrics (eccentricity, orientation).

Field Comparison Metrics
------------------------
field_stability : Correlation between two firing rate maps.
field_shift_distance : Distance between field centroids across sessions.
compute_field_emd : Earth Mover's Distance between rate distributions.
in_out_field_ratio : Ratio of in-field to out-of-field firing rate.
rate_map_coherence : Spatial coherence of a firing rate map.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem
    from neurospatial.ops._types import DistanceMetric

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    # Field geometry
    "field_size",
    "rate_map_centroid",
    "field_shape_metrics",
    # Field comparison
    "field_stability",
    "field_shift_distance",
    "compute_field_emd",
    "in_out_field_ratio",
    "rate_map_coherence",
]


def field_size(
    env: Environment,
    field_bins: NDArray[np.int64],
) -> float:
    """Compute field size (area) in physical units.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    field_bins : array
        Bin indices comprising the field.

    Returns
    -------
    size : float
        Field size: the summed measures of its bins, an area (e.g. cm^2) for
        2-D environments, a length (e.g. cm) on a track or 1-D grid, and a
        volume in 3-D.

    Notes
    -----
    Each bin's measure is ``env.bin_sizes``: ``bin_size**2`` on a 2-D grid,
    the hexagon or triangle area on hexagonal and mesh layouts, and the bin's
    length along a track.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import field_size
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> size = field_size(env, field_bins)
    >>> size > 0
    True
    """
    # Get bin sizes (property, not method)
    bin_sizes = env.bin_sizes

    # Sum areas of field bins
    total_size = np.sum(bin_sizes[field_bins])

    return float(total_size)


def _relative_measures(measures: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bin measures as fractions of the largest, shape like ``measures``.

    A measure-weighted statistic depends only on the ratios of the measures.
    Dividing by the largest makes equal bins weigh exactly 1, so on a regular
    grid the weighted statistic is the unweighted one, bit for bit.
    """
    relative: NDArray[np.float64] = measures / np.max(measures, initial=0.0)
    return relative


def rate_map_centroid(
    env: Environment,
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    *,
    method: DistanceMetric = "euclidean",
) -> NDArray[np.float64]:
    """Compute firing-rate-weighted centroid of place field.

    Parameters
    ----------
    env : Environment
        Spatial environment.
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    field_bins : array
        Bin indices comprising the field.
    method : {"euclidean", "geodesic"}, default "euclidean"
        Method for computing centroid:

        - ``"euclidean"``: Weighted mean in Euclidean space. Fast but may
          place centroid off-track for irregular geometries.
        - ``"geodesic"``: Weighted medoid using graph distances. Finds the bin
          within the field that minimizes weighted graph distance to all
          other field bins. Always on-track and respects maze geometry.

    Returns
    -------
    centroid : array, shape (n_dims,)
        Weighted center of mass in physical coordinates.

    Raises
    ------
    CartesianOnlyError
        If ``method="euclidean"`` and ``env`` is polar: averaging
        ``(distance, angle)`` bin centers is not a position. Use
        ``method="geodesic"``, which returns a field bin.

    Notes
    -----
    Each field bin is weighted by the firing it holds, its rate times its
    measure (``env.bin_sizes``: area, length or volume), so the centroid is
    that of the continuous rate field and does not depend on how finely a
    region is binned. On a regular grid every bin has the same measure and
    the weight is the rate alone.

    For ``method="euclidean"``, the centroid is the weighted mean position:

    .. math::

        \\mathbf{c} = \\frac{\\sum_i r_i \\mu_i \\mathbf{p}_i}{\\sum_i r_i \\mu_i}

    where :math:`r_i` is the firing rate, :math:`\\mu_i` the measure and
    :math:`\\mathbf{p}_i` the center of bin :math:`i`. A field with no firing
    returns its geometric centroid, :math:`\\sum_i \\mu_i \\mathbf{p}_i /
    \\sum_i \\mu_i`.

    For ``method="geodesic"``, the centroid is the field bin that minimizes the
    weighted sum of graph distances:

    .. math::

        c = \\arg\\min_j \\sum_i r_i \\mu_i \\, d_G(i, j)

    where :math:`d_G(i, j)` is the shortest path distance in the environment
    graph between bins :math:`i` and :math:`j`. This approach is preferred
    for mazes and complex geometries where Euclidean distances can cross
    walls.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import rate_map_centroid
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.random.rand(env.n_bins) * 5
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> centroid = rate_map_centroid(env, firing_rate, field_bins)
    >>> centroid.shape
    (2,)

    Use graph-based centroid for maze environments:

    >>> centroid_graph = rate_map_centroid(
    ...     env, firing_rate, field_bins, method="geodesic"
    ... )
    """
    if method == "euclidean":
        require_cartesian(
            env.coordinate_system,
            "rate_map_centroid(method='euclidean')",
            why=(
                "A rate-weighted mean of bin centers averages them as Cartesian "
                "positions."
            ),
            fix=(
                "use method='geodesic', which returns the field bin that minimizes "
                "rate-weighted graph distance"
            ),
        )

    # Each field bin holds rate x measure of firing: a bin twice as large
    # counts twice, so splitting a bin does not move the centroid.
    field_positions = env.bin_centers[field_bins]
    measures = _relative_measures(env.bin_sizes[field_bins])
    field_mass = firing_rate[field_bins] * measures

    centroid: NDArray[np.float64]

    if method == "euclidean":
        total_mass = np.sum(field_mass)
        if total_mass == 0:
            # No firing: the geometric centroid of the field.
            centroid = np.sum(field_positions * measures[:, None], axis=0) / np.sum(
                measures
            )
        else:
            centroid = (
                np.sum(field_positions * field_mass[:, None], axis=0) / total_mass
            )

    elif method == "geodesic":
        # Weighted medoid: the field bin that minimizes the sum of
        # (rate x measure x graph distance) to all field bins.

        from neurospatial.ops.distance import pairwise_distances

        # distances[i, j]: graph distance from field bin i to candidate j,
        # 0 on the diagonal and inf between disconnected bins.
        distances = pairwise_distances(env.connectivity, field_bins)
        with np.errstate(invalid="ignore"):
            # A zero-rate bin with no path adds 0 * inf = NaN; a NaN cost, like
            # an infinite one, never wins.
            costs = field_mass @ distances
        finite = costs < np.inf
        best = int(np.argmin(np.where(finite, costs, np.inf))) if finite.any() else 0
        centroid = field_positions[best]

    else:  # pragma: no cover
        # Runtime safety check (type system enforces at compile time)
        msg = f"Unknown method: {method}. Must be 'euclidean' or 'geodesic'."  # type: ignore[unreachable]
        raise ValueError(msg)

    return centroid


# A map whose spread over the compared bins is at most this fraction of its
# largest rate is flat. Dividing smoothed counts by smoothed occupancy leaves
# rounding of order 1e-14 relative (a flat map from the package's estimators
# spreads tens to hundreds of ulps), and correlating that noise reports a
# spurious score.
# A real field is many orders of magnitude deeper than 1e-12 of its peak.
_CONSTANT_RELATIVE_SPREAD = 1e-12


def _is_constant(
    values: NDArray[np.float64], common: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Whether each row is flat, up to rounding, over its ``common`` bins.

    A row is flat when ``max - min <= _CONSTANT_RELATIVE_SPREAD *
    max(|values|)`` over its common bins: the spread is rounding from
    computing the rates, not a field. Rows without common bins count as flat;
    the caller reports them as having too few bins first.
    """
    highest = np.where(common, values, -np.inf).max(axis=1, initial=-np.inf)
    lowest = np.where(common, values, np.inf).min(axis=1, initial=np.inf)
    magnitude = np.maximum(np.abs(highest), np.abs(lowest))
    tolerance = _CONSTANT_RELATIVE_SPREAD * magnitude
    constant: NDArray[np.bool_] = highest - lowest <= tolerance
    return constant


def field_stability(
    rate_map_1: NDArray[np.float64],
    rate_map_2: NDArray[np.float64],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Compute stability between two firing rate maps (correlation).

    Stability quantifies how consistent spatial firing is across sessions
    or trial halves. Used to assess place field reliability and memory.

    Parameters
    ----------
    rate_map_1 : array, shape (n_bins,)
        First firing rate map (Hz).
    rate_map_2 : array, shape (n_bins,)
        Second firing rate map (Hz).
    method : {'pearson', 'spearman'}, default='pearson'
        Correlation method. Pearson for linear correlation, Spearman
        for rank-based correlation.

    Returns
    -------
    stability : float
        Correlation coefficient in range [-1, 1]. Higher values indicate
        more stable place fields. NaN when the correlation is undefined:
        fewer than 3 bins are finite in both maps (two points always give
        exactly +-1), or either map is flat over those bins. NaN is not 0
        ("uncorrelated"); skip such cells with ``np.nanmean`` or
        ``np.isfinite`` when averaging across cells.

    Notes
    -----
    **Interpretation**:
    - High stability (r > 0.7): Reliable, stable place field
    - Medium stability (0.3 < r < 0.7): Moderately stable
    - Low stability (r < 0.3): Unstable or remapped field

    **Edge Cases**:
    - A map flat over the paired bins returns NaN (correlation undefined).
      Flat means its highest and lowest rates differ by at most ``1e-12``
      times its largest absolute rate, the rule ``compare_spatial_rates``
      uses: smoothed estimators leave a flat map with rounding of order
      ``1e-14`` relative, and correlating that rounding is not a measurement.
    - Fewer than 3 bins finite in both maps return NaN (2 points give exactly
      +-1 whatever the maps, so the correlation is not a measurement)

    **Pearson vs Spearman**: Pearson measures linear correlation of firing
    rates. Spearman measures rank correlation and is robust to outliers.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._field_metrics import field_stability
    >>> # Identical maps -> perfect correlation
    >>> rate_map = np.random.rand(100) * 5
    >>> stability = field_stability(rate_map, rate_map, method="pearson")
    >>> bool(np.abs(stability - 1.0) < 1e-6)
    True

    References
    ----------
    .. [1] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code.
           Science 261(5124).
    """
    # Remove NaN values
    valid_mask = np.isfinite(rate_map_1) & np.isfinite(rate_map_2)
    map1_clean = rate_map_1[valid_mask]
    map2_clean = rate_map_2[valid_mask]

    # Fewer than 3 paired bins: 2 points always give r = +-1 (not a measurement)
    # and fewer give no correlation; neither is "no similarity"
    if len(map1_clean) < 3:
        return np.nan

    # A map flat up to rounding has no field to correlate
    paired = np.vstack([map1_clean, map2_clean])
    if _is_constant(paired, np.ones(paired.shape, dtype=bool)).any():
        return np.nan

    # Compute correlation
    if method == "pearson":
        correlation, _ = stats.pearsonr(map1_clean, map2_clean)
    elif method == "spearman":
        correlation, _ = stats.spearmanr(map1_clean, map2_clean)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    return float(correlation)


def rate_map_coherence(
    env: Environment,
    firing_rate: NDArray[np.float64],
) -> float:
    """Compute spatial coherence of a firing rate map.

    Spatial coherence (Muller & Kubie, 1989) measures the smoothness of
    spatial firing: the Fisher z-transform (arctanh) of the Pearson
    correlation between each bin's firing rate and the mean rate of its
    spatial neighbors. High coherence indicates smooth, spatially structured
    firing. Low coherence indicates noisy or scattered firing.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second). NaN marks an
        unsampled bin.

    Returns
    -------
    float
        Spatial coherence, the Fisher z ``arctanh(r)`` of the correlation
        ``r``: 0 for no correlation and unbounded (``inf`` when ``r == 1``).
        ``np.tanh(coherence)`` recovers ``r``. Returns NaN if:
        - All firing rates are zero or constant (no variance)
        - All rates are NaN
        - Fewer than two bins have a rate and a fully observed neighborhood

    Notes
    -----
    **Algorithm**:

    1. For each bin i, compute mean firing rate of neighbors: m_i = mean(r_j) for j in neighbors(i)
    2. Compute the Pearson correlation r between bin rates r_i and neighbor means m_i
    3. Coherence = arctanh(r)

    Muller & Kubie (1989) report this z-transformed correlation, as does the
    BNT MATLAB toolbox's ``analyses.coherence`` by default. Some packages
    return ``r`` itself; check which one before comparing numbers.

    **Interpretation**:

    - **High coherence (> 0.7, r > 0.6)**: Smooth, spatially structured firing
    - **Medium coherence (0.3-0.7)**: Some spatial structure but with noise
    - **Low coherence (< 0.3)**: Noisy, poorly defined spatial firing

    **Graph-based approach**:

    This implementation uses `env.connectivity` to determine spatial neighbors,
    making it applicable to irregular environments and graphs with obstacles.
    Muller & Kubie (1989) average the eight surrounding pixels of a square
    grid, the neighborhood of a grid built with diagonal connectivity (the
    ``Environment.from_samples`` default). Bins with an unsampled (NaN)
    neighbor are left out rather than treating the missing rate as zero.

    References
    ----------
    Muller, R. U., & Kubie, J. L. (1989). The firing of hippocampal place cells
        predicts the future position of freely moving rats. Journal of Neuroscience,
        9(12), 4101-4110.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import rate_map_coherence
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Smooth Gaussian field (high coherence)
    >>> firing_rate_smooth = np.zeros(env.n_bins)
    >>> for i in range(env.n_bins):
    ...     center = env.bin_centers[i]
    ...     dist = np.linalg.norm(center - np.array([0, 0]))
    ...     firing_rate_smooth[i] = 5.0 * np.exp(-(dist**2) / (2 * 8**2))
    >>>
    >>> coherence_smooth = rate_map_coherence(env, firing_rate_smooth)
    >>> print(f"Smooth field coherence: {coherence_smooth:.3f}")  # doctest: +SKIP
    Smooth field coherence: 0.850

    See Also
    --------
    spatial_information : Spatial information (bits/spike)
    sparsity : Spatial sparsity
    field_stability : Temporal stability of firing rate maps
    """
    # Validate inputs
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )

    # Remove NaN values for computing neighbor means
    # But track which bins are valid for final correlation
    valid_bins = np.isfinite(firing_rate)

    if not np.any(valid_bins):
        # All NaN
        return np.nan

    if np.all(firing_rate[valid_bins] == firing_rate[valid_bins][0]):
        # All values are identical (constant map, no variance)
        return np.nan

    # Mean of each bin's neighbors. neighbor_reduce returns NaN for a bin with
    # any missing neighbor, so only bins whose whole neighborhood is observed
    # enter the correlation below.
    from neurospatial.ops.graph import neighbor_reduce

    neighbor_means = neighbor_reduce(
        env,
        firing_rate,
        op="mean",
        include_self=False,
    )

    # Now compute correlation between bin rates and their neighbor means
    # Only use bins where both the bin and its neighbor mean are valid
    valid_for_corr = valid_bins & np.isfinite(neighbor_means)

    if np.sum(valid_for_corr) < 2:
        # Need at least 2 points for correlation
        return np.nan

    bin_rates = firing_rate[valid_for_corr]
    neighbor_rate_means = neighbor_means[valid_for_corr]

    # Check for zero variance (would cause correlation to fail)
    if np.std(bin_rates) == 0 or np.std(neighbor_rate_means) == 0:
        return np.nan

    correlation, _ = stats.pearsonr(bin_rates, neighbor_rate_means)
    with np.errstate(divide="ignore"):
        return float(np.arctanh(correlation))


def in_out_field_ratio(
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
) -> float:
    """Compute ratio of in-field to out-of-field mean firing rate.

    This metric quantifies how much stronger firing is inside the place field
    compared to outside. Higher values indicate a more distinct place field with
    strong spatial selectivity.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    field_bins : NDArray[np.int64], shape (n_field_bins,)
        Indices of bins belonging to the place field.

    Returns
    -------
    float
        Ratio of in-field to out-of-field mean firing rate. Returns:
        - NaN if field is empty or covers all bins
        - NaN if out-of-field rate is zero and in-field rate is also zero
        - inf if out-of-field rate is zero but in-field rate is positive

    Notes
    -----
    **Formula**:

    .. math::

        R = \\frac{\\bar{r}_{\\text{in}}}{\\bar{r}_{\\text{out}}}

    where :math:`\\bar{r}_{\\text{in}}` is the mean firing rate inside the
    field and :math:`\\bar{r}_{\\text{out}}` is the mean rate outside.

    **Interpretation**:

    - **Ratio = 1.0**: No spatial selectivity (same firing inside and out)
    - **Ratio = 2-5**: Moderate place field (2-5x stronger inside)
    - **Ratio > 10**: Strong place field (10x or more stronger inside)

    **NaN handling**: NaN values in firing_rate are excluded from both
    in-field and out-of-field calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._field_metrics import in_out_field_ratio
    >>>
    >>> # Strong place field (10x ratio)
    >>> firing_rate = np.ones(100) * 1.0
    >>> firing_rate[40:50] = 10.0  # Field bins have 10 Hz
    >>> field_bins = np.arange(40, 50)
    >>> ratio = in_out_field_ratio(firing_rate, field_bins)
    >>> print(f"Ratio: {ratio:.1f}")  # doctest: +SKIP
    Ratio: 10.0

    See Also
    --------
    selectivity : Peak rate / mean rate

    References
    ----------
    .. [1] Jung et al. (1994). Comparison of spatial firing characteristics of
           units in dorsal and ventral hippocampus of the rat. J Neurosci 14(12).
    """
    # Validate field_bins
    if len(field_bins) == 0:
        return np.nan

    if len(field_bins) >= len(firing_rate):
        # Field covers entire environment
        return np.nan

    # Create mask for in-field and out-of-field bins
    in_field_mask = np.zeros(len(firing_rate), dtype=bool)
    in_field_mask[field_bins] = True

    # Handle NaN values
    valid_mask = np.isfinite(firing_rate)

    # In-field: bins in field AND valid
    in_valid = in_field_mask & valid_mask
    # Out-field: bins NOT in field AND valid
    out_valid = (~in_field_mask) & valid_mask

    if not np.any(in_valid) or not np.any(out_valid):
        return np.nan

    # Compute mean rates
    in_field_rate = np.mean(firing_rate[in_valid])
    out_field_rate = np.mean(firing_rate[out_valid])

    # Handle division by zero
    if out_field_rate == 0:
        if in_field_rate > 0:
            return np.inf
        else:
            return np.nan

    ratio = in_field_rate / out_field_rate

    return float(ratio)


def field_shape_metrics(
    env: Environment[CartesianSystem],
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
) -> dict[str, float]:
    """Compute geometric shape metrics for a place field.

    Analyzes the spatial geometry of a place field, including eccentricity,
    orientation, and extent along principal axes. Useful for characterizing
    field morphology and detecting elongated or circular fields.

    Parameters
    ----------
    env : Environment
        Spatial environment for bin positions. Must be 2D (n_dims == 2).

    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    field_bins : NDArray[np.int64], shape (n_field_bins,)
        Indices of bins belonging to the place field.
    Returns
    -------
    dict[str, float]
        Dictionary with shape metrics:
        - 'eccentricity': float in [0, 1], where 0 = circular, 1 = linear
        - 'major_axis_length': float, full width of the field along its
          major axis, ``2 * sqrt(lambda_max)``: two rate-weighted standard
          deviations of bin position (+/- 1 SD), in environment units
        - 'minor_axis_length': float, the same along the minor axis
        - 'orientation': float, angle of the major axis in radians from the
          first spatial dimension, in ``[-pi/2, pi/2)``. An axis has no
          direction, so a field along x is 0 and one along y is ``-pi/2``.
        - 'area': float, physical area of the field bins (e.g. cm^2), the
          same as :func:`field_size`

        Returns dict with NaN values if field is empty or environment is not 2D.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar: eccentricity and orientation come from the
        covariance of Cartesian bin centers.

    Notes
    -----
    **Eccentricity**: Computed from eigenvalues of the spatial covariance matrix
    of rate-weighted bin positions:

    .. math::

        e = \\sqrt{1 - \\frac{\\lambda_{\\text{min}}}{\\lambda_{\\text{max}}}}

    where :math:`\\lambda_{\\text{min}}` and :math:`\\lambda_{\\text{max}}` are
    the smallest and largest eigenvalues.

    **Interpretation**:

    - **Eccentricity = 0**: Circular field (equal extent in all directions)
    - **Eccentricity = 0.5**: Moderately elongated field
    - **Eccentricity -> 1**: Highly elongated, linear field

    **Orientation**: Angle of the major axis (eigenvector corresponding to
    largest eigenvalue) relative to the first spatial dimension. Useful for
    detecting field alignment with environmental features.

    **2D only**: This implementation currently supports only 2D environments.
    3D shape analysis would require different metrics (e.g., sphericity).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import field_shape_metrics
    >>>
    >>> # Create 2D environment
    >>> data = np.random.randn(1000, 2) * 20
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Create elongated field along x-axis
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Find bins near y=0, x in [0, 20]
    >>> centers = env.bin_centers
    >>> elongated_mask = (
    ...     (np.abs(centers[:, 1]) < 2) & (centers[:, 0] > 0) & (centers[:, 0] < 20)
    ... )
    >>> field_bins = np.where(elongated_mask)[0]
    >>> firing_rate[field_bins] = 10.0
    >>>
    >>> # Compute shape metrics
    >>> shape = field_shape_metrics(env, firing_rate, field_bins)
    >>> print(f"Eccentricity: {shape['eccentricity']:.2f}")  # doctest: +SKIP
    Eccentricity: 0.87

    See Also
    --------
    rate_map_centroid : Compute field center of mass
    field_size : Compute field area

    References
    ----------
    .. [1] Muller & Kubie (1989). The effects of changes in the environment on
           the spatial firing of hippocampal complex-spike cells. J Neurosci 9(1).
    .. [2] Knierim et al. (1995). Place cells, head direction cells, and the
           learning of landmark stability. J Neurosci 15(3).
    """
    require_cartesian(
        env.coordinate_system,
        "field_shape_metrics()",
        why=(
            "Eccentricity and orientation come from the covariance of Cartesian bin "
            "centers."
        ),
        fix=(
            "compute field shape on an allocentric environment, for example "
            "Environment.from_samples(positions, bin_size=...)"
        ),
    )
    # Initialize with NaN defaults
    result = {
        "eccentricity": np.nan,
        "major_axis_length": np.nan,
        "minor_axis_length": np.nan,
        "orientation": np.nan,
        "area": np.nan,
    }

    # Validate inputs
    if len(field_bins) == 0:
        return result

    if env.n_dims != 2:
        warnings.warn(
            f"field_shape_metrics currently only supports 2D environments, got {env.n_dims}D. "
            "Returning NaN values.",
            category=UserWarning,
            stacklevel=2,
        )
        return result

    # Get bin positions for field
    positions = env.bin_centers[field_bins]  # shape (n_field_bins, 2)

    # Get firing rates for weighting
    rates = firing_rate[field_bins]

    # Handle NaN values
    valid_mask = np.isfinite(rates)
    if not np.any(valid_mask):
        return result

    positions_valid = positions[valid_mask]
    rates_valid = rates[valid_mask]

    # Normalize rates to use as weights
    rate_weights = rates_valid / np.sum(rates_valid)

    # Compute rate-weighted centroid
    centroid = np.sum(rate_weights[:, np.newaxis] * positions_valid, axis=0)

    # Compute rate-weighted covariance matrix using einsum (vectorized outer product)
    # cov[j,k] = sum_i rate_weights[i] * centered[i,j] * centered[i,k]
    centered = positions_valid - centroid  # shape (n_valid, 2)
    cov = np.einsum("i,ij,ik->jk", rate_weights, centered, centered)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract metrics
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[1]

    # Eccentricity
    eccentricity = np.sqrt(1 - lambda_min / lambda_max) if lambda_max > 0 else 0.0

    # Axis lengths: full width at +/- 1 rate-weighted SD along each axis.
    major_axis = 2 * np.sqrt(lambda_max)
    minor_axis = 2 * np.sqrt(lambda_min)

    # Orientation of the major axis. The eigenvector's sign is arbitrary, so
    # fold its angle to an axial angle in [-pi/2, pi/2).
    direction = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    orientation = (direction + np.pi / 2) % np.pi - np.pi / 2

    area = field_size(env, field_bins)

    result.update(
        {
            "eccentricity": float(eccentricity),
            "major_axis_length": float(major_axis),
            "minor_axis_length": float(minor_axis),
            "orientation": float(orientation),
            "area": area,
        }
    )

    return result


def field_shift_distance(
    env_1: Environment[CartesianSystem],
    firing_rate_1: NDArray[np.float64],
    field_bins_1: NDArray[np.int64],
    env_2: Environment[CartesianSystem],
    firing_rate_2: NDArray[np.float64],
    field_bins_2: NDArray[np.int64],
    *,
    metric: DistanceMetric = "euclidean",
) -> float:
    """Compute distance between field centroids across sessions/environments.

    This metric quantifies how much a place field has shifted in position between
    two recording sessions or environments. Useful for detecting remapping,
    field stability, and spatial representation changes.

    Parameters
    ----------
    env_1 : Environment
        Spatial environment for first session.
    firing_rate_1 : NDArray[np.float64], shape (n_bins_1,)
        Firing rate map from first session (Hz or spikes/second).
    field_bins_1 : NDArray[np.int64], shape (n_field_bins_1,)
        Indices of bins belonging to place field in first session.
    env_2 : Environment
        Spatial environment for second session.
    firing_rate_2 : NDArray[np.float64], shape (n_bins_2,)
        Firing rate map from second session (Hz or spikes/second).
    field_bins_2 : NDArray[np.int64], shape (n_field_bins_2,)
        Indices of bins belonging to place field in second session.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric. ``"euclidean"`` is straight-line distance between
        the rate-weighted centroids, and works across different
        environments in one coordinate frame. ``"geodesic"`` is the shortest
        path along the connectivity graph between the bins containing the
        two centroids; it respects barriers and boundaries and requires
        ``env_1`` and ``env_2`` to have the same bins and connectivity.

    Returns
    -------
    float
        Distance between field centroids in the environments' spatial units.
        NaN if either field has no finite, positive-rate bins (no centroid),
        or, for ``"geodesic"``, if a centroid falls outside every bin (a
        warning says which). ``inf`` for ``"geodesic"`` when the two
        centroids lie in parts of the environment with no path between them.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"euclidean"`` or ``"geodesic"``.
    IncompatibleEnvironmentError
        If ``metric="geodesic"`` and ``env_1`` and ``env_2`` differ in their
        bins or connectivity: there is no single graph to measure a path on.
    CartesianOnlyError
        If ``env_1`` or ``env_2`` is polar: field centroids are Cartesian
        rate-weighted means of bin centers.

    Notes
    -----
    **Euclidean distance** (``metric="euclidean"``):

    Computes straight-line distance between rate-weighted field centroids:

    .. math::

        d = \\|c_1 - c_2\\|_2

    where :math:`c_1` and :math:`c_2` are the centroids in continuous space.

    **Geodesic distance** (``metric="geodesic"``):

    Computes shortest path distance along environment connectivity graph,
    respecting barriers and boundaries.

    **Cross-session alignment**:

    For comparing across sessions, environments should be aligned (e.g., using
    estimate_transform and Environment.apply_transform) to account for
    camera shifts, rotation, or scaling. If environments are not aligned,
    distance will include alignment error.

    **Use cases**:

    - **Rate remapping**: Same field location (distance ~ 0), different rates
    - **Global remapping**: Different field location (distance > 0)
    - **Field stability**: Measure distance across repeated sessions

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import field_shift_distance
    >>>
    >>> # Create two environments (same layout)
    >>> data = np.random.randn(1000, 2) * 20
    >>> env1 = Environment.from_samples(data, bin_size=2.0, name="session1")
    >>> env2 = Environment.from_samples(data, bin_size=2.0, name="session2")
    >>>
    >>> # Create place field in env1
    >>> firing_rate_1 = np.zeros(env1.n_bins)
    >>> centers1 = env1.bin_centers
    >>> field_mask_1 = np.linalg.norm(centers1 - [10, 10], axis=1) < 5
    >>> field_bins_1 = np.where(field_mask_1)[0]
    >>> firing_rate_1[field_bins_1] = 10.0
    >>>
    >>> # Create shifted field in env2 (shifted by ~7 units)
    >>> firing_rate_2 = np.zeros(env2.n_bins)
    >>> centers2 = env2.bin_centers
    >>> field_mask_2 = np.linalg.norm(centers2 - [15, 15], axis=1) < 5
    >>> field_bins_2 = np.where(field_mask_2)[0]
    >>> firing_rate_2[field_bins_2] = 10.0
    >>>
    >>> # Compute shift distance (env-first canonical argument order)
    >>> shift = field_shift_distance(
    ...     env1,
    ...     firing_rate_1,
    ...     field_bins_1,
    ...     env2,
    ...     firing_rate_2,
    ...     field_bins_2,
    ... )
    >>> print(f"Field shifted by: {shift:.1f} units")  # doctest: +SKIP
    Field shifted by: 7.1 units

    See Also
    --------
    rate_map_centroid : Compute field center of mass
    field_stability : Correlation-based stability measure

    References
    ----------
    .. [1] Leutgeb et al. (2005). Independent codes for spatial and episodic memory
           in hippocampal neuronal ensembles. Science 309(5734).
    .. [2] Colgin et al. (2008). Understanding memory through hippocampal remapping.
           Trends Neurosci 31(9).
    """
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            f"metric must be 'euclidean' or 'geodesic', got {metric!r}.\n"
            "Fix: pass metric='euclidean' for straight-line distance or "
            "metric='geodesic' for the path through the environment."
        )
    for argument, env in (("env_1", env_1), ("env_2", env_2)):
        require_cartesian(
            env.coordinate_system,
            f"field_shift_distance({argument}=...)",
            why=(
                "It measures the straight-line distance between Cartesian field "
                "centroids."
            ),
            fix=(
                "compare field positions on allocentric environments, for example "
                "Environment.from_samples(positions, bin_size=...)"
            ),
        )
    if metric == "geodesic":
        from neurospatial._exceptions import IncompatibleEnvironmentError
        from neurospatial.environment import _environment_differences

        bins, topology = _environment_differences(env_1, env_2)
        mismatch = bins or topology
        if mismatch is not None:
            raise IncompatibleEnvironmentError(
                "field_shift_distance(metric='geodesic') needs env_1 and env_2 "
                f"to have the same bins and connectivity; they differ in "
                f"{mismatch}. A path is measured on one environment's graph, "
                "and the other session's bins are not on it.",
                fix=(
                    "use metric='euclidean' to compare centroids across "
                    "environments in one coordinate frame, or compute both rate "
                    "maps on the same environment"
                ),
                first=env_1,
                second=env_2,
            )

    # Compute centroids for both fields
    centroid_1 = rate_map_centroid(env_1, firing_rate_1, field_bins_1)
    centroid_2 = rate_map_centroid(env_2, firing_rate_2, field_bins_2)

    # Check for NaN centroids
    if np.any(np.isnan(centroid_1)) or np.any(np.isnan(centroid_2)):
        return np.nan

    if metric == "euclidean":
        return float(np.linalg.norm(centroid_1 - centroid_2))

    # Same bins and connectivity: measure the path on env_1's graph.
    centroid_bins = env_1.bin_at(np.stack([centroid_1, centroid_2]))
    if np.any(centroid_bins < 0):
        warnings.warn(
            "A field centroid falls outside every bin of the environment (for "
            "example the centroid of a curved field lies in a wall), so it has "
            "no geodesic position. Returning NaN; use metric='euclidean' for "
            "this field.",
            category=UserWarning,
            stacklevel=2,
        )
        return np.nan
    return float(env_1.distance_between(bins=centroid_bins))


def compute_field_emd(
    env: Environment,
    firing_rate_1: NDArray[np.float64],
    firing_rate_2: NDArray[np.float64],
    *,
    metric: DistanceMetric = "euclidean",
    normalize: bool = True,
) -> float:
    """Compute Earth Mover's Distance (EMD) between two firing rate distributions.

    The Earth Mover's Distance (also known as Wasserstein distance or optimal
    transport distance) measures the minimum cost to transform one distribution
    into another. Unlike simple measures like correlation or mean squared error,
    EMD respects the spatial structure of the environment.

    This implementation supports both Euclidean distance (straight-line) and
    geodesic distance (shortest path through the environment's connectivity graph).
    Geodesic EMD is particularly useful for complex environments with barriers,
    mazes, or non-convex layouts where Euclidean distance is misleading.

    Parameters
    ----------
    env : Environment
        Spatial environment defining bin positions and connectivity.
    firing_rate_1 : NDArray[np.float64], shape (n_bins,)
        First firing rate distribution across spatial bins. NaN marks an
        unsampled bin.
    firing_rate_2 : NDArray[np.float64], shape (n_bins,)
        Second firing rate distribution across spatial bins. NaN marks an
        unsampled bin.
    metric : str, default="euclidean"
        Distance metric to use. Options:
        - "euclidean": Straight-line distance between bin centers
        - "geodesic": Shortest path distance through connectivity graph
    normalize : bool, default=True
        If True, normalize each distribution to sum to 1.0 over the bins
        finite in both maps before computing EMD. If False, use raw firing
        rates (their sums over those bins must already be equal).

    Returns
    -------
    emd : float
        Earth Mover's Distance between the two distributions.

        - With ``normalize=True``: the mean distance the firing-rate mass
          moves, in the environment's distance units (e.g. cm). Moving a
          whole field 10 cm gives 10.0.
        - With ``normalize=False``: cost in units of (rate x distance).
        - NaN if either distribution has zero total mass (with a warning).
        - ``inf`` with ``metric="geodesic"`` when the maps put different
          mass in parts of the environment with no path between them, so
          no transport plan exists.

    Raises
    ------
    ValueError
        If firing_rate arrays have different lengths.
        If firing_rate arrays don't match env.n_bins.
        If metric is not "euclidean" or "geodesic".
        If normalize=False and distributions have different total mass.
    CartesianOnlyError
        If ``metric="euclidean"`` and ``env`` is polar: the ground distance
        would treat ``(distance, angle)`` bin centers as ``(x, y)``. Use
        ``metric="geodesic"``.

    Warns
    -----
    UserWarning
        If normalized distributions have no mass (all zeros or NaN).

    See Also
    --------
    field_shift_distance : Distance between field centroids.

    Notes
    -----
    The Earth Mover's Distance is computed by solving the optimal transport problem.

    **Missing bins.** Only bins finite in both maps are compared: an
    unsampled (NaN) bin in either map is left out of both, and with
    ``normalize=True`` each map is renormalized over the remaining bins.
    Filling NaN with zero instead would treat unvisited places as silent
    and move mass onto them.

    **Metric choice:**

    - **Euclidean**: Fast, works for any environment, but ignores barriers and walls.
      Use for simple open fields or when computational speed is critical.

    - **Geodesic**: Slower, respects environment structure (barriers, walls, connectivity).
      Use for complex environments like mazes, multi-room layouts, or non-convex arenas.
      Mass moves only along paths in the connectivity graph, never between
      disconnected parts of the environment.

    **Interpretation:**

    - EMD = 0: Identical distributions
    - Small EMD: Distributions are similar and nearby in space
    - Large EMD: Distributions are different or far apart in space

    **Computational complexity:**

    - Euclidean: O(n^2) for distance matrix + O(n^3) for optimization
    - Geodesic: O(n^2 log n) for all-pairs shortest paths + O(n^3) for optimization

    For large environments (n_bins > 1000), consider subsampling or using
    approximate methods.

    References
    ----------
    .. [1] Rubner, Y., Tomasi, C., & Guibas, L. J. (2000). The Earth Mover's
           Distance as a Metric for Image Retrieval. International Journal of
           Computer Vision, 40(2), 99-121.
    .. [2] Villani, C. (2009). Optimal Transport: Old and New. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._field_metrics import compute_field_emd
    >>>
    >>> # Create environment
    >>> data = np.random.randn(1000, 2) * 20
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Create two similar firing rate distributions
    >>> field1 = np.exp(-0.1 * np.linalg.norm(env.bin_centers - [0, 0], axis=1) ** 2)
    >>> field2 = np.exp(-0.1 * np.linalg.norm(env.bin_centers - [5, 0], axis=1) ** 2)
    >>>
    >>> # Compute EMD with Euclidean distance (skipped in the doctest run: the
    >>> # optimal-transport solve is O(n^3) in the bin count and can take minutes
    >>> # / exhaust memory on a full environment -- see "Computational complexity")
    >>> emd_euclidean = compute_field_emd(  # doctest: +SKIP
    ...     env, field1, field2, metric="euclidean"
    ... )
    >>> print(f"Euclidean EMD: {emd_euclidean:.3f}")  # doctest: +SKIP
    """
    from scipy.optimize import linprog
    from scipy.spatial.distance import pdist, squareform

    # Validate inputs
    if len(firing_rate_1) != len(firing_rate_2):
        raise ValueError(
            f"firing_rate arrays must have same length, got {len(firing_rate_1)} and {len(firing_rate_2)}"
        )

    if len(firing_rate_1) != env.n_bins:
        raise ValueError(
            f"firing_rate arrays must match env.n_bins ({env.n_bins}), got {len(firing_rate_1)}"
        )

    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")
    if metric == "euclidean":
        require_cartesian(
            env.coordinate_system,
            "compute_field_emd(metric='euclidean')",
            why="Its ground distance is the straight line between bin centers.",
            fix=(
                "use metric='geodesic', whose ground distance follows physical graph "
                "edge lengths"
            ),
        )

    # Compare only bins observed in both maps. Unsampled (NaN) bins are
    # missing, not zero rate: filling them with zero would make mass in one
    # map move onto bins the other session never sampled.
    firing_rate_1 = np.asarray(firing_rate_1, dtype=np.float64)
    firing_rate_2 = np.asarray(firing_rate_2, dtype=np.float64)
    common_bins = np.flatnonzero(
        np.isfinite(firing_rate_1) & np.isfinite(firing_rate_2)
    )
    firing_rate_1 = firing_rate_1[common_bins]
    firing_rate_2 = firing_rate_2[common_bins]

    # Normalize distributions if requested
    if normalize:
        sum1 = np.sum(firing_rate_1)
        sum2 = np.sum(firing_rate_2)

        if sum1 == 0 or sum2 == 0:
            warnings.warn(
                "One or both distributions have zero total mass on the bins "
                "finite in both maps. Returning NaN.",
                category=UserWarning,
                stacklevel=2,
            )
            return np.nan

        firing_rate_1 = firing_rate_1 / sum1
        firing_rate_2 = firing_rate_2 / sum2
    else:
        # Check that unnormalized distributions have equal total mass
        sum1 = np.sum(firing_rate_1)
        sum2 = np.sum(firing_rate_2)
        if not np.isclose(sum1, sum2, rtol=1e-6):
            raise ValueError(
                f"Unnormalized distributions must have equal total mass on the "
                f"bins finite in both maps. Got {sum1:.6f} and {sum2:.6f}. Set "
                f"normalize=True to auto-normalize."
            )

    # Filter to bins with non-zero mass in either distribution
    # This reduces problem size significantly for sparse fields
    nonzero_mask = (firing_rate_1 > 0) | (firing_rate_2 > 0)
    n_active = np.sum(nonzero_mask)

    if n_active == 0:
        # Both distributions are all zeros
        return 0.0

    if n_active == 1:
        # Only one bin has mass - EMD is zero if same bin, undefined otherwise
        # Since distributions are normalized, must be same bin
        return 0.0

    # Get active bins and their distributions
    active_bins = common_bins[nonzero_mask]
    p = firing_rate_1[nonzero_mask]
    q = firing_rate_2[nonzero_mask]

    # Compute distance matrix between active bins
    if metric == "euclidean":
        # Euclidean distance between bin centers using scipy (vectorized)
        positions = env.bin_centers[active_bins]
        # pdist computes condensed distance matrix, squareform converts to full
        dist_matrix = squareform(pdist(positions, metric="euclidean"))

    else:  # metric == "geodesic"
        from neurospatial.ops.distance import geodesic_distance_matrix

        full_geodesic = geodesic_distance_matrix(env.connectivity, weight="distance")
        dist_matrix = full_geodesic[np.ix_(active_bins, active_bins)]

    # Mass cannot travel between parts of the environment with no path
    # between them. Label each active bin by the first active bin it can
    # reach (reachability is an equivalence relation); if any part holds
    # different mass in the two maps, no transport plan exists.
    reachable = np.isfinite(dist_matrix)
    component = np.argmax(reachable, axis=1)
    mass_1 = np.bincount(component, weights=p, minlength=len(p))
    mass_2 = np.bincount(component, weights=q, minlength=len(q))
    if not np.allclose(mass_1, mass_2, rtol=1e-9, atol=1e-12 * np.sum(p)):
        return np.inf

    # Solve the optimal transport problem using linear programming
    # Variables: T[i,j] = mass transported from bin i (source) to bin j (target)
    # Objective: minimize sum(T[i,j] * dist[i,j])
    # Constraints:
    #   - sum_j T[i,j] = p[i]  (all mass from source i is transported)
    #   - sum_i T[i,j] = q[j]  (all mass to target j is received)
    #   - T[i,j] >= 0, and T[i,j] = 0 where no path joins i and j

    n = len(p)

    # Flatten distance matrix for objective function (unreachable pairs carry
    # no mass, so their cost is irrelevant)
    c = np.where(reachable, dist_matrix, 0.0).flatten()
    upper = np.where(reachable.flatten(), np.inf, 0.0)

    # Equality constraints: Ax = b
    # Row constraints: sum over j of T[i,j] = p[i]
    # Column constraints: sum over i of T[i,j] = q[j]
    #
    # Vectorized construction using Kronecker products:
    # - Row constraints: each row i has 1s in columns [i*n, i*n+1, ..., i*n+n-1]
    #   This is kron(I_n, ones(1,n)) = block diagonal of row vectors
    # - Column constraints: each column j has 1s in columns [j, n+j, 2n+j, ...]
    #   This is kron(ones(1,n), I_n) = tiled identity matrices
    row_constraints = np.kron(np.eye(n), np.ones((1, n)))  # Shape: (n, n*n)
    col_constraints = np.kron(np.ones((1, n)), np.eye(n))  # Shape: (n, n*n)
    a_eq = np.vstack([row_constraints, col_constraints])  # Shape: (2n, n*n)

    b_eq = np.concatenate([p, q])

    # Solve linear program
    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=np.column_stack([np.zeros_like(upper), upper]),
        method="highs",
    )

    if not result.success:
        warnings.warn(
            f"Optimal transport optimization failed: {result.message}. Returning NaN.",
            category=UserWarning,
            stacklevel=2,
        )
        return np.nan

    # EMD is the optimal cost
    emd = float(result.fun)
    return emd
