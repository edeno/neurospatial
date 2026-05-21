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
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

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
    env: EnvironmentProtocol,
    field_bins: NDArray[np.int64],
) -> float:
    """Compute field size (area) in physical units.

    Parameters
    ----------
    env : EnvironmentProtocol
        Spatial environment.
    field_bins : array
        Bin indices comprising the field.

    Returns
    -------
    size : float
        Field area in squared physical units (e.g., cm^2).

    Notes
    -----
    Size is computed as the sum of individual bin areas. For regular grids,
    each bin has area bin_size^2. For irregular graphs, area is estimated
    from Voronoi cell volumes.

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


def rate_map_centroid(
    env: Environment,
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    *,
    method: Literal["euclidean", "geodesic"] = "euclidean",
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

    Notes
    -----
    For ``method="euclidean"``, centroid is computed as the firing-rate-weighted
    mean position:

    .. math::

        \\mathbf{c} = \\frac{\\sum_i r_i \\mathbf{p}_i}{\\sum_i r_i}

    where :math:`r_i` is firing rate and :math:`\\mathbf{p}_i` is position
    of bin :math:`i`.

    For ``method="geodesic"``, the centroid is the bin that minimizes the
    weighted sum of graph distances:

    .. math::

        c = \\arg\\min_j \\sum_i r_i \\cdot d_G(i, j)

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
    import networkx as nx

    # Get positions and rates for field bins
    field_positions = env.bin_centers[field_bins]
    field_rates = firing_rate[field_bins]

    centroid: NDArray[np.float64]

    if method == "euclidean":
        # Compute weighted centroid in Euclidean space
        total_rate = np.sum(field_rates)
        if total_rate == 0:
            # Unweighted centroid if no firing
            centroid = field_positions.mean(axis=0)
        else:
            centroid = (
                np.sum(field_positions * field_rates[:, None], axis=0) / total_rate
            )

    elif method == "geodesic":
        # Compute weighted medoid using graph distances
        # Find bin that minimizes sum of (rate * graph_distance) to all other bins

        if len(field_bins) == 1:
            # Single bin - return its position
            centroid = field_positions[0]
        else:
            # Get graph and compute distances between field bins
            graph = env.connectivity

            # Compute weighted cost for each candidate bin
            min_cost = np.inf
            best_bin_idx = 0

            for j, candidate_bin in enumerate(field_bins):
                cost = 0.0
                for i, source_bin in enumerate(field_bins):
                    if i != j:
                        # Get graph distance
                        try:
                            dist = nx.shortest_path_length(
                                graph, source_bin, candidate_bin, weight="distance"
                            )
                        except nx.NetworkXNoPath:
                            # No path - use large penalty
                            dist = np.inf
                        cost += field_rates[i] * dist

                if cost < min_cost:
                    min_cost = cost
                    best_bin_idx = j

            centroid = field_positions[best_bin_idx]

    else:  # pragma: no cover
        # Runtime safety check (type system enforces at compile time)
        msg = f"Unknown method: {method}. Must be 'euclidean' or 'geodesic'."  # type: ignore[unreachable]
        raise ValueError(msg)

    return centroid


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
        more stable place fields.

    Notes
    -----
    **Interpretation**:
    - High stability (r > 0.7): Reliable, stable place field
    - Medium stability (0.3 < r < 0.7): Moderately stable
    - Low stability (r < 0.3): Unstable or remapped field

    **Edge Cases**:
    - Constant arrays (zero variance) return NaN (correlation undefined)
    - Arrays with <2 valid bins return 0.0

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

    if len(map1_clean) < 2:
        return 0.0

    # Check for constant arrays (zero variance) - correlation undefined
    if np.std(map1_clean) == 0 or np.std(map2_clean) == 0:
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
    firing_rate: NDArray[np.float64],
    env: Environment,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """Compute spatial coherence of a firing rate map.

    Spatial coherence measures the smoothness of spatial firing patterns by
    computing the correlation between each bin's firing rate and the mean rate
    of its spatial neighbors. High coherence indicates smooth, spatially
    structured firing. Low coherence indicates noisy or scattered firing.

    This metric was introduced by Muller & Kubie (1989) to assess the quality
    of place field representations.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second).
    env : EnvironmentProtocol
        Spatial environment containing bin centers and connectivity.
    method : {'pearson', 'spearman'}, optional
        Correlation method. Default is 'pearson'.
        - 'pearson': Pearson correlation (linear relationship)
        - 'spearman': Spearman correlation (monotonic relationship)

    Returns
    -------
    float
        Spatial coherence in range [-1, 1]. Returns NaN if:
        - All firing rates are zero or constant (no variance)
        - All rates are NaN
        - Insufficient valid bins after NaN removal

    Notes
    -----
    **Algorithm**:

    1. For each bin i, compute mean firing rate of neighbors: m_i = mean(r_j) for j in neighbors(i)
    2. Compute correlation between bin rates r_i and neighbor means m_i
    3. Coherence = corr(r, m)

    **Interpretation**:

    - **High coherence (> 0.7)**: Smooth, spatially structured firing (good place field)
    - **Medium coherence (0.3-0.7)**: Some spatial structure but with noise
    - **Low coherence (< 0.3)**: Noisy, poorly defined spatial firing

    **Graph-based approach**:

    This implementation uses `env.connectivity` to determine spatial neighbors,
    making it applicable to irregular environments and graphs with obstacles.
    For regular grids, results should match Muller & Kubie (1989) approach.

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
    >>> coherence_smooth = rate_map_coherence(firing_rate_smooth, env)
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

    # Compute mean of neighbors for each bin using neighbor_reduce
    # Use nanmean to handle NaN values in neighbors
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

    # Compute correlation
    if method == "pearson":
        coherence, _ = stats.pearsonr(bin_rates, neighbor_rate_means)
    elif method == "spearman":
        coherence, _ = stats.spearmanr(bin_rates, neighbor_rate_means)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson' or 'spearman'.")

    return float(coherence)


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
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    env: Environment,
) -> dict[str, float]:
    """Compute geometric shape metrics for a place field.

    Analyzes the spatial geometry of a place field, including eccentricity,
    orientation, and extent along principal axes. Useful for characterizing
    field morphology and detecting elongated or circular fields.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    field_bins : NDArray[np.int64], shape (n_field_bins,)
        Indices of bins belonging to the place field.
    env : Environment
        Spatial environment for bin positions. Must be 2D (n_dims == 2).

    Returns
    -------
    dict[str, float]
        Dictionary with shape metrics:
        - 'eccentricity': float in [0, 1], where 0 = circular, 1 = linear
        - 'major_axis_length': float, extent along major axis (same units as env)
        - 'minor_axis_length': float, extent along minor axis (same units as env)
        - 'orientation': float, angle of major axis in radians [-pi/2, pi/2]
        - 'area': float, spatial extent of field (number of bins)

        Returns dict with NaN values if field is empty or environment is not 2D.

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
    >>> shape = field_shape_metrics(firing_rate, field_bins, env)
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

    # Axis lengths (2 standard deviations = ~95% of data)
    major_axis = 2 * np.sqrt(lambda_max)
    minor_axis = 2 * np.sqrt(lambda_min)

    # Orientation (angle of major axis)
    # eigenvectors[:, 0] is the major axis direction
    orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Area (number of bins)
    area = float(len(field_bins))

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
    env_1: Environment,
    firing_rate_1: NDArray[np.float64],
    field_bins_1: NDArray[np.int64],
    env_2: Environment,
    firing_rate_2: NDArray[np.float64],
    field_bins_2: NDArray[np.int64],
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
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
        the rate-weighted centroids. ``"geodesic"`` is the shortest path
        along the connectivity graph and requires ``env_1`` and ``env_2``
        to be the same environment or aligned environments with compatible
        connectivity; geodesic distance respects barriers and boundaries.

    Returns
    -------
    float
        Distance between field centroids in spatial units (same units as environment).
        Returns NaN if either field is empty or centroid calculation fails.

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
    estimate_transform and apply_transform_to_environment) to account for
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
    # Compute centroids for both fields
    centroid_1 = rate_map_centroid(env_1, firing_rate_1, field_bins_1)
    centroid_2 = rate_map_centroid(env_2, firing_rate_2, field_bins_2)

    # Check for NaN centroids
    if np.any(np.isnan(centroid_1)) or np.any(np.isnan(centroid_2)):
        return np.nan

    if metric not in ("euclidean", "geodesic"):
        raise ValueError(f"metric must be 'euclidean' or 'geodesic', got '{metric}'")

    if metric == "geodesic":
        # Geodesic distance using environment connectivity
        # Validate that centroids fall within environment bounds
        bin_1 = env_1.bin_at(centroid_1.reshape(1, -1))[0]
        bin_2 = env_2.bin_at(centroid_2.reshape(1, -1))[0]

        # Check if bins are valid (centroids in bounds)
        if bin_1 < 0 or bin_2 < 0:
            warnings.warn(
                "One or both centroids fall outside environment bounds. "
                "Cannot compute geodesic distance. Returning NaN.",
                category=UserWarning,
                stacklevel=2,
            )
            return np.nan

        # Check if environments are compatible for geodesic distance
        if env_1 is not env_2 and env_1.n_bins != env_2.n_bins:
            # Different environment objects - check if they have same bins
            warnings.warn(
                f"Environments have different number of bins ({env_1.n_bins} vs {env_2.n_bins}). "
                "Geodesic distance requires compatible environments. Falling back to Euclidean distance.",
                category=UserWarning,
                stacklevel=2,
            )
            # Fall back to Euclidean
            distance = float(np.linalg.norm(centroid_1 - centroid_2))
            return distance

        # Compute geodesic distance using centroids (coordinates), not bin indices
        try:
            geodesic_dist = cast("EnvironmentProtocol", env_1).distance_between(
                centroid_1, centroid_2
            )
            return float(geodesic_dist)
        except Exception as e:
            warnings.warn(
                f"Failed to compute geodesic distance: {e}. Falling back to Euclidean distance.",
                category=UserWarning,
                stacklevel=2,
            )
            # Fall back to Euclidean
            distance = float(np.linalg.norm(centroid_1 - centroid_2))
            return distance
    else:
        # Euclidean distance between centroids
        distance = float(np.linalg.norm(centroid_1 - centroid_2))
        return distance


def compute_field_emd(
    env: Environment,
    firing_rate_1: NDArray[np.float64],
    firing_rate_2: NDArray[np.float64],
    *,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
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
    firing_rate_1 : NDArray[np.float64], shape (n_bins,)
        First firing rate distribution across spatial bins.
    firing_rate_2 : NDArray[np.float64], shape (n_bins,)
        Second firing rate distribution across spatial bins.
    env : Environment
        Spatial environment defining bin positions and connectivity.
    metric : str, default="euclidean"
        Distance metric to use. Options:
        - "euclidean": Straight-line distance between bin centers
        - "geodesic": Shortest path distance through connectivity graph
    normalize : bool, default=True
        If True, normalize distributions to sum to 1.0 before computing EMD.
        If False, use raw firing rates (distributions must already sum to equal values).

    Returns
    -------
    emd : float
        Earth Mover's Distance between the two distributions.
        - For normalized distributions: unitless distance in [0, inf)
        - For unnormalized distributions: cost in units of (rate x distance)
        - Returns NaN if distributions cannot be computed (e.g., all zeros)

    Raises
    ------
    ValueError
        If firing_rate arrays have different lengths.
        If firing_rate arrays don't match env.n_bins.
        If metric is not "euclidean" or "geodesic".
        If normalize=False and distributions have different total mass.

    Warns
    -----
    UserWarning
        If distributions contain NaN values (they will be set to zero).
        If normalized distributions have no mass (all zeros or NaN).

    See Also
    --------
    field_shift_distance : Distance between field centroids.

    Notes
    -----
    The Earth Mover's Distance is computed by solving the optimal transport problem.

    **Metric choice:**

    - **Euclidean**: Fast, works for any environment, but ignores barriers and walls.
      Use for simple open fields or when computational speed is critical.

    - **Geodesic**: Slower, respects environment structure (barriers, walls, connectivity).
      Use for complex environments like mazes, multi-room layouts, or non-convex arenas.

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
    >>> # Compute EMD with Euclidean distance
    >>> emd_euclidean = compute_field_emd(env, field1, field2, metric="euclidean")
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

    # Handle NaN values
    firing_rate_1 = firing_rate_1.copy()
    firing_rate_2 = firing_rate_2.copy()

    if np.any(~np.isfinite(firing_rate_1)) or np.any(~np.isfinite(firing_rate_2)):
        warnings.warn(
            "Firing rate distributions contain NaN values. Setting to zero.",
            category=UserWarning,
            stacklevel=2,
        )
        firing_rate_1[~np.isfinite(firing_rate_1)] = 0.0
        firing_rate_2[~np.isfinite(firing_rate_2)] = 0.0

    # Normalize distributions if requested
    if normalize:
        sum1 = np.sum(firing_rate_1)
        sum2 = np.sum(firing_rate_2)

        if sum1 == 0 or sum2 == 0:
            warnings.warn(
                "One or both distributions have zero total mass. Returning NaN.",
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
                f"Unnormalized distributions must have equal total mass. "
                f"Got {sum1:.6f} and {sum2:.6f}. Set normalize=True to auto-normalize."
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
    active_bins = np.where(nonzero_mask)[0]
    p = firing_rate_1[nonzero_mask]
    q = firing_rate_2[nonzero_mask]

    # Compute distance matrix between active bins
    if metric == "euclidean":
        # Euclidean distance between bin centers using scipy (vectorized)
        positions = env.bin_centers[active_bins]
        # pdist computes condensed distance matrix, squareform converts to full
        dist_matrix = squareform(pdist(positions, metric="euclidean"))

    else:  # metric == "geodesic"
        # Geodesic distance using precomputed full distance matrix (vectorized)
        from neurospatial.ops.distance import geodesic_distance_matrix

        # Compute full geodesic distance matrix once
        full_geodesic = geodesic_distance_matrix(
            env.connectivity, env.n_bins, weight="distance"
        )

        # Extract submatrix for active bins (vectorized indexing)
        dist_matrix = full_geodesic[np.ix_(active_bins, active_bins)]

        # Find disconnected pairs (inf distances) and replace with Euclidean
        disconnected_mask = np.isinf(dist_matrix)
        disconnected_count = np.sum(disconnected_mask) // 2  # Count unique pairs

        if disconnected_count > 0:
            # Compute Euclidean fallback distances for disconnected pairs
            positions = env.bin_centers[active_bins]
            euclidean_dist = squareform(pdist(positions, metric="euclidean"))
            dist_matrix = np.where(disconnected_mask, euclidean_dist, dist_matrix)

            n = len(active_bins)
            warnings.warn(
                f"Found {disconnected_count} disconnected bin pairs out of {n * (n - 1) // 2} total pairs. "
                f"Using Euclidean distance for disconnected pairs.",
                category=UserWarning,
                stacklevel=2,
            )

    # Solve the optimal transport problem using linear programming
    # Variables: T[i,j] = mass transported from bin i (source) to bin j (target)
    # Objective: minimize sum(T[i,j] * dist[i,j])
    # Constraints:
    #   - sum_j T[i,j] = p[i]  (all mass from source i is transported)
    #   - sum_i T[i,j] = q[j]  (all mass to target j is received)
    #   - T[i,j] >= 0

    n = len(p)

    # Flatten distance matrix for objective function
    c = dist_matrix.flatten()

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
        bounds=(0, None),
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
