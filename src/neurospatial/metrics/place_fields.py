"""
Place field detection and single-cell spatial metrics.

This module implements standard neuroscience metrics for place cell analysis,
validated against field-standard packages (opexebo, neurocode, buzcode).

Key metrics:
- Place field detection (iterative peak-based, neurocode approach)
- Field size and centroid
- Skaggs spatial information (bits/spike)
- Sparsity (Skaggs et al. 1996)
- Field stability (correlation between sessions)

References
----------
.. [1] O'Keefe & Dostrovsky (1971). The hippocampus as a spatial map.
       Brain Research 34(1).
.. [2] Skaggs et al. (1996). Theta phase precession in hippocampal neuronal
       populations and the compression of temporal sequences. Hippocampus 6(2).
.. [3] Muller & Kubie (1989). The effects of changes in the environment on
       the spatial firing of hippocampal complex-spike cells. J Neurosci 9(1).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from neurospatial import Environment


def detect_place_fields(
    firing_rate: NDArray[np.float64],
    env: Environment,
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    max_mean_rate: float = 10.0,
    detect_subfields: bool = True,
) -> list[NDArray[np.int64]]:
    """
    Detect place fields using iterative peak-based approach (neurocode method).

    This implements the field-standard algorithm used by neurocode (AyA Lab)
    with support for subfield discrimination and interneuron exclusion.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz) from neuron.
    env : Environment
        Spatial environment for binning.
    threshold : float, default=0.2
        Fraction of peak rate for field boundary detection (0-1).
        Standard value is 0.2 (20% of peak).
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9 bins.
    max_mean_rate : float, default=10.0
        Maximum mean firing rate (Hz). Neurons exceeding this are excluded
        as putative interneurons (vandermeerlab convention).
    detect_subfields : bool, default=True
        If True, recursively detect subfields within large fields using
        higher thresholds. This discriminates coalescent place fields.

    Returns
    -------
    fields : list of arrays
        List of place fields, where each field is a 1D array of bin indices
        (integers) belonging to that field. Empty list if no fields detected.

    Notes
    -----
    **Algorithm (neurocode approach)**:

    1. **Interneuron exclusion**: If mean rate > max_mean_rate, return no fields
    2. **Peak detection**: Find global maximum in firing rate map
    3. **Field segmentation**: Threshold at fraction of peak to define boundary
    4. **Connected component**: Extract bins above threshold connected to peak
    5. **Size filtering**: Discard fields smaller than min_size
    6. **Subfield recursion**: If detect_subfields=True, recursively apply
       higher thresholds (0.5, 0.7) to discriminate coalescent fields
    7. **Iteration**: Remove detected field bins and repeat until no peaks remain

    **Interneuron exclusion**: Following vandermeerlab convention, neurons with
    mean firing rate > 10 Hz are excluded as putative interneurons. Pyramidal
    cells (place cells) typically fire at 0.5-5 Hz.

    **Subfield detection**: When two place fields are close together, they may
    appear as a single broad field at low thresholds. Recursive thresholding
    at 0.5× and 0.7× peak discriminates true subfields.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.place_fields import detect_place_fields
    >>> # Create synthetic place cell
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # Add Gaussian place field at center
    >>> for i in range(env.n_bins):
    ...     dist = np.linalg.norm(env.bin_centers[i])
    ...     firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(firing_rate, env)
    >>> len(fields)  # doctest: +SKIP
    1

    See Also
    --------
    field_size : Compute area of place field
    field_centroid : Compute weighted center of mass
    skaggs_information : Spatial information content

    References
    ----------
    .. [1] neurocode repository (AyA Lab, Cornell): FindPlaceFields.m
    .. [2] Wilson & McNaughton (1993). Dynamics of hippocampal ensemble code
           for space. Science 261(5124).

    """
    # Validate inputs
    if firing_rate.shape[0] != env.n_bins:
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"env.n_bins ({env.n_bins})"
        )

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # Set default min_size
    if min_size is None:
        min_size = 9  # Standard minimum (3×3 bins for 2D)

    # Interneuron exclusion
    mean_rate = np.nanmean(firing_rate)
    if mean_rate > max_mean_rate:
        return []  # Putative interneuron

    # Make a copy to modify during iteration
    rate_map = firing_rate.copy()
    fields = []

    # Iteratively find fields
    while True:
        # Handle all-NaN case
        if not np.any(np.isfinite(rate_map)):
            break  # No valid values remaining

        # Find peak
        peak_idx = int(np.nanargmax(rate_map))
        peak_rate = rate_map[peak_idx]

        # Check if peak is meaningful
        if peak_rate <= 0 or not np.isfinite(peak_rate):
            break

        # Threshold at fraction of peak
        threshold_rate = peak_rate * threshold

        # Find bins above threshold
        above_threshold = rate_map >= threshold_rate

        # Extract connected component containing peak
        field_bins = _extract_connected_component(peak_idx, above_threshold, env)

        # Check minimum size
        if len(field_bins) < min_size:
            # Remove this small field and continue
            rate_map[field_bins] = 0
            continue

        # Check for subfields (recursive thresholding)
        if detect_subfields and len(field_bins) > min_size * 2:
            # Try higher thresholds to discriminate subfields
            subfields = _detect_subfields(
                firing_rate[field_bins], field_bins, peak_rate, env, min_size
            )
            if len(subfields) > 1:
                # Found subfields - add them separately
                fields.extend(subfields)
            else:
                # No subfields - add as single field
                fields.append(field_bins)
        else:
            # Add field
            fields.append(field_bins)

        # Remove field bins from rate map
        rate_map[field_bins] = 0

        # Check if any meaningful peaks remain
        if np.nanmax(rate_map) < threshold_rate:
            break

    return fields


def _extract_connected_component(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """
    Extract connected component of bins from seed using graph connectivity.

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component.

    """
    # Flood fill using graph connectivity
    component_set = {seed_idx}
    frontier = [seed_idx]

    while frontier:
        current = frontier.pop(0)
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


def _detect_subfields(
    field_rates: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    peak_rate: float,
    env: Environment,
    min_size: int,
) -> list[NDArray[np.int64]]:
    """
    Recursively detect subfields using higher thresholds.

    Parameters
    ----------
    field_rates : array
        Firing rates within field bins.
    field_bins : array
        Bin indices of field.
    peak_rate : float
        Peak firing rate in field.
    env : Environment
        Spatial environment.
    min_size : int
        Minimum field size.

    Returns
    -------
    subfields : list of arrays
        List of subfield bin indices. If only one subfield found,
        returns list with original field.

    """
    # Try thresholds: 0.5 and 0.7 of peak
    subfield_thresholds = [0.5, 0.7]

    for thresh in subfield_thresholds:
        threshold_rate = peak_rate * thresh
        above_threshold = field_rates >= threshold_rate

        # Find connected components
        subfields = []
        remaining_mask = above_threshold.copy()

        while remaining_mask.any():
            # Find a seed
            seed_local_idx = np.where(remaining_mask)[0][0]
            seed_global_idx = field_bins[seed_local_idx]

            # Build mask in global coordinates
            global_mask = np.zeros(env.n_bins, dtype=bool)
            global_mask[field_bins[above_threshold]] = True

            # Extract component
            component_global = _extract_connected_component(
                seed_global_idx, global_mask, env
            )

            if len(component_global) >= min_size:
                subfields.append(component_global)

            # Remove from remaining mask
            for bin_idx in component_global:
                # Find local index
                local_indices = np.where(field_bins == bin_idx)[0]
                if len(local_indices) > 0:
                    remaining_mask[local_indices[0]] = False

        # If found multiple subfields, return them
        if len(subfields) > 1:
            return subfields

    # No subfields found
    return [field_bins]


def field_size(
    field_bins: NDArray[np.int64],
    env: Environment,
) -> float:
    """
    Compute field size (area) in physical units.

    Parameters
    ----------
    field_bins : array
        Bin indices comprising the field.
    env : EnvironmentProtocol
        Spatial environment.

    Returns
    -------
    size : float
        Field area in squared physical units (e.g., cm²).

    Notes
    -----
    Size is computed as the sum of individual bin areas. For regular grids,
    each bin has area ≈ bin_size². For irregular graphs, area is estimated
    from Voronoi cell volumes.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.place_fields import field_size
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> size = field_size(field_bins, env)
    >>> size > 0
    True

    """
    # Get bin sizes (property, not method)
    bin_sizes = env.bin_sizes

    # Sum areas of field bins
    total_size = np.sum(bin_sizes[field_bins])

    return float(total_size)


def field_centroid(
    firing_rate: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute firing-rate-weighted centroid of place field.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    field_bins : array
        Bin indices comprising the field.
    env : EnvironmentProtocol
        Spatial environment.

    Returns
    -------
    centroid : array, shape (n_dims,)
        Weighted center of mass in physical coordinates.

    Notes
    -----
    Centroid is computed as the firing-rate-weighted mean position:

    .. math::

        \\mathbf{c} = \\frac{\\sum_i r_i \\mathbf{p}_i}{\\sum_i r_i}

    where :math:`r_i` is firing rate and :math:`\\mathbf{p}_i` is position
    of bin :math:`i`.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics.place_fields import field_centroid
    >>> positions = np.random.randn(1000, 2) * 10
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> firing_rate = np.random.rand(env.n_bins) * 5
    >>> field_bins = np.array([0, 1, 2, 3, 4])
    >>> centroid = field_centroid(firing_rate, field_bins, env)
    >>> centroid.shape
    (2,)

    """
    # Get positions and rates for field bins
    field_positions = env.bin_centers[field_bins]
    field_rates = firing_rate[field_bins]

    # Compute weighted centroid
    total_rate = np.sum(field_rates)
    centroid: NDArray[np.float64]
    if total_rate == 0:
        # Unweighted centroid if no firing
        centroid = field_positions.mean(axis=0)
    else:
        centroid = np.sum(field_positions * field_rates[:, None], axis=0) / total_rate

    return centroid


def skaggs_information(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """
    Compute Skaggs spatial information (bits per spike).

    Spatial information quantifies how much information each spike conveys
    about the animal's spatial location.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    occupancy : array, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).
    base : float, default=2.0
        Logarithm base. Use 2.0 for bits, np.e for nats.

    Returns
    -------
    information : float
        Spatial information in bits per spike (if base=2.0).
        Returns 0.0 if mean rate is zero.

    Notes
    -----
    **Formula (Skaggs et al. 1993)**:

    .. math::

        I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log \\left( \\frac{r_i}{\\bar{r}} \\right)

    where :math:`p_i` is occupancy probability, :math:`r_i` is firing rate
    in bin :math:`i`, and :math:`\\bar{r}` is mean firing rate.

    **Interpretation**:
    - Place cells typically have 1-3 bits/spike
    - Higher values indicate more spatially selective firing
    - Zero information means uniform firing (no spatial information)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.place_fields import skaggs_information
    >>> # Uniform firing → zero information
    >>> firing_rate = np.ones(100) * 3.0
    >>> occupancy = np.ones(100) / 100
    >>> info = skaggs_information(firing_rate, occupancy)
    >>> np.abs(info) < 1e-6  # Should be ~0
    True

    References
    ----------
    .. [1] Skaggs et al. (1993). An information-theoretic approach to
           deciphering the hippocampal code. NIPS.

    """
    # Normalize occupancy to probability
    occupancy_prob = occupancy / np.sum(occupancy)

    # Mean firing rate
    mean_rate = np.sum(occupancy_prob * firing_rate)

    if mean_rate == 0:
        return 0.0

    # Compute information
    information = 0.0
    for i in range(len(firing_rate)):
        if occupancy_prob[i] > 0 and firing_rate[i] > 0:
            ratio = firing_rate[i] / mean_rate
            information += occupancy_prob[i] * ratio * np.log(ratio) / np.log(base)

    return float(information)


def sparsity(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """
    Compute sparsity of spatial firing (Skaggs et al. 1996).

    Sparsity measures what fraction of the environment elicits significant
    firing. Lower values indicate sparser, more selective place fields.

    Parameters
    ----------
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz).
    occupancy : array, shape (n_bins,)
        Occupancy probability (normalized to sum to 1).

    Returns
    -------
    sparsity : float
        Sparsity value in range [0, 1]. Lower values indicate sparser firing.

    Notes
    -----
    **Formula (Skaggs et al. 1996)**:

    .. math::

        S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

    where :math:`p_i` is occupancy probability and :math:`r_i` is firing rate.

    **Interpretation**:
    - Range: [0, 1]
    - Low sparsity (0.1-0.3): Sparse, selective place field
    - High sparsity (~1.0): Uniform firing throughout environment
    - Typical place cells: 0.1-0.3

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.place_fields import sparsity
    >>> # Uniform firing → high sparsity
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100) / 100
    >>> spars = sparsity(firing_rate, occupancy)
    >>> spars > 0.9  # Close to 1
    True

    References
    ----------
    .. [1] Skaggs et al. (1996). Theta phase precession in hippocampal
           neuronal populations. Hippocampus 6(2).

    """
    # Normalize occupancy to probability
    occupancy_prob = occupancy / np.sum(occupancy)

    # Compute sparsity
    numerator = np.sum(occupancy_prob * firing_rate) ** 2
    denominator = np.sum(occupancy_prob * firing_rate**2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def field_stability(
    rate_map_1: NDArray[np.float64],
    rate_map_2: NDArray[np.float64],
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """
    Compute stability between two firing rate maps (correlation).

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
    >>> from neurospatial.metrics.place_fields import field_stability
    >>> # Identical maps → perfect correlation
    >>> rate_map = np.random.rand(100) * 5
    >>> stability = field_stability(rate_map, rate_map, method="pearson")
    >>> np.abs(stability - 1.0) < 1e-6
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
    """
    Compute spatial coherence of a firing rate map.

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
    >>> from neurospatial.metrics import rate_map_coherence
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
    >>>
    >>> # Random noise (low coherence)
    >>> firing_rate_noisy = np.random.rand(env.n_bins) * 5.0
    >>> coherence_noisy = rate_map_coherence(firing_rate_noisy, env)
    >>> print(f"Noisy field coherence: {coherence_noisy:.3f}")  # doctest: +SKIP
    Noisy field coherence: 0.120

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
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
    from neurospatial.primitives import neighbor_reduce

    neighbor_means = neighbor_reduce(
        firing_rate,
        env,
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


def selectivity(
    firing_rate: NDArray[np.float64],
    occupancy: NDArray[np.float64],
) -> float:
    """
    Compute spatial selectivity (peak rate / mean rate).

    Selectivity measures how spatially selective a cell's firing is. Higher
    values indicate the cell fires strongly in a small region and weakly
    elsewhere. A value of 1.0 indicates uniform firing throughout the
    environment.

    This metric is used in opexebo and provides a simple, interpretable measure
    of place field quality.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map (Hz or spikes/second).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy probability (normalized to sum to 1).

    Returns
    -------
    float
        Selectivity value, always >= 1.0. Returns NaN if:
        - Mean rate is zero (division by zero)
        - All firing rates are NaN
        Returns infinity if peak rate is positive but mean rate is zero.

    Notes
    -----
    **Formula**:

    .. math::

        S = \\frac{r_{\\text{peak}}}{\\bar{r}}

    where :math:`r_{\\text{peak}}` is the maximum firing rate and
    :math:`\\bar{r}` is the occupancy-weighted mean firing rate.

    **Interpretation**:

    - **Selectivity = 1.0**: Uniform firing (peak equals mean)
    - **Selectivity = 2-5**: Moderately selective place field
    - **Selectivity > 10**: Highly selective place field (fires in small region)

    **NaN handling**: NaN values in firing_rate are excluded from peak and mean
    calculations. Occupancy is renormalized to valid bins.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import selectivity
    >>>
    >>> # Uniform firing → selectivity = 1.0
    >>> firing_rate = np.ones(100) * 5.0
    >>> occupancy = np.ones(100) / 100
    >>> select = selectivity(firing_rate, occupancy)
    >>> print(f"Uniform: {select:.2f}")  # doctest: +SKIP
    Uniform: 1.00
    >>>
    >>> # Highly selective cell (fires in one bin)
    >>> firing_rate_selective = np.zeros(100)
    >>> firing_rate_selective[50] = 100.0
    >>> select_high = selectivity(firing_rate_selective, occupancy)
    >>> print(f"Selective: {select_high:.1f}")  # doctest: +SKIP
    Selective: 100.0

    See Also
    --------
    skaggs_information : Spatial information (bits/spike)
    sparsity : Spatial sparsity
    rate_map_coherence : Spatial coherence

    References
    ----------
    .. [1] opexebo package (Moser Lab):
           https://github.com/kavli-ntnu/opexebo
    """
    # Handle NaN values
    valid_mask = np.isfinite(firing_rate) & np.isfinite(occupancy)

    if not np.any(valid_mask):
        # All NaN
        return np.nan

    # Get valid values
    firing_rate_valid = firing_rate[valid_mask]
    occupancy_valid = occupancy[valid_mask]

    # Normalize occupancy to probability
    occupancy_prob = occupancy_valid / np.sum(occupancy_valid)

    # Peak firing rate
    peak_rate = np.max(firing_rate_valid)

    # Mean firing rate (occupancy-weighted)
    mean_rate = np.sum(occupancy_prob * firing_rate_valid)

    # Compute selectivity
    if mean_rate == 0:
        # Division by zero
        if peak_rate > 0:
            return np.inf
        else:
            return np.nan

    selectivity_value = peak_rate / mean_rate

    return float(selectivity_value)
