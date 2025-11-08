"""
Grid cell metrics for spatial analysis.

Implements spatial autocorrelation and grid score (Sargolini et al., 2006) with
support for both regular 2D grids (FFT-based) and irregular graph topologies
(graph-based distance autocorrelation).

References
----------
Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M. B.,
    & Moser, E. I. (2006). Conjunctive representation of position, direction, and
    velocity in entorhinal cortex. Science, 312(5774), 758-762.
    https://doi.org/10.1126/science.1125572
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import fft, ndimage, stats

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol


def spatial_autocorrelation(
    firing_rate: NDArray[np.float64],
    env: EnvironmentProtocol,
    *,
    method: Literal["auto", "fft", "graph"] = "auto",
    max_distance: float | None = None,
    n_distance_bins: int = 50,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spatial autocorrelation of a firing rate map.

    Spatial autocorrelation measures the similarity between a cell's firing pattern
    and shifted versions of itself. For grid cells, this reveals the characteristic
    hexagonal periodicity. This implementation supports both regular 2D grids
    (FFT-based, opexebo-compatible) and irregular graph topologies (graph-based).

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second).
    env : EnvironmentProtocol
        Spatial environment containing bin centers and connectivity.
    method : {'auto', 'fft', 'graph'}, optional
        Autocorrelation computation method:
        - 'auto': Automatically select FFT for regular 2D grids, graph otherwise
        - 'fft': FFT-based 2D autocorrelation (requires regular 2D grid)
        - 'graph': Graph-based distance autocorrelation (works on any topology)
        Default is 'auto'.
    max_distance : float, optional
        Maximum distance for graph-based method (in physical units). If None,
        uses the environment's maximum extent. Ignored for FFT method.
    n_distance_bins : int, optional
        Number of distance bins for graph-based method. Default is 50.
        Ignored for FFT method.

    Returns
    -------
    autocorr : NDArray[np.float64], shape (height, width) OR tuple
        **FFT method** ('fft' or 'auto' on regular grid):
            Returns 2D autocorrelation map with same shape as environment grid.
            Center corresponds to zero lag. Values in [-1, 1].

        **Graph method** ('graph' or 'auto' on irregular topology):
            Returns tuple (distances, correlations):
            - distances : NDArray, shape (n_distance_bins,) - Distance bin centers
            - correlations : NDArray, shape (n_distance_bins,) - Autocorrelation at each distance

    Raises
    ------
    ValueError
        If firing_rate shape doesn't match env.n_bins.
        If method='fft' but environment is not a regular 2D grid.
        If all firing rates are NaN or constant.

    Notes
    -----
    **FFT Method** (regular 2D grids):

    1. Reshape firing rate to 2D grid shape
    2. Normalize: subtract mean, handle NaN
    3. Compute 2D autocorrelation via FFT convolution
    4. Normalize by unbiased variance estimate
    5. Output shape matches environment grid

    **Graph Method** (irregular topologies):

    1. Compute pairwise geodesic distances between all bins
    2. Bin distances into distance bins
    3. For each distance bin, compute correlation between firing rates
       of bin pairs at that distance
    4. Returns 1D distance-correlation profile

    **Interpretation**:

    - **Hexagonal grid cells**: Show 6-fold rotational symmetry in FFT method,
      periodic peaks at characteristic spacing in graph method
    - **Place cells**: Typically show single central peak, no periodicity
    - **Non-spatial cells**: Flat or noisy autocorrelation

    **When to use each method**:

    - Use **FFT method** for rectangular open-field recordings, compatibility
      with opexebo, or when you need 2D autocorrelogram for grid_score
    - Use **graph method** for irregular environments, tracks, or when you need
      distance-based characterization

    **Computational complexity**:

    - FFT method: O(n log n) where n = n_bins (very fast)
    - Graph method: O(n²) for distance computation (slower for large graphs)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import spatial_autocorrelation
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Create grid-like firing pattern
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # ... populate with grid cell firing pattern ...
    >>>
    >>> # FFT method (for regular 2D grid)
    >>> autocorr_2d = spatial_autocorrelation(firing_rate, env, method="fft")
    >>> print(autocorr_2d.shape)  # doctest: +SKIP
    (20, 20)
    >>>
    >>> # Graph method (works on any topology)
    >>> distances, correlations = spatial_autocorrelation(
    ...     firing_rate, env, method="graph", n_distance_bins=30
    ... )
    >>> print(distances.shape, correlations.shape)  # doctest: +SKIP
    (30,) (30,)

    See Also
    --------
    grid_score : Compute grid score from 2D autocorrelation
    periodicity_score : Graph-based periodicity metric

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    # Validate inputs
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )

    if method not in ("auto", "fft", "graph"):
        raise ValueError(f"method must be 'auto', 'fft', or 'graph', got '{method}'")

    if n_distance_bins <= 0:
        raise ValueError(f"n_distance_bins must be positive, got {n_distance_bins}")

    # Handle all-NaN
    if np.all(np.isnan(firing_rate)):
        raise ValueError("All firing rates are NaN")

    # Handle constant firing rate
    valid_rates = firing_rate[np.isfinite(firing_rate)]
    if len(valid_rates) > 0 and np.all(valid_rates == valid_rates[0]):
        raise ValueError(
            "All valid firing rates are constant. Autocorrelation undefined."
        )

    # Determine method
    if method == "auto":
        # Check if environment is a regular 2D grid
        method = _detect_grid_method(env)

    # Dispatch to appropriate method
    if method == "fft":
        return _spatial_autocorrelation_fft(firing_rate, env)
    else:  # method == "graph"
        return _spatial_autocorrelation_graph(
            firing_rate, env, max_distance=max_distance, n_distance_bins=n_distance_bins
        )


def _detect_grid_method(env: EnvironmentProtocol) -> Literal["fft", "graph"]:
    """
    Detect whether environment supports FFT-based autocorrelation.

    Parameters
    ----------
    env : EnvironmentProtocol
        Environment to check.

    Returns
    -------
    method : {'fft', 'graph'}
        'fft' if environment is a regular 2D grid, 'graph' otherwise.
    """
    # Check if environment has grid_shape attribute (regular grid)
    if not hasattr(env.layout, "grid_shape"):
        return "graph"

    grid_shape = env.layout.grid_shape

    # grid_shape can be None or tuple
    if grid_shape is None or len(grid_shape) != 2:
        return "graph"

    # Check if grid is reasonably regular (not too sparse)
    # If active_mask exists, check that most bins are active
    if hasattr(env.layout, "active_mask"):
        active_mask = env.layout.active_mask
        if active_mask is not None:
            # If less than 50% of bins are active, treat as irregular
            fill_fraction = np.sum(active_mask) / active_mask.size
            if fill_fraction < 0.5:
                return "graph"

    return "fft"


def _spatial_autocorrelation_fft(
    firing_rate: NDArray[np.float64],
    env: EnvironmentProtocol,
) -> NDArray[np.float64]:
    """
    Compute 2D spatial autocorrelation using FFT (for regular grids).

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map.
    env : EnvironmentProtocol
        Environment (must be regular 2D grid).

    Returns
    -------
    autocorr : NDArray[np.float64], shape (height, width)
        2D autocorrelation map.

    Raises
    ------
    ValueError
        If environment is not a regular 2D grid.
    """
    # Validate environment has grid structure
    if not hasattr(env.layout, "grid_shape"):
        raise ValueError(
            "FFT method requires environment with grid_shape attribute. "
            "Use method='graph' for irregular topologies."
        )

    grid_shape = env.layout.grid_shape

    if grid_shape is None:
        raise ValueError(
            "FFT method requires environment with grid_shape. "
            "Use method='graph' for irregular topologies."
        )

    if len(grid_shape) != 2:
        raise ValueError(
            f"FFT method requires 2D grid, got {len(grid_shape)}D grid. "
            "Use method='graph' for non-2D environments."
        )

    # Reshape to 2D grid (grid_shape is now guaranteed to be tuple[int, int])
    rate_map_2d = np.full(grid_shape, np.nan, dtype=np.float64)

    # Fill in values from firing_rate
    # Need to map flat bin indices to grid indices
    if hasattr(env.layout, "active_mask"):
        active_mask = env.layout.active_mask
        if active_mask is not None:
            rate_map_2d[active_mask] = firing_rate
        else:
            # Reshape flat firing rate to 2D grid
            reshaped = firing_rate.reshape(grid_shape)
            rate_map_2d[:, :] = reshaped
    else:
        # Reshape flat firing rate to 2D grid
        reshaped = firing_rate.reshape(grid_shape)
        rate_map_2d[:, :] = reshaped

    # Handle NaN: replace with 0 for computation
    # (NaN indicates unvisited bins, contribute zero to autocorrelation)
    valid_mask = np.isfinite(rate_map_2d)
    rate_map_clean = np.where(valid_mask, rate_map_2d, 0.0)

    # Normalize: subtract mean of valid bins
    valid_rates = rate_map_2d[valid_mask]
    if len(valid_rates) == 0:
        raise ValueError("No valid (non-NaN) firing rates")

    mean_rate = np.mean(valid_rates)
    rate_map_normalized = rate_map_clean - mean_rate

    # Compute 2D autocorrelation via FFT
    # autocorr(r) = IFFT(|FFT(rate_map)|^2)
    fft_rate = fft.fft2(rate_map_normalized)
    power_spectrum = np.abs(fft_rate) ** 2
    autocorr = fft.ifft2(power_spectrum).real

    # Shift zero lag to center
    autocorr = fft.fftshift(autocorr)

    # Normalize by variance and number of valid overlaps
    # This ensures autocorr[center] ≈ 1.0
    variance = np.var(valid_rates)
    if variance == 0:
        raise ValueError("Firing rate has zero variance")

    n_valid = np.sum(valid_mask)
    autocorr = autocorr / (variance * n_valid)

    # Return as NDArray[np.float64]
    return np.asarray(autocorr, dtype=np.float64)


def _spatial_autocorrelation_graph(
    firing_rate: NDArray[np.float64],
    env: EnvironmentProtocol,
    *,
    max_distance: float | None = None,
    n_distance_bins: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute distance-based autocorrelation using graph geodesic distances.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map.
    env : EnvironmentProtocol
        Environment.
    max_distance : float, optional
        Maximum distance to compute autocorrelation. If None, uses maximum
        extent of environment.
    n_distance_bins : int, optional
        Number of distance bins.

    Returns
    -------
    distances : NDArray[np.float64], shape (n_distance_bins,)
        Distance bin centers.
    correlations : NDArray[np.float64], shape (n_distance_bins,)
        Autocorrelation at each distance.
    """
    # Filter out NaN bins
    valid_bins = np.where(np.isfinite(firing_rate))[0]

    if len(valid_bins) < 2:
        raise ValueError("Need at least 2 valid (non-NaN) bins")

    valid_rates = firing_rate[valid_bins]

    # Compute pairwise geodesic distances between valid bins
    # For efficiency, compute all-pairs shortest path distances
    try:
        # Get subgraph of valid bins
        valid_bin_set = set(valid_bins.tolist())
        subgraph = env.connectivity.subgraph(valid_bin_set)

        # Compute all-pairs shortest path lengths
        distances_dict = dict(
            nx.all_pairs_dijkstra_path_length(subgraph, weight="distance")
        )
    except Exception as e:
        raise ValueError(f"Failed to compute graph distances: {e}") from e

    # Build pairwise distance and rate difference arrays
    pairwise_distances_list: list[float] = []
    rate_pairs_i_list: list[float] = []
    rate_pairs_j_list: list[float] = []

    for i, bin_i in enumerate(valid_bins):
        for j, bin_j in enumerate(valid_bins):
            if i >= j:  # Avoid duplicate pairs and self-pairs
                continue

            if int(bin_j) in distances_dict.get(int(bin_i), {}):
                dist = distances_dict[int(bin_i)][int(bin_j)]
                pairwise_distances_list.append(dist)
                rate_pairs_i_list.append(float(valid_rates[i]))
                rate_pairs_j_list.append(float(valid_rates[j]))

    if len(pairwise_distances_list) == 0:
        raise ValueError("No valid bin pairs found (graph may be disconnected)")

    pairwise_distances_array = np.array(pairwise_distances_list, dtype=np.float64)
    rate_pairs_i = np.array(rate_pairs_i_list, dtype=np.float64)
    rate_pairs_j = np.array(rate_pairs_j_list, dtype=np.float64)

    # Determine max distance
    if max_distance is None:
        max_distance = float(np.max(pairwise_distances_array))

    # Create distance bins
    distance_bin_edges = np.linspace(0, max_distance, n_distance_bins + 1)
    distance_bin_centers = (distance_bin_edges[:-1] + distance_bin_edges[1:]) / 2

    # Compute correlation for each distance bin
    correlations = np.full(n_distance_bins, np.nan)

    for d_idx in range(n_distance_bins):
        d_min = distance_bin_edges[d_idx]
        d_max = distance_bin_edges[d_idx + 1]

        # Find pairs in this distance bin
        in_bin = (pairwise_distances_array >= d_min) & (
            pairwise_distances_array < d_max
        )

        if np.sum(in_bin) < 2:
            # Not enough pairs for correlation
            continue

        rates_i_bin = rate_pairs_i[in_bin]
        rates_j_bin = rate_pairs_j[in_bin]

        # Compute Pearson correlation between rates at distance d
        if np.std(rates_i_bin) == 0 or np.std(rates_j_bin) == 0:
            # No variance, correlation undefined
            continue

        corr, _ = stats.pearsonr(rates_i_bin, rates_j_bin)
        correlations[d_idx] = corr

    return distance_bin_centers, correlations


def grid_score(
    autocorr_2d: NDArray[np.float64],
    *,
    inner_radius_fraction: float = 0.2,
    outer_radius_fraction: float = 0.5,
) -> float:
    """
    Compute grid score from 2D spatial autocorrelation.

    Grid score quantifies hexagonal periodicity by measuring rotational symmetry
    in the autocorrelogram. Hexagonal grids show high correlation at 60° and 120°
    rotations (hexagon vertices) and low correlation at 30°, 90°, 150° rotations.
    Implements the algorithm from Sargolini et al. (2006).

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (height, width)
        2D spatial autocorrelation map (from spatial_autocorrelation with method='fft').
        Should be centered (zero lag at center).
    inner_radius_fraction : float, optional
        Inner radius of annular region as fraction of image semi-axis.
        Default is 0.2 (20% of half-width). Must be in (0, 1).
    outer_radius_fraction : float, optional
        Outer radius of annular region as fraction of image semi-axis.
        Default is 0.5 (50% of half-width). Must be in (inner_radius_fraction, 1].

    Returns
    -------
    score : float
        Grid score in range [-2, 2]. Higher values indicate stronger hexagonal
        periodicity. Returns NaN if:
        - Autocorrelation has NaN or infinite values in annular region
        - Annular region is too small (< 10 pixels)
        - Rotation correlation fails

    Notes
    -----
    **Algorithm** (Sargolini et al., 2006):

    1. Define annular region between inner and outer radii (excludes central peak)
    2. Rotate autocorrelogram by angles: 30°, 60°, 90°, 120°, 150°
    3. Compute Pearson correlation between original and rotated within annulus
    4. Grid score = min(r60, r120) - max(r30, r90, r150)

    **Interpretation**:

    - **score > 0.4**: Strong hexagonal grid (typical threshold for grid cells)
    - **score ≈ 0**: No hexagonal structure (place cells, non-spatial cells)
    - **score < 0**: Anti-hexagonal structure (rare)

    **Annular region selection**:

    The annular region should:
    - Exclude central peak (inner_radius > 0)
    - Capture first ring of grid peaks (outer_radius chosen based on expected spacing)
    - Typical values: inner=0.2, outer=0.5 (works for most grid cells)

    If you know the expected grid spacing in bins, you can set radii accordingly:
    - inner_radius ≈ 0.5 × spacing (exclude center)
    - outer_radius ≈ 1.5 × spacing (capture first ring)

    **Differences from opexebo**:

    This implementation is compatible with opexebo's grid_score but with these differences:
    - Works on autocorrelogram from neurospatial's spatial_autocorrelation
    - Annular region defined as fraction of image size (opexebo uses absolute pixels)
    - Uses scipy.ndimage.rotate for rotations (same as opexebo)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import spatial_autocorrelation, grid_score
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Create grid cell firing pattern (simplified example)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # ... populate with hexagonal grid pattern ...
    >>>
    >>> # Compute autocorrelation
    >>> autocorr_2d = spatial_autocorrelation(firing_rate, env, method="fft")
    >>>
    >>> # Compute grid score
    >>> score = grid_score(autocorr_2d)
    >>> print(f"Grid score: {score:.3f}")  # doctest: +SKIP
    Grid score: 0.623

    See Also
    --------
    spatial_autocorrelation : Compute 2D autocorrelation map
    periodicity_score : Graph-based alternative for irregular topologies

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    # Validate inputs
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")

    if not (0 < inner_radius_fraction < 1):
        raise ValueError(
            f"inner_radius_fraction must be in (0, 1), got {inner_radius_fraction}"
        )

    if not (inner_radius_fraction < outer_radius_fraction <= 1):
        raise ValueError(
            f"outer_radius_fraction must be in (inner_radius_fraction, 1], "
            f"got {outer_radius_fraction} with inner={inner_radius_fraction}"
        )

    # Handle NaN/inf
    if not np.all(np.isfinite(autocorr_2d)):
        return np.nan

    height, width = autocorr_2d.shape

    # Create annular mask (ring region between inner and outer radii)
    center_y, center_x = height / 2, width / 2

    # Use minimum of height/width for radius calculation (handle non-square)
    semi_axis = min(height, width) / 2

    inner_radius = inner_radius_fraction * semi_axis
    outer_radius = outer_radius_fraction * semi_axis

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    # Annular mask
    annular_mask = (distance_from_center >= inner_radius) & (
        distance_from_center <= outer_radius
    )

    # Check if annular region is large enough
    if np.sum(annular_mask) < 10:
        # Too few pixels for reliable correlation
        return np.nan

    # Rotation angles (degrees)
    angles = [30, 60, 90, 120, 150]

    # Compute correlation for each rotation
    correlations = {}

    for angle in angles:
        # Rotate autocorrelation map
        rotated = ndimage.rotate(
            autocorr_2d, angle, reshape=False, order=1, mode="constant", cval=0.0
        )

        # Extract values in annular region for original and rotated
        orig_vals = autocorr_2d[annular_mask]
        rot_vals = rotated[annular_mask]

        # Remove NaN/inf from rotated (interpolation may introduce them)
        valid = np.isfinite(orig_vals) & np.isfinite(rot_vals)

        if np.sum(valid) < 10:
            # Not enough valid points
            return np.nan

        orig_vals_clean = orig_vals[valid]
        rot_vals_clean = rot_vals[valid]

        # Check for zero variance
        if np.std(orig_vals_clean) == 0 or np.std(rot_vals_clean) == 0:
            return np.nan

        # Compute Pearson correlation
        corr, _ = stats.pearsonr(orig_vals_clean, rot_vals_clean)
        correlations[angle] = corr

    # Grid score = min(r60, r120) - max(r30, r90, r150)
    r60 = correlations[60]
    r120 = correlations[120]
    r30 = correlations[30]
    r90 = correlations[90]
    r150 = correlations[150]

    score = min(r60, r120) - max(r30, r90, r150)

    return float(score)


def periodicity_score(
    distances: NDArray[np.float64],
    correlations: NDArray[np.float64],
    *,
    min_peaks: int = 3,
    prominence_threshold: float = 0.1,
) -> float:
    """
    Compute periodicity score from distance-correlation profile (graph-based).

    This is a graph-based alternative to grid_score for irregular topologies.
    Instead of analyzing 2D rotational symmetry, it detects regular spacing
    in the distance-correlation profile. Grid cells show periodic peaks at
    multiples of the grid spacing.

    Parameters
    ----------
    distances : NDArray[np.float64], shape (n_bins,)
        Distance bin centers (from spatial_autocorrelation with method='graph').
    correlations : NDArray[np.float64], shape (n_bins,)
        Autocorrelation values at each distance.
    min_peaks : int, optional
        Minimum number of peaks required to compute periodicity. Default is 3.
        Grid cells typically show 3+ peaks.
    prominence_threshold : float, optional
        Minimum peak prominence (height above surrounding baseline).
        Default is 0.1. Higher values require more distinct peaks.

    Returns
    -------
    score : float
        Periodicity score in range [0, 1]. Higher values indicate stronger
        periodic structure. Returns NaN if:
        - Too few peaks detected (< min_peaks)
        - Peak spacing has high variance (irregular)
        - All correlations are NaN

    Notes
    -----
    **Algorithm**:

    1. Exclude zero-lag region (distance < distance_bin_width)
    2. Find peaks in correlation profile using scipy.signal.find_peaks
    3. Require minimum peak prominence (distinctness)
    4. Compute inter-peak distances (spacing between consecutive peaks)
    5. Periodicity score = 1 - (std(spacing) / mean(spacing))
       - Score ≈ 1: Highly regular spacing (strong grid)
       - Score ≈ 0: Irregular spacing (no grid)

    **Interpretation**:

    - **score > 0.6**: Likely grid cell with regular spacing
    - **score 0.3-0.6**: Weak periodicity or noisy grid
    - **score < 0.3**: No periodic structure (place cell, noise)

    **Comparison to grid_score**:

    - **grid_score**: 2D rotational symmetry, requires regular 2D grid
    - **periodicity_score**: 1D spacing regularity, works on any graph topology
    - Use grid_score for open-field recordings (opexebo compatibility)
    - Use periodicity_score for tracks, irregular environments, or graphs

    **When to use this method**:

    - Irregular environments (non-rectangular, obstacles)
    - 1D tracks (linear environments)
    - When FFT-based autocorrelation is not feasible
    - Exploratory analysis to detect any periodic structure

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import spatial_autocorrelation, periodicity_score
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Create grid cell firing pattern
    >>> firing_rate = np.zeros(env.n_bins)
    >>> # ... populate with grid cell firing ...
    >>>
    >>> # Compute graph-based autocorrelation
    >>> distances, correlations = spatial_autocorrelation(
    ...     firing_rate, env, method="graph", n_distance_bins=50
    ... )
    >>>
    >>> # Compute periodicity score
    >>> score = periodicity_score(distances, correlations)
    >>> print(f"Periodicity score: {score:.3f}")  # doctest: +SKIP
    Periodicity score: 0.845

    See Also
    --------
    spatial_autocorrelation : Compute distance-correlation profile
    grid_score : 2D rotational symmetry (FFT-based)

    Notes
    -----
    This is an experimental metric not present in opexebo. It provides a
    graph-compatible alternative to traditional grid score analysis.
    """
    # Validate inputs
    if distances.shape != correlations.shape:
        raise ValueError(
            f"distances and correlations must have same shape, "
            f"got {distances.shape} and {correlations.shape}"
        )

    if len(distances) == 0:
        raise ValueError("distances and correlations cannot be empty")

    if min_peaks < 2:
        raise ValueError(f"min_peaks must be >= 2, got {min_peaks}")

    if not (0 < prominence_threshold < 1):
        raise ValueError(
            f"prominence_threshold must be in (0, 1), got {prominence_threshold}"
        )

    # Handle NaN
    valid_mask = np.isfinite(correlations) & np.isfinite(distances)
    if not np.any(valid_mask):
        return np.nan

    distances_clean = distances[valid_mask]
    correlations_clean = correlations[valid_mask]

    if len(distances_clean) < min_peaks:
        return np.nan

    # Exclude zero-lag region (first bin, typically autocorr = 1.0)
    # Start from second bin to avoid trivial central peak
    if len(distances_clean) > 1:
        distances_clean = distances_clean[1:]
        correlations_clean = correlations_clean[1:]

    if len(distances_clean) < min_peaks:
        return np.nan

    # Find peaks in correlation profile
    from scipy.signal import find_peaks

    # Compute prominence threshold in absolute units
    corr_range = np.max(correlations_clean) - np.min(correlations_clean)
    prominence_abs = prominence_threshold * corr_range

    peaks, _ = find_peaks(
        correlations_clean,
        prominence=prominence_abs,
    )

    if len(peaks) < min_peaks:
        # Not enough peaks
        return np.nan

    # Get peak distances (physical distances of peaks)
    peak_distances = distances_clean[peaks]

    # Compute inter-peak spacing (distance between consecutive peaks)
    inter_peak_spacing = np.diff(peak_distances)

    if len(inter_peak_spacing) < 2:
        # Need at least 2 spacings for variance
        return np.nan

    # Compute regularity of spacing
    mean_spacing = np.mean(inter_peak_spacing)
    std_spacing = np.std(inter_peak_spacing)

    if mean_spacing == 0:
        return np.nan

    # Periodicity score = 1 - coefficient of variation
    # High score = regular spacing, low score = irregular spacing
    cv = std_spacing / mean_spacing
    score = 1.0 - cv

    # Clip to [0, 1]
    score = np.clip(score, 0.0, 1.0)

    return float(score)
