"""
Grid cell metrics for spatial analysis.

Implements spatial autocorrelation and grid score (Sargolini et al., 2006) with
support for both regular 2D grids (FFT-based) and irregular graph topologies
(graph-based distance autocorrelation).

This module provides:
- `spatial_autocorrelation`: Compute 2D or distance-based autocorrelation
- `grid_score`: Standard hexagonal periodicity score (Sargolini et al., 2006)
- `grid_scale`: Grid spacing from autocorrelogram peak detection
- `grid_orientation`: Grid orientation with consensus algorithm
- `grid_properties`: Combined function returning all grid metrics
- `periodicity_score`: Graph-based alternative for irregular topologies

References
----------
Sargolini, F., Fyhn, M., Hafting, T., McNaughton, B. L., Witter, M. P., Moser, M. B.,
    & Moser, E. I. (2006). Conjunctive representation of position, direction, and
    velocity in entorhinal cortex. Science, 312(5774), 758-762.
    https://doi.org/10.1126/science.1125572

Brandon, M. P., Bogaard, A. R., Libby, C. P., Connerney, M. A., Guber, K., &
    Bhattarai, B. (2011). Reduction of theta rhythm dissociates grid cell
    spatial periodicity from directional tuning. Science, 332(6029), 595-599.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import fft, ndimage, stats
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from neurospatial import Environment

# Axis flip detection threshold for orientation consensus algorithm.
# This is 1/4 of the 60° hexagonal periodicity. When angles cluster
# near ±15° from the 0°/60° boundary, axis flip correction is applied.
_AXIS_FLIP_THRESHOLD_DEG = 15.0


@dataclass
class GridProperties:
    """
    Container for grid cell metrics extracted from autocorrelogram.

    Attributes
    ----------
    score : float
        Standard grid score in range [-2, 2]. Higher values indicate stronger
        hexagonal periodicity. Computed as min(r60, r120) - max(r30, r90, r150).
    scale : float
        Grid spacing (distance between fields) in the same units as bin_size.
        Computed as median distance from center to 6 nearest peaks.
    orientation : float
        Grid orientation in degrees, range [0, 60). The angle of the grid
        relative to horizontal, normalized to the 60° periodicity of hexagonal
        grids.
    orientation_std : float
        Standard deviation of orientation estimates across the 6 peaks.
        Lower values indicate more consistent/reliable orientation.
    peak_coords : NDArray[np.float64]
        Coordinates of detected peaks relative to autocorrelogram center,
        shape (n_peaks, 2) where columns are (row_offset, col_offset).
        Typically 6 peaks for a well-formed grid.
    n_peaks : int
        Number of peaks detected. Grid cells typically have 6.

    Notes
    -----
    All metrics are computed from a single peak detection pass for efficiency.
    If fewer than 3 peaks are detected, scale and orientation will be NaN.

    Examples
    --------
    >>> from neurospatial.metrics import grid_properties, spatial_autocorrelation
    >>> autocorr = spatial_autocorrelation(firing_rate, env, method="fft")
    >>> props = grid_properties(autocorr, bin_size=2.0)
    >>> print(f"Score: {props.score:.2f}, Scale: {props.scale:.1f} cm")
    Score: 0.85, Scale: 42.3 cm
    """

    score: float
    scale: float
    orientation: float
    orientation_std: float
    peak_coords: NDArray[np.float64]
    n_peaks: int


def spatial_autocorrelation(
    firing_rate: NDArray[np.float64],
    env: Environment,
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


def _detect_grid_method(env: Environment) -> Literal["fft", "graph"]:
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
    env: Environment,
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
    env: Environment,
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


def _find_autocorr_peaks(
    autocorr_2d: NDArray[np.float64],
    *,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
    exclude_center_radius: float | None = None,
    max_peaks: int = 7,
) -> NDArray[np.float64]:
    """
    Find peaks in 2D autocorrelogram, excluding central peak.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (height, width)
        2D autocorrelation map with center at zero lag.
    min_distance : int, optional
        Minimum distance between peaks in pixels. Default is 5.
    threshold_rel : float, optional
        Minimum peak height as fraction of max value. Default is 0.1.
    exclude_center_radius : float, optional
        Radius around center to exclude (excludes central peak).
        If None, uses 10% of the smaller dimension.
    max_peaks : int, optional
        Maximum number of peaks to return. Default is 7 (center + 6 surrounding).

    Returns
    -------
    peak_coords : NDArray[np.float64], shape (n_peaks, 2)
        Peak coordinates relative to center, as (row_offset, col_offset).
        Sorted by distance from center (closest first).
    """
    from skimage.feature import peak_local_max

    height, width = autocorr_2d.shape
    center_y, center_x = height / 2, width / 2

    # Determine center exclusion radius
    if exclude_center_radius is None:
        exclude_center_radius = min(height, width) * 0.1

    # Find peaks using skimage
    # threshold_abs is relative to the autocorrelogram range
    valid_vals = autocorr_2d[np.isfinite(autocorr_2d)]
    if len(valid_vals) == 0:
        return np.array([]).reshape(0, 2)

    threshold_abs = threshold_rel * (np.max(valid_vals) - np.min(valid_vals))

    # Handle NaN by replacing with minimum value
    autocorr_clean = np.where(np.isfinite(autocorr_2d), autocorr_2d, np.min(valid_vals))

    try:
        peaks = peak_local_max(
            autocorr_clean,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=True,
            num_peaks=max_peaks * 2,  # Get extras to filter
        )
    except Exception:
        # Fallback if peak_local_max fails
        return np.array([]).reshape(0, 2)

    if len(peaks) == 0:
        return np.array([]).reshape(0, 2)

    # Convert to center-relative coordinates
    peak_coords_relative = peaks.astype(np.float64) - np.array([center_y, center_x])

    # Compute distances from center
    distances = np.sqrt(
        peak_coords_relative[:, 0] ** 2 + peak_coords_relative[:, 1] ** 2
    )

    # Exclude central peak (within exclude_center_radius)
    non_central = distances > exclude_center_radius
    peak_coords_relative = peak_coords_relative[non_central]
    distances = distances[non_central]

    if len(peak_coords_relative) == 0:
        return np.array([]).reshape(0, 2)

    # Sort by distance from center
    sort_idx = np.argsort(distances)
    peak_coords_relative = peak_coords_relative[sort_idx]

    # Limit to max_peaks
    if len(peak_coords_relative) > max_peaks:
        peak_coords_relative = peak_coords_relative[:max_peaks]

    return np.asarray(peak_coords_relative, dtype=np.float64)


def grid_scale(
    autocorr_2d: NDArray[np.float64],
    bin_size: float,
    *,
    n_peaks: int = 6,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> float:
    """
    Compute grid spacing (scale) from 2D autocorrelogram.

    Grid scale is the characteristic distance between adjacent grid fields,
    computed as the median distance from the autocorrelogram center to the
    6 nearest peaks (the first ring of hexagonal symmetry).

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (height, width)
        2D spatial autocorrelation map (from spatial_autocorrelation with method='fft').
        Should be centered (zero lag at center).
    bin_size : float
        Size of each spatial bin in physical units (e.g., cm).
        Used to convert pixel distances to physical distances.
    n_peaks : int, optional
        Number of peaks to use for computing scale. Default is 6 (first ring).
        For noisy data, using fewer peaks may be more robust.
    min_distance : int, optional
        Minimum distance between peaks in pixels. Default is 5.
    threshold_rel : float, optional
        Minimum peak height as fraction of autocorrelogram range. Default is 0.1.

    Returns
    -------
    scale : float
        Grid spacing in physical units (same as bin_size).
        Returns NaN if fewer than 3 peaks are detected.

    Notes
    -----
    **Algorithm**:

    1. Find local maxima in autocorrelogram using peak detection
    2. Exclude central peak (self-correlation)
    3. Sort remaining peaks by distance from center
    4. Take n_peaks closest peaks (typically 6 for hexagonal grid)
    5. Compute median distance to these peaks
    6. Convert from pixels to physical units using bin_size

    **Interpretation**:

    - Scale represents the spacing between grid field centers
    - For hexagonal grids, all 6 first-ring peaks should be at similar distances
    - High variance in peak distances suggests poor grid regularity

    **Relationship to grid_score**:

    - grid_score measures rotational symmetry (quality of hexagonal pattern)
    - grid_scale measures the characteristic period (spacing between fields)
    - Both are needed to fully characterize a grid cell

    Examples
    --------
    >>> from neurospatial.metrics import spatial_autocorrelation, grid_scale
    >>> autocorr = spatial_autocorrelation(firing_rate, env, method="fft")
    >>> scale = grid_scale(autocorr, bin_size=2.0)
    >>> print(f"Grid spacing: {scale:.1f} cm")
    Grid spacing: 42.5 cm

    See Also
    --------
    grid_orientation : Extract grid orientation angle
    grid_properties : Combined function for all grid metrics
    grid_score : Measure hexagonal periodicity

    References
    ----------
    Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
        Microstructure of a spatial map in the entorhinal cortex. Nature,
        436(7052), 801-806.
    """
    # Validate inputs
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")

    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    if n_peaks < 1:
        raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")

    # Find peaks
    peak_coords = _find_autocorr_peaks(
        autocorr_2d,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        max_peaks=n_peaks + 1,  # Extra in case some are filtered
    )

    if len(peak_coords) < 3:
        # Not enough peaks to compute reliable scale
        return np.nan

    # Compute distances from center (peaks are already center-relative)
    distances_pixels = np.sqrt(peak_coords[:, 0] ** 2 + peak_coords[:, 1] ** 2)

    # Use up to n_peaks closest peaks
    distances_to_use = distances_pixels[: min(n_peaks, len(distances_pixels))]

    # Compute median distance and convert to physical units
    scale = float(np.median(distances_to_use)) * bin_size

    return scale


def grid_orientation(
    autocorr_2d: NDArray[np.float64],
    *,
    n_peaks: int = 6,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> tuple[float, float]:
    """
    Compute grid orientation from 2D autocorrelogram.

    Grid orientation is the angle of the grid pattern relative to horizontal,
    normalized to the [0, 60) degree range due to the 60° rotational symmetry
    of hexagonal grids.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (height, width)
        2D spatial autocorrelation map (from spatial_autocorrelation with method='fft').
        Should be centered (zero lag at center).
    n_peaks : int, optional
        Number of peaks to use for computing orientation. Default is 6.
    min_distance : int, optional
        Minimum distance between peaks in pixels. Default is 5.
    threshold_rel : float, optional
        Minimum peak height as fraction of autocorrelogram range. Default is 0.1.

    Returns
    -------
    orientation : float
        Grid orientation in degrees, range [0, 60).
        Returns NaN if fewer than 3 peaks are detected.
    orientation_std : float
        Standard deviation of orientation estimates across peaks.
        Lower values indicate more reliable orientation.
        Returns NaN if fewer than 3 peaks are detected.

    Notes
    -----
    **Algorithm** (adapted from gridcell_metrics / opexebo):

    1. Find local maxima in autocorrelogram (excluding central peak)
    2. Compute angle from center to each peak using arctan2
    3. Reduce all angles to [0, 60) range using modulo 60
    4. Detect axis flips: if angles cluster around 30°, flip some angles
       to achieve consensus (handles ambiguity in hexagonal symmetry)
    5. Return mean and standard deviation of adjusted angles

    **Interpretation**:

    - Orientation indicates the rotation of the grid pattern
    - Low std indicates a well-formed, regular hexagonal grid
    - High std may indicate distorted grid or poor peak detection

    **Axis flip handling**:

    Hexagonal grids have 6-fold symmetry, meaning peaks at 0°, 60°, 120°, etc.
    are equivalent. When peaks cluster around 30°, some may be at 30° and
    others at 90° (30° + 60°). The algorithm detects this and adjusts angles
    to achieve consensus.

    Examples
    --------
    >>> from neurospatial.metrics import spatial_autocorrelation, grid_orientation
    >>> autocorr = spatial_autocorrelation(firing_rate, env, method="fft")
    >>> orientation, orientation_std = grid_orientation(autocorr)
    >>> print(f"Orientation: {orientation:.1f}° ± {orientation_std:.1f}°")
    Orientation: 23.5° ± 2.1°

    See Also
    --------
    grid_scale : Extract grid spacing
    grid_properties : Combined function for all grid metrics
    grid_score : Measure hexagonal periodicity

    References
    ----------
    Brandon, M. P., et al. (2011). Reduction of theta rhythm dissociates
        grid cell spatial periodicity from directional tuning. Science.
    """
    # Validate inputs
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")

    # Find peaks
    peak_coords = _find_autocorr_peaks(
        autocorr_2d,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        max_peaks=n_peaks + 1,
    )

    if len(peak_coords) < 3:
        # Not enough peaks to compute reliable orientation
        return np.nan, np.nan

    # Compute angles from center to each peak
    # Note: peak_coords are (row_offset, col_offset), so (y, x)
    # arctan2 takes (y, x) and returns angle in radians
    angles_rad = np.arctan2(peak_coords[:, 0], peak_coords[:, 1])
    angles_deg = np.degrees(angles_rad)

    # Normalize to [0, 360)
    angles_deg = angles_deg % 360

    # Reduce to [0, 60) using 60° periodicity
    orientations = angles_deg % 60

    # Axis flip detection and correction (from gridcell_metrics)
    # If angles cluster around 30°, some may be at 30° and others at 90° (mod 60 = 30)
    # This creates a bimodal distribution around 0° and 60° in the reduced space

    # Check distance to 60° for each angle
    dist_to_60 = 60 - orientations
    # If closer to 60° than to 0°, subtract 60° (will give negative values)
    adjusted = np.where(dist_to_60 < orientations, orientations - 60, orientations)

    # Now angles are in [-30, 30] range
    # Check if there's a split around 0° (some positive, some negative near ±threshold)
    # If there's high spread, we may have axis flip issue
    # Detect by checking if angles are split between positive and negative
    n_positive = np.sum(adjusted > _AXIS_FLIP_THRESHOLD_DEG)
    n_negative = np.sum(adjusted < -_AXIS_FLIP_THRESHOLD_DEG)

    if n_positive > 0 and n_negative > 0:
        # Axis flip detected - force all to same sign based on majority
        if n_positive > n_negative:
            # Make all positive: wrap negative values
            adjusted = np.where(adjusted < 0, adjusted + 60, adjusted)
        else:
            # Make all negative: wrap positive values
            adjusted = np.where(adjusted > 0, adjusted - 60, adjusted)

    # Compute mean and std
    mean_orientation = float(np.nanmean(adjusted))
    std_orientation = float(np.nanstd(adjusted))

    # Normalize mean to [0, 60)
    mean_orientation = mean_orientation % 60

    # If result is very close to 60, set to 0 (both represent same orientation)
    if np.abs(mean_orientation - 60) < 0.1:
        mean_orientation = 0.0

    return mean_orientation, std_orientation


def grid_properties(
    autocorr_2d: NDArray[np.float64],
    bin_size: float,
    *,
    inner_radius_fraction: float = 0.2,
    outer_radius_fraction: float = 0.5,
    n_peaks: int = 6,
    min_distance: int = 5,
    threshold_rel: float = 0.1,
) -> GridProperties:
    """
    Compute all grid cell metrics from 2D autocorrelogram.

    This function efficiently computes grid score, scale, orientation, and
    peak coordinates from a single pass of peak detection. Use this instead
    of calling grid_score, grid_scale, and grid_orientation separately when
    you need multiple metrics.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (height, width)
        2D spatial autocorrelation map (from spatial_autocorrelation with method='fft').
        Should be centered (zero lag at center).
    bin_size : float
        Size of each spatial bin in physical units (e.g., cm).
        Used to convert pixel distances to physical distances.
    inner_radius_fraction : float, optional
        Inner radius for grid_score annular region as fraction of image semi-axis.
        Default is 0.2 (20% of half-width).
    outer_radius_fraction : float, optional
        Outer radius for grid_score annular region. Default is 0.5.
    n_peaks : int, optional
        Number of peaks to use for scale and orientation. Default is 6.
    min_distance : int, optional
        Minimum distance between peaks in pixels. Default is 5.
    threshold_rel : float, optional
        Minimum peak height as fraction of autocorrelogram range. Default is 0.1.

    Returns
    -------
    GridProperties
        Dataclass containing:
        - score: Grid score [-2, 2]
        - scale: Grid spacing in physical units
        - orientation: Grid orientation in degrees [0, 60)
        - orientation_std: Standard deviation of orientation
        - peak_coords: Detected peak coordinates (n_peaks, 2)
        - n_peaks: Number of peaks detected

    Notes
    -----
    **When to use this function**:

    - When you need multiple grid metrics (more efficient than separate calls)
    - When you want access to detected peak coordinates
    - When building analysis pipelines that need all grid properties

    **Efficiency**:

    This function performs peak detection once and reuses results for both
    scale and orientation computation. The grid_score is computed separately
    using the rotation-correlation method.

    **Relationship to individual functions**:

    The results are equivalent to calling:
    - `grid_score(autocorr_2d, ...)` - matches exactly
    - `grid_scale(autocorr_2d, bin_size, ...)` - may differ slightly
    - `grid_orientation(autocorr_2d, ...)` - may differ slightly

    Note: `grid_properties` performs peak detection once and reuses results,
    while `grid_scale` and `grid_orientation` each perform independent peak
    detection. Due to the stochastic nature of peak detection near thresholds,
    scale and orientation values may differ slightly (typically <1%) between
    `grid_properties` and individual function calls. For exact consistency,
    use `grid_properties` when you need multiple metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import spatial_autocorrelation, grid_properties

    >>> # Create environment and compute autocorrelation
    >>> positions = np.random.randn(5000, 2) * 50
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> # firing_rate = ... (grid cell activity)
    >>> # autocorr = spatial_autocorrelation(firing_rate, env, method="fft")

    >>> # Get all grid properties at once
    >>> # props = grid_properties(autocorr, bin_size=2.0)
    >>> # print(f"Score: {props.score:.2f}")
    >>> # print(f"Scale: {props.scale:.1f} cm")
    >>> # print(f"Orientation: {props.orientation:.1f}° ± {props.orientation_std:.1f}°")
    >>> # print(f"Detected {props.n_peaks} peaks")

    See Also
    --------
    grid_score : Grid score only
    grid_scale : Grid spacing only
    grid_orientation : Grid orientation only
    GridProperties : Return type dataclass

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    Hafting et al. (2005). Microstructure of a spatial map in the entorhinal
        cortex. Nature, 436(7052), 801-806.
    """
    # Validate inputs
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")

    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    # Compute grid score (uses rotation-correlation method)
    score = grid_score(
        autocorr_2d,
        inner_radius_fraction=inner_radius_fraction,
        outer_radius_fraction=outer_radius_fraction,
    )

    # Find peaks once for scale and orientation
    peak_coords = _find_autocorr_peaks(
        autocorr_2d,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        max_peaks=n_peaks + 1,
    )

    n_detected = len(peak_coords)

    if n_detected < 3:
        # Not enough peaks for reliable scale/orientation
        return GridProperties(
            score=score,
            scale=np.nan,
            orientation=np.nan,
            orientation_std=np.nan,
            peak_coords=peak_coords,
            n_peaks=n_detected,
        )

    # Compute scale from peak distances
    distances_pixels = np.sqrt(peak_coords[:, 0] ** 2 + peak_coords[:, 1] ** 2)
    distances_to_use = distances_pixels[: min(n_peaks, len(distances_pixels))]
    scale = float(np.median(distances_to_use)) * bin_size

    # Compute orientation from peak angles
    angles_rad = np.arctan2(peak_coords[:, 0], peak_coords[:, 1])
    angles_deg = np.degrees(angles_rad) % 360
    orientations = angles_deg % 60

    # Axis flip correction
    dist_to_60 = 60 - orientations
    adjusted = np.where(dist_to_60 < orientations, orientations - 60, orientations)

    n_positive = np.sum(adjusted > _AXIS_FLIP_THRESHOLD_DEG)
    n_negative = np.sum(adjusted < -_AXIS_FLIP_THRESHOLD_DEG)

    if n_positive > 0 and n_negative > 0:
        if n_positive > n_negative:
            adjusted = np.where(adjusted < 0, adjusted + 60, adjusted)
        else:
            adjusted = np.where(adjusted > 0, adjusted - 60, adjusted)

    orientation = float(np.nanmean(adjusted)) % 60
    orientation_std = float(np.nanstd(adjusted))

    if np.abs(orientation - 60) < 0.1:
        orientation = 0.0

    return GridProperties(
        score=score,
        scale=scale,
        orientation=orientation,
        orientation_std=orientation_std,
        peak_coords=peak_coords,
        n_peaks=n_detected,
    )
