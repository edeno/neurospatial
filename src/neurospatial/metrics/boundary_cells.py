"""
Boundary cell metrics for spatial analysis.

Implements border score (Solstad et al., 2008) and related boundary cell analyses.

References
----------
Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008).
    Representation of geometric borders in the entorhinal cortex. Science,
    322(5909), 1865-1868. https://doi.org/10.1126/science.1166466
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol


def border_score(
    firing_rate: NDArray[np.float64],
    env: EnvironmentProtocol,
    *,
    threshold: float = 0.3,
    min_area: float = 0.0,
) -> float:
    """
    Compute border score for a spatial firing rate map.

    The border score quantifies how much a cell's firing field is aligned with
    environmental boundaries (walls). It ranges from -1 (center-preferring) to
    +1 (perfect border cell). Implements the algorithm from Solstad et al. (2008).

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second).
    env : EnvironmentProtocol
        Spatial environment containing bin centers and connectivity.
    threshold : float, optional
        Fraction of peak firing rate used to segment the field. Default is 0.3
        (30% of peak), following Solstad et al. (2008).
    min_area : float, optional
        Minimum field area (in physical units) to compute border score. Fields
        smaller than this return NaN. Default is 0.0 (no filtering). For rat
        hippocampal data, Solstad et al. (2008) used 200 cm². Adjust based on
        your bin size and environment scale.

    Returns
    -------
    float
        Border score in range [-1, 1]. Returns NaN if:
        - All firing rates are zero or NaN
        - Peak firing rate is zero or NaN
        - Field area is below min_area
        - No field bins after thresholding

    Notes
    -----
    **Algorithm** (adapted for irregular graphs):

    1. Segment field at threshold: bins where `firing_rate >= threshold * peak`
    2. Compute boundary coverage (cM): fraction of boundary bins in field
    3. Compute normalized mean distance (d): mean distance from field bins to
       nearest boundary bin, normalized by environment diameter
    4. Border score: `(cM - d) / (cM + d)`

    **Interpretation**:

    - **+1**: Perfect border cell (field covers boundary, far from center)
    - **0**: No boundary preference (uniform or mixed)
    - **-1**: Anti-border (field in center, far from boundaries)

    **Differences from Solstad et al. (2008)**:

    - Original algorithm uses rectangular arenas with 4 discrete walls
    - This implementation generalizes to irregular graph-based environments
    - Boundary coverage computed over all boundary bins (not per-wall)
    - Distance computed using graph shortest paths (not Euclidean)

    References
    ----------
    Solstad et al. (2008). Representation of geometric borders in the entorhinal
        cortex. Science, 322(5909), 1865-1868.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import border_score
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Create border cell (high firing along boundary)
    >>> firing_rate = np.zeros(env.n_bins)
    >>> boundary_bins = env.boundary_bins
    >>> firing_rate[boundary_bins] = 5.0
    >>>
    >>> score = border_score(firing_rate, env)
    >>> print(f"Border score: {score:.3f}")  # doctest: +SKIP
    Border score: 0.850

    See Also
    --------
    neurospatial.metrics.detect_place_fields : Detect place fields
    Environment.boundary_bins : Get boundary bin indices
    """
    # Validate inputs
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )

    if not (0 < threshold < 1):
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}. "
            "Typically 0.3 (30% of peak)."
        )

    if min_area < 0:
        raise ValueError(f"min_area must be non-negative, got {min_area}")

    # Handle all-NaN or all-zero
    if np.all(np.isnan(firing_rate)) or np.all(firing_rate == 0):
        return np.nan

    # Find peak firing rate (ignore NaN)
    peak_rate = np.nanmax(firing_rate)
    if peak_rate == 0 or np.isnan(peak_rate):
        return np.nan

    # Segment field at threshold
    field_mask = firing_rate >= (threshold * peak_rate)
    field_bins = np.where(field_mask)[0]

    if len(field_bins) == 0:
        return np.nan

    # Check field area (sum of bin areas)
    from neurospatial.metrics.place_fields import field_size

    area = field_size(field_bins, env)
    if area < min_area:
        return np.nan

    # Get boundary bins
    boundary_bins = env.boundary_bins

    if len(boundary_bins) == 0:
        # No boundaries detected (shouldn't happen in normal environments)
        return np.nan

    # Compute boundary coverage (cM)
    # Fraction of boundary bins that are in the field
    boundary_in_field = np.isin(boundary_bins, field_bins)
    coverage = np.sum(boundary_in_field) / len(boundary_bins)

    # Compute mean distance from field bins to nearest boundary bin
    # Use multi-source Dijkstra for efficiency (single pass for all boundary bins)
    try:
        # Compute shortest distances from ALL boundary bins to all reachable nodes
        distances_from_boundary = nx.multi_source_dijkstra_path_length(
            env.connectivity,
            sources=boundary_bins.tolist(),
            weight="distance",
        )
    except Exception:
        return np.nan

    # For each field bin, get distance to nearest boundary (already computed above)
    distances_to_boundary = []
    for field_bin in field_bins:
        if int(field_bin) in distances_from_boundary:
            distances_to_boundary.append(distances_from_boundary[int(field_bin)])

    if len(distances_to_boundary) == 0:
        return np.nan

    mean_distance = np.mean(distances_to_boundary)

    # Normalize distance by environment extent (maximum spatial extent)
    # Compute extent as the diagonal of the bounding box
    mins = np.min(env.bin_centers, axis=0)
    maxs = np.max(env.bin_centers, axis=0)
    extent = np.linalg.norm(maxs - mins)

    if extent == 0:
        return np.nan

    normalized_distance = mean_distance / extent

    # Compute border score: (cM - d) / (cM + d)
    denominator = coverage + normalized_distance
    if denominator == 0:
        return np.nan

    score = (coverage - normalized_distance) / denominator

    # Ensure score is in [-1, 1] (should be by construction, but numerical precision)
    score = np.clip(score, -1.0, 1.0)

    return float(score)
