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

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol


def border_score(
    firing_rate: NDArray[np.float64],
    env: EnvironmentProtocol,
    *,
    threshold: float = 0.3,
    min_area: float = 0.0,
    distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
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
    distance_metric : {'geodesic', 'euclidean'}, optional
        Distance metric for computing distance from field bins to boundary bins.
        - 'geodesic': Graph shortest path distance (default). Respects environment
          connectivity, appropriate for irregular environments or those with obstacles.
        - 'euclidean': Straight-line distance in physical space. Generally faster
          for large environments (no graph traversal). Appropriate for simple,
          open environments without obstacles.
        Default is 'geodesic' to match the original implementation and ensure
        compatibility with irregular graph structures.

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

    **Differences from opexebo and Solstad et al. (2008)**:

    This implementation differs from opexebo's border_score in several ways:

    1. **Coverage**: Overall boundary coverage (not per-wall). This implementation
       computes a single coverage metric across all boundary bins, whereas opexebo
       computes coverage for each wall (N/S/E/W) separately and uses the maximum.

    2. **Distance**: Geodesic or Euclidean (not Manhattan). Opexebo uses Manhattan
       (taxicab) distance from arena edges. This implementation supports both
       geodesic (graph shortest path) and Euclidean (straight-line) distances.

    3. **Boundaries**: Automatic graph-based detection (not wall specification).
       Opexebo requires explicit arena shape ("square", "rect", "circle"). This
       implementation automatically detects boundaries via ``env.boundary_bins``.

    4. **Environment**: Works on any graph (not rectangular arenas only). This
       implementation generalizes to irregular environments with obstacles.

    **Computing per-wall coverage** (for rectangular arenas):

    If you need per-wall coverage analysis like opexebo, use
    `compute_region_coverage()` with wall regions:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import border_score, compute_region_coverage
    >>> from shapely.geometry import box
    >>>
    >>> # Create rectangular environment
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>>
    >>> # Define wall regions (for a 40x40 arena centered at origin)
    >>> wall_width = 5.0  # Width of wall region in cm
    >>> env.regions.add("north_wall", polygon=box(-20, 15, 20, 20))
    >>> env.regions.add("south_wall", polygon=box(-20, -20, 20, -15))
    >>> env.regions.add("east_wall", polygon=box(15, -20, 20, 20))
    >>> env.regions.add("west_wall", polygon=box(-20, -20, -15, 20))
    >>>
    >>> # Compute coverage for each wall
    >>> field_bins = np.where(firing_rate >= 0.3 * np.nanmax(firing_rate))[0]
    >>> coverage = compute_region_coverage(field_bins, env)
    >>> for wall_name in ["north_wall", "south_wall", "east_wall", "west_wall"]:
    ...     print(f"{wall_name}: {coverage[wall_name]:.2%}")  # doctest: +SKIP

    **When to use opexebo vs. neurospatial**:

    - Use **opexebo** for rectangular arenas requiring exact reproduction of
      Solstad et al. (2008) or comparison with Moser lab publications
    - Use **neurospatial** for irregular environments, graphs with obstacles,
      or when you need flexible distance metrics

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

    if distance_metric not in ("geodesic", "euclidean"):
        raise ValueError(
            f"distance_metric must be 'geodesic' or 'euclidean', got '{distance_metric}'"
        )

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
    if distance_metric == "geodesic":
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

    else:  # distance_metric == "euclidean"
        # Compute Euclidean distances in physical space (vectorized)
        boundary_positions = env.bin_centers[boundary_bins]
        field_positions = env.bin_centers[field_bins]

        # Vectorized computation using broadcasting
        # Shape: (n_field_bins, n_boundary_bins, n_dims)
        diff = field_positions[:, np.newaxis, :] - boundary_positions[np.newaxis, :, :]
        # Shape: (n_field_bins, n_boundary_bins)
        distances_matrix = np.linalg.norm(diff, axis=2)
        # Shape: (n_field_bins,) - minimum distance to any boundary bin
        distances_to_boundary = np.min(distances_matrix, axis=1)

        if len(distances_to_boundary) == 0:
            return np.nan

        mean_distance = float(np.mean(distances_to_boundary))

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


def compute_region_coverage(
    field_bins: NDArray[np.int64],
    env: Environment,
    *,
    regions: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute field coverage for each region.

    This function calculates what fraction of each spatial region is covered
    by a place field. Useful for determining wall preferences in border cells
    or analyzing field overlap with task-relevant zones.

    Parameters
    ----------
    field_bins : array of int
        Bin indices comprising the field (e.g., from detect_place_fields).
    env : Environment
        Spatial environment with defined regions.
    regions : list of str, optional
        List of region names to analyze. If None, analyzes all regions
        defined in env.regions.

    Returns
    -------
    coverage : dict[str, float]
        Dictionary mapping region name to coverage fraction [0, 1].
        Coverage is computed as:
        (number of region bins in field) / (total number of region bins)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.metrics import compute_region_coverage
    >>> from shapely.geometry import box
    >>>
    >>> # Create environment with wall regions
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> env.regions.add("north", polygon=box(-20, 15, 20, 20))
    >>> env.regions.add("south", polygon=box(-20, -20, 20, -15))
    >>> env.regions.add("east", polygon=box(15, -20, 20, 20))
    >>> env.regions.add("west", polygon=box(-20, -20, -15, 20))
    >>>
    >>> # Create field along north wall
    >>> firing_rate = np.zeros(env.n_bins)
    >>> north_bins = np.where(env.mask_for_region("north"))[0]
    >>> firing_rate[north_bins] = 5.0
    >>> field_bins = np.where(firing_rate > 0)[0]
    >>>
    >>> # Compute coverage per wall
    >>> coverage = compute_region_coverage(field_bins, env)
    >>> for wall, cov in sorted(coverage.items()):
    ...     print(f"{wall}: {cov:.1%}")  # doctest: +SKIP
    east: 0.0%
    north: 100.0%
    south: 0.0%
    west: 0.0%

    See Also
    --------
    border_score : Compute overall border preference score
    detect_place_fields : Detect place fields from firing rate maps

    Notes
    -----
    This function is particularly useful for:

    - **Border cell analysis**: Determine which wall a border cell prefers
    - **Task-relevant regions**: Check if fields overlap with reward zones, start boxes, etc.
    - **Multi-zone analysis**: Quantify field distribution across spatial zones

    The coverage metric is simply the Jaccard coefficient between the field
    and each region, but computed more efficiently.
    """
    # Get region names to analyze
    if regions is None:
        regions = list(env.regions.keys())

    # Validate regions exist
    for region_name in regions:
        if region_name not in env.regions:
            raise ValueError(
                f"Region '{region_name}' not found. "
                f"Available regions: {list(env.regions.keys())}"
            )

    # Compute coverage for each region
    coverage = {}
    for region_name in regions:
        # Get bins in this region
        region_mask = env.mask_for_region(region_name)
        region_bins = np.where(region_mask)[0]

        if len(region_bins) == 0:
            # Empty region
            coverage[region_name] = 0.0
        else:
            # Compute fraction of region bins that are in field
            coverage[region_name] = float(
                np.sum(np.isin(region_bins, field_bins)) / len(region_bins)
            )

    return coverage
