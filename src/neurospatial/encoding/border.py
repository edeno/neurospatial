"""
Boundary cell metrics for spatial analysis.

Implements the border score of Solstad et al. (2008) for arenas of any planar
shape, and per-region field coverage.

References
----------
Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I. (2008).
    Representation of geometric borders in the entorhinal cortex. Science,
    322(5909), 1865-1868. https://doi.org/10.1126/science.1166466
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from neurospatial._exceptions import RegionNotFoundError, _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._types import DistanceMetric

# Solstad et al. (2008): a field is at least 200 cm^2 of neighboring bins at
# or above 30% of the peak rate.
BORDER_FIELD_MIN_AREA_CM2 = 200.0


def border_score(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    threshold: float = 0.3,
    min_area: float | None = None,
    metric: DistanceMetric = "geodesic",
    boundary_source: BoundarySource = "physical",
) -> float:
    """
    Border score of a firing rate map (Solstad et al., 2008).

    Border cells of the medial entorhinal cortex fire along one or more walls
    of the arena. The border score compares how much of one wall a firing
    field covers with how far the cell's firing lies from the walls::

        b = (cM - dm) / (cM + dm)

    It ranges from -1 (firing away from every wall) to +1 (a thin field along
    the whole of a wall). On a rectangular arena it is the score of Solstad
    et al. (2008); the same definition applies to any planar arena.

    Parameters
    ----------
    env : Environment
        Planar environment whose bins are exact cells (grid, hexagonal,
        triangular mesh or polygon cells).
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz). NaN marks unvisited bins.
    threshold : float, default=0.3
        Fraction of the peak firing rate that defines field bins, following
        Solstad et al. (2008). Must be in (0, 1).
    min_area : float, optional
        Smallest area of one field, in squared position units (for example
        cm² when ``env.units`` is ``"cm"``). Smaller fields are ignored. The
        default, None, is the 200 cm² of Solstad et al. (2008), converted to
        the environment's units; environments without metric length units
        (``env.units`` unset, pixels or custom units) need an explicit value.
        Pass 0 to keep every field.
    metric : {'geodesic', 'euclidean'}, default='geodesic'
        Distance from field bins to the nearest wall.

        - 'geodesic': straight-line distance from a wall-touching bin's center
          to its wall, plus the shortest path through the environment's
          connectivity graph. Respects obstacles and internal walls.
        - 'euclidean': straight-line distance from each bin center to the
          nearest wall.

        The two agree on an open rectangular arena.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Which walls to score against.

        - 'physical': the measured arena: the rings of its asserted footprint
          (for example ``Environment.from_polygon(outline, ...)``) and any
          declared internal walls. Every piece of wall must be owned by a
          retained bin with a finite firing rate: the bin whose cell touches
          it or, where a curved or diagonal wall runs through a grid cell
          that was not kept, the neighboring bin that owns positions there.
        - 'analysis': the outline of the retained bins (within the arena's
          domain, when known), plus declared internal walls. An approximation
          for arenas whose walls were not measured.

    Returns
    -------
    float
        Border score in [-1, 1]. NaN when the map has no positive finite peak,
        when no field reaches ``min_area``, or, with ``metric='geodesic'``,
        when a field bin has no path to a wall.

    Raises
    ------
    ValueError
        If an argument is invalid, or if ``min_area`` is None and the
        environment's units are not a metric length.
    NotImplementedError
        If the walls are unavailable: with ``boundary_source='physical'``, when
        the arena's footprint was not measured (or a combined component has
        none), or a wall is owned by no retained bin with a finite firing
        rate; with either source, when the bins are not exact planar cells.
        Missing observations are not evidence of zero firing; choose
        ``boundary_source='analysis'`` explicitly when the retained bins'
        outline is the intended measurement.

    Notes
    -----
    **Algorithm.**

    1. *Fields*: groups of neighboring bins (in the environment's
       connectivity graph) whose rate is at least ``threshold`` times the
       peak rate, each covering at least ``min_area``.
    2. *Walls*: each closed outline (an arena ring, or the retained bins'
       outline) is split into separate walls at its corners, where it turns
       by more than 50° within two bin widths. A rectangle has four walls; a
       circle, or any outline that only curves gently, is one wall. Each named
       internal wall (segments sharing a name) is one more wall.
    3. *Coverage* ``cM``: the largest fraction of one wall's length touched by
       the bins of one field, over all fields and walls.
    4. *Distance* ``dm``: the firing-rate-weighted mean distance from the bins
       of all fields to the nearest wall, divided by the radius of the
       largest circle that fits in the arena without crossing a wall (half
       the shorter side of a rectangle).

    **Interpretation.** A field covering one whole wall with all its firing
    against that wall scores near 1; a field in the middle of the arena
    scores -1. A cell that fires everywhere also scores high: on a square its
    single field covers every wall and its mean distance is a third of the
    half side, so ``b = 0.5``. The border score alone therefore does not
    identify a border cell; ``SpatialRatesResult.label_cell_types`` also
    requires spatial information, and a shuffle test is needed for a verdict.

    The usual cutoff of 0.5 was chosen for rectangular boxes. On other shapes
    the same field gives a different score: a circular arena is one wall, so
    a field along a quarter of it covers only 0.25 of its wall and scores
    lower than the same field along one side of a square. Calibrate a cutoff
    for non-rectangular arenas, for example against shuffled spike trains.

    **Comparison with BNT and opexebo.** On a rectangular arena this is the
    border score of the BNT (Moser lab) and opexebo implementations, with
    three differences. (1) Distances run from each bin's center to the wall,
    so a bin against the wall is half a bin away; BNT and opexebo count whole
    bins from the edge, so that bin is one bin away and their scores are
    lower by an amount that shrinks with bin size. (2) Fields follow Solstad
    et al. (2008) (``threshold=0.3``, ``min_area`` 200 cm²); BNT and opexebo
    take fields from their place-field detection (by default 0.2 of the
    peak, at least 9 bins and a peak of at least 1 Hz). (3) BNT and opexebo
    look past unvisited bins along a wall for the first visited bin; here an
    unvisited wall bin makes the physical score unavailable. This
    implementation also scores polygonal, circular and multi-room arenas and
    internal walls, which they do not.

    References
    ----------
    Solstad, T., Boccara, C. N., Kropff, E., Moser, M. B., & Moser, E. I.
        (2008). Representation of geometric borders in the entorhinal cortex.
        Science, 322(5909), 1865-1868.

    Examples
    --------
    A cell firing along the west wall of a 100 cm box:

    >>> import numpy as np
    >>> import shapely
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.border import border_score
    >>> env = Environment.from_polygon(
    ...     shapely.box(0, 0, 100, 100), bin_size=2.5, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> wall_cell = np.where(x < 7.5, 10.0, 0.2)
    >>> round(border_score(env, wall_cell), 2)
    0.87
    >>> center_cell = 0.2 + 10.0 * np.exp(
    ...     -np.sum((env.bin_centers - 50.0) ** 2, axis=1) / (2 * 10.0**2)
    ... )
    >>> round(border_score(env, center_cell), 2)
    -1.0

    See Also
    --------
    compute_region_coverage : Field coverage of named regions
    neurospatial.encoding.SpatialRatesResult.label_cell_types : Cell-type labels
    """
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )
    _validate_border_arguments(threshold, min_area, metric, boundary_source)

    from neurospatial.encoding._physical_border import (
        _border_score,
        _require_wall_rates,
        border_geometry,
    )

    geometry = border_geometry(env, boundary_source)
    if boundary_source == "physical":
        _require_wall_rates(geometry, firing_rate)
    min_area = _resolve_min_area(env, min_area)

    fields = _border_fields(env, firing_rate, threshold, min_area)
    if not fields:
        return np.nan
    return _border_score(fields, firing_rate, metric, geometry)


def _validate_border_arguments(
    threshold: float,
    min_area: float | None,
    metric: str,
    boundary_source: str,
) -> None:
    """Refuse invalid border-score arguments before any computation."""
    if not (0 < threshold < 1):
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}. "
            "Typically 0.3 (30% of peak)."
        )
    if min_area is not None and not (np.isfinite(min_area) and min_area >= 0):
        raise ValueError(f"min_area must be non-negative and finite, got {min_area}")
    if metric not in ("geodesic", "euclidean"):
        raise ValueError(f"metric must be 'geodesic' or 'euclidean', got '{metric}'")
    if boundary_source not in ("physical", "analysis"):
        raise ValueError("boundary_source must be 'physical' or 'analysis'.")


def _resolve_min_area(env: Environment, min_area: float | None) -> float:
    """``min_area``, or Solstad's 200 cm² in the environment's units."""
    if min_area is not None:
        return float(min_area)
    from neurospatial._units import units_per_centimeter

    per_cm = units_per_centimeter(env.units)
    if per_cm is None:
        raise ValueError(
            _format_error(
                "border_score needs min_area: the environment's units "
                f"({env.units!r}) are not a known length, so the 200 cm² minimum "
                "field area of Solstad et al. (2008) cannot be expressed in them.",
                why="Why: min_area is an area in squared position units.",
                fix="declare the units with env.with_metadata(units='cm'), or pass "
                "min_area=<smallest field area in squared position units> (0 keeps "
                "every field); for label_cell_types, pass border_scores computed "
                "that way",
            )
        )
    return BORDER_FIELD_MIN_AREA_CM2 * per_cm**2


def _border_fields(
    env: Environment,
    firing_rate: NDArray[np.float64],
    threshold: float,
    min_area: float,
) -> list[NDArray[np.intp]]:
    """Connected groups of bins at or above ``threshold`` of the peak rate.

    Groups smaller than ``min_area`` are dropped. Empty when the map has no
    positive finite peak.
    """
    finite = np.isfinite(firing_rate)
    if not finite.any():
        return []
    peak = np.max(firing_rate[finite])
    if not peak > 0:
        return []
    above = np.zeros(env.n_bins, dtype=bool)
    above[finite] = firing_rate[finite] >= threshold * peak
    pairs = env.binning.connections.pairs
    linked = pairs[above[pairs[:, 0]] & above[pairs[:, 1]]]
    graph = coo_array(
        (np.ones(len(linked)), (linked[:, 0], linked[:, 1])),
        shape=(env.n_bins, env.n_bins),
    )
    _, labels = connected_components(graph, directed=False)
    bins = np.flatnonzero(above)
    groups, group = np.unique(labels[bins], return_inverse=True)
    areas = np.bincount(group, weights=env.bin_sizes[bins], minlength=len(groups))
    return [bins[group == index] for index in np.flatnonzero(areas >= min_area)]


def compute_region_coverage(
    env: Environment,
    field_bins: NDArray[np.int64],
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
    env : Environment
        Spatial environment with defined regions.
    field_bins : array of int
        Bin indices comprising the field (e.g., from detect_place_fields).
    regions : list of str, optional
        List of region names to analyze. If None, analyzes all regions
        defined in env.regions.

    Returns
    -------
    coverage : dict[str, float]
        Dictionary mapping region name to coverage fraction [0, 1]:
        (number of region bins in field) / (total number of region bins).
        NaN for a region that contains no bins.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.border import compute_region_coverage
    >>> from shapely.geometry import box
    >>>
    >>> # Create environment with wall regions
    >>> positions = np.random.randn(5000, 2) * 20
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> from neurospatial.regions import Regions
    >>> regions = Regions(env.regions.values())
    >>> _ = regions.add("north", polygon=box(-20, 15, 20, 20))
    >>> _ = regions.add("south", polygon=box(-20, -20, 20, -15))
    >>> _ = regions.add("east", polygon=box(15, -20, 20, 20))
    >>> _ = regions.add("west", polygon=box(-20, -20, -15, 20))
    >>> env = env.with_regions(regions)
    >>>
    >>> # Create field along north wall
    >>> firing_rate = np.zeros(env.n_bins)
    >>> north_bins = np.where(env.region_mask("north"))[0]
    >>> firing_rate[north_bins] = 5.0
    >>> field_bins = np.where(firing_rate > 0)[0]
    >>>
    >>> # Compute coverage per wall
    >>> coverage = compute_region_coverage(env, field_bins)
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

    Coverage is the fraction of the region's bins that lie in the field,
    ``|region ∩ field| / |region|``. It is not symmetric: a field larger than
    the region can still cover it completely.
    """
    # Get region names to analyze
    if regions is None:
        regions = list(env.regions.keys())

    # Validate regions exist
    for region_name in regions:
        if region_name not in env.regions:
            raise RegionNotFoundError(
                region_name,
                available=list(env.regions),
                argument="regions",
                in_list=True,
            )

    # Compute coverage for each region
    coverage = {}
    for region_name in regions:
        # Get bins in this region
        region_mask = env.region_mask(region_name)
        region_bins = np.where(region_mask)[0]

        if len(region_bins) == 0:
            # No bins to cover: the fraction is undefined, not zero.
            coverage[region_name] = np.nan
        else:
            # Compute fraction of region bins that are in field
            coverage[region_name] = float(
                np.sum(np.isin(region_bins, field_bins)) / len(region_bins)
            )

    return coverage


__all__ = [
    "border_score",
    "compute_region_coverage",
]
