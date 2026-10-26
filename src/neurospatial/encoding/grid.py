"""
Grid cell analysis.

This module provides tools for analyzing grid cells: spatial autocorrelation
(2D FFT for regular grids, 1D radial profile for irregular topologies), grid
score (hexagonal periodicity), grid scale (spacing), grid orientation (folded
into ``GridProperties`` since v0.4), and periodicity score for
irregular topologies.

Implements spatial autocorrelation and grid score (Sargolini et al., 2006).

Imports
-------
All functions can be imported directly from this module:

>>> from neurospatial.encoding.grid import (
...     grid_score,
...     spatial_autocorrelation,
...     spatial_autocorrelation_radial,
...     grid_scale,
...     grid_properties,
...     periodicity_score,
...     GridProperties,
... )

Or via the encoding package:

>>> from neurospatial.encoding import grid_score, GridProperties

Functions
---------
spatial_autocorrelation
    Compute the 2D FFT autocorrelogram of a firing rate map (requires a
    regular 2D grid environment).
spatial_autocorrelation_radial
    Compute the 1D distance-binned autocorrelation profile via the
    connectivity graph (works on any topology).
grid_score
    Compute grid score (hexagonal periodicity) from a 2D autocorrelogram.
grid_scale
    Compute grid spacing from a 2D autocorrelogram.
grid_properties
    Compute all grid cell metrics (score, scale, orientation,
    orientation_std, peak_coords) from a 2D autocorrelogram.
periodicity_score
    Compute periodicity score from a distance-correlation profile
    (graph-based / radial).

Classes
-------
GridProperties
    Container dataclass for grid cell metrics.

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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import fft, ndimage
from scipy.signal import find_peaks

from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry import CartesianSystem

__all__ = [
    # Dataclass
    "GridProperties",
    # Functions
    "autocorrelation_pixel_size",
    "grid_properties",
    "grid_scale",
    "grid_score",
    "periodicity_score",
    "spatial_autocorrelation",
    "spatial_autocorrelation_radial",
]

# Hexagonal lattices repeat every 60 degrees, so orientation is an angle
# modulo pi / 3 radians.
_HEX_PERIOD = np.pi / 3
# Axis flip detection threshold for the orientation consensus: a quarter of
# the period. When angles cluster near +/-15 degrees of the 0 / 60 degree
# boundary, the axis flip correction is applied.
_AXIS_FLIP_THRESHOLD = _HEX_PERIOD / 4


@dataclass(repr=False)
class GridProperties(ResultMixin):
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
        Grid orientation in radians, range [0, pi/3): the angle of the field
        lattice in the environment's allocentric frame, from +x (East)
        counterclockwise toward +y (North), modulo pi/3, the period of a
        hexagonal lattice.
    orientation_std : float
        Standard deviation of the orientation estimates of the peaks, in
        radians. Lower values indicate a more consistent orientation.
    peak_coords : NDArray[np.float64]
        Coordinates of detected peaks relative to autocorrelogram center,
        shape (n_peaks, 2) where columns are ``(x_offset, y_offset)`` in
        pixels of the autocorrelogram, along the environment's x and y
        (axis 0 and axis 1 of the map). Typically 6 peaks for a well-formed
        grid.

        .. note::

           **Breaking change in v0.4**: this attribute previously held
           ``(row_offset, col_offset)`` (image-array convention with the
           y-axis flipped). It now matches the rest of the package's
           Cartesian ``(x, y)`` convention. Downstream code that read
           ``peak_coords[:, 0]`` as a row index must swap to
           ``peak_coords[:, 1]`` (``y``).
    n_peaks : int
        Number of peaks detected. Grid cells typically have 6.

    Notes
    -----
    All metrics are computed from a single peak detection pass for efficiency.
    If fewer than 3 peaks are detected, scale and orientation will be NaN.

    Examples
    --------
    >>> from neurospatial.encoding.grid import (
    ...     grid_properties,
    ...     spatial_autocorrelation,
    ... )  # doctest: +SKIP
    >>> autocorr = spatial_autocorrelation(env, firing_rate)  # doctest: +SKIP
    >>> props = grid_properties(autocorr, bin_size=2.0)  # doctest: +SKIP
    >>> print(
    ...     f"Score: {props.score:.2f}, Scale: {props.scale:.1f} cm"
    ... )  # doctest: +SKIP
    Score: 0.85, Scale: 42.3 cm  # doctest: +SKIP
    """

    score: float
    scale: float
    orientation: float
    orientation_std: float
    peak_coords: NDArray[np.float64]
    n_peaks: int

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "scale": self.scale,
            "orientation": self.orientation,
            "orientation_std": self.orientation_std,
            "n_peaks": self.n_peaks,
        }


def spatial_autocorrelation(
    env: Environment[CartesianSystem],
    firing_rate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the 2D spatial autocorrelogram of a firing rate map.

    Spatial autocorrelation measures the similarity between a cell's firing
    pattern and shifted versions of itself; for grid cells this reveals the
    characteristic hexagonal periodicity. This entry point computes the 2D
    autocorrelogram that :func:`grid_score` and :func:`grid_properties`
    read. For environments without a regular 2D grid of rectangular cells,
    use :func:`spatial_autocorrelation_radial` instead, which returns a 1D
    distance-correlation profile from the connectivity graph.

    Parameters
    ----------
    env : Environment
        Spatial environment with a regular 2D grid layout.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second). NaN marks unvisited
        bins.

    Returns
    -------
    autocorr : NDArray[np.float64], shape (2 * n_x - 1, 2 * n_y - 1)
        Pearson correlation of the map with itself at every shift, with
        square pixels; each pixel side is :func:`autocorrelation_pixel_size`,
        the smaller cell width. Zero lag is the center pixel and lags run to
        the full extent of the map in each direction, so for an ``(n_x,
        n_y)`` grid the map is ``(2 * n_x - 1, 2 * n_y - 1)``. Values in
        [-1, 1]; NaN at shifts where fewer than 20 pairs of visited bins
        overlap. When the two cell widths differ (by more than 1/100 of a
        cell across the grid) the lags along the wider axis are interpolated
        onto the finer spacing, so the map is larger along that axis.

    Raises
    ------
    ValueError
        If ``firing_rate.shape != (env.n_bins,)``.
        If all firing rates are NaN or constant.
    NotImplementedError
        If ``env`` is not a regular 2D grid of rectangular cells, one width
        along each axis, with at least half of them active, for example a
        hexagonal, mesh, track or 3D environment, or a grid with uneven edges
        (use :func:`spatial_autocorrelation_radial` instead).
    CartesianOnlyError
        If ``env`` is polar: a ``(distance, angle)`` grid is not a Cartesian
        lattice, so its 2D autocorrelogram and grid score are meaningless.

    Notes
    -----
    At each shift ``(dx, dy)`` the map is overlaid on a copy of itself moved
    by that many cells, without wrapping around the edges, and the value is
    the Pearson correlation over the pairs of bins that are both visited.
    Unvisited bins (NaN) and inactive grid cells are left out of each pair
    set rather than counted as zero firing, so a gap in coverage does not
    pull correlations toward zero. Shifts with fewer than 20 overlapping
    pairs are NaN: near the edge of the map so few bins overlap that their
    correlation is mostly noise. The six
    sums each correlation needs (pair count, the sums and sums of squares of
    both sides, and the sum of products) are computed for all shifts at once
    with zero-padded FFTs.

    Interpretation:

    - **Hexagonal grid cells**: 6-fold rotational symmetry around the
      central peak.
    - **Place cells**: Typically a single central peak, no periodicity.
    - **Non-spatial cells**: Flat or noisy autocorrelation.

    Complexity: ``O(n log n)`` where ``n = n_bins``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.grid import spatial_autocorrelation
    >>> edges = np.arange(0.0, 42.0, 2.0)  # 40 cm box, 2 cm bins
    >>> env = Environment.from_grid_mask(
    ...     np.ones((20, 20), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> rate = np.random.default_rng(0).gamma(2.0, 1.0, env.n_bins)
    >>> autocorr = spatial_autocorrelation(env, rate)
    >>> autocorr.shape
    (39, 39)
    >>> float(autocorr[19, 19])  # zero lag
    1.0

    See Also
    --------
    grid_score : Compute grid score from this 2D autocorrelogram.
    spatial_autocorrelation_radial : 1D distance-correlation profile
        for irregular environments.

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    _require_fft_grid(env, "spatial_autocorrelation()")
    _validate_autocorr_input(env, firing_rate)

    rate_map = np.full(env.grid_shape, np.nan)
    rate_map[env.active_mask] = firing_rate
    autocorr = _linear_autocorrelation(rate_map)
    cell_widths = tuple(float(np.median(np.diff(edge))) for edge in env.grid_edges)
    return _square_lags(autocorr, cell_widths)


def _require_fft_grid(env: Environment, operation: str) -> None:
    """Refuse environments without a regular 2D grid for the FFT lag map.

    Parameters
    ----------
    env : Environment
        Environment to check.
    operation : str
        The refused call, named in the error.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    NotImplementedError
        If ``env`` has no 2D grid of rectangular cells, its cells along an
        axis are not one width, or fewer than half of its grid cells are
        active.
    """
    require_cartesian(
        env.coordinate_system,
        operation,
        why="The autocorrelogram shifts the rate map along Cartesian x and y.",
        fix=(
            "compute it on an allocentric environment, for example "
            "Environment.from_samples(positions, bin_size=...)"
        ),
    )
    if _detect_grid_method(env) == "fft":
        return
    if len(env.grid_edges) != 2:
        found = (
            f"this environment's {env.layout_type} layout has no 2D grid of "
            "rectangular cells"
        )
    elif not _uniform_spacing(env.grid_edges):
        axis = next(
            axis
            for axis, edge in enumerate(env.grid_edges)
            if not _uniform_spacing((edge,))
        )
        widths = np.diff(env.grid_edges[axis])
        found = (
            f"its cells along axis {axis} are not all one width: widths "
            f"range from {widths.min():.6g} to {widths.max():.6g}"
        )
    else:
        found = f"only {env.n_bins} of its {env.active_mask.size} grid cells are active"
    raise NotImplementedError(
        _format_error(
            f"{operation} needs a regular 2D grid of rectangular cells, one width "
            f"along each axis, with at least half of them active; {found}.",
            why=(
                "Why: the FFT shifts the rate map one cell at a time along "
                "rows and columns, so each shift is one distance only when "
                "the cells along an axis share one width."
            ),
            fix=(
                "use spatial_autocorrelation_radial(env, firing_rate), which "
                "works on every layout, with periodicity_score"
            ),
        )
    )


def autocorrelation_pixel_size(env: Environment) -> float:
    """Physical side of the square pixels of :func:`spatial_autocorrelation`.

    The autocorrelogram's pixels are square, with the smaller of the grid's
    two cell widths as their side, so a distance in pixels times this size is
    a distance in the environment's units. Pass it as ``bin_size`` to
    :func:`grid_scale` and :func:`grid_properties`.

    Parameters
    ----------
    env : Environment
        Environment with a regular 2D grid, as for
        :func:`spatial_autocorrelation`.

    Returns
    -------
    float
        Pixel side in the environment's units, e.g. cm.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    NotImplementedError
        If :func:`spatial_autocorrelation` refuses ``env``: no regular 2D
        grid of rectangular cells (hexagonal, mesh, track or 3D layouts),
        uneven cells along an axis, or fewer than half of the cells active.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.grid import autocorrelation_pixel_size
    >>> mask = np.ones((4, 5), dtype=bool)
    >>> edges = (np.linspace(0.0, 12.0, 5), np.linspace(0.0, 10.0, 6))
    >>> env = Environment.from_grid_mask(mask, grid_edges=edges)
    >>> autocorrelation_pixel_size(env)
    2.0
    """
    _require_fft_grid(env, "autocorrelation_pixel_size()")
    return min(float(np.median(np.diff(edge))) for edge in env.grid_edges)


# Below this many pixels of accumulated lag error, treating the two cell widths
# as one changes no lag by more than 1/100 of a cell, so the autocorrelogram is
# used as computed. Grids fitted to a recorded range (from_samples) can differ
# between axes by a few percent and are then resampled.
_SQUARE_LAG_TOLERANCE = 0.01


def _square_lags(
    autocorr: NDArray[np.float64], widths: tuple[float, ...]
) -> NDArray[np.float64]:
    """Resample a lag map with per-axis cell widths onto square pixels.

    Parameters
    ----------
    autocorr : ndarray of float64, shape (n_x, n_y)
        Lag map in cell steps, zero lag at index ``n // 2`` along each axis.
    widths : tuple of float
        Physical cell width along each axis.

    Returns
    -------
    ndarray of float64
        Lag map with square pixels whose side is ``min(widths)``, zero lag at
        the center. Returned unchanged when the two widths are equal to within
        ``_SQUARE_LAG_TOLERANCE`` pixels across the map.
    """
    side = min(widths)
    shape = autocorr.shape
    error = max(
        n * (width / side - 1.0) for n, width in zip(shape, widths, strict=True)
    )
    if error <= _SQUARE_LAG_TOLERANCE:
        return autocorr
    centers = [n // 2 for n in shape]
    axes = []
    for n, c, width in zip(shape, centers, widths, strict=True):
        below = int(np.floor(c * width / side + 1e-9))
        above = int(np.floor((n - 1 - c) * width / side + 1e-9))
        # Keep the parity of the computed map, so zero lag sits where
        # fftshift put it: index size // 2, with one more lag below zero
        # than above it on an even axis.
        odd = n % 2
        reach = min(below, above + 1 - odd)
        lags = np.arange(-reach, reach + odd)
        axes.append(c + lags * side / width)
    coordinates = np.meshgrid(*axes, indexing="ij")
    squared = ndimage.map_coordinates(autocorr, coordinates, order=1, mode="nearest")
    return np.asarray(squared, dtype=np.float64)


def spatial_autocorrelation_radial(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    max_distance: float | None = None,
    n_distance_bins: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute 1D distance-binned spatial autocorrelation.

    For each distance bin, computes the Pearson correlation between firing
    rates of bin pairs separated by that geodesic distance. Use this on
    irregular topologies (graphs, masked grids) where the FFT-based
    :func:`spatial_autocorrelation` is undefined.

    Parameters
    ----------
    env : Environment
        Spatial environment containing bin centers and connectivity.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Spatial firing rate map (Hz or spikes/second).
    max_distance : float, optional
        Maximum geodesic distance to consider. If None, uses the environment's
        maximum extent.
    n_distance_bins : int, default=50
        Number of distance bins.

    Returns
    -------
    distances : NDArray[np.float64], shape (n_distance_bins,)
        Distance bin centers.
    correlations : NDArray[np.float64], shape (n_distance_bins,)
        Autocorrelation at each distance.

    Raises
    ------
    ValueError
        If ``firing_rate.shape != (env.n_bins,)``.
        If ``n_distance_bins <= 0``.
        If all firing rates are NaN or constant.

    See Also
    --------
    spatial_autocorrelation : 2D FFT-based autocorrelogram for regular grids.
    """
    _validate_autocorr_input(env, firing_rate)

    if n_distance_bins <= 0:
        raise ValueError(f"n_distance_bins must be positive, got {n_distance_bins}")

    return _spatial_autocorrelation_graph(
        env,
        firing_rate,
        max_distance=max_distance,
        n_distance_bins=n_distance_bins,
    )


def _validate_autocorr_input(
    env: Environment, firing_rate: NDArray[np.float64]
) -> None:
    """Shared input validation for the public autocorrelation entry points.

    Pulled out so :func:`spatial_autocorrelation` and
    :func:`spatial_autocorrelation_radial` apply the same shape /
    legitimate-NaN checks before dispatching to the method-specific
    implementation.
    """
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate.shape must be ({env.n_bins},), got {firing_rate.shape}"
        )
    if np.all(np.isnan(firing_rate)):
        raise ValueError("All firing rates are NaN")
    valid_rates = firing_rate[np.isfinite(firing_rate)]
    if len(valid_rates) > 0 and np.all(valid_rates == valid_rates[0]):
        raise ValueError(
            "All valid firing rates are constant. Autocorrelation undefined."
        )


def _detect_grid_method(env: Environment) -> Literal["fft", "graph"]:
    """
    Detect whether environment supports FFT-based autocorrelation.

    Parameters
    ----------
    env : Environment
        Environment to check.

    Returns
    -------
    method : {'fft', 'graph'}
        'fft' if environment is a regular 2D grid of rectangular cells,
        'graph' otherwise.
    """
    # Hexagonal lattices also have a 2D grid shape, but no rectangular cell
    # edges: their rows are offset, so an FFT over them is not a 2D lag map.
    if len(env.grid_edges) != 2:
        return "graph"
    # A lag of k cells is a distance only when the cells along an axis share
    # one width; the two axes may differ, and the lags are rescaled to them.
    if not _uniform_spacing(env.grid_edges):
        return "graph"

    # Check if grid is reasonably regular (not too sparse): if less than 50%
    # of bins are active, treat it as irregular.
    active_mask = env.active_mask
    if np.sum(active_mask) / active_mask.size < 0.5:
        return "graph"

    return "fft"


# Cell widths may differ by this fraction of the median width and still count
# as one width. Edges computed as origin + i * width carry rounding errors of a
# few ulp of the coordinates, far below this; a relative width error of 1e-6
# shifts the farthest lag of a 10,000-cell axis by at most 0.01 cell.
_SPACING_RTOL = 1e-6


def _uniform_spacing(edges: tuple[NDArray[np.float64], ...]) -> bool:
    """Whether the cells along each axis share one width; axes may differ.

    Parameters
    ----------
    edges : tuple of ndarray
        Cell edges along each axis, ``env.grid_edges``.
    """
    for edge in edges:
        widths = np.diff(edge)
        reference = float(np.median(widths))
        if np.max(np.abs(widths - reference)) > _SPACING_RTOL * reference:
            return False
    return True


# A lag's correlation needs at least this many pairs of visited bins that
# overlap at that lag; fewer leave it NaN. Near the map's edge the overlap
# shrinks to a few bins whose correlation is mostly noise.
_MIN_OVERLAP_BINS = 20


def _linear_autocorrelation(rate_map: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson correlation of a rate map with itself at every 2D lag.

    For each lag the map is shifted against itself without wrap-around, and
    the correlation is taken over the pairs of bins that are both finite at
    that lag. Missing bins (NaN) are left out of every sum, not set to zero.

    Parameters
    ----------
    rate_map : ndarray of float64, shape (n_x, n_y)
        Rate map on the grid, NaN at unvisited or inactive bins.

    Returns
    -------
    ndarray of float64, shape (2 * n_x - 1, 2 * n_y - 1)
        Correlation at lag ``(i - (n_x - 1), j - (n_y - 1))`` at index
        ``(i, j)``: zero lag is the center. NaN where fewer than
        ``_MIN_OVERLAP_BINS`` pairs overlap or either side of the overlap is
        constant.
    """
    valid = np.isfinite(rate_map)
    # Standardizing first keeps the summed moments near one, so the
    # differences of sums below lose no precision; Pearson correlation does
    # not change under it.
    values = rate_map[valid]
    standardized = np.where(valid, (rate_map - values.mean()) / values.std(), 0.0)
    shape = tuple(2 * n - 1 for n in rate_map.shape)
    padded = tuple(fft.next_fast_len(n, real=True) for n in shape)

    def spectrum(array: NDArray[np.float64]) -> NDArray[np.complex128]:
        return np.asarray(fft.rfft2(array, s=padded))

    def overlap_sum(
        first: NDArray[np.complex128], second: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        # sum_i a[i] * b[i + lag] over the zero-padded maps, for every lag up
        # to n - 1 in each direction; the padding leaves no wrap-around.
        circular = fft.irfft2(np.conj(first) * second, s=padded)
        rows = np.r_[padded[0] - shape[0] // 2 : padded[0], : shape[0] // 2 + 1]
        cols = np.r_[padded[1] - shape[1] // 2 : padded[1], : shape[1] // 2 + 1]
        return np.asarray(circular[np.ix_(rows, cols)], dtype=np.float64)

    mask_f = spectrum(valid.astype(np.float64))
    x_f = spectrum(standardized)
    xx_f = spectrum(standardized**2)
    n_pairs = np.rint(overlap_sum(mask_f, mask_f))
    sum_first = overlap_sum(x_f, mask_f)
    sum_second = overlap_sum(mask_f, x_f)
    sum_squares_first = overlap_sum(xx_f, mask_f)
    sum_squares_second = overlap_sum(mask_f, xx_f)
    sum_products = overlap_sum(x_f, x_f)

    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = n_pairs * sum_products - sum_first * sum_second
        variance_first = n_pairs * sum_squares_first - sum_first**2
        variance_second = n_pairs * sum_squares_second - sum_second**2
        # A side of the overlap whose values are all equal has a variance of
        # rounding error only: its correlation is undefined.
        floor = 1e-9 * n_pairs**2
        defined = (
            (n_pairs >= _MIN_OVERLAP_BINS)
            & (variance_first > floor)
            & (variance_second > floor)
        )
        autocorr = covariance / np.sqrt(variance_first * variance_second)
    return np.where(defined, np.clip(autocorr, -1.0, 1.0), np.nan)


def _spatial_autocorrelation_graph(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    max_distance: float | None = None,
    n_distance_bins: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute distance-based autocorrelation using graph geodesic distances.

    Parameters
    ----------
    env : Environment
        Environment.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        Firing rate map.
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
    from neurospatial.ops.distance import pairwise_distances

    # Filter out NaN bins
    valid_bins = np.where(np.isfinite(firing_rate))[0]

    if len(valid_bins) < 2:
        raise ValueError("Need at least 2 valid (non-NaN) bins")

    valid_rates = firing_rate[valid_bins]
    n_valid = len(valid_bins)

    dist_matrix = pairwise_distances(env.connectivity, valid_bins)  # (n_valid, n_valid)

    # Vectorized extraction of upper triangle (avoid nested loops)
    triu_indices = np.triu_indices(n_valid, k=1)
    pair_distances = dist_matrix[triu_indices]
    rates_i = valid_rates[triu_indices[0]]
    rates_j = valid_rates[triu_indices[1]]

    # Filter out infinite distances (disconnected pairs)
    finite_mask = np.isfinite(pair_distances)
    pair_distances = pair_distances[finite_mask]
    rates_i = rates_i[finite_mask]
    rates_j = rates_j[finite_mask]

    if len(pair_distances) == 0:
        raise ValueError("No valid bin pairs found (graph may be disconnected)")

    # Determine max distance
    if max_distance is None:
        max_distance = float(np.max(pair_distances))

    # Create distance bins
    distance_bin_edges = np.linspace(0, max_distance, n_distance_bins + 1)
    distance_bin_centers = (distance_bin_edges[:-1] + distance_bin_edges[1:]) / 2

    # Vectorized bin assignment using searchsorted
    # Match original behavior: d_min <= d < d_max (excludes values at exactly max_distance)
    bin_indices = np.searchsorted(distance_bin_edges, pair_distances, side="right") - 1
    # Values at or beyond last edge get bin_indices >= n_distance_bins, mark as invalid
    bin_indices = np.where(bin_indices >= n_distance_bins, -1, bin_indices)
    # Values below first edge also invalid
    bin_indices = np.where(bin_indices < 0, -1, bin_indices)

    # Compute correlation for each distance bin using optimized grouped approach
    # Sort by bin index for efficient contiguous memory access
    valid_bin_mask = bin_indices >= 0
    if not np.any(valid_bin_mask):
        return distance_bin_centers, np.full(n_distance_bins, np.nan)

    sort_idx = np.argsort(bin_indices[valid_bin_mask])
    sorted_bins = bin_indices[valid_bin_mask][sort_idx]
    sorted_rates_i = rates_i[valid_bin_mask][sort_idx]
    sorted_rates_j = rates_j[valid_bin_mask][sort_idx]

    # Find bin boundaries using unique (sorted data makes this efficient)
    unique_bins, bin_counts = np.unique(sorted_bins, return_counts=True)
    bin_starts = np.concatenate([[0], np.cumsum(bin_counts)[:-1]])

    correlations = np.full(n_distance_bins, np.nan)
    for b, start, count in zip(unique_bins, bin_starts, bin_counts, strict=True):
        if count < 2:
            # Not enough pairs for correlation
            continue
        end = start + count
        ri = sorted_rates_i[start:end]
        rj = sorted_rates_j[start:end]

        # Check for zero variance
        if np.std(ri) == 0 or np.std(rj) == 0:
            # No variance, correlation undefined
            continue

        # Inline Pearson correlation (faster than scipy.stats.pearsonr for small arrays)
        ri_centered = ri - ri.mean()
        rj_centered = rj - rj.mean()
        corr = np.dot(ri_centered, rj_centered) / (
            np.linalg.norm(ri_centered) * np.linalg.norm(rj_centered)
        )
        correlations[b] = corr

    return distance_bin_centers, correlations


def grid_score(autocorr_2d: NDArray[np.float64]) -> float:
    """
    Compute grid score from 2D spatial autocorrelation.

    Grid score quantifies hexagonal periodicity by measuring rotational symmetry
    in the autocorrelogram. Hexagonal grids show high correlation at 60° and 120°
    rotations (hexagon vertices) and low correlation at 30°, 90°, 150° rotations.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (n_x, n_y)
        2D spatial autocorrelogram, as :func:`spatial_autocorrelation` returns
        it: zero lag at index ``n // 2`` along each axis, NaN at lags without
        a correlation. An even axis's first entry, the most negative lag, has
        no positive partner and is not used, so rotations are about the
        zero-lag pixel.

    Returns
    -------
    score : float
        Grid score in range [-2, 2]. Higher values indicate stronger hexagonal
        periodicity. NaN when the autocorrelogram has no central peak
        bounded inside the map (no correlation above the central-peak
        threshold at zero lag, or a region that reaches the map's edge or its
        undefined lags), or when fewer than three ring radii fit between the
        central peak and the edge of the defined lags.

    Notes
    -----
    **Algorithm**, the expanding-annulus grid score of the Moser group's
    analysis toolbox (BNT ``gridnessScore``) and opexebo:

    1. Central peak: the connected region of lags around zero lag where the
       correlation exceeds 0.2 of its zero-lag value. Its inner radius is
       the radius of a disk of the same area, so it grows with the size of
       the firing fields.
    2. Rotate the autocorrelogram by 30°, 60°, 90°, 120° and 150° about zero
       lag.
    3. For each outer radius from just outside the central peak to the edge
       of the map, in steps of one pixel, take the annulus between the
       central peak and that radius, and compute the Pearson correlation
       between the autocorrelogram and each rotation over the lags defined
       in the map and in all five rotations:
       ``min(r60, r120) - max(r30, r90, r150)``.
    4. The score is the largest mean of three consecutive radii.

    Sweeping the outer radius makes the score independent of the grid
    spacing and of the arena size, as long as the first ring of peaks lies
    inside the map; a fixed annulus misses the ring of a widely spaced grid.

    **Interpretation**:

    - **score > 0.4**: hexagonal grid (a common grid-cell threshold)
    - **score ≈ 0**: no hexagonal structure (place cells, non-spatial cells)
    - **score < 0**: four-fold or other non-hexagonal symmetry

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.grid import grid_score, spatial_autocorrelation
    >>> edges = np.arange(0.0, 102.5, 2.5)  # 100 cm box, 2.5 cm bins
    >>> env = Environment.from_grid_mask(
    ...     np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> wavenumber = 4 * np.pi / (np.sqrt(3) * 40.0)  # 40 cm grid spacing
    >>> rate = sum(
    ...     np.cos(wavenumber * env.bin_centers @ [np.cos(a), np.sin(a)])
    ...     for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
    ... )
    >>> score = grid_score(spatial_autocorrelation(env, np.exp(rate)))
    >>> bool(score > 0.4)
    True

    See Also
    --------
    spatial_autocorrelation : Compute 2D autocorrelation map
    grid_properties : Score, spacing and orientation together

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")

    centered = _zero_lag_centered(np.asarray(autocorr_2d, dtype=np.float64))
    central_radius = _central_peak_radius(centered)
    if np.isnan(central_radius):
        return np.nan
    edge = (min(centered.shape) - 1) // 2
    radii = np.arange(int(np.ceil(_MIN_OUTER_RADIUS * central_radius)), edge + 1)
    if radii.size < _N_SCORE_RADII:
        return np.nan

    distance = _lag_distance(centered.shape)
    used = np.isfinite(centered) & (distance > central_radius) & (distance <= edge)
    rotations = []
    for angle in _ROTATION_ANGLES:
        rotated, defined = _rotate_about_zero_lag(centered, angle)
        rotations.append(rotated)
        used &= defined

    # Pearson correlation in every annulus at once: cumulative sums over the
    # lags sorted by distance, read at each outer radius.
    order = np.argsort(distance[used], kind="stable")
    x = centered[used][order]
    end = np.searchsorted(distance[used][order], radii, side="right")
    n = end.astype(np.float64)

    def at_radii(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([[0.0], np.cumsum(values)])[end]

    sum_x, sum_xx = at_radii(x), at_radii(x**2)
    correlations = []
    for rotated in rotations:
        y = rotated[used][order]
        with np.errstate(divide="ignore", invalid="ignore"):
            covariance = n * at_radii(x * y) - sum_x * at_radii(y)
            variance = (n * sum_xx - sum_x**2) * (n * at_radii(y**2) - at_radii(y) ** 2)
            correlations.append(
                np.where(variance > 0, covariance / np.sqrt(variance), np.nan)
            )
    r30, r60, r90, r120, r150 = correlations
    per_radius = np.minimum(r60, r120) - np.maximum(np.maximum(r30, r90), r150)

    window = np.lib.stride_tricks.sliding_window_view(per_radius, _N_SCORE_RADII)
    means = window.mean(axis=1)
    if not np.any(np.isfinite(means)):
        return np.nan
    return float(np.nanmax(means))


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
        Distance bin centers (from spatial_autocorrelation_radial; 1D distance profile).
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
    >>> import numpy as np  # doctest: +SKIP
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.encoding.grid import (  # doctest: +SKIP
    ...     spatial_autocorrelation,
    ...     periodicity_score,
    ... )
    >>>
    >>> # Create environment
    >>> positions = np.random.randn(5000, 2) * 20  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>>
    >>> # Create grid cell firing pattern
    >>> firing_rate = np.zeros(env.n_bins)  # doctest: +SKIP
    >>> # ... populate with grid cell firing ...
    >>>
    >>> # Compute graph-based autocorrelation
    >>> distances, correlations = spatial_autocorrelation(  # doctest: +SKIP
    ...     env, firing_rate, n_distance_bins=50
    ... )
    >>>
    >>> # Compute periodicity score
    >>> score = periodicity_score(distances, correlations)  # doctest: +SKIP
    >>> print(f"Periodicity score: {score:.3f}")  # doctest: +SKIP
    Periodicity score: 0.845  # doctest: +SKIP

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


def _zero_lag_centered(autocorr_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center an autocorrelogram on its zero-lag pixel.

    Parameters
    ----------
    autocorr_2d : ndarray of float64, shape (n_0, n_1)
        Lag map with zero lag at index ``n // 2`` along each axis, as
        ``scipy.fft.fftshift`` and :func:`spatial_autocorrelation` place it.

    Returns
    -------
    ndarray of float64
        View with every axis of odd length and zero lag at its middle index
        ``(n - 1) / 2``. An even axis loses its first entry, the most
        negative lag, which has no positive partner.
    """
    return autocorr_2d[
        tuple(slice(1, None) if n % 2 == 0 else slice(None) for n in autocorr_2d.shape)
    ]


# Rotations compared by the grid score, in degrees: a hexagonal lattice maps
# onto itself at 60 and 120 and away from itself at 30, 90 and 150.
_ROTATION_ANGLES = (30.0, 60.0, 90.0, 120.0, 150.0)
# The grid score is the largest mean over this many consecutive annulus radii.
_N_SCORE_RADII = 3
# The smallest outer annulus radius, in central-peak radii. A neighboring
# field lies at least one central-peak diameter from zero lag, so a narrower
# annulus holds only the shoulder of the central peak.
_MIN_OUTER_RADIUS = 2.0
# The central peak is the region around zero lag where the correlation exceeds
# this fraction of its zero-lag value.
_CENTRAL_PEAK_THRESHOLD = 0.2


def _lag_distance(shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Distance in pixels of every lag from the center of an odd-sized map."""
    offsets = np.ogrid[tuple(slice(-(n // 2), n // 2 + 1) for n in shape)]
    return np.asarray(np.sqrt(sum(offset**2 for offset in offsets)), dtype=np.float64)


def _central_peak_radius(centered: NDArray[np.float64]) -> float:
    """Radius in pixels of the central peak of a zero-lag-centered map.

    The central peak is the set of lags reached from zero lag by stepping to
    4-connected neighbors whose correlation does not rise and stays above
    ``_CENTRAL_PEAK_THRESHOLD`` of the zero-lag value. A neighboring field
    rises out of the dip around the central peak, so it is not included even
    when that dip stays above the threshold. The radius is that of a disk of
    the same area.

    Parameters
    ----------
    centered : ndarray of float64, shape (n_x, n_y)
        Autocorrelogram with odd axes and zero lag at the middle index.

    Returns
    -------
    float
        Equivalent radius in pixels, or NaN when the zero-lag correlation is
        not positive or the peak reaches the map's edge or an undefined
        (NaN) lag, so its extent is unknown.
    """
    center = tuple(n // 2 + 1 for n in centered.shape)
    # A NaN border marks the map's edge like an undefined lag.
    values = np.pad(centered, 1, constant_values=np.nan)
    peak = values[center]
    if not peak > 0:
        return np.nan
    above = np.nan_to_num(values, nan=-np.inf) > _CENTRAL_PEAK_THRESHOLD * peak
    region = np.zeros(values.shape, dtype=bool)
    region[center] = True
    while True:
        downhill = np.zeros_like(region)
        for axis in (0, 1):
            for step in (1, -1):
                from_region = np.roll(region, step, axis=axis)
                downhill |= from_region & (values <= np.roll(values, step, axis=axis))
        grown = downhill & above & ~region
        if not grown.any():
            break
        region |= grown
    if np.any(ndimage.binary_dilation(region) & np.isnan(values)):
        return np.nan
    return float(np.sqrt(np.count_nonzero(region) / np.pi))


def _rotate_about_zero_lag(
    centered: NDArray[np.float64], angle: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Rotate a zero-lag-centered map, marking lags the rotation leaves undefined.

    Parameters
    ----------
    centered : ndarray of float64, shape (n_x, n_y)
        Map with odd axes and zero lag at the middle index; NaN where
        undefined.
    angle : float
        Rotation in degrees.

    Returns
    -------
    rotated : ndarray of float64, shape (n_x, n_y)
        Bilinear rotation about the zero-lag pixel.
    defined : ndarray of bool, shape (n_x, n_y)
        Lags whose interpolated value uses only defined lags inside the map.
    """
    valid = np.isfinite(centered)
    rotated = ndimage.rotate(
        np.where(valid, centered, 0.0), angle, reshape=False, order=1, cval=0.0
    )
    coverage = ndimage.rotate(
        valid.astype(np.float64), angle, reshape=False, order=1, cval=0.0
    )
    return np.asarray(rotated, dtype=np.float64), coverage > 1.0 - 1e-9


def _find_autocorr_peaks(
    autocorr_2d: NDArray[np.float64],
    *,
    threshold_rel: float,
    max_peaks: int,
) -> NDArray[np.float64]:
    """
    Find the peaks of a 2D autocorrelogram nearest its central peak.

    A peak is a local maximum outside the central peak (see
    :func:`grid_score`) whose correlation is at least ``threshold_rel`` times
    the zero-lag correlation. Peaks closer together than the central peak's
    radius are one field, of which the higher is kept. Both lengths come from
    the field size in the map, so the search does not depend on the pixel
    size.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (n_x, n_y)
        2D autocorrelogram with zero lag at index ``n // 2`` along each axis;
        NaN at undefined lags.
    threshold_rel : float
        Minimum peak correlation as a fraction of the zero-lag correlation.
    max_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    peak_coords : NDArray[np.float64], shape (n_peaks, 2)
        Peak coordinates relative to zero lag, in pixels, as
        ``(x_offset, y_offset)``: axis 0 of the map runs along the
        environment's x and axis 1 along y, as :func:`spatial_autocorrelation`
        lays it out. Sorted by distance from zero lag (closest first). Empty
        when the map has no bounded central peak.
    """
    from skimage.feature import peak_local_max

    centered = _zero_lag_centered(np.asarray(autocorr_2d, dtype=np.float64))
    central_radius = _central_peak_radius(centered)
    if np.isnan(central_radius):
        return np.empty((0, 2))
    center = np.array([n // 2 for n in centered.shape])
    valid = np.isfinite(centered)
    floor = np.min(centered[valid]) - 1.0
    peaks = peak_local_max(
        np.where(valid, centered, floor),
        min_distance=max(1, int(np.floor(central_radius))),
        threshold_abs=threshold_rel * centered[tuple(center)],
        exclude_border=False,
    )
    offsets = peaks.astype(np.float64) - center
    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    outside = distances > central_radius
    order = np.argsort(distances[outside], kind="stable")[:max_peaks]
    return np.asarray(offsets[outside][order], dtype=np.float64)


def _validate_peak_arguments(
    autocorr_2d: NDArray[np.float64],
    bin_size: float,
    n_peaks: int,
    threshold_rel: float,
) -> None:
    """Check the arguments shared by :func:`grid_scale` and :func:`grid_properties`."""
    if autocorr_2d.ndim != 2:
        raise ValueError(f"autocorr_2d must be 2D array, got {autocorr_2d.ndim}D")
    if not bin_size > 0:
        raise ValueError(
            f"bin_size must be positive (env units, e.g., cm), got {bin_size}. "
            "Pass autocorrelation_pixel_size(env), the side of the "
            "autocorrelogram's pixels."
        )
    if isinstance(n_peaks, bool) or not isinstance(n_peaks, (int, np.integer)):
        raise TypeError(f"n_peaks must be an integer, got {n_peaks!r}")
    if n_peaks < 3:
        raise ValueError(
            f"n_peaks must be >= 3, got {n_peaks}: the spacing is the median "
            "distance to at least three peaks of the first ring."
        )
    if not 0 <= threshold_rel < 1:
        raise ValueError(
            f"threshold_rel must be in [0, 1), got {threshold_rel}: it is the "
            "minimum peak correlation as a fraction of the zero-lag correlation."
        )


def grid_scale(
    autocorr_2d: NDArray[np.float64],
    *,
    bin_size: float,
    n_peaks: int = 6,
    threshold_rel: float = 0.1,
) -> float:
    """
    Compute grid spacing (scale) from 2D autocorrelogram.

    Grid scale is the characteristic distance between adjacent grid fields,
    computed as the median distance from the autocorrelogram center to the
    6 nearest peaks (the first ring of hexagonal symmetry).

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (n_x, n_y)
        2D spatial autocorrelogram from :func:`spatial_autocorrelation`, zero
        lag at index ``n // 2`` along each axis.
    bin_size : float
        Side of the autocorrelogram's square pixels in the environment's
        units (e.g., cm). Pass ``autocorrelation_pixel_size(env)``, which
        differs from the environment's bin size when its two cell widths
        differ.
    n_peaks : int, default=6
        Number of nearest peaks used: 6 is the first ring of a hexagonal
        lattice. At least 3.
    threshold_rel : float, default=0.1
        Minimum peak correlation as a fraction of the zero-lag correlation
        (which is 1 for :func:`spatial_autocorrelation`). In [0, 1).

    Returns
    -------
    scale : float
        Grid spacing in the units of ``bin_size``. NaN if fewer than 3 peaks
        are found.

    Raises
    ------
    ValueError
        If ``autocorr_2d`` is not 2D, ``bin_size`` is not positive,
        ``n_peaks < 3`` or ``threshold_rel`` is outside [0, 1).
    TypeError
        If ``n_peaks`` is not an integer.

    Notes
    -----
    **Algorithm**:

    1. Find the central peak as :func:`grid_score` does.
    2. Find local maxima outside it with correlation at least
       ``threshold_rel``; maxima closer together than the central peak's
       radius are one field. Both lengths follow the field size, so the
       spacing does not depend on the pixel size.
    3. Take the ``n_peaks`` maxima nearest zero lag and return the median of
       their distances, converted to physical units with ``bin_size``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.grid import (
    ...     autocorrelation_pixel_size,
    ...     grid_scale,
    ...     spatial_autocorrelation,
    ... )
    >>> edges = np.arange(0.0, 102.5, 2.5)  # 100 cm box, 2.5 cm bins
    >>> env = Environment.from_grid_mask(
    ...     np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> wavenumber = 4 * np.pi / (np.sqrt(3) * 40.0)  # 40 cm grid spacing
    >>> rate = sum(
    ...     np.cos(wavenumber * env.bin_centers @ [np.cos(a), np.sin(a)])
    ...     for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
    ... )
    >>> autocorr = spatial_autocorrelation(env, np.exp(rate))
    >>> scale = grid_scale(autocorr, bin_size=autocorrelation_pixel_size(env))
    >>> bool(abs(scale - 40.0) < 2.5)
    True

    See Also
    --------
    grid_properties : Combined function for all grid metrics
    grid_score : Measure hexagonal periodicity

    References
    ----------
    Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
        Microstructure of a spatial map in the entorhinal cortex. Nature,
        436(7052), 801-806.
    """
    _validate_peak_arguments(autocorr_2d, bin_size, n_peaks, threshold_rel)
    peak_coords = _find_autocorr_peaks(
        autocorr_2d, threshold_rel=threshold_rel, max_peaks=n_peaks
    )
    if len(peak_coords) < 3:
        return np.nan
    return float(np.median(np.hypot(peak_coords[:, 0], peak_coords[:, 1]))) * bin_size


def grid_properties(
    autocorr_2d: NDArray[np.float64],
    *,
    bin_size: float,
    n_peaks: int = 6,
    threshold_rel: float = 0.1,
) -> GridProperties:
    """
    Compute all grid cell metrics from 2D autocorrelogram.

    Computes grid score, spacing, orientation and the peaks they are read
    from in one call; the spacing equals :func:`grid_scale` with the same
    arguments and the score equals :func:`grid_score`.

    Parameters
    ----------
    autocorr_2d : NDArray[np.float64], shape (n_x, n_y)
        2D spatial autocorrelogram from :func:`spatial_autocorrelation`.
        Zero lag at index ``n // 2`` along each axis; axis 0 runs along the
        environment's x and axis 1 along y, so orientation is measured in
        the environment's frame.
    bin_size : float
        Side of the autocorrelogram's square pixels in the environment's
        units (e.g., cm). Pass ``autocorrelation_pixel_size(env)``, which
        differs from the environment's bin size when its two cell widths
        differ.
    n_peaks : int, default=6
        Number of nearest peaks used for spacing and orientation: 6 is the
        first ring of a hexagonal lattice. At least 3.
    threshold_rel : float, default=0.1
        Minimum peak correlation as a fraction of the zero-lag correlation.
        In [0, 1).

    Returns
    -------
    GridProperties
        Score, spacing, orientation, orientation spread and the peaks used.

    Raises
    ------
    ValueError
        If ``autocorr_2d`` is not 2D, ``bin_size`` is not positive,
        ``n_peaks < 3`` or ``threshold_rel`` is outside [0, 1).
    TypeError
        If ``n_peaks`` is not an integer.

    Notes
    -----
    Peaks are found as in :func:`grid_scale`. Orientation is the mean angle
    of the peaks modulo 60 degrees, after moving angles near the 0/60
    boundary to the side most peaks are on. Scale and orientation are NaN
    when fewer than 3 peaks are found.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.grid import (
    ...     autocorrelation_pixel_size,
    ...     grid_properties,
    ...     spatial_autocorrelation,
    ... )
    >>> edges = np.arange(0.0, 102.5, 2.5)  # 100 cm box, 2.5 cm bins
    >>> env = Environment.from_grid_mask(
    ...     np.ones((40, 40), dtype=bool), grid_edges=(edges, edges)
    ... )
    >>> wavenumber = 4 * np.pi / (np.sqrt(3) * 40.0)  # 40 cm grid spacing
    >>> rate = sum(
    ...     np.cos(wavenumber * env.bin_centers @ [np.cos(a), np.sin(a)])
    ...     for a in (0.3, 0.3 + np.pi / 3, 0.3 + 2 * np.pi / 3)
    ... )
    >>> props = grid_properties(
    ...     spatial_autocorrelation(env, np.exp(rate)),
    ...     bin_size=autocorrelation_pixel_size(env),
    ... )
    >>> bool(props.score > 0.4), bool(abs(props.scale - 40.0) < 2.5)
    (True, True)

    See Also
    --------
    grid_score : Grid score only
    grid_scale : Grid spacing only
    GridProperties : Return type dataclass

    References
    ----------
    Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    Hafting et al. (2005). Microstructure of a spatial map in the entorhinal
        cortex. Nature, 436(7052), 801-806.
    """
    _validate_peak_arguments(autocorr_2d, bin_size, n_peaks, threshold_rel)
    score = grid_score(autocorr_2d)
    peak_coords = _find_autocorr_peaks(
        autocorr_2d, threshold_rel=threshold_rel, max_peaks=n_peaks
    )
    n_detected = len(peak_coords)
    if n_detected < 3:
        return GridProperties(
            score=score,
            scale=np.nan,
            orientation=np.nan,
            orientation_std=np.nan,
            peak_coords=peak_coords,
            n_peaks=n_detected,
        )

    scale = float(np.median(np.hypot(peak_coords[:, 0], peak_coords[:, 1]))) * bin_size

    # peak_coords is (x, y), so arctan2(y, x) reads column 1 then column 0.
    orientations = np.arctan2(peak_coords[:, 1], peak_coords[:, 0]) % _HEX_PERIOD

    # Axis flip correction
    adjusted = np.where(
        _HEX_PERIOD - orientations < orientations,
        orientations - _HEX_PERIOD,
        orientations,
    )

    n_positive = np.sum(adjusted > _AXIS_FLIP_THRESHOLD)
    n_negative = np.sum(adjusted < -_AXIS_FLIP_THRESHOLD)

    if n_positive > 0 and n_negative > 0:
        if n_positive > n_negative:
            adjusted = np.where(adjusted < 0, adjusted + _HEX_PERIOD, adjusted)
        else:
            adjusted = np.where(adjusted > 0, adjusted - _HEX_PERIOD, adjusted)

    orientation = float(np.nanmean(adjusted)) % _HEX_PERIOD
    orientation_std = float(np.nanstd(adjusted))

    # A mean just below the period is orientation 0 (within 0.1 degree).
    if _HEX_PERIOD - orientation < np.radians(0.1):
        orientation = 0.0

    return GridProperties(
        score=score,
        scale=scale,
        orientation=orientation,
        orientation_std=orientation_std,
        peak_coords=peak_coords,
        n_peaks=n_detected,
    )
