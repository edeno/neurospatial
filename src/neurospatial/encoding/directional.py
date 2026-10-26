"""Directional rate computation for head direction cells.

This module provides result classes and compute functions for directional firing
rate analysis, specifically for head direction (HD) cells. HD cells fire
preferentially when an animal faces a particular direction.

Unlike spatial encoding, directional encoding does not require an Environment.
Directional data is represented by bin centers (angles in radians) rather than
spatial coordinates.

Result Classes
--------------
DirectionalRateResult
    Single-neuron directional tuning curve with convenience methods
DirectionalRatesResult
    Multi-neuron directional tuning curves with batch methods and iteration

Compute Functions
-----------------
compute_directional_rate
    Compute directional firing rate for one neuron
compute_directional_rates
    Compute directional firing rates for multiple neurons

Examples
--------
>>> import numpy as np
>>> from neurospatial.encoding.directional import DirectionalRateResult

>>> # Create result (typically from compute_directional_rate)
>>> n_bins = 60  # 6 degree resolution
>>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
>>> firing_rate = np.random.rand(n_bins) * 10  # Hz
>>> occupancy = np.ones(n_bins) * 0.5  # seconds per bin
>>> result = DirectionalRateResult(
...     firing_rate=firing_rate,
...     occupancy=occupancy,
...     bin_centers=bin_centers,
...     bin_size=np.pi / 30,  # 6 degrees in radians
...     bandwidth=None,
... )

See Also
--------
neurospatial.stats.circular : Circular statistics utilities
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.ops._types import BackendType
    from neurospatial.stats.shuffle import ShuffleTestResult

from neurospatial._intervals import (
    interval_valid_mask,
    resolve_time_windows,
    run_time_bounds,
)
from neurospatial._observation import validate_break_before
from neurospatial._results import _copy_analysis_settings
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.encoding._base import (
    PopulationRateMixin,
    SingleUnitRateMixin,
    _to_numpy,
)
from neurospatial.encoding._significance import (
    check_criterion,
    check_mode_keywords,
    resolve_shuffle_settings,
)
from neurospatial.encoding._spatial_counts import (
    _warn_if_population_silent,
)

__all__ = [
    # Result classes
    "DirectionalRateResult",
    "DirectionalRatesResult",
    # Compute functions
    "compute_directional_rate",
    "compute_directional_rates",
    # Convenience functions
    "head_direction_cell_significance",
    "is_head_direction_cell",
    "plot_head_direction_tuning",
]


def _half_max_halfwidth(
    rates: NDArray[np.float64],
    peak_idx: int,
    half_max: float,
    bin_size: float,
    *,
    step: int,
) -> float:
    """Distance (radians) from peak to the half-max crossing in one direction.

    Walks circularly from ``peak_idx`` in ``step`` (+1 or -1), skipping NaN
    bins, and linearly interpolates the crossing between the last finite bin
    above half-max and the first finite bin below it. ``offset`` counts bins
    from the peak so the geometric distance is preserved even when NaN bins are
    skipped. Returns NaN if no finite below-half-max bin is found.

    Parameters
    ----------
    rates : ndarray of shape (n_bins,)
        Directional firing rate per angular bin (may contain NaN for
        unvisited bins).
    peak_idx : int
        Index of the peak bin.
    half_max : float
        Half of the peak firing rate.
    bin_size : float
        Angular bin width in radians.
    step : int
        Search direction: +1 (increasing index) or -1 (decreasing index).

    Returns
    -------
    float
        Half-width at half-maximum in radians for this direction, or NaN if
        no finite below-half-max bin is found.
    """
    n_bins = len(rates)
    prev_offset = 0
    prev_rate = rates[peak_idx]
    for offset in range(1, n_bins // 2 + 1):
        idx = (peak_idx + step * offset) % n_bins
        r = rates[idx]
        if not np.isfinite(r):
            continue  # skip unvisited bin; keep accumulating offset distance
        if r < half_max:
            denom = r - prev_rate
            frac = 0.0 if denom == 0 else (half_max - prev_rate) / denom
            # Interpolate between the last finite-above bin (prev_offset) and
            # this finite-below bin (offset), in units of bins-from-peak.
            return (prev_offset + frac * (offset - prev_offset)) * bin_size
        prev_offset = offset
        prev_rate = r
    return float(np.nan)


HEAD_DIRECTION_THRESHOLDS = MappingProxyType({"min_mvl": 0.4, "alpha": 0.05})


def _directional_xarray_attrs(
    result: DirectionalRateResult | DirectionalRatesResult,
) -> dict[str, Any]:
    """Coordinate and estimator metadata specific to this rate family."""
    return {"units": "radians", "bin_size": result.bin_size}


def _directional_preferred_direction(rates: NDArray, centers: NDArray) -> float:
    """Shared circular metric with the existing valid-bin/count rules."""
    from neurospatial.stats.circular import circular_mean

    rates = np.asarray(rates, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # Mask out NaN values (from unvisited bins) before computing circular mean
    valid_mask = ~np.isnan(rates)
    if not np.any(valid_mask):
        return float(np.nan)

    return circular_mean(centers[valid_mask], weights=rates[valid_mask])


def _directional_mean_vector_length(rates: NDArray, centers: NDArray) -> float:
    """Shared circular metric with the existing valid-bin/count rules."""
    from neurospatial.stats.circular import mean_resultant_length

    rates = np.asarray(rates, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # Mask out NaN values (from unvisited bins) before computing MVL
    valid_mask = ~np.isnan(rates)
    if not np.any(valid_mask):
        return float(np.nan)

    return mean_resultant_length(centers[valid_mask], weights=rates[valid_mask])


_MIN_TUNING_TEST_SPIKES = 3


def _tuning_pvalues(
    rates: NDArray, occupancy: NDArray, centers: NDArray, spike_counts: NDArray | None
) -> NDArray[np.float64]:
    """Occupancy-corrected test of one preferred direction, one p-value per unit.

    Score test of the Poisson model ``log rate(θ) = a + b cos θ + c sin θ``
    with occupancy (seconds) as exposure, against ``b = c = 0``. With occupied
    bins ``i`` (occupancy ``o_i``, spike count ``c_i``, unit vector ``u_i``),
    ``O = Σ o_i``, ``N = Σ c_i``, ``λ = N / O`` and ``m = Σ o_i u_i``:

    ``S = Σ c_i u_i - λ m``, ``V = λ (Σ o_i u_i u_iᵀ - m mᵀ / O)`` and
    ``T = Sᵀ V⁻¹ S``, which is χ² with 2 degrees of freedom under the null,
    so ``p = exp(-T / 2)``. With uniform occupancy, ``T = 2 N R²`` (``R``
    the count-weighted mean resultant length): the large-sample Rayleigh test.

    Parameters
    ----------
    rates : ndarray of shape (n_units, n_bins)
        Firing rates in Hz; used only to rebuild counts when ``spike_counts``
        is None.
    occupancy : ndarray of shape (n_bins,)
        Unsmoothed time in each heading bin, seconds.
    centers : ndarray of shape (n_bins,)
        Bin centers in radians.
    spike_counts : ndarray of shape (n_units, n_bins) or None
        Unsmoothed spike counts. None rebuilds them as ``rates * occupancy``.

    Returns
    -------
    ndarray of shape (n_units,)
        P-values; NaN when fewer than three spikes fall in occupied bins or
        the occupied headings cannot separate a cosine from a sine term
        (fewer than three distinct directions).
    """
    rates = np.atleast_2d(np.asarray(rates, dtype=np.float64))
    occupancy = np.asarray(occupancy, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    counts = (
        rates * occupancy
        if spike_counts is None
        else np.atleast_2d(np.asarray(spike_counts, dtype=np.float64))
    )
    # Unvisited bins carry no exposure, so spikes assigned to them cannot be
    # tested against their heading.
    used = (occupancy > 0) & np.isfinite(centers) & np.isfinite(counts)
    exposure = np.where(used, occupancy, 0.0)  # (n_units, n_bins)
    counts = np.where(used, counts, 0.0)
    unit = np.stack([np.cos(centers), np.sin(centers)], axis=1)
    unit = np.where(np.isfinite(unit), unit, 0.0)  # (n_bins, 2)

    total_time = exposure.sum(axis=1)  # (n_units,)
    n_spikes = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = n_spikes / total_time
        m = exposure @ unit  # (n_units, 2)
        score = counts @ unit - lam[:, None] * m
        # Occupancy-weighted covariance of the heading unit vectors, times
        # total time: the Fisher information divided by λ.
        second = np.einsum("kb,bi,bj->kij", exposure, unit, unit)
        cov = second - m[:, :, None] * m[:, None, :] / total_time[:, None, None]
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        quad = (
            cov[:, 1, 1] * score[:, 0] ** 2
            - 2.0 * cov[:, 0, 1] * score[:, 0] * score[:, 1]
            + cov[:, 0, 0] * score[:, 1] ** 2
        ) / det
        statistic = quad / lam
    scale = np.trace(cov, axis1=1, axis2=2) ** 2
    valid = (
        (n_spikes >= _MIN_TUNING_TEST_SPIKES) & (total_time > 0) & (det > 1e-12 * scale)
    )
    pvalues = np.full(len(counts), np.nan)
    pvalues[valid] = np.exp(-0.5 * np.maximum(statistic[valid], 0.0))
    return pvalues


def _directional_tuning_width(rates: NDArray, bin_size: float) -> float:
    """Half-width at half-maximum in radians; NaN for a map with no finite rate."""
    rates = np.asarray(rates, dtype=np.float64)
    if not np.any(np.isfinite(rates)):
        return float(np.nan)

    # Find peak
    peak_idx = int(np.nanargmax(rates))
    peak_rate = rates[peak_idx]
    half_max = peak_rate / 2.0

    # Check for flat tuning curve
    if np.nanmin(rates) >= half_max:
        # All rates are above half-max, can't compute HWHM
        return float(np.nan)

    # Search for half-max crossings on both sides using circular indexing.
    # The helper skips NaN (unvisited) bins so an unvisited bin between the
    # peak and the crossing no longer aborts the search.
    right_width = _half_max_halfwidth(rates, peak_idx, half_max, bin_size, step=+1)
    left_width = _half_max_halfwidth(rates, peak_idx, half_max, bin_size, step=-1)

    # Average the finite half-widths. If only one side crosses (the other
    # is masked-out NaN all the way round), report the single finite side
    # rather than NaN.
    halves = np.array([left_width, right_width])
    finite = halves[np.isfinite(halves)]
    if finite.size == 0:
        return float(np.nan)
    return float(finite.mean())


_MAX_UNVISITED_ARC = np.pi / 2


def _largest_unvisited_arc(occupancy: ArrayLike) -> float:
    """Longest circular run of unvisited heading bins, in radians.

    Parameters
    ----------
    occupancy : ndarray of shape (n_bins,)
        Unsmoothed time per heading bin, seconds; the bins tile the circle.

    Returns
    -------
    float
        Run length times the bin width ``2π / n_bins``: 0 when every bin was
        visited, 2π when none was.
    """
    occupancy = np.asarray(occupancy, dtype=np.float64)
    n_bins = occupancy.size
    unvisited = ~(occupancy > 0)
    if unvisited.all():
        return float(2 * np.pi)
    # Start the scan just after a visited bin so a run crossing the wrap is
    # counted once.
    start = int(np.flatnonzero(~unvisited)[0]) + 1
    run = np.roll(unvisited, -start).astype(np.int8)
    edges = np.diff(np.concatenate([[0], run, [0]]))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    longest = int(lengths.max()) if lengths.size else 0
    return float(longest * 2 * np.pi / n_bins)


def _warn_if_headings_incomplete(occupancy: ArrayLike) -> None:
    """Warn that an MVL verdict is unreliable when a wide heading arc is unvisited."""
    arc = _largest_unvisited_arc(occupancy)
    if arc <= _MAX_UNVISITED_ARC:
        return
    warnings.warn(
        f"The animal never faced a {np.degrees(arc):.0f}° arc of headings "
        f"(largest_unvisited_arc() = {arc:.2f} rad). Mean vector length uses "
        "only visited headings, so it is inflated for untuned cells (about "
        "0.3 when 270° are visited, 0.4 when 240° are) and the MVL screen is "
        "unreliable. For a verdict, use is_head_direction_cell(..., "
        'criterion="shuffle") or head_direction_cell_significance(...) with '
        "the raw arrays.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _head_direction_classify(
    rates: NDArray,
    occupancy: NDArray,
    bin_centers: NDArray,
    spike_counts: NDArray | None,
    *,
    min_mvl: float,
    alpha: float,
) -> NDArray[np.bool_]:
    """The shared MVL and occupancy-corrected tuning-test screen."""
    _warn_if_headings_incomplete(occupancy)
    mvls = np.array(
        [_directional_mean_vector_length(row, bin_centers) for row in rates]
    )
    pvalues = _tuning_pvalues(rates, occupancy, bin_centers, spike_counts)
    # NaN MVLs or p-values compare False, so an untestable unit is not flagged.
    return np.asarray((mvls >= min_mvl) & (pvalues < alpha), dtype=np.bool_)


def _directional_summary_frame(
    result: DirectionalRateResult | DirectionalRatesResult, *, index: Sequence[Hashable]
) -> pd.DataFrame:
    """Build identical circular metric columns for one or many units."""
    import pandas as pd

    rates = np.atleast_2d(np.asarray(result._get_rates(), dtype=np.float64))
    occupancy = np.asarray(result.occupancy, dtype=np.float64)
    centers = np.asarray(result.bin_centers, dtype=np.float64)
    counts = (
        None
        if result.spike_counts is None
        else np.atleast_2d(np.asarray(result.spike_counts, dtype=np.float64))
    )
    directions = np.array(
        [_directional_preferred_direction(row, centers) for row in rates]
    )
    mvls = np.array([_directional_mean_vector_length(row, centers) for row in rates])
    widths = np.array(
        [_directional_tuning_width(row, result.bin_size) for row in rates]
    )
    columns = {
        "peak_rate": np.atleast_1d(result.peak_firing_rate()),
        "mean_vector_length": mvls,
        "preferred_direction_deg": np.degrees(directions),
        "tuning_width_deg": np.degrees(widths),
        "preferred_direction": directions,
        "tuning_width": widths,
        "tuning_pvalue": _tuning_pvalues(rates, occupancy, centers, counts),
        "is_head_direction_cell": _head_direction_classify(
            rates, occupancy, centers, counts, **HEAD_DIRECTION_THRESHOLDS
        ),
    }
    df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
    df.attrs["units"] = {
        "peak_rate": "Hz",
        "mean_vector_length": "",
        "preferred_direction": "rad",
        "preferred_direction_deg": "deg",
        "tuning_width": "rad",
        "tuning_width_deg": "deg",
        "tuning_pvalue": "",
        "largest_unvisited_arc": "rad",
    }
    df.attrs["classification_thresholds"] = dict(HEAD_DIRECTION_THRESHOLDS)
    df.attrs["direction_frame"] = result.direction_frame
    df.attrs["largest_unvisited_arc"] = _largest_unvisited_arc(occupancy)
    df.attrs.update(result._analysis_attrs())
    return df


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class DirectionalRateResult(SingleUnitRateMixin):
    """Result of directional rate computation for a single neuron.

    This class wraps a directional tuning curve with its associated metadata
    (occupancy, bin centers, bin size, smoothing parameters). Unlike spatial
    results, directional results do not require an Environment since head
    direction is independent of spatial position.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate by direction in Hz. Shape is (n_bins,) where n_bins is
        the number of angular bins (e.g., 60 bins for 6° resolution).
    occupancy : ArrayLike
        Time spent at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Center of each angular bin in radians [-π, π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    bandwidth : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.
    spike_counts : ArrayLike or None, optional
        Raw (unsmoothed) spike count per angular bin, shape (n_bins,), used by
        :meth:`tuning_pvalue`. None for results constructed without counts
        (e.g. from external tuning curves), in which case counts are rebuilt
        as ``firing_rate * occupancy``, which is exact only for unsmoothed
        rates.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by direction in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians, in [-π, π). Shape is (n_bins,).
    bin_size : float
        Angular bin width in radians.
    bandwidth : float or None
        Smoothing bandwidth in radians, or None.
    spike_counts : ArrayLike or None
        Raw (unsmoothed) spike count per angular bin, shape (n_bins,), or
        None if the result was built without counts.
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation.
    direction_frame : {"allocentric"}
        Head direction is measured in the world frame (0 = East, π/2 =
        North). Recorded in ``summary()`` and ``summary_table().attrs``.

    Other Parameters
    ----------------
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, meet the recorded minimum occupancy and have
    finite rates for every unit. Angular and polar results use their own bin
    coordinates; unsupported bins are not evidence of neural silence.

    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **No Environment dependency**: Unlike SpatialRateResult, directional results
    do not store an Environment. Head direction is a 1D circular variable that
    does not depend on spatial position. The bin_centers field provides the
    angular coordinates needed for analysis and plotting.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import DirectionalRateResult

    >>> # Create a simple tuning curve
    >>> n_bins = 60
    >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    >>> # Von Mises-like tuning with peak at π/2 (90°)
    >>> firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1))
    >>> occupancy = np.ones(n_bins) * 0.5

    >>> result = DirectionalRateResult(
    ...     firing_rate=firing_rate,
    ...     occupancy=occupancy,
    ...     bin_centers=bin_centers,
    ...     bin_size=np.pi / 30,
    ...     bandwidth=None,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (60,)
    >>> result.bin_size  # 6 degrees in radians
    0.10471975511965977

    See Also
    --------
    DirectionalRatesResult : Batch version for multiple neurons
    compute_directional_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    bandwidth: float | None
    spike_counts: ArrayLike | None = None
    unit_id: int | str | None = None
    direction_frame: ClassVar[Literal["allocentric"]] = "allocentric"

    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _xarray_attrs(self) -> dict[str, Any]:
        return _directional_xarray_attrs(self)

    @property
    def method(self) -> str:
        """Estimator: ``"circular_gaussian"`` when smoothed, else ``"binned"``."""
        return "binned" if self.bandwidth is None else "circular_gaussian"

    def summary_table(self) -> pd.DataFrame:
        """Per-unit metrics with the same columns as the population table.

        Returns
        -------
        pandas.DataFrame
            One row indexed by this unit label, or <NA> when no label was supplied.

        Notes
        -----
        Labels are fixed-threshold heuristics; the family defaults are in
        ``df.attrs["classification_thresholds"]``. For shuffle significance,
        call head_direction_cell_significance with the raw arrays.
        See the family predicate's Notes for information/statistic bias.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> table = result[0].summary_table()
        >>> len(table)
        1
        """
        return _directional_summary_frame(self, index=self._row_unit_ids().tolist())

    def _headline_metrics(self) -> dict[str, float]:
        """Cheap, NaN-safe metrics for the singular summary."""
        if not np.any(np.isfinite(_to_numpy(self.firing_rate))):
            return {
                "preferred_direction": float("nan"),
                "mean_vector_length": float("nan"),
            }
        return {
            "preferred_direction": float(self.preferred_direction()),
            "mean_vector_length": float(self.mean_vector_length()),
        }

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: directional results store bin centers
        # directly on the dataclass (no Environment).
        return np.asarray(self.bin_centers, dtype=np.float64)

    def _bin_center_columns(self) -> dict[str, NDArray[np.float64]]:
        # Directional results have no Environment; the bin center is the
        # angular center (radians). Emit it under the shared vocabulary name.
        return {"bin_center_angle": np.asarray(self.bin_centers, dtype=np.float64)}

    def plot(
        self,
        ax: Axes | PolarAxes | None = None,
        polar: bool = True,
        **kwargs: Any,
    ) -> Axes | PolarAxes:
        """Plot the directional tuning curve.

        Creates a visualization of the firing rate as a function of direction.
        By default, creates a polar plot which provides an intuitive circular
        representation of head direction tuning.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes, optional
            Axes to plot on. If None, creates a new figure with appropriate
            projection (polar or Cartesian based on the ``polar`` parameter).
        polar : bool, default=True
            If True, create a polar plot (circular representation).
            If False, create a Cartesian plot (angle on x-axis, rate on y-axis).
        **kwargs : dict
            Additional keyword arguments passed to matplotlib's plot function.
            Common options include:
            - color: Line color (default: matplotlib default)
            - linewidth: Line width
            - linestyle: Line style ('-', '--', ':', etc.)
            - label: Legend label

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes
            The axes object containing the plot. Returns PolarAxes if
            ``polar=True``, otherwise regular Axes.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> ax = result.plot()  # Creates polar plot  # doctest: +SKIP

        See Also
        --------
        matplotlib.pyplot.polar : Underlying polar plot function
        """
        import matplotlib.pyplot as plt

        # Convert to numpy for plotting
        rates = np.asarray(self.firing_rate, dtype=np.float64)
        centers = np.asarray(self.bin_centers, dtype=np.float64)

        # Close the curve by appending first point to end
        rates_closed = np.concatenate([rates, [rates[0]]])
        centers_closed = np.concatenate([centers, [centers[0] + 2 * np.pi]])

        if ax is None:
            if polar:
                _, ax = plt.subplots(subplot_kw={"projection": "polar"})
            else:
                _, ax = plt.subplots()

        ax.plot(centers_closed, rates_closed, **kwargs)

        if not polar:
            ax.set_xlabel("Direction (rad)")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_xlim(-np.pi, np.pi)

        return ax

    def preferred_direction(self) -> float:
        """Compute the preferred direction (circular mean weighted by firing rate).

        The preferred direction is the circular mean of the bin centers weighted
        by the firing rate, representing the direction at which the neuron fires
        most strongly on average.

        Returns
        -------
        float
            Preferred direction in radians, range [-π, π].

        Notes
        -----
        Uses the circular mean formula:

            θ_pref = arctan2(Σ(r_i * sin(θ_i)), Σ(r_i * cos(θ_i)))

        where r_i is the firing rate and θ_i is the bin center.

        This is a more robust measure than simply taking the bin with maximum
        firing rate, as it accounts for the overall shape of the tuning curve.

        For neurons with uniform firing (no directional preference), the result
        may not be meaningful, but a value will still be returned.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> # Create tuning curve with peak at π/2 (90 degrees)
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> pref = result.preferred_direction()
        >>> bool(np.abs(pref - np.pi / 2) < 0.1)  # Close to 90 degrees
        True

        See Also
        --------
        preferred_direction_deg : Same result in degrees
        mean_vector_length : Strength of directional tuning
        neurospatial.stats.circular.circular_mean : Underlying circular mean function
        """
        return _directional_preferred_direction(
            np.asarray(self.firing_rate, dtype=np.float64),
            np.asarray(self.bin_centers, dtype=np.float64),
        )

    def preferred_direction_deg(self) -> float:
        """Compute the preferred direction in degrees.

        This is a convenience method that returns the preferred direction
        converted to degrees.

        Returns
        -------
        float
            Preferred direction in degrees, range [-180, 180].

        See Also
        --------
        preferred_direction : Same result in radians

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> pref_deg = result.preferred_direction_deg()
        >>> bool(np.abs(pref_deg - 90) < 6)  # Close to 90 degrees
        True
        """
        return float(np.degrees(self.preferred_direction()))

    def peak_firing_rate(self) -> float:
        """Get the maximum firing rate.

        Returns the peak (maximum) firing rate across all directional bins.
        NaN values are ignored when computing the maximum.

        Returns
        -------
        float
            Maximum firing rate in Hz.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> firing_rate = 10.0 * np.exp(2.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> peak = result.peak_firing_rate()
        >>> peak > 0
        True

        See Also
        --------
        preferred_direction : Direction of peak firing
        """
        rates = np.asarray(self.firing_rate, dtype=np.float64)
        return float(np.nanmax(rates))

    def mean_vector_length(self) -> float:
        """Compute the mean vector length (MVL) of the tuning curve.

        The mean vector length quantifies the concentration of directional tuning.
        It ranges from 0 (uniform firing) to 1 (all firing at one direction).
        This is the standard measure of head direction cell tuning strength.

        Returns
        -------
        float
            Mean vector length in [0, 1]. Higher values indicate sharper tuning.

        Notes
        -----
        The mean vector length is computed as the resultant of unit vectors
        at each direction, weighted by the firing rate:

            R = |Σ(r_i * exp(i * θ_i))| / Σ(r_i)

        This is equivalent to:

            R = sqrt(Σ(r_i * cos(θ_i))² + Σ(r_i * sin(θ_i))²) / Σ(r_i)

        where r_i is the firing rate and θ_i is the bin center.

        Only visited bins enter the sum, so an untuned cell recorded while
        the animal faced a limited range of headings has an inflated MVL
        (about 0.3 when 270° are visited, 0.4 when 240° are). Check
        :meth:`largest_unvisited_arc`. The cutoff of 0.4 used by
        :meth:`is_head_direction_cell` is a common screening convention;
        check it against the MVL distribution of your own recordings.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> # Sharply tuned neuron
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> mvl = result.mean_vector_length()
        >>> mvl > 0.5  # Sharply tuned neuron has high MVL
        True

        See Also
        --------
        preferred_direction : Direction of peak firing
        tuning_pvalue : Occupancy-corrected test of directional tuning
        neurospatial.stats.circular.mean_resultant_length : Underlying function
        """
        return _directional_mean_vector_length(
            np.asarray(self.firing_rate, dtype=np.float64),
            np.asarray(self.bin_centers, dtype=np.float64),
        )

    def tuning_width(self) -> float:
        """Compute the tuning width (half-width at half-maximum) in radians.

        The tuning width is the angular distance from the peak at which the
        firing rate drops to half of its maximum value. This is computed as
        the half-width at half-maximum (HWHM) of the tuning curve.

        Returns
        -------
        float
            Tuning width in radians, range (0, π]. Returns NaN if the tuning
            curve is flat (uniform firing) or if HWHM cannot be determined.

        Notes
        -----
        The algorithm finds the peak location, then searches outward in both
        directions to find where the firing rate drops below half-maximum.
        The tuning width is the average of the left and right half-widths.

        For a von Mises tuning curve with concentration κ, the theoretical
        HWHM is approximately:

            HWHM ≈ arccos(1 - (1/κ) * ln(2))

        **Interpretation guidelines**:

        - Width < 30° (π/6): Very sharp tuning
        - Width 30-60° (π/6 to π/3): Sharp tuning (typical HD cell)
        - Width 60-90° (π/3 to π/2): Moderate tuning
        - Width > 90° (> π/2): Broad tuning

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> # Sharply tuned neuron (kappa=5)
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> width = result.tuning_width()
        >>> width < np.pi / 3  # Sharp tuning < 60 degrees
        True

        See Also
        --------
        tuning_width_deg : Same result in degrees
        mean_vector_length : Alternative tuning strength measure
        """
        return _directional_tuning_width(
            np.asarray(self.firing_rate, dtype=np.float64), self.bin_size
        )

    def tuning_width_deg(self) -> float:
        """Compute the tuning width (half-width at half-maximum) in degrees.

        This is a convenience method that returns the tuning width
        converted to degrees.

        Returns
        -------
        float
            Tuning width in degrees, range (0, 180]. Returns NaN if
            tuning width cannot be determined.

        See Also
        --------
        tuning_width : Same result in radians

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> width_deg = result.tuning_width_deg()
        >>> width_deg < 60  # Sharp tuning < 60 degrees
        True
        """
        return float(np.degrees(self.tuning_width()))

    def largest_unvisited_arc(self) -> float:
        """Widest arc of headings the animal never faced, in radians.

        Mean vector length and preferred direction use only visited heading
        bins. When a wide arc is never visited, an untuned cell's tuning curve
        covers only part of the circle and its MVL is inflated (about 0.3
        when 270° are visited, 0.4 when 240° are). The MVL screen warns when
        this arc exceeds 90°.

        Returns
        -------
        float
            Longest run of zero-occupancy bins times the bin width, in
            [0, 2π]; 0 when every heading bin was visited.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> occupancy = np.ones(n_bins)
        >>> occupancy[:15] = 0.0  # a 90° arc never faced
        >>> result = DirectionalRateResult(
        ...     firing_rate=np.where(occupancy > 0, 2.0, np.nan),
        ...     occupancy=occupancy,
        ...     bin_centers=np.linspace(-np.pi, np.pi, n_bins, endpoint=False),
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> round(float(np.degrees(result.largest_unvisited_arc())))
        90
        """
        return _largest_unvisited_arc(self.occupancy)

    def tuning_pvalue(self) -> float:
        """Test for one preferred direction, given where the animal faced.

        Tests whether the firing rate varies with heading as
        ``log rate(θ) = a + b cos θ + c sin θ`` against a constant rate,
        using the time spent at each heading (``occupancy``) as exposure.
        Because occupancy enters the test, a cell that fires at a constant
        rate is not called tuned when the animal simply faced some directions
        more often than others.

        Returns
        -------
        float
            P-value in [0, 1]; small values indicate directional tuning.
            NaN when fewer than 3 spikes fall in visited bins, or when the
            visited headings span fewer than three distinct directions.

        Notes
        -----
        This is the score test of a Poisson regression of the spike counts
        on ``cos θ`` and ``sin θ`` with log occupancy as offset. With visited
        bins ``i`` (occupancy ``o_i`` in seconds, unsmoothed spike count
        ``c_i``, unit vector ``u_i = (cos θ_i, sin θ_i)``), ``O = Σ o_i``,
        ``N = Σ c_i``, ``λ = N / O`` and ``m = Σ o_i u_i``:

            S = Σ c_i u_i - λ m
            V = λ (Σ o_i u_i u_iᵀ - m mᵀ / O)
            T = Sᵀ V⁻¹ S,   p = P(χ²₂ > T) = exp(-T / 2)

        With uniform occupancy ``T = 2 N R²``, where ``R`` is the mean
        resultant length of the spike headings, so the test reduces to the
        large-sample Rayleigh test.

        **Assumptions.** Spikes are independent given heading. Bursting and
        slow drifts in rate or heading (common in real recordings) make the
        p-value too small, so treat it as a screen. For a verdict, use
        ``is_head_direction_cell(..., criterion="shuffle")`` or
        :func:`head_direction_cell_significance` on the raw arrays. The test
        targets one preferred direction and has little power for bimodal
        tuning. With many spikes even weak tuning is significant; check
        :meth:`mean_vector_length` as well.

        Counts come from ``spike_counts``; when it is None they are rebuilt
        as ``firing_rate * occupancy``, which is exact only for unsmoothed
        rates.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> occupancy = np.full(n_bins, 0.5)
        >>> # Sharply tuned neuron
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=occupancy,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ...     spike_counts=np.round(firing_rate * occupancy),
        ... )
        >>> bool(result.tuning_pvalue() < 0.05)
        True

        See Also
        --------
        mean_vector_length : Tuning strength measure
        is_head_direction_cell : Screen combining MVL and this p-value
        head_direction_cell_significance : Circular-shift test on raw arrays
        """
        return float(
            _tuning_pvalues(
                np.asarray(self.firing_rate, dtype=np.float64),
                np.asarray(self.occupancy, dtype=np.float64),
                np.asarray(self.bin_centers, dtype=np.float64),
                None
                if self.spike_counts is None
                else np.asarray(self.spike_counts, dtype=np.float64),
            )[0]
        )

    def is_head_direction_cell(
        self, *, min_mvl: float | None = None, alpha: float | None = None
    ) -> bool:
        """Screen for a head direction cell.

        A neuron passes the screen if it meets both criteria:

        1. Mean vector length (MVL) >= min_mvl (default 0.4)
        2. :meth:`tuning_pvalue` < alpha (default 0.05)

        Parameters
        ----------
        min_mvl : float or None, default=None
            Minimum mean vector length. None resolves to
            ``HEAD_DIRECTION_THRESHOLDS["min_mvl"]`` (0.4), a common screening
            convention rather than a calibrated threshold; check it against
            the MVL distribution of your own recordings and brain region.
        alpha : float or None, default=None
            Level for :meth:`tuning_pvalue`. None resolves to 0.05.

        Returns
        -------
        bool
            True if the neuron passes both criteria.

        Warns
        -----
        UserWarning
            When the animal never faced a heading arc wider than 90°
            (:meth:`largest_unvisited_arc`), because MVL is then inflated.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> # Sharply tuned neuron
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> result.is_head_direction_cell()  # Sharply tuned neuron should be classified as HD cell
        True

        See Also
        --------
        mean_vector_length : Tuning strength measure
        tuning_pvalue : Occupancy-corrected test of directional tuning
        interpretation : Human-readable summary of classification

        Notes
        -----
        This is a threshold screen, not a verdict. :meth:`tuning_pvalue`
        tests one preferred direction given heading occupancy and assumes
        spikes are independent given heading; bursting and slow drifts in rate
        make its p-value too small. For a verdict, call
        head_direction_cell_significance(...) or
        is_head_direction_cell(..., criterion="shuffle") with the raw arrays;
        a result does not keep the arrays it was computed from.
        """
        min_mvl = HEAD_DIRECTION_THRESHOLDS["min_mvl"] if min_mvl is None else min_mvl
        alpha = HEAD_DIRECTION_THRESHOLDS["alpha"] if alpha is None else alpha
        return bool(
            _head_direction_classify(
                np.atleast_2d(np.asarray(self.firing_rate, dtype=np.float64)),
                np.asarray(self.occupancy, dtype=np.float64),
                np.asarray(self.bin_centers, dtype=np.float64),
                None
                if self.spike_counts is None
                else np.atleast_2d(np.asarray(self.spike_counts, dtype=np.float64)),
                min_mvl=min_mvl,
                alpha=alpha,
            )[0]
        )

    def interpretation(self, min_mvl: float | None = None) -> str:
        """Human-readable interpretation of head direction metrics.

        Summarizes the neuron's directional tuning and the screen result.
        For neurons that pass the screen, includes preferred direction, tuning
        strength, peak rate and the tuning-test p-value. For the others,
        explains which criteria were not met.

        Parameters
        ----------
        min_mvl : float or None, default=None
            Minimum mean vector length threshold for HD cell classification.
            Same parameter as in :meth:`is_head_direction_cell`; None resolves
            to HEAD_DIRECTION_THRESHOLDS["min_mvl"] (0.4).

        Returns
        -------
        str
            Multi-line string containing:

            **For HD cells:**
            - Header indicating HD cell classification
            - Preferred direction (degrees)
            - Mean vector length (with threshold)
            - Peak firing rate (Hz)
            - Tuning width (degrees)
            - Tuning-test p-value

            **For non-HD cells:**
            - Header indicating non-HD cell
            - Explanation of which criteria failed

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> # Sharply tuned neuron
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> print(result.interpretation())  # doctest: +SKIP
        *** HEAD DIRECTION CELL ***
        Preferred direction: 90.0 deg
        Mean vector length: 0.xxx (threshold = 0.4)
        Peak firing rate: 10.0 Hz
        Tuning width (HWHM): xx.x deg
        Tuning test: p = 0.xxxx

        See Also
        --------
        is_head_direction_cell : Boolean classification method
        mean_vector_length : Tuning strength
        tuning_pvalue : Occupancy-corrected test of directional tuning
        """
        lines = []
        min_mvl = HEAD_DIRECTION_THRESHOLDS["min_mvl"] if min_mvl is None else min_mvl
        alpha = HEAD_DIRECTION_THRESHOLDS["alpha"]

        mvl = self.mean_vector_length()
        pval = self.tuning_pvalue()
        is_hd = self.is_head_direction_cell(min_mvl=min_mvl, alpha=alpha)

        if is_hd:
            lines.append("*** HEAD DIRECTION CELL ***")
            lines.append(
                f"Preferred direction: {self.preferred_direction_deg():.1f} deg"
            )
            lines.append(f"Mean vector length: {mvl:.3f} (threshold = {min_mvl})")
            lines.append(f"Peak firing rate: {self.peak_firing_rate():.1f} Hz")
            lines.append(f"Tuning width (HWHM): {self.tuning_width_deg():.1f} deg")
            lines.append(f"Tuning test: p = {pval:.4f}")
        else:
            lines.append("Not classified as HD cell")
            if np.isnan(mvl):
                lines.append(
                    "  - Mean vector length undefined: no firing in visited "
                    "heading bins"
                )
            elif mvl < min_mvl:
                lines.append(f"  - Mean vector length too low: {mvl:.3f} < {min_mvl}")
                lines.append(
                    "    The cutoff is a screening convention; check it against "
                    "the MVL distribution of your recordings."
                )
            if np.isnan(pval):
                lines.append(
                    "  - Tuning test undefined: fewer than 3 spikes or fewer "
                    "than 3 visited heading directions"
                )
            elif pval >= alpha:
                lines.append(
                    f"  - Tuning test not significant: p = {pval:.3f} >= {alpha}"
                )

        return "\n".join(lines)


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class DirectionalRatesResult(PopulationRateMixin[DirectionalRateResult]):
    """Result of directional rate computation for multiple neurons.

    This class wraps directional tuning curves for a population of neurons
    with shared occupancy and bin structure. It provides iteration over
    individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rates by direction in Hz. Shape is (n_neurons, n_bins) where
        n_bins is the number of angular bins. Each row is a single neuron's
        tuning curve.
    occupancy : ArrayLike
        Time spent at each direction in seconds. Shape is (n_bins,). Shared
        across all neurons.
    bin_centers : ArrayLike
        Center of each angular bin in radians [-π, π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    bandwidth : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.
    spike_counts : ArrayLike or None, optional
        Raw (unsmoothed) spike counts per angular bin, shape
        (n_neurons, n_bins), used by :meth:`tuning_pvalues` and forwarded to
        each per-neuron result. None if the batch was built without counts;
        counts are then rebuilt as ``firing_rates * occupancy``.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates by direction in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians, in [-π, π). Shape is (n_bins,).
    bin_size : float
        Angular bin width in radians.
    bandwidth : float or None
        Smoothing bandwidth in radians, or None.
    spike_counts : ArrayLike or None
        Raw (unsmoothed) spike counts per angular bin, shape
        (n_neurons, n_bins), or None.
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    direction_frame : {"allocentric"}
        Head direction is measured in the world frame (0 = East, π/2 =
        North). Recorded in ``summary()`` and ``summary_table().attrs``.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. A unit_id column or named/non-default index aligns rows
        by identity; an unnamed default RangeIndex uses unit order.
        Mismatched or duplicate labels raise ValueError.

    Other Parameters
    ----------------
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, meet the recorded minimum occupancy and have
    finite rates for every unit. Angular and polar results use their own bin
    coordinates; unsupported bins are not evidence of neural silence.

    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `DirectionalRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import DirectionalRatesResult

    >>> # Create batch result (5 neurons)
    >>> rng = np.random.default_rng(0)
    >>> n_neurons = 5
    >>> n_bins = 60
    >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    >>> firing_rates = rng.random((n_neurons, n_bins)) * 10
    >>> occupancy = np.ones(n_bins) * 0.5

    >>> result = DirectionalRatesResult(
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     bin_centers=bin_centers,
    ...     bin_size=np.pi / 30,
    ...     bandwidth=None,
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape
    (5, 60)
    >>> len(result)
    5

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'DirectionalRateResult'

    >>> # Iterate over neurons
    >>> peaks = [float(np.max(r.firing_rate)) for r in result]
    >>> len(peaks)
    5

    See Also
    --------
    DirectionalRateResult : Single-neuron version
    compute_directional_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    bandwidth: float | None
    spike_counts: ArrayLike | None = None  # shape (n_neurons, n_bins)
    direction_frame: ClassVar[Literal["allocentric"]] = "allocentric"
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)

    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _xarray_attrs(self) -> dict[str, Any]:
        return _directional_xarray_attrs(self)

    @property
    def method(self) -> str:
        """Estimator: ``"circular_gaussian"`` when smoothed, else ``"binned"``."""
        return "binned" if self.bandwidth is None else "circular_gaussian"

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: directional results store bin centers
        # directly on the dataclass (no Environment).
        return np.asarray(self.bin_centers, dtype=np.float64)

    def _bin_center_columns(self) -> dict[str, NDArray[np.float64]]:
        # Directional results have no Environment; the bin center is the
        # angular center (radians). Emit it under the shared vocabulary name.
        return {"bin_center_angle": np.asarray(self.bin_centers, dtype=np.float64)}

    def __getitem__(self, idx: int) -> DirectionalRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Unit position (0-based), the row of :attr:`firing_rates`. Select
            by label with ``sel(unit_id=...)``.

        Returns
        -------
        DirectionalRateResult
            Result for the specified neuron with shared occupancy,
            bin_centers, bin_size, and bandwidth.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> single = result[0]
        >>> single.firing_rate.shape
        (60,)
        """
        idx = self._unit_position(idx)
        counts = self.spike_counts
        return self._unit_result(
            DirectionalRateResult,
            idx,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            bandwidth=self.bandwidth,
            spike_counts=(None if counts is None else np.asarray(counts)[idx]),
        )

    def plot(
        self,
        idx: int,
        ax: Axes | PolarAxes | None = None,
        polar: bool = True,
        **kwargs: Any,
    ) -> Axes | PolarAxes:
        """Plot the directional tuning curve for a specific neuron.

        Creates a visualization of the firing rate as a function of direction
        for the specified neuron. By default, creates a polar plot.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed).
        ax : matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes, optional
            Axes to plot on. If None, creates a new figure with appropriate
            projection (polar or Cartesian based on the ``polar`` parameter).
        polar : bool, default=True
            If True, create a polar plot (circular representation).
            If False, create a Cartesian plot (angle on x-axis, rate on y-axis).
        **kwargs : dict
            Additional keyword arguments passed to matplotlib's plot function.

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes
            The axes object containing the plot.

        Examples
        --------
        >>> ax = result.plot(0)  # Plot first neuron  # doctest: +SKIP
        >>> ax = result.plot(0, polar=False, color="red")  # doctest: +SKIP

        See Also
        --------
        DirectionalRateResult.plot : Plot method for single neuron
        """
        return self[idx].plot(ax=ax, polar=polar, **kwargs)

    def preferred_directions(self) -> NDArray[np.float64]:
        """Compute preferred directions for all neurons.

        The preferred direction is the circular mean of the bin centers
        weighted by the firing rate.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_neurons,) with preferred directions in radians,
            range [-π, π].

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> pref_dirs = result.preferred_directions()
        >>> pref_dirs.shape
        (3,)

        See Also
        --------
        DirectionalRateResult.preferred_direction : Single-neuron method
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates, dtype=np.float64)
        centers: NDArray[np.float64] = np.asarray(self.bin_centers, dtype=np.float64)

        # Match the single-neuron path: NaN bins are dropped before the
        # circular-mean reduction, not propagated through. Without this
        # mask any NaN bin makes the whole row's direction NaN, which
        # disagrees with iterating singles.
        finite = np.isfinite(rates)
        finite_rates = np.where(finite, rates, 0.0)

        weight_sums = finite_rates.sum(axis=1)
        valid = weight_sums > 0
        safe_sums = np.where(valid, weight_sums, 1.0)
        weights_norm = finite_rates / safe_sums[:, np.newaxis]

        mean_cos = weights_norm @ np.cos(centers)
        mean_sin = weights_norm @ np.sin(centers)

        pref_dirs: NDArray[np.float64] = np.arctan2(mean_sin, mean_cos)
        pref_dirs[~valid] = np.nan
        return pref_dirs

    def mean_vector_lengths(self) -> NDArray[np.float64]:
        """Compute mean vector lengths for all neurons.

        The mean vector length quantifies the concentration of directional
        tuning. It ranges from 0 (uniform firing) to 1 (all firing at one
        direction).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_neurons,) with mean vector lengths in [0, 1].
            Higher values indicate sharper tuning.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> mvls = result.mean_vector_lengths()
        >>> mvls.shape
        (3,)

        See Also
        --------
        DirectionalRateResult.mean_vector_length : Single-neuron method
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates, dtype=np.float64)
        centers: NDArray[np.float64] = np.asarray(self.bin_centers, dtype=np.float64)

        # NaN bins are dropped before the reduction (see preferred_directions).
        finite = np.isfinite(rates)
        finite_rates = np.where(finite, rates, 0.0)

        weight_sums = finite_rates.sum(axis=1)
        valid = weight_sums > 0
        safe_sums = np.where(valid, weight_sums, 1.0)
        weights_norm = finite_rates / safe_sums[:, np.newaxis]

        mean_cos = weights_norm @ np.cos(centers)
        mean_sin = weights_norm @ np.sin(centers)

        mvls: NDArray[np.float64] = np.sqrt(mean_cos**2 + mean_sin**2)
        mvls[~valid] = np.nan
        return mvls

    def tuning_widths(self) -> NDArray[np.float64]:
        """Compute tuning widths (HWHM) for all neurons.

        The tuning width is the angular distance from the peak at which the
        firing rate drops to half of its maximum value.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_neurons,) with tuning widths in radians,
            range (0, π]. Contains NaN for neurons with flat tuning curves.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> widths = result.tuning_widths()
        >>> widths.shape
        (3,)

        See Also
        --------
        DirectionalRateResult.tuning_width : Single-neuron method
        """
        n_neurons = len(self)
        widths = np.empty(n_neurons, dtype=np.float64)

        for i in range(n_neurons):
            widths[i] = self[i].tuning_width()

        return widths

    def largest_unvisited_arc(self) -> float:
        """Widest arc of headings the animal never faced, in radians.

        Shared by all units, because they share occupancy. See
        :meth:`DirectionalRateResult.largest_unvisited_arc`.

        Returns
        -------
        float
            Longest run of zero-occupancy bins times the bin width, in [0, 2π].
        """
        return _largest_unvisited_arc(self.occupancy)

    def tuning_pvalues(self) -> NDArray[np.float64]:
        """Occupancy-corrected tuning-test p-value for every neuron.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_neurons,) with p-values in [0, 1], NaN where the
            test is undefined. See :meth:`DirectionalRateResult.tuning_pvalue`
            for the test and its assumptions.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> result.tuning_pvalues().shape
        (3,)

        See Also
        --------
        DirectionalRateResult.tuning_pvalue : Single-neuron method
        """
        return _tuning_pvalues(
            np.asarray(self.firing_rates, dtype=np.float64),
            np.asarray(self.occupancy, dtype=np.float64),
            np.asarray(self.bin_centers, dtype=np.float64),
            None
            if self.spike_counts is None
            else np.asarray(self.spike_counts, dtype=np.float64),
        )

    def classify(
        self, *, min_mvl: float | None = None, alpha: float | None = None
    ) -> NDArray[np.bool_]:
        """Screen neurons for head direction cells.

        A neuron passes the screen if it meets both criteria:

        1. Mean vector length (MVL) >= min_mvl (default 0.4)
        2. :meth:`tuning_pvalues` < alpha (default 0.05)

        This is the single-type boolean predicate ("is this an HD cell") for
        the batch result.

        Parameters
        ----------
        min_mvl : float or None, default=None
            Minimum mean vector length. None resolves to 0.4, a common
            screening convention; check it against your own recordings.
        alpha : float or None, default=None
            Level for the tuning test. None resolves to 0.05.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (n_neurons,). True indicates the neuron
            passes the screen.

        Warns
        -----
        UserWarning
            When the animal never faced a heading arc wider than 90°
            (:meth:`largest_unvisited_arc`), because MVL is then inflated.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> is_hd = result.classify()
        >>> is_hd.shape
        (3,)
        >>> n_hd_cells = int(np.sum(is_hd))

        See Also
        --------
        DirectionalRateResult.is_head_direction_cell : Single-neuron method

        Notes
        -----
        This is a threshold screen, not a verdict. The tuning test checks one
        preferred direction given heading occupancy and assumes spikes are
        independent given heading; bursting and slow drifts in rate make its
        p-value too small. For a verdict, call
        head_direction_cell_significance(...) with the raw arrays; a result
        does not keep the arrays it was computed from.
        None thresholds resolve through HEAD_DIRECTION_THRESHOLDS.
        """
        min_mvl = HEAD_DIRECTION_THRESHOLDS["min_mvl"] if min_mvl is None else min_mvl
        alpha = HEAD_DIRECTION_THRESHOLDS["alpha"] if alpha is None else alpha
        return _head_direction_classify(
            np.asarray(self.firing_rates, dtype=np.float64),
            np.asarray(self.occupancy, dtype=np.float64),
            np.asarray(self.bin_centers, dtype=np.float64),
            None
            if self.spike_counts is None
            else np.asarray(self.spike_counts, dtype=np.float64),
            min_mvl=min_mvl,
            alpha=alpha,
        )

    def summary_table(
        self,
        unit_ids: Sequence[str | int] | None = None,
    ) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        Computes all directional metrics and returns one row per unit, indexed
        by ``unit_id``, with scalar metric columns. This is the per-unit
        summary for filtering, sorting, and population tables. For the dense
        per-bin frame (one row per ``(unit, bin)``) use :meth:`to_dataframe`.

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Identity labels for the index, one per unit. If ``None``, the
            result's own :attr:`unit_ids` are used.

        Returns
        -------
        pd.DataFrame
            One row per unit, indexed by ``unit_id``, with columns in order:

            - peak_rate: maximum firing rate (Hz)
            - mean_vector_length: mean vector length [0, 1]
            - preferred_direction_deg: preferred direction in degrees [-180, 180]
            - tuning_width_deg: tuning width (HWHM) in degrees (0, 180]
            - preferred_direction: preferred direction in radians [-π, π]
            - tuning_width: tuning width (HWHM) in radians (0, π]
            - tuning_pvalue: occupancy-corrected tuning-test p-value
            - is_head_direction_cell: threshold screen with default thresholds

        Raises
        ------
        ValueError
            If unit_ids has a different length than the number of units, or
            repeats a label.

        Notes
        -----
        Labels are fixed-threshold heuristics; the family defaults are in
        ``df.attrs["classification_thresholds"]``. For shuffle significance,
        call head_direction_cell_significance with the raw arrays.
        See the family predicate's Notes for information/statistic bias.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``preferred_directions()``, ``mean_vector_lengths()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_head_direction_cell"] == True]``
        - Sort: ``df.sort_values("mean_vector_length", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> result = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> df = result.summary_table()
        >>> len(df)
        3
        >>> "is_head_direction_cell" in df.columns
        True
        >>> df.index.name
        'unit_id'

        >>> # Filter for HD cells
        >>> hd_cells = df[df["is_head_direction_cell"]]

        >>> # Sort by mean vector length
        >>> top_cells = df.sort_values("mean_vector_length", ascending=False)

        >>> # Custom unit identifiers
        >>> df = result.summary_table(unit_ids=["unit_0", "unit_1", "unit_2"])
        >>> list(df.index)
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        to_dataframe : Dense per-bin frame (one row per (unit, bin)).
        classify : HD cell classification
        preferred_directions : Batch preferred direction computation
        mean_vector_lengths : Batch mean vector length computation
        """
        n_neurons = len(self)

        if unit_ids is None:
            index_ids: list[str | int] = list(np.asarray(self.unit_ids))
        else:
            from neurospatial._results import resolve_unit_ids

            # Validate as an object array so mixed int/str labels are
            # neither coerced to strings nor merged; keep them as given.
            index_ids = list(unit_ids)
            resolve_unit_ids(
                np.asarray(index_ids, dtype=object),
                n_neurons,
                context="DirectionalRatesResult.summary_table",
            )

        return _directional_summary_frame(self, index=index_ids)


class _DirectionalCounts(NamedTuple):
    """Unsmoothed directional counts with the binning and provenance used."""

    spike_counts: NDArray[np.float64]  # (n_units, n_bins), unsmoothed
    occupancy: NDArray[np.float64]  # (n_bins,) seconds
    bin_centers: NDArray[np.float64]  # (n_bins,) radians
    bin_size: float  # radians, 2*pi / n_bins actually used
    bandwidth: float | None  # radians
    analysis_settings: dict[str, Any]


def _count_directional(
    spike_trains: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float | None,
    bandwidth: float | None,
    angle_unit: Literal["rad", "deg"],
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    backend: str,
    n_jobs: int = 1,
    break_before: NDArray[np.bool_] | None = None,
) -> _DirectionalCounts:
    """Bin headings, count spikes and build provenance for directional rates.

    Parameters
    ----------
    spike_trains : sequence of ndarray of shape (n_spikes_i,)
        Validated spike times, one array per unit.
    times : ndarray of shape (n_time,)
        Trajectory timestamps in seconds.
    headings : ndarray of shape (n_time,)
        Heading at each timestamp, in ``angle_unit``.
    bin_size : float or None
        Requested bin size, in ``angle_unit``; None means 6°.
    bandwidth : float or None
        Requested smoothing bandwidth, in ``angle_unit``; ``None`` records
        an unsmoothed (binned) analysis.
    angle_unit : {"rad", "deg"}
        Unit of ``headings``, ``bin_size`` and ``bandwidth``.
    max_gap : float or None
        Maximum timestamp gap, in seconds, bridged within a tracked run.
    epochs, spike_window
        Resolved time windows.
    backend : str
        Resolved backend recorded in the provenance.
    n_jobs : int, default=1
        Parallel jobs for spike counting.
    break_before : ndarray of bool, shape (n_time,), or None
        Validated declared recording breaks.

    Returns
    -------
    _DirectionalCounts
        Counts of shape ``(n_units, n_bins)`` and occupancy ``(n_bins,)``,
        both unsmoothed, with the bin spacing and bandwidth in radians.
    """
    from neurospatial.encoding._directional_counts import (
        prepare_directional_binning,
    )
    from neurospatial.encoding._provenance import rate_analysis_settings

    # Counts and occupancy share one frame mask, including a single unit.
    prepared, bin_centers = prepare_directional_binning(
        times,
        headings,
        bin_size,
        angle_unit=angle_unit,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    )
    spike_counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)

    # Compute actual bin_size from bin_centers (handles non-divisible bin_size)
    # The binning layer rounds n_bins = int(round(2π / bin_size)), so the actual
    # bin spacing may differ from the requested bin_size
    actual_bin_size_rad = 2 * np.pi / len(bin_centers)

    # Convert bandwidth to radians for storage
    if angle_unit == "deg":
        bandwidth_rad = np.radians(bandwidth) if bandwidth else None
    else:
        bandwidth_rad = bandwidth

    settings = rate_analysis_settings(
        prepared.support,
        method="binned" if bandwidth_rad is None else "circular_gaussian",
        bandwidth=bandwidth_rad,
        backend=backend,
        bin_size=float(actual_bin_size_rad),
        angle_unit="rad",
        input_angle_unit=angle_unit,
    )
    return _DirectionalCounts(
        spike_counts=spike_counts,
        occupancy=occupancy,
        bin_centers=bin_centers,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        analysis_settings=settings,
    )


def compute_directional_rate(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float | None = None,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
) -> DirectionalRateResult:
    """Compute directional firing rate for one neuron.

    Computes a directional tuning curve from spike times and head direction
    data. The result is a DirectionalRateResult object containing the firing
    rate map, occupancy, and metadata.

    Head direction is a circular variable, not a position, so this function
    bins headings into angular bins and takes no :class:`Environment`, unlike
    :func:`compute_spatial_rate`, :func:`compute_egocentric_rate` and
    :func:`compute_view_rate`.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South.
        Any finite angle in ``angle_unit`` is accepted (for example
        [-π, π), [0, 2π) or unwrapped); NaN marks a missing sample.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.behavior.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float or None, default=None
        Width of angular bins, in ``angle_unit``. None means 6° (π/30 rad,
        60 bins) in either unit. The bin count is rounded to tile the circle,
        and ``result.bin_size`` records the width used, in radians.
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curve. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for output arrays (smoothing uses NumPy/SciPy)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    DirectionalRateResult
        Result object containing:

        - ``firing_rate``: Firing rate by direction in Hz, shape (n_bins,)
        - ``occupancy``: Time at each direction in seconds, shape (n_bins,)
        - ``bin_centers``: Angular bin centers in radians [-π, π), shape (n_bins,)
        - ``bin_size``: Bin width in radians
        - ``bandwidth``: Smoothing bandwidth in radians (or None)

    Raises
    ------
    ValueError
        If angle_unit is not 'rad' or 'deg'.

    See Also
    --------
    compute_directional_rates : Batch version for multiple neurons
    DirectionalRateResult : Result class with convenience methods

    Notes
    -----
    An interval is analyzed only if it is no longer than ``max_gap``, does
    not cross a declared break, starts at a finite heading and lies inside
    ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    The function uses the binning layer (``_directional_counts.py``) to convert
    spike times to spike counts and compute occupancy, then optionally applies
    Gaussian smoothing.

    **Algorithm**:

    1. Bin spike train into angular bins based on head direction at spike time
    2. Compute occupancy (time spent at each direction)
    3. Compute raw firing rate (spike counts / occupancy)
    4. Apply Gaussian smoothing if ``bandwidth`` is provided

    **Smoothing**: Uses circular Gaussian smoothing to handle the wrap-around
    at ±π. The smoothing is applied to both spike counts and occupancy
    separately (not to the ratio) to preserve proper rate estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import compute_directional_rate

    >>> # Create trajectory and spike times
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 60, 1800)  # 60 seconds at 30 Hz
    >>> headings = rng.uniform(0, 2 * np.pi, 1800)
    >>> spike_times = np.sort(rng.uniform(0, 60, 100))  # 100 spikes, sorted

    >>> # Compute directional rate (radians)
    >>> result = compute_directional_rate(spike_times, times, headings)
    >>> result.firing_rate.shape
    (60,)

    >>> # With smoothing
    >>> result = compute_directional_rate(
    ...     spike_times, times, headings, bandwidth=np.pi / 6
    ... )

    >>> # Using degrees
    >>> headings_deg = np.degrees(headings)
    >>> result = compute_directional_rate(
    ...     spike_times, times, headings_deg, bin_size=6.0, angle_unit="deg"
    ... )
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Convert inputs to arrays (1D required; validated below)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(
        times,
        headings=headings,
        context="compute_directional_rate",
        heading_unit=angle_unit,
    )
    validate_spike_times(spike_times, context="compute_directional_rate")
    breaks = validate_break_before(
        break_before, len(times), call="compute_directional_rate"
    )

    counts = _count_directional(
        [spike_times],
        times,
        headings,
        bin_size=bin_size,
        bandwidth=bandwidth,
        angle_unit=angle_unit,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        backend=resolved_backend,
        break_before=breaks,
    )
    spike_counts = counts.spike_counts[0]
    occupancy = counts.occupancy
    bin_centers = counts.bin_centers
    actual_bin_size_rad = counts.bin_size
    bandwidth_rad = counts.bandwidth
    settings = counts.analysis_settings

    # Apply smoothing if requested
    if bandwidth_rad is not None:
        from scipy.ndimage import gaussian_filter1d

        # Convert bandwidth to number of bins
        sigma_bins = bandwidth_rad / actual_bin_size_rad

        # Apply circular Gaussian smoothing
        # Use mode='wrap' for circular boundary conditions
        spike_counts_smooth = gaussian_filter1d(
            spike_counts, sigma=sigma_bins, mode="wrap"
        )
        occupancy_smooth = gaussian_filter1d(occupancy, sigma=sigma_bins, mode="wrap")
    else:
        spike_counts_smooth = spike_counts
        occupancy_smooth = occupancy

    # Compute firing rate (handle division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = spike_counts_smooth / occupancy_smooth
        # Set unvisited bins to NaN
        firing_rate[occupancy_smooth == 0] = np.nan

    # Convert to JAX arrays if JAX backend is selected
    occupancy_out: ArrayLike = occupancy
    bin_centers_out: ArrayLike = bin_centers
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        firing_rate = jnp.asarray(firing_rate)
        occupancy_out = jnp.asarray(occupancy)
        bin_centers_out = jnp.asarray(bin_centers)

    return DirectionalRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy_out,
        bin_centers=bin_centers_out,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        spike_counts=spike_counts,
        spike_window=resolved_spike_window,
        analysis_settings=settings,
    )


def compute_directional_rates(
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float | None = None,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> DirectionalRatesResult:
    """Compute directional firing rates for multiple neurons.

    This is the batch version of ``compute_directional_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (occupancy, bin centers) once and optionally parallelizes
    spike counting with joblib.

    Like :func:`compute_directional_rate`, it takes no :class:`Environment`:
    head direction is a circular variable, not a position.

    Parameters
    ----------
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South.
        Any finite angle in ``angle_unit`` is accepted (for example
        [-π, π), [0, 2π) or unwrapped); NaN marks a missing sample.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.behavior.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float or None, default=None
        Width of angular bins, in ``angle_unit``. None means 6° (π/30 rad,
        60 bins) in either unit. The bin count is rounded to tile the circle,
        and ``result.bin_size`` records the width used, in radians.
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curves. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    n_jobs : int, default=1
        Number of parallel jobs for spike binning. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for output arrays (smoothing uses NumPy/SciPy)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.

    Returns
    -------
    DirectionalRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each bin in seconds, shape ``(n_bins,)``
        - ``bin_centers``: Angular bin centers in radians [-π, π), shape ``(n_bins,)``
        - ``bin_size``: Bin width in radians
        - ``bandwidth``: Smoothing bandwidth in radians (or None)

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    See Also
    --------
    compute_directional_rate : Single-neuron version
    DirectionalRatesResult : Result class with batch methods

    Warns
    -----
    UserWarning
        When at least five units are all silent for at least 60 seconds of
        continuously tracked time and ``spike_window`` was not supplied.
        This is a heuristic for possible recording outages: it cannot detect
        an outage for a single unit or one shorter than 60 seconds, and its
        absence is not proof that recording coverage is correct.

    Notes
    -----
    An interval is analyzed only if it is no longer than ``max_gap``, does
    not cross a declared break, starts at a finite heading and lies inside
    ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Efficiency advantages over calling ``compute_directional_rate()`` in a loop**:

    1. Occupancy is computed once and shared across all neurons
    2. Bin centers are computed once
    3. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      shared computation matters
    - **Single**: One neuron, or when you need individual control per neuron

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import compute_directional_rates

    >>> # Create trajectory and spike times
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 60, 1800)  # 60 seconds at 30 Hz
    >>> headings = rng.uniform(0, 2 * np.pi, 1800)
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 60, 100)),  # Neuron 0
    ...     np.sort(rng.uniform(0, 60, 150)),  # Neuron 1
    ...     np.sort(rng.uniform(0, 60, 80)),  # Neuron 2
    ... ]

    >>> # Compute batch
    >>> result = compute_directional_rates(spike_times, times, headings)
    >>> result.firing_rates.shape
    (3, 60)

    >>> # With parallelization
    >>> result = compute_directional_rates(spike_times, times, headings, n_jobs=-1)

    >>> # Using degrees
    >>> headings_deg = np.degrees(headings)
    >>> result = compute_directional_rates(
    ...     spike_times, times, headings_deg, bin_size=6.0, angle_unit="deg"
    ... )
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    from neurospatial._spikes import as_labelled_spike_trains
    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._validation import (
        validate_spike_trains,
        validate_trajectory,
    )

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Normalize spike times to canonical list-of-arrays format and resolve the
    # per-unit identity labels (defaults to arange). A labelled input keeps its
    # own labels; a differing unit_ids= raises.
    spike_times_list, resolved_unit_ids, labels_supplied = as_labelled_spike_trains(
        spike_times, unit_ids, context="compute_directional_rates"
    )
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays (1D required; validated below)
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(
        times,
        headings=headings,
        context="compute_directional_rates",
        heading_unit=angle_unit,
    )
    validate_spike_trains(spike_times_list, context="compute_directional_rates")
    breaks = validate_break_before(
        break_before, len(times), call="compute_directional_rates"
    )

    # Recording coverage uses tracked runs, independently of speed or frame bins.
    _warn_if_population_silent(
        spike_times_list,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
    )

    counts = _count_directional(
        spike_times_list,
        times,
        headings,
        bin_size=bin_size,
        bandwidth=bandwidth,
        angle_unit=angle_unit,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        backend=resolved_backend,
        n_jobs=n_jobs,
        break_before=breaks,
    )
    spike_counts_batch = counts.spike_counts
    occupancy = counts.occupancy
    bin_centers = counts.bin_centers
    n_bins = len(bin_centers)
    actual_bin_size_rad = counts.bin_size
    bandwidth_rad = counts.bandwidth
    settings = counts.analysis_settings

    # Precompute smoothed occupancy once (instead of per-neuron)
    if bandwidth_rad is not None:
        from scipy.ndimage import gaussian_filter1d

        sigma_bins = bandwidth_rad / actual_bin_size_rad
        occupancy_smooth: NDArray[np.float64] = gaussian_filter1d(
            occupancy, sigma=sigma_bins, mode="wrap"
        )
    else:
        occupancy_smooth = occupancy

    # Handle empty neuron list
    if n_neurons == 0:
        empty_rates: ArrayLike = np.empty((0, n_bins), dtype=np.float64)
        empty_counts: ArrayLike = np.empty((0, n_bins), dtype=np.float64)
        occupancy_result: ArrayLike = occupancy
        bin_centers_result: ArrayLike = bin_centers
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            empty_rates = jnp.asarray(empty_rates)
            occupancy_result = jnp.asarray(occupancy)
            bin_centers_result = jnp.asarray(bin_centers)
        return DirectionalRatesResult(
            firing_rates=empty_rates,
            occupancy=occupancy_result,
            bin_centers=bin_centers_result,
            bin_size=actual_bin_size_rad,
            bandwidth=bandwidth_rad,
            spike_counts=empty_counts,
            unit_ids=resolved_unit_ids if labels_supplied else None,
            spike_window=resolved_spike_window,
            analysis_settings=settings,
        )

    # Helper function to process a single neuron's spike train
    def _process_neuron(
        spike_counts: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute (firing_rate, spike_counts) from shared-mask counts."""

        # Apply smoothing if requested
        if bandwidth_rad is not None:
            from scipy.ndimage import gaussian_filter1d

            spike_counts_smooth: NDArray[np.float64] = gaussian_filter1d(
                spike_counts, sigma=sigma_bins, mode="wrap"
            )
        else:
            spike_counts_smooth = spike_counts

        # Compute firing rate (handle division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            firing_rate: NDArray[np.float64] = spike_counts_smooth / occupancy_smooth
            # Set unvisited bins to NaN
            firing_rate[occupancy_smooth == 0] = np.nan

        # Return the unsmoothed counts for the tuning test.
        return firing_rate, spike_counts

    # Spike counting already honors n_jobs in the shared binning path.
    processed = [_process_neuron(counts) for counts in spike_counts_batch]

    firing_rates = np.array([rate for rate, _ in processed], dtype=np.float64)
    spike_counts_all: ArrayLike = np.array(
        [counts for _, counts in processed], dtype=np.float64
    )

    # Convert to JAX arrays if JAX backend is selected
    firing_rates_out: ArrayLike = firing_rates
    occupancy_out: ArrayLike = occupancy
    bin_centers_out: ArrayLike = bin_centers
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        firing_rates_out = jnp.asarray(firing_rates)
        occupancy_out = jnp.asarray(occupancy)
        bin_centers_out = jnp.asarray(bin_centers)

    return DirectionalRatesResult(
        firing_rates=firing_rates_out,
        occupancy=occupancy_out,
        bin_centers=bin_centers_out,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        spike_counts=spike_counts_all,
        unit_ids=resolved_unit_ids if labels_supplied else None,
        spike_window=resolved_spike_window,
        analysis_settings=settings,
    )


# ==============================================================================
# Convenience Functions
# ==============================================================================


def is_head_direction_cell(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    criterion: Literal["threshold", "shuffle"] = "threshold",
    min_mvl: float | None = None,
    alpha: float | None = None,
    n_shuffles: int | None = None,
    min_shift: float | None = None,
    rng: np.random.Generator | int | None = None,
    unit_id: Hashable | None = None,
    bin_size: float | None = None,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
) -> bool:
    """Classify one neuron by the head_direction_cell screen or circular-shift test.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South.
        Any finite angle in ``angle_unit`` is accepted (for example
        [-π, π), [0, 2π) or unwrapped); NaN marks a missing sample.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.behavior.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float or None, default=None
        Width of angular bins, in ``angle_unit``. None means 6° (π/30 rad,
        60 bins) in either unit. The bin count is rounded to tile the circle,
        and ``result.bin_size`` records the width used, in radians.
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curve. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for output arrays (smoothing uses NumPy/SciPy)
        - 'auto': Use JAX if available, otherwise NumPy

    criterion : {"threshold", "shuffle"}, default="threshold"
        Screen the observed statistic or test circular-shift significance.
    min_mvl : float or None, default=None
        Inclusive screen cutoff; None resolves to the family threshold constant.
    alpha : float or None, default=None
        P-value level for the occupancy-corrected tuning test (threshold
        mode) and the shuffle test; None resolves to 0.05.
    n_shuffles : int or None, default=None
        Number of circular shifts in shuffle mode; None resolves to 1000.
    min_shift : float or None, default=None
        Minimum shift in analyzed seconds; None resolves to 20.0.
    rng : numpy.random.Generator, int or None, default=None
        Shuffle random source; an integer seed with the same unit label is stable
        across single and population calls.
    unit_id : hashable or None, default=None
        Shuffle stream label; None uses label 0. Match the population's label.

    Returns
    -------
    bool
        Whether the chosen criterion is met.

    Raises
    ------
    ValueError
        If the criterion, inputs, or mode-specific keywords are invalid.

    Warns
    -----
    UserWarning
        In threshold mode, when the animal never faced a heading arc wider
        than 90°, because mean vector length is then inflated.

    Notes
    -----
    The threshold screen requires MVL >= min_mvl and
    ``DirectionalRateResult.tuning_pvalue() < alpha``. The MVL cutoff of 0.4
    is a common screening convention; check it against your own recordings.
    The tuning test checks one preferred direction given heading occupancy and
    assumes spikes are independent given heading; bursting and slow drifts
    in rate make its p-value too small. For publication, report the
    circular-shift test (``criterion="shuffle"``) and its assumptions.

    Circular shifting costs about n_shuffles recomputes of the plural map.
    Its null assumes stable firing statistics on the joined analyzed clock;
    recording gaps and excluded epochs are never shift destinations.
    Compare p_value < alpha; a significant association alone does not establish
    cell identity. Results keep no raw arrays or recompute closures.
    Threshold keywords belong only to the screen; shuffle keywords belong
    only to the shuffle. Passing a keyword for the other mode raises.

    See Also
    --------
    head_direction_cell_significance : Population significance on raw arrays.
    compute_directional_rate : Compute the map without classification.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import is_head_direction_cell
    >>> # Screen many neurons
    >>> times = np.linspace(0, 60, 1800)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1800)
    >>> spike_times = np.sort(np.random.uniform(0, 60, 100))
    >>> result = is_head_direction_cell(spike_times, times, headings)
    >>> type(result)
    <class 'bool'>
    """
    check_criterion(criterion, ("threshold", "shuffle"), call="is_head_direction_cell")
    check_mode_keywords(
        criterion,
        threshold={"min_mvl": min_mvl},
        shuffle={
            "n_shuffles": n_shuffles,
            "min_shift": min_shift,
            "rng": rng,
            "unit_id": unit_id,
        },
        call="is_head_direction_cell",
    )
    breaks = validate_break_before(
        break_before, len(times), call="is_head_direction_cell"
    )
    if criterion == "shuffle":
        label = 0 if unit_id is None else unit_id
        level = HEAD_DIRECTION_THRESHOLDS["alpha"] if alpha is None else alpha
        shuffle_result = head_direction_cell_significance(
            [spike_times],
            times,
            headings,
            unit_ids=[label],
            **resolve_shuffle_settings(n_shuffles, min_shift),
            rng=rng,
            bin_size=bin_size,
            bandwidth=bandwidth,
            angle_unit=angle_unit,
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
            spike_window=spike_window,
            backend=backend,
        )[label]
        return shuffle_result.p_value < level
    min_mvl = HEAD_DIRECTION_THRESHOLDS["min_mvl"] if min_mvl is None else min_mvl
    alpha = HEAD_DIRECTION_THRESHOLDS["alpha"] if alpha is None else alpha
    result = compute_directional_rate(
        spike_times,
        times,
        headings,
        bin_size=bin_size,
        bandwidth=bandwidth,
        angle_unit=angle_unit,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        spike_window=spike_window,
        backend=backend,
    )
    return result.is_head_direction_cell(min_mvl=min_mvl, alpha=alpha)


def plot_head_direction_tuning(
    result: DirectionalRateResult,
    ax: Axes | PolarAxes | None = None,
    *,
    polar: bool = True,
    show_preferred_direction: bool = True,
    show_metrics: bool = True,
    color: str = "C0",
    fill_alpha: float = 0.3,
    **kwargs: Any,
) -> Axes | PolarAxes:
    """Plot head direction tuning curve with metrics overlay.

    Creates standard head direction tuning visualization with optional polar
    or linear projection. Polar plots draw angles as in the arena: 0 = East
    (right), π/2 = North (up), counter-clockwise.

    Parameters
    ----------
    result : DirectionalRateResult
        Result from ``compute_directional_rate()``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with appropriate projection.
    polar : bool, default=True
        If True, create a polar plot (circular representation).
        If False, create a Cartesian plot (angle on x-axis, rate on y-axis).
    show_preferred_direction : bool, default=True
        If True, mark the preferred direction with a radial line.
    show_metrics : bool, default=True
        If True, show metrics text box with preferred direction, MVL, and peak rate.
    color : str, default='C0'
        Color for tuning curve line and fill.
    fill_alpha : float, default=0.3
        Alpha (transparency) for filled area under curve.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's plot function.

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes
        The axes object with the plot.

    Notes
    -----
    **Polar plot conventions**:

    - Angles are drawn as in the arena, 0 = East (right), π/2 = North (up),
      counter-clockwise, matching the heading convention of
      ``heading_from_velocity``. A caller-supplied polar axis is reset to
      this orientation.
    - Curve is closed (first point appended at end)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import (
    ...     compute_directional_rate,
    ...     plot_head_direction_tuning,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 60, 1800)
    >>> headings = rng.uniform(0, 2 * np.pi, 1800)
    >>> spike_times = np.sort(rng.uniform(0, 60, 100))
    >>> result = compute_directional_rate(spike_times, times, headings)
    >>> ax = plot_head_direction_tuning(result)  # doctest: +SKIP

    See Also
    --------
    DirectionalRateResult.plot : Basic plotting method on result object
    """
    import matplotlib.pyplot as plt

    # Convert to numpy for plotting
    rates = np.asarray(result.firing_rate, dtype=np.float64)
    centers = np.asarray(result.bin_centers, dtype=np.float64)

    # Close the curve by appending first point to end
    rates_closed = np.concatenate([rates, [rates[0]]])
    centers_closed = np.concatenate([centers, [centers[0] + 2 * np.pi]])

    if ax is None:
        if polar:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            _, ax = plt.subplots()

    if polar:
        polar_ax = cast("PolarAxes", ax)

        # Draw angles as in the arena: 0 = East (right), counter-clockwise.
        polar_ax.set_theta_zero_location("E")
        polar_ax.set_theta_direction(1)

        # Plot tuning curve
        polar_ax.plot(centers_closed, rates_closed, color=color, linewidth=2, **kwargs)

        # Fill under curve
        polar_ax.fill(centers_closed, rates_closed, color=color, alpha=fill_alpha)

        # Set angle labels in degrees
        polar_ax.set_thetagrids(
            [0, 45, 90, 135, 180, 225, 270, 315],
            ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
        )

        # Mark preferred direction
        if show_preferred_direction:
            pfd = result.preferred_direction()
            peak_rate = result.peak_firing_rate()
            polar_ax.plot(
                [pfd, pfd],
                [0, peak_rate],
                color="red",
                linewidth=2,
                linestyle="--",
                zorder=3,
            )

    else:
        # Linear projection
        x_closed = np.degrees(centers_closed)
        ax.set_xlabel("Head Direction (deg)")
        ax.set_xlim(-180, 180)

        # Plot tuning curve
        ax.plot(x_closed, rates_closed, color=color, linewidth=2, **kwargs)

        # Fill under curve
        ax.fill(x_closed, rates_closed, color=color, alpha=fill_alpha)

        ax.set_ylabel("Firing Rate (Hz)")

        # Mark preferred direction
        if show_preferred_direction:
            pfd_deg = result.preferred_direction_deg()
            ax.axvline(pfd_deg, color="red", linewidth=2, linestyle="--", zorder=3)

    # Show metrics text box if requested
    if show_metrics:
        metrics_text = (
            f"PFD: {result.preferred_direction_deg():.1f}°\n"
            f"MVL: {result.mean_vector_length():.3f}\n"
            f"Peak: {result.peak_firing_rate():.1f} Hz"
        )
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    return ax


def head_direction_cell_significance(
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float | None = None,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> dict[Hashable, ShuffleTestResult]:
    """Test head direction tuning against circular spike-time shifts.

    Recompute the population map for the supplied arrays and for each shifted
    train, using exactly the observed map's valid intervals. Shuffles preserve
    counts on the compressed analyzed clock and never enter recording gaps.
    This costs about n_shuffles population-map recomputes. The result's
    is_significant property uses 0.05; compare p_value < alpha for another level.

    Parameters
    ----------
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South.
        Any finite angle in ``angle_unit`` is accepted (for example
        [-π, π), [0, 2π) or unwrapped); NaN marks a missing sample.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.behavior.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float or None, default=None
        Width of angular bins, in ``angle_unit``. None means 6° (π/30 rad,
        60 bins) in either unit. The bin count is rounded to tile the circle,
        and ``result.bin_size`` records the width used, in radians.
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curves. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    n_jobs : int, default=1
        Number of parallel jobs for spike binning. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for output arrays (smoothing uses NumPy/SciPy)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.
    n_shuffles : int, default=1000
        Positive number of null map recomputes.
    min_shift : float, default=20.0
        Minimum shift in either direction, seconds of analyzed time.
    rng : numpy.random.Generator, int or None, default=None
        Integer seeds give stable per-unit shifts regardless of population order.

    Returns
    -------
    dict[Hashable, ShuffleTestResult]
        Per-unit observed score, null scores, corrected p-value and z-score,
        keyed by unit label in input order. Invalid observed scores yield NaN p-values.

    Raises
    ------
    ValueError
        For invalid inputs/settings, duplicate/conflicting unit labels or
        insufficient analyzed time.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import head_direction_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> headings = np.angle(np.exp(1j * 0.3 * times))  # turns through all headings
    >>> trains = [np.sort(rng.uniform(0, 59.9, 40))]
    >>> results = head_direction_cell_significance(
    ...     trains, times, headings, n_shuffles=20, rng=0
    ... )
    >>> list(results)
    [0]
    >>> results[0].n_shuffles
    20
    """
    from neurospatial._intervals import resolve_time_windows
    from neurospatial._results import resolve_unit_ids
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial.encoding._directional_counts import directional_frame_bins
    from neurospatial.encoding._significance import (
        run_shuffle_test,
        snapshot_options,
        to_shuffle_results,
    )
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    trains, input_ids = _as_spike_trains_with_ids(spike_times)
    trains = [np.array(train, dtype=np.float64, copy=True) for train in trains]
    times = np.array(times, dtype=np.float64, copy=True)
    headings = np.array(headings, dtype=np.float64, copy=True)
    ids = np.array(
        resolve_unit_ids(
            unit_ids,
            len(trains),
            input_ids=input_ids,
            context="head_direction_cell_significance",
        ),
        copy=True,
    )
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    options: dict[str, Any] = {
        "bin_size": bin_size,
        "bandwidth": bandwidth,
        "angle_unit": angle_unit,
        "max_gap": max_gap,
        "n_jobs": n_jobs,
        "backend": backend,
    }
    options = snapshot_options(
        options,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        unit_ids=ids,
    )
    validate_trajectory(
        times,
        headings=headings,
        context="head_direction_cell_significance",
        heading_unit=angle_unit,
    )
    for train in trains:
        validate_spike_times(train, context="head_direction_cell_significance")
    breaks = validate_break_before(
        break_before, len(times), call="head_direction_cell_significance"
    )
    options["break_before"] = breaks
    frame_bins, _ = directional_frame_bins(headings, bin_size, angle_unit=angle_unit)
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=options["break_before"],
    )
    windows = run_time_bounds(times, mask)

    def statistic(shifted: list[NDArray[np.float64]]) -> ArrayLike:
        # The heading-unit check already ran once above; do not repeat it for
        # every shuffle.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*look like degrees", category=UserWarning
            )
            rates = compute_directional_rates(shifted, times, headings, **options)
        return rates.mean_vector_lengths()

    observed, null = run_shuffle_test(
        statistic,
        trains,
        windows,
        ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
    return to_shuffle_results(observed, null, ids)
