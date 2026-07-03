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
>>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

from neurospatial.encoding._base import SpatialResultMixin

__all__ = [
    # Result classes
    "DirectionalRateResult",
    "DirectionalRatesResult",
    # Compute functions
    "compute_directional_rate",
    "compute_directional_rates",
    # Convenience functions
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


@dataclass(frozen=True)
class DirectionalRateResult(SpatialResultMixin):
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
        Center of each angular bin in radians [0, 2π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    bandwidth : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.
    spike_counts : ArrayLike or None, optional
        Raw (unsmoothed) spike count per angular bin, shape (n_bins,). Used as
        count weights for the Rayleigh test. None for results constructed
        without counts (e.g. from external tuning curves), in which case
        rayleigh_pvalue() falls back to occupancy-implied counts.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by direction in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians. Shape is (n_bins,).
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

    Notes
    -----
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
    >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
            ax.set_xlim(0, 2 * np.pi)

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        from neurospatial.stats.circular import circular_mean

        rates = np.asarray(self.firing_rate, dtype=np.float64)
        centers = np.asarray(self.bin_centers, dtype=np.float64)

        # Mask out NaN values (from unvisited bins) before computing circular mean
        valid_mask = ~np.isnan(rates)
        if not np.any(valid_mask):
            return float(np.nan)

        return circular_mean(centers[valid_mask], weights=rates[valid_mask])

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        """Compute the mean vector length (Rayleigh MVL) of the tuning curve.

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

        **Interpretation guidelines (Taube et al., 1990)**:

        - MVL < 0.2: Weak or no directional tuning
        - MVL 0.2-0.4: Moderate tuning
        - MVL > 0.4: Strong tuning (typical HD cell threshold)

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        rayleigh_pvalue : Statistical test for non-uniformity
        neurospatial.stats.circular.mean_resultant_length : Underlying function
        """
        from neurospatial.stats.circular import mean_resultant_length

        rates = np.asarray(self.firing_rate, dtype=np.float64)
        centers = np.asarray(self.bin_centers, dtype=np.float64)

        # Mask out NaN values (from unvisited bins) before computing MVL
        valid_mask = ~np.isnan(rates)
        if not np.any(valid_mask):
            return float(np.nan)

        return mean_resultant_length(centers[valid_mask], weights=rates[valid_mask])

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        rates = np.asarray(self.firing_rate, dtype=np.float64)

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
        right_width = _half_max_halfwidth(
            rates, peak_idx, half_max, self.bin_size, step=+1
        )
        left_width = _half_max_halfwidth(
            rates, peak_idx, half_max, self.bin_size, step=-1
        )

        # Average the finite half-widths. If only one side crosses (the other
        # is masked-out NaN all the way round), report the single finite side
        # rather than NaN.
        halves = np.array([left_width, right_width])
        finite = halves[np.isfinite(halves)]
        if finite.size == 0:
            return float(np.nan)
        return float(finite.mean())

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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

    def rayleigh_pvalue(self) -> float:
        """Compute the Rayleigh test p-value for directional non-uniformity.

        The Rayleigh test evaluates whether the directional tuning is
        statistically significant (i.e., the firing rate distribution is
        non-uniform across directions).

        Returns
        -------
        float
            P-value from the Rayleigh test in [0, 1]. Small values (< 0.05)
            indicate significant directional tuning. Returns ``nan`` when
            fewer than 3 angular bins have a positive spike count, since the
            test cannot be evaluated reliably with so few populated bins.

        Notes
        -----
        The Rayleigh test uses the mean vector length (R) to compute a
        test statistic. The bin centers are the angles and the **spike
        counts** are the frequency weights:

            z = sum(spike_counts) * R²

        Spike counts (not firing rates in Hz) are the correct weights: the
        Rayleigh statistic treats the weights as event frequencies, so using
        rates would make the p-value depend on bin dwell-time and the absolute
        rate scale. When ``spike_counts`` is None (e.g. results built from an
        external tuning curve), counts are reconstructed as
        ``firing_rate * occupancy``, which is still a count and keeps the
        statistic scale-correct.

        Only **occupied** heading bins (positive occupancy and finite firing
        rate) contribute to the test. Spikes assigned to a bin the animal
        never visited (zero occupancy, NaN rate) are excluded, so an unvisited
        bin can never drive significance. If fewer than 3 spikes fall in
        occupied bins, the result is undefined and ``nan`` is returned.

        **Interpretation**:

        - p < 0.001: Very strong evidence for directional tuning
        - p < 0.01: Strong evidence
        - p < 0.05: Significant evidence
        - p >= 0.05: No significant directional tuning

        **Caveats**:

        - The test assumes unimodal tuning. For bimodal cells (rare), it may
          fail to detect significant tuning.
        - With many bins and high spike counts, even weak tuning may be
          statistically significant but biologically meaningless. Always
          check mean_vector_length() as well.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        >>> # Sharply tuned neuron
        >>> firing_rate = 10.0 * np.exp(5.0 * (np.cos(bin_centers - np.pi / 2) - 1))
        >>> result = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> pval = result.rayleigh_pvalue()
        >>> pval < 0.05  # Significant non-uniformity
        True

        Building a result *with* explicit spike counts uses those counts as
        the Rayleigh weights; without them, counts are reconstructed from
        ``firing_rate * occupancy``:

        >>> occupancy = np.ones(n_bins) * 0.5
        >>> counts = np.round(firing_rate * occupancy)
        >>> with_counts = DirectionalRateResult(
        ...     firing_rate=firing_rate,
        ...     occupancy=occupancy,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ...     spike_counts=counts,
        ... )
        >>> with_counts.rayleigh_pvalue() < 0.05
        True

        See Also
        --------
        mean_vector_length : Tuning strength measure
        is_head_direction_cell : Classification combining MVL and p-value
        neurospatial.stats.circular.rayleigh_test : Underlying test function
        """
        from neurospatial.stats.circular import rayleigh_test

        centers = np.asarray(self.bin_centers, dtype=np.float64)
        rates = np.asarray(self.firing_rate, dtype=np.float64)

        # Count weights for the Rayleigh test. The test treats weights as
        # FREQUENCIES (z = sum(weights) * R**2); firing rate in Hz is the wrong
        # quantity because it is occupancy- and rate-scale-dependent.
        if self.spike_counts is not None:
            counts = np.asarray(self.spike_counts, dtype=np.float64)
        else:
            # Fallback for results built without raw counts: reconstruct an
            # integer-like count from rate * occupancy. Still a count, so the
            # statistic remains scale-correct.
            occ = np.asarray(self.occupancy, dtype=np.float64)
            counts = rates * occ

        # Drop bins with no valid weight (unvisited -> NaN rate / NaN count,
        # or zero count). A bin with zero spikes contributes nothing to the
        # resultant and must not be passed as a zero weight that still counts
        # toward n. Crucially, also exclude bins the animal never occupied:
        # an unvisited heading bin has zero occupancy (and therefore a NaN
        # firing rate), yet raw spike counts assigned to such a bin would
        # otherwise drive spurious significance. Requiring positive occupancy
        # AND a finite rate guarantees an unvisited bin can never contribute,
        # while a genuinely-visited cell concentrated in 1-2 occupied bins
        # (occupancy > 0, finite rate) still counts.
        occupancy = np.asarray(self.occupancy, dtype=np.float64)
        firing_rate = np.asarray(self.firing_rate, dtype=np.float64)
        valid = (
            np.isfinite(centers)
            & np.isfinite(counts)
            & (counts > 0)
            & np.isfinite(firing_rate)
            & (occupancy > 0)
        )
        # Gate on the total spike count, not the number of occupied bins. A
        # strongly-tuned cell can concentrate all its spikes in 1-2 angular
        # bins; the effective sample size for the weighted Rayleigh test is
        # sum(counts), so reject only when too few spikes are present overall.
        if counts[valid].sum() < 3:
            return float(np.nan)

        _, pval = rayleigh_test(centers[valid], weights=counts[valid])
        return pval

    def is_head_direction_cell(self, min_mvl: float = 0.4, alpha: float = 0.05) -> bool:
        """Classify as head direction cell.

        A neuron is classified as a head direction (HD) cell if it meets
        both of the following criteria (Taube et al., 1990):

        1. Mean vector length (MVL) > min_mvl (default 0.4)
        2. Rayleigh test p-value < alpha (default 0.05)

        Parameters
        ----------
        min_mvl : float, default=0.4
            Minimum mean vector length threshold.

            **How was 0.4 chosen?**

            This threshold comes from Taube et al. (1990) analyzing
            postsubicular HD cells in rats. Empirically:

            - Classic HD cells: 0.5-0.8
            - Borderline HD cells: 0.3-0.5
            - Non-HD cells: 0.1-0.3

            **When to adjust:**

            - Other brain regions: May need 0.3-0.5
            - Different species: Validate threshold first
            - Noisy recordings: Consider 0.3 (more permissive)
            - Publication quality: Use 0.5 (more conservative)

        alpha : float, default=0.05
            Significance level for Rayleigh test. A neuron must have a
            p-value below this threshold to be classified as an HD cell.

        Returns
        -------
        bool
            True if the neuron passes both HD cell criteria.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        rayleigh_pvalue : Statistical test for non-uniformity
        interpretation : Human-readable summary of classification

        References
        ----------
        Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells
            recorded from the postsubiculum in freely moving rats. I.
            Description and quantitative analysis. J Neurosci, 10(2), 420-435.
        """
        return self.mean_vector_length() > min_mvl and self.rayleigh_pvalue() < alpha

    def interpretation(self, min_mvl: float = 0.4) -> str:
        """Human-readable interpretation of head direction metrics.

        Provides a comprehensive summary of the neuron's directional tuning
        properties and classification status. For neurons classified as HD
        cells, includes preferred direction, tuning strength, and peak rate.
        For non-HD cells, explains which criteria were not met.

        Parameters
        ----------
        min_mvl : float, default=0.4
            Minimum mean vector length threshold for HD cell classification.
            Same parameter as in :meth:`is_head_direction_cell`.

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
            - Rayleigh test p-value

            **For non-HD cells:**
            - Header indicating non-HD cell
            - Explanation of which criteria failed
            - Guidance on threshold selection

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRateResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        Rayleigh test: p = 0.xxxx

        See Also
        --------
        is_head_direction_cell : Boolean classification method
        mean_vector_length : Tuning strength
        rayleigh_pvalue : Statistical significance
        """
        lines = []
        alpha = 0.05  # Fixed significance level for Rayleigh test

        mvl = self.mean_vector_length()
        pval = self.rayleigh_pvalue()
        is_hd = self.is_head_direction_cell(min_mvl=min_mvl, alpha=alpha)

        if is_hd:
            lines.append("*** HEAD DIRECTION CELL ***")
            lines.append(
                f"Preferred direction: {self.preferred_direction_deg():.1f} deg"
            )
            lines.append(f"Mean vector length: {mvl:.3f} (threshold = {min_mvl})")
            lines.append(f"Peak firing rate: {self.peak_firing_rate():.1f} Hz")
            lines.append(f"Tuning width (HWHM): {self.tuning_width_deg():.1f} deg")
            lines.append(f"Rayleigh test: p = {pval:.4f}")
        else:
            lines.append("Not classified as HD cell")
            if mvl < min_mvl:
                lines.append(f"  - Mean vector length too low: {mvl:.3f} < {min_mvl}")
                lines.append(
                    f"    How was {min_mvl} chosen? Default 0.4 is from "
                    "Taube et al. (1990) analyzing"
                )
                lines.append("    postsubicular HD cells in rats. Empirically:")
                lines.append("      Classic HD cells: 0.5-0.8")
                lines.append("      Borderline HD cells: 0.3-0.5")
                lines.append("      Non-HD cells: 0.1-0.3")
                lines.append("    When to adjust:")
                lines.append("      - Other brain regions: May need 0.3-0.5")
                lines.append("      - Different species: Validate threshold first")
                lines.append("      - Noisy recordings: Consider 0.3 (more permissive)")
                lines.append("      - Publication quality: Use 0.5 (more conservative)")
            if pval >= alpha:
                lines.append(
                    f"  - Rayleigh test not significant: p = {pval:.3f} >= {alpha}"
                )

        return "\n".join(lines)


@dataclass(frozen=True)
class DirectionalRatesResult(SpatialResultMixin):
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
        Center of each angular bin in radians [0, 2π). Shape is (n_bins,).
    bin_size : float
        Width of each angular bin in radians.
    bandwidth : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.
    spike_counts : ArrayLike or None, optional
        Raw (unsmoothed) spike counts per angular bin, shape
        (n_neurons, n_bins). Forwarded to each per-neuron result for the
        Rayleigh test. None if the batch was built without counts.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates by direction in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time at each direction in seconds. Shape is (n_bins,).
    bin_centers : ArrayLike
        Angular bin centers in radians. Shape is (n_bins,).
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
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. Rides alongside the rates for downstream filtering/grouping.

    Notes
    -----
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
    >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        from neurospatial._results import resolve_unit_ids, validate_unit_table

        n_units = int(np.asarray(self.firing_rates).shape[0])
        object.__setattr__(
            self,
            "unit_ids",
            resolve_unit_ids(self.unit_ids, n_units),
        )
        validate_unit_table(self.unit_table, n_units, context="DirectionalRatesResult")

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: directional results store bin centers
        # directly on the dataclass (no Environment).
        return np.asarray(self.bin_centers, dtype=np.float64)

    def _bin_center_columns(self) -> dict[str, NDArray[np.float64]]:
        # Directional results have no Environment; the bin center is the
        # angular center (radians). Emit it under the shared vocabulary name.
        return {"bin_center_angle": np.asarray(self.bin_centers, dtype=np.float64)}

    def to_xarray(self) -> Any:
        """Convert the tuning curves to a labeled :class:`xarray.Dataset`.

        Wraps the ``(n_units, n_bins)`` directional tuning-curve matrix in a
        labeled :class:`xarray.Dataset` with dims ``("unit_id", "bin")``. The
        ``unit_id`` index coordinate holds the real per-unit identity labels
        (:attr:`unit_ids`). Directional results have no spatial environment;
        the ``bin`` dimension indexes angular bins and carries a
        ``bin_center_angle`` non-index coordinate (radians, from
        :attr:`bin_centers`).

        Returns
        -------
        xarray.Dataset
            Dataset with data var ``firing_rate`` (Hz, dims
            ``("unit_id", "bin")``), data var ``occupancy`` (seconds, dims
            ``("bin",)``), index coord ``unit_id`` = :attr:`unit_ids`,
            ``bin_center_angle`` coord (radians) on ``bin``, and ``attrs``
            carrying ``units`` (``"radians"``), ``bandwidth``, and
            ``software_version``.

        Raises
        ------
        ValueError
            If :attr:`unit_ids` contains duplicate labels.
        ImportError
            If ``xarray`` is not installed (optional dependency).
        """
        from neurospatial._results import (
            build_population_dataset,
            software_version,
        )

        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        attrs: dict[str, Any] = {
            "units": "radians",
            "bandwidth": self.bandwidth,
            "software_version": software_version(),
        }
        return build_population_dataset(
            rates,
            np.asarray(self.unit_ids),
            bin_centers=np.asarray(self.bin_centers, dtype=np.float64),
            occupancy=np.asarray(self.occupancy, dtype=np.float64),
            attrs=attrs,
        )

    def __len__(self) -> int:
        """Return number of units.

        Returns
        -------
        int
            Number of neurons (first dimension of firing_rates).
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return int(rates.shape[0])

    def __getitem__(self, idx: int) -> DirectionalRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Neuron index (0-indexed).

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        counts = self.spike_counts
        return DirectionalRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            bandwidth=self.bandwidth,
            spike_counts=(None if counts is None else np.asarray(counts)[idx]),
            unit_id=np.asarray(self.unit_ids)[idx].item(),
        )

    def __iter__(self) -> Iterator[DirectionalRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        DirectionalRateResult
            Result for each neuron in order.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
        >>> rng = np.random.default_rng(0)
        >>> results = DirectionalRatesResult(
        ...     firing_rates=rng.random((3, n_bins)) * 10,
        ...     occupancy=np.ones(n_bins) * 0.5,
        ...     bin_centers=bin_centers,
        ...     bin_size=np.pi / 30,
        ...     bandwidth=None,
        ... )
        >>> peaks = [float(np.max(result.firing_rate)) for result in results]
        >>> len(peaks)
        3
        """
        for i in range(len(self)):
            yield self[i]

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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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

    def classify(
        self, *, min_mvl: float = 0.4, alpha: float = 0.05
    ) -> NDArray[np.bool_]:
        """Classify neurons as head direction cells.

        A neuron is classified as a head direction (HD) cell if it meets
        both criteria (Taube et al., 1990):

        1. Mean vector length (MVL) > min_mvl (default 0.4)
        2. Rayleigh test p-value < alpha (default 0.05)

        This is the single-type boolean predicate ("is this an HD cell") for
        the batch result.

        Parameters
        ----------
        min_mvl : float, default=0.4
            Minimum mean vector length threshold.
        alpha : float, default=0.05
            Significance level for Rayleigh test.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (n_neurons,). True indicates the neuron
            is classified as an HD cell.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.directional import DirectionalRatesResult
        >>> n_bins = 60
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        """
        n_neurons = len(self)
        is_hd = np.empty(n_neurons, dtype=np.bool_)

        for i in range(n_neurons):
            is_hd[i] = self[i].is_head_direction_cell(min_mvl=min_mvl, alpha=alpha)

        return is_hd

    def detect_hd_cells(
        self, min_mvl: float = 0.4, alpha: float = 0.05
    ) -> NDArray[np.bool_]:
        """Deprecated alias for :meth:`classify`.

        .. deprecated:: 0.6
            ``detect_hd_cells`` is deprecated since 0.6; use
            :meth:`classify` instead. Removed in 0.7.

        Parameters
        ----------
        min_mvl : float, default=0.4
            Minimum mean vector length threshold.
        alpha : float, default=0.05
            Significance level for Rayleigh test.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape (n_neurons,). True indicates an HD cell.
        """
        warnings.warn(
            "detect_hd_cells is deprecated since 0.6, use classify; removed in 0.7",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.classify(min_mvl=min_mvl, alpha=alpha)

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
            One row per unit, indexed by ``unit_id``, with columns:

            - preferred_direction: preferred direction in radians [-π, π]
            - preferred_direction_deg: preferred direction in degrees [-180, 180]
            - mean_vector_length: mean vector length [0, 1]
            - tuning_width: tuning width (HWHM) in radians (0, π]
            - tuning_width_deg: tuning width (HWHM) in degrees (0, 180]
            - peak_rate: maximum firing rate (Hz)
            - is_head_direction_cell: whether classified as HD cell (using default thresholds)

        Raises
        ------
        ValueError
            If unit_ids has a different length than the number of units.

        Notes
        -----
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
        >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
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
        import pandas as pd

        n_neurons = len(self)

        if unit_ids is None:
            index_ids: list[str | int] = list(np.asarray(self.unit_ids))
        else:
            index_ids = list(unit_ids)
            if len(index_ids) != n_neurons:
                raise ValueError(
                    f"unit_ids has {len(index_ids)} elements but "
                    f"result contains {n_neurons} units"
                )

        # Compute all metrics
        pref_dirs = self.preferred_directions()
        mvls = self.mean_vector_lengths()
        widths = self.tuning_widths()
        peaks = self.peak_firing_rate()
        is_hd = self.classify()

        # Build data dictionary
        data: dict[str, Any] = {
            "preferred_direction": pref_dirs,
            "preferred_direction_deg": np.degrees(pref_dirs),
            "mean_vector_length": mvls,
            "tuning_width": widths,
            "tuning_width_deg": np.degrees(widths),
            "peak_rate": peaks,
            "is_head_direction_cell": is_hd,
        }

        return pd.DataFrame(data, index=pd.Index(index_ids, name="unit_id"))


def compute_directional_rate(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = np.pi / 30,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> DirectionalRateResult:
    """Compute directional firing rate for one neuron.

    Computes a directional tuning curve from spike times and head direction
    data. The result is a DirectionalRateResult object containing the firing
    rate map, occupancy, and metadata.

    .. note::

       Directional encoding is the documented exception to the v0.4
       canonical "env first" argument order for encoding functions
       (see :ref:`canonical-argument-order` in the project guide).
       Heading is a circular angular variable, not a position in a
       spatial environment, so this function operates on circular
       heading bins (``bin_size``, ``angle_unit``) rather than an
       :class:`Environment`. There is no ``env`` parameter and no
       internal Environment shim — keeping the signature
       heading-domain native is intentional. Sister classifiers
       :func:`compute_spatial_rate`, :func:`compute_egocentric_rate`,
       and :func:`compute_view_rate` all take ``env`` first because
       they operate over a discretized spatial environment.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South,
        wrapped to [-π, π] (or to [0, 360°) when ``angle_unit="deg"``).
        Units determined by ``angle_unit``.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.ops.egocentric.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float, default=π/30 (6 degrees)
        Width of angular bins. Units match ``angle_unit``.
        Default produces 60 bins (6° resolution).
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curve. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees

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
        - ``bin_centers``: Angular bin centers in radians, shape (n_bins,)
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
    The function uses the binning layer (``_directional_binning.py``) to convert
    spike times to spike counts and compute occupancy, then optionally applies
    Gaussian smoothing.

    **Algorithm**:

    1. Bin spike train into angular bins based on head direction at spike time
    2. Compute occupancy (time spent at each direction)
    3. Compute raw firing rate (spike counts / occupancy)
    4. Apply Gaussian smoothing if ``bandwidth`` is provided

    **Smoothing**: Uses circular Gaussian smoothing to handle the wrap-around
    at 0/2π. The smoothing is applied to both spike counts and occupancy
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
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._directional_binning import (
        bin_directional_spike_train,
        compute_directional_occupancy,
    )
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Convert inputs to arrays (1D required; validated below)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(times, headings=headings, context="compute_directional_rate")
    validate_spike_times(spike_times, context="compute_directional_rate")

    # Compute occupancy and bin centers
    occupancy, bin_centers = compute_directional_occupancy(
        times, headings, bin_size, angle_unit=angle_unit
    )

    # Bin spike train
    spike_counts = bin_directional_spike_train(
        spike_times, times, headings, bin_size, angle_unit=angle_unit
    )

    # Compute actual bin_size from bin_centers (handles non-divisible bin_size)
    # The binning layer rounds n_bins = int(round(2π / bin_size)), so the actual
    # bin spacing may differ from the requested bin_size
    n_bins = len(bin_centers)
    actual_bin_size_rad = 2 * np.pi / n_bins

    # Convert bandwidth to radians for storage
    if angle_unit == "deg":
        bandwidth_rad = np.radians(bandwidth) if bandwidth else None
    else:
        bandwidth_rad = bandwidth

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
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        firing_rate = jnp.asarray(firing_rate)
        occupancy = jnp.asarray(occupancy)
        bin_centers = jnp.asarray(bin_centers)

    return DirectionalRateResult(
        firing_rate=firing_rate,
        occupancy=occupancy,
        bin_centers=bin_centers,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        spike_counts=spike_counts,
    )


def compute_directional_rates(
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = np.pi / 30,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> DirectionalRatesResult:
    """Compute directional firing rates for multiple neurons.

    This is the batch version of ``compute_directional_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (occupancy, bin centers) once and optionally parallelizes
    spike counting with joblib.

    .. note::

       Like :func:`compute_directional_rate`, this function is the documented
       exception to the v0.4 canonical "env first" argument order for
       encoding functions (see :ref:`canonical-argument-order` in the
       project guide). Heading is a circular angular variable, not a
       position in a spatial environment, so this signature is
       heading-domain native and intentionally takes no
       :class:`Environment`. Sister batch encoders
       (:func:`compute_spatial_rates`, :func:`compute_egocentric_rates`,
       :func:`compute_view_rates`) keep their env-first signatures.

    Parameters
    ----------
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically

        All formats are coerced to per-neuron spike trains via ``as_spike_trains()``.
    times : ndarray, shape (n_samples,)
        Timestamps of head direction samples in seconds.
    headings : ndarray, shape (n_samples,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South,
        wrapped to [-π, π] (or to [0, 360°) when ``angle_unit="deg"``).
        Units determined by ``angle_unit``.

        **Movement heading vs. head direction.** This function expects the
        animal's *head direction* (where the head points, typically from a
        head-mounted LED pair or pose tracking). A velocity-derived heading
        (e.g. from :func:`neurospatial.ops.egocentric.heading_from_velocity`)
        is the direction of *movement*, which equals head direction only when
        the animal moves the way it faces. Feeding movement heading here and
        reporting the result as a "head direction cell" is a common
        methodological mislabel — keep the two distinct.
    bin_size : float, default=π/30 (6 degrees)
        Width of angular bins. Units match ``angle_unit``.
        Default produces 60 bins (6° resolution).
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth for the tuning curves. Units match
        ``angle_unit``. If None, no smoothing is applied.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``headings``, ``bin_size``, and ``bandwidth``.

        - 'rad': angles in radians
        - 'deg': angles in degrees

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
        indexing/iterating. Defaults to ``np.arange(n_neurons)``. A
        wrong-length value raises ``ValueError``.

    Returns
    -------
    DirectionalRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each bin in seconds, shape ``(n_bins,)``
        - ``bin_centers``: Angular bin centers in radians, shape ``(n_bins,)``
        - ``bin_size``: Bin width in radians
        - ``bandwidth``: Smoothing bandwidth in radians (or None)

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    See Also
    --------
    compute_directional_rate : Single-neuron version
    DirectionalRatesResult : Result class with batch methods

    Notes
    -----
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
    from neurospatial.encoding._backend import (
        SUPPORTED_BACKENDS,
        get_backend_name,
        is_jax_available,
    )
    from neurospatial.encoding._directional_binning import (
        bin_directional_spike_train,
        compute_directional_occupancy,
    )
    from neurospatial.encoding._spikes import as_spike_trains
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Supported backends are: {', '.join(repr(b) for b in SUPPORTED_BACKENDS)}"
        )

    # Resolve backend (handles "auto" → "numpy" or "jax")
    # This raises ImportError if backend="jax" and JAX is unavailable
    resolved_backend = get_backend_name(backend)

    # Validate angle_unit
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")

    # Normalize spike times to canonical format
    spike_times_list: list[NDArray[np.float64]] = as_spike_trains(spike_times)
    n_neurons = len(spike_times_list)

    # Resolve and validate per-unit identity labels (defaults to arange).
    from neurospatial._results import resolve_unit_ids

    resolved_unit_ids = resolve_unit_ids(
        unit_ids, n_neurons, context="compute_directional_rates"
    )

    # Convert inputs to arrays (1D required; validated below)
    times = np.asarray(times, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    validate_trajectory(times, headings=headings, context="compute_directional_rates")
    for i, st in enumerate(spike_times_list):
        validate_spike_times(st, context=f"compute_directional_rates (neuron {i})")

    # Precompute shared quantities: occupancy and bin centers
    occupancy, bin_centers = compute_directional_occupancy(
        times, headings, bin_size, angle_unit=angle_unit
    )
    n_bins = len(bin_centers)

    # Compute actual bin_size from bin_centers (handles non-divisible bin_size)
    # The binning layer rounds n_bins = int(round(2π / bin_size)), so the actual
    # bin spacing may differ from the requested bin_size
    actual_bin_size_rad = 2 * np.pi / n_bins

    # Convert bandwidth to radians for storage
    if angle_unit == "deg":
        bandwidth_rad = np.radians(bandwidth) if bandwidth else None
    else:
        bandwidth_rad = bandwidth

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
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            empty_rates = jnp.asarray(empty_rates)
            occupancy = jnp.asarray(occupancy)
            bin_centers = jnp.asarray(bin_centers)
        return DirectionalRatesResult(
            firing_rates=empty_rates,
            occupancy=occupancy,
            bin_centers=bin_centers,
            bin_size=actual_bin_size_rad,
            bandwidth=bandwidth_rad,
            spike_counts=empty_counts,
            unit_ids=resolved_unit_ids,
        )

    # Helper function to process a single neuron's spike train
    def _process_neuron(
        neuron_spikes: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Bin spike train and compute (firing_rate, spike_counts) for one neuron."""
        spike_counts = bin_directional_spike_train(
            neuron_spikes, times, headings, bin_size, angle_unit=angle_unit
        )

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

        # Return the unsmoothed counts for the Rayleigh test weights.
        return firing_rate, spike_counts

    # Process neurons (sequential or parallel)
    if n_jobs == 1 or n_neurons <= 1:
        # Sequential processing
        processed = [_process_neuron(spikes) for spikes in spike_times_list]
    else:
        # Parallel processing with joblib
        from joblib import Parallel, delayed

        processed = Parallel(n_jobs=n_jobs)(
            delayed(_process_neuron)(spikes) for spikes in spike_times_list
        )

    firing_rates = np.array([rate for rate, _ in processed], dtype=np.float64)
    spike_counts_all: ArrayLike = np.array(
        [counts for _, counts in processed], dtype=np.float64
    )

    # Convert to JAX arrays if JAX backend is selected
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        firing_rates = jnp.asarray(firing_rates)
        occupancy = jnp.asarray(occupancy)
        bin_centers = jnp.asarray(bin_centers)

    return DirectionalRatesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        bin_centers=bin_centers,
        bin_size=actual_bin_size_rad,
        bandwidth=bandwidth_rad,
        spike_counts=spike_counts_all,
        unit_ids=resolved_unit_ids,
    )


# ==============================================================================
# Convenience Functions
# ==============================================================================


def is_head_direction_cell(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    bin_size: float = np.pi / 30,
    bandwidth: float | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    min_mvl: float = 0.4,
    alpha: float = 0.05,
) -> bool:
    """Quick check: Is this a head direction cell?

    Convenience function for fast screening of neurons. Computes directional
    tuning and checks if the neuron meets HD cell criteria.

    For detailed metrics, use ``compute_directional_rate()`` and inspect
    the result's methods (``is_head_direction_cell()``, ``mean_vector_length()``, etc.).

    .. note::

       Like :func:`compute_directional_rate`, this function is the documented
       exception to the v0.4 canonical "env first" argument order for
       encoding functions (see :ref:`canonical-argument-order` in the
       project guide). Heading is a circular angular variable, not a
       position in a spatial environment, so this signature is
       heading-domain native and intentionally takes no
       :class:`Environment`. Sister classifiers
       (:func:`is_object_vector_cell`, :func:`is_spatial_view_cell`) keep
       their env-first signatures because they operate on spatial
       (allocentric) firing fields.

    Parameters
    ----------
    spike_times : ndarray of shape (n_spikes,)
        Times of spikes (same time units as times).
    times : ndarray of shape (n_frames,)
        Timestamps corresponding to each head direction sample.
    headings : ndarray of shape (n_frames,)
        Head direction at each time point. **Allocentric (world-frame)
        convention**: 0 = East, π/2 = North, π = West, -π/2 = South,
        wrapped to ``[-π, π]`` (or ``[0, 360°]`` when
        ``angle_unit="deg"``). Units determined by ``angle_unit``.

        **Movement heading vs. head direction.** This classifier expects the
        animal's *head direction* (where the head points). A velocity-derived
        heading (e.g. from
        :func:`neurospatial.ops.egocentric.heading_from_velocity`) is the
        direction of *movement*, which equals head direction only when the
        animal moves the way it faces. Calling a cell a "head direction cell"
        from movement heading is a common methodological mislabel — keep the
        two distinct.
    bin_size : float, default=π/30 (6 degrees)
        Width of angular bins. Units match ``angle_unit``.
    bandwidth : float or None, default=None
        Gaussian smoothing bandwidth. Units match ``angle_unit``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of headings and bin_size.
    min_mvl : float, default=0.4
        Minimum mean vector length threshold.
    alpha : float, default=0.05
        Significance level for Rayleigh test.

    Returns
    -------
    bool
        True if neuron passes HD cell criteria.

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

    See Also
    --------
    compute_directional_rate : Full directional rate computation
    DirectionalRateResult.is_head_direction_cell : HD cell classification on result object
    """
    from neurospatial.encoding._validation import validate_classifier_trajectory

    # Validate inputs OUTSIDE the try so genuine input errors propagate
    # (a typo such as angle_unit="degrees" must surface as a ValueError,
    # not be swallowed by the except below into a False classification).
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")
    validate_classifier_trajectory(
        spike_times, times, headings, context="is_head_direction_cell"
    )

    try:
        result = compute_directional_rate(
            spike_times,
            times,
            headings,
            bin_size=bin_size,
            bandwidth=bandwidth,
            angle_unit=angle_unit,
        )
    except (ValueError, RuntimeError):
        # Computation passed validation but produced no usable tuning
        # (e.g. no spikes in any visited bin) -> not an HD cell.
        return False
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
    or linear projection. Polar plots show 0° at the top (North) with
    clockwise direction following neuroscience convention.

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

    - 0° at top (North): Uses ``theta_zero_location='N'``
    - Clockwise direction: Uses ``theta_direction=-1``
    - Curve is closed (first point appended at end)

    These conventions match standard neuroscience visualization where
    0° = facing forward/north, 90° = facing right/east.

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

        # Configure polar plot: 0° at top (North), clockwise direction
        polar_ax.set_theta_zero_location("N")
        polar_ax.set_theta_direction(-1)

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
        x_closed[-1] = 360.0  # Ensure last point is at 360
        ax.set_xlabel("Head Direction (deg)")
        ax.set_xlim(0, 360)

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
