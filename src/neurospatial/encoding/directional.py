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

Compute Functions (to be implemented in Tasks 3.8-3.9)
------------------------------------------------------
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
...     smoothing_sigma=None,
... )

See Also
--------
neurospatial.encoding.head_direction : Legacy head direction analysis module
neurospatial.stats.circular : Circular statistics utilities
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

__all__ = [
    "DirectionalRateResult",
    "DirectionalRatesResult",
]


@dataclass(frozen=True)
class DirectionalRateResult:
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
    smoothing_sigma : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.

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
    smoothing_sigma : float or None
        Smoothing bandwidth in radians, or None.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **No Environment dependency**: Unlike SpatialRateResult, directional results
    do not store an Environment. Head direction is a 1D circular variable that
    does not depend on spatial position. The bin_centers field provides the
    angular coordinates needed for analysis and plotting.

    Convenience methods (plot, preferred_direction, mean_vector_length, etc.)
    are implemented in Tasks 3.2-3.4.

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
    ...     smoothing_sigma=None,
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
    smoothing_sigma: float | None

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
        ...     smoothing_sigma=None,
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
        ...     smoothing_sigma=None,
        ... )
        >>> pref = result.preferred_direction()
        >>> np.abs(pref - np.pi / 2) < 0.1  # Close to 90 degrees
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

        return circular_mean(centers, weights=rates)

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
        ...     smoothing_sigma=None,
        ... )
        >>> pref_deg = result.preferred_direction_deg()
        >>> np.abs(pref_deg - 90) < 6  # Close to 90 degrees
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
        ...     smoothing_sigma=None,
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
        ...     smoothing_sigma=None,
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

        return mean_resultant_length(centers, weights=rates)

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
        ...     smoothing_sigma=None,
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
        peak_idx = np.nanargmax(rates)
        peak_rate = rates[peak_idx]
        half_max = peak_rate / 2.0

        # Check for flat tuning curve
        if np.nanmin(rates) >= half_max:
            # All rates are above half-max, can't compute HWHM
            return float(np.nan)

        n_bins = len(rates)

        # Search for half-max crossings on both sides
        # Use circular indexing

        # Search right (increasing index)
        right_width = 0.0
        for offset in range(1, n_bins // 2 + 1):
            idx = (peak_idx + offset) % n_bins
            if rates[idx] < half_max:
                # Interpolate to find exact crossing
                prev_idx = (peak_idx + offset - 1) % n_bins
                frac = (half_max - rates[prev_idx]) / (rates[idx] - rates[prev_idx])
                right_width = (offset - 1 + frac) * self.bin_size
                break
        else:
            # Never crossed half-max (shouldn't happen if we passed the check above)
            return float(np.nan)

        # Search left (decreasing index)
        left_width = 0.0
        for offset in range(1, n_bins // 2 + 1):
            idx = (peak_idx - offset) % n_bins
            if rates[idx] < half_max:
                # Interpolate to find exact crossing
                prev_idx = (peak_idx - offset + 1) % n_bins
                frac = (half_max - rates[prev_idx]) / (rates[idx] - rates[prev_idx])
                left_width = (offset - 1 + frac) * self.bin_size
                break
        else:
            # Never crossed half-max
            return float(np.nan)

        # Return average of left and right half-widths
        return float((left_width + right_width) / 2.0)

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
        ...     smoothing_sigma=None,
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
            indicate significant directional tuning.

        Notes
        -----
        The Rayleigh test uses the mean vector length (R) to compute a
        test statistic. For a sample weighted by firing rates:

            z = n_eff * R²

        where n_eff is the effective sample size (related to total spikes).

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
        ...     smoothing_sigma=None,
        ... )
        >>> pval = result.rayleigh_pvalue()
        >>> pval < 0.05  # Significant non-uniformity
        True

        See Also
        --------
        mean_vector_length : Tuning strength measure
        is_hd_cell : Classification combining MVL and p-value
        neurospatial.stats.circular.rayleigh_test : Underlying test function
        """
        from neurospatial.stats.circular import rayleigh_test

        rates = np.asarray(self.firing_rate, dtype=np.float64)
        centers = np.asarray(self.bin_centers, dtype=np.float64)

        # The rayleigh_test uses the bin centers as angles and firing rates as weights
        _, pval = rayleigh_test(centers, weights=rates)

        return pval

    def is_hd_cell(self, min_mvl: float = 0.4, alpha: float = 0.05) -> bool:
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
        ...     smoothing_sigma=None,
        ... )
        >>> result.is_hd_cell()  # Sharply tuned neuron should be classified as HD cell
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
            Same parameter as in :meth:`is_hd_cell`.

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
        ...     smoothing_sigma=None,
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
        is_hd_cell : Boolean classification method
        mean_vector_length : Tuning strength
        rayleigh_pvalue : Statistical significance
        """
        lines = []
        alpha = 0.05  # Fixed significance level for Rayleigh test

        mvl = self.mean_vector_length()
        pval = self.rayleigh_pvalue()
        is_hd = self.is_hd_cell(min_mvl=min_mvl, alpha=alpha)

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
class DirectionalRatesResult:
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
    smoothing_sigma : float or None
        Gaussian smoothing bandwidth in radians, or None if unsmoothed.

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
    smoothing_sigma : float or None
        Smoothing bandwidth in radians, or None.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `DirectionalRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    Batch methods (preferred_directions, mean_vector_lengths, etc.) are
    implemented in Tasks 3.5-3.6.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.directional import DirectionalRatesResult

    >>> # Create batch result (5 neurons)
    >>> n_neurons = 5
    >>> n_bins = 60
    >>> bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    >>> firing_rates = np.random.rand(n_neurons, n_bins) * 10
    >>> occupancy = np.ones(n_bins) * 0.5

    >>> result = DirectionalRatesResult(
    ...     firing_rates=firing_rates,
    ...     occupancy=occupancy,
    ...     bin_centers=bin_centers,
    ...     bin_size=np.pi / 30,
    ...     smoothing_sigma=None,
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
    >>> for i, r in enumerate(result):
    ...     peak_rate = np.max(r.firing_rate)
    ...     print(f"Neuron {i}: peak = {peak_rate:.2f} Hz")

    See Also
    --------
    DirectionalRateResult : Single-neuron version
    compute_directional_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    bin_centers: ArrayLike
    bin_size: float
    smoothing_sigma: float | None

    def __len__(self) -> int:
        """Return number of neurons.

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
            bin_centers, bin_size, and smoothing_sigma.

        Examples
        --------
        >>> single = result[0]
        >>> single.firing_rate.shape
        (n_bins,)
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        return DirectionalRateResult(
            firing_rate=rates[idx],
            occupancy=self.occupancy,
            bin_centers=self.bin_centers,
            bin_size=self.bin_size,
            smoothing_sigma=self.smoothing_sigma,
        )

    def __iter__(self) -> Iterator[DirectionalRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        DirectionalRateResult
            Result for each neuron in order.

        Examples
        --------
        >>> for result in results:
        ...     peak = np.max(result.firing_rate)
        ...     print(f"Peak rate: {peak:.2f} Hz")
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
        >>> pref_dirs = result.preferred_directions()
        >>> pref_dirs.shape
        (n_neurons,)

        See Also
        --------
        DirectionalRateResult.preferred_direction : Single-neuron method
        """
        from neurospatial.stats.circular import circular_mean

        rates: NDArray[np.float64] = np.asarray(self.firing_rates, dtype=np.float64)
        centers: NDArray[np.float64] = np.asarray(self.bin_centers, dtype=np.float64)

        n_neurons = len(self)
        pref_dirs = np.empty(n_neurons, dtype=np.float64)

        for i in range(n_neurons):
            pref_dirs[i] = circular_mean(centers, weights=rates[i])

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
        >>> mvls = result.mean_vector_lengths()
        >>> mvls.shape
        (n_neurons,)

        See Also
        --------
        DirectionalRateResult.mean_vector_length : Single-neuron method
        """
        from neurospatial.stats.circular import mean_resultant_length

        rates: NDArray[np.float64] = np.asarray(self.firing_rates, dtype=np.float64)
        centers: NDArray[np.float64] = np.asarray(self.bin_centers, dtype=np.float64)

        n_neurons = len(self)
        mvls = np.empty(n_neurons, dtype=np.float64)

        for i in range(n_neurons):
            mvls[i] = mean_resultant_length(centers, weights=rates[i])

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
        >>> widths = result.tuning_widths()
        >>> widths.shape
        (n_neurons,)

        See Also
        --------
        DirectionalRateResult.tuning_width : Single-neuron method
        """
        n_neurons = len(self)
        widths = np.empty(n_neurons, dtype=np.float64)

        for i in range(n_neurons):
            widths[i] = self[i].tuning_width()

        return widths

    def peak_firing_rates(self) -> NDArray[np.float64]:
        """Get peak firing rates for all neurons.

        Returns the maximum firing rate across all directional bins for each
        neuron. NaN values are ignored.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_neurons,) with peak firing rates in Hz.

        Examples
        --------
        >>> peaks = result.peak_firing_rates()
        >>> peaks.shape
        (n_neurons,)

        See Also
        --------
        DirectionalRateResult.peak_firing_rate : Single-neuron method
        """
        rates: NDArray[np.float64] = np.asarray(self.firing_rates, dtype=np.float64)
        result: NDArray[np.float64] = np.nanmax(rates, axis=1)
        return result

    def detect_hd_cells(
        self, min_mvl: float = 0.4, alpha: float = 0.05
    ) -> NDArray[np.bool_]:
        """Classify neurons as head direction cells.

        A neuron is classified as a head direction (HD) cell if it meets
        both criteria (Taube et al., 1990):

        1. Mean vector length (MVL) > min_mvl (default 0.4)
        2. Rayleigh test p-value < alpha (default 0.05)

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
        >>> is_hd = result.detect_hd_cells()
        >>> is_hd.shape
        (n_neurons,)
        >>> n_hd_cells = np.sum(is_hd)

        See Also
        --------
        DirectionalRateResult.is_hd_cell : Single-neuron method
        """
        n_neurons = len(self)
        is_hd = np.empty(n_neurons, dtype=np.bool_)

        for i in range(n_neurons):
            is_hd[i] = self[i].is_hd_cell(min_mvl=min_mvl, alpha=alpha)

        return is_hd

    def to_dataframe(
        self,
        neuron_ids: Sequence[str | int] | None = None,
    ) -> pd.DataFrame:
        """Export metrics to DataFrame for exploratory analysis.

        Computes all directional metrics and exports them to a pandas DataFrame
        for easy filtering, sorting, and analysis.

        Parameters
        ----------
        neuron_ids : sequence of str or int, optional
            Identifiers for each neuron. If None, uses integer indices
            (0, 1, 2, ..., n_neurons-1).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - neuron_id: identifier for each neuron
            - preferred_direction: preferred direction in radians [-π, π]
            - preferred_direction_deg: preferred direction in degrees [-180, 180]
            - mean_vector_length: mean vector length [0, 1]
            - tuning_width: tuning width (HWHM) in radians (0, π]
            - tuning_width_deg: tuning width (HWHM) in degrees (0, 180]
            - peak_rate: maximum firing rate (Hz)
            - is_hd_cell: whether classified as HD cell (using default thresholds)

        Raises
        ------
        ValueError
            If neuron_ids has a different length than the number of neurons.

        Notes
        -----
        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``preferred_directions()``, ``mean_vector_lengths()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_hd_cell"] == True]``
        - Sort: ``df.sort_values("mean_vector_length", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_rate")``

        Examples
        --------
        >>> result = DirectionalRatesResult(...)
        >>> df = result.to_dataframe()
        >>> print(df.head())
           neuron_id  preferred_direction  preferred_direction_deg  ...

        >>> # Filter for HD cells
        >>> hd_cells = df[df["is_hd_cell"]]
        >>> print(f"Found {len(hd_cells)} HD cells")

        >>> # Sort by mean vector length
        >>> top_cells = df.sort_values("mean_vector_length", ascending=False).head(10)

        >>> # Custom neuron identifiers
        >>> df = result.to_dataframe(neuron_ids=["unit_0", "unit_1", "unit_2"])

        See Also
        --------
        detect_hd_cells : HD cell classification
        preferred_directions : Batch preferred direction computation
        mean_vector_lengths : Batch mean vector length computation
        """
        import pandas as pd

        n_neurons = len(self)

        # Use integer indices if no neuron_ids provided
        if neuron_ids is None:
            neuron_ids_list: list[str | int] = list(range(n_neurons))
        else:
            neuron_ids_list = list(neuron_ids)
            if len(neuron_ids_list) != n_neurons:
                raise ValueError(
                    f"neuron_ids has {len(neuron_ids_list)} elements but "
                    f"result contains {n_neurons} neurons"
                )

        # Compute all metrics
        pref_dirs = self.preferred_directions()
        mvls = self.mean_vector_lengths()
        widths = self.tuning_widths()
        peaks = self.peak_firing_rates()
        is_hd = self.detect_hd_cells()

        # Build data dictionary
        data: dict[str, Any] = {
            "neuron_id": neuron_ids_list,
            "preferred_direction": pref_dirs,
            "preferred_direction_deg": np.degrees(pref_dirs),
            "mean_vector_length": mvls,
            "tuning_width": widths,
            "tuning_width_deg": np.degrees(widths),
            "peak_rate": peaks,
            "is_hd_cell": is_hd,
        }

        return pd.DataFrame(data)
