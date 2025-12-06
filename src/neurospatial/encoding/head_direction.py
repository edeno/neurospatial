"""
Head direction cell analysis module.

This module provides tools for analyzing head direction (HD) cells, neurons
that fire preferentially when an animal faces a particular direction.
HD cells are found in various brain regions including the postsubiculum,
anterodorsal thalamus, and lateral mammillary nucleus.

Which Function Should I Use?
----------------------------
**Computing tuning curve from raw data?**
    Use ``head_direction_tuning_curve()`` to compute firing rate as a function
    of head direction from spike times and head direction time series.

**Analyzing tuning curve properties?**
    Use ``head_direction_metrics()`` to compute preferred direction, mean
    vector length, tuning width, and HD cell classification.

**Screening many neurons (100s-1000s)?**
    Use ``is_head_direction_cell()`` for fast boolean filtering without
    manually computing tuning curves and metrics.

**Visualizing HD tuning?**
    Use ``plot_head_direction_tuning()`` for standard polar or linear plots.

Typical Workflow
----------------
1. Compute tuning curve from spike times and head directions::

    >>> bin_centers, firing_rates = head_direction_tuning_curve(  # doctest: +SKIP
    ...     head_directions, spike_times, position_times,
    ...     bin_size=6.0, angle_unit='deg'
    ... )

2. Compute metrics and classify::

    >>> metrics = head_direction_metrics(bin_centers, firing_rates)  # doctest: +SKIP
    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if metrics.is_hd_cell:  # doctest: +SKIP
    ...     print(f"Preferred direction: {metrics.preferred_direction_deg:.1f}°")

3. Visualize::

    >>> plot_head_direction_tuning(bin_centers, firing_rates, metrics)  # doctest: +SKIP

Common Parameters
-----------------
Most functions accept ``angle_unit`` parameter: ``'rad'`` (default) or ``'deg'``.
HD research commonly uses degrees, but we default to radians for scipy
compatibility. Use ``angle_unit='deg'`` if your data is in degrees.

References
----------
Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells recorded
    from the postsubiculum in freely moving rats. I. Description and
    quantitative analysis. Journal of Neuroscience, 10(2), 420-435.
Sargolini, F. et al. (2006). Conjunctive representation of position, direction,
    and velocity in entorhinal cortex. Science, 312(5774), 758-762.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from neurospatial.stats.circular import (
    _mean_resultant_length,
    _to_radians,
    _validate_tuning_data,
    # Re-export for convenience in HD workflow
    circular_mean,
    mean_resultant_length,
    rayleigh_test,
)

__all__: list[str] = [
    "HeadDirectionMetrics",
    "circular_mean",
    "head_direction_metrics",
    "head_direction_tuning_curve",
    "is_head_direction_cell",
    "mean_resultant_length",
    "plot_head_direction_tuning",
    "rayleigh_test",
]


@dataclass
class HeadDirectionMetrics:
    """
    Metrics for head direction cell analysis.

    Attributes
    ----------
    preferred_direction : float
        Peak direction in radians [0, 2pi].
    preferred_direction_deg : float
        Peak direction in degrees [0, 360].
    mean_vector_length : float
        Rayleigh vector length (0-1). Higher values indicate sharper tuning.
        Typical HD cells have values > 0.4.
    peak_firing_rate : float
        Maximum firing rate (Hz).
    tuning_width : float
        Approximate half-width at half-maximum (HWHM) in radians.
        Computed from bin counts, so accuracy depends on bin_size.
    tuning_width_deg : float
        Approximate HWHM in degrees.
    is_hd_cell : bool
        True if passes HD cell criteria.
    rayleigh_pval : float
        P-value from Rayleigh test.
    min_vector_length_threshold : float
        The threshold used for HD cell classification.

    Notes
    -----
    **Classification Criteria**:

    A neuron is classified as an HD cell if:

    - Mean vector length > min_vector_length (default 0.4)
    - Rayleigh test p-value < 0.05

    These criteria follow Taube et al. (1990) and subsequent literature.

    Examples
    --------
    >>> metrics = head_direction_metrics(bins, rates)  # doctest: +SKIP
    >>> if metrics.is_hd_cell:  # doctest: +SKIP
    ...     print(
    ...         f"HD cell! Preferred direction: {metrics.preferred_direction_deg:.1f} deg"
    ...     )
    ...     print(f"Tuning width: {metrics.tuning_width_deg:.1f} deg")
    """

    preferred_direction: float
    preferred_direction_deg: float
    mean_vector_length: float
    peak_firing_rate: float
    tuning_width: float
    tuning_width_deg: float
    is_hd_cell: bool
    rayleigh_pval: float
    min_vector_length_threshold: float = 0.4

    def interpretation(self) -> str:
        """
        Human-readable interpretation of head direction metrics.

        Returns
        -------
        str
            Multi-line interpretation.
        """
        lines = []
        threshold = self.min_vector_length_threshold

        if self.is_hd_cell:
            lines.append("*** HEAD DIRECTION CELL ***")
            lines.append(f"Preferred direction: {self.preferred_direction_deg:.1f} deg")
            lines.append(
                f"Mean vector length: {self.mean_vector_length:.3f} "
                f"(threshold = {threshold})"
            )
            lines.append(f"Peak firing rate: {self.peak_firing_rate:.1f} Hz")
            lines.append(f"Tuning width (HWHM): {self.tuning_width_deg:.1f} deg")
            lines.append(f"Rayleigh test: p = {self.rayleigh_pval:.4f}")
        else:
            lines.append("Not classified as HD cell")
            if self.mean_vector_length < threshold:
                lines.append(
                    f"  - Mean vector length too low: "
                    f"{self.mean_vector_length:.3f} < {threshold}"
                )
                lines.append(
                    f"    How was {threshold} chosen? Default 0.4 is from "
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
            if self.rayleigh_pval >= 0.05:
                lines.append(
                    f"  - Rayleigh test not significant: p = {self.rayleigh_pval:.3f} >= 0.05"
                )

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation with interpretation."""
        return self.interpretation()


def head_direction_tuning_curve(
    head_directions: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    position_times: NDArray[np.float64],
    *,
    bin_size: float = 6.0,
    angle_unit: Literal["rad", "deg"] = "rad",
    smoothing_window: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute head direction tuning curve from spike times and head direction data.

    This function computes firing rate as a function of head direction by:
    1. Binning head directions into angular bins
    2. Computing occupancy (time spent in each bin) using actual time deltas
    3. Counting spikes in each bin by interpolating head direction at spike times
    4. Computing firing rate = spike count / occupancy
    5. Optionally applying Gaussian smoothing with circular boundary handling

    Parameters
    ----------
    head_directions : ndarray of shape (n_frames,)
        Head direction at each time point. Units determined by ``angle_unit``.
    spike_times : ndarray of shape (n_spikes,)
        Times of spikes in the same units as ``position_times``.
    position_times : ndarray of shape (n_frames,)
        Timestamps corresponding to each head direction sample.
        Must be monotonically increasing.
    bin_size : float, default=6.0
        Width of angular bins. Units match ``angle_unit`` (radians by default).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of ``head_directions`` and ``bin_size``.
        Note: HD research commonly uses degrees. If your data is in degrees,
        use ``angle_unit='deg'``.
    smoothing_window : int, default=5
        Standard deviation of Gaussian smoothing kernel in bins.
        Set to 0 to disable smoothing.

    Returns
    -------
    bin_centers : ndarray of shape (n_bins,)
        Center of each angular bin in radians.
    firing_rates : ndarray of shape (n_bins,)
        Firing rate (Hz) in each bin.

    Raises
    ------
    ValueError
        If head_directions and position_times have different lengths.
        If position_times are not monotonically increasing.
        If fewer than 3 samples provided.

    See Also
    --------
    head_direction_metrics : Compute tuning curve properties.
    is_head_direction_cell : Fast screening for HD cells.

    Notes
    -----
    **Occupancy calculation**: Uses actual time deltas between frames
    (``np.diff(position_times)``) rather than assuming uniform sampling.
    This correctly handles dropped frames and variable sampling rates.
    The last frame is excluded from occupancy since we don't know how
    long the animal stayed at that position.

    **Spike assignment**: Spikes are assigned to bins using nearest-neighbor
    lookup (not interpolation) to correctly handle circular discontinuities.
    Linear interpolation would give wrong results when head direction crosses
    the 0°/360° boundary (e.g., 350° to 10° would incorrectly interpolate to
    180°). Spikes outside the recording window are excluded.

    **Bin edge handling**: Head directions are assigned using ``np.digitize``,
    which assigns values in range ``[bin_edges[i], bin_edges[i+1])`` to bin ``i``.
    Values exactly at 2π are wrapped to bin 0.

    **Circular smoothing**: Gaussian smoothing uses ``mode='wrap'`` to
    correctly handle the circular boundary (0° neighbors 360°).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import head_direction_tuning_curve
    >>> # Create sample data: 10 seconds at 30 Hz
    >>> position_times = np.linspace(0, 10, 300)
    >>> head_directions = np.random.default_rng(42).uniform(0, 360, 300)
    >>> spike_times = np.random.default_rng(42).uniform(0, 10, 50)
    >>> bin_centers, firing_rates = head_direction_tuning_curve(
    ...     head_directions,
    ...     spike_times,
    ...     position_times,
    ...     bin_size=30.0,
    ...     angle_unit="deg",
    ... )
    >>> len(bin_centers)  # 360 / 30 = 12 bins
    12
    >>> firing_rates.shape
    (12,)
    """
    # Convert inputs to arrays
    head_directions = np.asarray(head_directions, dtype=np.float64).ravel()
    spike_times = np.asarray(spike_times, dtype=np.float64).ravel()
    position_times = np.asarray(position_times, dtype=np.float64).ravel()

    # Validate inputs
    if len(head_directions) != len(position_times):
        raise ValueError(
            f"head_directions and position_times must have the same length. "
            f"Got head_directions: {len(head_directions)}, "
            f"position_times: {len(position_times)}.\n"
            f"Fix: Ensure both arrays represent the same time series."
        )

    if len(position_times) < 3:
        raise ValueError(
            f"Need at least 3 samples to compute tuning curve. "
            f"Got {len(position_times)} samples.\n"
            f"Fix: Provide more data points."
        )

    # Check strict monotonicity (no duplicates, no decreasing)
    time_diffs = np.diff(position_times)
    if np.any(time_diffs <= 0):
        n_problems = np.sum(time_diffs <= 0)
        raise ValueError(
            f"position_times must be strictly increasing (no duplicates). "
            f"Found {n_problems} non-increasing time steps.\n"
            f"Fix: Remove duplicate timestamps or check for timestamp errors."
        )

    # Convert to radians if needed
    head_directions_rad = _to_radians(head_directions, angle_unit)
    bin_size_rad = np.radians(bin_size) if angle_unit == "deg" else bin_size

    # Compute bin edges and centers
    n_bins = int(np.round(2 * np.pi / bin_size_rad))
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Wrap head directions to [0, 2*pi)
    head_directions_wrapped = head_directions_rad % (2 * np.pi)

    # Compute occupancy using actual time deltas
    # Each frame i contributes the time until frame i+1
    # The last frame is excluded (we don't know how long the animal stayed there)
    time_deltas = np.diff(position_times)

    # Assign each frame (except last) to a bin
    # head_directions_wrapped[:-1] has n-1 elements, matching time_deltas
    frame_bins = np.digitize(head_directions_wrapped[:-1], bin_edges) - 1
    # Handle edge case: value exactly at 2*pi goes to bin n_bins, wrap to 0
    frame_bins[frame_bins >= n_bins] = 0

    # Compute occupancy per bin using vectorized bincount (much faster than loop)
    occupancy = np.bincount(frame_bins, weights=time_deltas, minlength=n_bins).astype(
        np.float64
    )

    # Count spikes per bin
    spike_counts = np.zeros(n_bins, dtype=np.float64)

    if len(spike_times) > 0:
        # Filter spikes to valid time range
        valid_mask = (spike_times >= position_times[0]) & (
            spike_times <= position_times[-1]
        )
        valid_spike_times = spike_times[valid_mask]

        if len(valid_spike_times) > 0:
            # Use nearest-neighbor assignment to avoid circular interpolation issues
            # np.interp would give wrong results when head direction crosses 0/2pi
            # (e.g., 350° to 10° would interpolate to 180° instead of ~0°)
            spike_indices = (
                np.searchsorted(position_times, valid_spike_times, side="right") - 1
            )
            spike_indices = np.clip(spike_indices, 0, len(head_directions_wrapped) - 1)
            spike_hd = head_directions_wrapped[spike_indices]

            # Assign spikes to bins
            spike_bins = np.digitize(spike_hd, bin_edges) - 1
            spike_bins[spike_bins >= n_bins] = 0

            # Count spikes per bin using vectorized bincount
            spike_counts = np.bincount(spike_bins, minlength=n_bins).astype(np.float64)

    # Compute firing rates (Hz) = spike count / occupancy
    # Handle division by zero (bins with no occupancy get zero rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(occupancy > 0, spike_counts / occupancy, 0.0)

    # Apply Gaussian smoothing with circular boundary
    if smoothing_window > 0:
        firing_rates = gaussian_filter1d(firing_rates, smoothing_window, mode="wrap")

    return bin_centers, firing_rates


def head_direction_metrics(
    bin_centers: NDArray[np.float64],
    firing_rates: NDArray[np.float64],
    *,
    min_vector_length: float = 0.4,
) -> HeadDirectionMetrics:
    """
    Compute head direction cell metrics from tuning curve.

    Parameters
    ----------
    bin_centers : ndarray of shape (n_bins,)
        Center of each angular bin (radians).
    firing_rates : ndarray of shape (n_bins,)
        Firing rate in each bin (Hz).
    min_vector_length : float, default=0.4
        Minimum Rayleigh vector length to classify as HD cell.

        **How was 0.4 chosen?**

        This threshold comes from Taube et al. (1990) analyzing postsubicular
        HD cells in rats. Empirically:

        - Classic HD cells: 0.5-0.8
        - Borderline HD cells: 0.3-0.5
        - Non-HD cells: 0.1-0.3

        **When to adjust**:

        - Other brain regions: May need 0.3-0.5
        - Different species: Validate threshold first
        - Noisy recordings: Consider 0.3 (more permissive)
        - Publication quality: Use 0.5 (more conservative)

    Returns
    -------
    HeadDirectionMetrics
        Dataclass with preferred_direction, mean_vector_length,
        peak_firing_rate, tuning_width, is_hd_cell, rayleigh_pval.

    Raises
    ------
    ValueError
        If bin_centers and firing_rates have different lengths.
        If all firing rates are zero.
        If all firing rates are constant.

    See Also
    --------
    head_direction_tuning_curve : Compute tuning curve.
    is_head_direction_cell : Quick boolean check.

    Notes
    -----
    **Mean Vector Length** (Rayleigh vector):

        R = |sum(rate_i * exp(i*theta_i))| / sum(rate_i)

    **Preferred Direction**:

        PFD = arg(sum(rate_i * exp(i*theta_i)))

    **Tuning Width**: Approximate half-width at half-maximum (HWHM),
    computed by counting bins above half-maximum. For more accurate
    measurement, use smaller bin_size or fit a parametric model.

    Examples
    --------
    >>> from neurospatial.metrics import (
    ...     head_direction_tuning_curve,
    ...     head_direction_metrics,
    ... )
    >>> bins, rates = head_direction_tuning_curve(hd, spikes, times)  # doctest: +SKIP
    >>> metrics = head_direction_metrics(bins, rates)  # doctest: +SKIP
    >>> print(metrics)  # doctest: +SKIP

    References
    ----------
    Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells.
        J Neurosci, 10(2), 420-435.
    """
    # Validate input data (checks length match, non-empty, non-negative,
    # all-zero, and constant firing rates)
    bin_centers, firing_rates = _validate_tuning_data(
        bin_centers, firing_rates, require_variation=True
    )

    # Compute mean resultant length using centralized helper (uses scipy if available)
    mean_vector_length = _mean_resultant_length(bin_centers, weights=firing_rates)

    # Compute preferred direction (circular mean weighted by firing rate)
    total_rate = np.sum(firing_rates)
    weights = firing_rates / total_rate
    mean_cos = np.sum(weights * np.cos(bin_centers))
    mean_sin = np.sum(weights * np.sin(bin_centers))
    preferred_direction = np.arctan2(mean_sin, mean_cos) % (2 * np.pi)

    # Peak firing rate
    peak_firing_rate = float(np.max(firing_rates))

    # Half-width at half-max (HWHM) approximation
    half_max = peak_firing_rate / 2
    above_half = firing_rates >= half_max

    # Count bins above half-max (with circular wrapping)
    # This is approximate; for exact HWHM, would need interpolation
    if np.any(above_half):
        # Find transitions
        extended = np.concatenate([above_half, above_half[:1]])
        transitions = np.diff(extended.astype(int))
        rises = np.where(transitions == 1)[0]
        falls = np.where(transitions == -1)[0]

        if len(rises) > 0 and len(falls) > 0:
            # Calculate width in bins
            bin_width = (
                bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else np.pi / 30
            )
            n_above = np.sum(above_half)
            tuning_width = n_above * bin_width / 2  # HWHM = FWHM / 2
        else:
            tuning_width = np.pi / 2  # Default if can't compute
    else:
        tuning_width = np.nan

    # Rayleigh test on weighted angles
    _, pval = rayleigh_test(bin_centers, weights=firing_rates)

    # Classification
    is_hd_cell = (mean_vector_length > min_vector_length) and (pval < 0.05)

    return HeadDirectionMetrics(
        preferred_direction=float(preferred_direction),
        preferred_direction_deg=float(np.degrees(preferred_direction)),
        mean_vector_length=float(mean_vector_length),
        peak_firing_rate=float(peak_firing_rate),
        tuning_width=float(tuning_width),
        tuning_width_deg=float(np.degrees(tuning_width)),
        is_hd_cell=is_hd_cell,
        rayleigh_pval=float(pval),
        min_vector_length_threshold=min_vector_length,
    )


def is_head_direction_cell(
    head_directions: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    position_times: NDArray[np.float64],
    **kwargs: Any,
) -> bool:
    """
    Quick check: Is this a head direction cell?

    Convenience function for fast screening.
    For detailed metrics, use ``head_direction_tuning_curve`` + ``head_direction_metrics``.

    Parameters
    ----------
    head_directions : ndarray of shape (n_frames,)
        Head direction at each time point.
    spike_times : ndarray of shape (n_spikes,)
        Times of spikes (same time units as position_times).
    position_times : ndarray of shape (n_frames,)
        Timestamps corresponding to each head direction sample.
    **kwargs : dict
        Additional arguments passed to ``head_direction_tuning_curve``.

    Returns
    -------
    bool
        True if neuron passes HD cell criteria.

    Examples
    --------
    >>> from neurospatial.metrics import is_head_direction_cell
    >>> # Screen many neurons
    >>> for i, (hd, spikes, times) in enumerate(all_neurons):  # doctest: +SKIP
    ...     if is_head_direction_cell(hd, spikes, times):
    ...         print(f"Neuron {i} is an HD cell")
    """
    try:
        bins, rates = head_direction_tuning_curve(
            head_directions, spike_times, position_times, **kwargs
        )
        metrics = head_direction_metrics(bins, rates)
        return metrics.is_hd_cell
    except ValueError:
        return False


def plot_head_direction_tuning(
    bin_centers: NDArray[np.float64],
    firing_rates: NDArray[np.float64],
    metrics: HeadDirectionMetrics | None = None,
    ax: Axes | PolarAxes | None = None,
    *,
    projection: Literal["polar", "linear"] = "polar",
    angle_unit: Literal["deg", "rad"] = "rad",
    show_metrics: bool = True,
    color: str = "C0",
    fill_alpha: float = 0.3,
    line_kwargs: dict[str, Any] | None = None,
    fill_kwargs: dict[str, Any] | None = None,
) -> Axes | PolarAxes:
    """
    Plot head direction tuning curve.

    Creates standard head direction tuning visualization with optional polar
    or linear projection. Polar plots show 0° at the top (North) with
    clockwise direction following neuroscience convention.

    Parameters
    ----------
    bin_centers : ndarray of shape (n_bins,)
        Center of each angular bin (radians).
    firing_rates : ndarray of shape (n_bins,)
        Firing rate in each bin (Hz).
    metrics : HeadDirectionMetrics, optional
        If provided, mark preferred direction and optionally show metrics.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with appropriate projection.
    projection : {'polar', 'linear'}, default='polar'
        Plot projection type.
    angle_unit : {'deg', 'rad'}, default='rad'
        Unit for angle labels on axes. Use ``'deg'`` if you prefer degrees.
    show_metrics : bool, default=True
        If True and metrics provided, show metrics text box.

        **Rationale for default=True**: Head direction tuning curves are typically
        shown with key metrics (preferred direction, vector length) overlaid to
        aid interpretation. This is the standard presentation in neuroscience
        publications (Taube et al., 1990).
    color : str, default='C0'
        Color for tuning curve line and fill.
    fill_alpha : float, default=0.3
        Alpha (transparency) for filled area under curve.
    line_kwargs : dict, optional
        Additional keyword arguments for line plot.
    fill_kwargs : dict, optional
        Additional keyword arguments for fill.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Raises
    ------
    ValueError
        If bin_centers and firing_rates have different lengths.

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
    >>> from neurospatial.metrics import (
    ...     head_direction_tuning_curve,
    ...     head_direction_metrics,
    ...     plot_head_direction_tuning,
    ... )
    >>> bins, rates = head_direction_tuning_curve(hd, spikes, times)  # doctest: +SKIP
    >>> metrics = head_direction_metrics(bins, rates)  # doctest: +SKIP
    >>> ax = plot_head_direction_tuning(bins, rates, metrics)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    # Validate input data (checks length match, non-empty, non-negative)
    # Note: require_variation=False because plotting can handle flat curves
    bin_centers, firing_rates = _validate_tuning_data(
        bin_centers, firing_rates, require_variation=False
    )

    # Create figure if needed
    if ax is None:
        if projection == "polar":
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            _, ax = plt.subplots()

    # Default kwargs
    line_defaults: dict[str, Any] = {"color": color, "linewidth": 2, "zorder": 2}
    fill_defaults: dict[str, Any] = {"color": color, "alpha": fill_alpha, "zorder": 1}

    # Merge with user kwargs
    line_kw = {**line_defaults, **(line_kwargs or {})}
    fill_kw = {**fill_defaults, **(fill_kwargs or {})}

    # Close the curve (append first point at end)
    angles_closed = np.concatenate([bin_centers, [bin_centers[0] + 2 * np.pi]])
    rates_closed = np.concatenate([firing_rates, [firing_rates[0]]])

    if projection == "polar":
        # Import PolarAxes at runtime for cast

        polar_ax = cast("PolarAxes", ax)

        # Configure polar plot: 0° at top (North), clockwise direction
        polar_ax.set_theta_zero_location("N")
        polar_ax.set_theta_direction(-1)

        # Plot tuning curve
        polar_ax.plot(angles_closed, rates_closed, **line_kw)

        # Fill under curve
        polar_ax.fill(angles_closed, rates_closed, **fill_kw)

        # Set angle labels
        if angle_unit == "deg":
            polar_ax.set_thetagrids(
                [0, 45, 90, 135, 180, 225, 270, 315],
                ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
            )
        else:
            # Radians with pi notation
            polar_ax.set_thetagrids(
                [0, 45, 90, 135, 180, 225, 270, 315],
                ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4"],
            )

    else:
        # Linear projection
        # Convert to display unit for x-axis
        if angle_unit == "deg":
            x_closed = np.degrees(angles_closed)
            x_closed[-1] = 360.0  # Ensure last point is at 360
            ax.set_xlabel("Head Direction (deg)")
            ax.set_xlim(0, 360)
        else:
            x_closed = angles_closed
            ax.set_xlabel("Head Direction (rad)")
            ax.set_xlim(0, 2 * np.pi)

        # Plot tuning curve
        ax.plot(x_closed, rates_closed, **line_kw)

        # Fill under curve
        ax.fill(x_closed, rates_closed, **fill_kw)

        ax.set_ylabel("Firing Rate (Hz)")

    # Mark preferred direction if metrics provided
    if metrics is not None:
        pfd = metrics.preferred_direction
        peak_rate = metrics.peak_firing_rate

        if projection == "polar":
            # Draw line from origin to preferred direction (use polar_ax)
            polar_ax.plot(
                [pfd, pfd],
                [0, peak_rate],
                color="red",
                linewidth=2,
                linestyle="--",
                zorder=3,
            )
        else:
            pfd_display = np.degrees(pfd) if angle_unit == "deg" else pfd
            ax.axvline(pfd_display, color="red", linewidth=2, linestyle="--", zorder=3)

        # Show metrics text box if requested
        if show_metrics:
            metrics_text = (
                f"PFD: {metrics.preferred_direction_deg:.1f}°\n"
                f"MVL: {metrics.mean_vector_length:.3f}\n"
                f"Peak: {metrics.peak_firing_rate:.1f} Hz"
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
