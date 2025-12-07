"""
Phase precession analysis for place cells.

This module provides tools for detecting and analyzing phase precession,
the phenomenon where place cell spikes occur at progressively earlier
theta phases as an animal moves through a place field.

Which Function Should I Use?
----------------------------
**Screening many neurons (100s-1000s)?**
    Use ``has_phase_precession()`` for fast boolean filtering.

**Need phase precession slope for publication?**
    Use ``phase_precession()`` for full analysis with slope, offset, and fit quality.
    This is what you want for figures and reporting.

**Visualizing phase precession?**
    Use ``plot_phase_precession()`` for standard doubled-axis visualization.

Common Use Cases
----------------
**Screening many neurons (fast filtering)**:

>>> # Fast boolean check - use for initial filtering
>>> if has_phase_precession(positions, phases):  # doctest: +SKIP
...     print("Neuron shows phase precession")

**Publication-quality analysis (full metrics)**:

Use for detailed analysis and publication::

    from neurospatial.encoding.phase_precession import (
        phase_precession,
        plot_phase_precession,
    )

    result = phase_precession(positions, phases)
    print(result)  # Automatic interpretation with slope, correlation, fit
    plot_phase_precession(positions, phases, result)

References
----------
O'Keefe, J. & Recce, M.L. (1993). Phase relationship between hippocampal
    place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
Kempter, R. et al. (2012). Quantifying circular-linear associations.
    J Neurosci Methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.stats.circular import (
    _mean_resultant_length,
    _to_radians,
    _validate_paired_input,
    circular_linear_correlation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "PhasePrecessionResult",
    "has_phase_precession",
    "phase_precession",
    "plot_phase_precession",
]


@dataclass
class PhasePrecessionResult:
    """
    Results from phase precession analysis.

    Attributes
    ----------
    slope : float
        Slope of phase-position relationship (radians per position unit).
    slope_units : str
        Units of slope (e.g., "rad/cm").
    offset : float
        Phase offset at position 0 (radians).
    correlation : float
        Circular-linear correlation coefficient in [0, 1].
    pval : float
        P-value for the correlation.
    mean_resultant_length : float
        Mean resultant length of phase residuals in [0, 1].
        Higher values indicate better linear fit.
    """

    slope: float
    slope_units: str
    offset: float
    correlation: float
    pval: float
    mean_resultant_length: float

    def is_significant(self, alpha: float = 0.05) -> bool:
        """
        Check if phase precession is statistically significant.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        bool
            True if p-value is below alpha.
        """
        return self.pval < alpha

    def interpretation(self) -> str:
        """
        Generate human-readable interpretation of results.

        Returns
        -------
        str
            Multi-line interpretation of the phase precession analysis.
        """
        lines = []

        # Significance
        if self.is_significant():
            lines.append(f"SIGNIFICANT phase-position relationship (p={self.pval:.4f})")
        else:
            lines.append(
                f"No significant phase-position relationship (p={self.pval:.4f})"
            )

        # Direction
        if self.slope < 0:
            lines.append(
                f"Phase PRECESSION detected (slope={self.slope:.3f} {self.slope_units})"
            )
        elif self.slope > 0:
            lines.append(
                f"Phase RECESSION detected (slope={self.slope:.3f} {self.slope_units})"
            )
        else:
            lines.append("No phase shift with position")

        # Correlation strength
        if self.correlation > 0.5:
            strength = "strong"
        elif self.correlation > 0.3:
            strength = "moderate"
        elif self.correlation > 0.1:
            strength = "weak"
        else:
            strength = "very weak"
        lines.append(f"Correlation: {strength} (r={self.correlation:.3f})")

        # Fit quality
        if self.mean_resultant_length > 0.7:
            fit = "excellent"
        elif self.mean_resultant_length > 0.5:
            fit = "good"
        elif self.mean_resultant_length > 0.3:
            fit = "moderate"
        else:
            fit = "poor"
        lines.append(f"Fit quality: {fit} (R={self.mean_resultant_length:.3f})")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return interpretation when printed."""
        return self.interpretation()


def phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    *,
    slope_bounds: tuple[float, float] = (-2 * np.pi, 2 * np.pi),
    position_range: tuple[float, float] | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    min_spikes: int = 10,
) -> PhasePrecessionResult:
    """
    Analyze phase precession in place cell data.

    Fits a linear phase-position relationship using maximum likelihood
    estimation (maximizing mean resultant length of residuals).

    Parameters
    ----------
    positions : ndarray of shape (n_spikes,)
        Position at each spike in arbitrary position units.
    phases : ndarray of shape (n_spikes,)
        Spike phase relative to LFP theta in radians or degrees.
    slope_bounds : tuple of float, default=(-2*pi, 2*pi)
        Bounds for slope optimization (radians per position unit).
    position_range : tuple of (float, float), optional
        If provided as ``(pos_min, pos_max)``, positions are normalized to
        [0, 1] before fitting. This changes the slope units from
        ``rad/position_unit`` to ``rad/normalized_position``, where the slope
        represents phase change across the entire normalized field.
        **Use case**: When comparing precession across fields of different
        sizes, normalization makes slopes comparable.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.
    min_spikes : int, default=10
        Minimum number of spikes required for analysis.

    Returns
    -------
    PhasePrecessionResult
        Dataclass with slope, offset, correlation, p-value, and fit quality.

    Raises
    ------
    ValueError
        If insufficient spikes or invalid input.

    See Also
    --------
    has_phase_precession : Quick boolean check.
    plot_phase_precession : Visualize phase precession.

    Notes
    -----
    Phase precession (O'Keefe & Recce, 1993) is characterized by spikes
    occurring at progressively earlier theta phases as the animal moves
    through a place field. This manifests as a negative slope in the
    phase-position relationship.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.phase_precession import phase_precession
    >>> positions = np.linspace(0, 50, 100)  # 0-50 cm
    >>> phases = 2 * np.pi - positions * 0.1  # Negative slope
    >>> result = phase_precession(positions, phases)
    >>> print(result)  # doctest: +SKIP
    """
    from scipy.optimize import fminbound
    from scipy.stats import circmean

    # Convert to arrays and radians if needed
    positions = np.asarray(positions, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    phases = _to_radians(phases, angle_unit)

    # Validate paired inputs (handles length mismatch, NaN removal)
    positions, phases = _validate_paired_input(
        positions, phases, "positions", "phases", min_samples=min_spikes
    )

    # Wrap phases to [0, 2*pi] to ensure consistent residual calculation
    # during optimization. Input phases may be in any range after angle_unit
    # conversion, but optimization assumes [0, 2*pi] range.
    phases = phases % (2 * np.pi)

    # Handle position normalization
    # Note: If position_range is provided, the correlation is computed on
    # normalized positions [0, 1], not the original position values.
    slope_units = "rad/position_unit"
    if position_range is not None:
        pos_min, pos_max = position_range
        if pos_max <= pos_min:
            raise ValueError(
                f"position_range must have pos_max > pos_min. "
                f"Got pos_min={pos_min}, pos_max={pos_max}.\n"
                f"Fix: Ensure position_range=(min, max) where max > min."
            )
        positions = (positions - pos_min) / (pos_max - pos_min)
        slope_units = "rad/normalized_position (0-1)"

    # Define objective function: negative mean resultant length of residuals
    # We minimize this to find the slope that maximizes mean resultant length
    def _neg_mean_resultant_length(slope: float) -> float:
        residuals = (phases - slope * positions) % (2 * np.pi)
        return -_mean_resultant_length(residuals)

    # The objective function has multiple local minima due to circular nature
    # Use grid search to find a good starting region, then refine
    n_grid = 100
    grid_slopes = np.linspace(slope_bounds[0], slope_bounds[1], n_grid)
    grid_values = np.array([_neg_mean_resultant_length(s) for s in grid_slopes])

    # Find the best region from grid search
    best_idx = np.argmin(grid_values)

    # Define a narrow search window around the best grid point
    window_width = (slope_bounds[1] - slope_bounds[0]) / n_grid * 2
    local_bounds = (
        max(slope_bounds[0], grid_slopes[best_idx] - window_width),
        min(slope_bounds[1], grid_slopes[best_idx] + window_width),
    )

    # Refine using bounded minimization in the best region
    optimal_slope, neg_mrl, _ierr, _numfunc = fminbound(
        _neg_mean_resultant_length,
        local_bounds[0],
        local_bounds[1],
        full_output=True,
    )

    # Compute mean resultant length at optimal slope
    mean_resultant_length = -neg_mrl

    # Compute residuals at optimal slope
    residuals = (phases - optimal_slope * positions) % (2 * np.pi)

    # Compute offset as circular mean of residuals
    offset = float(circmean(residuals, high=2 * np.pi, low=0))

    # Compute correlation using circular-linear correlation
    correlation, pval = circular_linear_correlation(phases, positions)

    return PhasePrecessionResult(
        slope=float(optimal_slope),
        slope_units=slope_units,
        offset=offset,
        correlation=correlation,
        pval=pval,
        mean_resultant_length=float(mean_resultant_length),
    )


def has_phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    min_correlation: float = 0.2,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> bool:
    """
    Quick check for significant phase precession.

    Parameters
    ----------
    positions : ndarray of shape (n_spikes,)
        Position at each spike.
    phases : ndarray of shape (n_spikes,)
        Spike phase relative to LFP theta in radians or degrees.
    alpha : float, default=0.05
        Significance level for correlation test.
    min_correlation : float, default=0.2
        Minimum correlation coefficient required.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.

    Returns
    -------
    bool
        True if significant phase precession detected (p < alpha,
        r >= min_correlation, and negative slope).

    See Also
    --------
    phase_precession : Full analysis with metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.phase_precession import has_phase_precession
    >>> # Random data - no precession expected
    >>> positions = np.random.default_rng(42).uniform(0, 100, 50)
    >>> phases = np.random.default_rng(42).uniform(0, 2 * np.pi, 50)
    >>> has_phase_precession(positions, phases)  # doctest: +SKIP
    False
    """
    try:
        result = phase_precession(positions, phases, angle_unit=angle_unit)
        return (
            result.pval < alpha
            and result.correlation >= min_correlation
            and result.slope < 0
        )
    except ValueError:
        return False


def plot_phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    result: PhasePrecessionResult | None = None,
    ax: Axes | None = None,
    *,
    position_label: str = "Position",
    show_fit: bool = True,
    marker_size: float = 20.0,
    marker_alpha: float = 0.6,
    show_doubled_note: bool = True,
    scatter_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
) -> Axes:
    """
    Plot phase precession with doubled phase axis.

    Creates standard phase precession visualization with phases plotted
    twice (0-2pi and 2pi-4pi) following O'Keefe & Recce convention.

    Parameters
    ----------
    positions : ndarray of shape (n_spikes,)
        Position at each spike.
    phases : ndarray of shape (n_spikes,)
        Spike phase relative to LFP theta (radians).
    result : PhasePrecessionResult, optional
        If provided, overlay fitted line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    position_label : str, default="Position"
        Label for x-axis.
    show_fit : bool, default=True
        If True and result provided, show fitted line.

        **Rationale for default=True**: Phase precession analysis is primarily
        about fitting the phase-position relationship. Showing the fit line
        is essential for visualizing the slope that defines precession.
    marker_size : float, default=20.0
        Size of scatter markers.
    marker_alpha : float, default=0.6
        Alpha of scatter markers.
    show_doubled_note : bool, default=True
        If True, add annotation explaining doubled phase axis.

        **Rationale for default=True**: The doubled-axis convention (O'Keefe &
        Recce, 1993) may be unfamiliar to new users. The annotation helps
        prevent confusion about why each spike appears twice.
    scatter_kwargs : dict, optional
        Additional keyword arguments for scatter plot.
    line_kwargs : dict, optional
        Additional keyword arguments for fitted line.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Notes
    -----
    Following O'Keefe & Recce (1993), phases are plotted twice: once in
    the [0, 2π] range and again in the [2π, 4π] range. This doubled
    representation makes it easier to see the phase precession relationship
    without wrapping artifacts.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.phase_precession import (
    ...     phase_precession,
    ...     plot_phase_precession,
    ... )
    >>> positions = np.linspace(0, 50, 100)
    >>> phases = (2 * np.pi - positions * 0.1) % (2 * np.pi)
    >>> result = phase_precession(positions, phases)
    >>> ax = plot_phase_precession(positions, phases, result)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    # Convert to arrays
    positions = np.asarray(positions, dtype=np.float64).ravel()
    phases = np.asarray(phases, dtype=np.float64).ravel()

    # Validate matching lengths
    if len(positions) != len(phases):
        raise ValueError(
            f"positions and phases must have the same length. "
            f"Got positions: {len(positions)}, phases: {len(phases)}."
        )

    # Wrap phases to [0, 2pi]
    phases = phases % (2 * np.pi)

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Default kwargs
    scatter_defaults: dict = {"c": "C0", "zorder": 2}
    line_defaults: dict = {"color": "red", "linewidth": 2, "zorder": 3}

    # Merge with user kwargs
    scatter_kw = {**scatter_defaults, **(scatter_kwargs or {})}
    line_kw = {**line_defaults, **(line_kwargs or {})}

    # Plot phases doubled (0-2pi and 2pi-4pi)
    # This is the O'Keefe & Recce convention for visualizing phase precession
    positions_doubled = np.concatenate([positions, positions])
    phases_doubled = np.concatenate([phases, phases + 2 * np.pi])

    ax.scatter(
        positions_doubled,
        phases_doubled,
        s=marker_size,
        alpha=marker_alpha,
        **scatter_kw,
    )

    # Plot fitted lines if result provided and show_fit is True
    if result is not None and show_fit:
        # Create fitted line
        pos_range = np.linspace(positions.min(), positions.max(), 100)
        fitted_phases = (result.offset + result.slope * pos_range) % (2 * np.pi)

        # Plot in both the lower [0, 2pi] and upper [2pi, 4pi] regions
        ax.plot(pos_range, fitted_phases, **line_kw)
        ax.plot(pos_range, fitted_phases + 2 * np.pi, **line_kw)

    # Set y-axis with pi labels
    y_ticks = [0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi]
    y_labels = ["0", "π", "2π", "3π", "4π"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(0, 4 * np.pi)

    # Labels
    ax.set_xlabel(position_label)
    ax.set_ylabel("Phase (radians)")

    # Add annotation explaining doubled phase axis
    if show_doubled_note:
        ax.text(
            0.02,
            0.98,
            "Note: Each point appears twice\n(same data at phase and phase+2π)",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    return ax
