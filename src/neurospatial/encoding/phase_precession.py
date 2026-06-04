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
    "theta_phase",
]


def theta_phase(
    lfp: NDArray[np.float64],
    sampling_rate: float,
    *,
    band: tuple[float, float] = (6.0, 10.0),
) -> NDArray[np.float64]:
    """Extract instantaneous theta phase from a local field potential (LFP).

    Band-pass filters ``lfp`` to the theta band, then takes the phase of the
    Hilbert analytic signal. The returned phase is wrapped to ``[0, 2*pi)``
    radians, the convention that :func:`phase_precession` (and the rest of this
    module) consumes — so the output is drop-in for phase-precession analysis
    once you select the phases at spike times.

    A zero-phase (forward-backward) Butterworth filter is used so the phase is
    not time-shifted relative to the input.

    Parameters
    ----------
    lfp : ndarray of shape (n_samples,)
        Local field potential trace (a voltage time series), one channel.
        Assumed uniformly sampled at ``sampling_rate``.
    sampling_rate : float
        Sampling rate of ``lfp`` in Hz.
    band : tuple of (float, float), default=(6.0, 10.0)
        ``(low, high)`` theta-band edges in Hz for the band-pass filter.

    Returns
    -------
    ndarray of shape (n_samples,)
        Instantaneous theta phase in radians, wrapped to ``[0, 2*pi)``, one
        value per input sample. Phase increases through the theta cycle.

    Raises
    ------
    ValueError
        If ``lfp`` is not 1-D, ``lfp`` contains any non-finite value
        (NaN or Inf — a single one makes the zero-phase filter return an
        all-NaN trace), ``lfp`` is too short for the zero-phase filter
        (``len(lfp)`` must exceed the filter ``padlen``; see Notes),
        ``sampling_rate`` is not positive, ``band`` is not ``(low, high)``
        with ``0 < low < high``, or the high edge is not below the Nyquist
        frequency (``sampling_rate / 2``).

    See Also
    --------
    phase_precession : Consumes spike phases (in radians) from this function.
    has_phase_precession : Quick boolean precession screen.

    Notes
    -----
    Only :mod:`scipy` is used (``scipy.signal``); no new dependency is
    introduced. This function takes an LFP array the caller already has — it
    does not load or spike-sort data.

    To obtain the spike phases that :func:`phase_precession` expects, sample
    this per-sample phase at the spike times (e.g. by interpolating the
    *unwrapped* phase onto the spike times, then re-wrapping to ``[0, 2*pi)``).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.phase_precession import (
    ...     theta_phase,
    ...     phase_precession,
    ... )
    >>> # Synthesize a pure 8 Hz theta sinusoid sampled at 1 kHz.
    >>> sampling_rate = 1000.0
    >>> t = np.arange(0, 5, 1 / sampling_rate)
    >>> lfp = np.sin(2 * np.pi * 8.0 * t)
    >>> phase = theta_phase(lfp, sampling_rate, band=(6, 10))
    >>> phase.shape == lfp.shape
    True
    >>> bool(phase.min() >= 0 and phase.max() < 2 * np.pi)
    True
    >>> # Phases are drop-in for phase_precession (no reshaping):
    >>> positions = np.linspace(0, 50, phase.size)
    >>> result = phase_precession(positions, phase, rng=0)
    >>> isinstance(result.slope, float)
    True
    """
    from scipy.signal import butter, filtfilt, hilbert

    from neurospatial._validation import validate_finite

    lfp = np.asarray(lfp, dtype=np.float64)
    if lfp.ndim != 1:
        raise ValueError(
            f"lfp must be a 1-D array of shape (n_samples,), got shape {lfp.shape}.\n"
            f"Fix: pass a single LFP channel, e.g. lfp[:, channel]."
        )
    # Reject non-finite samples up front: a single NaN/Inf makes filtfilt
    # return an all-NaN trace, silently producing all-NaN phases. validate_finite
    # also coerces to float64.
    lfp = validate_finite(lfp, name="lfp")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            f"sampling_rate must be a positive number (Hz), got {sampling_rate}."
        )
    low, high = float(band[0]), float(band[1])
    if not (0 < low < high):
        raise ValueError(f"band must be (low, high) with 0 < low < high, got {band}.")
    nyquist = sampling_rate / 2.0
    if high >= nyquist:
        raise ValueError(
            f"band high edge ({high} Hz) must be below the Nyquist frequency "
            f"({nyquist} Hz = sampling_rate / 2). "
            f"Fix: lower the band or raise sampling_rate."
        )

    # Zero-phase band-pass so the extracted phase is not time-shifted.
    # butter(..., btype="bandpass") with no output="sos" returns
    # transfer-function (b, a) coefficients, NOT second-order sections.
    b, a = butter(N=4, Wn=(low / nyquist, high / nyquist), btype="bandpass")

    # filtfilt's default padding is padlen = 3 * max(len(a), len(b)); it
    # raises an opaque "padlen" error when len(lfp) <= padlen. Precheck so the
    # caller gets a domain message stating the minimum length instead.
    padlen = 3 * max(len(a), len(b))
    if len(lfp) <= padlen:
        raise ValueError(
            f"lfp is too short for the zero-phase theta filter: got "
            f"{len(lfp)} sample(s), but the 4th-order Butterworth band-pass "
            f"requires more than {padlen} samples (filtfilt padlen = "
            f"{padlen}).\n"
            f"Fix: pass a longer LFP segment (at least {padlen + 1} samples)."
        )

    filtered = filtfilt(b, a, lfp)

    analytic = hilbert(filtered)
    # np.angle returns (-pi, pi]; wrap to [0, 2*pi) for the consumer convention.
    phase: NDArray[np.float64] = np.asarray(
        np.angle(analytic) % (2 * np.pi), dtype=np.float64
    )
    return phase


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
        Circular-linear correlation coefficient in [0, 1]. A descriptive,
        slope-independent effect size (Mardia & Jupp); it does not determine
        ``pval``.
    pval : float
        Shuffle p-value at the fitted slope. Computed by permuting the
        phase-position pairing, re-fitting the slope on each shuffle, and
        comparing the mean resultant length of residuals to the observed fit.
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
        """Check if phase precession is statistically significant.

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
        """Generate human-readable interpretation of results.

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


def _best_residual_mrl(
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    slope_bounds: tuple[float, float],
) -> float:
    """Coarse-grid maximum of the residual mean resultant length over the slope.

    Evaluates a fixed coarse slope grid and returns the largest mean resultant
    length (MRL) of the phase residuals. This is the statistic used for both
    the observed value and each shuffle in the permutation null, so the
    comparison is unbiased. A fixed coarse grid (rather than the full adaptive
    grid + Brent refinement of :func:`_fit_slope`) bounds the per-shuffle cost:
    the null only needs a comparable statistic, not a precision slope.

    Parameters
    ----------
    phases : ndarray of shape (n_spikes,)
        Spike phases in radians, wrapped to ``[0, 2*pi)``.
    positions : ndarray of shape (n_spikes,)
        Position at each spike (already normalized if requested).
    slope_bounds : tuple of float
        ``(lo, hi)`` bounds for the slope search (radians per position unit).

    Returns
    -------
    float
        Maximum residual MRL over the coarse slope grid, in [0, 1].
    """

    def _neg_mean_resultant_length(slope: float) -> float:
        residuals = (phases - slope * positions) % (2 * np.pi)
        return -_mean_resultant_length(residuals)

    grid_slopes = np.linspace(slope_bounds[0], slope_bounds[1], 100)
    grid_values = np.array([_neg_mean_resultant_length(s) for s in grid_slopes])
    return float(-grid_values.min())


def _fit_slope(
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    slope_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Fit the precession slope and return ``(optimal_slope, mrl)``.

    Maximizes the mean resultant length of the phase residuals over the slope
    using a data-adaptive grid search followed by a bounded Brent refinement,
    and returns the fitted slope (needed for the offset and the reported
    ``slope``) alongside the maximized MRL.

    Parameters
    ----------
    phases : ndarray of shape (n_spikes,)
        Spike phases in radians, wrapped to ``[0, 2*pi)``.
    positions : ndarray of shape (n_spikes,)
        Position at each spike (already normalized if requested).
    slope_bounds : tuple of float
        ``(lo, hi)`` bounds for the slope search (radians per position unit).

    Returns
    -------
    optimal_slope : float
        Fitted slope (radians per position unit).
    mrl : float
        Maximized residual mean resultant length, in [0, 1].
    """
    from scipy.optimize import minimize_scalar

    def _neg_mean_resultant_length(slope: float) -> float:
        residuals = (phases - slope * positions) % (2 * np.pi)
        return -_mean_resultant_length(residuals)

    # The circular objective is multimodal: as a function of slope it has a
    # main lobe at the true slope surrounded by side-lobes, and the main lobe
    # gets *narrower* as the position span grows (its half-width in slope is
    # ~pi / position_span). A fixed coarse grid can therefore step right over
    # the main lobe and bracket a side-lobe minimum instead.
    #
    # Make the grid data-adaptive: sample finely enough that at least a few
    # points fall inside the main lobe regardless of position span, then
    # refine within the single grid cell bracketing the best grid point.
    span = float(slope_bounds[1] - slope_bounds[0])
    position_span = float(np.ptp(positions))
    samples_per_lobe = 4
    if position_span > 0:
        lobe_half_width = np.pi / position_span
        target_spacing = lobe_half_width / samples_per_lobe
        n_grid = int(np.ceil(span / target_spacing)) + 1
    else:
        n_grid = 100
    # Clamp to a sensible range: never coarser than the original 100-point
    # grid, never so dense the O(n_grid) sweep dominates runtime.
    n_grid = int(np.clip(n_grid, 100, 20000))

    grid_slopes = np.linspace(slope_bounds[0], slope_bounds[1], n_grid)
    grid_values = np.array([_neg_mean_resultant_length(s) for s in grid_slopes])

    # Bracket the best grid point by its immediate neighbors (one grid cell on
    # each side) so the refinement isolates a single lobe, then polish with a
    # bounded scalar minimizer (Brent within bounds).
    best_idx = int(np.argmin(grid_values))
    lo = grid_slopes[max(best_idx - 1, 0)]
    hi = grid_slopes[min(best_idx + 1, n_grid - 1)]

    result_opt = minimize_scalar(
        _neg_mean_resultant_length,
        bounds=(lo, hi),
        method="bounded",
    )
    optimal_slope = float(result_opt.x)
    neg_mrl = float(result_opt.fun)

    # Guard against the rare case where the refinement lands above the best
    # grid sample (e.g. a degenerate bracket): keep the grid optimum instead.
    if grid_values[best_idx] < neg_mrl:
        optimal_slope = float(grid_slopes[best_idx])
        neg_mrl = float(grid_values[best_idx])

    return optimal_slope, float(-neg_mrl)


def phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    *,
    slope_bounds: tuple[float, float] = (-2 * np.pi, 2 * np.pi),
    position_range: tuple[float, float] | None = None,
    angle_unit: Literal["rad", "deg"] = "rad",
    min_spikes: int = 10,
    n_shuffles: int = 1000,
    rng: int | np.random.Generator | None = None,
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
    n_shuffles : int, default=1000
        Number of permutation shuffles used to build the null distribution
        for the p-value. Each shuffle re-fits the slope on the shuffled
        phase-position pairing (see Notes). Larger values give finer
        p-value resolution at higher cost.
    rng : int, numpy.random.Generator, or None, optional
        Seed or generator for the permutation shuffles. Pass a fixed value
        for a deterministic ``pval``.

    Returns
    -------
    PhasePrecessionResult
        Dataclass with slope, offset, correlation, p-value, and fit quality.
        ``pval`` is a **shuffle p-value at the fitted slope** (see Notes);
        ``correlation`` is a slope-independent descriptive circular-linear
        effect size.

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

    **P-value.** ``pval`` is a permutation p-value whose statistic is the
    mean resultant length (MRL) of phase residuals at the fitted slope — the
    same quantity the fit maximizes. The null is built by permuting the
    phase-position pairing and **re-fitting the slope on each shuffle**, so
    the test asks whether the fitted precession is stronger than chance,
    rather than testing a slope-free circular-linear association. A +1
    smoothing is applied (Phipson & Smyth, 2010) so ``pval`` is never exactly
    zero.

    **Performance.** The null re-fits the slope per shuffle. To bound latency,
    the per-shuffle re-fit uses a fixed coarse slope grid (no Brent
    refinement); the observed fit keeps the full data-adaptive grid + Brent
    search. The cost scales with ``n_shuffles``; reduce it for faster
    screening.

    ``correlation`` is the circular-linear correlation (Mardia & Jupp),
    reported as a descriptive, slope-independent effect size — it does not
    determine ``pval``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.phase_precession import phase_precession
    >>> positions = np.linspace(0, 50, 100)  # 0-50 cm
    >>> phases = 2 * np.pi - positions * 0.1  # Negative slope
    >>> result = phase_precession(positions, phases, rng=0)
    >>> print(result)  # doctest: +SKIP
    """
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

    # Fit the slope that maximizes the mean resultant length (MRL) of the
    # phase residuals (full data-adaptive grid + Brent refinement).
    optimal_slope, mean_resultant_length = _fit_slope(phases, positions, slope_bounds)

    # Compute residuals at optimal slope
    residuals = (phases - optimal_slope * positions) % (2 * np.pi)

    # Compute offset as circular mean of residuals
    offset = float(circmean(residuals, high=2 * np.pi, low=0))

    # Shuffle null: break the phase<->position pairing, re-fit the slope on
    # each shuffle, and compare the resulting MRL. This makes the p-value test
    # the SAME hypothesis the slope fit optimizes (a real position-dependent
    # phase relationship), instead of the slope-free circular-linear
    # correlation that ignores the fitted slope entirely. A fixed coarse grid
    # is used per shuffle (refine=False) to bound the per-shuffle cost.
    #
    # The observed statistic is computed with the SAME coarse procedure as the
    # null (not the refined fit) so the comparison is unbiased: the refined
    # `mean_resultant_length` would sit slightly above its coarse-grid null and
    # spuriously deflate the p-value. The refined MRL is still reported as the
    # fit-quality field.
    observed_mrl = _best_residual_mrl(phases, positions, slope_bounds)

    rng = np.random.default_rng(rng)
    n_shuffles_eff = int(n_shuffles)
    null_mrls = np.empty(n_shuffles_eff, dtype=np.float64)
    for i in range(n_shuffles_eff):
        shuffled_pos = rng.permutation(positions)
        null_mrls[i] = _best_residual_mrl(phases, shuffled_pos, slope_bounds)
    # +1 smoothing avoids p == 0 (Phipson & Smyth 2010).
    pval = float((np.sum(null_mrls >= observed_mrl) + 1) / (n_shuffles_eff + 1))

    # Report the circular-linear correlation alongside as a descriptive
    # effect size (slope-independent), NOT as the significance.
    correlation, _ = circular_linear_correlation(angles=phases, linear_values=positions)

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
    n_shuffles: int = 200,
    rng: int | np.random.Generator | None = None,
) -> bool:
    """Quick check for significant phase precession.

    Parameters
    ----------
    positions : ndarray of shape (n_spikes,)
        Position at each spike.
    phases : ndarray of shape (n_spikes,)
        Spike phase relative to LFP theta in radians or degrees.
    alpha : float, default=0.05
        Significance level for the shuffle test.
    min_correlation : float, default=0.2
        Minimum correlation coefficient required.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.
    n_shuffles : int, default=200
        Number of permutation shuffles for the p-value. A smaller default
        than :func:`phase_precession` (1000) keeps screening fast since this
        function is intended for filtering many neurons.
    rng : int, numpy.random.Generator, or None, optional
        Seed or generator for the shuffles. Pass a fixed value for a
        deterministic result.

    Returns
    -------
    bool
        True if significant phase precession detected (p < alpha,
        r >= min_correlation, and negative slope).

    See Also
    --------
    phase_precession : Full analysis with metrics.

    Notes
    -----
    Genuine input errors (length mismatch, invalid ``angle_unit``) raise
    rather than returning ``False``. Only an insufficient-data ``ValueError``
    from the fit (too few valid spikes) maps to ``False``.

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
    from neurospatial._validation import validate_lengths

    # Validate inputs OUTSIDE the try so genuine input errors propagate.
    if angle_unit not in ("rad", "deg"):
        raise ValueError(f"angle_unit must be 'rad' or 'deg', got '{angle_unit}'")
    positions = np.asarray(positions, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    validate_lengths({"positions": positions, "phases": phases})

    try:
        result = phase_precession(
            positions,
            phases,
            angle_unit=angle_unit,
            n_shuffles=n_shuffles,
            rng=rng,
        )
    except ValueError:
        # Too few spikes after NaN-dropping -> cannot assess precession.
        return False
    return (
        result.pval < alpha
        and result.correlation >= min_correlation
        and result.slope < 0
    )


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
