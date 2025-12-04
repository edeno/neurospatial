"""
Circular statistics and phase precession analysis.

This module provides circular statistics functions for analyzing directional
data common in neuroscience: spike phase relative to LFP, head direction,
and phase precession in place cells.

Which Function Should I Use?
----------------------------
**Screening many neurons (100s-1000s)?**
    Use ``has_phase_precession()`` for fast boolean filtering, or
    ``phase_position_correlation()`` if you need correlation strength.

**Need phase precession slope for publication?**
    Use ``phase_precession()`` for full analysis with slope, offset, and fit quality.
    This is what you want for figures and reporting.

**Custom circular statistics?**
    - ``rayleigh_test()`` - Test for non-uniform circular distribution
    - ``circular_linear_correlation()`` - Generic angular-linear correlation
    - ``circular_circular_correlation()`` - Compare two circular variables

Angle Units
-----------
All functions accept an ``angle_unit`` parameter: ``'rad'`` (default) or ``'deg'``.
Internally, all computations use radians. Output angles are in radians unless
otherwise specified.

**Note**: Most neuroscience software uses degrees by default (especially for
head direction). We use radians to match scipy conventions. If your data is
in degrees, set ``angle_unit='deg'``.

Common Use Cases
----------------
**Screening many neurons (fast filtering)**:

>>> # Fast boolean check - use for initial filtering
>>> if has_phase_precession(positions, phases):
...     print("Neuron shows phase precession")

**Correlation strength only (no slope needed)**:

>>> # Returns correlation coefficient and p-value
>>> r, p = phase_position_correlation(phases, positions)
>>> if p < 0.05 and r > 0.2:
...     print(f"Significant correlation: r={r:.3f}")

**Publication-quality analysis (full metrics)**:

>>> # Use for detailed analysis and publication
>>> from neurospatial.metrics import phase_precession, plot_phase_precession
>>> result = phase_precession(positions, phases)
>>> print(result)  # Automatic interpretation with slope, correlation, fit
>>> plot_phase_precession(positions, phases, result)

**Test for non-uniformity (e.g., preferred theta phase)**:

>>> z, p = rayleigh_test(angles)

**Circular-circular correlation (e.g., phase coherence between electrodes)**:

>>> r, p = circular_circular_correlation(angles1, angles2)

References
----------
Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
    World Scientific.
Kempter, R. et al. (2012). Quantifying circular-linear associations.
    J Neurosci Methods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "PhasePrecessionResult",
    "circular_circular_correlation",
    "circular_linear_correlation",
    "has_phase_precession",
    "phase_position_correlation",
    "phase_precession",
    "plot_phase_precession",
    "rayleigh_test",
]

# Feature-detect scipy.stats.directional_stats (added in scipy 1.9.0)
try:
    from scipy.stats import directional_stats as _scipy_directional_stats

    _HAS_DIRECTIONAL_STATS = True
except ImportError:
    _HAS_DIRECTIONAL_STATS = False
    _scipy_directional_stats = None  # type: ignore[assignment,unused-ignore]


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _to_radians(
    angles: NDArray[np.float64],
    angle_unit: Literal["rad", "deg"],
) -> NDArray[np.float64]:
    """
    Convert angles to radians if needed.

    Parameters
    ----------
    angles : array
        Input angles.
    angle_unit : {'rad', 'deg'}
        Unit of input angles.

    Returns
    -------
    array
        Angles in radians.
    """
    if angle_unit == "deg":
        return np.radians(angles)
    return angles


def _mean_resultant_length(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute mean resultant length (Rayleigh vector length).

    This is an internal helper function. Uses scipy.stats.directional_stats
    when available (scipy >= 1.9.0), otherwise falls back to direct computation.

    Parameters
    ----------
    angles : array
        Angles in radians.
    weights : array, optional
        Weights for each angle. If None, uniform weights.

    Returns
    -------
    float
        Mean resultant length R in [0, 1].

    Notes
    -----
    The mean resultant length is defined as:

        R = |mean(exp(i * angles))| = |sum(w * exp(i * angles))| / sum(w)

    where w are the weights (uniform if not specified).
    """
    if len(angles) == 0:
        return np.nan

    # Convert to unit vectors
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    if weights is not None:
        # Normalize weights
        weights = np.asarray(weights, dtype=np.float64)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.nan
        weights_norm = weights / weight_sum
        mean_cos = np.sum(weights_norm * cos_angles)
        mean_sin = np.sum(weights_norm * sin_angles)
    elif _HAS_DIRECTIONAL_STATS:
        # Use scipy when available (more thoroughly tested)
        vectors = np.column_stack([cos_angles, sin_angles])
        result = _scipy_directional_stats(vectors)
        return float(result.mean_resultant_length)
    else:
        # Fallback: direct computation
        mean_cos = np.mean(cos_angles)
        mean_sin = np.mean(sin_angles)

    return float(np.sqrt(mean_cos**2 + mean_sin**2))


def _validate_circular_input(
    angles: NDArray[np.float64],
    name: str = "angles",
    *,
    min_samples: int = 3,
    check_range: bool = True,
) -> NDArray[np.float64]:
    """
    Validate circular input array.

    Parameters
    ----------
    angles : array
        Input angles to validate.
    name : str
        Name for error messages.
    min_samples : int
        Minimum required samples.
    check_range : bool
        If True, warn if angles outside [0, 2pi].

    Returns
    -------
    array
        Validated angles (NaN removed).

    Raises
    ------
    ValueError
        If validation fails with actionable error message.
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()

    # Check for NaN
    nan_mask = np.isnan(angles)
    n_nan = np.sum(nan_mask)
    if n_nan > 0:
        if n_nan == len(angles):
            raise ValueError(
                f"All {name} values are NaN ({len(angles)} samples). "
                f"Cannot compute circular statistics.\n"
                f"\n"
                f"Diagnostic steps:\n"
                f"1. Check your data: print({name}[:10]) to see actual values\n"
                f"2. If all zeros: spike detection may have failed\n"
                f"3. If all identical: phase extraction may be broken\n"
                f"4. Check data types: {name}.dtype should be float, not object\n"
                f"\n"
                f"Common causes:\n"
                f"  - Wrong variable passed (e.g., spike times instead of phases)\n"
                f"  - Array indexing error (e.g., phases[spike_indices] where "
                f"indices are all -1)\n"
                f"  - Phase computation failed (check LFP signal quality)\n"
                f"\n"
                f"Quick fix: Verify extraction with: np.isnan({name}).sum()"
            )
        # Remove NaN and warn
        angles = angles[~nan_mask]
        warnings.warn(
            f"Removed {n_nan} NaN values from {name}. "
            f"Proceeding with {len(angles)} valid samples.",
            stacklevel=2,
        )

    # Check for Inf
    if np.any(np.isinf(angles)):
        n_inf = np.sum(np.isinf(angles))
        raise ValueError(
            f"{name} contains {n_inf} infinite values. "
            f"Cannot compute circular statistics.\n"
            f"Fix: Remove or replace infinite values before calling this function."
        )

    # Check minimum samples
    if len(angles) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples for circular statistics. "
            f"Got {len(angles)} valid samples in {name}.\n"
            f"Fix: Provide more data points or use a different analysis method."
        )

    # Check range (warning only)
    if check_range and (np.any(angles < 0) or np.any(angles > 2 * np.pi)):
        warnings.warn(
            f"{name} contains values outside [0, 2pi]. "
            f"Range: [{angles.min():.3f}, {angles.max():.3f}]. "
            f"Values will be wrapped to [0, 2pi] using modulo.",
            stacklevel=2,
        )
        angles = angles % (2 * np.pi)

    return angles


def _validate_paired_input(
    arr1: NDArray[np.float64],
    arr2: NDArray[np.float64],
    name1: str,
    name2: str,
    *,
    min_samples: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Validate paired arrays have matching lengths.

    Returns
    -------
    tuple of arrays
        Validated arrays with NaN pairs removed.

    Raises
    ------
    ValueError
        If lengths don't match or insufficient samples.
    """
    arr1 = np.asarray(arr1, dtype=np.float64).ravel()
    arr2 = np.asarray(arr2, dtype=np.float64).ravel()

    if len(arr1) != len(arr2):
        raise ValueError(
            f"{name1} and {name2} must have same length. "
            f"Got {name1}: {len(arr1)}, {name2}: {len(arr2)}.\n"
            f"Fix: Ensure both arrays represent the same events "
            f"(e.g., same spikes)."
        )

    # Remove pairs where either is NaN
    nan_mask = np.isnan(arr1) | np.isnan(arr2)
    if np.any(nan_mask):
        n_removed = np.sum(nan_mask)
        warnings.warn(
            f"Removed {n_removed} pairs containing NaN values. "
            f"Proceeding with {len(arr1) - n_removed} valid pairs.",
            stacklevel=2,
        )
        arr1 = arr1[~nan_mask]
        arr2 = arr2[~nan_mask]

    # Check minimum samples after NaN removal
    if len(arr1) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} valid pairs for analysis. "
            f"Got {len(arr1)} pairs after removing NaN values.\n"
            f"Fix: Provide more data points."
        )

    return arr1, arr2


# =============================================================================
# Public API - Placeholder stubs for future implementation
# =============================================================================


def rayleigh_test(
    angles: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    weights: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """
    Rayleigh test for non-uniformity of circular data.

    Tests H0: angles are uniformly distributed on the circle.
    Rejection indicates a preferred direction exists.

    Parameters
    ----------
    angles : array, shape (n,)
        Sample of angles. Values outside [0, 2pi] (or [0, 360] for degrees)
        will be wrapped.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.
    weights : array, shape (n,), optional
        Weights for each angle (e.g., spike counts per bin).
        If None, uniform weights are used.

    Returns
    -------
    z : float
        Rayleigh z-statistic (n * R^2 where R is mean resultant length).
        Range: [0, n]. Higher values indicate stronger directionality.
    pval : float
        P-value from Rayleigh approximation with finite-sample correction.
        Small p-values (< 0.05) indicate non-uniform distribution.

    Raises
    ------
    ValueError
        If angles array is empty, all NaN, or too short.

    See Also
    --------
    circular_linear_correlation : Test correlation with linear variable.

    Notes
    -----
    Uses finite-sample correction from Mardia & Jupp (2000, Section 5.3.2,
    p. 94) for accurate p-values when n < 50.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import rayleigh_test
    >>> # Uniform distribution - expect high p-value
    >>> uniform_angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> z, p = rayleigh_test(uniform_angles)
    >>> p > 0.5
    True
    """
    # Convert to radians if needed
    angles = np.asarray(angles, dtype=np.float64)
    angles = _to_radians(angles, angle_unit)

    # Validate input (handles NaN, Inf, minimum samples)
    angles = _validate_circular_input(
        angles, "angles", min_samples=3, check_range=False
    )

    n = len(angles)

    # Compute mean resultant length
    r_mean = _mean_resultant_length(angles, weights=weights)

    # Compute effective sample size for weighted data
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        # Effective sample size: sum(w)^2 / sum(w^2)
        # This accounts for unequal weighting
        n_eff: float = float(np.sum(weights) ** 2 / np.sum(weights**2))
    else:
        n_eff = float(n)

    # Rayleigh z-statistic: z = n * R^2
    z = n_eff * r_mean**2

    # P-value with finite-sample correction (Mardia & Jupp, p. 94)
    # For large n, p = exp(-z) is a good approximation
    # For small n, we use the correction formula
    pval = float(np.exp(-z))

    # Apply finite-sample correction for more accurate p-values
    # From Mardia & Jupp (2000), Section 5.3.2, equation 5.3.6
    if n_eff < 50:
        # Correction terms
        term1 = (2 * z - z**2) / (4 * n_eff)
        term2 = (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n_eff**2)
        pval = pval * (1 + term1 - term2)

    # Ensure p-value is in valid range
    pval = float(np.clip(pval, 0.0, 1.0))

    return float(z), pval


def circular_linear_correlation(
    angles: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Circular-linear correlation coefficient.

    Computes the correlation between a circular and a linear variable
    using the Mardia & Jupp formula.

    Parameters
    ----------
    angles : array, shape (n,)
        Circular variable (e.g., spike phases).
    values : array, shape (n,)
        Linear variable (e.g., positions).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    r : float
        Correlation coefficient in [0, 1]. Note: always non-negative.
    pval : float
        P-value from chi-squared distribution with 2 df.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient samples.

    See Also
    --------
    phase_position_correlation : Alias with neuroscience naming.
    circular_circular_correlation : For two circular variables.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_linear_correlation
    >>> positions = np.linspace(0, 100, 50)
    >>> phases = np.linspace(0, 2 * np.pi, 50)  # Perfect linear relationship
    >>> r, p = circular_linear_correlation(phases, positions)
    >>> r > 0.9
    True
    """
    raise NotImplementedError("circular_linear_correlation not yet implemented")


def phase_position_correlation(
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Correlation between spike phases and positions.

    This is an alias for ``circular_linear_correlation`` with neuroscience
    naming conventions.

    Parameters
    ----------
    phases : array, shape (n,)
        Spike phases relative to LFP theta.
    positions : array, shape (n,)
        Position at each spike.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.

    Returns
    -------
    r : float
        Correlation coefficient in [0, 1].
    pval : float
        P-value from chi-squared distribution.

    See Also
    --------
    circular_linear_correlation : Generic version.
    phase_precession : Full phase precession analysis with slope.
    """
    return circular_linear_correlation(phases, positions, angle_unit=angle_unit)


def circular_circular_correlation(
    angles1: NDArray[np.float64],
    angles2: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Circular-circular correlation coefficient.

    Computes the Fisher & Lee correlation coefficient between two
    circular variables.

    Parameters
    ----------
    angles1, angles2 : array, shape (n,)
        Paired circular measurements.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    r : float
        Correlation coefficient in [-1, 1].
    pval : float
        P-value from normal approximation.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient samples.

    See Also
    --------
    circular_linear_correlation : For circular-linear correlation.

    Notes
    -----
    The correlation is symmetric: r(a1, a2) == r(a2, a1).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_circular_correlation
    >>> angles1 = np.random.default_rng(42).uniform(0, 2 * np.pi, 100)
    >>> angles2 = angles1 + 0.1 * np.random.default_rng(42).standard_normal(100)
    >>> r, p = circular_circular_correlation(angles1, angles2)
    >>> r > 0.8
    True
    """
    raise NotImplementedError("circular_circular_correlation not yet implemented")


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
    positions : array, shape (n,)
        Position at each spike.
    phases : array, shape (n,)
        Spike phase relative to LFP theta.
    slope_bounds : tuple of float, default=(-2*pi, 2*pi)
        Bounds for slope optimization (radians per position unit).
    position_range : tuple of float, optional
        If provided, normalize positions to [0, 1]. This changes slope units!
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
    >>> from neurospatial.metrics import phase_precession
    >>> positions = np.linspace(0, 50, 100)  # 0-50 cm
    >>> phases = 2 * np.pi - positions * 0.1  # Negative slope
    >>> result = phase_precession(positions, phases)
    >>> print(result)
    """
    raise NotImplementedError("phase_precession not yet implemented")


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
    positions : array, shape (n,)
        Position at each spike.
    phases : array, shape (n,)
        Spike phase relative to LFP theta.
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
    >>> from neurospatial.metrics import has_phase_precession
    >>> # Random data - no precession expected
    >>> positions = np.random.default_rng(42).uniform(0, 100, 50)
    >>> phases = np.random.default_rng(42).uniform(0, 2 * np.pi, 50)
    >>> has_phase_precession(positions, phases)
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
    positions : array, shape (n,)
        Position at each spike.
    phases : array, shape (n,)
        Spike phase relative to LFP theta (radians).
    result : PhasePrecessionResult, optional
        If provided, overlay fitted line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    position_label : str, default="Position"
        Label for x-axis.
    show_fit : bool, default=True
        If True and result provided, show fitted line.
    marker_size : float, default=20.0
        Size of scatter markers.
    marker_alpha : float, default=0.6
        Alpha of scatter markers.
    show_doubled_note : bool, default=True
        If True, add annotation explaining doubled phase axis.
    scatter_kwargs : dict, optional
        Additional kwargs for scatter plot.
    line_kwargs : dict, optional
        Additional kwargs for fitted line.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import phase_precession, plot_phase_precession
    >>> positions = np.linspace(0, 50, 100)
    >>> phases = (2 * np.pi - positions * 0.1) % (2 * np.pi)
    >>> result = phase_precession(positions, phases)
    >>> ax = plot_phase_precession(positions, phases, result)
    """
    raise NotImplementedError("plot_phase_precession not yet implemented")
