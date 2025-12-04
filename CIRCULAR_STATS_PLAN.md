# Plan: Add Circular Statistics, Phase Precession, and Head Direction Analysis

## Overview

Add a new `neurospatial.metrics.circular` module for circular statistics, phase precession analysis, and head direction cell metrics. This fills gaps identified in repository evaluations:

- **Phase precession**: From CINPLA/phase-precession evaluation
- **Head direction cells**: From PeyracheLab tutorial and mwsoft64 evaluation
- **Circular statistics**: Foundation for both, using scipy where possible

## Design Principles

1. **Use scipy.stats where possible**: `circmean`, `circstd`, `directional_stats`
2. **Implement what scipy lacks**: Rayleigh test, circular-linear correlation, circular-circular correlation
3. **No new dependencies**: Uses existing numpy, scipy, matplotlib only
4. **Match existing patterns**: Follow neurospatial conventions (NumPy docstrings, type hints, TYPE_CHECKING guards)
5. **Standalone functions**: Like `place_fields.py` and `grid_cells.py`, not Environment methods
6. **Consistent angle units**: Support both degrees and radians with explicit `angle_unit` parameter

## Implementation Sources (Verified)

All algorithms verified against authoritative sources:

- **CircStats R package** (Jammalamadaka & SenGupta, 2001) - `r.test`, `circ.cor`
- **CINPLA/phase-precession** (Kempter et al., 2012) - `cl_corr` (circular-linear correlation with slope fitting)
- **pycircstat** (Berens, 2009) - `corrcc`, `corrcl`, `mean`, `resultant_vector_length`
- **Astropy** - `rayleightest` (with finite-sample correction)
- **Pingouin** - `circ_corrcl` (Mardia & Jupp formula)

**Note**: Uses `scipy.stats.directional_stats` when available (scipy >= 1.9.0).
The project already requires scipy >= 1.10.0, so no version bump is needed.
A fallback implementation is provided for older scipy versions.

## Scipy Coverage Analysis

| Function | scipy.stats | Need to Implement |
|----------|-------------|-------------------|
| Circular mean | `circmean` | No - use scipy |
| Circular std | `circstd` | No - use scipy |
| Mean resultant length | `directional_stats().mean_resultant_length` | No - use scipy |
| Rayleigh test | No | **Yes** (uses `directional_stats` internally) |
| Circular-linear correlation | No | **Yes** (uses `scipy.stats.pearsonr`) |
| Circular-circular correlation | No | **Yes** (uses `scipy.stats.circmean`) |

### Implementation: Mean Resultant Length

**Decision**: Use `scipy.stats.directional_stats` for consistency and maintainability:

```python
from scipy.stats import directional_stats

def _mean_resultant_length(angles: NDArray[np.float64]) -> float:
    """
    Compute mean resultant length (Rayleigh vector length) using scipy.

    Parameters
    ----------
    angles : array
        Angles in radians.

    Returns
    -------
    float
        Mean resultant length R in [0, 1].
    """
    if len(angles) == 0:
        return np.nan
    # Convert angles to unit vectors for directional_stats
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    result = directional_stats(vectors)
    return float(result.mean_resultant_length)
```

**Rationale**: Using scipy ensures we benefit from well-tested, maintained code.
This follows the project's design principle of using scipy where possible.

## Module Structure

```
src/neurospatial/metrics/
├── circular.py          # NEW: Circular statistics and phase precession
├── head_direction.py    # NEW: Head direction cell analysis
├── __init__.py          # Update exports
└── ... (existing)
```

---

## Part 1: `circular.py` - Circular Statistics Foundation

### Module Header

```python
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
Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics. World Scientific.
Kempter, R. et al. (2012). Quantifying circular-linear associations. J Neurosci Methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import fminbound

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "rayleigh_test",
    "circular_linear_correlation",
    "circular_circular_correlation",
    "phase_precession",
    "phase_position_correlation",
    "plot_phase_precession",
    "PhasePrecessionResult",
    "has_phase_precession",
]
```

### 1.1 Angle Unit Conversion, Mean Resultant Length, and Input Validation

```python
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


# Feature-detect scipy.stats.directional_stats (added in scipy 1.9.0)
try:
    from scipy.stats import directional_stats as _scipy_directional_stats
    _HAS_DIRECTIONAL_STATS = True
except ImportError:
    _HAS_DIRECTIONAL_STATS = False
    _scipy_directional_stats = None  # type: ignore[assignment]


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
                f"  - Array indexing error (e.g., phases[spike_indices] where indices are all -1)\n"
                f"  - Phase computation failed (check LFP signal quality)\n"
                f"\n"
                f"Quick fix: Verify extraction with: np.isnan({name}).sum()"
            )
        # Remove NaN and warn
        angles = angles[~nan_mask]
        import warnings
        warnings.warn(
            f"Removed {n_nan} NaN values from {name}. "
            f"Proceeding with {len(angles)} valid samples.",
            stacklevel=2,
        )

    # Check for Inf
    if np.any(np.isinf(angles)):
        n_inf = np.sum(np.isinf(angles))
        raise ValueError(
            f"{name} contains {n_inf} infinite values. Cannot compute circular statistics.\n"
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
    if check_range:
        if np.any(angles < 0) or np.any(angles > 2 * np.pi):
            import warnings
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
            f"Fix: Ensure both arrays represent the same events (e.g., same spikes)."
        )

    # Remove pairs where either is NaN
    nan_mask = np.isnan(arr1) | np.isnan(arr2)
    if np.any(nan_mask):
        n_removed = np.sum(nan_mask)
        import warnings
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
```

### 1.2 Rayleigh Test for Uniformity

```python
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
    head_direction_metrics : Uses Rayleigh test for HD cell classification.

    Notes
    -----
    Uses finite-sample correction from Mardia & Jupp (2000, Section 5.3.2,
    p. 94) for accurate p-values when n < 50. For large n, the correction
    is negligible.

    The correction formula is::

        p_corrected = exp(-z) * (1 + (2z - z²)/(4n) - (24z - 132z² + 76z³ - 9z⁴)/(288n²))

    Examples
    --------
    >>> from neurospatial.metrics import rayleigh_test
    >>> # Test if spikes have preferred theta phase (radians)
    >>> z, p = rayleigh_test(spike_phases)
    >>> if p < 0.05:
    ...     print(f"Significant phase preference (z={z:.2f}, p={p:.4f})")

    >>> # Test with firing rate weights
    >>> z, p = rayleigh_test(bin_centers, weights=firing_rates)

    >>> # Using degrees
    >>> z, p = rayleigh_test(head_directions_deg, angle_unit='deg')

    References
    ----------
    Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
        Section 5.3.2, p. 94 (finite-sample correction).
    Zar, J.H. (1999). Biostatistical Analysis. 4th ed. Chapter 27.
    """
    # Convert to radians if needed
    angles = _to_radians(np.asarray(angles, dtype=np.float64), angle_unit)

    # Validate input
    angles = _validate_circular_input(angles, "angles", min_samples=3)

    n = len(angles)

    # Handle weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if len(weights) != n:
            raise ValueError(
                f"weights must have same length as angles. "
                f"Got angles: {n}, weights: {len(weights)}."
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if np.sum(weights) == 0:
            raise ValueError("weights sum to zero. Cannot compute weighted statistics.")

        # Weighted mean resultant length using centralized helper
        rbar = _mean_resultant_length(angles, weights=weights)
        # Effective sample size for weighted data (approximate)
        # Note: p-values are approximate for heavily skewed weights
        n_eff = np.sum(weights) ** 2 / np.sum(weights**2)
    else:
        # Unweighted mean resultant length using centralized helper
        rbar = _mean_resultant_length(angles)
        n_eff = n

    # Rayleigh statistic
    z = n_eff * rbar**2

    # P-value with finite-sample correction (Mardia & Jupp, 2000, p. 94)
    pval = np.exp(-z)
    if n_eff < 50:
        # Finite-sample correction
        temp = 1 + (2 * z - z**2) / (4 * n_eff)
        temp -= (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n_eff**2)
        pval = pval * temp

    # Ensure p-value is in valid range
    pval = np.clip(pval, 0.0, 1.0)

    return float(z), float(pval)
```

### 1.3 Circular-Linear Correlation

```python
def circular_linear_correlation(
    angles: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Correlation between a circular and a linear variable.

    Computes the angular-linear correlation coefficient using the
    Mardia & Jupp (2000) formula, which properly handles the circular
    nature of one variable.

    Parameters
    ----------
    angles : array, shape (n,)
        Circular variable (e.g., spike phases, head direction).
        Values outside [0, 2pi] will be wrapped.
    values : array, shape (n,)
        Linear variable (e.g., position, time, speed).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    r : float
        Angular-linear correlation coefficient in [0, 1].
        Note: Unlike Pearson correlation, this is always non-negative.
    pval : float
        P-value from chi-squared test with 2 degrees of freedom.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient data.

    See Also
    --------
    phase_position_correlation : Alias with neuroscience-friendly naming.
    phase_precession : Full phase precession analysis with regression.
    circular_circular_correlation : For two circular variables.

    Notes
    -----
    **Formula** (Mardia & Jupp, 2000, Chapter 11.2, pp. 244-245):

        r_xc = corr(values, cos(angles))
        r_xs = corr(values, sin(angles))
        r_cs = corr(sin(angles), cos(angles))

        r = sqrt((r_xc^2 + r_xs^2 - 2*r_xc*r_xs*r_cs) / (1 - r_cs^2))

    **Interpretation**:

    - r > 0.3: Moderate correlation (typical threshold for phase precession)
    - r > 0.5: Strong correlation
    - p < 0.05: Statistically significant

    Examples
    --------
    >>> from neurospatial.metrics import circular_linear_correlation
    >>> # Correlate spike phase with position
    >>> r, p = circular_linear_correlation(spike_phases, spike_positions)
    >>> print(f"Correlation: r={r:.3f}, p={p:.4f}")

    >>> # Using degrees
    >>> r, p = circular_linear_correlation(phases_deg, positions, angle_unit='deg')

    References
    ----------
    Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
        Chapter 11.2, pp. 244-245 (angular-linear correlation).
    Berens, P. (2009). CircStat: A MATLAB Toolbox for Circular Statistics.
    """
    # Convert to radians if needed
    angles = _to_radians(np.asarray(angles, dtype=np.float64), angle_unit)
    values = np.asarray(values, dtype=np.float64)

    # Validate paired inputs first (handles NaN removal)
    angles, values = _validate_paired_input(angles, values, "angles", "values", min_samples=5)

    # Validate circular input (range check and wrapping)
    # Note: Don't remove more values here since we need paired data
    if np.any(angles < 0) or np.any(angles > 2 * np.pi):
        import warnings
        warnings.warn(
            f"angles contains values outside [0, 2pi]. "
            f"Range: [{angles.min():.3f}, {angles.max():.3f}]. "
            f"Values will be wrapped to [0, 2pi] using modulo.",
            stacklevel=2,
        )
        angles = angles % (2 * np.pi)

    n = len(angles)

    # Compute Pearson correlations using scipy
    rxs, _ = stats.pearsonr(values, np.sin(angles))
    rxc, _ = stats.pearsonr(values, np.cos(angles))
    rcs, _ = stats.pearsonr(np.sin(angles), np.cos(angles))

    # Handle edge case where rcs = 1 (would cause division by zero)
    if np.abs(rcs) > 0.9999:
        # Degenerate case: sin and cos perfectly correlated
        import warnings
        warnings.warn(
            "Degenerate case: sin(angles) and cos(angles) are highly correlated. "
            "This may indicate all angles are nearly identical. Returning r=0.",
            stacklevel=2,
        )
        return 0.0, 1.0

    # Compute angular-linear correlation (Mardia & Jupp, Ch. 11.2)
    r_squared = (rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2)

    # Handle numerical issues
    if r_squared < 0:
        r_squared = 0.0  # Can happen due to floating point errors
    r = np.sqrt(r_squared)

    # P-value from chi-squared with 2 df
    pval = stats.chi2.sf(n * r**2, 2)

    return float(r), float(pval)


def phase_position_correlation(
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Correlation between spike phase and position.

    Alias for `circular_linear_correlation` with neuroscience-friendly naming.
    Use this when analyzing phase precession in place cells.

    Parameters
    ----------
    phases : array, shape (n,)
        Spike phases (circular variable).
    positions : array, shape (n,)
        Position values in environment units (linear variable).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.

    Returns
    -------
    correlation : float
        Angular-linear correlation coefficient in [0, 1].
    pval : float
        Statistical significance.

    See Also
    --------
    circular_linear_correlation : Generic circular-linear correlation.
    phase_precession : Full phase precession analysis with regression.

    Examples
    --------
    >>> from neurospatial.metrics import phase_position_correlation
    >>> # Quick check for phase-position relationship
    >>> r, p = phase_position_correlation(spike_phases, spike_positions)
    >>> if p < 0.05 and r > 0.2:
    ...     print("Significant phase-position correlation detected")
    """
    return circular_linear_correlation(phases, positions, angle_unit=angle_unit)
```

### 1.4 Circular-Circular Correlation

```python
def circular_circular_correlation(
    angles1: NDArray[np.float64],
    angles2: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Correlation between two circular variables.

    Uses the Fisher & Lee (1983) circular correlation coefficient,
    which is appropriate when both variables are angular.

    Parameters
    ----------
    angles1, angles2 : array, shape (n,)
        Two circular variables.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles (applies to both arrays).

    Returns
    -------
    r : float
        Circular correlation coefficient in [-1, 1].
        - r > 0: positive association
        - r < 0: negative association
        - r = 0: no circular correlation
    pval : float
        P-value from normal approximation.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient data.

    See Also
    --------
    circular_linear_correlation : When one variable is linear.

    Notes
    -----
    **Formula** (Fisher & Lee, 1983; Jammalamadaka & SenGupta, 2001, p. 176):

        rho = sum(sin(a1 - mean1) * sin(a2 - mean2)) /
              sqrt(sum(sin(a1 - mean1)^2) * sum(sin(a2 - mean2)^2))

    **Interpretation**:

    - |r| > 0.3: Moderate association
    - |r| > 0.5: Strong association
    - Positive r: angles increase together
    - Negative r: angles move in opposite directions

    Examples
    --------
    >>> from neurospatial.metrics import circular_circular_correlation
    >>> # Compare two phase measurements
    >>> r, p = circular_circular_correlation(phase_electrode1, phase_electrode2)
    >>> print(f"Phase coherence: r={r:.3f}, p={p:.4f}")

    >>> # Using degrees
    >>> r, p = circular_circular_correlation(hd1_deg, hd2_deg, angle_unit='deg')

    References
    ----------
    Fisher, N.I. & Lee, A.J. (1983). A correlation coefficient for circular data.
        Biometrika, 70(2), 327-332.
    Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
        World Scientific, p. 176.
    """
    # Convert to radians if needed
    angles1 = _to_radians(np.asarray(angles1, dtype=np.float64), angle_unit)
    angles2 = _to_radians(np.asarray(angles2, dtype=np.float64), angle_unit)

    # Validate paired inputs first (handles NaN removal and length check)
    angles1, angles2 = _validate_paired_input(angles1, angles2, "angles1", "angles2", min_samples=5)

    n = len(angles1)

    # Compute circular means using scipy
    alpha_bar = stats.circmean(angles1, high=2 * np.pi, low=0)
    beta_bar = stats.circmean(angles2, high=2 * np.pi, low=0)

    # Compute deviations from circular mean
    sin_dev1 = np.sin(angles1 - alpha_bar)
    sin_dev2 = np.sin(angles2 - beta_bar)

    # Compute correlation coefficient (Fisher & Lee formula)
    num = np.sum(sin_dev1 * sin_dev2)
    den = np.sqrt(np.sum(sin_dev1**2) * np.sum(sin_dev2**2))

    if den == 0:
        # Degenerate case: no variation in one or both variables
        import warnings
        warnings.warn(
            "No variation in one or both circular variables. "
            "This may indicate constant angles. Returning r=0.",
            stacklevel=2,
        )
        return 0.0, 1.0

    rho = num / den

    # Clip to valid range (numerical stability)
    rho = np.clip(rho, -1.0, 1.0)

    # Significance test (normal approximation)
    l20 = np.mean(sin_dev1**2)
    l02 = np.mean(sin_dev2**2)
    l22 = np.mean(sin_dev1**2 * sin_dev2**2)

    if l22 == 0:
        import warnings
        warnings.warn(
            "Cannot compute p-value: variance product is zero. Returning p=1.0.",
            stacklevel=2,
        )
        return float(rho), 1.0

    ts = np.sqrt((n * l20 * l02) / l22) * rho
    pval = 2 * (1 - stats.norm.cdf(np.abs(ts)))

    return float(rho), float(pval)
```

---

## Part 2: Phase Precession Analysis

### 2.1 Phase Precession Result Dataclass

```python
@dataclass
class PhasePrecessionResult:
    """
    Results from phase precession analysis.

    Attributes
    ----------
    slope : float
        Phase change per unit position (rad/unit).
        Negative slope indicates phase precession (typical for place cells).
        Positive slope indicates phase recession.
    slope_units : str
        Units of the slope. Either "rad/position_unit" (default) or
        "rad/place_field" (when position_range was used).
    offset : float
        Phase offset at position=0 (radians, in [0, 2pi]).
    correlation : float
        Circular-linear correlation coefficient in [0, 1].
    pval : float
        P-value for the correlation.
    mean_resultant_length : float
        Goodness of fit measure in [0, 1]. Higher values indicate
        residuals are tightly clustered, meaning better model fit.

    Notes
    -----
    The model is: phase = slope * position + offset (mod 2pi)

    Examples
    --------
    >>> result = phase_precession(positions, phases)
    >>> print(result)  # Shows interpretation
    >>> if result.is_significant():
    ...     print(f"Slope: {result.slope:.3f} {result.slope_units}")
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
            Significance threshold.

        Returns
        -------
        bool
            True if pval < alpha.
        """
        return self.pval < alpha

    def interpretation(self) -> str:
        """
        Human-readable interpretation of phase precession results.

        Returns
        -------
        str
            Multi-line interpretation with significance assessment,
            direction (precession vs recession), and quality metrics.

        Examples
        --------
        >>> result = phase_precession(positions, phases)
        >>> print(result.interpretation())
        """
        lines = []

        # Significance assessment
        if self.pval < 0.001:
            lines.append("Significance: *** Highly significant (p < 0.001)")
        elif self.pval < 0.01:
            lines.append("Significance: ** Significant (p < 0.01)")
        elif self.pval < 0.05:
            lines.append("Significance: * Marginally significant (p < 0.05)")
        else:
            lines.append(f"Significance: Not significant (p = {self.pval:.3f})")

        # Direction interpretation
        if self.slope < -0.1:
            lines.append(f"Direction: Phase PRECESSION (slope = {self.slope:.3f} {self.slope_units})")
            lines.append("    > Spikes occur at earlier phases as animal moves through field")
        elif self.slope > 0.1:
            lines.append(f"Direction: Phase RECESSION (slope = {self.slope:.3f} {self.slope_units})")
            lines.append("    > Spikes occur at later phases as animal moves through field")
        else:
            lines.append(f"Direction: No clear trend (slope = {self.slope:.3f}, near zero)")

        # Correlation strength (thresholds from hippocampal literature)
        if self.correlation > 0.4:
            lines.append(f"Correlation: Strong (r = {self.correlation:.3f}, threshold = 0.3)")
        elif self.correlation > 0.3:
            lines.append(f"Correlation: Moderate (r = {self.correlation:.3f}, threshold = 0.3)")
            lines.append("    > r > 0.3 is typical threshold for phase precession")
        elif self.correlation > 0.2:
            lines.append(f"Correlation: Weak (r = {self.correlation:.3f})")
            lines.append("    > Below typical threshold (0.3) but may still be meaningful")
        else:
            lines.append(f"Correlation: Very weak (r = {self.correlation:.3f})")

        # Fit quality (typical range for place cells: 0.1-0.6)
        if self.mean_resultant_length > 0.3:
            lines.append(f"Fit quality: Good (R = {self.mean_resultant_length:.3f}, threshold = 0.3)")
            lines.append("    > R indicates tight clustering around regression line")
        else:
            lines.append(f"Fit quality: Poor (R = {self.mean_resultant_length:.3f} < 0.3)")
            lines.append("    > Low R suggests noisy data or weak phase precession")
            lines.append("    > Typical range for hippocampal place cells: 0.1-0.6")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Automatic interpretation when printing result."""
        header = (
            f"PhasePrecessionResult(slope={self.slope:.4f} {self.slope_units}, "
            f"offset={self.offset:.4f}, r={self.correlation:.4f}, "
            f"p={self.pval:.4f}, R={self.mean_resultant_length:.4f})"
        )
        return f"{header}\n\n{self.interpretation()}"
```

### 2.2 Phase Precession Analysis Function

```python
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
    Analyze phase precession between position and spike phase.

    Fits the model: phase = slope * position + offset (mod 2pi)

    Uses the method of Kempter et al. (2012) which finds the slope
    that maximizes the mean resultant length of the residuals.

    Parameters
    ----------
    positions : array, shape (n_spikes,)
        Position of animal at each spike time (in environment units, e.g., cm).
    phases : array, shape (n_spikes,)
        LFP theta phase at each spike time (will be wrapped to [0, 2pi]).
    slope_bounds : tuple of float, default=(-2pi, 2pi)
        Bounds for slope search (radians per position unit).

        **Default (-2pi, 2pi)** allows one complete phase cycle per position unit.

        **Physical interpretation examples**:

        - Default (-2pi, 2pi): One full cycle per position unit.
          Example: 40cm place field -> phase changes 0 to 360 deg across field
        - Tight (-pi, pi): Half cycle per unit.
          Use for: Large place fields (>50cm) where you expect slow precession
        - Wide (-4pi, 4pi): Two full cycles per unit.
          Use for: Small place fields (<20cm) or suspected multiple cycles

        **How to choose**: Look at your data first with ``plot_phase_precession()``.
        If the fitted line is horizontal, try wider bounds. If fit looks
        wrong, try tighter bounds.
    position_range : tuple of float, optional
        If provided, normalize positions to [0, 1] within this range before fitting.

        .. warning::
           Using ``position_range`` changes slope units!

           - **Without** position_range: slope is in rad/position_unit (e.g., rad/cm)
           - **With** position_range: slope is in rad/place_field (normalized)

           **DO NOT compare slopes from analyses with and without position_range.**

        **When to use**:

        - Comparing precession across different-sized place fields
        - When position units vary between recordings
        - To get slope in units of "radians per place field"

        Example: For 40cm place field, use position_range=(0, 40)
        to get slope in "radians per 40cm traversal".
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input phases.
    min_spikes : int, default=10
        Minimum number of spikes required for reliable analysis.

        **Trade-off**: More spikes = higher confidence but fewer neurons pass.

        **Recommendations**:

        - 10: Fast screening (accepts borderline cells, ~15% false positives)
        - 20: Balanced analysis (good confidence, ~5% false positives)
        - 50: Publication quality (high confidence, <1% false positives,
          but may reject weak-but-real phase precession)

        When in doubt, use 20 for initial analysis, then verify with
        visual inspection using ``plot_phase_precession()``.

    Returns
    -------
    PhasePrecessionResult
        Dataclass with slope, slope_units, offset, correlation, pval,
        mean_resultant_length. Use ``print(result)`` for automatic interpretation.

    Raises
    ------
    ValueError
        If inputs are invalid with actionable error message.

    See Also
    --------
    phase_position_correlation : Quick correlation without regression.
    plot_phase_precession : Visualize results.
    has_phase_precession : Quick boolean check.

    Notes
    -----
    **Algorithm** (Kempter et al., 2012):

    1. For each candidate slope s in [slope_min, slope_max]:
       - Compute residuals: r_i = phase_i - s * position_i
       - Compute mean resultant length R(s) of residuals
    2. Find slope s* that maximizes R(s) using bounded optimization
    3. Compute offset as circular mean of (phase - s* * position)
    4. Compute circular-linear correlation and p-value

    **Interpretation**:

    - Negative slope: phase precession (earlier phase at later positions)
    - Positive slope: phase recession
    - High correlation (>0.3): significant phase-position relationship
      (typical threshold from hippocampal literature)
    - High mean_resultant_length (>0.3): good fit quality
      (typical range for place cells: 0.1-0.6)

    Examples
    --------
    >>> from neurospatial.metrics import phase_precession
    >>> # Basic usage
    >>> result = phase_precession(spike_positions, spike_phases)
    >>> print(result)
    PhasePrecessionResult(slope=-1.2500 rad/position_unit, offset=3.1416, ...)

    Significance: *** Highly significant (p < 0.001)
    Direction: Phase PRECESSION (slope = -1.250 rad/position_unit)
        > Spikes occur at earlier phases as animal moves through field
    Correlation: Strong (r = 0.623, threshold = 0.3)
    Fit quality: Good (R = 0.452, threshold = 0.3)

    >>> # Check significance and interpret
    >>> if result.is_significant():
    ...     print(f"Phase changes {result.slope:.2f} {result.slope_units}")

    >>> # Normalize to place field width (changes slope units!)
    >>> result = phase_precession(positions, phases, position_range=(10, 50))
    >>> # Now slope is in rad/place_field, not rad/cm

    References
    ----------
    Kempter, R., Leibold, C., Buzsaki, G., Diba, K., & Schmidt, R. (2012).
        Quantifying circular-linear associations: Hippocampal phase precession.
        Journal of Neuroscience Methods, 207(1), 113-124.
    O'Keefe, J. & Recce, M.L. (1993). Phase relationship between hippocampal
        place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
    """
    # Convert to radians if needed
    phases = _to_radians(np.asarray(phases, dtype=np.float64), angle_unit)
    positions = np.asarray(positions, dtype=np.float64)

    # Validate inputs
    positions, phases = _validate_paired_input(
        positions, phases, "positions", "phases", min_samples=min_spikes
    )

    n = len(positions)
    if n < min_spikes:
        raise ValueError(
            f"Need at least {min_spikes} spikes for reliable phase precession analysis. "
            f"Got {n} spikes.\n"
            f"Possible fixes:\n"
            f"  - Use longer recording session\n"
            f"  - Combine multiple passes through place field\n"
            f"  - Lower min_spikes (but reduces reliability)\n"
            f"  - Use phase_position_correlation() instead (needs fewer samples)"
        )

    # Wrap phases to [0, 2pi]
    phases = phases % (2 * np.pi)

    # Validate slope bounds
    if slope_bounds[0] >= slope_bounds[1]:
        raise ValueError(
            f"slope_bounds[0] must be < slope_bounds[1]. Got {slope_bounds}."
        )

    # Determine slope units and normalize positions if requested
    if position_range is not None:
        pos_min, pos_max = position_range
        if pos_min >= pos_max:
            raise ValueError(
                f"position_range[0] must be < position_range[1]. Got {position_range}."
            )
        positions = (positions - pos_min) / (pos_max - pos_min)
        slope_units = "rad/place_field"

        # Warn user about unit change
        import warnings
        warnings.warn(
            "position_range is normalizing positions to [0, 1].\n"
            "IMPORTANT: This changes slope units!\n"
            "  - Without position_range: slope is rad/position_unit (e.g., rad/cm)\n"
            "  - With position_range: slope is rad/place_field (normalized)\n"
            "DO NOT compare slopes from analyses with and without position_range.",
            UserWarning,
            stacklevel=2,
        )
    else:
        slope_units = "rad/position_unit"

    # Define optimization function using centralized helper
    def _neg_residual_mrl(slope: float) -> float:
        """Negative mean resultant length of residuals (for minimization)."""
        residuals = phases - slope * positions
        return -_mean_resultant_length(residuals)

    # Find optimal slope by maximizing mean resultant length of residuals
    optimal_slope, neg_mrl, _, _ = fminbound(
        _neg_residual_mrl,
        slope_bounds[0],
        slope_bounds[1],
        full_output=True,
    )

    # Goodness of fit is the mean resultant length at optimal slope
    mrl = -neg_mrl  # Convert back from negative

    # Compute phase offset (circular mean of residuals)
    residuals = phases - optimal_slope * positions
    offset = stats.circmean(residuals, high=2 * np.pi, low=0)

    # Ensure offset is in [0, 2pi]
    offset = offset % (2 * np.pi)

    # Compute correlation using circular-linear correlation
    correlation, pval = circular_linear_correlation(phases, positions)

    return PhasePrecessionResult(
        slope=float(optimal_slope),
        slope_units=slope_units,
        offset=float(offset),
        correlation=float(correlation),
        pval=float(pval),
        mean_resultant_length=float(mrl),
    )


def has_phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    min_correlation: float = 0.2,
) -> bool:
    """
    Quick check: Does this neuron show significant phase precession?

    Convenience function for fast screening without detailed analysis.
    For full analysis, use `phase_precession()`.

    Parameters
    ----------
    positions, phases : array
        Same as `phase_precession()`.
    alpha : float, default=0.05
        Significance threshold.
    min_correlation : float, default=0.2
        Minimum correlation to consider meaningful.

    Returns
    -------
    bool
        True if significant phase precession detected (p < alpha,
        correlation >= min_correlation, and negative slope).

    See Also
    --------
    phase_precession : Full analysis with detailed results.

    Examples
    --------
    >>> from neurospatial.metrics import has_phase_precession
    >>> # Screen many neurons quickly
    >>> for i, (pos, phase) in enumerate(all_neurons):
    ...     if has_phase_precession(pos, phase):
    ...         print(f"Neuron {i} shows phase precession!")
    """
    try:
        result = phase_precession(positions, phases)
        return (
            result.pval < alpha
            and result.correlation >= min_correlation
            and result.slope < 0  # Precession, not recession
        )
    except ValueError:
        # Not enough data
        return False
```

### 2.3 Phase Precession Plot

```python
def plot_phase_precession(
    positions: NDArray[np.float64],
    phases: NDArray[np.float64],
    result: PhasePrecessionResult | None = None,
    *,
    ax: "Axes | None" = None,
    position_label: str = "Position",
    show_fit: bool = True,
    marker_size: float = 20.0,
    marker_alpha: float = 0.6,
    show_doubled_note: bool = True,
    scatter_kwargs: dict | None = None,
    line_kwargs: dict | None = None,
) -> "Axes":
    """
    Plot phase precession with doubled phase axis (0-4pi).

    Each spike is plotted twice (at theta and theta+2pi) to emphasize the linear
    relationship. This is the standard visualization from O'Keefe & Recce (1993).

    **IMPORTANT**: Each data point appears TWICE in this plot. This is NOT a bug -
    it's the standard convention that makes linear trends visually obvious.

    Parameters
    ----------
    positions : array, shape (n_spikes,)
        Position of animal at each spike time.
    phases : array, shape (n_spikes,)
        LFP theta phase at each spike time (radians, [0, 2pi]).
    result : PhasePrecessionResult, optional
        If provided, overlay the fitted regression line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    position_label : str, default="Position"
        Label for x-axis.
    show_fit : bool, default=True
        If True and result provided, show fitted line.
    marker_size : float, default=20.0
        Size of scatter markers.
    marker_alpha : float, default=0.6
        Alpha transparency of markers.
    show_doubled_note : bool, default=True
        If True, add text annotation explaining doubled phase axis.
    scatter_kwargs : dict, optional
        Additional keyword arguments passed to ``ax.scatter()``.
    line_kwargs : dict, optional
        Additional keyword arguments passed to ``ax.plot()`` for fit line.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    See Also
    --------
    phase_precession : Compute phase precession analysis.

    Notes
    -----
    **Why does each spike appear TWICE?**

    This makes phase precession VISIBLE! The doubled phase convention is essential:

    1. **Without doubling**: Phases wrap 0 -> 2pi -> 0, hiding the trend.
       You see a scattered cloud of points.

    2. **With doubling**: Linear trend is obvious (diagonal line).
       The wrap-around appears as a continuous slope.

    This is NOT a bug - it's the standard convention from O'Keefe & Recce (1993)
    used in all major phase precession papers.

    **Interpretation**:

    - Negative slope (top-left to bottom-right): phase precession
    - Positive slope (bottom-left to top-right): phase recession
    - Horizontal scatter: no phase-position relationship

    **Alternative (single cycle)**:

    To plot original phases only (0-2pi), use standard scatter:

        >>> fig, ax = plt.subplots()
        >>> ax.scatter(positions, phases % (2*np.pi), alpha=0.6)
        >>> ax.set_ylim(0, 2*np.pi)

    Examples
    --------
    >>> from neurospatial.metrics import phase_precession, plot_phase_precession
    >>> result = phase_precession(positions, phases)
    >>> ax = plot_phase_precession(positions, phases, result)
    >>> plt.show()

    References
    ----------
    O'Keefe, J. & Recce, M.L. (1993). Phase relationship between hippocampal
        place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Set up kwargs with defaults
    scatter_kw = {"s": marker_size, "alpha": marker_alpha, "c": "black", "edgecolors": "none"}
    if scatter_kwargs is not None:
        scatter_kw.update(scatter_kwargs)

    line_kw = {"color": "red", "linewidth": 2}
    if line_kwargs is not None:
        line_kw.update(line_kwargs)

    # Ensure phases are in [0, 2pi]
    phases_normalized = phases % (2 * np.pi)

    # Plot phases twice: at theta and theta+2pi
    positions_doubled = np.concatenate([positions, positions])
    phases_doubled = np.concatenate([phases_normalized, phases_normalized + 2 * np.pi])

    ax.scatter(positions_doubled, phases_doubled, **scatter_kw)

    # Add fitted line if result provided
    if show_fit and result is not None:
        pos_range = np.array([positions.min(), positions.max()])
        # Model: phase = slope * position + offset
        fit_phases = result.slope * pos_range + result.offset
        # Plot fit line twice (for both phase copies)
        ax.plot(pos_range, fit_phases, label="Fit", **line_kw)
        ax.plot(pos_range, fit_phases + 2 * np.pi, **line_kw)

    # Format axes
    ax.set_xlabel(position_label)
    ax.set_ylabel("Theta phase (rad)")
    ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
    ax.set_ylim(0, 4 * np.pi)

    # Add annotation explaining doubled phase
    if show_doubled_note:
        ax.text(
            0.02,
            0.98,
            "WHY does each spike appear TWICE?\n"
            "This makes phase precession visible!\n"
            "\n"
            "Without doubling: Phases wrap 0→2π→0, hiding the trend\n"
            "With doubling: Linear trend is obvious (diagonal line)\n"
            "\n"
            "This is NOT a bug - it's the standard convention\n"
            "from O'Keefe & Recce (1993)\n"
            "\n"
            "To plot single cycle: ax.scatter(pos, phase % (2*np.pi))",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
        )

    return ax
```

---

## Part 3: `head_direction.py` - Head Direction Cell Analysis

### Module Header

```python
"""
Head direction cell analysis.

This module provides functions for computing head direction tuning curves
and classifying head direction cells.

Typical Workflow
----------------
1. Compute tuning curve:

   >>> from neurospatial.metrics import head_direction_tuning_curve
   >>> bins, rates = head_direction_tuning_curve(
   ...     head_directions, spike_times, position_times
   ... )

2. Compute metrics:

   >>> from neurospatial.metrics import head_direction_metrics
   >>> metrics = head_direction_metrics(bins, rates)
   >>> if metrics.is_hd_cell:
   ...     print(f"HD cell with preferred direction: {metrics.preferred_direction_deg:.1f} deg")

3. (Optional) Visualize:

   >>> from neurospatial.metrics import plot_head_direction_tuning
   >>> plot_head_direction_tuning(bins, rates, metrics)

References
----------
Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells recorded
    from the postsubiculum in freely moving rats. J Neurosci, 10(2), 420-435.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

# Import from sibling module (internal use)
from .circular import rayleigh_test, _mean_resultant_length

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "head_direction_tuning_curve",
    "head_direction_metrics",
    "HeadDirectionMetrics",
    "plot_head_direction_tuning",
    "is_head_direction_cell",
]
```

### 3.1 Head Direction Tuning Curve

```python
def head_direction_tuning_curve(
    head_directions: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    position_times: NDArray[np.float64],
    *,
    bin_size: float = 6.0,
    angle_unit: Literal["deg", "rad"] = "deg",
    smoothing_window: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute head direction tuning curve.

    Parameters
    ----------
    head_directions : array, shape (n_samples,)
        Head direction at each position sample. Units determined by `angle_unit`.
    spike_times : array, shape (n_spikes,)
        Times of spikes (same time units as position_times).
    position_times : array, shape (n_samples,)
        Times corresponding to head_directions. Must be strictly increasing
        (monotonic) for interpolation to behave correctly.
    bin_size : float, default=6.0
        Angular bin size. Units determined by `angle_unit`.
        Default is 6 degrees, standard in HD cell literature.
    angle_unit : {'deg', 'rad'}, default='deg'
        Unit for input angles and bin_size.
        - 'deg': degrees (0-360)
        - 'rad': radians (0-2pi)
    smoothing_window : int, default=5
        Width of Gaussian smoothing window (number of bins).
        Set to 0 to disable smoothing.

        **Note**: Smoothing uses circular boundary handling to properly
        wrap around at 0/360 degrees.

    Returns
    -------
    bin_centers : array, shape (n_bins,)
        Center of each angular bin (radians, regardless of input unit).
    firing_rates : array, shape (n_bins,)
        Firing rate in each bin (Hz).

    Raises
    ------
    ValueError
        If inputs are invalid with actionable error message.

    See Also
    --------
    head_direction_metrics : Compute metrics from tuning curve.
    plot_head_direction_tuning : Visualize tuning curve.

    Notes
    -----
    **Algorithm**:

    1. Bin head directions into angular bins
    2. For each spike, find nearest position sample and assign to bin
    3. Compute occupancy (time spent) per bin using actual time deltas
    4. Firing rate = spike count / occupancy
    5. Apply Gaussian smoothing with circular boundary handling

    **Occupancy calculation**: Uses actual inter-sample time intervals
    (``np.diff(position_times)``) rather than assuming uniform sampling.

    **Why this matters**: If your camera drops frames (30 fps nominal but
    sometimes skips), we calculate occupancy from actual timestamps. This
    prevents artificially inflated firing rates in bins where data was sparse.

    **Smoothing**: Uses Gaussian kernel with `mode='wrap'` to handle
    circular boundary. This ensures 0 deg and 360 deg are treated as adjacent.

    Examples
    --------
    >>> from neurospatial.metrics import head_direction_tuning_curve
    >>> # Using degrees (default)
    >>> bins, rates = head_direction_tuning_curve(
    ...     head_directions_deg, spike_times, position_times
    ... )

    >>> # Using radians
    >>> bins, rates = head_direction_tuning_curve(
    ...     head_directions_rad, spike_times, position_times,
    ...     bin_size=np.radians(6), angle_unit='rad'
    ... )
    """
    # Convert to radians if needed
    if angle_unit == "deg":
        head_directions = np.radians(head_directions)
        bin_size_rad = np.radians(bin_size)
    else:
        bin_size_rad = bin_size

    # Validate inputs
    if len(head_directions) != len(position_times):
        raise ValueError(
            f"head_directions and position_times must have same length. "
            f"Got {len(head_directions)} and {len(position_times)}."
        )

    if len(position_times) < 2:
        raise ValueError(
            f"Need at least two position samples to compute head direction tuning curve. "
            f"Got {len(position_times)}."
        )

    if len(spike_times) == 0:
        raise ValueError(
            "No spikes provided. Cannot compute tuning curve.\n"
            "Fix: Check spike detection or use a longer recording."
        )

    # Check for monotonic timestamps
    if not np.all(np.diff(position_times) > 0):
        raise ValueError(
            "position_times must be strictly increasing (monotonic). "
            "Found non-monotonic timestamps.\n"
            "Fix: Sort your data by time or check for duplicate timestamps."
        )

    # Wrap to [0, 2pi]
    head_directions = head_directions % (2 * np.pi)

    # Create bins
    n_bins = int(np.ceil(2 * np.pi / bin_size_rad))
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute occupancy per bin using actual time deltas
    # This handles variable sampling rates correctly
    dt = np.diff(position_times)
    dt = np.concatenate([dt, dt[-1:]])  # Repeat last delta for length match
    hd_bins = np.digitize(head_directions, bin_edges) - 1
    hd_bins = np.clip(hd_bins, 0, n_bins - 1)
    occupancy = np.bincount(hd_bins, weights=dt, minlength=n_bins)

    # Assign spikes to bins
    spike_hd = np.interp(spike_times, position_times, head_directions)
    spike_hd = spike_hd % (2 * np.pi)
    spike_bins = np.digitize(spike_hd, bin_edges) - 1
    spike_bins = np.clip(spike_bins, 0, n_bins - 1)
    spike_counts = np.bincount(spike_bins, minlength=n_bins)

    # Compute firing rates (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(occupancy > 0, spike_counts / occupancy, 0.0)

    # Check for non-finite values (can happen with very small occupancy)
    if not np.all(np.isfinite(firing_rates)):
        raise ValueError(
            "Computed firing rates contain inf or NaN. This may indicate "
            "extremely short occupancy in some bins.\n"
            "Fix: Use larger bin_size or check for timestamp issues."
        )

    # Apply Gaussian smoothing with circular boundary
    if smoothing_window > 0:
        sigma = smoothing_window / 2.355  # Convert FWHM to sigma
        firing_rates = ndimage.gaussian_filter1d(
            firing_rates, sigma=sigma, mode="wrap"
        )

    return bin_centers, firing_rates.astype(np.float64)
```

### 3.2 Head Direction Metrics

```python
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

    Notes
    -----
    **Classification Criteria**:

    A neuron is classified as an HD cell if:

    - Mean vector length > min_vector_length (default 0.4)
    - Rayleigh test p-value < 0.05

    These criteria follow Taube et al. (1990) and subsequent literature.

    Examples
    --------
    >>> metrics = head_direction_metrics(bins, rates)
    >>> if metrics.is_hd_cell:
    ...     print(f"HD cell! Preferred direction: {metrics.preferred_direction_deg:.1f} deg")
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

    def interpretation(self) -> str:
        """
        Human-readable interpretation of head direction metrics.

        Returns
        -------
        str
            Multi-line interpretation.
        """
        lines = []

        if self.is_hd_cell:
            lines.append("*** HEAD DIRECTION CELL ***")
            lines.append(f"Preferred direction: {self.preferred_direction_deg:.1f} deg")
            lines.append(f"Mean vector length: {self.mean_vector_length:.3f} (threshold = 0.4)")
            lines.append(f"Peak firing rate: {self.peak_firing_rate:.1f} Hz")
            lines.append(f"Tuning width (HWHM): {self.tuning_width_deg:.1f} deg")
            lines.append(f"Rayleigh test: p = {self.rayleigh_pval:.4f}")
        else:
            lines.append("Not classified as HD cell")
            if self.mean_vector_length < 0.4:
                lines.append(f"  - Mean vector length too low: {self.mean_vector_length:.3f} < 0.4")
                lines.append("    How was 0.4 chosen? From Taube et al. (1990) analyzing")
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
                lines.append(f"  - Rayleigh test not significant: p = {self.rayleigh_pval:.3f} >= 0.05")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation with interpretation."""
        return self.interpretation()


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
    bin_centers : array, shape (n_bins,)
        Center of each angular bin (radians).
    firing_rates : array, shape (n_bins,)
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
    >>> from neurospatial.metrics import head_direction_tuning_curve, head_direction_metrics
    >>> bins, rates = head_direction_tuning_curve(hd, spikes, times)
    >>> metrics = head_direction_metrics(bins, rates)
    >>> print(metrics)

    References
    ----------
    Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells.
        J Neurosci, 10(2), 420-435.
    """
    if len(bin_centers) != len(firing_rates):
        raise ValueError(
            f"bin_centers and firing_rates must have same length. "
            f"Got {len(bin_centers)} and {len(firing_rates)}."
        )

    if np.sum(firing_rates) == 0:
        raise ValueError(
            "All firing rates are zero. Cannot compute HD metrics.\n"
            "Fix: Check if neuron has any spikes in this recording."
        )

    # Check for constant (non-zero) firing rates
    if np.ptp(firing_rates) == 0:
        raise ValueError(
            "All firing rates are constant. Cannot compute HD metrics.\n"
            "Fix: Neuron shows no directional tuning (uniform firing)."
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
    peak_idx = np.argmax(firing_rates)
    peak_firing_rate = firing_rates[peak_idx]

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
            bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else np.pi / 30
            n_above = np.sum(above_half)
            tuning_width = n_above * bin_width / 2  # HWHM = FWHM / 2
        else:
            tuning_width = np.pi / 2  # Default if can't compute
    else:
        tuning_width = np.nan

    # Rayleigh test on weighted angles
    z, pval = rayleigh_test(bin_centers, weights=firing_rates)

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
    )


def is_head_direction_cell(
    head_directions: NDArray[np.float64],
    spike_times: NDArray[np.float64],
    position_times: NDArray[np.float64],
    **kwargs,
) -> bool:
    """
    Quick check: Is this a head direction cell?

    Convenience function for fast screening.
    For detailed metrics, use `head_direction_tuning_curve` + `head_direction_metrics`.

    Parameters
    ----------
    head_directions, spike_times, position_times : array
        Same as `head_direction_tuning_curve`.
    **kwargs
        Additional arguments passed to `head_direction_tuning_curve`.

    Returns
    -------
    bool
        True if neuron passes HD cell criteria.

    Examples
    --------
    >>> from neurospatial.metrics import is_head_direction_cell
    >>> # Screen many neurons
    >>> for i, (hd, spikes, times) in enumerate(all_neurons):
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
```

### 3.3 Head Direction Tuning Plot

```python
def plot_head_direction_tuning(
    bin_centers: NDArray[np.float64],
    firing_rates: NDArray[np.float64],
    metrics: HeadDirectionMetrics | None = None,
    *,
    ax: "Axes | None" = None,
    projection: Literal["polar", "linear"] = "polar",
    angle_display_unit: Literal["deg", "rad"] = "deg",
    show_metrics: bool = True,
    color: str = "blue",
    fill_alpha: float = 0.3,
    line_kwargs: dict | None = None,
    fill_kwargs: dict | None = None,
) -> "Axes":
    """
    Plot head direction tuning curve.

    Parameters
    ----------
    bin_centers : array, shape (n_bins,)
        Center of each angular bin (radians).
    firing_rates : array, shape (n_bins,)
        Firing rate in each bin (Hz).
    metrics : HeadDirectionMetrics, optional
        If provided, overlay preferred direction and metrics.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        For polar projection, creates polar axes.
    projection : {'polar', 'linear'}, default='polar'
        - 'polar': Polar plot (standard for HD cells)
        - 'linear': Linear plot (easier to read exact values)
    angle_display_unit : {'deg', 'rad'}, default='deg'
        Unit for displaying angles on plot axes.
    show_metrics : bool, default=True
        If True, add text box with key metrics.
    color : str, default='blue'
        Color for tuning curve.
    fill_alpha : float, default=0.3
        Alpha for filled area under curve.
    line_kwargs : dict, optional
        Additional keyword arguments passed to ``ax.plot()``.
    fill_kwargs : dict, optional
        Additional keyword arguments passed to ``ax.fill()`` or ``ax.fill_between()``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from neurospatial.metrics import (
    ...     head_direction_tuning_curve, head_direction_metrics, plot_head_direction_tuning
    ... )
    >>> bins, rates = head_direction_tuning_curve(hd, spikes, times)
    >>> metrics = head_direction_metrics(bins, rates)
    >>> plot_head_direction_tuning(bins, rates, metrics)
    >>> plt.show()

    >>> # Linear view for detailed analysis
    >>> plot_head_direction_tuning(bins, rates, metrics, projection='linear')
    """
    import matplotlib.pyplot as plt

    if ax is None:
        if projection == "polar":
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(10, 4))

    # Set up kwargs with defaults
    line_kw = {"color": color, "linewidth": 2}
    if line_kwargs is not None:
        line_kw.update(line_kwargs)

    fill_kw = {"color": color, "alpha": fill_alpha}
    if fill_kwargs is not None:
        fill_kw.update(fill_kwargs)

    # Convert to display units
    if angle_display_unit == "deg":
        display_centers = np.degrees(bin_centers)
        angle_label = "Head Direction (deg)"
    else:
        display_centers = bin_centers
        angle_label = "Head Direction (rad)"

    if projection == "polar":
        # Close the curve by appending first point
        angles_closed = np.concatenate([bin_centers, [bin_centers[0]]])
        rates_closed = np.concatenate([firing_rates, [firing_rates[0]]])

        ax.plot(angles_closed, rates_closed, **line_kw)
        ax.fill(angles_closed, rates_closed, **fill_kw)

        # Mark preferred direction
        if metrics is not None:
            ax.plot(
                [metrics.preferred_direction],
                [metrics.peak_firing_rate],
                "r^",
                markersize=10,
                label=f"PFD: {metrics.preferred_direction_deg:.1f} deg",
            )
            ax.legend(loc="upper right")

        ax.set_theta_zero_location("N")  # 0 deg at top
        ax.set_theta_direction(-1)  # Clockwise

    else:  # linear
        ax.plot(display_centers, firing_rates, **line_kw)
        ax.fill_between(display_centers, firing_rates, **fill_kw)

        # Mark preferred direction
        if metrics is not None:
            pfd_display = (
                metrics.preferred_direction_deg
                if angle_display_unit == "deg"
                else metrics.preferred_direction
            )
            ax.axvline(pfd_display, color="red", linestyle="--", label="PFD")
            ax.legend()

        ax.set_xlabel(angle_label)
        ax.set_ylabel("Firing Rate (Hz)")

        if angle_display_unit == "deg":
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])

    # Add metrics text box
    if show_metrics and metrics is not None:
        text = (
            f"PFD: {metrics.preferred_direction_deg:.1f} deg\n"
            f"MVL: {metrics.mean_vector_length:.3f}\n"
            f"Peak: {metrics.peak_firing_rate:.1f} Hz"
        )
        if metrics.is_hd_cell:
            text = "HD CELL\n" + text

        if projection == "polar":
            ax.text(
                0.02, 0.98, text,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        else:
            ax.text(
                0.98, 0.98, text,
                transform=ax.transAxes,
                fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    return ax
```

---

## Part 4: Integration and Exports

### 4.1 Update `metrics/__init__.py`

```python
from neurospatial.metrics.circular import (
    rayleigh_test,
    circular_linear_correlation,
    circular_circular_correlation,
    phase_precession,
    phase_position_correlation,
    plot_phase_precession,
    PhasePrecessionResult,
    has_phase_precession,
)
from neurospatial.metrics.head_direction import (
    head_direction_tuning_curve,
    head_direction_metrics,
    HeadDirectionMetrics,
    plot_head_direction_tuning,
    is_head_direction_cell,
)

__all__ = [
    # ... existing exports ...
    # Circular statistics
    "rayleigh_test",
    "circular_linear_correlation",
    "circular_circular_correlation",
    "phase_precession",
    "phase_position_correlation",
    "plot_phase_precession",
    "PhasePrecessionResult",
    "has_phase_precession",
    # Head direction
    "head_direction_tuning_curve",
    "head_direction_metrics",
    "HeadDirectionMetrics",
    "plot_head_direction_tuning",
    "is_head_direction_cell",
]
```

---

## Implementation Order

1. **circular.py - Core functions** (no dependencies)
   - `_to_radians()` (internal helper for angle_unit conversion)
   - `_mean_resultant_length()` (internal helper, uses scipy if available)
   - `_validate_circular_input()`, `_validate_paired_input()`
   - `rayleigh_test()`
   - `circular_linear_correlation()`, `phase_position_correlation()`
   - `circular_circular_correlation()`

2. **circular.py - Phase precession** (depends on core)
   - `PhasePrecessionResult` dataclass
   - `phase_precession()`
   - `has_phase_precession()`
   - `plot_phase_precession()`

3. **head_direction.py** (depends on circular.py)
   - `head_direction_tuning_curve()`
   - `HeadDirectionMetrics` dataclass
   - `head_direction_metrics()`
   - `is_head_direction_cell()`
   - `plot_head_direction_tuning()`

4. **Tests**
   - `tests/metrics/test_circular.py`
   - `tests/metrics/test_head_direction.py`

5. **Update exports**
   - `metrics/__init__.py`

---

## Test Strategy

### Anchor Tests with Known Values

1. **Rayleigh test**:
   - Uniform distribution: p > 0.5 (high p-value, no preferred direction)
   - Von Mises with high concentration: p < 0.001 (significant directionality)
   - Known R values from literature examples

2. **Circular-linear correlation**:
   - Perfect linear phase-position: r -> 1.0
   - Random phases with positions: r -> 0
   - Synthetic precession with known slope

3. **Phase precession**:
   - Synthetic data with known slope: recovered slope within tolerance
   - No precession (random phases): slope near zero, high p-value

4. **HD tuning curve**:
   - Synthetic HD cell with Gaussian tuning: correct preferred direction
   - Uniform firing: low MVL, not classified as HD cell

### Synthetic Slope Recovery Test

```python
def test_phase_precession_recovers_known_slope():
    """Verify phase_precession recovers slope and offset from synthetic data."""
    rng = np.random.default_rng(42)

    # Known parameters
    true_slope = -1.5  # rad/cm
    true_offset = np.pi / 2
    noise_level = 0.3

    # Generate synthetic data
    n_spikes = 200
    positions = rng.uniform(0, 40, n_spikes)  # 40 cm place field
    phases = (true_slope * positions + true_offset) % (2 * np.pi)
    phases += rng.normal(0, noise_level, n_spikes)
    phases = phases % (2 * np.pi)

    # Analyze
    result = phase_precession(positions, phases)

    # Check recovery within tolerance
    # Slope should be within 20% of true value
    np.testing.assert_allclose(result.slope, true_slope, rtol=0.2)

    # Offset should be within 0.5 rad
    # Note: offset comparison needs to handle circular wrapping
    offset_diff = np.abs(result.offset - true_offset)
    offset_diff = min(offset_diff, 2 * np.pi - offset_diff)
    assert offset_diff < 0.5, f"Offset difference {offset_diff} > 0.5"

    # Should be significant
    assert result.is_significant()
    assert result.correlation > 0.3


def test_phase_precession_no_relationship():
    """Verify phase_precession detects no relationship in random data."""
    rng = np.random.default_rng(42)

    n_spikes = 200
    positions = rng.uniform(0, 40, n_spikes)
    phases = rng.uniform(0, 2 * np.pi, n_spikes)

    result = phase_precession(positions, phases)

    # Should not be significant
    assert result.pval > 0.05
    assert result.correlation < 0.2
```

### Scipy Fallback Test

```python
def test_mean_resultant_length_fallback(monkeypatch):
    """Verify fallback computation matches scipy when available."""
    from neurospatial.metrics import circular

    # Test data
    rng = np.random.default_rng(42)
    angles = rng.vonmises(np.pi, 2.0, 100)

    # Get result with scipy (if available)
    if circular._HAS_DIRECTIONAL_STATS:
        result_scipy = circular._mean_resultant_length(angles)

        # Monkeypatch to force fallback
        monkeypatch.setattr(circular, "_HAS_DIRECTIONAL_STATS", False)

        # Get result with fallback
        result_fallback = circular._mean_resultant_length(angles)

        # Should match within floating point tolerance
        np.testing.assert_allclose(result_scipy, result_fallback, rtol=1e-10)
    else:
        # scipy not available, just verify fallback works
        result = circular._mean_resultant_length(angles)
        assert 0 <= result <= 1


def test_rayleigh_fallback_path(monkeypatch):
    """Verify Rayleigh test works with fallback MRL computation."""
    from neurospatial.metrics import circular

    rng = np.random.default_rng(42)
    angles = rng.vonmises(np.pi, 2.0, 100)

    # Get result with scipy
    z1, p1 = circular.rayleigh_test(angles)

    # Force fallback
    monkeypatch.setattr(circular, "_HAS_DIRECTIONAL_STATS", False)

    # Get result with fallback
    z2, p2 = circular.rayleigh_test(angles)

    # Should match
    np.testing.assert_allclose(z1, z2, rtol=1e-10)
    np.testing.assert_allclose(p1, p2, rtol=1e-10)
```

### Validation Against Reference Implementations

Add dev dependency for validation (not production):

```toml
[tool.uv.dev-dependencies]
pycircstat2 = "^0.1.0"  # For validation tests only
```

Test functions:

```python
@pytest.mark.parametrize("n_samples", [50, 100, 500])
def test_rayleigh_vs_pycircstat(n_samples):
    """Compare rayleigh_test against pycircstat2."""
    pytest.importorskip("pycircstat2")
    from pycircstat2 import rayleigh

    rng = np.random.default_rng(42)
    angles = rng.vonmises(np.pi, 2.0, n_samples)
    z_ours, p_ours = rayleigh_test(angles)
    p_ref, z_ref = rayleigh(angles)

    np.testing.assert_allclose(z_ours, z_ref, rtol=0.01)
    np.testing.assert_allclose(p_ours, p_ref, rtol=0.01)
```

### Edge Cases

- Empty arrays -> ValueError with message
- All same angle -> R = 1.0, p -> 0
- All NaN -> ValueError with message
- Mixed NaN -> Warning, proceed with valid data
- Angles outside [0, 2pi] -> Warning, wrap automatically
- Mismatched array lengths -> ValueError with message
- Constant (non-zero) firing rates -> ValueError with message
- Non-monotonic timestamps -> ValueError with message
- Insufficient position samples -> ValueError with message

### Property-Based Tests (using hypothesis)

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(0, 2*np.pi), min_size=10, max_size=1000))
def test_rayleigh_r_bounds(angles):
    """R should always be in [0, 1]."""
    z, p = rayleigh_test(np.array(angles))
    n = len(angles)
    r = np.sqrt(z / n)
    assert 0 <= r <= 1


@given(
    st.lists(st.floats(0, 2*np.pi), min_size=10, max_size=100),
    st.lists(st.floats(0, 2*np.pi), min_size=10, max_size=100),
)
def test_circular_correlation_symmetric(a1, a2):
    """Correlation should be symmetric."""
    a1, a2 = np.array(a1), np.array(a2)
    n = min(len(a1), len(a2))
    r1, _ = circular_circular_correlation(a1[:n], a2[:n])
    r2, _ = circular_circular_correlation(a2[:n], a1[:n])
    np.testing.assert_allclose(r1, r2, rtol=1e-10)
```

---

## Dependencies

**Uses existing neurospatial dependencies only**:

- numpy
- scipy (stats, optimize, ndimage)
- matplotlib (for plotting functions)

**No new dependencies required.**

**Development dependencies for validation tests**:

- pycircstat2 (optional, dev-only)
- hypothesis (already in dev dependencies)

---

## Computational Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| `rayleigh_test` | O(n) | Single pass over data |
| `circular_linear_correlation` | O(n) | Three Pearson correlations |
| `circular_circular_correlation` | O(n) | Two passes (mean, correlation) |
| `phase_precession` | O(n * k) | k = optimization iterations (~50-100) |
| `head_direction_tuning_curve` | O(n + m) | n = samples, m = spikes |
| `head_direction_metrics` | O(b) | b = number of bins (~60) |

---

## References

### Primary Sources

- **Jammalamadaka, S.R. & SenGupta, A. (2001)**. Topics in Circular Statistics. World Scientific.
  - Foundation for Rayleigh test, circular-circular correlation (p. 176)
- **Kempter, R., Leibold, C., Buzsaki, G., Diba, K., & Schmidt, R. (2012)**. Quantifying circular-linear associations: Hippocampal phase precession. J Neurosci Methods, 207(1), 113-124.
  - Phase precession algorithm
- **Mardia, K.V. & Jupp, P.E. (2000)**. Directional Statistics. Wiley.
  - Circular-linear correlation (Chapter 11.2, pp. 244-245), Rayleigh test (Section 5.3.2, p. 94)
- **Taube, J.S., Muller, R.U., & Ranck, J.B. (1990)**. Head-direction cells. J Neurosci, 10(2), 420-435.
  - HD cell classification criteria
- **O'Keefe, J. & Recce, M.L. (1993)**. Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
  - Phase precession discovery, doubled-phase plotting convention

### Implementation References

- [CircStats R package](https://cran.r-project.org/web/packages/CircStats/) - `r.test`, `circ.cor`
- [CINPLA/phase-precession](https://github.com/CINPLA/phase-precession) - `cl_corr` (circular-linear correlation with slope)
- [pycircstat](https://github.com/circstat/pycircstat) - `corrcc`, `corrcl`, `mean`, `resultant_vector_length`
- [Astropy circstats](https://docs.astropy.org/en/stable/stats/circ.html) - `rayleightest`
- [pingouin circular](https://pingouin-stats.org/build/html/generated/pingouin.circ_corrcl.html) - `circ_corrcl`

### Scipy Functions Used

- [scipy.stats.circmean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html)
- [scipy.stats.directional_stats](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.directional_stats.html) (scipy >= 1.9.0)
- [scipy.stats.pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
- [scipy.stats.chi2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
- [scipy.stats.norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
- [scipy.optimize.fminbound](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fminbound.html)
- [scipy.ndimage.gaussian_filter1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html)
