"""
Core circular statistics functions.

This module provides fundamental circular statistics functions for analyzing
directional data common in neuroscience: spike phase relative to LFP, head
direction, and other angular measurements.

Which Function Should I Use?
----------------------------
**Test for preferred direction?**
    Use ``rayleigh_test()`` to test if angles are uniformly distributed.

**Correlate angle with position/time?**
    Use ``circular_linear_correlation()`` for angular-linear correlation.
    ``phase_position_correlation()`` is an alias with neuroscience naming.

**Correlate two angular variables?**
    Use ``circular_circular_correlation()`` (e.g., phase coherence).

**Phase precession analysis?**
    See ``neurospatial.metrics.phase_precession`` module for
    ``phase_precession()``, ``has_phase_precession()``, and
    ``plot_phase_precession()``.

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
**Test for non-uniformity (e.g., preferred theta phase)**:

>>> z, p = rayleigh_test(angles)  # doctest: +SKIP

**Correlation strength (no slope needed)**:

>>> # Returns correlation coefficient and p-value
>>> r, p = phase_position_correlation(phases, positions)  # doctest: +SKIP
>>> if p < 0.05 and r > 0.2:  # doctest: +SKIP
...     print(f"Significant correlation: r={r:.3f}")

**Circular-circular correlation (e.g., phase coherence between electrodes)**:

>>> r, p = circular_circular_correlation(angles1, angles2)  # doctest: +SKIP

References
----------
Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
    World Scientific.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

__all__ = [
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "circular_circular_correlation",
    "circular_linear_correlation",
    "phase_position_correlation",
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
# Circular Basis Functions for GLM Design Matrices
# =============================================================================


@dataclass
class CircularBasisResult:
    """
    Result from circular_basis() function.

    Contains sin/cos components for use in GLM design matrices, plus the
    original angles for reference.

    Attributes
    ----------
    sin_component : ndarray, shape (n_samples,)
        Sine of angles: sin(angles).
    cos_component : ndarray, shape (n_samples,)
        Cosine of angles: cos(angles).
    angles : ndarray, shape (n_samples,)
        Original angles (in radians).

    Properties
    ----------
    design_matrix : ndarray, shape (n_samples, 2)
        Design matrix with columns [sin_component, cos_component].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_basis
    >>> angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> result = circular_basis(angles)
    >>> result.design_matrix.shape
    (100, 2)
    """

    sin_component: NDArray[np.float64]
    cos_component: NDArray[np.float64]
    angles: NDArray[np.float64]

    @property
    def design_matrix(self) -> NDArray[np.float64]:
        """
        Return design matrix with columns [sin_component, cos_component].

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Design matrix suitable for GLM.
        """
        return np.column_stack([self.sin_component, self.cos_component])


def circular_basis(
    angles: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> CircularBasisResult:
    """
    Compute sine/cosine basis functions for circular variables.

    Creates a design matrix for use in GLMs (Generalized Linear Models) where
    the sin and cos components capture circular modulation. This is the standard
    approach for including circular predictors in regression models.

    Parameters
    ----------
    angles : array, shape (n_samples,)
        Circular variable (e.g., head direction, LFP phase).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    CircularBasisResult
        Result object containing:
        - sin_component: sin(angles)
        - cos_component: cos(angles)
        - angles: original angles (in radians)
        - design_matrix property: (n_samples, 2) array for GLM

    See Also
    --------
    circular_basis_metrics : Compute amplitude/phase from GLM coefficients.

    Notes
    -----
    **Why sin/cos basis?**

    For a circular predictor theta, using sin(theta) and cos(theta) as separate
    predictors allows the model to capture any phase and amplitude of circular
    modulation. The fitted coefficients (beta_sin, beta_cos) can be converted to:

    - Amplitude: sqrt(beta_sin^2 + beta_cos^2)
    - Phase: atan2(beta_sin, beta_cos)

    **GLM Workflow**:

    1. Create design matrix: ``X = circular_basis(angles).design_matrix``
    2. Fit GLM: ``model.fit(X, y)``
    3. Get metrics: ``amplitude, phase, pval = circular_basis_metrics(
           beta_sin, beta_cos, cov_matrix)``

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_basis
    >>> # Create basis for head direction
    >>> angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> result = circular_basis(angles)
    >>> result.design_matrix.shape
    (100, 2)

    >>> # Use with statsmodels GLM
    >>> # X = result.design_matrix
    >>> # model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
    >>> # fit = model.fit()
    >>> # beta_sin, beta_cos = fit.params[1:3]
    """
    # Convert to radians if needed
    angles = np.asarray(angles, dtype=np.float64).ravel()
    angles_rad = _to_radians(angles, angle_unit)

    # Compute sin/cos components
    sin_component = np.sin(angles_rad)
    cos_component = np.cos(angles_rad)

    return CircularBasisResult(
        sin_component=sin_component,
        cos_component=cos_component,
        angles=angles_rad,
    )


def _wald_test_magnitude(
    beta_sin: float,
    beta_cos: float,
    cov_matrix: NDArray[np.float64],
) -> float:
    """
    Wald test for significance of circular modulation magnitude.

    Tests H0: amplitude = sqrt(beta_sin^2 + beta_cos^2) = 0.

    Parameters
    ----------
    beta_sin : float
        Coefficient for sin component from GLM.
    beta_cos : float
        Coefficient for cos component from GLM.
    cov_matrix : ndarray, shape (2, 2)
        Covariance matrix for [beta_sin, beta_cos].

    Returns
    -------
    float
        P-value from chi-squared distribution with 2 df.

    Notes
    -----
    Uses the Wald test statistic: W = beta.T @ inv(cov) @ beta ~ chi2(2)
    where beta = [beta_sin, beta_cos].
    """
    beta = np.array([beta_sin, beta_cos])

    # Wald statistic: beta.T @ inv(cov) @ beta
    # Use solve instead of inv for numerical stability
    try:
        wald_stat = float(beta @ np.linalg.solve(cov_matrix, beta))
    except np.linalg.LinAlgError:
        # Singular covariance matrix
        warnings.warn(
            "Covariance matrix is singular. Cannot compute p-value.",
            stacklevel=3,
        )
        return np.nan

    # P-value from chi-squared with 2 df
    pval = float(1.0 - chi2.cdf(wald_stat, df=2))

    return float(np.clip(pval, 0.0, 1.0))


def circular_basis_metrics(
    beta_sin: float,
    beta_cos: float,
    cov_matrix: NDArray[np.float64] | None = None,
) -> tuple[float, float, float | None]:
    """
    Compute amplitude, phase, and p-value from GLM coefficients.

    Given fitted coefficients for sin and cos basis functions, compute:
    - Amplitude (strength of modulation)
    - Phase (preferred angle)
    - P-value (statistical significance via Wald test)

    Parameters
    ----------
    beta_sin : float
        Coefficient for sin(angle) from fitted GLM.
    beta_cos : float
        Coefficient for cos(angle) from fitted GLM.
    cov_matrix : ndarray, shape (2, 2), optional
        Covariance matrix for [beta_sin, beta_cos] from GLM fit.
        If provided, computes p-value via Wald test.

    Returns
    -------
    amplitude : float
        Modulation amplitude: sqrt(beta_sin^2 + beta_cos^2).
        Larger values indicate stronger circular modulation.
    phase : float
        Preferred phase angle in radians, range [-pi, pi].
        Computed as atan2(beta_sin, beta_cos).
    pvalue : float or None
        P-value testing H0: amplitude = 0 (no modulation).
        None if cov_matrix not provided.

    See Also
    --------
    circular_basis : Create design matrix for GLM.

    Notes
    -----
    **Interpretation**:

    - amplitude > 0.3 typically indicates meaningful modulation
    - pvalue < 0.05 indicates statistically significant modulation
    - phase tells you the preferred angle (e.g., preferred head direction)

    **Standard Errors**:

    The p-value is computed using the Wald test, which tests whether the
    joint effect of (beta_sin, beta_cos) is significantly different from zero.
    This is preferred over testing each coefficient separately.

    Examples
    --------
    >>> from neurospatial.metrics import circular_basis_metrics
    >>> # After fitting GLM with circular basis
    >>> beta_sin, beta_cos = 0.5, 0.3
    >>> cov = np.array([[0.01, 0.001], [0.001, 0.01]])
    >>> amplitude, phase, pval = circular_basis_metrics(beta_sin, beta_cos, cov)
    >>> print(f"Amplitude: {amplitude:.2f}, Phase: {np.degrees(phase):.1f}°")
    Amplitude: 0.58, Phase: 59.0°
    """
    # Compute amplitude: sqrt(beta_sin^2 + beta_cos^2)
    amplitude = float(np.sqrt(beta_sin**2 + beta_cos**2))

    # Compute phase: atan2(beta_sin, beta_cos)
    phase = float(np.arctan2(beta_sin, beta_cos))

    # Compute p-value if covariance provided
    pvalue: float | None = None
    if cov_matrix is not None:
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        if cov_matrix.shape != (2, 2):
            raise ValueError(f"cov_matrix must be shape (2, 2), got {cov_matrix.shape}")
        pvalue = _wald_test_magnitude(beta_sin, beta_cos, cov_matrix)

    return amplitude, phase, pvalue


# =============================================================================
# Public API - Core Circular Statistics
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

    Notes
    -----
    The circular-linear correlation is computed using the formula from
    Mardia & Jupp (2000, Section 11.3):

        r^2 = (r_xs^2 + r_xc^2 - 2*r_xs*r_xc*r_cs) / (1 - r_cs^2)

    where:
    - r_xs: Pearson correlation between linear variable x and sin(angles)
    - r_xc: Pearson correlation between linear variable x and cos(angles)
    - r_cs: Pearson correlation between cos(angles) and sin(angles)

    The result is always non-negative because r is the square root of r^2.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_linear_correlation
    >>> positions = np.linspace(0, 100, 50)
    >>> phases = np.linspace(0, 2 * np.pi, 50)  # Linear relationship
    >>> r, p = circular_linear_correlation(phases, positions)
    >>> r > 0.5 and p < 0.05  # Significant correlation
    True
    """
    from scipy.stats import chi2, pearsonr

    # Convert to arrays and radians if needed
    angles = np.asarray(angles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    angles = _to_radians(angles, angle_unit)

    # Validate paired inputs (handles length mismatch, NaN removal)
    angles, values = _validate_paired_input(
        angles, values, "angles", "values", min_samples=3
    )

    n = len(angles)

    # Compute sin and cos of angles
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    # Check for degenerate case: no variation in linear variable
    values_std = np.std(values)
    if values_std < 1e-10:
        warnings.warn(
            "Linear variable has no variation (constant values). "
            "Circular-linear correlation is undefined. Returning r=0.",
            stacklevel=2,
        )
        return 0.0, 1.0

    # Compute Pearson correlations
    # r_xs: correlation between x (linear) and sin(angles)
    # r_xc: correlation between x (linear) and cos(angles)
    # r_cs: correlation between cos(angles) and sin(angles)
    r_xs, _ = pearsonr(values, sin_angles)
    r_xc, _ = pearsonr(values, cos_angles)
    r_cs, _ = pearsonr(cos_angles, sin_angles)

    # Handle degenerate case: r_cs near 1 (cos and sin perfectly correlated)
    # This shouldn't happen for real circular data but handle it anyway
    if np.abs(r_cs) > 0.9999:
        warnings.warn(
            "Degenerate case: cos(angles) and sin(angles) are nearly perfectly "
            "correlated. This suggests angles have very limited range. "
            "Returning r=0.",
            stacklevel=2,
        )
        return 0.0, 1.0

    # Mardia & Jupp formula for circular-linear correlation
    # r^2 = (r_xs^2 + r_xc^2 - 2*r_xs*r_xc*r_cs) / (1 - r_cs^2)
    r_squared = (r_xs**2 + r_xc**2 - 2 * r_xs * r_xc * r_cs) / (1 - r_cs**2)

    # Ensure r_squared is non-negative (numerical issues can make it slightly negative)
    r_squared = max(0.0, r_squared)

    # r is always non-negative by definition
    r = float(np.sqrt(r_squared))

    # P-value from chi-squared distribution with 2 degrees of freedom
    # Test statistic: n * r^2 follows chi-squared(2) under null hypothesis
    chi2_stat = n * r_squared
    pval = float(1.0 - chi2.cdf(chi2_stat, df=2))

    # Ensure p-value is in valid range
    pval = float(np.clip(pval, 0.0, 1.0))

    return r, pval


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

    Computes the Fisher & Lee (1983) circular correlation coefficient
    between two circular variables.

    Parameters
    ----------
    angles1, angles2 : array, shape (n,)
        Paired circular measurements.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles (applies to both arrays).

    Returns
    -------
    r : float
        Circular correlation coefficient in [-1, 1].
        - r > 0: positive association (angles increase together)
        - r < 0: negative association (angles move in opposite directions)
        - r = 0: no circular correlation
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
    **Formula** (Fisher & Lee, 1983; Jammalamadaka & SenGupta, 2001, p. 176):

        rho = sum(sin(a1 - mean1) * sin(a2 - mean2)) /
              sqrt(sum(sin(a1 - mean1)^2) * sum(sin(a2 - mean2)^2))

    **Interpretation**:

    - |r| > 0.3: Moderate association
    - |r| > 0.5: Strong association

    **Important Properties**:

    - Symmetric: r(a1, a2) == r(a2, a1)
    - Invariant to constant offsets: r(a, a + c) = 1.0 for any constant c
      (because deviations from circular means remain identical)
    - Anticorrelation requires reflection: r(a, -a) approaches -1.0

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import circular_circular_correlation
    >>> angles1 = np.random.default_rng(42).uniform(0, 2 * np.pi, 100)
    >>> angles2 = angles1 + 0.1 * np.random.default_rng(42).standard_normal(100)
    >>> r, p = circular_circular_correlation(angles1, angles2)
    >>> r > 0.8
    True

    References
    ----------
    Fisher, N.I. & Lee, A.J. (1983). A correlation coefficient for circular data.
        Biometrika, 70(2), 327-332.
    Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
        World Scientific, p. 176.
    """
    from scipy import stats

    # Convert to radians if needed
    angles1 = np.asarray(angles1, dtype=np.float64)
    angles2 = np.asarray(angles2, dtype=np.float64)
    angles1 = _to_radians(angles1, angle_unit)
    angles2 = _to_radians(angles2, angle_unit)

    # Validate paired inputs (handles NaN removal and length check)
    angles1, angles2 = _validate_paired_input(
        angles1, angles2, "angles1", "angles2", min_samples=5
    )

    n = len(angles1)

    # Compute circular means using scipy
    alpha_bar = float(stats.circmean(angles1, high=2 * np.pi, low=0))
    beta_bar = float(stats.circmean(angles2, high=2 * np.pi, low=0))

    # Compute deviations from circular mean
    sin_dev1 = np.sin(angles1 - alpha_bar)
    sin_dev2 = np.sin(angles2 - beta_bar)

    # Compute correlation coefficient (Fisher & Lee formula)
    num = np.sum(sin_dev1 * sin_dev2)
    den = np.sqrt(np.sum(sin_dev1**2) * np.sum(sin_dev2**2))

    if den == 0:
        # Degenerate case: no variation in one or both variables
        warnings.warn(
            "No variation in one or both circular variables. "
            "This may indicate constant angles. Returning r=0.",
            stacklevel=2,
        )
        return 0.0, 1.0

    rho = num / den

    # Clip to valid range (numerical stability)
    rho = float(np.clip(rho, -1.0, 1.0))

    # Significance test (normal approximation)
    # From Jammalamadaka & SenGupta (2001), p. 177
    l20 = np.mean(sin_dev1**2)
    l02 = np.mean(sin_dev2**2)
    l22 = np.mean(sin_dev1**2 * sin_dev2**2)

    if l22 == 0:
        warnings.warn(
            "Cannot compute p-value: variance product is zero. Returning p=1.0.",
            stacklevel=2,
        )
        return rho, 1.0

    # Test statistic follows standard normal under null hypothesis
    ts = np.sqrt((n * l20 * l02) / l22) * rho
    pval = float(2 * (1 - stats.norm.cdf(np.abs(ts))))

    # Ensure p-value is in valid range
    pval = float(np.clip(pval, 0.0, 1.0))

    return rho, pval
