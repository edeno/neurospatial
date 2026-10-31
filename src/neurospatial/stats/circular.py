"""
Circular statistics functions.

This module provides circular statistics functions for analyzing directional
data common in neuroscience: spike phase relative to LFP, head direction,
and other angular measurements.

Imports
-------
>>> from neurospatial.stats.circular import rayleigh_test
>>> from neurospatial.stats import circular_mean, mean_resultant_length

Which Function Should I Use?
----------------------------
**Test for preferred direction?**
    Use ``rayleigh_test()`` to test if angles are uniformly distributed.

**Correlate angle with position/time?**
    Use ``circular_linear_correlation()`` for angular-linear correlation.
    ``phase_position_correlation()`` is an alias with neuroscience naming.

**Correlate two angular variables?**
    Use ``circular_circular_correlation()`` (e.g., phase coherence).

**Compute circular mean/variance?**
    Use ``circular_mean()``, ``circular_variance()``, ``mean_resultant_length()``.

**Wrap angles to a standard range?**
    Use ``wrap_angle()`` to wrap angles to (-π, π].

**Phase precession analysis?**
    See ``neurospatial.encoding.phase_precession`` module for
    ``compute_phase_precession()``, ``has_phase_precession()``, and
    ``plot_phase_precession()``.

**GLM-based circular regression?**
    Use ``circular_basis()`` to create design matrix [sin(φ), cos(φ)].
    Fit with statsmodels/scikit-learn, then interpret with ``circular_basis_metrics()``.
    Visualize with ``plot_circular_basis_tuning()``.

    Works for any circular variable:

    - **Head direction** tuning (GLM alternative to binned tuning curves)
    - **Theta phase** modulation (spike-field coherence, phase precession)
    - **Running direction** preference
    - **Time-of-day** effects (circadian rhythms)

Angle Units
-----------
Functions that take angles accept an ``angle_unit`` parameter: ``'rad'``
(default) or ``'deg'``; any other value raises. Computations use radians, and
output angles are in radians, the package-wide convention. If your data is in
degrees, set ``angle_unit='deg'``.

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

**GLM circular tuning (head direction or theta phase)**::

    # 1. Create design matrix from the heading in every time bin (not only
    #    at spikes): headings and spike_counts have shape (n_time_bins,)
    result = circular_basis(headings)  # or: circular_basis(theta_phases)
    X = result.design_matrix  # shape (n_time_bins, 2): [sin, cos]

    # 2. Fit a Poisson GLM (statsmodels example); the log(dt) offset makes
    #    exp(intercept) a rate in Hz
    import statsmodels.api as sm

    X_with_intercept = sm.add_constant(X)
    model = sm.GLM(
        spike_counts,
        X_with_intercept,
        family=sm.families.Poisson(),
        offset=np.full(len(spike_counts), np.log(dt)),
    ).fit()

    # 3. Extract and interpret coefficients
    beta_sin, beta_cos = model.params[1], model.params[2]
    cov = model.cov_params()[1:3, 1:3]
    amplitude, phase, pval = circular_basis_metrics(beta_sin, beta_cos, cov)

    # 4. Quick significance check
    if is_modulated(beta_sin, beta_cos, cov):
        print(f"Significant modulation: {amplitude:.2f} at {np.degrees(phase):.0f}°")

    # 5. Visualize
    plot_circular_basis_tuning(beta_sin, beta_cos, intercept=model.params[0])

References
----------
Mardia, K.V. & Jupp, P.E. (2000). Directional Statistics. Wiley.
Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
    World Scientific.
Kramer, M.A. & Eden, U.T. (2016). Case Studies in Neural Data Analysis. MIT Press.
    Chapter 11: Point Process Models of Spike-Field Coherence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import to_radians, wrap_angle
from neurospatial._results import ReadOnlyArraysMixin
from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

__all__ = [  # noqa: RUF022
    # Tests and correlations
    "rayleigh_test",
    "circular_linear_correlation",
    "circular_circular_correlation",
    "phase_position_correlation",
    # Descriptive statistics
    "circular_mean",
    "circular_variance",
    "mean_resultant_length",
    # Angle utilities
    "wrap_angle",
    # GLM circular basis
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "is_modulated",
    "plot_circular_basis_tuning",
]

# =============================================================================
# Internal Helper Functions
# =============================================================================


def _validate_weights(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    name: str = "weights",
) -> NDArray[np.float64]:
    """
    Validate per-angle weights for weighted circular statistics.

    Weights must be the same length as ``angles`` (a length-1 array is a
    mismatch, never broadcast), finite (no NaN/Inf), and non-negative. This
    guards the weighted resultant-length / Rayleigh paths against silently
    producing out-of-range or ``nan`` statistics (e.g. a NaN/Inf weight
    flowing into ``np.sum(weights)``).

    Parameters
    ----------
    angles : ndarray of shape (n,)
        Angles the weights apply to (already raveled to 1-D).
    weights : ndarray of shape (n,)
        Per-angle weights, interpreted as counts/frequencies.
    name : str, default='weights'
        Argument name for error messages.

    Returns
    -------
    ndarray of shape (n,)
        Validated weights as a float64 array.

    Raises
    ------
    ValueError
        If lengths differ, any weight is non-finite (NaN/Inf), or any weight
        is negative.
    """
    from neurospatial._validation import validate_finite, validate_lengths

    weights = np.asarray(weights, dtype=np.float64).ravel()
    validate_lengths({"angles": angles, name: weights})
    # Reject non-finite weights (NaN/Inf) before they flow into sum(weights)
    # and silently turn weighted statistics into nan.
    weights = validate_finite(weights, name=name)
    if np.any(weights < 0):
        n_neg = int(np.sum(weights < 0))
        first = int(np.argmax(weights < 0))
        raise ValueError(
            f"{name} must be non-negative (interpreted as counts/frequencies). "
            f"Found {n_neg} negative value(s) (first at index {first}: "
            f"{weights[first]!r}). "
            f"Fix: pass non-negative weights such as spike counts, not signed "
            f"quantities."
        )
    return weights


def _mean_vector(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """
    Weighted mean of (cos, sin) over validated angles.

    Parameters
    ----------
    angles : ndarray of shape (n,)
        Angles in radians, without NaN.
    weights : ndarray of shape (n,), optional
        Validated non-negative weights. If None, uniform weights are used.

    Returns
    -------
    mean_cos, mean_sin : float
        Both NaN when no angle carries weight (including empty input).
    """
    if weights is None:
        if len(angles) == 0:
            return np.nan, np.nan
        return float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))
    total = np.sum(weights)
    if total == 0:
        return np.nan, np.nan
    normalized = weights / total
    return (
        float(np.sum(normalized * np.cos(angles))),
        float(np.sum(normalized * np.sin(angles))),
    )


def _mean_resultant_length(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute mean resultant length (Rayleigh vector length).

    Parameters
    ----------
    angles : ndarray of shape (n,)
        Angles in radians, without NaN.
    weights : ndarray of shape (n,), optional
        Validated non-negative weights. If None, uniform weights are used.

    Returns
    -------
    float
        Mean resultant length R in [0, 1]; NaN when no angle carries weight.

    Notes
    -----
    The mean resultant length is defined as:

        R = |mean(exp(i * angles))| = |sum(w * exp(i * angles))| / sum(w)

    where w are the weights (uniform if not specified).
    """
    mean_cos, mean_sin = _mean_vector(angles, weights)
    return float(np.sqrt(mean_cos**2 + mean_sin**2))


def _valid_angles(
    angles: NDArray[np.float64],
    angle_unit: Literal["rad", "deg"],
    weights: NDArray[np.float64] | None = None,
    *,
    min_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """
    Angles in radians with NaN angles, and their weights, dropped.

    Parameters
    ----------
    angles : ndarray of shape (n,)
        Input angles in ``angle_unit``; raveled to 1-D.
    angle_unit : {'rad', 'deg'}
        Unit of ``angles``.
    weights : ndarray of shape (n,), optional
        Per-angle weights, validated (equal length, finite, non-negative)
        and co-filtered by the NaN mask of ``angles``.
    min_samples : int
        Fewest angles that must remain. With 0, empty or all-NaN input
        returns no angles, so descriptive statistics come out NaN.

    Returns
    -------
    angles : ndarray of shape (m,)
        Angles in radians without NaN, where m <= n.
    weights : ndarray of shape (m,) or None
        The weights of the remaining angles, or None if none were passed.

    Raises
    ------
    ValueError
        If ``angle_unit`` is unknown, an angle is infinite, every angle is
        NaN while ``min_samples > 0``, or fewer than ``min_samples`` remain.
    """
    angles = to_radians(angles, angle_unit).ravel()
    if weights is not None:
        weights = _validate_weights(angles, weights)

    nan_mask = np.isnan(angles)
    n_nan = int(np.sum(nan_mask))
    if n_nan and n_nan == len(angles) and min_samples > 0:
        raise ValueError(
            f"All angles are NaN ({len(angles)} samples). "
            f"Cannot compute circular statistics.\n"
            f"\n"
            f"Diagnostic steps:\n"
            f"1. Check your data: print(angles[:10]) to see actual values\n"
            f"2. If all zeros: spike detection may have failed\n"
            f"3. If all identical: phase extraction may be broken\n"
            f"4. Check data types: angles.dtype should be float, not object\n"
            f"\n"
            f"Common causes:\n"
            f"  - Wrong variable passed (e.g., spike times instead of phases)\n"
            f"  - Array indexing error (e.g., phases[spike_indices] where "
            f"indices are all -1)\n"
            f"  - Phase computation failed (check LFP signal quality)\n"
            f"\n"
            f"Quick fix: Verify extraction with: np.isnan(angles).sum()"
        )
    if n_nan:
        angles = angles[~nan_mask]
        if weights is not None:
            weights = weights[~nan_mask]
        if len(angles):
            warnings.warn(
                f"Removed {n_nan} NaN values from angles. "
                f"Proceeding with {len(angles)} valid samples.",
                category=UserWarning,
                stacklevel=_external_stacklevel(),
            )

    if np.any(np.isinf(angles)):
        n_inf = np.sum(np.isinf(angles))
        raise ValueError(
            f"angles contains {n_inf} infinite values. "
            f"Cannot compute circular statistics.\n"
            f"Fix: Remove or replace infinite values before calling this function."
        )

    if len(angles) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples for circular statistics. "
            f"Got {len(angles)} valid samples in angles.\n"
            f"Fix: Provide more data points or use a different analysis method."
        )

    return angles, weights


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

    Parameters
    ----------
    arr1 : ndarray of shape (n,)
        First array of paired values.
    arr2 : ndarray of shape (n,)
        Second array of paired values, must match length of arr1.
    name1 : str
        Name of first array for error messages.
    name2 : str
        Name of second array for error messages.
    min_samples : int, default=3
        Minimum required number of valid samples after NaN removal.

    Returns
    -------
    arr1_valid : ndarray of shape (m,)
        Validated first array with NaN pairs removed, where m <= n.
    arr2_valid : ndarray of shape (m,)
        Validated second array with NaN pairs removed, where m <= n.

    Raises
    ------
    ValueError
        If lengths don't match or insufficient samples after NaN removal.
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
            category=UserWarning,
            stacklevel=_external_stacklevel(),
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
    angles : ndarray of shape (n_samples,)
        Sample of angles. NaN angles are dropped with a warning.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.
    weights : ndarray of shape (n_samples,), optional
        Whole-number weights for each angle, interpreted as **counts**
        (e.g., spike counts per phase bin). If None, uniform unit weights
        are used. Weighting an angle by ``k`` is equivalent to observing
        that angle ``k`` times; the test statistic is
        ``z = sum(weights) * R**2`` (see Notes). Occupancy times or rates
        are not counts: the p-value would change with their units.

    Returns
    -------
    z : float
        Rayleigh z-statistic (n * R^2 where R is mean resultant length).
        Range: [0, n], with n = sum(weights) for weighted input. Higher
        values indicate stronger directionality.
    pval : float
        P-value in (0, 1] from Zar's approximation (see Notes). Small
        p-values (< 0.05) indicate a non-uniform distribution.

    Raises
    ------
    ValueError
        If ``angle_unit`` is unknown; if angles are empty, all NaN, contain
        Inf, or number fewer than 3; if ``weights`` is not the same length
        as ``angles`` (a length-1 array is rejected, not broadcast), is not
        finite, or contains negative or non-integer values; or if the total
        weight (sum of weights) is below 3.

    See Also
    --------
    circular_linear_correlation : Test correlation with linear variable.

    Notes
    -----
    The unweighted statistic is ``z = n * R**2`` where ``R`` is the mean
    resultant length and ``n`` the number of angles. The p-value is Zar's
    approximation (Zar 1999, p. 617), as in CircStat's ``circ_rtest``::

        p = exp(sqrt(1 + 4n + 4(n**2 - (n R)**2)) - (1 + 2n))

    It is positive, decreases as ``R`` grows, and is accurate for small
    samples, where ``exp(-z)`` is anti-conservative. For n < 6 it is
    conservative (p too large) below p = 0.01.

    **Weighted variant.** When ``weights`` are supplied this is the
    grouped-data Rayleigh test: weights are frequencies, the resultant
    length is the weighted mean resultant length
    ``R = |sum_i w_i exp(i theta_i)| / sum_i w_i``, and ``n = sum_i w_i``.
    With unit weights this reduces exactly to the unweighted test, and
    integer weights match physically replicating each angle by its weight.
    This is intentionally *not* the effective-sample-size form
    ``n_eff = (sum w)**2 / sum(w**2)``. When the counts come from phase
    bins, ``R`` is biased low by the bin width (grouping).

    References
    ----------
    Zar, J.H. (1999). Biostatistical Analysis, 4th ed. Prentice Hall.
    Berens, P. (2009). CircStat: a MATLAB toolbox for circular statistics.
        Journal of Statistical Software, 31(10).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import rayleigh_test
    >>> # Uniform distribution - expect high p-value
    >>> uniform_angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> z, p = rayleigh_test(uniform_angles)
    >>> p > 0.5
    True
    """
    # The weighted test's sample size is the total count, not the number of
    # angles: a tuned cell can put 100 spikes into one or two phase bins. So
    # the 3-sample minimum applies to n_eff below.
    angles, weights = _valid_angles(
        angles, angle_unit, weights, min_samples=1 if weights is not None else 3
    )
    if weights is not None and np.any(weights != np.round(weights)):
        raise ValueError(
            "rayleigh_test weights must be whole-number counts (e.g., spike "
            "counts per phase bin). The test's sample size is sum(weights), so "
            "non-count weights such as occupancy times or rates make the "
            "p-value depend on their units.\n"
            "Fix: pass integer counts, or use mean_resultant_length(angles, "
            "weights=...) for a descriptive weighted concentration."
        )
    n_eff = float(np.sum(weights)) if weights is not None else float(len(angles))
    if n_eff < 3:
        raise ValueError(
            f"Need at least 3 total weight (sum of counts) for the weighted "
            f"Rayleigh test. Got sum(weights) = {n_eff:g}.\n"
            f"Fix: provide more spikes/counts (all-zero counts carry no "
            f"data), or use a different analysis method."
        )
    r_mean = _mean_resultant_length(angles, weights=weights)

    # Rayleigh z-statistic: z = n_eff * R^2.
    z = n_eff * r_mean**2

    # Zar (1999, p. 617), with the exponent sqrt(a) - b rewritten as
    # (a - b**2) / (sqrt(a) + b) = -4 R_n**2 / (sqrt(a) + b) to avoid
    # cancellation for large n.
    r_n = n_eff * r_mean
    root = np.sqrt(1 + 4 * n_eff + 4 * (n_eff**2 - r_n**2))
    pval = float(np.exp(-4 * r_n**2 / (root + 1 + 2 * n_eff)))
    return float(z), pval


def circular_linear_correlation(
    angles: NDArray[np.float64],
    linear_values: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> tuple[float, float]:
    """
    Circular-linear correlation coefficient.

    Computes the correlation between a circular and a linear variable
    using the Mardia & Jupp formula.

    Parameters
    ----------
    angles : ndarray of shape (n_samples,)
        Circular variable (e.g., spike phases) in radians or degrees.
    linear_values : ndarray of shape (n_samples,)
        Linear variable (e.g., positions, time).
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    r : float
        Correlation coefficient in [0, 1]. It has no sign, so it cannot tell
        phase precession from procession; for slope and direction use
        ``compute_phase_precession``. NaN, with a ``RuntimeWarning``, when
        the correlation is undefined: constant ``linear_values`` or
        constant angles.
    pval : float
        P-value from the large-sample chi-squared approximation (see Notes).
        NaN when ``r`` is undefined.

    Raises
    ------
    ValueError
        If ``angle_unit`` is unknown, the arrays have different lengths, or
        fewer than 3 pairs remain after dropping pairs with NaN.

    See Also
    --------
    phase_position_correlation : Alias with neuroscience naming.
    circular_circular_correlation : For two circular variables.

    Notes
    -----
    ``r`` is the multiple correlation of the linear variable on cos(angles)
    and sin(angles): ``r**2`` is the fraction of its variance a least-squares
    fit ``x ~ a + b cos(angles) + c sin(angles)`` explains (Mardia & Jupp
    2000, ch. 11). With full-rank cos/sin columns this equals

        r^2 = (r_xs^2 + r_xc^2 - 2*r_xs*r_xc*r_cs) / (1 - r_cs^2)

    where r_xs, r_xc and r_cs are the Pearson correlations of x with
    sin(angles), of x with cos(angles), and of cos(angles) with sin(angles).
    The projection form stays defined when cos and sin are collinear, for
    example when the angles take only two values; the fit then has one
    degree of freedom instead of two.

    Under independence, ``n * r**2`` is approximately chi-squared with as many
    degrees of freedom as the fit (2, or 1 for collinear cos/sin). The
    approximation is conservative for small samples (n < 20).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_linear_correlation
    >>> positions = np.linspace(0, 100, 50)
    >>> phases = np.linspace(0, 2 * np.pi, 50)  # Linear relationship
    >>> r, p = circular_linear_correlation(phases, positions)
    >>> r > 0.5 and p < 0.05  # Significant correlation
    True
    """
    # scipy.stats is imported on use; it adds ~0.2 s to importing the package.
    from scipy.stats import chi2

    angles = to_radians(angles, angle_unit)

    # Validate paired inputs (handles length mismatch, NaN removal)
    angles, linear_values = _validate_paired_input(
        angles, linear_values, "angles", "linear_values", min_samples=3
    )

    n = len(angles)

    if np.ptp(linear_values) == 0:
        warnings.warn(
            "linear_values are constant, so the circular-linear correlation is "
            "undefined. Returning r=NaN, p=NaN.",
            category=RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
        return np.nan, np.nan

    # Project the centered linear variable onto the span of the centered
    # cos/sin columns. cos and sin lie in [-1, 1], so an absolute singular
    # value cutoff (RMS spread 1e-10) decides which directions exist.
    x = linear_values - linear_values.mean()
    basis = np.column_stack([np.cos(angles), np.sin(angles)])
    basis -= basis.mean(axis=0)
    u, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10 * np.sqrt(n)))
    if rank == 0:
        warnings.warn(
            "angles are constant, so the circular-linear correlation is "
            "undefined. Returning r=NaN, p=NaN.",
            category=RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
        return np.nan, np.nan

    projection = u[:, :rank].T @ x
    r_squared = min(float(projection @ projection) / float(x @ x), 1.0)
    pval = float(chi2.sf(n * r_squared, df=rank))
    return float(np.sqrt(r_squared)), pval


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
    phases : ndarray of shape (n_spikes,)
        Spike phases relative to LFP theta in radians or degrees.
    positions : ndarray of shape (n_spikes,)
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

    Computes the Jammalamadaka & Sarma (1988) circular correlation
    coefficient between two circular variables, as CircStat's
    ``circ_corrcc``.

    Parameters
    ----------
    angles1 : ndarray of shape (n_samples,)
        First circular variable in radians or degrees.
    angles2 : ndarray of shape (n_samples,)
        Second circular variable in radians or degrees. Must have same
        length as ``angles1``.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles (applies to both arrays).

    Returns
    -------
    r : float
        Circular correlation coefficient in [-1, 1].
        - r > 0: positive association (angles increase together)
        - r < 0: negative association (angles move in opposite directions)
        - r = 0: no circular correlation
        NaN, with a ``RuntimeWarning``, when either variable is constant.
    pval : float
        Two-sided p-value from the large-sample normal approximation. NaN
        when ``r`` is undefined or the statistic's variance is zero.

    Raises
    ------
    ValueError
        If ``angle_unit`` is unknown, the arrays have different lengths, or
        fewer than 5 pairs remain after dropping pairs with NaN.

    See Also
    --------
    circular_linear_correlation : For circular-linear correlation.

    Notes
    -----
    **Formula** (Jammalamadaka & SenGupta, 2001, p. 176):

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
    >>> from neurospatial.stats.circular import circular_circular_correlation
    >>> angles1 = np.random.default_rng(42).uniform(0, 2 * np.pi, 100)
    >>> angles2 = angles1 + 0.1 * np.random.default_rng(42).standard_normal(100)
    >>> r, p = circular_circular_correlation(angles1, angles2)
    >>> r > 0.8
    True

    This is not the Fisher & Lee (1983) coefficient, which is built from
    pairwise angle differences and gives different values.

    References
    ----------
    Jammalamadaka, S.R. & Sarma, Y.R. (1988). A correlation coefficient for
        angular variables. In Statistical Theory and Data Analysis II,
        349-364. North-Holland.
    Jammalamadaka, S.R. & SenGupta, A. (2001). Topics in Circular Statistics.
        World Scientific, p. 176.
    """
    from scipy import stats

    angles1 = to_radians(angles1, angle_unit)
    angles2 = to_radians(angles2, angle_unit)

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

    # A spread below 1e-10 rad RMS is rounding error around a constant angle.
    if min(np.mean(sin_dev1**2), np.mean(sin_dev2**2)) <= 1e-20:
        warnings.warn(
            "One or both circular variables are constant, so the "
            "circular-circular correlation is undefined. Returning r=NaN, p=NaN.",
            category=RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
        return np.nan, np.nan

    # Jammalamadaka & Sarma coefficient
    num = np.sum(sin_dev1 * sin_dev2)
    den = np.sqrt(np.sum(sin_dev1**2) * np.sum(sin_dev2**2))
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
            "Cannot compute the p-value: no pair varies in both variables at "
            "once, so the statistic's variance is zero. Returning p=NaN.",
            category=RuntimeWarning,
            stacklevel=_external_stacklevel(),
        )
        return rho, np.nan

    # Test statistic follows standard normal under null hypothesis
    ts = np.sqrt((n * l20 * l02) / l22) * rho
    pval = float(2 * stats.norm.sf(np.abs(ts)))

    # Ensure p-value is in valid range
    pval = float(np.clip(pval, 0.0, 1.0))

    return rho, pval


# =============================================================================
# Public API - Circular Descriptive Statistics
# =============================================================================


def circular_mean(
    angles: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute the circular mean (mean direction) of angles.

    Parameters
    ----------
    angles : ndarray of shape (n_samples,)
        Sample of angles. NaN angles (and their weights) are dropped with a
        warning.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.
    weights : ndarray of shape (n_samples,), optional
        Weights for each angle. If None, uniform weights are used.
        Must be the same length as ``angles``, finite and non-negative.

    Returns
    -------
    float
        Circular mean in radians, range [-π, π], whatever ``angle_unit``.
        NaN when no angle carries weight, or when the mean direction is
        undefined because the resultant vector is zero (e.g., two opposite
        angles).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_mean
    >>> angles = np.array([0, np.pi / 4, np.pi / 2])
    >>> mean = circular_mean(angles)
    >>> bool(np.abs(mean - np.pi / 4) < 0.1)
    True
    """
    mean_cos, mean_sin = _mean_vector(
        *_valid_angles(angles, angle_unit, weights, min_samples=0)
    )
    # A resultant below rounding error has no direction (NaN fails the test).
    if not np.hypot(mean_cos, mean_sin) > 1e-12:
        return np.nan
    return float(np.arctan2(mean_sin, mean_cos))


def circular_variance(
    angles: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute the circular variance of angles.

    The circular variance is defined as V = 1 - R, where R is the mean
    resultant length.

    Parameters
    ----------
    angles : ndarray of shape (n_samples,)
        Sample of angles. NaN angles (and their weights) are dropped with a
        warning.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.
    weights : ndarray of shape (n_samples,), optional
        Weights for each angle. If None, uniform weights are used.
        Must be the same length as ``angles``, finite and non-negative.

    Returns
    -------
    float
        Circular variance in [0, 1]. 0 indicates all angles are identical,
        1 indicates maximum dispersion (uniform distribution). NaN when no
        angle carries weight.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_variance
    >>> # Concentrated distribution - low variance
    >>> concentrated = np.array([0.1, 0.15, 0.05, 0.12])
    >>> var = circular_variance(concentrated)
    >>> var < 0.1
    True
    """
    return 1.0 - _mean_resultant_length(
        *_valid_angles(angles, angle_unit, weights, min_samples=0)
    )


def mean_resultant_length(
    angles: NDArray[np.float64],
    *,
    angle_unit: Literal["rad", "deg"] = "rad",
    weights: NDArray[np.float64] | None = None,
) -> float:
    """
    Compute the mean resultant length (Rayleigh vector length).

    The mean resultant length R measures concentration of circular data.
    It equals 1 if all angles are identical and approaches 0 for uniform
    distributions.

    Parameters
    ----------
    angles : ndarray of shape (n_samples,)
        Sample of angles. NaN angles (and their weights) are dropped with a
        warning.
    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.
    weights : ndarray of shape (n_samples,), optional
        Weights for each angle. If None, uniform weights are used.
        Must be the same length as ``angles``, finite and non-negative.

    Returns
    -------
    float
        Mean resultant length R in [0, 1]. NaN when no angle carries weight.

    Notes
    -----
    The mean resultant length is defined as:

        R = |mean(exp(i * angles))| = sqrt(mean(cos)^2 + mean(sin)^2)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import mean_resultant_length
    >>> # All angles identical - R should be 1
    >>> r = mean_resultant_length(np.array([0, 0, 0, 0]))
    >>> bool(np.abs(r - 1.0) < 0.01)
    True
    """
    return _mean_resultant_length(
        *_valid_angles(angles, angle_unit, weights, min_samples=0)
    )


# =============================================================================
# Data Classes - GLM Circular Basis
# =============================================================================


@dataclass(frozen=True, eq=False)
class CircularBasisResult(ReadOnlyArraysMixin):
    """
    Result from circular_basis() function.

    Holds the angles of a circular predictor; the sin/cos components for a
    GLM design matrix are derived from them.

    Attributes
    ----------
    angles : ndarray, shape (n_samples,)
        Angles in radians (converted from degrees if ``angle_unit='deg'``).

    Properties
    ----------
    sin_component : ndarray, shape (n_samples,)
        Sine of angles: sin(angles).
    cos_component : ndarray, shape (n_samples,)
        Cosine of angles: cos(angles).
    design_matrix : ndarray, shape (n_samples, 2)
        Design matrix with columns [sin_component, cos_component].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_basis
    >>> angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    >>> result = circular_basis(angles)
    >>> result.design_matrix.shape
    (100, 2)
    """

    angles: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own the angles as a read-only float64 copy."""
        object.__setattr__(self, "angles", np.asarray(self.angles, dtype=np.float64))
        self._own_arrays()

    @property
    def sin_component(self) -> NDArray[np.float64]:
        """Sine of the angles, shape (n_samples,)."""
        return np.sin(self.angles)

    @property
    def cos_component(self) -> NDArray[np.float64]:
        """Cosine of the angles, shape (n_samples,)."""
        return np.cos(self.angles)

    @property
    def design_matrix(self) -> NDArray[np.float64]:
        """
        Return design matrix with columns [sin_component, cos_component].

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Design matrix suitable for GLM.
        """
        return np.column_stack([np.sin(self.angles), np.cos(self.angles)])


# =============================================================================
# Internal Helper Functions - GLM Circular Basis
# =============================================================================


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
        P-value from chi-squared distribution with 2 df; NaN, with a
        warning, when ``cov_matrix`` is singular.

    Raises
    ------
    ValueError
        If the coefficients or ``cov_matrix`` are not finite, or
        ``cov_matrix`` is not symmetric positive semi-definite.

    Notes
    -----
    Uses the Wald test statistic: W = beta.T @ inv(cov) @ beta ~ chi2(2)
    where beta = [beta_sin, beta_cos].
    """
    from scipy.stats import chi2

    beta = np.array([beta_sin, beta_cos], dtype=np.float64)
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(cov_matrix))):
        raise ValueError(
            f"beta_sin, beta_cos and cov_matrix must be finite, got "
            f"beta=({beta_sin!r}, {beta_cos!r}) and cov_matrix={cov_matrix.tolist()}.\n"
            f"Fix: check that the GLM fit converged before testing its "
            f"coefficients."
        )
    scale = float(np.max(np.abs(cov_matrix)))
    if not np.allclose(cov_matrix, cov_matrix.T, rtol=1e-8, atol=1e-12 * scale) or (
        np.linalg.eigvalsh(cov_matrix).min() < -1e-12 * scale
    ):
        raise ValueError(
            f"cov_matrix must be a symmetric positive semi-definite covariance "
            f"matrix, got {cov_matrix.tolist()}.\n"
            f"Fix: pass model.cov_params()[1:3, 1:3], the rows and columns "
            f"for [sin, cos] of the fitted GLM's covariance."
        )

    # Wald statistic: beta.T @ inv(cov) @ beta
    # Use solve instead of inv for numerical stability
    try:
        wald_stat = float(beta @ np.linalg.solve(cov_matrix, beta))
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix is singular. Cannot compute p-value; returning NaN.",
            stacklevel=_external_stacklevel(),
            category=UserWarning,
        )
        return np.nan

    return float(chi2.sf(wald_stat, df=2))


# =============================================================================
# Public API - GLM Circular Basis Functions
# =============================================================================


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
        Circular variable, one value per GLM sample (e.g., per time bin).
        Common use cases:

        - **Head direction**: Animal's facing direction
        - **Theta phase**: LFP phase in each time bin
        - **Running direction**: Direction of movement
        - **Time of day**: Circadian phase

    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles. Use ``'deg'`` for angles recorded in degrees.

    Returns
    -------
    CircularBasisResult
        Result object containing:

        - angles: angles in radians
        - sin_component: sin(angles)
        - cos_component: cos(angles)
        - design_matrix property: (n_samples, 2) array for GLM

    Raises
    ------
    ValueError
        If ``angle_unit`` is unknown, or ``angles`` is not a 1-D array of
        finite values.

    See Also
    --------
    circular_basis_metrics : Compute amplitude/phase from GLM coefficients.
    is_modulated : Quick significance test for circular modulation.
    plot_circular_basis_tuning : Visualize fitted tuning curve.

    Notes
    -----
    **Why sin/cos basis?**

    For a circular predictor θ, using sin(θ) and cos(θ) as separate
    predictors allows the model to capture any phase and amplitude of circular
    modulation. The fitted coefficients (β_sin, β_cos) can be converted to:

    - Amplitude: sqrt(β_sin² + β_cos²) — modulation strength
    - Phase: atan2(β_sin, β_cos) — preferred direction/phase

    **GLM vs. Binned Tuning Curves**:

    - **Binned**: Non-parametric, intuitive, good for visualization.
      Requires many spikes per bin.
    - **GLM** (``circular_basis``): Parametric, statistically rigorous, can
      include covariates. Better for low spike counts.

    Examples
    --------
    **Head direction tuning (GLM approach)**:

    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_basis
    >>> # Head direction in each time bin (in degrees)
    >>> headings = np.array([45, 50, 48, 52, 180, 185, 170])  # degrees
    >>> result = circular_basis(headings, angle_unit="deg")
    >>> result.design_matrix.shape
    (7, 2)

    **Theta phase modulation**:

    >>> # LFP phase in each time bin (in radians)
    >>> theta_phases = np.random.default_rng(0).uniform(0, 2 * np.pi, 100)
    >>> result = circular_basis(theta_phases)  # angle_unit='rad' is default
    >>> X = result.design_matrix
    >>> # Now fit a Poisson GLM of the spike counts per time bin, e.g.
    >>> # sm.GLM(spike_counts, sm.add_constant(X), family=sm.families.Poisson())
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError(
            f"angles must be a 1-D array of finite values, got shape "
            f"{angles.shape} with {int(np.sum(~np.isfinite(angles)))} "
            f"non-finite value(s).\n"
            f"Fix: pass one angle per GLM sample, and drop the samples (with "
            f"their spike counts) where the angle is missing."
        )
    return CircularBasisResult(angles=to_radians(angles, angle_unit))


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
        Preferred phase angle in radians, range [-π, π].
        Computed as atan2(beta_sin, beta_cos).
    pvalue : float or None
        Wald-test p-value for H0: amplitude = 0 (no modulation). None if
        ``cov_matrix`` is not provided; NaN, with a warning, if it is
        singular.

    Raises
    ------
    ValueError
        If ``cov_matrix`` is not 2x2, is not finite, or is not symmetric
        positive semi-definite, or the coefficients are not finite.

    See Also
    --------
    circular_basis : Create design matrix for GLM.

    Notes
    -----
    **Interpretation** (log link, as in a Poisson GLM):

    - the rate peaks at ``phase`` and is lowest opposite it; the
      peak-to-trough rate ratio is ``exp(2 * amplitude)``, so amplitude 0.2
      is a 1.49-fold and 0.5 a 2.7-fold modulation
    - pvalue < 0.05 indicates statistically significant modulation

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import circular_basis_metrics
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


def is_modulated(
    beta_sin: float,
    beta_cos: float,
    cov_matrix: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    min_magnitude: float = 0.2,
) -> bool:
    """
    Quick check: Do GLM coefficients show significant circular modulation?

    This is a convenience function that combines statistical significance
    (Wald test p-value) with a practical magnitude threshold.

    Parameters
    ----------
    beta_sin : float
        Coefficient for sin(angle) from fitted GLM.
    beta_cos : float
        Coefficient for cos(angle) from fitted GLM.
    cov_matrix : ndarray, shape (2, 2)
        Covariance matrix for [beta_sin, beta_cos] from GLM fit.
        Required for significance testing.
    alpha : float, default=0.05
        Significance level for Wald test.
    min_magnitude : float, default=0.2
        Minimum modulation magnitude (amplitude) required.
        Amplitude = sqrt(beta_sin^2 + beta_cos^2).

        With a log link (Poisson GLM) the peak-to-trough rate ratio is
        ``exp(2 * amplitude)``, so the default 0.2 asks for at least a
        1.49-fold modulation of the rate. This is a screening convention,
        not a statistical criterion; set it from the effect size that
        matters for your question.

    Returns
    -------
    bool
        True if BOTH:
        - p-value < alpha (statistically significant)
        - magnitude >= min_magnitude (practically meaningful)
        False otherwise, including when ``cov_matrix`` is singular (the
        p-value is then NaN, with a warning).

    Raises
    ------
    ValueError
        If ``cov_matrix`` is not a finite, symmetric positive semi-definite
        2x2 matrix, or the coefficients are not finite.

    See Also
    --------
    circular_basis_metrics : For full analysis with all metrics.

    Notes
    -----
    **Why both thresholds?**

    Statistical significance alone can occur with very weak modulation
    when sample sizes are large. Requiring both significance AND a
    minimum magnitude ensures the modulation is both reliable and
    biologically meaningful.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.stats.circular import is_modulated
    >>> # Strong significant modulation
    >>> is_modulated(2.0, 2.0, np.array([[0.01, 0], [0, 0.01]]))
    True
    >>> # Weak (non-significant) modulation
    >>> is_modulated(0.1, 0.1, np.array([[1.0, 0], [0, 1.0]]))
    False
    """
    amplitude, _, pvalue = circular_basis_metrics(
        beta_sin, beta_cos, np.asarray(cov_matrix, dtype=np.float64)
    )
    # A covariance was given, so there is a p-value (NaN when singular).
    return bool(cast("float", pvalue) < alpha and amplitude >= min_magnitude)


def plot_circular_basis_tuning(
    beta_sin: float,
    beta_cos: float,
    angles: NDArray[np.float64] | None = None,
    rates: NDArray[np.float64] | None = None,
    ax: Axes | PolarAxes | None = None,
    *,
    intercept: float = 0.0,
    cov_matrix: NDArray[np.float64] | None = None,
    projection: Literal["polar", "linear"] = "polar",
    n_points: int = 100,
    show_data: bool = False,
    show_fit: bool = True,
    show_ci: bool = False,
    ci: float = 0.95,
    color: str = "C0",
    data_color: str = "gray",
    data_alpha: float = 0.5,
    ci_alpha: float = 0.3,
    line_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
) -> Axes | PolarAxes:
    """
    Plot tuning curve from GLM circular basis coefficients.

    Visualizes the fitted circular tuning curve from a GLM with sin/cos
    basis functions, optionally overlaying raw binned data.

    Parameters
    ----------
    beta_sin : float
        Coefficient for sin(angle) from fitted GLM.
    beta_cos : float
        Coefficient for cos(angle) from fitted GLM.
    angles : array, shape (n_bins,), optional
        Center of each angular bin (radians) for raw data overlay.
        Required if ``show_data=True``.
    rates : array, shape (n_bins,), optional
        Firing rate or response in each bin.
        Required if ``show_data=True``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with appropriate projection.
    intercept : float, default=0.0
        Intercept coefficient from GLM. The curve is
        ``exp(intercept + beta_sin sin + beta_cos cos)``: a rate in Hz if
        the GLM had a ``log(dt)`` offset, otherwise the expected count per
        time bin.
    cov_matrix : ndarray, shape (2, 2), optional
        Covariance matrix for [beta_sin, beta_cos] from GLM fit.
        Required if ``show_ci=True`` to compute confidence bands. The band
        leaves out the intercept's uncertainty, so it is narrower than a
        full pointwise interval for the rate.
    projection : {'polar', 'linear'}, default='polar'
        Plot projection type.
    n_points : int, default=100
        Number of points for smooth fitted curve.
    show_data : bool, default=False
        If True, overlay raw data points. Requires ``angles`` and ``rates``.
    show_fit : bool, default=True
        If True, show smooth curve from GLM fit.
    show_ci : bool, default=False
        If True, show confidence band around fitted curve. Requires ``cov_matrix``.
    ci : float, default=0.95
        Confidence level for band (e.g., 0.95 for 95% CI).
    color : str, default='C0'
        Color for fitted curve.
    data_color : str, default='gray'
        Color for data points.
    data_alpha : float, default=0.5
        Alpha (transparency) for data points.
    ci_alpha : float, default=0.3
        Alpha (transparency) for confidence band fill.
    line_kwargs : dict, optional
        Additional kwargs for line plot (fitted curve).
    scatter_kwargs : dict, optional
        Additional kwargs for scatter plot (data points).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.

    Raises
    ------
    ValueError
        If ``show_data=True`` but ``angles`` or ``rates`` not provided.
        If ``show_ci=True`` but ``cov_matrix`` not provided.

    See Also
    --------
    circular_basis_metrics : Compute amplitude/phase from coefficients.
    circular_basis : Create design matrix for GLM.

    Notes
    -----
    The curve assumes a log link (Poisson GLM). Data points from
    ``show_data`` share its axis, so give ``rates`` in the curve's units.

    The polar projection draws angles as in the arena: 0 = East (right),
    π/2 = North (up), counter-clockwise, the convention in which
    ``circular_basis_metrics`` reports the preferred direction. A
    caller-supplied polar axis is reset to this orientation.

    Examples
    --------
    **Head direction GLM tuning curve**:

    >>> import numpy as np
    >>> from neurospatial.stats.circular import plot_circular_basis_tuning
    >>> # After fitting GLM to head direction data
    >>> # Coefficients: preferred direction ~45° with strong modulation
    >>> beta_sin, beta_cos = 0.7, 0.7  # atan2(0.7, 0.7) ≈ 45°
    >>> ax = plot_circular_basis_tuning(
    ...     beta_sin, beta_cos, intercept=2.0
    ... )  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    # Validate show_data requirements
    if show_data and (angles is None or rates is None):
        raise ValueError(
            "show_data=True requires both angles and rates arguments.\n\n"
            "To show model fit over raw data:\n"
            "  plot_circular_basis_tuning(beta_sin, beta_cos, "
            "angles=bins, rates=rates, show_data=True)\n\n"
            "To show only the fitted curve:\n"
            "  plot_circular_basis_tuning(beta_sin, beta_cos, show_data=False)"
        )

    # Validate show_ci requirements
    if show_ci and cov_matrix is None:
        raise ValueError(
            "show_ci=True requires cov_matrix argument.\n\n"
            "To show confidence bands:\n"
            "  cov = model.cov_params()[1:3, 1:3]  # Extract sin/cos covariance\n"
            "  plot_circular_basis_tuning(beta_sin, beta_cos, cov_matrix=cov, show_ci=True)"
        )

    # Validate cov_matrix shape if provided
    if cov_matrix is not None:
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        if cov_matrix.shape != (2, 2):
            raise ValueError(
                f"cov_matrix must be a 2x2 matrix for [beta_sin, beta_cos].\n"
                f"Got shape: {cov_matrix.shape}\n\n"
                f"Extract from GLM fit:\n"
                f"  cov = model.cov_params()[1:3, 1:3]  # Rows/cols for sin, cos"
            )

    # Create figure if needed
    if ax is None:
        if projection == "polar":
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        else:
            _, ax = plt.subplots()

    # Default kwargs
    line_defaults: dict = {"color": color, "linewidth": 2, "zorder": 2}
    scatter_defaults: dict = {
        "color": data_color,
        "alpha": data_alpha,
        "s": 30,
        "zorder": 1,
    }

    # Merge with user kwargs
    line_kw = {**line_defaults, **(line_kwargs or {})}
    scatter_kw = {**scatter_defaults, **(scatter_kwargs or {})}

    # Generate smooth curve for fit
    theta_smooth = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # GLM prediction: lambda(theta) = exp(intercept + beta_cos*cos + beta_sin*sin)
    linear_pred = (
        intercept + beta_cos * np.cos(theta_smooth) + beta_sin * np.sin(theta_smooth)
    )
    # Transform to response scale (Poisson GLM uses exp link)
    fitted_response = np.exp(linear_pred)

    # Compute confidence bands if requested
    ci_lower: NDArray[np.float64] | None = None
    ci_upper: NDArray[np.float64] | None = None

    if show_ci and cov_matrix is not None:
        # Design matrix for each angle: [sin(θ), cos(θ)]
        design = np.column_stack([np.sin(theta_smooth), np.cos(theta_smooth)])

        # Variance of linear predictor at each angle
        var_linear = np.sum((design @ cov_matrix) * design, axis=1)
        se_linear = np.sqrt(np.maximum(var_linear, 0))

        # Z-score for confidence level
        z = stats.norm.ppf((1 + ci) / 2)

        # CI on linear scale, then transform
        linear_lower = linear_pred - z * se_linear
        linear_upper = linear_pred + z * se_linear

        # Transform to response scale
        ci_lower = np.exp(linear_lower)
        ci_upper = np.exp(linear_upper)

    # Close the curve (append first point)
    theta_closed = np.concatenate([theta_smooth, [theta_smooth[0] + 2 * np.pi]])
    fitted_closed = np.concatenate([fitted_response, [fitted_response[0]]])

    # Close CI bands if computed
    if ci_lower is not None and ci_upper is not None:
        ci_lower_closed = np.concatenate([ci_lower, [ci_lower[0]]])
        ci_upper_closed = np.concatenate([ci_upper, [ci_upper[0]]])
    else:
        ci_lower_closed = None
        ci_upper_closed = None

    if projection == "polar":
        polar_ax = cast("PolarAxes", ax)

        # Draw angles as in the arena: 0 = East (right), counter-clockwise.
        polar_ax.set_theta_zero_location("E")
        polar_ax.set_theta_direction(1)

        # Plot confidence band (behind curve)
        if show_ci and ci_lower_closed is not None and ci_upper_closed is not None:
            polar_ax.fill_between(
                theta_closed,
                ci_lower_closed,
                ci_upper_closed,
                color=color,
                alpha=ci_alpha,
                zorder=0,
            )

        # Plot fitted curve
        if show_fit:
            polar_ax.plot(theta_closed, fitted_closed, **line_kw)

        # Plot data points
        if show_data and angles is not None and rates is not None:
            angles_arr = np.asarray(angles, dtype=np.float64).ravel()
            rates_arr = np.asarray(rates, dtype=np.float64).ravel()
            polar_ax.scatter(angles_arr, rates_arr, **scatter_kw)

    else:
        # Linear projection
        # Plot confidence band (behind curve)
        if show_ci and ci_lower_closed is not None and ci_upper_closed is not None:
            ax.fill_between(
                theta_closed,
                ci_lower_closed,
                ci_upper_closed,
                color=color,
                alpha=ci_alpha,
                zorder=0,
            )

        # Plot fitted curve
        if show_fit:
            ax.plot(theta_closed, fitted_closed, **line_kw)

        # Plot data points
        if show_data and angles is not None and rates is not None:
            angles_arr = np.asarray(angles, dtype=np.float64).ravel()
            rates_arr = np.asarray(rates, dtype=np.float64).ravel()
            ax.scatter(angles_arr, rates_arr, **scatter_kw)

        ax.set_xlabel("Angle (rad)")
        ax.set_ylabel("Response")
        ax.set_xlim(0, 2 * np.pi)

    return ax
