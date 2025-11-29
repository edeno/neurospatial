"""Trajectory analysis for decoded position sequences.

This module provides functions for fitting and detecting trajectories in
posterior probability sequences from Bayesian decoding. These are useful for
analyzing replay events and sequential activation patterns.

Functions
---------
fit_isotonic_trajectory
    Fit monotonic trajectory using isotonic regression.
fit_linear_trajectory
    Fit linear trajectory with optional Monte Carlo uncertainty.
detect_trajectory_radon
    Detect linear trajectory using Radon transform.

Result Dataclasses
------------------
IsotonicFitResult
    Container for isotonic regression results.
LinearFitResult
    Container for linear regression results.
RadonDetectionResult
    Container for Radon transform detection results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class IsotonicFitResult:
    """Container for isotonic regression trajectory fit results.

    Isotonic regression fits a monotonic (either strictly increasing or
    decreasing) function to the decoded position sequence. This is useful
    for detecting replay events that represent sequential traversal of
    spatial locations.

    Attributes
    ----------
    fitted_positions : NDArray[np.float64]
        Fitted monotonic positions in bin index space, shape (n_time_bins,).
        These are constrained to be monotonically increasing or decreasing.
    r_squared : float
        Coefficient of determination (R²) measuring goodness of fit.
        Range [0, 1] where 1 indicates perfect fit.
    direction : {"increasing", "decreasing"}
        Direction of the fitted monotonic trajectory.
        "increasing" means position increases with time (forward replay).
        "decreasing" means position decreases with time (reverse replay).
    residuals : NDArray[np.float64]
        Residuals (observed - fitted) for each time bin, shape (n_time_bins,).
        Useful for diagnostics and outlier detection.

    See Also
    --------
    fit_isotonic_trajectory : Function that produces this result.
    LinearFitResult : For linear (non-monotonic constrained) fits.

    Examples
    --------
    >>> result = fit_isotonic_trajectory(posterior, times)
    >>> print(f"R² = {result.r_squared:.3f}, direction = {result.direction}")
    R² = 0.923, direction = increasing
    >>> replay_speed = np.diff(result.fitted_positions) / np.diff(times)
    """

    fitted_positions: NDArray[np.float64]
    r_squared: float
    direction: Literal["increasing", "decreasing"]
    residuals: NDArray[np.float64]


@dataclass(frozen=True)
class LinearFitResult:
    """Container for linear trajectory fit results.

    Linear regression fits a straight line to the decoded position sequence.
    This provides slope (replay speed) and intercept estimates, optionally
    with uncertainty quantification via Monte Carlo sampling.

    Attributes
    ----------
    slope : float
        Slope of the fitted line in bin indices per second.
        Positive slope indicates forward replay, negative indicates reverse.
        To convert to environment units: speed_cm_s = slope * bin_size.
    intercept : float
        Intercept of the fitted line (bin index at t=0).
        Note: t=0 refers to the start of the time array, not absolute time.
    r_squared : float
        Coefficient of determination (R²) measuring goodness of fit.
        Range [0, 1] where 1 indicates perfect linear relationship.
    slope_std : float | None
        Standard error of the slope estimate.
        Only available when method="sample" (Monte Carlo estimation).
        None when method="map" (deterministic fit).

    See Also
    --------
    fit_linear_trajectory : Function that produces this result.
    IsotonicFitResult : For monotonic (non-linear) fits.

    Examples
    --------
    >>> result = fit_linear_trajectory(env, posterior, times, method="sample", rng=42)
    >>> print(f"slope = {result.slope:.2f} ± {result.slope_std:.2f} bins/s")
    slope = 15.32 ± 1.24 bins/s
    >>> replay_speed_cm_s = result.slope * env.bin_size
    """

    slope: float
    intercept: float
    r_squared: float
    slope_std: float | None


@dataclass(frozen=True)
class RadonDetectionResult:
    """Container for Radon transform trajectory detection results.

    The Radon transform treats the posterior as a 2D image (time × position)
    and finds the line with maximum integrated probability mass. This is
    particularly effective for detecting diagonal stripe patterns in replay
    posteriors.

    Attributes
    ----------
    angle_degrees : float
        Detected trajectory angle in degrees.
        0° = horizontal (constant position, no movement).
        90° = vertical (instantaneous teleportation).
        Positive angles indicate forward replay, negative indicate reverse.
    score : float
        Radon transform value at the detected peak.
        Higher values indicate stronger linear structure in the posterior.
        Useful for comparing replay "quality" across events.
    offset : float
        Perpendicular offset from origin in the Radon parameterization.
        Combined with angle, uniquely identifies the detected line.
    sinogram : NDArray[np.float64]
        Full Radon transform (sinogram) array, shape (n_angles, n_offsets).
        Useful for visualization and secondary peak detection.

    See Also
    --------
    detect_trajectory_radon : Function that produces this result.

    Notes
    -----
    The Radon transform requires the optional dependency `scikit-image`.
    Install with: ``pip install scikit-image`` or
    ``pip install neurospatial[trajectory]``

    Examples
    --------
    >>> result = detect_trajectory_radon(posterior)
    >>> print(f"Detected angle: {result.angle_degrees:.1f}°, score: {result.score:.3f}")
    Detected angle: 42.0°, score: 0.847
    """

    angle_degrees: float
    score: float
    offset: float
    sinogram: NDArray[np.float64]


def fit_isotonic_trajectory(
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    increasing: bool | None = None,
    method: Literal["map", "expected"] = "expected",
) -> IsotonicFitResult:
    """Fit monotonic trajectory using isotonic regression.

    Isotonic regression fits a monotonic (non-decreasing or non-increasing)
    function to the decoded position sequence. This is useful for detecting
    replay events that represent sequential traversal of spatial locations.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Each row should sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time values for each time bin. Used as the independent variable (x)
        in the regression.
    increasing : bool | None, optional
        Direction constraint for the isotonic fit:
        - True: Fit increasing (non-decreasing) monotonic function
        - False: Fit decreasing (non-increasing) monotonic function
        - None (default): Try both directions and return the one with higher R²
    method : {"map", "expected"}, optional
        How to extract position from the posterior for fitting:
        - "map": Use argmax (maximum a posteriori) bin index
        - "expected" (default): Use weighted mean (expected) bin index

    Returns
    -------
    IsotonicFitResult
        Container with fitted_positions, r_squared, direction, and residuals.

    Raises
    ------
    ValueError
        If method is not "map" or "expected".

    See Also
    --------
    IsotonicFitResult : Container for isotonic fit results.
    fit_linear_trajectory : For unconstrained linear fits.

    Notes
    -----
    This function uses scikit-learn's IsotonicRegression, which implements
    the pool adjacent violators algorithm (PAVA). The R² is computed as:

    .. math::

        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}` is the sum of squared residuals and :math:`SS_{tot}`
    is the total sum of squares.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.trajectory import fit_isotonic_trajectory
    >>>
    >>> # Create posterior with increasing positions
    >>> n_time_bins, n_bins = 20, 50
    >>> posterior = np.zeros((n_time_bins, n_bins))
    >>> for t in range(n_time_bins):
    ...     posterior[t, t * 2] = 1.0  # Delta posteriors
    >>> times = np.linspace(0, 1, n_time_bins)
    >>>
    >>> result = fit_isotonic_trajectory(posterior, times)
    >>> print(f"R² = {result.r_squared:.3f}, direction = {result.direction}")
    R² = 1.000, direction = increasing
    """
    from sklearn.isotonic import IsotonicRegression

    if method not in ("map", "expected"):
        raise ValueError(f"method must be 'map' or 'expected', got {method!r}")

    # Extract positions from posterior
    n_bins = posterior.shape[1]
    bin_indices = np.arange(n_bins)

    if method == "map":
        # Use argmax positions
        positions = np.argmax(posterior, axis=1).astype(np.float64)
    else:
        # Use expected (weighted mean) positions
        positions = np.sum(posterior * bin_indices, axis=1)

    # If direction is specified, fit once
    if increasing is not None:
        iso_reg = IsotonicRegression(increasing=increasing)
        fitted = iso_reg.fit_transform(times, positions)
        direction: Literal["increasing", "decreasing"] = (
            "increasing" if increasing else "decreasing"
        )
        residuals = positions - fitted
        r_squared = _compute_r_squared(positions, fitted)

        return IsotonicFitResult(
            fitted_positions=fitted,
            r_squared=float(r_squared),
            direction=direction,
            residuals=residuals,
        )

    # Try both directions and return the one with better R²
    iso_inc = IsotonicRegression(increasing=True)
    fitted_inc = iso_inc.fit_transform(times, positions)
    r2_inc = _compute_r_squared(positions, fitted_inc)

    iso_dec = IsotonicRegression(increasing=False)
    fitted_dec = iso_dec.fit_transform(times, positions)
    r2_dec = _compute_r_squared(positions, fitted_dec)

    if r2_inc >= r2_dec:
        return IsotonicFitResult(
            fitted_positions=fitted_inc,
            r_squared=float(r2_inc),
            direction="increasing",
            residuals=positions - fitted_inc,
        )
    else:
        return IsotonicFitResult(
            fitted_positions=fitted_dec,
            r_squared=float(r2_dec),
            direction="decreasing",
            residuals=positions - fitted_dec,
        )


def _compute_r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Compute R² (coefficient of determination).

    Parameters
    ----------
    observed : NDArray[np.float64]
        Observed values.
    predicted : NDArray[np.float64]
        Predicted (fitted) values.

    Returns
    -------
    float
        R² value in [0, 1]. Returns 0 if total variance is zero.
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "fit_isotonic_trajectory",
]
