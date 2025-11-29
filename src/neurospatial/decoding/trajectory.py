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


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
]
