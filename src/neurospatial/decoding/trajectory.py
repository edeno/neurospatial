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

    from neurospatial import Environment

# Check for optional scikit-image dependency
_SKIMAGE_AVAILABLE = False
try:
    from skimage.transform import radon

    _SKIMAGE_AVAILABLE = True
except ImportError:
    radon = None


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
        Standard deviation of slope estimates from Monte Carlo sampling.
        Measures the spread of sampled slopes across different draws from
        the posterior. Only available when method="sample".
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


def fit_linear_trajectory(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    n_samples: int = 1000,
    method: Literal["map", "sample"] = "sample",
    rng: np.random.Generator | int | None = None,
) -> LinearFitResult:
    """Fit linear trajectory to posterior using linear regression.

    Fits a straight line to the decoded position sequence, providing slope
    (replay speed) and intercept estimates. Optionally uses Monte Carlo
    sampling to quantify uncertainty in the slope estimate.

    Parameters
    ----------
    env : Environment
        Spatial environment (provides bin_centers for coordinate transforms).
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Each row should sum to 1.
    times : NDArray[np.float64], shape (n_time_bins,)
        Time values for each time bin. Used as the independent variable (x)
        in the regression. Need not be uniformly spaced.
    n_samples : int, optional
        Number of Monte Carlo samples from the posterior for uncertainty
        estimation (only used when method="sample"). Default is 1000.
    method : {"map", "sample"}, optional
        How to fit the linear trajectory:
        - "map": Use argmax positions directly. Fast but ignores uncertainty.
        - "sample" (default): Sample from posterior, fit to each sample,
          average coefficients. Provides uncertainty estimate via slope_std.
    rng : np.random.Generator | int | None, optional
        Random number generator for reproducibility (method="sample" only).
        - If Generator: Use directly
        - If int: Seed for np.random.default_rng()
        - If None: Use default RNG (not reproducible)

    Returns
    -------
    LinearFitResult
        Container with slope, intercept, r_squared, and slope_std.
        slope_std is None when method="map".

    Raises
    ------
    ValueError
        If method is not "map" or "sample".
        If posterior is not 2D.
        If times length doesn't match posterior shape.

    See Also
    --------
    LinearFitResult : Container for linear fit results.
    fit_isotonic_trajectory : For monotonic (non-linear) fits.

    Notes
    -----
    **Slope interpretation**: The slope is in units of bin indices per second.
    To convert to environment units (e.g., cm/s):

    .. code-block:: python

        speed_cm_s = result.slope * env.bin_size  # for regular grids

    **Sampling implementation** (for method="sample"):

    Sampling is performed in cumulative-sum space to handle peaky posteriors:

    .. code-block:: python

        cumsum = np.cumsum(posterior, axis=1)
        u = rng.random((n_samples, n_time_bins, 1))
        samples = np.argmax(cumsum >= u, axis=-1)

    This avoids numerical issues with np.random.choice on posteriors
    with probabilities very close to 0 or 1.

    **Edge cases**:

    - **Constant positions**: If all decoded positions are identical, R² = 1.0
      (perfect fit to horizontal line). Check if slope ≈ 0 to detect this case.
    - **Constant times**: If all time values are identical, slope is undefined
      and set to 0.0 with intercept = mean position.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.decoding.trajectory import fit_linear_trajectory
    >>>
    >>> # Create environment and posterior with linear trajectory
    >>> positions = np.linspace(0, 100, 1000).reshape(-1, 1)
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> n_time_bins = 20
    >>> posterior = np.zeros((n_time_bins, env.n_bins))
    >>> for t in range(n_time_bins):
    ...     posterior[t, t * 2] = 1.0  # Linear trajectory
    >>> times = np.linspace(0, 1, n_time_bins)
    >>>
    >>> result = fit_linear_trajectory(env, posterior, times, method="map")
    >>> print(f"slope = {result.slope:.2f} bins/s, R² = {result.r_squared:.3f}")
    slope = 38.00 bins/s, R² = 1.000
    """
    if method not in ("map", "sample"):
        raise ValueError(f"method must be 'map' or 'sample', got {method!r}")

    # Input validation
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be 2D, got shape {posterior.shape}")

    n_time_bins = posterior.shape[0]

    if len(times) != n_time_bins:
        raise ValueError(
            f"times length ({len(times)}) must match posterior shape[0] ({n_time_bins})"
        )

    if method == "map":
        # Use argmax positions directly
        positions = np.argmax(posterior, axis=1).astype(np.float64)

        # Simple linear regression
        slope, intercept = _fit_line(times, positions)
        predicted = slope * times + intercept
        r_squared = _compute_r_squared(positions, predicted)

        return LinearFitResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            slope_std=None,
        )

    # method == "sample"
    # Set up RNG using shared helper for consistency
    from neurospatial.decoding.shuffle import _ensure_rng

    rng_gen = _ensure_rng(rng)

    # Sample positions from posterior using cumulative sum approach
    # This is numerically stable for peaky posteriors
    cumsum = np.cumsum(posterior, axis=1)

    # Generate uniform random values: (n_samples, n_time_bins, 1)
    u = rng_gen.random((n_samples, n_time_bins, 1))

    # Find bin indices where cumsum exceeds uniform values
    # Shape: (n_samples, n_time_bins)
    sampled_positions = np.argmax(cumsum >= u, axis=-1).astype(np.float64)

    # Fit line to each sample
    slopes = np.empty(n_samples)
    intercepts = np.empty(n_samples)

    for i in range(n_samples):
        slopes[i], intercepts[i] = _fit_line(times, sampled_positions[i])

    # Average coefficients
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    slope_std = np.std(slopes, ddof=1)

    # Compute R² using mean coefficients on expected (weighted mean) positions
    n_bins = posterior.shape[1]
    bin_indices = np.arange(n_bins)
    expected_positions = np.sum(posterior * bin_indices, axis=1)
    predicted = mean_slope * times + mean_intercept
    r_squared = _compute_r_squared(expected_positions, predicted)

    return LinearFitResult(
        slope=float(mean_slope),
        intercept=float(mean_intercept),
        r_squared=float(r_squared),
        slope_std=float(slope_std),
    )


def _fit_line(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Fit a line y = slope*x + intercept using least squares.

    Parameters
    ----------
    x : NDArray[np.float64]
        Independent variable.
    y : NDArray[np.float64]
        Dependent variable.

    Returns
    -------
    tuple[float, float]
        (slope, intercept) of the fitted line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        # All x values are the same - slope is undefined
        return 0.0, float(y_mean)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return float(slope), float(intercept)


def detect_trajectory_radon(
    posterior: NDArray[np.float64],
    *,
    theta_range: tuple[float, float] = (-90, 90),
    theta_step: float = 1.0,
) -> RadonDetectionResult:
    """Detect linear trajectory using Radon transform.

    Treats the posterior as a 2D image (time × position) and finds the line
    with maximum integrated probability mass. This is particularly useful
    for detecting diagonal stripe patterns in replay posteriors.

    Parameters
    ----------
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability distribution over spatial bins for each time bin.
        Treated as a 2D image where rows are time bins and columns are spatial bins.
    theta_range : tuple[float, float], optional
        Range of angles to search in degrees. Default is (-90, 90).
        - 0° corresponds to a horizontal line (constant position).
        - 90° corresponds to a vertical line (instantaneous jump).
        - Positive angles indicate forward replay.
        - Negative angles indicate reverse replay.
    theta_step : float, optional
        Angular resolution in degrees. Default is 1.0.
        Smaller values give finer angle resolution but take longer to compute.

    Returns
    -------
    RadonDetectionResult
        Container with angle_degrees, score, offset, and sinogram.

    Raises
    ------
    ImportError
        If scikit-image is not installed. Install with:
        ``pip install scikit-image`` or ``pip install neurospatial[trajectory]``

    See Also
    --------
    RadonDetectionResult : Container for Radon detection results.
    fit_linear_trajectory : For parametric linear fits.
    fit_isotonic_trajectory : For monotonic trajectory fits.

    Notes
    -----
    The Radon transform computes line integrals of the posterior at all
    specified angles. The angle with the highest integrated mass indicates
    the dominant trajectory direction.

    **Interpretation of angles:**

    The detected angle is in Radon space, where:

    - θ = 0° means the line is perpendicular to the first axis (time axis),
      i.e., a horizontal line in the posterior (constant position).
    - θ = 90° means the line is perpendicular to the second axis (position axis),
      i.e., a vertical line (instantaneous position change).
    - θ ≈ 45° typically indicates forward replay (position increases with time).
    - θ ≈ -45° typically indicates reverse replay (position decreases with time).

    **Time uniformity assumption:**

    Time bins are assumed to be uniformly spaced for image interpretation.
    For non-uniform times, consider interpolating the posterior to a uniform
    time grid first.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.decoding.trajectory import detect_trajectory_radon
    >>>
    >>> # Create posterior with diagonal pattern (forward replay)
    >>> n_time_bins, n_bins = 50, 50
    >>> posterior = np.zeros((n_time_bins, n_bins))
    >>> for t in range(n_time_bins):
    ...     posterior[t, t] = 1.0  # Perfect diagonal
    >>>
    >>> result = detect_trajectory_radon(posterior)
    >>> print(f"Detected angle: {result.angle_degrees:.1f}°")  # doctest: +SKIP
    Detected angle: 45.0°
    """
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for Radon transform trajectory detection. "
            "Install with: pip install scikit-image\n"
            "Or install neurospatial with trajectory extras: "
            "pip install neurospatial[trajectory]"
        )

    # Generate angle array
    theta_start, theta_end = theta_range
    # Use endpoint=False to get consistent behavior with theta_step
    n_angles = int((theta_end - theta_start) / theta_step)
    theta = np.linspace(theta_start, theta_end, n_angles, endpoint=False)

    # Compute Radon transform
    # The posterior is treated as an image with:
    # - rows = time bins (vertical axis in image)
    # - columns = spatial bins (horizontal axis in image)
    sinogram = radon(posterior, theta=theta, circle=False)

    # Find the peak in the sinogram
    # sinogram has shape (n_offsets, n_angles)
    peak_idx = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    offset_idx, angle_idx = peak_idx

    # Get the detected angle
    angle_degrees = float(theta[angle_idx])

    # Get the score (peak value)
    score = float(sinogram[offset_idx, angle_idx])

    # Compute offset in the original coordinate system
    # The offset index corresponds to the distance from the center
    n_offsets = sinogram.shape[0]
    # Center offset (Radon transform centers the projection)
    center_offset = (n_offsets - 1) / 2
    offset = float(offset_idx - center_offset)

    # Transpose sinogram to match (n_angles, n_offsets) convention
    # (scikit-image's radon() returns (n_offsets, n_angles))
    sinogram_transposed = sinogram.T

    return RadonDetectionResult(
        angle_degrees=angle_degrees,
        score=score,
        offset=offset,
        sinogram=sinogram_transposed,
    )


__all__ = [
    "IsotonicFitResult",
    "LinearFitResult",
    "RadonDetectionResult",
    "detect_trajectory_radon",
    "fit_isotonic_trajectory",
    "fit_linear_trajectory",
]
