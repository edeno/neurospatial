"""
GLM circular basis functions for parametric circular regression.

This module provides functions for fitting Generalized Linear Models (GLMs)
with circular predictors using sin/cos basis functions. This is the standard
parametric approach for analyzing circular tuning (head direction, theta phase,
running direction, circadian rhythms).

Which Function Should I Use?
----------------------------
**Create design matrix for GLM?**
    Use ``circular_basis()`` to get [sin(θ), cos(θ)] columns.

**Interpret GLM coefficients?**
    Use ``circular_basis_metrics()`` to get amplitude, phase, p-value.

**Quick significance check?**
    Use ``is_modulated()`` for a one-liner True/False.

**Visualize GLM fit?**
    Use ``plot_circular_basis_tuning()`` for polar or linear plots.

GLM Workflow
------------
1. Create design matrix: ``X = circular_basis(angles).design_matrix``
2. Fit GLM: ``model.fit(sm.add_constant(X), y)``
3. Extract coefficients: ``beta_sin, beta_cos = model.params[1:3]``
4. Get metrics: ``amplitude, phase, pval = circular_basis_metrics(...)``
5. Visualize: ``plot_circular_basis_tuning(beta_sin, beta_cos, ...)``

See Also
--------
neurospatial.metrics.circular : Core circular statistics (rayleigh_test, etc.)
neurospatial.metrics.head_direction : Head direction cell analysis
neurospatial.metrics.phase_precession : Phase precession analysis

References
----------
Kramer, M.A. & Eden, U.T. (2016). Case Studies in Neural Data Analysis. MIT Press.
    Chapter 11: Point Process Models of Spike-Field Coherence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from neurospatial.metrics.circular import _to_radians

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

__all__ = [
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "is_modulated",
    "plot_circular_basis_tuning",
]


# =============================================================================
# Data Classes
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


# =============================================================================
# Internal Helper Functions
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


# =============================================================================
# Public API - Circular Basis Functions
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
        Circular variable. Common use cases:

        - **Head direction**: Animal's facing direction (typically in degrees)
        - **Theta phase**: LFP phase at spike times (0 to 2π radians)
        - **Running direction**: Direction of movement
        - **Time of day**: Circadian phase

    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles. Use ``'deg'`` for head direction data that
        is typically recorded in degrees.

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
    is_modulated : Quick significance test for circular modulation.
    plot_circular_basis_tuning : Visualize fitted tuning curve.
    head_direction_tuning_curve : Non-parametric binned tuning curve.

    Notes
    -----
    **Why sin/cos basis?**

    For a circular predictor θ, using sin(θ) and cos(θ) as separate
    predictors allows the model to capture any phase and amplitude of circular
    modulation. The fitted coefficients (β_sin, β_cos) can be converted to:

    - Amplitude: sqrt(β_sin² + β_cos²) — modulation strength
    - Phase: atan2(β_sin, β_cos) — preferred direction/phase

    **GLM vs. Binned Tuning Curves**:

    - **Binned** (``head_direction_tuning_curve``): Non-parametric, intuitive,
      good for visualization. Requires many spikes per bin.
    - **GLM** (``circular_basis``): Parametric, statistically rigorous, can
      include covariates. Better for low spike counts.

    **GLM Workflow**:

    1. Create design matrix: ``X = circular_basis(angles).design_matrix``
    2. Fit GLM: ``model.fit(X, y)``
    3. Get metrics: ``amplitude, phase, pval = circular_basis_metrics(...)``
    4. Visualize: ``plot_circular_basis_tuning(beta_sin, beta_cos, ...)``

    Examples
    --------
    **Head direction tuning (GLM approach)**:

    >>> import numpy as np
    >>> from neurospatial.metrics import circular_basis
    >>> # Head direction at each spike (in degrees)
    >>> hd_at_spikes = np.array([45, 50, 48, 52, 180, 185, 170])  # degrees
    >>> result = circular_basis(hd_at_spikes, angle_unit="deg")
    >>> result.design_matrix.shape
    (7, 2)

    **Theta phase modulation**:

    >>> # LFP phase at each spike (in radians)
    >>> theta_phases = np.random.uniform(0, 2 * np.pi, 100)
    >>> result = circular_basis(theta_phases)  # angle_unit='rad' is default
    >>> X = result.design_matrix
    >>> # Now fit GLM: model.fit(sm.add_constant(X), spike_counts)
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
    >>> import numpy as np
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

    Returns
    -------
    bool
        True if BOTH:
        - p-value < alpha (statistically significant)
        - magnitude >= min_magnitude (practically meaningful)
        False otherwise.

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

    **Default thresholds**:

    - alpha=0.05: Standard significance level
    - min_magnitude=0.2: Weak but detectable modulation

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics import is_modulated
    >>> # Strong significant modulation
    >>> is_modulated(2.0, 2.0, np.array([[0.01, 0], [0, 0.01]]))
    True
    >>> # Weak (non-significant) modulation
    >>> is_modulated(0.1, 0.1, np.array([[1.0, 0], [0, 1.0]]))
    False
    """
    amplitude, _, pvalue = circular_basis_metrics(beta_sin, beta_cos, cov_matrix)

    # pvalue should never be None since cov_matrix is required, but handle it
    if pvalue is None:
        return False

    return pvalue < alpha and amplitude >= min_magnitude


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
        Intercept (baseline) coefficient from GLM. For Poisson GLM, this
        controls the baseline firing rate: exp(intercept).
    cov_matrix : ndarray, shape (2, 2), optional
        Covariance matrix for [beta_sin, beta_cos] from GLM fit.
        Required if ``show_ci=True`` to compute confidence bands.
    projection : {'polar', 'linear'}, default='polar'
        Plot projection type.
    n_points : int, default=100
        Number of points for smooth fitted curve.
    show_data : bool, default=False
        If True, overlay raw data points. Requires ``angles`` and ``rates``.

        **Rationale for default=False**: GLM visualization typically focuses
        on the smooth parametric fit. Raw data can obscure the model prediction
        and is available via ``plot_head_direction_tuning`` for binned analysis.
    show_fit : bool, default=True
        If True, show smooth curve from GLM fit.

        **Rationale for default=True**: The whole purpose of this function is
        to visualize the GLM fit. Showing the fit curve is the primary use case.
    show_ci : bool, default=False
        If True, show confidence band around fitted curve. Requires ``cov_matrix``.

        **Rationale for default=False**: Confidence bands require the covariance
        matrix from the GLM fit, which isn't always available. Users who want
        uncertainty visualization explicitly opt-in.
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
        If ``cov_matrix`` is not a 2x2 matrix.

    See Also
    --------
    circular_basis_metrics : Compute amplitude/phase from coefficients.
    circular_basis : Create design matrix for GLM.
    plot_head_direction_tuning : Non-parametric binned tuning curve visualization.

    Notes
    -----
    **GLM Model**:

    For a Poisson GLM with circular basis, the expected response is:

        λ(θ) = exp(intercept + β_cos*cos(θ) + β_sin*sin(θ))

    This can be rewritten as:

        λ(θ) = exp(intercept) * exp(R * cos(θ - φ))

    where:

    - R = sqrt(β_sin² + β_cos²) is the modulation amplitude
    - φ = atan2(β_sin, β_cos) is the preferred angle

    **When to use this vs. plot_head_direction_tuning**:

    - **GLM fit** (this function): Parametric, smooth curve from regression.
      Use when you've fit a GLM and want to show the model prediction.
    - **Binned tuning** (``plot_head_direction_tuning``): Non-parametric,
      shows actual data. Use for exploratory visualization.

    **Polar plot conventions**:

    - 0 at top (North): Uses ``theta_zero_location='N'``
    - Clockwise direction: Uses ``theta_direction=-1``

    Examples
    --------
    **Head direction GLM tuning curve**:

    >>> import numpy as np
    >>> from neurospatial.metrics import plot_circular_basis_tuning
    >>> # After fitting GLM to head direction data
    >>> # Coefficients: preferred direction ~45° with strong modulation
    >>> beta_sin, beta_cos = 0.7, 0.7  # atan2(0.7, 0.7) ≈ 45°
    >>> ax = plot_circular_basis_tuning(
    ...     beta_sin, beta_cos, intercept=2.0
    ... )  # doctest: +SKIP

    **Theta phase modulation**:

    >>> # After fitting GLM to theta phase data
    >>> # Coefficients: preferred phase near trough (π) with moderate modulation
    >>> beta_sin, beta_cos = 0.0, -0.5  # atan2(0, -0.5) = π
    >>> ax = plot_circular_basis_tuning(
    ...     beta_sin, beta_cos, intercept=1.5, projection="linear"
    ... )  # doctest: +SKIP

    **With binned data overlay**:

    >>> bin_centers = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    >>> rates = np.exp(2 + 0.7 * np.cos(bin_centers - np.pi / 4))  # Simulated
    >>> ax = plot_circular_basis_tuning(
    ...     0.7, 0.7, angles=bin_centers, rates=rates, intercept=2.0, show_data=True
    ... )  # doctest: +SKIP

    **With 95% confidence band (shows uncertainty)**:

    >>> # Covariance matrix from GLM fit
    >>> cov = np.array(
    ...     [[0.01, 0.002], [0.002, 0.01]]
    ... )  # From model.cov_params()[1:3, 1:3]
    >>> ax = plot_circular_basis_tuning(
    ...     0.7, 0.7, intercept=2.0, cov_matrix=cov, show_ci=True, ci=0.95
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
    # For visualization, we use the linear predictor
    linear_pred = (
        intercept + beta_cos * np.cos(theta_smooth) + beta_sin * np.sin(theta_smooth)
    )
    # Transform to response scale (Poisson GLM uses exp link)
    fitted_response = np.exp(linear_pred)

    # Compute confidence bands if requested
    ci_lower: NDArray[np.float64] | None = None
    ci_upper: NDArray[np.float64] | None = None

    if show_ci and cov_matrix is not None:
        # Compute standard error of linear predictor using delta method
        # linear_pred = intercept + beta_cos*cos(θ) + beta_sin*sin(θ)
        # Gradient w.r.t. [beta_sin, beta_cos] is [sin(θ), cos(θ)]
        # SE^2 = gradient.T @ cov @ gradient

        # Design matrix for each angle: [sin(θ), cos(θ)]
        design = np.column_stack([np.sin(theta_smooth), np.cos(theta_smooth)])

        # Variance of linear predictor at each angle
        var_linear = np.sum((design @ cov_matrix) * design, axis=1)
        se_linear = np.sqrt(np.maximum(var_linear, 0))  # Ensure non-negative

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
        # Import PolarAxes at runtime for cast
        polar_ax = cast("PolarAxes", ax)

        # Configure polar plot: 0° at top (North), clockwise direction
        polar_ax.set_theta_zero_location("N")
        polar_ax.set_theta_direction(-1)

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
