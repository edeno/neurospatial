# Plan: Circular Basis Functions for GLM Design Matrices

## Overview

Add circular basis functions to the existing `neurospatial.metrics.circular` module (NOT a new module). This provides convenience functions for constructing cosine/sine design matrices for GLM fitting. Users can use these with any GLM framework (statsmodels, scikit-learn, PyTorch, etc.).

**Key change**: Also fix `head_direction.py` to default to radians for consistency across all circular modules.

**Key insight from Mark Kramer's tutorial**: The GLM framework uses a trigonometric basis:

```
λ_t = exp(β₀ + β₁·cos(φ(t)) + β₂·sin(φ(t)))
```

This decouples overall firing rate (β₀) from phase modulation (β₁, β₂). The magnitude and preferred phase can be recovered from the fitted coefficients:

- **Magnitude**: `sqrt(β₁² + β₂²)` - modulation strength
- **Preferred phase**: `arctan2(β₂, β₁)` - peak phase

## Design Principles

1. **No GLM dependency**: We provide the design matrix and coefficient interpretation, users fit the model
2. **Mirror head direction module structure**: `CircularBasisResult` dataclass, convenience functions, plotting
3. **Reuse existing helpers**: `_to_radians()`, `_mean_resultant_length()`, `_validate_circular_input()` from `circular.py`
4. **Support multiple harmonics**: Allow higher-order harmonics (cos(2φ), sin(2φ), etc.) for more flexible tuning
5. **Task-oriented naming**: Functions named for what users DO, not how it's implemented
6. **Consistent parameters**: Use `include_intercept` everywhere (not `has_intercept`)

## Module Structure

```
src/neurospatial/metrics/
├── circular.py           # UPDATED: Add circular basis functions here
├── head_direction.py     # UPDATED: Change angle_unit default to 'rad'
├── phase_precession.py   # Existing phase precession
└── __init__.py           # Update exports
```

**Rationale for merging into `circular.py`**:

- Basis functions are core circular statistics, not a separate domain
- Reduces module fragmentation (3 modules instead of 4)
- Clearer mental model: `circular.py` = all fundamental circular operations

---

## Part 0: Fix `head_direction.py` Default to Radians

**Change `angle_unit` default from `'deg'` to `'rad'`** in all functions:

```python
# head_direction.py - BEFORE
def head_direction_tuning_curve(
    ...
    angle_unit: Literal["rad", "deg"] = "deg",  # OLD
):

# head_direction.py - AFTER
def head_direction_tuning_curve(
    ...
    angle_unit: Literal["rad", "deg"] = "rad",  # NEW - consistent with other modules
):
```

**Functions to update**:

- `head_direction_tuning_curve()` (line ~204)
- `plot_head_direction_tuning()` - rename `angle_display_unit` to `angle_unit` (line ~606)

**Add warning to module docstring**:

```python
"""
...

Angle Units
-----------
All functions default to radians (``angle_unit='rad'``) for consistency with
scipy and other neurospatial.metrics functions.

**Head direction research commonly uses degrees.** If your data is in degrees::

    head_direction_tuning_curve(hd, spikes, times, angle_unit='deg')

...
"""
```

---

## Part 1: Update `circular.py` Module Header

Add to the existing "Which Function Should I Use?" section:

```python
"""
...

**GLM-based circular regression?**
    Use ``circular_basis()`` to create design matrix [1, cos(φ), sin(φ), ...].
    Fit with statsmodels/scikit-learn, then interpret with ``circular_basis_metrics()``.

...
"""
```

Add new section after existing docstring content:

```python
"""
...

GLM Basis Functions
-------------------
For parametric analysis using Generalized Linear Models:

1. Create design matrix::

    >>> X = circular_basis(theta_phases)

2. Fit GLM (statsmodels example)::

    >>> import statsmodels.api as sm
    >>> model = sm.GLM(spike_counts, X, family=sm.families.Poisson())
    >>> result = model.fit()

3. Interpret coefficients::

    >>> metrics = circular_basis_metrics(
    ...     result.params, covariance_matrix=result.cov_params()
    ... )
    >>> print(metrics)  # Human-readable: preferred phase, modulation, significance

**Framework-specific guidance**:

- statsmodels: Use ``include_intercept=True`` (statsmodels doesn't add one)
- scikit-learn: Use ``include_intercept=False`` (sklearn adds intercept automatically)

References
----------
...
Kramer, M.A. & Eden, U.T. (2016). Case Studies in Neural Data Analysis. MIT Press.
    Chapter 11: Point Process Models of Spike-Field Coherence.
"""
```

---

## Part 2: Core Functions

### 2.1 Design Matrix Construction

```python
def circular_basis(
    angles: NDArray[np.float64],
    *,
    n_harmonics: int = 1,
    include_intercept: bool = True,
    angle_unit: Literal["rad", "deg"] = "rad",
) -> NDArray[np.float64]:
    """
    Construct cosine/sine basis design matrix for GLM fitting.

    Creates [1, cos(φ), sin(φ), cos(2φ), sin(2φ), ...] suitable for
    fitting circular-linear relationships with any GLM package.

    Parameters
    ----------
    angles : array, shape (n,)
        Circular variable (e.g., LFP theta phase, head direction).
    n_harmonics : int, default=1
        Number of harmonics. Start with 1; increase only if model
        comparison (AIC/BIC) justifies it.
    include_intercept : bool, default=True
        Include column of ones for intercept term.

        **Framework-specific guidance**:
        - statsmodels: Use True (statsmodels does NOT add intercept)
        - scikit-learn: Use False (scikit-learn adds intercept automatically)

    angle_unit : {'rad', 'deg'}, default='rad'
        Unit of input angles.

    Returns
    -------
    X : array, shape (n, 1 + 2*n_harmonics) or (n, 2*n_harmonics)
        Design matrix. Columns: [1, cos(φ), sin(φ), ...] if intercept.

    Examples
    --------
    **With statsmodels**::

        >>> X = circular_basis(phases)  # include_intercept=True
        >>> model = sm.GLM(spike_counts, X, family=sm.families.Poisson()).fit()

    **With scikit-learn**::

        >>> X = circular_basis(phases, include_intercept=False)  # sklearn adds it
        >>> model = PoissonRegressor().fit(X, spike_counts)
    """
```

### 2.2 Coefficient Interpretation

```python
@dataclass
class CircularBasisResult:
    """
    Interpretation of fitted circular basis coefficients.

    Attributes
    ----------
    harmonic_magnitudes : list[float]
        Modulation strength for each harmonic [R₁, R₂, ...].
        R = sqrt(β_cos² + β_sin²). Higher = stronger modulation.
    harmonic_phases : list[float]
        Preferred phase for each harmonic [φ₁, φ₂, ...] in radians.
        φ = arctan2(β_sin, β_cos).
    intercept : float | None
        Baseline coefficient β₀. None if no intercept was fitted.
    is_significant : bool
        True if modulation is statistically significant.
        False if covariance_matrix was not provided.
    pval : float | None
        P-value for H0: no modulation. None if covariance not provided.

    Notes
    -----
    For a single-harmonic Poisson GLM:

        λ(φ) = exp(β₀) · exp(R·cos(φ - φ_pref))

    where R = magnitude and φ_pref = preferred_angle.
    """

    harmonic_magnitudes: list[float]
    harmonic_phases: list[float]
    intercept: float | None
    is_significant: bool
    pval: float | None

    @property
    def magnitude(self) -> float:
        """Modulation strength of first harmonic."""
        return self.harmonic_magnitudes[0]

    @property
    def preferred_angle(self) -> float:
        """Preferred phase/direction of first harmonic (radians)."""
        return self.harmonic_phases[0]

    @property
    def preferred_angle_deg(self) -> float:
        """Preferred phase/direction in degrees."""
        return float(np.degrees(self.harmonic_phases[0]))

    def interpretation(self) -> str:
        """Human-readable interpretation."""
        lines = []

        # Significance
        if self.pval is None:
            lines.append("Significance: CANNOT TEST (no covariance matrix provided)")
            lines.append("  To test: metrics = circular_basis_metrics(coefs, covariance_matrix=...)")
        elif self.is_significant:
            lines.append(f"SIGNIFICANT modulation (p={self.pval:.4f})")
        else:
            lines.append(f"No significant modulation (p={self.pval:.4f})")

        # Preferred angle and magnitude
        lines.append(f"Preferred angle: {self.preferred_angle_deg:.1f}°")
        lines.append(f"Modulation strength: {self.magnitude:.3f}")

        # Higher harmonics if present
        if len(self.harmonic_magnitudes) > 1:
            lines.append("Higher harmonics:")
            for i, (mag, phase) in enumerate(
                zip(self.harmonic_magnitudes[1:], self.harmonic_phases[1:], strict=True)
            ):
                lines.append(f"  Harmonic {i+2}: magnitude={mag:.3f}, phase={np.degrees(phase):.1f}°")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.interpretation()


def circular_basis_metrics(
    coefficients: NDArray[np.float64],
    *,
    n_harmonics: int = 1,
    include_intercept: bool = True,
    covariance_matrix: NDArray[np.float64] | None = None,
) -> CircularBasisResult:
    """
    Compute magnitude and preferred angle from fitted GLM coefficients.

    Parameters
    ----------
    coefficients : array
        Fitted coefficients from GLM. Expected order:
        [β₀, β₁_cos, β₁_sin, β₂_cos, β₂_sin, ...] if include_intercept=True
        [β₁_cos, β₁_sin, β₂_cos, β₂_sin, ...] if include_intercept=False
    n_harmonics : int, default=1
        Number of harmonics that were fitted.
    include_intercept : bool, default=True
        Whether first coefficient is intercept.
    covariance_matrix : array, optional
        Covariance matrix for significance testing (Wald test).
        If None, pval=None and is_significant=False.

    Returns
    -------
    CircularBasisResult
        Dataclass with magnitude, preferred_angle, and significance.

    Raises
    ------
    ValueError
        If coefficients length doesn't match expected (with helpful message).

    Examples
    --------
    >>> coefs = result.params  # [β₀, β_cos, β_sin]
    >>> cov = result.cov_params()
    >>> metrics = circular_basis_metrics(coefs, covariance_matrix=cov)
    >>> print(metrics)
    SIGNIFICANT modulation (p=0.0032)
    Preferred angle: 45.2°
    Modulation strength: 0.82
    """
    # Validate coefficient length
    expected_len = (1 if include_intercept else 0) + 2 * n_harmonics
    if len(coefficients) != expected_len:
        raise ValueError(
            f"coefficients length {len(coefficients)} doesn't match expected {expected_len}.\n"
            f"Expected: {'1 (intercept) + ' if include_intercept else ''}{2 * n_harmonics} (harmonics).\n\n"
            f"Common causes:\n"
            f"  - Used include_intercept={'not ' + str(include_intercept)} in circular_basis()\n"
            f"  - Used different n_harmonics value\n\n"
            f"Fix: Match parameters between circular_basis() and circular_basis_metrics()."
        )
```

### 2.3 Convenience Function for Quick Check

```python
def is_modulated(
    coefficients: NDArray[np.float64],
    covariance_matrix: NDArray[np.float64],
    *,
    alpha: float = 0.05,
    min_magnitude: float = 0.2,
    include_intercept: bool = True,
    n_harmonics: int = 1,
) -> bool:
    """
    Quick check: Do these coefficients show significant circular modulation?

    Parameters
    ----------
    coefficients : array
        Fitted GLM coefficients.
    covariance_matrix : array
        Covariance matrix from GLM fit (required for significance test).
    alpha : float, default=0.05
        Significance level.
    min_magnitude : float, default=0.2
        Minimum modulation magnitude required.
    include_intercept : bool, default=True
        Whether first coefficient is intercept.
    n_harmonics : int, default=1
        Number of harmonics that were fitted.

    Returns
    -------
    bool
        True if p < alpha AND magnitude >= min_magnitude.

    See Also
    --------
    circular_basis_metrics : For full analysis with all metrics.
    """
    metrics = circular_basis_metrics(
        coefficients,
        n_harmonics=n_harmonics,
        include_intercept=include_intercept,
        covariance_matrix=covariance_matrix,
    )
    return metrics.is_significant and metrics.magnitude >= min_magnitude
```

---

## Part 3: Visualization

### 3.1 Tuning Curve from Coefficients

```python
def plot_circular_basis_tuning(
    coefficients: NDArray[np.float64],
    angles: NDArray[np.float64] | None = None,
    rates: NDArray[np.float64] | None = None,
    metrics: CircularBasisResult | None = None,
    ax: Axes | PolarAxes | None = None,
    *,
    n_harmonics: int = 1,
    include_intercept: bool = True,
    projection: Literal["polar", "linear"] = "polar",
    n_points: int = 100,
    show_data: bool = True,
    show_fit: bool = True,
    color: str = "C0",
    **kwargs,
) -> Axes:
    """
    Plot tuning curve from GLM coefficients with optional data overlay.

    Parameters
    ----------
    coefficients : array
        Fitted GLM coefficients.
    angles, rates : array, optional
        Raw binned data to overlay. Required if show_data=True.
    metrics : CircularBasisResult, optional
        If provided, mark preferred direction and show metrics text.
    ax : Axes, optional
        Axes to plot on. Creates new figure if None.
    n_harmonics : int, default=1
        Number of harmonics fitted.
    include_intercept : bool, default=True
        Whether first coefficient is intercept.
    projection : {'polar', 'linear'}, default='polar'
        Plot type.
    n_points : int, default=100
        Number of points for smooth curve.
    show_data : bool, default=True
        If True, overlay raw data points. Requires angles and rates.
    show_fit : bool, default=True
        If True, show smooth curve from GLM fit.
    color : str, default='C0'
        Color for curve and markers.

    Returns
    -------
    Axes
        The axes object with the plot.

    Raises
    ------
    ValueError
        If show_data=True but angles or rates not provided.

    Examples
    --------
    >>> # Show model fit over raw data
    >>> bin_centers, firing_rates = head_direction_tuning_curve(...)
    >>> plot_circular_basis_tuning(
    ...     coeffs, angles=bin_centers, rates=firing_rates, metrics=metrics
    ... )
    """
    if show_data and (angles is None or rates is None):
        raise ValueError(
            "show_data=True requires both angles and rates arguments.\n\n"
            "To show model fit over raw data:\n"
            "  plot_circular_basis_tuning(coeffs, angles=bins, rates=rates)\n\n"
            "To show only the fitted curve:\n"
            "  plot_circular_basis_tuning(coeffs, show_data=False)"
        )
```

---

## Part 4: Tests

### 4.1 Test File Structure

Add to existing `tests/metrics/test_circular.py`:

```python
# Test categories (~25 tests):

class TestCircularBasis:
    """Test design matrix construction."""
    - test_single_harmonic_shape  # (n,) -> (n, 3) with intercept
    - test_multiple_harmonics_shape  # n_harmonics=2 -> (n, 5)
    - test_without_intercept  # (n,) -> (n, 2)
    - test_degree_input  # angle_unit='deg' converts correctly
    - test_values_correct  # cos/sin computed correctly
    - test_empty_input_raises  # Empty array -> ValueError
    - test_nan_handling  # NaN in input -> validated


class TestCircularBasisMetrics:
    """Test coefficient interpretation."""
    - test_pure_cosine_modulation  # [0, 1, 0] -> magnitude=1, phase=0
    - test_pure_sine_modulation  # [0, 0, 1] -> magnitude=1, phase=π/2
    - test_45_degree_preference  # [0, 1, 1] -> phase=π/4
    - test_no_modulation  # [0, 0, 0] -> magnitude=0
    - test_with_covariance_significant  # p < 0.05
    - test_without_covariance  # pval=None, is_significant=False
    - test_multiple_harmonics  # harmonic_magnitudes has correct length
    - test_coefficient_length_mismatch_raises  # Helpful error message
    - test_interpretation_string  # __str__ returns readable text


class TestIsModulated:
    """Test convenience function."""
    - test_significant_strong_modulation  # True
    - test_not_significant  # p > 0.05 -> False
    - test_weak_modulation_below_threshold  # magnitude < 0.2 -> False


class TestPlotCircularBasisTuning:
    """Test visualization."""
    - test_polar_plot_creates_figure
    - test_linear_plot_creates_figure
    - test_show_data_requires_angles_rates  # ValueError if missing
    - test_show_fit_only  # show_data=False works
    - test_preferred_direction_marker  # When metrics provided
```

---

## Part 5: Exports

Update `src/neurospatial/metrics/__init__.py`:

```python
from neurospatial.metrics.circular import (
    # ... existing exports ...
    CircularBasisResult,
    circular_basis,
    circular_basis_metrics,
    is_modulated,
    plot_circular_basis_tuning,
)

__all__ = [
    # ... existing exports ...
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "is_modulated",
    "plot_circular_basis_tuning",
]
```

Update `src/neurospatial/metrics/circular.py` `__all__`:

```python
__all__ = [
    # Existing
    "circular_circular_correlation",
    "circular_linear_correlation",
    "phase_position_correlation",
    "rayleigh_test",
    # New
    "CircularBasisResult",
    "circular_basis",
    "circular_basis_metrics",
    "is_modulated",
    "plot_circular_basis_tuning",
]
```

---

## Implementation Order

1. **M0: Fix `head_direction.py`** - Change `angle_unit` default to `'rad'`, update docstrings
2. **M1: Core functions** - Add to `circular.py`: `circular_basis`, `CircularBasisResult`, `circular_basis_metrics`
3. **M2: Convenience function** - Add to `circular.py`: `is_modulated`
4. **M3: Visualization** - Add to `circular.py`: `plot_circular_basis_tuning`
5. **M4: Tests** - Add to `tests/metrics/test_circular.py` (~25 new tests)
6. **M5: Integration** - Update exports in `__init__.py`, verify imports work

---

## Success Criteria

1. All tests pass: `uv run pytest tests/metrics/test_circular.py -v`
2. Type checks pass: `uv run mypy src/neurospatial/metrics/circular.py src/neurospatial/metrics/head_direction.py`
3. Linting passes: `uv run ruff check src/neurospatial/metrics/`
4. `head_direction.py` uses radians by default (breaking change documented)
5. Example workflow runs:

```python
from neurospatial.metrics import circular_basis, circular_basis_metrics
import statsmodels.api as sm
import numpy as np

# User's spike phase data
phases = np.random.uniform(0, 2*np.pi, 1000)
spike_counts = np.random.poisson(5, 1000)

# Create design matrix
X = circular_basis(phases)

# Fit GLM
model = sm.GLM(spike_counts, X, family=sm.families.Poisson())
result = model.fit()

# Interpret coefficients
metrics = circular_basis_metrics(result.params, covariance_matrix=result.cov_params())
print(metrics)  # Shows preferred phase, magnitude, significance
```

---

## Dependencies

**Required (already in project):**

- numpy
- scipy (for Wald test significance)
- matplotlib

**No new dependencies needed.**

---

## Implementation Notes

### Wald Test for Magnitude Significance

The p-value tests H0: magnitude = 0 using the delta method:

```python
def _wald_test_magnitude(
    coef_cos: float,
    coef_sin: float,
    cov_matrix: NDArray[np.float64],
    offset: int,
) -> float:
    """Test H0: magnitude = 0 using delta method."""
    from scipy import stats

    mag = np.sqrt(coef_cos**2 + coef_sin**2)
    if mag < 1e-10:
        return 1.0  # No modulation

    # Gradient of sqrt(cos^2 + sin^2) wrt [cos, sin]
    grad = np.array([coef_cos / mag, coef_sin / mag])

    # Variance via delta method: var(f(x)) ≈ grad' @ cov @ grad
    cov_slice = cov_matrix[offset : offset + 2, offset : offset + 2]
    var = grad @ cov_slice @ grad

    # Wald statistic: z = estimate / SE
    z = mag / np.sqrt(var)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(pval)
```

---

## Changes from Initial Plan (Based on Review Feedback)

1. **Merged into `circular.py`** - No new module; basis functions are core circular stats
2. **Fixed `angle_unit` inconsistency** - `head_direction.py` now defaults to `'rad'` like all other modules
3. **Removed redundant fields** - `CircularBasisResult` uses properties for `magnitude`/`preferred_angle` instead of storing twice
4. **Renamed `is_phase_modulated()` to `is_modulated()`** - Generic name matches generic module
5. **Deleted head direction integration (Part 3)** - Scope creep; users can call `circular_basis()` directly
6. **Added "Which Function Should I Use?"** - Navigation section in module docstring
7. **Added "Before You Start"** - GLM package requirements and framework-specific guidance
8. **Added coefficient length validation** - Helpful error message for common mistake
9. **Made `pval` nullable** - Explicit when covariance not provided
10. **Consistent parameter naming** - `include_intercept` everywhere
11. **`show_data=True` default** - Plot validates and shows data by default
12. **Renamed `angle_display_unit` to `angle_unit`** - Consistent naming in `plot_head_direction_tuning()`
