# SCRATCHPAD - Circular Basis Functions Implementation

## Current Status

**Completed**: M0, M1, M2, M3, M4, M5, M6
**Status**: All milestones complete! Feature complete for circular basis functions with uncertainty visualization.

## Completed Tasks

### M0: Fix head_direction.py Default to Radians ✓

- [x] M0.1: Updated `head_direction_tuning_curve()` default from `"deg"` to `"rad"`
- [x] M0.2: Renamed `angle_display_unit` to `angle_unit` in `plot_head_direction_tuning()`
- [x] M0.2: Changed `plot_head_direction_tuning()` default from `"deg"` to `"rad"`
- [x] M0.3: Module docstring already had correct documentation at lines 46-50
- [x] M0.4: Updated tests to use `angle_unit` instead of `angle_display_unit`
- [x] M0.5: All 76 tests pass, ruff and mypy clean

### M1: Implement Circular Basis Functions ✓

- [x] Added `dataclass` import and `scipy.stats.chi2` import
- [x] Created `CircularBasisResult` dataclass with sin_component, cos_component, angles
- [x] Added `design_matrix` property returning (n_samples, 2) array
- [x] Implemented `circular_basis()` function
- [x] Implemented `_wald_test_magnitude()` helper for p-value calculation
- [x] Implemented `circular_basis_metrics()` function
- [x] Updated `__all__` in circular.py
- [x] Updated exports in `metrics/__init__.py`
- [x] Added 20 new tests (all passing)
- [x] All 79 circular tests pass, ruff and mypy clean

### M2: Implement is_modulated() Convenience Function ✓

- [x] Implemented `is_modulated(beta_sin, beta_cos, cov_matrix, *, alpha=0.05, min_magnitude=0.2)`
- [x] Returns True if BOTH statistically significant (p < alpha) AND practically meaningful (amplitude >= min_magnitude)
- [x] Added to `__all__` in circular.py and metrics/__init__.py
- [x] Added 10 tests including edge cases:
  - zero coefficients
  - singular covariance matrix
  - custom alpha/min_magnitude thresholds
- [x] Code review passed with approval
- [x] All 87 circular tests pass, ruff and mypy clean

## Decisions

1. Implementation now matches documentation (radians default)
2. Tests explicitly pass `angle_unit="deg"` so no behavior changes
3. Following TDD: red-green-refactor cycle
4. CircularBasisResult stores angles in radians internally
5. Wald test uses `np.linalg.solve` instead of `inv` for numerical stability
6. `is_modulated()` takes `beta_sin`, `beta_cos` separately (not packed array) - matches API of `circular_basis_metrics()`
7. `is_modulated()` returns False for NaN p-values (singular covariance) due to NaN comparison semantics
8. Confidence bands use delta method: Var(X @ beta) = X @ Cov @ X.T, then exp() link for Poisson GLM

### M3: Implement plot_circular_basis_tuning() Visualization ✓

- [x] Added matplotlib imports (TYPE_CHECKING for Axes, PolarAxes; Any, cast)
- [x] Implemented `plot_circular_basis_tuning(beta_sin, beta_cos, ...)` function
- [x] Parameters: intercept, projection, n_points, show_data, show_fit, color, data_color, data_alpha, line_kwargs, scatter_kwargs
- [x] Added to `__all__` in circular.py and metrics/__init__.py
- [x] Added 12 tests (all passing)
- [x] Code review passed with approval
- [x] All 101 circular tests pass, ruff and mypy clean

### M4 & M5: Tests and Integration ✓

- [x] All test classes existed from M1/M2 (TestCircularBasis, TestCircularBasisMetrics, TestIsModulated)
- [x] Added TestPlotCircularBasisTuning (12 tests) in M3
- [x] All 101 circular tests passing
- [x] All exports working from `neurospatial.metrics`
- [x] mypy and ruff clean

### Documentation Enhancement (M5.6) ✓

- [x] Enhanced module docstring with "GLM-based circular regression" guidance
- [x] Listed multi-domain use cases: head direction, theta phase, running direction, circadian
- [x] Enhanced `circular_basis()` docstring with domain examples and GLM vs binned comparison
- [x] Enhanced `plot_circular_basis_tuning()` docstring with domain examples
- [x] Added cross-references between functions

### M6: Uncertainty Visualization ✓

- [x] Added confidence band parameters to `plot_circular_basis_tuning()`:
  - `cov_matrix`: 2x2 covariance matrix for beta_sin, beta_cos
  - `show_ci`: bool to enable confidence bands (default False)
  - `ci`: confidence level (default 0.95)
  - `ci_alpha`: transparency for fill_between (default 0.3)
- [x] Implemented delta method for computing standard errors of linear predictor
- [x] Applied exp() link function to get confidence bands on rate scale
- [x] Added validation: `show_ci=True` requires `cov_matrix`
- [x] Added validation: `cov_matrix` must be shape (2, 2)
- [x] Works with both polar and linear projections
- [x] Added 6 new tests (18 total for TestPlotCircularBasisTuning)
- [x] All 107 circular tests pass, ruff and mypy clean

## Blockers

None - all milestones complete!

## Notes

- The module docstring already claimed radians was default, so fixing implementation aligned with docs
- All tests in test_head_direction.py explicitly use `angle_unit="deg"`, so the default change doesn't affect them
- CircularBasisResult follows same pattern as HeadDirectionMetrics and PhasePrecessionResult
- `is_modulated()` follows pattern of `is_head_direction_cell()` for convenience boolean functions
