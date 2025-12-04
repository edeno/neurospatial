# SCRATCHPAD - Circular Basis Functions Implementation

## Current Status

**Completed**: M0, M1, M2
**Next**: M3 - Visualization (plot_circular_basis_tuning)

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

## Next Steps

1. Commit M2 changes
2. Start M3: Add plot_circular_basis_tuning() visualization

## Blockers

None currently.

## Notes

- The module docstring already claimed radians was default, so fixing implementation aligned with docs
- All tests in test_head_direction.py explicitly use `angle_unit="deg"`, so the default change doesn't affect them
- CircularBasisResult follows same pattern as HeadDirectionMetrics and PhasePrecessionResult
- `is_modulated()` follows pattern of `is_head_direction_cell()` for convenience boolean functions
