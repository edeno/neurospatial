# SCRATCHPAD - Circular Basis Functions Implementation

## Current Status

**Completed**: M0, M1
**Next**: M2 - is_modulated() convenience function

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

## Decisions

1. Implementation now matches documentation (radians default)
2. Tests explicitly pass `angle_unit="deg"` so no behavior changes
3. Following TDD: red-green-refactor cycle
4. CircularBasisResult stores angles in radians internally
5. Wald test uses `np.linalg.solve` instead of `inv` for numerical stability

## Next Steps

1. Commit M1 changes
2. Start M2: Add is_modulated() convenience function

## Blockers

None currently.

## Notes

- The module docstring already claimed radians was default, so fixing implementation aligned with docs
- All tests in test_head_direction.py explicitly use `angle_unit="deg"`, so the default change doesn't affect them
- CircularBasisResult follows same pattern as HeadDirectionMetrics and PhasePrecessionResult
