# SCRATCHPAD - Circular Basis Functions Implementation

## Current Status
**Completed**: M0 - Fix head_direction.py defaults to radians
**Next**: M1 - Core Functions (circular_basis, CircularBasisResult, circular_basis_metrics)

## Completed Tasks

### M0: Fix head_direction.py Default to Radians ✓
- [x] M0.1: Updated `head_direction_tuning_curve()` default from `"deg"` to `"rad"`
- [x] M0.2: Renamed `angle_display_unit` to `angle_unit` in `plot_head_direction_tuning()`
- [x] M0.2: Changed `plot_head_direction_tuning()` default from `"deg"` to `"rad"`
- [x] M0.3: Module docstring already had correct documentation at lines 46-50
- [x] M0.4: Updated tests to use `angle_unit` instead of `angle_display_unit`
- [x] M0.5: All 76 tests pass, ruff and mypy clean

## Decisions

1. Implementation now matches documentation (radians default)
2. Tests explicitly pass `angle_unit="deg"` so no behavior changes
3. Following TDD: red-green-refactor cycle

## Next Steps
1. Commit M0 changes
2. Start M1: Add circular basis functions to circular.py

## Blockers
None currently.

## Notes
- The module docstring already claimed radians was default, so fixing implementation aligned with docs
- All tests in test_head_direction.py explicitly use `angle_unit="deg"`, so the default change doesn't affect them
