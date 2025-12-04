# Circular Statistics Implementation - Scratchpad

**Started:** 2025-12-03
**Current Status:** Milestone 1.1 complete

---

## Session Notes

### 2025-12-03: Initial Implementation

**Completed:**
- Created `src/neurospatial/metrics/circular.py` with full module structure
- Implemented internal helper functions:
  - `_to_radians()` - angle unit conversion
  - `_mean_resultant_length()` - with scipy feature detection and fallback
  - `_validate_circular_input()` - comprehensive validation with diagnostics
  - `_validate_paired_input()` - paired array validation
- Created test file `tests/metrics/test_circular.py` with 16 tests
- All tests passing, ruff and mypy clean

**Design Decisions:**
1. Using scipy.stats.directional_stats when available (scipy >= 1.9.0)
2. Fallback implementation for weighted mean resultant length (scipy doesn't support weights)
3. Comprehensive error messages with diagnostic steps (following neurospatial patterns)
4. Warnings for data quality issues (NaN removal, angle wrapping)

**Next Steps:**
- Implement `circular_linear_correlation()` (Milestone 1.3)
- Write tests first (TDD)

### 2025-12-03: Rayleigh Test Implementation

**Completed:**
- Implemented `rayleigh_test()` with:
  - Finite-sample correction (Mardia & Jupp, p. 94)
  - Weighted version with effective sample size
  - Degree/radian support
- Added 10 tests for Rayleigh test
- Exported from `neurospatial.metrics`
- All 26 tests passing, ruff and mypy clean

### 2025-12-03: Circular-Linear Correlation Implementation

**Completed:**
- Implemented `circular_linear_correlation()` using Mardia & Jupp formula:
  - r^2 = (r_xs^2 + r_xc^2 - 2*r_xs*r_xc*r_cs) / (1 - r_cs^2)
  - Uses scipy.stats.pearsonr for component correlations
  - P-value from chi-squared(2) distribution
- Implemented `phase_position_correlation()` as alias
- Added 13 tests (11 for circular_linear_correlation, 2 for phase_position_correlation)
- All 39 tests passing, ruff and mypy clean
- Exported from `neurospatial.metrics`

**Design Decisions:**
1. The Mardia-Jupp circular-linear correlation has a theoretical maximum less than 1.0
   for a single-cycle linear relationship (~0.755 for phases spanning 0 to 2π). This is
   a known characteristic of this correlation measure due to sine/cosine transformations.
2. Handles degenerate cases:
   - Constant linear variable → warns and returns r=0, p=1.0
   - cos/sin perfectly correlated (limited angle range) → warns and returns r=0, p=1.0
3. Correlation is always non-negative (measures strength, not direction)

**Next Steps:**
- Implement `circular_circular_correlation()` (Milestone 1.4)
- Write tests first (TDD)

---

## Blockers

None currently.

---

## Open Questions

None currently.
