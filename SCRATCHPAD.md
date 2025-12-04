# Circular Statistics Implementation - Scratchpad

**Started:** 2025-12-03
**Current Status:** Milestone 2.2 complete (phase precession analysis)

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
- Implement `PhasePrecessionResult` dataclass (Milestone 2.1)
- Write tests first (TDD)

### 2025-12-03: Circular-Circular Correlation Implementation

**Completed:**
- Implemented `circular_circular_correlation()` using Fisher & Lee (1983) formula:
  - rho = sum(sin(a1-mean1) * sin(a2-mean2)) / sqrt(sum(sin(a1-mean1)^2) * sum(sin(a2-mean2)^2))
  - Uses scipy.stats.circmean for circular means
  - P-value from normal approximation (Jammalamadaka & SenGupta, 2001, p. 177)
- Added 13 tests for circular_circular_correlation
- All 52 tests passing, ruff and mypy clean
- Exported from `neurospatial.metrics`

**Design Decisions:**
1. Fisher & Lee formula is invariant to constant offsets (r(a, a+c) = 1.0)
   This is mathematically correct since deviations from circular means are unchanged.
2. Anticorrelation requires reflection (-angles), not just opposite direction.
3. Minimum 5 samples required (same as circular_linear_correlation).
4. Degenerate cases handled (no variation → r=0, p=1.0 with warning).

### 2025-12-04: Phase Precession Analysis Implementation

**Completed:**
- Discovered `PhasePrecessionResult` dataclass was already implemented (M2.1)
- Exported `PhasePrecessionResult`, `phase_precession`, `has_phase_precession` from `neurospatial.metrics`
- Implemented `phase_precession()` with:
  - Maximum likelihood estimation via mean resultant length optimization
  - Grid search (100 points) + fminbound refinement to find global optimum
  - Position normalization with warning about slope unit changes
  - Validation for invalid position_range (division by zero)
- Added 20 tests:
  - 6 for `PhasePrecessionResult` (is_significant, interpretation, str)
  - 14 for `phase_precession()` (slope recovery, random data, validation)
  - 5 for `has_phase_precession()` (precession/recession/random detection)
- All 77 tests passing, ruff and mypy clean

**Design Decisions:**
1. Two-stage optimization (grid search + refinement) handles multiple local minima
   due to circular nature of objective function. Without this, optimizer could find
   equivalent positive slope for negative data due to circular wrapping.
2. Added validation for position_range where pos_max <= pos_min to prevent
   silent division by zero.
3. Correlation computed after normalization if position_range is provided -
   documented in comments.

**Next Steps:**

- Implement head direction module (Milestone 3)
- Write tests first (TDD)

### 2025-12-04: Phase Precession Visualization Implementation

**Completed:**

- Implemented `plot_phase_precession()` with:
  - Doubled phase axis (0-4π) following O'Keefe & Recce convention
  - Each point appears twice (at phase and phase + 2π)
  - Fitted lines drawn in both phase regions when result provided
  - Y-axis with π labels (0, π, 2π, 3π, 4π)
  - Optional annotation explaining doubled convention
  - Customizable markers, colors, and kwargs
  - Input validation for mismatched array lengths
- Added 17 tests covering all functionality
- Exported `plot_phase_precession` from `neurospatial.metrics`
- All 94 circular statistics tests passing, ruff and mypy clean

**Design Decisions:**

1. Used scatter plot with doubled data rather than two separate scatter calls
   for consistent styling and simpler code.
2. Fitted lines also doubled (in both 0-2π and 2π-4π regions) to match data.
3. Added length validation with clear error message (not just matplotlib's cryptic error).
4. Default annotation explains the doubling convention to help new users.

### 2025-12-04: Phase Precession Module Refactoring

**Completed:**

- Extracted phase precession code from `circular.py` into new `phase_precession.py` module
- Created `src/neurospatial/metrics/phase_precession.py` (466 lines) containing:
  - `PhasePrecessionResult` dataclass
  - `phase_precession()` function
  - `has_phase_precession()` function
  - `plot_phase_precession()` function
- Slimmed `circular.py` from ~1200 to 732 lines (core circular stats only)
- Created `tests/metrics/test_phase_precession.py` (549 lines, 43 tests)
- Updated `metrics/__init__.py` to import from both modules
- All 90 tests passing (47 circular + 43 phase precession)
- Backward compatibility maintained - `from neurospatial.metrics import phase_precession` still works

**Design Decisions:**

1. Internal helpers (`_to_radians`, `_mean_resultant_length`, `_validate_*`) stay in
   `circular.py` and are imported by `phase_precession.py` - avoids circular imports.
2. Helpers removed from `__all__` in circular.py - they're private implementation details.
3. Module docstrings include "Which Function Should I Use?" decision trees.
4. Both code-reviewer and ux-reviewer approved the refactoring.

---

## Blockers

None currently.

---

## Open Questions

None currently.
