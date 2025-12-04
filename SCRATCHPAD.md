# Circular Statistics Implementation - Scratchpad

**Started:** 2025-12-03
**Current Status:** ALL MILESTONES COMPLETE - Circular Statistics Implementation Done!

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

- Implement `head_direction_tuning_curve()` (Milestone 3.2)
- Write tests first (TDD)

### 2025-12-04: Head Direction Module Setup (M3.1)

**Completed:**

- Created `src/neurospatial/metrics/head_direction.py` with:
  - Module docstring with "Which Function Should I Use?" guide
  - Typical workflow examples
  - Scientific references (Taube et al. 1990, Sargolini et al. 2006)
  - Empty `__all__` list (ready for exports)
- Created `tests/metrics/test_head_direction.py` with 8 tests for module setup
- All 8 tests passing, ruff and mypy clean

**Design Decisions:**

1. Module follows same pattern as `phase_precession.py` with decision tree docstring.
2. Imports will be added when functions are implemented (ruff removes unused).
3. Default to radians like other circular modules, but note that HD literature
   commonly uses degrees (hence `angle_unit='deg'` in examples).

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

### 2025-12-04: Head Direction Tuning Curve Implementation (M3.2)

**Completed:**

- Implemented `head_direction_tuning_curve()` with:
  - Proper validation (length match, minimum samples, strict monotonicity)
  - Occupancy calculation using actual time deltas (handles dropped frames)
  - Last frame excluded from occupancy (scientifically correct)
  - Nearest-neighbor spike assignment (avoids circular interpolation bug)
  - Vectorized occupancy and spike counting using `np.bincount()`
  - Gaussian smoothing with circular boundary (`mode='wrap'`)
  - Duplicate timestamp detection
- Added 19 tests covering all functionality
- Exported `head_direction_tuning_curve` from `neurospatial.metrics`
- All 27 head direction tests passing, ruff and mypy clean

**Design Decisions:**

1. Used nearest-neighbor spike assignment instead of linear interpolation
   to avoid circular discontinuity issues. Linear interpolation would give
   wrong results when head direction crosses 0°/360° boundary (e.g., 350° to 10°
   would incorrectly interpolate to 180°).
2. Last frame excluded from occupancy calculation - we don't know how long
   the animal stayed at that position, so including it would bias firing rates.
3. Strict timestamp validation (no duplicates) catches data quality issues.
4. Vectorized with `np.bincount()` for 100-1000x performance improvement
   over Python loops.
5. Default to degrees (`angle_unit='deg'`) for head direction since HD literature
   commonly uses degrees (unlike other circular functions that default to radians).

### 2025-12-04: HeadDirectionMetrics and HD Cell Classification (M3.3-M3.4)

**Completed:**

- Implemented `HeadDirectionMetrics` dataclass with:
  - All 8 required fields (preferred_direction, mean_vector_length, peak_firing_rate, etc.)
  - `interpretation()` method with detailed explanation for both HD cells and non-HD cells
  - Educational content about 0.4 threshold (from Taube et al. 1990)
  - `__str__()` method that delegates to `interpretation()`
- Implemented `head_direction_metrics()` with:
  - Input validation (length match, all-zero rates, constant rates)
  - Mean vector length computation using centralized `_mean_resultant_length()` helper
  - Preferred direction as weighted circular mean via `arctan2()`
  - Tuning width (HWHM) approximation via bin counting above half-max
  - Rayleigh test for statistical significance
  - HD cell classification: MVL > min_vector_length AND p < 0.05
- Implemented `is_head_direction_cell()` convenience function:
  - Combines `head_direction_tuning_curve()` + `head_direction_metrics()`
  - Returns False on any ValueError (fast screening)
  - Passes **kwargs to tuning curve computation
- Added 29 new tests (8 for dataclass, 15 for metrics function, 6 for convenience function)
- All 56 head direction tests passing, ruff and mypy clean
- Exported `HeadDirectionMetrics`, `head_direction_metrics`, `is_head_direction_cell` from `neurospatial.metrics`

**Design Decisions:**

1. HWHM approximation uses bin counting rather than interpolation - documented as
   approximate in docstring. For more accurate measurements, users should use
   smaller bin_size or fit a parametric model.
2. Classification criteria follow Taube et al. (1990): MVL > 0.4 AND p < 0.05.
   The `min_vector_length` parameter allows adjustment for different brain regions
   or species.
3. Interpretation method provides educational content about threshold choices,
   helping users understand why neurons are classified as they are.

**Next Steps:**

- Milestone 3 complete! Head direction analysis module finished.
- Next: Milestone 4 (Tests) and Milestone 5 (Integration and Documentation)

### 2025-12-04: Head Direction Visualization Implementation (M3.5)

**Completed:**

- Implemented `plot_head_direction_tuning()` with:
  - Polar projection (default) with 0° at North, clockwise direction
  - Linear projection option for alternative visualization
  - Curve closing for smooth polar plots (first point appended at end)
  - Preferred direction marker (dashed red line from origin)
  - Metrics text box showing PFD, MVL, and peak firing rate
  - Configurable colors, fill alpha, and kwargs for line/fill
  - Input validation (length match, empty arrays, negative rates)
  - Proper type annotations with `TYPE_CHECKING` guard for mypy
- Added 20 tests covering all functionality
- Exported `plot_head_direction_tuning` from `neurospatial.metrics`
- All 76 head direction tests passing, ruff and mypy clean

**Design Decisions:**

1. Used `cast("PolarAxes", ax)` with TYPE_CHECKING guard to satisfy mypy while
   keeping matplotlib import lazy (inside function). This follows the pattern
   from `plot_phase_precession()`.
2. Default to degrees (`angle_display_unit='deg'`) for display since HD literature
   commonly uses degrees, matching `head_direction_tuning_curve()` default.
3. Fill stored as patches (not collections) in polar plots - tests updated to
   check `ax.patches` rather than `ax.collections`.
4. Validation includes: length mismatch with actionable error message, empty
   arrays check, and negative firing rate validation.
5. Text box positioning uses `ax.transAxes` for consistent placement regardless
   of projection type.

### 2025-12-04: Property-Based Tests and Final Integration (M4.7, M5)

**Completed:**

- Added property-based hypothesis tests to `test_circular.py`:
  - `TestPropertyBasedRayleighTest`: R in [0,1], p in [0,1], z >= 0
  - `TestPropertyBasedCircularCircularCorrelation`: symmetry, r in [-1,1]
  - `TestPropertyBasedCircularLinearCorrelation`: r >= 0, finite outputs
- Fixed doctest issues in module docstrings:
  - Added `# doctest: +SKIP` markers for stochastic tests
  - Changed code examples to use `::` block notation where appropriate
  - Fixed assertion values (r > 0.5 instead of r > 0.9 for theoretical maximum)
- Verified all Milestone 5 criteria:
  - Full test suite: 166 tests pass
  - Type checks: mypy clean
  - Linting: ruff clean
  - Doctests: pass with +SKIP markers
  - All imports work correctly

**Design Decisions:**

1. Property-based tests use min_size=5 for circular-circular correlation
   (function requires minimum 5 samples).
2. Tests filter out degenerate cases (constant values, limited angle range)
   to focus on testing valid inputs.
3. Doctest +SKIP markers used for stochastic tests rather than removing
   examples - preserves documentation value.

---

## Summary

All 5 milestones complete:

1. **M1: Core Circular Statistics** - `_to_radians()`, `_mean_resultant_length()`,
   `rayleigh_test()`, `circular_linear_correlation()`, `circular_circular_correlation()`
2. **M2: Phase Precession** - `PhasePrecessionResult`, `phase_precession()`,
   `has_phase_precession()`, `plot_phase_precession()`
3. **M3: Head Direction** - `head_direction_tuning_curve()`, `HeadDirectionMetrics`,
   `head_direction_metrics()`, `is_head_direction_cell()`, `plot_head_direction_tuning()`
4. **M4: Tests** - 166 tests across 3 files, including property-based tests
5. **M5: Integration** - All exports, type checks, linting, and doctests pass

---

## Blockers

None - implementation complete.

---

## Open Questions

None - all resolved.
