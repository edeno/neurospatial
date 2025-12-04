# Circular Statistics Implementation Tasks

**Based on:** CIRCULAR_STATS_PLAN.md
**Goal:** Add circular statistics, phase precession, and head direction analysis to `neurospatial.metrics`

---

## Milestone 1: Core Circular Statistics (`circular.py`)

**Objective:** Create foundation module with angle conversion, validation, and basic statistics.

### 1.1 Module Setup and Internal Helpers

- [x] **Create `src/neurospatial/metrics/circular.py`**
  - Add module header with docstring (usage guide, function overview, references)
  - Add imports: `dataclass`, `TYPE_CHECKING`, `Literal`, `numpy`, `scipy.stats`, `scipy.optimize.fminbound`
  - Add `__all__` with all public exports

- [x] **Implement `_to_radians()`**
  - Parameters: `angles: NDArray`, `angle_unit: Literal["rad", "deg"]`
  - Returns: angles in radians
  - Simple conversion using `np.radians()` when needed

- [x] **Implement `_mean_resultant_length()`**
  - Feature-detect `scipy.stats.directional_stats` (scipy >= 1.9.0)
  - Use scipy when available, fallback to direct computation otherwise
  - Support optional `weights` parameter for weighted statistics
  - Handle empty arrays (return `np.nan`)
  - **Success criteria:** Returns float in [0, 1]

- [x] **Implement `_validate_circular_input()`**
  - Parameters: `angles`, `name`, `min_samples=3`, `check_range=True`
  - Check for all-NaN (raise with diagnostic message)
  - Check for Inf values (raise)
  - Check minimum samples (raise)
  - Warn and wrap if angles outside [0, 2pi]
  - **Success criteria:** Actionable error messages with diagnostic steps

- [x] **Implement `_validate_paired_input()`**
  - Parameters: `arr1`, `arr2`, `name1`, `name2`, `min_samples=3`
  - Check lengths match (raise if not)
  - Remove pairs where either is NaN (warn)
  - Check minimum samples after removal
  - **Success criteria:** Returns cleaned paired arrays

### 1.2 Rayleigh Test

- [ ] **Implement `rayleigh_test()`**
  - Parameters: `angles`, `angle_unit='rad'`, `weights=None`
  - Convert to radians, validate input
  - Handle optional weights with effective sample size
  - Compute z-statistic: `n * R^2`
  - Compute p-value with finite-sample correction (Mardia & Jupp, p. 94)
  - **Success criteria:**
    - Uniform distribution: p > 0.5
    - Von Mises (kappa=2): p < 0.001
    - z, pval both floats

### 1.3 Circular-Linear Correlation

- [ ] **Implement `circular_linear_correlation()`**
  - Parameters: `angles`, `values`, `angle_unit='rad'`
  - Validate paired inputs
  - Compute Pearson correlations: rxs, rxc, rcs using `scipy.stats.pearsonr`
  - Handle degenerate case (rcs near 1)
  - Compute r using Mardia & Jupp formula
  - Compute p-value from chi-squared with 2 df
  - **Success criteria:**
    - Perfect linear relationship: r -> 1.0
    - Random data: r -> 0
    - r always non-negative

- [ ] **Implement `phase_position_correlation()`**
  - Alias for `circular_linear_correlation` with neuroscience naming
  - Parameters: `phases`, `positions`, `angle_unit='rad'`
  - **Success criteria:** Same output as `circular_linear_correlation`

### 1.4 Circular-Circular Correlation

- [ ] **Implement `circular_circular_correlation()`**
  - Parameters: `angles1`, `angles2`, `angle_unit='rad'`
  - Validate paired inputs
  - Compute circular means using `scipy.stats.circmean`
  - Compute Fisher & Lee correlation coefficient
  - Handle degenerate case (no variation)
  - Compute p-value from normal approximation
  - **Success criteria:**
    - r in [-1, 1]
    - Symmetric: r(a1, a2) == r(a2, a1)

---

## Milestone 2: Phase Precession Analysis (`circular.py` continued)

**Objective:** Add phase precession detection and analysis for place cells.

### 2.1 PhasePrecessionResult Dataclass

- [ ] **Implement `PhasePrecessionResult`**
  - Fields: `slope`, `slope_units`, `offset`, `correlation`, `pval`, `mean_resultant_length`
  - Method: `is_significant(alpha=0.05)` -> bool
  - Method: `interpretation()` -> str with significance, direction, correlation strength, fit quality
  - Method: `__str__()` -> automatic interpretation when printing
  - **Success criteria:** `print(result)` shows human-readable analysis

### 2.2 Phase Precession Analysis

- [ ] **Implement `phase_precession()`**
  - Parameters: `positions`, `phases`, `slope_bounds=(-2pi, 2pi)`, `position_range=None`, `angle_unit='rad'`, `min_spikes=10`
  - Validate inputs, wrap phases to [0, 2pi]
  - Warn if `position_range` used (changes slope units!)
  - Define optimization: maximize mean resultant length of residuals
  - Use `scipy.optimize.fminbound` to find optimal slope
  - Compute offset as circular mean of residuals
  - Compute correlation using `circular_linear_correlation`
  - Return `PhasePrecessionResult`
  - **Success criteria:**
    - Synthetic data with known slope: recovered within 20%
    - Random phases: high p-value, low correlation

- [ ] **Implement `has_phase_precession()`**
  - Parameters: `positions`, `phases`, `alpha=0.05`, `min_correlation=0.2`
  - Quick boolean check using `phase_precession()`
  - Return True if: p < alpha, r >= min_correlation, slope < 0
  - Catch ValueError -> return False
  - **Success criteria:** Fast screening function

### 2.3 Phase Precession Visualization

- [ ] **Implement `plot_phase_precession()`**
  - Parameters: `positions`, `phases`, `result=None`, `ax=None`, `position_label`, `show_fit`, `marker_size`, `marker_alpha`, `show_doubled_note`, `scatter_kwargs`, `line_kwargs`
  - Plot doubled phase axis (0-4pi) per O'Keefe & Recce convention
  - Add annotation explaining WHY points appear twice
  - Overlay fitted line if result provided (plot twice for both copies)
  - Format y-axis with pi labels
  - **Success criteria:** Standard phase precession visualization

---

## Milestone 3: Head Direction Analysis (`head_direction.py`)

**Objective:** Create module for head direction cell tuning and classification.

### 3.1 Module Setup

- [ ] **Create `src/neurospatial/metrics/head_direction.py`**
  - Add module header with docstring (workflow, examples, references)
  - Add imports (import `rayleigh_test`, `_mean_resultant_length` from sibling)
  - Add `__all__`

### 3.2 Head Direction Tuning Curve

- [ ] **Implement `head_direction_tuning_curve()`**
  - Parameters: `head_directions`, `spike_times`, `position_times`, `bin_size=6.0`, `angle_unit='deg'`, `smoothing_window=5`
  - Convert to radians if needed
  - Validate: length match, minimum samples, spike count, monotonic timestamps
  - Compute occupancy using actual time deltas (handles dropped frames)
  - Assign spikes to bins via interpolation
  - Compute firing rates with division-by-zero handling
  - Apply Gaussian smoothing with `mode='wrap'` for circular boundary
  - **Success criteria:**
    - Returns bin_centers (radians), firing_rates (Hz)
    - Handles non-uniform sampling correctly

### 3.3 HeadDirectionMetrics Dataclass

- [ ] **Implement `HeadDirectionMetrics`**
  - Fields: `preferred_direction`, `preferred_direction_deg`, `mean_vector_length`, `peak_firing_rate`, `tuning_width`, `tuning_width_deg`, `is_hd_cell`, `rayleigh_pval`
  - Method: `interpretation()` -> str explaining classification
  - Method: `__str__()` -> automatic interpretation
  - **Success criteria:** `print(metrics)` explains why classified or not

### 3.4 Head Direction Metrics Computation

- [ ] **Implement `head_direction_metrics()`**
  - Parameters: `bin_centers`, `firing_rates`, `min_vector_length=0.4`
  - Validate: length match, non-zero rates, non-constant rates
  - Compute mean vector length using `_mean_resultant_length` with weights
  - Compute preferred direction (weighted circular mean)
  - Compute peak firing rate
  - Compute tuning width (HWHM approximation)
  - Run Rayleigh test with weighted angles
  - Classify: MVL > threshold AND p < 0.05
  - **Success criteria:**
    - Gaussian tuning: correct preferred direction
    - Uniform firing: not classified as HD cell

- [ ] **Implement `is_head_direction_cell()`**
  - Convenience function for fast screening
  - Calls `head_direction_tuning_curve` + `head_direction_metrics`
  - Catch ValueError -> return False
  - **Success criteria:** Single-function HD cell check

### 3.5 Head Direction Visualization

- [ ] **Implement `plot_head_direction_tuning()`**
  - Parameters: `bin_centers`, `firing_rates`, `metrics=None`, `ax=None`, `projection='polar'`, `angle_display_unit='deg'`, `show_metrics`, `color`, `fill_alpha`, `line_kwargs`, `fill_kwargs`
  - Support polar and linear projections
  - Close curve for polar plot (append first point)
  - Mark preferred direction
  - Add metrics text box
  - Set theta_zero_location='N', theta_direction=-1 for polar
  - **Success criteria:** Standard HD cell polar plot

---

## Milestone 4: Tests

**Objective:** Comprehensive test coverage with known values, edge cases, and property tests.

### 4.1 Test Setup

- [ ] **Create `tests/metrics/test_circular.py`**
- [ ] **Create `tests/metrics/test_head_direction.py`**
- [ ] **Add dev dependency** (optional): `pycircstat2` for validation

### 4.2 Core Statistics Tests

- [ ] **Test Rayleigh test**
  - Uniform distribution: p > 0.5
  - Von Mises (kappa=2): p < 0.001
  - Weighted: correct effective sample size
  - Edge case: all same angle -> R = 1.0

- [ ] **Test circular-linear correlation**
  - Perfect linear: r -> 1.0
  - Random data: r -> 0
  - Degenerate case: warns and returns r=0

- [ ] **Test circular-circular correlation**
  - Symmetry: r(a1, a2) == r(a2, a1)
  - Perfect correlation: r -> 1.0
  - Anti-correlation: r -> -1.0
  - No correlation: r -> 0

### 4.3 Phase Precession Tests

- [ ] **Test slope recovery**
  - Synthetic data with known slope (-1.5 rad/cm): recovered within 20%
  - Offset recovered within 0.5 rad

- [ ] **Test no relationship detection**
  - Random phases: p > 0.05, r < 0.2

- [ ] **Test `has_phase_precession`**
  - True for synthetic precession
  - False for random data
  - False when insufficient spikes

### 4.4 Head Direction Tests

- [ ] **Test tuning curve computation**
  - Correct bin centers
  - Firing rates in Hz
  - Occupancy calculation handles non-uniform sampling

- [ ] **Test HD cell classification**
  - Sharp Gaussian tuning: classified as HD cell
  - Uniform firing: not classified
  - Correct preferred direction

- [ ] **Test edge cases**
  - All-zero firing rates: ValueError
  - Constant firing rates: ValueError
  - Non-monotonic timestamps: ValueError

### 4.5 Validation Tests

- [ ] **Test scipy fallback**
  - `_mean_resultant_length`: scipy vs fallback match within 1e-10
  - `rayleigh_test`: results match with both paths

- [ ] **Test against pycircstat2** (if available)
  - `rayleigh_test`: z, p match within 1%
  - Run with parametrize over sample sizes

### 4.6 Edge Case Tests

- [ ] **Input validation tests**
  - Empty arrays -> ValueError
  - All NaN -> ValueError with diagnostic
  - Mixed NaN -> Warning, proceed
  - Angles outside [0, 2pi] -> Warning, wrap
  - Mismatched lengths -> ValueError
  - Insufficient samples -> ValueError

### 4.7 Property-Based Tests

- [ ] **Add hypothesis tests**
  - `rayleigh_test`: R always in [0, 1]
  - `circular_circular_correlation`: symmetric
  - All outputs finite for valid inputs

---

## Milestone 5: Integration and Documentation

**Objective:** Export functions and update package structure.

### 5.1 Update Exports

- [ ] **Update `src/neurospatial/metrics/__init__.py`**
  - Import all public functions from `circular.py`
  - Import all public functions from `head_direction.py`
  - Add to `__all__`

### 5.2 Verify Integration

- [ ] **Run full test suite**: `uv run pytest`
- [ ] **Run type checks**: `uv run mypy src/neurospatial/`
- [ ] **Run linting**: `uv run ruff check . && uv run ruff format .`

### 5.3 Documentation Verification

- [ ] **Verify all docstrings follow NumPy format**
- [ ] **Verify all functions have examples in docstrings**
- [ ] **Run doctests**: `uv run pytest --doctest-modules src/neurospatial/metrics/circular.py src/neurospatial/metrics/head_direction.py`

---

## Dependencies

**Required (already in project):**
- numpy
- scipy (stats, optimize, ndimage)
- matplotlib

**No new dependencies needed.**

**Optional dev dependencies:**
- pycircstat2 (validation only)
- hypothesis (property tests, already available)

---

## Success Criteria

1. All tests pass: `uv run pytest tests/metrics/test_circular.py tests/metrics/test_head_direction.py -v`
2. Type checks pass: `uv run mypy src/neurospatial/metrics/circular.py src/neurospatial/metrics/head_direction.py`
3. Linting passes: `uv run ruff check src/neurospatial/metrics/circular.py src/neurospatial/metrics/head_direction.py`
4. Imports work: `from neurospatial.metrics import rayleigh_test, phase_precession, head_direction_metrics`
5. Example workflow runs without errors:
   ```python
   from neurospatial.metrics import (
       rayleigh_test, phase_precession, plot_phase_precession,
       head_direction_tuning_curve, head_direction_metrics
   )
   ```

---

## Implementation Order

1. **M1.1-M1.4**: Core circular statistics (no dependencies)
2. **M2.1-M2.3**: Phase precession (depends on M1)
3. **M3.1-M3.5**: Head direction (depends on M1)
4. **M4.1-M4.7**: Tests (depends on M1-M3)
5. **M5.1-M5.3**: Integration (depends on M1-M4)

---

## Notes

- All functions use `angle_unit` parameter: `'rad'` (default) or `'deg'`
- Internal computations always in radians
- Error messages include diagnostic steps and fix suggestions
- Warnings for data quality issues (NaN removal, angle wrapping)
- Follow existing patterns in `place_fields.py` and `grid_cells.py`
