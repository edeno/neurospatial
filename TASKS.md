# Circular Basis Functions - Implementation Tasks

This document breaks down CIRCULAR_BASIS_PLAN.md into actionable tasks for Claude Code.

**Plan Summary**: Add circular basis functions to `neurospatial.metrics.circular` for GLM design matrix construction, plus fix `head_direction.py` to default to radians.

---

## Milestone 0: Fix head_direction.py Default to Radians

**Goal**: Change `angle_unit` default from `'deg'` to `'rad'` for consistency with other circular modules.

**Dependencies**: None (can be done first or in parallel with M1)

### Tasks

- [x] **M0.1**: Update `head_direction_tuning_curve()` default
  - File: `src/neurospatial/metrics/head_direction.py`
  - Change: `angle_unit: Literal["rad", "deg"] = "deg"` → `angle_unit: Literal["rad", "deg"] = "rad"`
  - Location: ~line 204

- [x] **M0.2**: Update `plot_head_direction_tuning()` parameter naming
  - File: `src/neurospatial/metrics/head_direction.py`
  - Change: Rename `angle_display_unit` to `angle_unit` for consistency
  - Change default: `"deg"` → `"rad"`
  - Location: ~line 606

- [x] **M0.3**: Add "Angle Units" section to module docstring
  - File: `src/neurospatial/metrics/head_direction.py`
  - Already present at lines 46-50 (module docstring)

- [x] **M0.4**: Update existing tests for head_direction.py
  - File: `tests/metrics/test_head_direction.py`
  - Updated tests to use `angle_unit` instead of `angle_display_unit`
  - Tests already explicitly passed `angle_unit='deg'`

- [x] **M0.5**: Run tests to verify no regressions
  - Command: `uv run pytest tests/metrics/test_head_direction.py -v`
  - All 76 tests pass

**Success Criteria**:
- `head_direction_tuning_curve()` defaults to `angle_unit='rad'`
- `plot_head_direction_tuning()` uses `angle_unit` (not `angle_display_unit`)
- All existing tests pass

---

## Milestone 1: Core Functions

**Goal**: Add `circular_basis()`, `CircularBasisResult`, and `circular_basis_metrics()` to `circular.py`.

**Dependencies**: None

### Tasks

- [x] **M1.1**: Add imports to circular.py
  - File: `src/neurospatial/metrics/circular.py`
  - Added: `from dataclasses import dataclass`
  - Added: `from scipy.stats import chi2` (for Wald test)

- [x] **M1.2**: Update module docstring with GLM guidance
  - Skipped: module docstring already contains relevant guidance
  - GLM workflow is documented in function docstrings instead

- [x] **M1.3**: Implement `circular_basis()` function
  - File: `src/neurospatial/metrics/circular.py`
  - Creates simple sin/cos basis (n_harmonics=1 by default)
  - Parameters: `angles`, `angle_unit='rad'`
  - Returns: `CircularBasisResult` with design_matrix property

- [x] **M1.4**: Implement `CircularBasisResult` dataclass
  - File: `src/neurospatial/metrics/circular.py`
  - Attributes: `sin_component`, `cos_component`, `angles`
  - Property: `design_matrix` returning (n_samples, 2) array

- [x] **M1.5**: Implement `_wald_test_magnitude()` helper
  - File: `src/neurospatial/metrics/circular.py`
  - Private function for significance testing
  - Uses Wald statistic: beta.T @ inv(cov) @ beta ~ chi2(2)
  - Returns p-value for H0: amplitude = 0

- [x] **M1.6**: Implement `circular_basis_metrics()` function
  - File: `src/neurospatial/metrics/circular.py`
  - Computes amplitude = sqrt(β_sin² + β_cos²)
  - Computes phase = arctan2(β_sin, β_cos)
  - Returns tuple: (amplitude, phase, pvalue)

- [x] **M1.7**: Add to `__all__` in circular.py and metrics/__init__.py
  - File: `src/neurospatial/metrics/circular.py`
  - File: `src/neurospatial/metrics/__init__.py`
  - Added: `"CircularBasisResult"`, `"circular_basis"`, `"circular_basis_metrics"`

**Success Criteria**:
- `circular_basis(angles)` returns correct shape design matrix
- `circular_basis_metrics(coefficients)` returns `CircularBasisResult` with correct values
- Error message when coefficient length doesn't match expected

---

## Milestone 2: Convenience Function

**Goal**: Add `is_modulated()` quick-check function.

**Dependencies**: M1 (uses `circular_basis_metrics`)

### Tasks

- [x] **M2.1**: Implement `is_modulated()` function
  - File: `src/neurospatial/metrics/circular.py`
  - Parameters: `beta_sin`, `beta_cos`, `cov_matrix` (required), `alpha=0.05`, `min_magnitude=0.2`
  - Returns: `True` if p < alpha AND magnitude >= min_magnitude
  - Calls `circular_basis_metrics()` internally
  - Added 10 tests including edge cases (zero coefficients, singular covariance)

- [x] **M2.2**: Add to `__all__` in circular.py and metrics/__init__.py
  - Add: `"is_modulated"`

**Success Criteria**:
- `is_modulated(beta_sin, beta_cos, cov)` returns boolean
- Returns `True` only when both statistically significant AND magnitude above threshold

---

## Milestone 3: Visualization

**Goal**: Add `plot_circular_basis_tuning()` for visualizing GLM fit.

**Dependencies**: M1 (uses `CircularBasisResult`)

### Tasks

- [x] **M3.1**: Add matplotlib imports
  - File: `src/neurospatial/metrics/circular.py`
  - Added: TYPE_CHECKING imports for `Axes` and `PolarAxes`
  - Added: `Any`, `cast` to typing imports

- [x] **M3.2**: Implement `plot_circular_basis_tuning()` function
  - File: `src/neurospatial/metrics/circular.py`
  - Parameters: `beta_sin`, `beta_cos`, `angles=None`, `rates=None`, `ax=None`, `intercept=0.0`, `projection='polar'`, `n_points=100`, `show_data=False`, `show_fit=True`, `color='C0'`, `data_color='gray'`, `data_alpha=0.5`, `line_kwargs`, `scatter_kwargs`
  - Raises `ValueError` if `show_data=True` but angles/rates not provided
  - Creates polar or linear plot based on projection
  - Uses exp() link function for Poisson GLM visualization
  - Added 12 tests (all passing)

- [x] **M3.3**: Add to `__all__` in circular.py and metrics/__init__.py
  - Added: `"plot_circular_basis_tuning"` to both files

**Success Criteria**:
- Polar plot shows smooth tuning curve from coefficients
- Raw data overlaid when angles/rates provided
- Preferred direction marked when metrics provided
- Clear error message when show_data=True but data missing

---

## Milestone 4: Tests

**Goal**: Add comprehensive tests for all new functions (~25 tests).

**Dependencies**: M1, M2, M3

### Tasks

- [ ] **M4.1**: Add `TestCircularBasis` class
  - File: `tests/metrics/test_circular.py`
  - Tests for design matrix construction:
    - `test_single_harmonic_shape`: (n,) → (n, 3) with intercept
    - `test_multiple_harmonics_shape`: n_harmonics=2 → (n, 5)
    - `test_without_intercept`: (n,) → (n, 2)
    - `test_degree_input`: angle_unit='deg' converts correctly
    - `test_values_correct`: cos/sin computed correctly for known angles
    - `test_empty_input_raises`: Empty array → ValueError
    - `test_nan_handling`: NaN in input validated

- [ ] **M4.2**: Add `TestCircularBasisMetrics` class
  - File: `tests/metrics/test_circular.py`
  - Tests for coefficient interpretation:
    - `test_pure_cosine_modulation`: [0, 1, 0] → magnitude=1, phase=0
    - `test_pure_sine_modulation`: [0, 0, 1] → magnitude=1, phase=π/2
    - `test_45_degree_preference`: [0, 1, 1] → phase=π/4
    - `test_no_modulation`: [0, 0, 0] → magnitude=0
    - `test_with_covariance_significant`: p < 0.05 when strong modulation
    - `test_without_covariance`: pval=None, is_significant=False
    - `test_multiple_harmonics`: harmonic_magnitudes has correct length
    - `test_coefficient_length_mismatch_raises`: Helpful error message
    - `test_interpretation_string`: __str__ returns readable text

- [ ] **M4.3**: Add `TestIsModulated` class
  - File: `tests/metrics/test_circular.py`
  - Tests for convenience function:
    - `test_significant_strong_modulation`: Returns True
    - `test_not_significant`: p > 0.05 → False
    - `test_weak_modulation_below_threshold`: magnitude < 0.2 → False

- [ ] **M4.4**: Add `TestPlotCircularBasisTuning` class
  - File: `tests/metrics/test_circular.py`
  - Tests for visualization:
    - `test_polar_plot_creates_figure`
    - `test_linear_plot_creates_figure`
    - `test_show_data_requires_angles_rates`: ValueError if missing
    - `test_show_fit_only`: show_data=False works
    - `test_preferred_direction_marker`: When metrics provided

- [ ] **M4.5**: Run all circular tests
  - Command: `uv run pytest tests/metrics/test_circular.py -v`

**Success Criteria**:
- All ~25 tests pass
- Tests cover edge cases (empty input, NaN, coefficient mismatch)
- Visualization tests use pytest-mpl or equivalent

---

## Milestone 5: Integration

**Goal**: Update exports and verify end-to-end workflow.

**Dependencies**: M0, M1, M2, M3, M4

### Tasks

- [ ] **M5.1**: Update `metrics/__init__.py` exports
  - File: `src/neurospatial/metrics/__init__.py`
  - Add imports:
    ```python
    from neurospatial.metrics.circular import (
        CircularBasisResult,
        circular_basis,
        circular_basis_metrics,
        is_modulated,
        plot_circular_basis_tuning,
    )
    ```
  - Add to `__all__`

- [ ] **M5.2**: Verify imports work from top level
  - Command: `uv run python -c "from neurospatial.metrics import circular_basis, circular_basis_metrics, CircularBasisResult, is_modulated, plot_circular_basis_tuning; print('OK')"`

- [ ] **M5.3**: Run type checking
  - Command: `uv run mypy src/neurospatial/metrics/circular.py src/neurospatial/metrics/head_direction.py`

- [ ] **M5.4**: Run linting
  - Command: `uv run ruff check src/neurospatial/metrics/`

- [ ] **M5.5**: Run full test suite
  - Command: `uv run pytest tests/metrics/ -v`

- [ ] **M5.6**: Test end-to-end workflow
  - Create test script verifying example from CIRCULAR_BASIS_PLAN.md:
    ```python
    from neurospatial.metrics import circular_basis, circular_basis_metrics
    import statsmodels.api as sm
    import numpy as np

    phases = np.random.uniform(0, 2*np.pi, 1000)
    spike_counts = np.random.poisson(5, 1000)

    X = circular_basis(phases)
    model = sm.GLM(spike_counts, X, family=sm.families.Poisson())
    result = model.fit()

    metrics = circular_basis_metrics(result.params, covariance_matrix=result.cov_params())
    print(metrics)
    ```

**Success Criteria**:
- All imports work from `neurospatial.metrics`
- Type checking passes with no errors
- Linting passes
- All tests pass
- End-to-end workflow executes without error

---

## Summary

| Milestone | Tasks | Dependencies | Est. Tests |
|-----------|-------|--------------|------------|
| M0: Fix head_direction.py | 5 | None | Update existing |
| M1: Core Functions | 7 | None | 0 (tested in M4) |
| M2: Convenience Function | 2 | M1 | 0 (tested in M4) |
| M3: Visualization | 3 | M1 | 0 (tested in M4) |
| M4: Tests | 5 | M1, M2, M3 | ~25 new tests |
| M5: Integration | 6 | All | 0 |

**Total**: 28 tasks across 6 milestones

**Recommended Order**: M0 and M1 can run in parallel → M2, M3 → M4 → M5

---

## Notes

- **No new dependencies required** - Uses existing numpy, scipy, matplotlib
- **Breaking change**: M0 changes `head_direction.py` default from degrees to radians
- **Follow NumPy docstring format** for all new functions
- **Use `uv run`** for all Python commands
