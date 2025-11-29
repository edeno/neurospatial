# Decoding Subpackage Implementation - Scratchpad

## Current Work

**Started**: 2025-11-28
**Current Milestone**: Milestone 2.2 Complete - Working on Correlation Metric

## Session Notes

### 2025-11-28 - Milestone 2.2 Complete (Confusion Matrix)

**Milestone 2.2 - Confusion Matrix**: ✅ COMPLETED

- Added `confusion_matrix()` function to `src/neurospatial/decoding/metrics.py`
- Supports two methods:
  - `method="map"`: Uses argmax for discrete bin predictions (integer counts)
  - `method="expected"`: Accumulates full posterior mass (probability-weighted)
- Created comprehensive test suite in `tests/decoding/test_metrics.py` (16 new tests)
- All tests pass (169 total in decoding/, 2 skipped)
- Code review passed with "APPROVE" rating
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Vectorized MAP method using `np.add.at()` for 10-100x speedup on large datasets
- Input validation: shape checks, bin range validation, method validation
- Helpful error messages with diagnostics (shows actual range found)
- Proper row/column semantics: rows=actual bins, columns=decoded bins
- Exported in `__init__.py` and `__all__` list

**Next task**: Milestone 2.3 - Correlation Metric (`decoding_correlation`)

### 2025-11-28 - Milestone 2.1 Complete (Error Metrics)

**Milestone 2.1 - Error Metrics**: ✅ COMPLETED

- Created `src/neurospatial/decoding/metrics.py` with two functions:
  - `decoding_error()` - Per-time-bin position error (Euclidean or graph-based)
  - `median_decoding_error()` - Median error summary statistic
- Created comprehensive test suite: `tests/decoding/test_metrics.py` (26 tests)
- All tests pass (152 total in decoding/, 2 skipped)
- Code review passed with "APPROVE" rating
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Euclidean metric uses vectorized `np.linalg.norm(diff, axis=1)`
- Graph metric uses `env.distance_between()` for shortest-path distances
- NaN values propagate correctly (explicit check for graph metric)
- `median_decoding_error` uses `np.nanmedian` for robust statistics
- Type annotations with `cast()` for mypy compatibility
- Comprehensive docstrings with NumPy format, examples, and type annotations

**Next task**: Milestone 2.2 - Confusion Matrix

### 2025-11-28 - Milestone 1.5 & 1.6 Complete (Phase 1 Done!)

**Milestone 1.5 - Estimate Functions**: ✅ COMPLETED

- Created `src/neurospatial/decoding/estimates.py` with five functions:
  - `map_estimate()` - Maximum a posteriori bin index
  - `map_position()` - MAP position in environment coordinates
  - `mean_position()` - Posterior mean position (expected value)
  - `entropy()` - Posterior entropy in bits (mask-based, numerically stable)
  - `credible_region()` - Highest posterior density region (HPD)
- Created comprehensive test suite: `tests/decoding/test_estimates.py` (27 tests)
- All tests pass (127 total in decoding/, 2 skipped)
- Code review passed with "APPROVE" rating
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Functions mirror DecodingResult properties for API consistency
- Standalone functions enable use without creating DecodingResult container
- Entropy uses mask-based computation (`np.where(p > 0, np.log2(p), 0.0)`)
- HPD region uses sorted probabilities + cumsum + searchsorted
- Comprehensive docstrings with NumPy format, examples, and type annotations
- Tests verify consistency with DecodingResult properties

**Milestone 1.6 - Phase 1 Tests**: ✅ COMPLETED

All Phase 1 tests pass:

- `test_result.py` - 28 tests
- `test_likelihood.py` - 28 tests
- `test_posterior.py` - 41 tests
- `test_estimates.py` - 27 tests
- Total: 127 passed, 2 skipped

**Next task**: Milestone 2 - Quality Metrics

- `decoding_error()` - Position error per time bin
- `median_decoding_error()` - Median error summary
- `confusion_matrix()` - Spatial confusion analysis
- `decoding_correlation()` - Weighted correlation

### 2025-11-28 - Milestone 1.4 Complete

**Milestone 1.4 - Posterior Normalization**: ✅ COMPLETED

- Created `src/neurospatial/decoding/posterior.py` with two functions:
  - `normalize_to_posterior()` - Log-sum-exp normalization with prior and degenerate handling
  - `decode_position()` - Main entry point combining likelihood + posterior
- Created comprehensive test suite: `tests/decoding/test_posterior.py` (41 tests)
- All tests pass (101 total in decoding/, 1 skipped)
- Code review passed with excellent rating
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Log-sum-exp trick for numerical stability: `ll -= ll.max(); post = exp(ll); post /= post.sum()`
- Prior handling: normalizes prior internally, supports 1D (stationary) and 2D (time-varying)
- Degenerate handling: three strategies ("uniform", "nan", "raise")
- Input validation: optional `validate=True` catches NaN/Inf in inputs
- NumPy 2.x doctest compatibility with `bool()` conversion
- Type annotations with `Literal` types for enums

**Next task**: Milestone 1.5 - Estimate Functions
- `map_estimate()`, `map_position()`, `mean_position()` - Already implemented as DecodingResult properties
- `entropy()` - Already implemented as DecodingResult.uncertainty property
- `credible_region()` - New: HPD bin indices

### 2025-11-28 - Milestone 1.3 Complete

**Milestone 1.3 - Likelihood Functions**: ✅ COMPLETED

- Created `src/neurospatial/decoding/likelihood.py` with two functions:
  - `log_poisson_likelihood()` - Primary function for numerically stable log-likelihood
  - `poisson_likelihood()` - Thin wrapper with underflow protection
- Formula: `sum_i [n_i * log(lambda_i * dt) - lambda_i * dt]`
- Omits `-log(n_i!)` term (constant across positions, cancels in normalization)
- Created comprehensive test suite: `tests/decoding/test_likelihood.py` (28 tests)
- All tests pass, doctests pass
- Code review passed - addressed mypy type issues with `cast()`
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Uses matrix multiplication for efficient vectorization
- `min_rate` parameter (default 1e-10) prevents log(0)
- Handles edge cases: zero spikes, single neuron/bin, extreme rates
- NumPy 2.x doctest compatibility with `bool()` conversion
- Type annotations with `cast("NDArray[np.float64]", ...)` for mypy

**Next task**: Milestone 1.4 - Posterior Normalization
- `normalize_to_posterior()` - Bayes rule with log-sum-exp trick
- `decode_position()` - Main entry point combining likelihood + posterior

### 2025-11-28 - Milestone 1.2 Complete

**Milestone 1.2 - DecodingResult Container**: ✅ COMPLETED

- Created `src/neurospatial/decoding/_result.py` with full `DecodingResult` dataclass
- Implemented cached properties: `map_estimate`, `map_position`, `mean_position`, `uncertainty`
- Implemented `n_time_bins` property
- Added `plot()` and `to_dataframe()` method stubs
- Created comprehensive test suite: `tests/decoding/test_result.py` (28 tests)
- All tests pass (32 total in decoding/, 1 skipped)
- Code review complete - addressed all mypy type errors
- Ruff and mypy pass with no issues

**Implementation highlights**:
- Used `@dataclass` (not frozen) to allow `@cached_property`
- Mask-based entropy computation avoids log(0) and doesn't bias results
- NumPy docstrings with examples and See Also sections
- Type annotations with `cast()` for mypy compatibility

**Next task**: Milestone 1.3 - Likelihood Functions
- `log_poisson_likelihood()` - primary likelihood function
- `poisson_likelihood()` - thin wrapper (discouraged for direct use)

### 2025-11-28 - Initial Setup

Starting implementation of the Bayesian decoding subpackage following PLAN.md and TASKS.md.

**Milestone 1.1 - Package Setup**: ✅ COMPLETED

- Created `src/neurospatial/decoding/` directory
- Created `__init__.py` with placeholder exports (DecodingResult, decode_position)
- Added `decoding` to main package `__init__.py` imports
- Tests pass: `tests/decoding/test_imports.py` (5 tests)
- Linting (ruff) and type checking (mypy) pass

## Decisions Made

- Following TDD approach as specified in workflow
- Using stateless functions (not classes) per PLAN.md design decisions
- DecodingResult is a dataclass with `@cached_property` for lazy computation
- Used `cast()` for mypy compatibility with numpy array returns
- Entropy uses mask-based computation (more accurate than global clipping)
- Dimension naming: 'x', 'y', 'z' for ≤3D, 'dim_0', 'dim_1', etc. for >3D

## Blockers

None currently.

## Questions

None currently.
