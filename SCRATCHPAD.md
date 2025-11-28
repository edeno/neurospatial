# Decoding Subpackage Implementation - Scratchpad

## Current Work

**Started**: 2025-11-28
**Current Milestone**: 1.4 Posterior Normalization

## Session Notes

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
