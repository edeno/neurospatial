# Decoding Subpackage Implementation - Scratchpad

## Current Work

**Started**: 2025-11-28
**Current Milestone**: Milestone 5.2 Complete - Visualization. Next: 5.3 Documentation

## Session Notes

### 2025-11-28 - Milestone 5.2 Complete (Visualization)

**Milestone 5.2 - Visualization**: ✅ COMPLETED

- Implemented `DecodingResult.plot()` method with full visualization capabilities:
  - Posterior probability heatmap with `imshow`
  - Optional colorbar (`colorbar=True`)
  - Optional MAP trajectory overlay (`show_map=True`) as white line
  - Proper time axis extent when `times` is provided
  - Customizable colormap and imshow kwargs
- Verified `DecodingResult.to_dataframe()` was already complete (updated docstring):
  - Exports time (optional), map_bin, map_x/y, mean_x/y, uncertainty
  - Smart dimension naming (x,y,z for ≤3D, dim_0,dim_1 for >3D)
- Added 23 new tests (8 for plot, 15 for to_dataframe)
- All 45 tests in test_result.py pass (1 skipped)
- All 448 decoding tests pass (3 skipped)
- Code review passed with APPROVE rating
- Ruff and mypy pass with no issues

**Implementation highlights**:
- `plot()` uses proper extent calculation for time axis
- MAP overlay uses float64 dtype for mypy compatibility
- Colorbar and show_map are keyword-only arguments
- to_dataframe() docstring updated with comprehensive examples
- Follows NumPy docstring format consistently

**Next task**: Milestone 5.3 - Documentation
- Add decoding quick reference to CLAUDE.md
- Create examples/bayesian_decoding.ipynb notebook

### 2025-11-28 - Milestone 5.1 Complete (Public API Finalization)

**Milestone 5.1 - Public API Finalization**: ✅ COMPLETED

- Added `decoding_error` and `median_decoding_error` to main `neurospatial/__init__.py` exports
- Updated `__all__` list in main package to include all four decoding exports per PLAN.md
- Added comprehensive import tests to `tests/decoding/test_imports.py` (19 tests)
- All decoding tests pass (430 passed, 3 skipped)
- Ruff and mypy pass with no issues

**Public API exports from main package (per PLAN.md)**:

- `DecodingResult` - Result container class
- `decode_position` - Main entry point function
- `decoding_error` - Per-time-bin position error
- `median_decoding_error` - Summary statistic

**Next task**: Milestone 5.2 - Visualization

- `DecodingResult.plot()` - Posterior heatmap
- `DecodingResult.to_dataframe()` - Pandas export

### 2025-11-28 - Milestone 4.5 Complete (Significance Testing Functions)

**Milestone 4.5 - Significance Testing Functions**: ✅ COMPLETED

- Added three new items to `src/neurospatial/decoding/shuffle.py`:
  - `compute_shuffle_pvalue()` - Monte Carlo p-value with Phipson-Smyth correction
  - `compute_shuffle_zscore()` - Z-score from null distribution
  - `ShuffleTestResult` - Frozen dataclass container for results
- Comprehensive test suite: 31 new tests (164 total in test_shuffle.py)
- Code review passed with "APPROVE" rating
- All tests pass, ruff and mypy pass
- Functions exported in `decoding/__init__.py`

**Implementation highlights**:
- `compute_shuffle_pvalue`:
  - Uses `(k + 1) / (n + 1)` formula (Phipson-Smyth correction)
  - Prevents zero p-values, important for FDR/Bonferroni corrections
  - Supports "greater", "less", "two-sided" tails
  - Greater: counts null >= observed; Less: counts null <= observed
  - Two-sided: 2 * min(p_greater, p_less), capped at 1.0
- `compute_shuffle_zscore`:
  - Standard formula: (observed - mean(null)) / std(null)
  - Returns NaN for zero variance or single null value
- `ShuffleTestResult`:
  - Frozen dataclass for immutability
  - Fields: observed_score, null_scores, p_value, z_score, shuffle_type, n_shuffles
  - Property: `is_significant` (p < 0.05)
  - Method: `plot()` for null distribution visualization with histogram

**Design decisions**:
- Phipson-Smyth correction ensures unbiased Monte Carlo p-values
- Z-score returns NaN for degenerate cases (better than inf or zero division errors)
- Plot method has lazy matplotlib import for optional dependency
- Type annotation uses Literal["greater", "less", "two-sided"] for tail parameter

**Milestone 4.6 - Phase 4 Tests**: ✅ COMPLETED (164 tests total in test_shuffle.py)

**Milestone 4 COMPLETE!** All shuffle functions implemented:
- `shuffle_time_bins()` ✅
- `shuffle_time_bins_coherent()` ✅
- `shuffle_cell_identity()` ✅
- `shuffle_place_fields_circular()` ✅
- `shuffle_place_fields_circular_2d()` ✅
- `shuffle_posterior_circular()` ✅
- `shuffle_posterior_weighted_circular()` ✅
- `generate_poisson_surrogates()` ✅
- `generate_inhomogeneous_poisson_surrogates()` ✅
- `compute_shuffle_pvalue()` ✅
- `compute_shuffle_zscore()` ✅
- `ShuffleTestResult` ✅

**Next task**: Milestone 5 - Integration & Documentation
- 5.1: Public API Finalization
- 5.2: Visualization (optional)
- 5.3: Documentation
- 5.4: Optional Dependencies Check

### 2025-11-28 - Milestone 4.4 Complete (Surrogate Generation)

**Milestone 4.4 - Surrogate Generation**: ✅ COMPLETED

- Added two new generator functions to `src/neurospatial/decoding/shuffle.py`:
  - `generate_poisson_surrogates()` - Homogeneous Poisson surrogates from mean firing rates
  - `generate_inhomogeneous_poisson_surrogates()` - Time-varying rate surrogates with smoothed rates
- Comprehensive test suite: 32 new tests (133 total in test_shuffle.py)
- Code review passed with "APPROVE" rating
- All tests pass, ruff and mypy pass
- Functions exported in `decoding/__init__.py`

**Implementation highlights**:
- `generate_poisson_surrogates`:
  - Computes mean spike count per neuron across all time bins
  - Each (time_bin, neuron) pair independently samples from Poisson(mean_count)
  - Destroys all temporal structure while preserving average rates
- `generate_inhomogeneous_poisson_surrogates`:
  - Uses `scipy.ndimage.uniform_filter1d()` to smooth spike counts in time
  - Smoothed counts become time-varying Poisson rate parameters
  - Preserves slow rate fluctuations, destroys fine temporal structure
  - Default smoothing_window=3 bins
- Both functions follow same generator pattern as other shuffle functions
- `dt` parameter accepted for API consistency but not used in computation (counts used directly as rates)

**Design decisions**:
- Generators yield surrogates one at a time for memory efficiency
- Empty inputs handled gracefully (yield empty arrays)
- Uses `_ensure_rng()` helper for consistent RNG handling
- Type annotations include `NDArray[np.int64]` return type
- Comprehensive docstrings with scientific rationale

**Code review feedback addressed**:
- Added exports to `decoding/__init__.py` (imports and `__all__` list)
- Updated module docstring to document both functions

**Next task**: Milestone 4.5 - Significance Testing Functions
- `compute_shuffle_pvalue()` - Monte Carlo p-value with correction
- `compute_shuffle_zscore()` - Z-score from null distribution
- `ShuffleTestResult` - Result container dataclass

### 2025-11-28 - Milestone 4.3 Complete (Posterior Shuffles)

**Milestone 4.3 - Posterior Shuffles**: ✅ COMPLETED

- Added two new generator functions to `src/neurospatial/decoding/shuffle.py`:
  - `shuffle_posterior_circular()` - Circularly shift each posterior row (time bin) independently
  - `shuffle_posterior_weighted_circular()` - Edge-aware circular shift with edge buffer restriction
- Comprehensive test suite: 32 new tests (101 total in test_shuffle.py)
- Code review passed with "APPROVE" rating
- All tests pass, ruff and mypy pass

**Implementation highlights**:
- `shuffle_posterior_circular`: Simple circular shifts with `np.roll()` per time bin
- `shuffle_posterior_weighted_circular`: Smart edge buffer algorithm:
  - Computes MAP position per time bin
  - If MAP near left edge (< edge_buffer): restricts max positive shift to avoid wrapping to far right
  - If MAP near right edge (>= n_bins - edge_buffer): restricts max negative shift to avoid wrapping to far left
  - Center positions: full circular shift allowed
  - Example (n_bins=20, edge_buffer=5): MAP at bin 2 → shifts in [-2, 10); MAP at bin 17 → shifts in [-7, 3)
- Both functions preserve normalization (rows still sum to 1.0)
- Type annotations with `type: ignore[index]` for numpy scalar indexing (mypy false positive)

**Design decisions**:
- Edge buffer default is 5 bins (typical for linear track decoding)
- The edge buffer restricts shift range, not final position
- Handles degenerate cases: empty posterior yields empty copies, single bin is identity
- All functions follow same generator pattern as temporal shuffles

**Bug fix during review**:
- Initial implementation of edge buffer didn't actually restrict shifts properly
- Test revealed MAP could wrap to opposite end despite being near an edge
- Fixed by computing restricted shift ranges that keep MAP in "center region"

**Next task**: Milestone 4.4 - Surrogate Generation
- `generate_poisson_surrogates()` - Poisson spike count surrogates
- `generate_inhomogeneous_poisson_surrogates()` - Smoothed rate surrogates

### 2025-11-28 - Milestone 4.2 Complete (Cell Identity Shuffles)

**Milestone 4.2 - Cell Identity Shuffles**: ✅ COMPLETED

- Added three new generator functions to `src/neurospatial/decoding/shuffle.py`:
  - `shuffle_cell_identity()` - Permute spike train-to-place field mapping (column shuffle)
  - `shuffle_place_fields_circular()` - Circularly shift each place field independently
  - `shuffle_place_fields_circular_2d()` - 2D circular shifts for grid environments
- Comprehensive test suite: 38 new tests (69 total in test_shuffle.py)
- Code review passed with "APPROVE" rating
- All tests pass, ruff and mypy pass

**Implementation highlights**:
- `shuffle_cell_identity`: Returns tuple `(shuffled_counts, original_models)` for API clarity
- `shuffle_place_fields_circular`: Uses `np.roll()` with independent shifts per neuron
- `shuffle_place_fields_circular_2d`: Requires full grid (no inactive bins), validates grid_shape
- Added validation for 2D environments (n_dims == 2) and grid layout (grid_shape attribute)
- Clear error messages for masked grids: suggests using 1D shuffle instead
- Type annotations with `type: ignore[index]` for numpy scalar indexing (mypy false positive)

**Design decisions**:
- 2D circular shuffle requires `prod(grid_shape) == n_bins` (no masked grids)
- Masked grids would have unclear semantics after circular shift
- Error message explains this and suggests `shuffle_place_fields_circular` for masked environments
- All functions follow same generator pattern as temporal shuffles

**Next task**: Milestone 4.3 - Posterior Shuffles
- `shuffle_posterior_circular()` - Circular shift posterior rows
- `shuffle_posterior_weighted_circular()` - Edge effect mitigation

### 2025-11-28 - Milestone 4.1 Complete (Core Temporal Shuffles)

**Milestone 4.1 - Core Temporal Shuffles**: ✅ COMPLETED

- Created `src/neurospatial/decoding/shuffle.py` with two generator functions:
  - `shuffle_time_bins()` - Randomly permute time bins (rows) of spike counts
  - `shuffle_time_bins_coherent()` - Same as above but emphasizes coherent permutation
- Key design decisions:
  - Generator pattern for memory efficiency (yields one shuffle at a time)
  - `rng` parameter accepts `Generator | int | None` for reproducibility
  - Helper function `_ensure_rng()` for RNG handling
  - `.copy()` on yield to prevent reference bugs
- Comprehensive test suite: 31 tests covering:
  - Shape and dtype preservation
  - Statistical invariants (total spikes, per-neuron, per-time-bin)
  - Reproducibility with seeds and Generators
  - Edge cases (empty arrays, single time bin, single neuron)
  - Generator laziness verification
- Code review passed with "APPROVE" rating
- All 283 decoding tests pass, ruff and mypy pass

**Implementation details**:
- Both functions currently have identical implementation (row permutation)
- Naming distinction clarifies intent in null hypothesis testing
- Docstrings include scientific rationale and usage examples
- Module docstring has shuffle category table (temporal, cell identity, posterior, surrogate)

**Next task**: Milestone 4.2 - Cell Identity Shuffles
- `shuffle_cell_identity()` - Permute neuron-to-place-field mapping
- `shuffle_place_fields_circular()` - Circularly shift each place field
- `shuffle_place_fields_circular_2d()` - 2D circular shifts

### 2025-11-28 - Milestone 3.4 Complete (Radon Transform Detection)

**Milestone 3.4 - Radon Transform Detection**: ✅ COMPLETED

- Implemented `detect_trajectory_radon()` function in `trajectory.py`:
  - Uses scikit-image's `skimage.transform.radon()` for Radon transform
  - Optional dependency check with graceful ImportError message
  - Configurable theta_range and theta_step for angular resolution
  - Returns `RadonDetectionResult` with angle_degrees, score, offset, sinogram
- Added import guard at module level for scikit-image
- Comprehensive docstring with angle interpretation notes
- Added 21 tests (20 pass, 1 skipped for import guard when scikit-image unavailable)
- Code review passed with "APPROVE" rating
- All 252 decoding tests pass, ruff and mypy pass

**Implementation details**:
- Radon transform treats posterior as 2D image (time × position)
- Uses `circle=False` for rectangular image interpretation
- Peak detection with `np.unravel_index(np.argmax(sinogram))`
- Offset computed relative to centered projection: `offset_idx - (n_offsets - 1) / 2`
- Sinogram transposed to (n_angles, n_offsets) convention for consistency with docstring

**Angle interpretation**:
- θ = 0°: Horizontal line (constant position)
- θ = ±45°: Diagonal (forward/reverse replay)
- θ = 90°: Vertical line (instantaneous position change)

**Milestone 3.5 - Phase 3 Tests**: ✅ COMPLETED (61 tests total in test_trajectory.py)

**Next task**: Milestone 4 - Shuffle-Based Significance Testing
- 4.1: Core temporal shuffles (shuffle_time_bins, shuffle_time_bins_coherent)
- 4.2: Cell identity shuffles
- 4.3: Posterior shuffles
- 4.4: Surrogate generation
- 4.5: Significance testing functions

### 2025-11-28 - Milestone 3.3 Complete (Linear Regression)

**Milestone 3.3 - Linear Regression**: ✅ COMPLETED

- Implemented `fit_linear_trajectory()` function in `trajectory.py`:
  - Two methods: `method="map"` (argmax) and `method="sample"` (Monte Carlo)
  - Uses cumulative-sum sampling for numerical stability with peaky posteriors
  - Handles `rng` parameter for reproducibility (int seed or Generator)
  - Returns `LinearFitResult` with slope, intercept, r_squared, slope_std
- Added input validation for posterior shape and times length
- Documented edge cases (constant positions, constant times)
- Added comprehensive test suite (13 new tests)
- Code review passed with "APPROVE" rating
- All 44 trajectory tests pass, ruff and mypy pass

**Implementation details**:
- Cumulative-sum sampling: `cumsum = np.cumsum(posterior, axis=1); samples = np.argmax(cumsum >= u, axis=-1)`
- Avoids issues with `np.random.choice` on near-zero probabilities
- Helper function `_fit_line()` implements basic least squares regression
- R² computed using `_compute_r_squared()` helper (shared with isotonic)

**Next task**: Milestone 3.4 - Radon Transform Detection (`detect_trajectory_radon`)

### 2025-11-28 - Milestone 3.2 Complete (Isotonic Regression)

**Milestone 3.2 - Isotonic Regression**: ✅ COMPLETED

- Implemented `fit_isotonic_trajectory()` function in `trajectory.py`:
  - Uses scikit-learn's `IsotonicRegression` (PAVA algorithm)
  - Supports `method="map"` (argmax) and `method="expected"` (weighted mean)
  - Auto-selects direction (increasing/decreasing) based on R² when `increasing=None`
  - Returns `IsotonicFitResult` with fitted_positions, r_squared, direction, residuals
- Added comprehensive test suite (15 new tests)
- All 27 trajectory tests pass, ruff and mypy pass

**Next task**: Milestone 3.3 - Linear Regression (`fit_linear_trajectory`)

### 2025-11-28 - Milestone 3.1 Complete (Result Dataclasses)

**Milestone 3.1 - Result Dataclasses**: ✅ COMPLETED

- Created `src/neurospatial/decoding/trajectory.py` with three frozen dataclasses:
  - `IsotonicFitResult`: fitted_positions, r_squared, direction, residuals
  - `LinearFitResult`: slope, intercept, r_squared, slope_std
  - `RadonDetectionResult`: angle_degrees, score, offset, sinogram
- All dataclasses have comprehensive NumPy docstrings with examples
- Exported in `decoding/__init__.py` and `__all__` list
- Created test suite: `tests/decoding/test_trajectory.py` (12 tests)
- All tests pass, ruff and mypy pass

**Next task**: Milestone 3.2 - Isotonic Regression (`fit_isotonic_trajectory`)

### 2025-11-28 - Milestone 2.4 Complete (Phase 2 Tests)

**Milestone 2.4 - Phase 2 Tests**: ✅ COMPLETED

- Verified all 64 tests in `tests/decoding/test_metrics.py` pass
- Test coverage confirmed for all Phase 2 requirements:
  - `decoding_error()` - known positions, graph metric, NaN handling
  - `median_decoding_error()` - basic functionality, NaN handling
  - `confusion_matrix()` - shapes, sums, MAP vs expected methods
  - `decoding_correlation()` - range, symmetry, weighted, NaN handling
- All tests pass (64 passed, 1 warning for expected NaN behavior)

**Phase 2 Complete!** All metrics functions implemented and tested:

- `decoding_error()` ✅
- `median_decoding_error()` ✅
- `confusion_matrix()` ✅
- `decoding_correlation()` ✅

**Next task**: Milestone 3 - Trajectory Analysis

- 3.1: Result dataclasses (IsotonicFitResult, LinearFitResult, RadonDetectionResult)
- 3.2: Isotonic regression
- 3.3: Linear regression
- 3.4: Radon transform detection

### 2025-11-28 - Milestone 2.3 Complete (Correlation Metric)

**Milestone 2.3 - Correlation Metric**: ✅ COMPLETED

- Added `decoding_correlation()` function to `src/neurospatial/decoding/metrics.py`
- Implemented weighted Pearson correlation with numerically stable centered formula
- Key features:
  - Uses `np.average()` for weighted means (stable computation)
  - Centers data before computing covariance (prevents catastrophic cancellation)
  - Handles multi-dimensional positions (computes per-dimension correlation, returns mean)
  - Excludes NaN values and zero-weight bins automatically
  - Returns NaN for edge cases: <2 valid samples, zero variance, weight sum overflow
- Created comprehensive test suite (22 new tests covering all edge cases)
- All tests pass (191 total in decoding/, 2 skipped)
- Code review passed with "APPROVE" rating
- Ruff and mypy pass with no issues

**Implementation highlights**:

- Numerically stable centered formula avoids cancellation for large datasets
- Explicit overflow check: `if weight_sum == 0 or not np.isfinite(weight_sum)`
- Per-dimension correlation with NaN propagation if any dimension has zero variance
- Comprehensive docstring with NumPy format, pseudo-code explanation, and examples
- Exported in `__init__.py` and `__all__` list

**Next task**: Milestone 2.4 - Phase 2 Tests (complete test suite for metrics.py)

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
