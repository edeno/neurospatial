# Neurospatial v0.3.0 Development Notes

## 2025-11-07: Milestone 0.1 - Prerequisites COMPLETED

### Task: Add `return_seconds` parameter to `env.occupancy()` method

**Status**: ✅ COMPLETE

**Files Modified**:

1. `src/neurospatial/environment/trajectory.py` - Added parameter and implementation
2. `src/neurospatial/environment/_protocols.py` - Updated Protocol definition
3. `tests/test_occupancy.py` - Added comprehensive test suite

**Implementation Details**:

- Added `return_seconds: bool = True` parameter (default True for backward compatibility)
- When `True`: returns time in seconds (time-weighted occupancy) - **EXISTING BEHAVIOR**
- When `False`: returns interval counts (unweighted, each interval = 1)
- Updated both "start" and "linear" time allocation methods
- All 24 tests pass (19 existing + 5 new)
- Mypy passes with zero errors

**Key Design Decisions**:

1. **Default to `True`**: Maintains backward compatibility - all existing code continues to work without changes
2. **Interval-based counting**: For `return_seconds=False`, we count the number of intervals (not samples), which is consistent with how occupancy is calculated
3. **Linear allocation handling**: For linear allocation with `return_seconds=False`, we normalize the proportional time allocations to sum to 1.0 per interval

**Test Coverage**:

- Basic true/false behavior with multiple bins
- Stationary samples (tests constant occupancy)
- Multiple bins with varying durations
- Interaction with speed filtering
- All tests use proper grid construction to avoid bin mapping issues

**Code Review Findings**:

- ✅ Type safety: Mypy passes with no errors
- ✅ Backward compatibility: Default behavior maintained
- ✅ Documentation: NumPy-style docstrings complete
- ✅ Test coverage: Comprehensive (5 new tests, all pass)
- ✅ Edge cases: Handled properly (empty arrays, single sample, etc.)

**Next Steps**:

Ready to move on to implementing the `spikes_to_field()` and `compute_place_field()` functions in Milestone 0.1.

---

## 2025-11-07: Milestone 0.1 - Spike → Field Conversion COMPLETE

### Task: Implement `spikes_to_field()` and `compute_place_field()` functions

**Status**: ✅ COMPLETE

**Files Created**:

1. `src/neurospatial/spike_field.py` - Core implementation module (346 lines)
2. `tests/test_spike_field.py` - Comprehensive test suite (14 tests, all pass)

**Files Modified**:

1. `src/neurospatial/environment/_protocols.py` - Added `occupancy()` and `smooth()` method signatures to Protocol
2. `src/neurospatial/__init__.py` - Added public API exports for new functions

**Implementation Details**:

**`spikes_to_field()` function:**
- Converts spike trains to occupancy-normalized firing rate fields (spikes/second)
- Parameters: `env, spike_times, times, positions, *, min_occupancy_seconds=0.0`
- **Default behavior**: Includes all bins (min_occupancy_seconds=0.0), no NaN filtering by default
- **Optional NaN filtering**: Set min_occupancy_seconds > 0 (e.g., 0.5) to exclude unreliable bins
- Full input validation: times/positions length check, 1D/2D position normalization, negative min_occupancy check
- Handles edge cases: empty spikes, out-of-bounds (time/space), all-NaN occupancy
- 1D trajectory support: accepts both `(n,)` and `(n, 1)` position shapes
- Comprehensive NumPy-style docstring with examples and LaTeX math
- Uses `env.occupancy(return_seconds=True)` for time-weighted normalization

**`compute_place_field()` convenience function:**
- One-liner combining `spikes_to_field()` + optional `env.smooth()`
- Parameters: same as `spikes_to_field` + `smoothing_bandwidth: float | None`
- Default: `min_occupancy_seconds=0.0` (no filtering), `smoothing_bandwidth=None` (no smoothing)
- Handles NaN values in smoothing: fills with 0, smooths, restores NaN
- If `smoothing_bandwidth=None`, equivalent to `spikes_to_field()`

**Test Coverage**: 14 comprehensive tests (100% pass rate)
- Synthetic data with known firing rate
- Min occupancy threshold (NaN masking)
- Empty spike trains
- Out-of-bounds spikes (time and space)
- 1D trajectories (both column vector and bare array)
- All-NaN occupancy edge case
- Manual computation verification
- Parameter order validation
- Input validation (mismatched lengths, negative min_occupancy)
- Smoothing with/without NaN handling

**Type Safety**:
- ✅ Mypy passes with zero errors
- ✅ No `type: ignore` comments
- ✅ Full type hints using `EnvironmentProtocol`
- ✅ Updated Protocol with `occupancy()` and `smooth()` signatures

**Code Quality**:
- ✅ Ruff check passes
- ✅ Ruff format applied
- ✅ NumPy-style docstrings throughout
- ✅ Comprehensive examples in docstrings

**Critical Fixes Applied** (from code review):
1. **1D trajectory bug**: Fixed missing normalization of positions to 2D at function start
2. **Validation**: Added check for negative `min_occupancy_seconds`
3. **Test coverage**: Added test for bare 1D positions `(n,)` without column dimension

**Known Limitations** (documented):
1. **Smoothing NaN handling**: Current approach (fill-with-0) can artificially reduce firing rates near unvisited regions. This is a pragmatic trade-off. For scientific applications requiring high precision near boundaries, users should call `spikes_to_field()` and `env.smooth()` separately with custom handling.

**Public API Additions**:
- `neurospatial.spikes_to_field(env, spike_times, times, positions, *, min_occupancy_seconds=0.0)`
- `neurospatial.compute_place_field(env, spike_times, times, positions, *, min_occupancy_seconds=0.0, smoothing_bandwidth=None)`

**Next Task**: Move to Milestone 0.3 - Documentation for Phase 0 primitives

---

## 2025-11-07: Milestone 0.2 - Reward Field Primitives COMPLETE

### Task: Implement `region_reward_field()` and `goal_reward_field()` functions

**Status**: ✅ COMPLETE

**Files Created**:

1. `src/neurospatial/reward.py` - Core implementation module (336 lines)
2. `tests/test_reward.py` - Comprehensive test suite (15 tests, all pass)

**Files Modified**:

1. `src/neurospatial/__init__.py` - Added public API exports for new functions

**Implementation Details**:

**`region_reward_field()` function:**
- Generates reward fields from named regions with three decay types:
  - `decay="constant"` - Binary reward (reward_value inside region, 0 outside)
  - `decay="linear"` - Linear decay from region boundary using distance field
  - `decay="gaussian"` - Smooth Gaussian falloff (requires bandwidth parameter)
- **Critical fix**: Gaussian decay rescales by max *within region* (not global max)
- Full input validation: region existence, bandwidth requirement for Gaussian
- Comprehensive NumPy-style docstring with RL references (Ng et al., 1999)
- Uses `Literal["constant", "linear", "gaussian"]` for type-safe decay parameter

**`goal_reward_field()` function:**
- Generates distance-based reward fields from goal bins with three decay types:
  - `decay="exponential"` - `scale * exp(-d/scale)` (most common in RL)
  - `decay="linear"` - Linear decay reaching zero at max_distance
  - `decay="inverse"` - Inverse distance `scale / (1 + d)`
- Handles scalar or array goal_bins input (converts scalar to array)
- Validates goal bin indices are in valid range
- Validates scale > 0 for exponential decay
- Multi-goal support: distance computed to nearest goal
- Uses `Literal["linear", "exponential", "inverse"]` for type-safe decay parameter

**Test Coverage**: 15 comprehensive tests (100% pass rate)
- All decay types for both functions
- Edge cases (multiple goals, scalar vs array, custom reward values)
- Error paths (missing bandwidth, invalid regions, out-of-range bins, negative scale)
- Parameter naming validation (ensures API stability)
- Numerical correctness (comparing against expected formulas)

**Type Safety**:
- ✅ Mypy passes with zero errors
- ✅ No `type: ignore` comments
- ✅ Full type hints using `EnvironmentProtocol`
- ✅ TYPE_CHECKING guards for imports

**Code Quality**:
- ✅ Ruff check passes (all linting rules satisfied)
- ✅ Ruff format applied (consistent code style)
- ✅ NumPy-style docstrings throughout
- ✅ Comprehensive examples in docstrings

**Code Review Findings** (all fixed):
- ✅ Removed unused `type: ignore` comments (mypy compliance)
- ✅ Fixed doctest failure (suppressed output from `regions.add()`)
- ✅ Exported functions in public API `__init__.py`
- ✅ All validation comprehensive and user-friendly

**Public API Additions**:
- `neurospatial.region_reward_field(env, region_name, *, reward_value=1.0, decay="constant", bandwidth=None)`
- `neurospatial.goal_reward_field(env, goal_bins, *, decay="exponential", scale=1.0, max_distance=None)`

**Design Decisions**:

1. **Consistent parameter naming**: Used `decay` (not `falloff` or `kind`) across both functions
2. **Environment-first order**: Matches project pattern (e.g., `spikes_to_field()`, `distance_field()`)
3. **Gaussian rescaling**: Rescales by max IN REGION to preserve intended reward magnitude
4. **Error messages**: Include diagnostic information (e.g., available regions, valid range)
5. **Scalar handling**: `goal_reward_field()` accepts scalar or array goal_bins for convenience

**Known Limitations** (documented):
1. Linear decay in `region_reward_field()` normalizes by global max distance (may give non-zero rewards far from region)
2. Could add optional `max_distance` parameter for local reward shaping (deferred as optional enhancement)

**Scientific Correctness**:
- ✅ Mathematically sound formulas validated against RL literature
- ✅ Proper integration with graph-based distance fields
- ✅ Boundary detection correct for region-based rewards
- ✅ Potential-based reward shaping (Ng et al., 1999) properly referenced

**Next Task**: Milestone 0.3 - Example notebook creation

---

## 2025-11-07: Milestone 0.3 - Documentation COMPLETE (Part 1/2)

### Task: Create user guide documentation for Phase 0 primitives

**Status**: ✅ COMPLETE (Documentation files created)

**Files Created**:

1. `docs/user-guide/spike-field-primitives.md` - Comprehensive spike-to-field conversion guide (260 lines)
2. `docs/user-guide/rl-primitives.md` - Complete RL reward field guide (490 lines)

**Documentation Coverage**:

**spike-field-primitives.md:**
- Converting spike trains to spatial fields
- Why occupancy normalization matters (neuroscience standard)
- Parameter order (env first, consistent API)
- `compute_place_field()` convenience function
- Min occupancy threshold best practices (0.5 seconds standard)
- Edge case handling (empty spikes, out-of-bounds, NaN, 1D trajectories)
- Complete code examples with visualizations
- Note about deferred batch operations (v0.3.1)

**rl-primitives.md:**
- Region-based reward field generation (constant, linear, gaussian decay)
- Goal-based reward field generation (exponential, linear, inverse decay)
- Reward shaping strategies and best practices
- Consistent `decay` parameter naming across functions
- Gaussian falloff rescaling (uses max IN REGION - critical fix documented)
- Cautions about reward shaping (Ng et al. 1999 reference)
- Potential-based reward shaping theory
- Combining reward sources
- Multiple goals support (Voronoi partitioning)
- Complete RL-specific examples

**Design Highlights**:

1. **Consistent Style**: Matches existing neurospatial documentation format
2. **Practical Examples**: Every concept has runnable code examples
3. **Best Practices**: Clear recommendations based on neuroscience/RL literature
4. **Warnings**: Explicit cautions about when shaping can hurt (Ng et al.)
5. **API References**: Cross-linked to related functions and concepts
6. **Mathematical Rigor**: Formulas and references for all decay functions

**Next Steps**:

- Run coverage tests for Phase 0 code
- Verify notebook executes without errors
- Complete Milestone 0.3 (verify all tests pass, documentation complete)

---

## 2025-11-07: Milestone 0.3 - Example Notebook COMPLETE (Part 2/2)

### Task: Create example notebook for Phase 0 primitives

**Status**: ✅ COMPLETE

**File Created**:

`examples/09_spike_field_basics.ipynb` (renumbered from 00 to fit existing sequence)

**Notebook Features**:

1. **Part 1: Spike Train to Firing Rate Maps**
   - Generate synthetic circular trajectory with Gaussian place cell
   - Create environment and compute occupancy
   - Convert spike train to firing rate with `spikes_to_field()`
   - Demonstrate min occupancy threshold filtering (0.5s standard)
   - Show `compute_place_field()` convenience function with smoothing
   - Visualize: occupancy vs raw vs filtered vs smoothed

2. **Part 2: Reward Fields for RL**
   - Region-based rewards (constant, linear, gaussian)
   - Goal-based distance rewards (exponential, linear, inverse)
   - Multi-goal support demonstration
   - Combining reward sources
   - Visual comparisons of all decay types

3. **Part 3: Best Practices**
   - Cautions about reward shaping (Ng et al. 1999)
   - Potential-based reward shaping theory
   - Testing reward designs
   - References to key papers

**Technical Enhancements**:

- Used jupyter-notebook-editor skill for proper pairing (jupytext)
- Applied scientific-figures-presentation principles:
  - Constrained layout for better spacing
  - Wong color palette for accessibility
  - Larger, readable fonts (12-14pt)
  - Improved marker sizes and line weights
  - Clear, bold titles
- Comprehensive markdown explanations throughout
- Estimated time: 20-25 minutes

**Validation**:

- ✅ Notebook paired with .py file via jupytext
- ✅ Proper numbering (09 instead of 00)
- ✅ All plotting code enhanced for clarity
- ✅ Complete examples for all Phase 0 functions

**Notebook Execution Fixes** (2025-11-07):

All plotting errors fixed and notebook executes successfully:

1. **Plotting API Fix** (10 sections fixed):
   - Replaced incorrect `env.plot(field, ax=axes[i])` calls
   - Used correct pattern: `ax.scatter(env.bin_centers[:, 0], env.bin_centers[:, 1], c=field)`
   - Added colorbars, axis labels, and aspect ratio settings
   - Applied to: occupancy/firing rate comparison, raw vs smoothed, region rewards (3 plots), goal rewards (3 plots), multi-goal reward, combined rewards (3 plots)

2. **NaN Handling Fix**:
   - Removed manual `env.smooth()` call that failed on NaN values
   - Kept only `compute_place_field()` which handles NaN properly
   - Added explanatory comment about NaN handling in smoothing

3. **Region Definition Fix**:
   - Changed from point region to circular polygon (12 cm radius)
   - Used `shapely.geometry.Point(goal_location).buffer(12.0)`
   - Ensures region has area and contains bin centers
   - Fixed goal_location to use existing bin center instead of hardcoded coordinates

4. **Goal Bin Selection Fix**:
   - Used `env.bin_centers[idx]` instead of hardcoded coordinates
   - Ensures goal locations are within environment bounds
   - Multi-goal example uses bins at 1/3 and 2/3 positions (opposite quadrants)

**Execution Result**:

```
SUCCESS: Notebook executed without errors
[NbConvertApp] Writing 655232 bytes to examples/09_spike_field_basics.ipynb
```

All 29 Phase 0 tests pass, notebook executes cleanly with all visualizations.

**Milestone 0.3 Status**: ✅ **COMPLETE**

---

## 2025-11-07: Milestone 0 - Final Verification COMPLETE

### Task: Verify all Milestone 0 Success Criteria

**Status**: ✅ COMPLETE

**Verification Results**:

1. **Tests**: ✅ All 29 tests pass (14 spike_field + 15 reward)
   - `uv run pytest tests/test_spike_field.py tests/test_reward.py -v`
   - 29 passed, 2 warnings (expected UserWarnings for edge cases)

2. **Type Safety**: ✅ Mypy passes with zero errors
   - `uv run mypy src/neurospatial/spike_field.py src/neurospatial/reward.py --ignore-missing-imports --warn-unused-ignores`
   - Success: no issues found in 2 source files

3. **Notebook**: ✅ `examples/09_spike_field_basics.ipynb` exists and executes successfully
   - Created 2025-11-07, 655KB size
   - All plotting errors fixed in previous session

4. **Documentation**: ✅ Complete
   - `docs/user-guide/spike-field-primitives.md` (12KB)
   - `docs/user-guide/rl-primitives.md` (15KB)

**Success Criteria Verified**:

- ✅ `spikes_to_field()` uses correct parameter order (env first)
- ✅ `compute_place_field()` convenience function works
- ✅ Min occupancy threshold correctly filters to NaN
- ✅ 1D and multi-D trajectory handling works
- ✅ Input validation comprehensive (11 validation tests)
- ✅ `region_reward_field()` supports all decay types with `decay` parameter
- ✅ `goal_reward_field()` supports all decay types with `decay` parameter
- ✅ Gaussian rescaling uses max IN REGION
- ✅ All tests pass with 100% pass rate
- ✅ Zero mypy errors
- ✅ Example notebook complete with visualizations
- ✅ Documentation complete and cross-referenced

**Updated Files**:
- `TASKS.md` - All Milestone 0 Success Criteria marked complete [x]

**Next Task**: Begin Milestone 1.1 - Differential Operator Matrix

---

## 2025-11-07: Milestone 1.1 - Differential Operator Matrix COMPLETE

### Task: Implement differential operator infrastructure

**Status**: ✅ COMPLETE

**Files Created**:

1. `src/neurospatial/differential.py` - Core differential operator module (145 lines)
2. `tests/test_differential.py` - Comprehensive test suite (13 tests, all pass)

**Files Modified**:

1. `src/neurospatial/environment/core.py` - Added `differential_operator` cached property
2. `src/neurospatial/environment/_protocols.py` - Updated Protocol with differential_operator property

**Implementation Details**:

**`compute_differential_operator(env)` function:**
- Extracts edge data from `env.connectivity` graph
- Computes sqrt of edge weights (distances) following PyGSP convention
- Builds sparse CSC matrix (n_bins × n_edges) using COO → CSC conversion
- Handles edge cases: empty graphs, disconnected components, single nodes
- Comprehensive NumPy-style docstring with PyGSP references
- Full type hints: `-> sparse.csc_matrix`
- Uses `Union[Environment, EnvironmentProtocol]` to satisfy both mypy contexts

**`Environment.differential_operator` cached property:**
- Added to `src/neurospatial/environment/core.py` (line 922-986)
- Uses `@cached_property` decorator for efficient reuse
- Includes `@check_fitted` decorator for safety
- Comprehensive NumPy-style docstring with examples
- Proper return type annotation: `-> sparse.csc_matrix`

**Mathematical Correctness:**
- Sign convention correct: source node gets -sqrt(w), destination gets +sqrt(w)
- Verified fundamental relationship: L = D @ D.T (Laplacian)
- Edge weights use sqrt(distance) scaling per graph signal processing convention

**Test Coverage**: 13 comprehensive tests (100% pass rate)
- Shape verification (n_bins, n_edges)
- Laplacian relationship: D @ D.T == nx.laplacian_matrix()
- Sparse format (CSC)
- Edge weight computation (sqrt scaling)
- Caching behavior (same object returned)
- Edge cases: single node, disconnected graph
- Regular grids, irregular spacing
- Symmetry preservation

**Type Safety**:
- ✅ Mypy passes with zero errors
- ✅ No `type: ignore` comments
- ✅ Full type hints using `EnvironmentProtocol` and `Environment`
- ✅ Protocol updated with proper return type

**Code Quality**:
- ✅ Ruff check passes (all linting rules satisfied)
- ✅ Ruff format applied (consistent code style)
- ✅ NumPy-style docstrings throughout
- ✅ Fixed variable naming to comply with PEP 8 (d_coo, d_csc)

**Code Review Findings** (all fixed):
- ✅ Added return type annotation to property
- ✅ Removed forward references to non-existent gradient/divergence functions
- ✅ Fixed Protocol return type (sparse.csc_matrix instead of Any)
- ✅ Fixed Union type hint to satisfy mypy in both contexts

**TDD Workflow Followed**:
1. ✅ Created comprehensive tests first (RED phase)
2. ✅ Verified tests failed (ModuleNotFoundError)
3. ✅ Implemented functionality (GREEN phase)
4. ✅ All 13 tests pass
5. ✅ Code review applied
6. ✅ Refactored based on feedback
7. ✅ Mypy and ruff pass

**Updated Files**:
- `TASKS.md` - All Milestone 1.1 checkboxes marked complete [x]

**Next Task**: Milestone 1.3 - Divergence Operator

---

## 2025-11-07: Milestone 1.2 - Gradient Operator COMPLETE

### Task: Implement `gradient(field, env)` function

**Status**: ✅ COMPLETE

**Files Created**:

1. Tests added to `tests/test_differential.py` - New `TestGradientOperator` class (4 tests, all pass)

**Files Modified**:

1. `src/neurospatial/differential.py` - Added `gradient()` function (lines 148-253)
2. `src/neurospatial/__init__.py` - Exported gradient in public API

**Implementation Details**:

**`gradient(field, env)` function:**
- Computes gradient of scalar field: `gradient(f) = D.T @ f`
- Input validation: checks `field.shape == (env.n_bins,)`
- Returns edge field with shape `(n_edges,)`
- Uses cached `env.differential_operator` property for efficiency
- Handles sparse matrix result conversion to dense NDArray[np.float64]
- Comprehensive NumPy-style docstring with graph signal processing references (Shuman et al., 2013)
- Two working docstring examples (constant field, linear field)
- Cross-references to divergence (future), compute_differential_operator, Environment.differential_operator

**Test Coverage**: 4 comprehensive tests (100% pass rate)
- Shape validation (n_edges,)
- Constant field gradient = 0
- Linear field gradient is constant on regular grid
- Input validation (wrong shape raises ValueError with diagnostic message)

**Type Safety**:
- ✅ Mypy passes with zero errors
- ✅ No `type: ignore` comments
- ✅ Full type hints using `NDArray[np.float64]` (precise type annotation)
- ✅ Union type `Environment | EnvironmentProtocol` for flexibility

**Code Quality**:
- ✅ Ruff check passes (all linting rules satisfied)
- ✅ Ruff format applied (consistent code style)
- ✅ NumPy-style docstring with Examples, Notes, References sections
- ✅ Proper variable naming (diff_op instead of D to comply with PEP 8)

**Code Review Findings** (code-reviewer agent):
- ✅ **APPROVED** - Production ready
- ✅ Mathematical correctness verified
- ✅ Input validation comprehensive
- ✅ Documentation excellent (references graph signal processing theory)
- ✅ Test coverage thorough
- ✅ Type safety perfect (NDArray type annotations added per review suggestion)
- ✅ No critical or blocking issues

**TDD Workflow Followed**:
1. ✅ Created 4 tests first in TestGradientOperator class
2. ✅ Verified tests FAIL with ImportError (RED phase)
3. ✅ Implemented gradient() function (GREEN phase)
4. ✅ All 4 tests pass
5. ✅ Applied code-reviewer agent
6. ✅ Fixed NDArray type annotation per review suggestion
7. ✅ Mypy and ruff pass with zero errors

**Public API Additions**:
- `neurospatial.gradient(field, env)` - Compute gradient of scalar field on graph

**Mathematical Foundation**:
- Gradient: scalar field → edge field (D.T @ f)
- Foundation for Laplacian: D @ D.T @ f = div(grad(f))
- Adjoint of divergence operation (to be implemented in M1.3)

**Performance**:
- Uses cached differential_operator for efficiency
- Sparse matrix operations for large graphs
- Result converted to dense array for user convenience

**Design Decisions**:
1. **Parameter order**: `gradient(field, env)` - field first, consistent with numpy conventions
2. **Return type**: Always dense NDArray[np.float64], never sparse (user-friendly)
3. **Validation**: Clear error message showing expected vs actual shape
4. **Type hints**: Precise NDArray[np.float64] annotations (not generic np.ndarray)
5. **Documentation**: Full mathematical context with graph signal processing references

**Known Limitations** (documented):
- None - implementation is complete and production-ready

**Next Task**: Milestone 1.3 - Divergence Operator (rename KL divergence, implement graph divergence)

---
