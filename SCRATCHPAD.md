# Neurospatial Simulation Development Scratchpad

**Last Updated**: 2025-11-11

## Current Status

### Milestone 1: ✅ COMPLETE

**Completed:**

- All module structure in place
- NeuralModel Protocol implemented
- PlaceCellModel implemented and tested
- simulate_trajectory_ou implemented and tested
- simulate_trajectory_sinusoidal implemented (tests skipped for 1D env)
- generate_poisson_spikes implemented and tested
- generate_population_spikes implemented and tested
- All mypy errors fixed (17 errors resolved)
- All doctests passing (8/8)
- All ruff warnings fixed
- Test suite: 41/44 tests passing (3 skipped for 1D environment requirement)
- Code review approved with all critical and important issues resolved
- ✅ Committed to main (commit ad96365)

### Milestone 2: 🚧 IN PROGRESS

**Completed:**

- ✅ `simulate_trajectory_laps()` implemented and tested
  - 7/7 tests passing
  - Full parameter validation (n_laps, speeds, pause_duration, sampling_frequency)
  - Path validation (bin index bounds checking)
  - Smart boundary handling using `map_points_to_bins()` for efficiency
  - Proper NumPy docstring format
  - Code review applied and all critical/important issues resolved
  - Mypy clean, ruff clean, doctests passing
  - Exported in `neurospatial.simulation.__init__.py`

- ✅ `BoundaryCellModel` implemented and tested
  - 10/10 tests passing
  - TDD approach (tests written first, verified failures, then implementation)
  - Supports omnidirectional (border cells) and directional (BVC) tuning
  - Both geodesic and euclidean distance metrics implemented
  - Gaussian distance tuning around preferred_distance
  - Von Mises directional tuning for BVCs (2D only)
  - Comprehensive parameter validation with clear error messages
  - Code review applied (all critical and high-priority issues fixed):
    - Added type annotation for `_distance_field: NDArray[np.float64] | None`
    - Added parameter validation (distances, rates, tolerances)
    - Fixed variable scope in directional tuning
    - Added `direction_tolerance` to `ground_truth` property
  - Mypy clean (no issues), ruff clean (all checks passed)
  - Exported in `neurospatial.simulation.models` and `neurospatial.simulation`

## Technical Decisions Made

### BoundaryCellModel Implementation

1. **boundary_bins is a property, not method**: Fixed from `env.boundary_bins()` → `env.boundary_bins`
2. **Variable scope clarification**: Precompute `boundary_centers` at start of `firing_rate()` to avoid scope issues across geodesic/euclidean branches
3. **Geodesic + Directional hybrid**: For directional BVCs with geodesic metric, use geodesic distance (path through graph) but Euclidean direction (geometric angle). This is scientifically correct - direction is inherently geometric, not path-based.
4. **Type annotation for mypy**: Explicit type annotations required for:
   - `_distance_field: NDArray[np.float64] | None` (before assignment)
   - `rates: NDArray[np.float64]` (before return) to avoid "Any" inference
5. **Parameter validation**: Comprehensive validation upfront with diagnostic error messages:
   - `preferred_distance >= 0`
   - `distance_tolerance > 0`
   - `max_rate > 0`
   - `baseline_rate >= 0`
   - `baseline_rate < max_rate`
   - `direction_tolerance > 0`
6. **ground_truth completeness**: Include all model parameters including `direction_tolerance` (needed to reproduce model)
7. **Distance precomputation**: For geodesic metric, precompute distance field from all boundary bins at initialization (O(n_bins²) one-time cost, then O(1) lookup per position)

### simulate_trajectory_laps() Implementation

1. **Speed clamping**: Uses absolute minimum (0.01) instead of relative to `speed_mean` to prevent division by zero regardless of parameter values
2. **Boundary handling**: Uses `map_points_to_bins()` for efficient vectorized boundary checking instead of looping with `env.contains()`
3. **Path validation**: Validates bin indices upfront with clear error messages before trajectory generation
4. **Mypy type safety**: Renamed `lap_boundaries` list to `lap_boundaries_list` to avoid list→array type reassignment errors
5. **Default path selection**: Uses `nx.shortest_path()` between environment extrema (min/max of first dimension) with graceful fallback
6. **Metadata structure**: Returns dict with 'lap_ids', 'lap_boundaries', and 'direction' arrays for complete lap characterization

### Mypy Type Fixes (Milestone 1)

1. **env.bin_at() returns array, not scalar**: Must use `int(env.bin_at(pos)[0])` to extract scalar
2. **dimension_ranges can be None**: Added None checks before indexing
3. **np.argmin() returns scalar**: Cast to int explicitly
4. **Optional distance_field**: Declared as `NDArray[np.float64] | None`
5. **Unreachable else blocks**: Removed (Literal types cover all cases)
6. **Return type shape mismatches**: Added `# type: ignore[return-value]` where numpy shape inference fails
7. **tqdm import**: Added `# type: ignore[import-untyped]`

### Doctest Reproducibility

- Added seeds to random trajectory generation
- Placed place fields at environment center for higher hit rate
- Increased duration and max_rate to ensure spike generation
- Changed exact value checks to range checks where appropriate

## Code Review Fixes Applied

Following the code-reviewer subagent's recommendations:

### Critical Fix (Completed)
- ✅ Removed unused type ignore comment in spikes.py

### Important Improvements (Completed)
- ✅ Added validation for anisotropic width + geodesic metric in PlaceCellModel.__init__
  - Raises clear ValueError explaining geodesic distance doesn't have directional components
  - Prevents scientifically meaningless results
- ✅ Added upfront validation for periodic boundary mode requirements
  - Now validates `env.dimension_ranges` at function entry (fail-fast)
  - Removed redundant check inside boundary handling loop
  - Provides clear error message with guidance

- ✅ `add_modulation()` implemented and tested
  - 16/16 tests passing (10 core + 6 edge case validation tests)
  - Non-homogeneous thinning algorithm for phase-locked firing
  - Full parameter validation (modulation_depth ∈ [0,1], modulation_freq > 0)
  - Special case: depth=0 returns all spikes (no modulation)
  - Comprehensive NumPy docstring with biological motivation
  - Code review applied and all critical/important issues resolved:
    - Fixed mypy type error (`.copy()` → `np.array(..., copy=True)`)
    - Added parameter validation with diagnostic error messages
    - Exported in `neurospatial.simulation.__init__.py`
    - Added 6 edge case tests (invalid params, single spike, high freq)
  - Mypy clean (no issues), ruff clean (all checks passed)
  - Exported in `neurospatial.simulation.__init__.py`

### add_modulation() Implementation

1. **Zero modulation depth special case**: When `modulation_depth=0.0`, return all spikes without random thinning. Formula `(1 + 0 * cos(phase)) / 2 = 0.5` would give uniform 50% acceptance, but zero depth should mean "no modulation".
2. **Parameter validation upfront**: Validate `modulation_depth ∈ [0,1]` and `modulation_freq > 0` to prevent silent failures with invalid inputs. Follows project pattern of explicit validation.
3. **Typed array copy**: Use `np.array(spike_times, dtype=np.float64, copy=True)` instead of `.copy()` to satisfy mypy's strict type checking.
4. **Phase computation**: Direct formula `2π * freq * spike_times + phase` without wrapping. NumPy's `np.cos()` handles large phase values correctly.
5. **Acceptance probability formula**: `(1 + depth * cos(phase)) / 2` ensures valid probabilities ∈ [0,1] for all phases when depth ∈ [0,1].
6. **Test robustness**: Used larger sample sizes (2000 spikes) and bigger depth differences (0.1 vs 0.95) to reduce probabilistic test flakiness.

## Milestone 2 Completion Summary

**Status**: ✅ COMPLETE

All Milestone 2 tasks successfully implemented, tested, and validated:

### Core Implementations

- ✅ `simulate_trajectory_laps()` - 7/7 tests passing (commit 6024dcc)
- ✅ `BoundaryCellModel` - 10/10 tests passing (commit 8558b68)
- ✅ `add_modulation()` - 16/16 tests passing (commit 5b90bc5)

### Documentation & Validation

- ✅ Comprehensive NumPy docstrings with examples for all functions
- ✅ Doctests: 12/12 passing across entire simulation module
- ✅ Test suite: 74 passed, 3 skipped
- ✅ Coverage: 73% overall (92% for spikes.py, 83% for place_cells.py, 75% for boundary_cells.py)
- ✅ Mypy: No issues found in 7 source files
- ✅ Ruff: All checks passed

### Test Breakdown

- `test_trajectory.py`: 17 tests (14 passed, 3 skipped for 1D trajectory)
- `test_models.py`: 24 tests (all passed)
- `test_spikes.py`: 32 tests (all passed)
- **Total**: 74 passed, 3 skipped

## Milestone 3: GridCellModel Implementation

**Status**: ✅ COMPLETE

### Implementation Details

1. **Hexagonal grid pattern**: Uses three plane waves at 60° intervals
   - Wave vector magnitude: `k = 4π / (√3 * grid_spacing)`
   - Three wave vectors rotated by `grid_orientation` at 0°, 60°, 120°
   - Grid pattern: `g(x) = (1/3) * Σ cos(k_i · (x - phase_offset))`
   - Rectification: `rate = baseline + (max - baseline) * max(0, g(x))`

2. **Parameter validation**: All parameters validated upfront
   - `grid_spacing > 0`
   - `max_rate > 0`
   - `baseline_rate >= 0` and `< max_rate`
   - `phase_offset` shape must be `(2,)`
   - 2D environment required (raises ValueError for 1D/3D)

3. **Code review improvements applied**:
   - ✅ Fixed doctest examples (undefined variables `arena_data` and `env`)
   - ✅ Added parameter validation (5 validators)
   - ✅ Added 6 parameter validation tests
   - ✅ Removed unused `field_width` parameter (was stored but never used in computation)
   - ✅ Enhanced docstring Notes with performance, stability, and grid spacing selection guidance
   - ✅ Fixed numpy bool doctest issue (`np.True_` → use `bool()` wrapper)

### Test Coverage

- **16 tests total**, all passing
  - 10 functional tests (2D requirement, initialization, hexagonal symmetry, spacing, orientation, output shape, bounds, ground truth, protocol compliance, default phase)
  - 6 validation tests (negative/zero grid_spacing, invalid rates, wrong phase_offset shape)
- **2 doctests**, both passing
- **Mypy**: Clean (no issues)
- **Ruff**: Clean (all checks passed)

### Key Design Decisions

1. **No field_width parameter**: Removed because unused in `firing_rate()` computation. Standard hexagonal grid model doesn't have variable bump width. Can add later if needed.

2. **Phase offset at origin by default**: `phase_offset=[0, 0]` creates grid with vertices at origin and periodic spacing.

3. **Performance**: O(n_positions) with low constant (3 cosine evaluations per position). Much faster than geodesic distance metrics.

4. **Numerical stability**: Cosine gratings bounded [-1, 1], so grid pattern bounded [-1, 1]. After rectification max(0, g(x)), pattern bounded [0, 1], preventing overflow/underflow.

## Next Steps

1. ✅ Implement GridCellModel - DONE
2. Commit GridCellModel implementation - Ready
3. Continue Milestone 3: Session API and validation helpers

## Blockers

None - GridCellModel complete and tested.
