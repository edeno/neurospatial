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

## Next Steps

1. ✅ Update TASKS.md checkboxes - DONE
2. ✅ Commit Milestone 1 completion - DONE (commit ad96365)
3. Begin Milestone 2 (Boundary cells + laps) - Awaiting user instruction

## Blockers

None - all critical functionality implemented and tested.
