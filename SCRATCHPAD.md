# SCRATCHPAD - neurospatial Development Notes

**Last Updated**: 2025-11-03
**Current Phase**: Phase 2 - Core Analysis Operations
**Current Status**: P0.1 Occupancy Complete

---

## Current Task

**Phase 2, Task P0.1**: Occupancy / Dwell Time (COMPLETE ✅)

### Steps:
1.  Read project documentation (CLAUDE.md, TASKS.md, ENVIRONMENT_OPS_PLAN.md)
2. � Create `tests/test_kernels.py` (TEST FIRST - TDD)
3. � Run tests and verify FAILURE
4. � Implement `src/neurospatial/kernels.py`
5. � Run tests until PASS
6. � Apply code review
7. � Refactor based on feedback
8. � Update docstrings and types

---

## Notes

### Design Decisions

**Kernel Infrastructure**:
- Will implement `compute_diffusion_kernels()` as main public function
- Two normalization modes: `transition` (column sums = 1) and `density` (volume-corrected)
- Foundation for all smoothing operations
- Performance warning: O(n�) complexity for matrix exponential

**Testing Strategy**:
- Test kernel symmetry on uniform grids
- Test normalization (both modes)
- Test mass conservation
- Test cache behavior
- Test across multiple layout types

---

## Blockers

None currently.

---

## Questions/Decisions Pending

None currently.

---

## Implementation Log

### 2025-11-03 - Session Start
- Read all project documentation
- Identified first task: Kernel infrastructure
- Ready to begin TDD cycle with test creation

- Created comprehensive test suite in tests/test_kernels.py
- Discovered existing src/neurospatial/kernels.py (untracked file)
- Running tests shows failures due to sparse matrix operations
- Need to fix implementation: scipy.sparse.linalg.expm returns dense array


## Phase 1 Complete! (2025-11-03)

### Summary
- Created comprehensive kernel infrastructure (22 tests, all passing)
- Implemented compute_diffusion_kernels() in src/neurospatial/kernels.py
- Added Environment.compute_kernel() wrapper method
- All code review feedback addressed
- Public API exported in __init__.py

### Files Created/Modified
- NEW: src/neurospatial/kernels.py (126 lines)
- NEW: tests/test_kernels.py (455 lines, 22 tests)
- MODIFIED: src/neurospatial/environment.py (added compute_kernel method + _kernel_cache field)
- MODIFIED: src/neurospatial/__init__.py (added compute_diffusion_kernels export)

### Code Quality
- NumPy docstring format: ✅
- Type safety: ✅ (100% coverage)
- Error handling: ✅ (comprehensive validation)
- Test coverage: ✅ (22/22 passing)
- TDD compliance: ✅
- Performance warnings: ✅ (O(n³) documented)

### Next Steps
- Phase 2: Core Analysis Operations (occupancy, sequences, transitions, connectivity)
- Ready to begin implementation

---

## Phase 2, P0.1 Complete! (2025-11-03)

### Summary
- Implemented `Environment.occupancy()` method for computing time spent in each bin
- Comprehensive test suite (19 tests, all passing)
- Follows strict TDD methodology
- All code review feedback addressed

### Implementation Details
- **Method**: `Environment.occupancy(times, positions, *, speed=None, min_speed=None, max_gap=0.5, kernel_bandwidth=None)`
- **Location**: src/neurospatial/environment.py (lines 1611-1820)
- **Features**:
  - Gap handling (default max_gap=0.5s to filter large time jumps)
  - Speed filtering (exclude slow periods via min_speed threshold)
  - Kernel smoothing (mode='transition' for mass conservation)
  - Mass conservation guaranteed: occupancy.sum() = total valid time
  - Efficient np.bincount for accumulation
  - Leverages spatial.map_points_to_bins with KDTree caching

### Key Design Decisions

1. **Time Interval Allocation**: Each Δt assigned to starting bin (simple, fast)
   - Future enhancement (P2.11): "linear" allocation across bin boundaries

2. **Kernel Smoothing Mode**: Use mode='transition' (not 'density')
   - Reason: Occupancy is a count field, not a density
   - transition mode: kernel columns sum to 1 → mass conservation
   - density mode: kernel columns integrate to 1 over bin volumes → would lose mass

3. **Default max_gap=0.5**: Prevents including unrealistic time gaps
   - Common in neuroscience: tracking systems can have dropouts
   - Users can set max_gap=None to include all intervals

4. **Monotonic Timestamp Validation**: Added after code review
   - Prevents silent errors from non-monotonic data
   - Provides diagnostic info (indices of violations)

5. **Removed outside_value parameter**: Unused in current implementation
   - Will be added in future when implementing "linear" allocation (P2.11)
   - Clean API: only parameters that are actually used

### Files Created/Modified
- NEW: tests/test_occupancy.py (390 lines, 19 tests)
- MODIFIED: src/neurospatial/environment.py (added occupancy() method, ~210 lines)

### Test Coverage
Test organization (6 test suites):
1. **TestOccupancyBasic**: Core functionality (4 tests)
2. **TestOccupancyGapHandling**: max_gap parameter (2 tests)
3. **TestOccupancySpeedFiltering**: min_speed parameter (2 tests)
4. **TestOccupancySmoothing**: kernel_bandwidth parameter (2 tests)
5. **TestOccupancyOutsideBehavior**: Points outside environment (2 tests)
6. **TestOccupancyValidation**: Input validation (3 tests)
7. **TestOccupancyMassConservation**: Property tests (2 tests)
8. **TestOccupancyPerformance**: Large trajectory (1 test)
9. **TestOccupancyMultipleLayouts**: Different layout types (2 tests)

### Code Quality Metrics
- NumPy docstring format: ✅
- Type safety: ✅ (Complete type annotations)
- Input validation: ✅ (Comprehensive with diagnostic errors)
- Test coverage: ✅ (19/19 passing)
- TDD compliance: ✅ (Tests written first, verified failure, then implementation)
- Linting: ✅ (ruff check passed)
- Code review: ✅ (All critical and quality issues addressed)

### Code Review Feedback Addressed
**Critical Issues Fixed**:
- ✅ Removed invalid `bin_sequence` reference in "See Also" section
- ✅ Removed unused `outside_value` parameter

**Quality Issues Fixed**:
- ✅ Converted regex patterns to raw strings (r"...") in tests
- ✅ Removed empty placeholder test
- ✅ Added monotonic timestamp validation with diagnostics
- ✅ Improved comment clarity about -1 bin indices

### Performance
- 100k samples: < 0.15s (tested)
- Efficient vectorized operations throughout
- KDTree caching via map_points_to_bins

### Next Steps
- Phase 2, Task P0.3: Transitions / Adjacency Matrix
- Ready to begin TDD cycle

---

## Phase 2, P0.2 Complete! (2025-11-03)

### Summary
- Implemented `Environment.bin_sequence()` method for trajectory-to-bin-sequence conversion
- Comprehensive test suite (24 tests, all passing)
- Follows strict TDD methodology
- All code review feedback addressed

### Implementation Details
- **Method**: `Environment.bin_sequence(times, positions, *, dedup=True, return_runs=False, outside_value=-1)`
- **Location**: src/neurospatial/environment.py (lines 1826-2053)
- **Features**:
  - Deduplication (default dedup=True collapses consecutive repeats)
  - Run-length encoding (return_runs=True provides start/end indices)
  - Outside handling (outside_value=-1 marks outside, None drops them)
  - Uses bin_at() which correctly returns -1 for points outside environment
  - Deterministic behavior via bin_at()'s underlying layout.point_to_bin_index()

### Key Design Decisions

1. **Validation Consistency**: Matched `occupancy()` validation patterns
   - Raise ValueError for non-monotonic times (not just warn)
   - Require 2D positions array (no auto-reshape from 1D)
   - Comprehensive diagnostics in error messages

2. **Run Encoding**: Correctly handles both dedup=True and dedup=False
   - With dedup: run boundaries based on deduplicated sequence
   - Without dedup: run boundaries based on full bin_indices array
   - Handles outside_value=None affecting run boundaries

3. **Outside Handling**: Uses bin_at() instead of map_points_to_bins
   - bin_at() returns -1 for points outside environment (correct behavior)
   - map_points_to_bins() always maps to nearest bin (incorrect for this use case)

4. **Type Safety**: Explicit dtype conversions
   - bin_indices converted to int32 (consistent with return type)
   - run boundaries use int64 (for large trajectory support)

### Files Created/Modified
- NEW: tests/test_bin_sequence.py (507 lines, 24 tests)
- MODIFIED: src/neurospatial/environment.py (added bin_sequence() method, ~235 lines)

### Test Coverage
Test organization (9 test suites):
1. **TestBinSequenceBasic**: Core functionality (4 tests)
2. **TestBinSequenceDeduplication**: dedup parameter (3 tests)
3. **TestBinSequenceRuns**: Run-length encoding (4 tests)
4. **TestBinSequenceOutsideBehavior**: outside_value parameter (3 tests)
5. **TestBinSequenceValidation**: Input validation (4 tests)
6. **TestBinSequenceMultipleLayouts**: Different layout types (2 tests)
7. **TestBinSequenceEdgeCases**: Boundary conditions (4 tests)

### Code Quality Metrics
- NumPy docstring format: ✅
- Type safety: ✅ (Complete type annotations)
- Input validation: ✅ (Comprehensive with diagnostic errors)
- Test coverage: ✅ (24/24 passing)
- TDD compliance: ✅ (Tests written first, verified failure, then implementation)
- Linting: ✅ (ruff check passed)
- Code review: ✅ (All quality issues addressed)

### Code Review Feedback Addressed
**Quality Issues Fixed**:
- ✅ Fixed time validation inconsistency (now raises ValueError like occupancy())
- ✅ Fixed positions.ndim validation (now requires 2D like occupancy())
- ✅ Updated docstring to reflect ValueError for non-monotonic times
- ✅ Updated test to expect ValueError instead of UserWarning

**Approved Aspects**:
- Excellent NumPy docstring with comprehensive examples
- Proper @check_fitted decorator usage
- Clear algorithm structure and efficient implementation
- Correct run-length encoding logic
- Comprehensive edge case handling

### Performance
- Expected: Fast for typical neuroscience trajectories (tested with various sizes)
- Efficient vectorized NumPy operations throughout
- bin_at() delegation to layout engine (varies by layout type)

### Next Steps (Current Status)
- Phase 2, Task P0.3: Transitions / Adjacency Matrix
- Ready to begin TDD cycle
