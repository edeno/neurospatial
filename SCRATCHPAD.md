# Directional Place Fields - Development Scratchpad

## Current Status
- **Date**: 2025-01-27
- **Working on**: Milestone 1 - Core Infrastructure

## Completed Tasks

### Task 1.1: DirectionalPlaceFields dataclass ✅
- Added frozen dataclass to `src/neurospatial/spike_field.py`
- Added `Mapping` and `dataclass` imports
- Comprehensive NumPy-style docstring with Attributes, Examples, See Also sections
- All 6 tests pass
- Ruff and mypy pass

### Task 1.2: _subset_spikes_by_time_mask helper ✅
- Added private helper function to `src/neurospatial/spike_field.py`
- Finds contiguous True segments in mask using np.diff on indices
- Uses searchsorted for O(log n) spike slicing per segment
- All 9 tests pass covering edge cases:
  - Single segment, multiple segments
  - Empty mask, all-True mask
  - No spikes, spikes at boundaries
  - Single timepoint segment
- Ruff and mypy pass

### Task 1.3: compute_directional_place_fields ✅
- Main function that computes place fields conditioned on direction labels
- Reuses compute_place_field for each direction (no code duplication)
- Excludes "other" label from results
- Returns DirectionalPlaceFields dataclass
- All 7 tests pass covering:
  - Constant labels equals compute_place_field
  - Two directions partition
  - "other" label exclusion
  - Empty spike train handling
  - Length mismatch error
  - All "other" returns empty
- Fixed mypy issue: `NDArray[object]` → `NDArray[np.object_]`
- Ruff and mypy pass

### Task 2.1: goal_pair_direction_labels ✅
- Added to `src/neurospatial/behavioral.py`
- Generates per-timepoint direction labels from trial data
- Uses arrow notation: "start→end" (Unicode right arrow U+2192)
- Failed trials (end_region=None) are labeled "other"
- Later trials overwrite earlier if overlapping (documented behavior)
- Inclusive boundaries: `>=` and `<=`
- All 9 tests pass covering:
  - Basic trials, outside trials, failed trials
  - Overlapping trials, empty inputs, boundary inclusive
  - Arrow notation format, multiple region names
- Comprehensive NumPy docstring
- Ruff and mypy pass

### Task 2.2: heading_direction_labels ✅
- Added to `src/neurospatial/behavioral.py`
- Generates per-timepoint direction labels from heading angle
- Accepts either (positions, times) or precomputed (speed, heading)
- Precomputed values take precedence if both provided
- Bins heading into sectors (e.g., "−180–−135°", "0–45°", etc.)
- Labels slow-moving periods (< min_speed) as "stationary"
- Uses en-dash (–, U+2013) and degree symbol (°) for professional labels
- Edge cases handled: empty arrays, single timepoint, mismatched lengths
- All 18 tests pass covering:
  - Straight paths (+x, +y directions)
  - Stationary detection and min_speed threshold
  - Precomputed vs computed kinematics (equivalence and precedence)
  - Input validation (error cases for missing/incomplete/mismatched inputs)
  - Binning behavior (n_directions customization, boundary cases)
  - Negative angles, output shape/dtype
- Comprehensive NumPy docstring with scientific detail
- Code review: APPROVED with length validation enhancement added
- Ruff and mypy pass

## Next Up
- Task 3.1: `directional_field_index` function (metrics/place_fields.py)

## Design Decisions
1. **Frozen dataclass**: Makes `DirectionalPlaceFields` immutable for safety
2. **Mapping type for fields**: Allows dict-like access while being more generic
3. **Tuple for labels**: Preserves iteration order (important for reproducibility)
4. **Inclusive boundaries**: Spikes exactly at segment start/end are included
5. **searchsorted efficiency**: O(log n) spike lookup vs O(n) linear scan

## Notes
- Following TDD: write test → fail → implement → pass → refactor
- Tests located in `tests/test_directional_place_fields.py`

## Blockers
None currently.
