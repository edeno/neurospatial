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

## Next Up
- Task 1.3: `compute_directional_place_fields` function

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
