# Events Module Implementation Tasks

**Last Updated**: 2025-12-04
**Source Plan**: [PLAN.md](PLAN.md)

---

## Overview

This task list breaks down the Events Module implementation into actionable steps. Each milestone corresponds to a phase from PLAN.md.

**Success Criteria** (from PLAN.md):

- [ ] All existing tests pass after integration
- [ ] NWB round-trip works for all event types
- [ ] GLM workflow example runs successfully
- [ ] PSTH matches scipy/elephant implementations (validated)
- [ ] Documentation is complete and follows codebase patterns
- [ ] Type checking passes (mypy)
- [ ] Linting passes (ruff)

---

## Milestone 1: Core Infrastructure

**Goal**: Create module structure with dataclasses and validation helpers.

**Dependencies**: None

### Tasks

- [x] **M1.1**: Create `src/neurospatial/events/` directory structure
  - Create `__init__.py` with lazy imports (follow `nwb/__init__.py` pattern)
  - Create `_core.py` for dataclasses and validation
  - Create empty `detection.py`, `intervals.py`, `regressors.py`, `alignment.py`

- [x] **M1.2**: Implement `PeriEventResult` dataclass in `_core.py`
  - Fields: `bin_centers`, `histogram`, `sem`, `n_events`, `window`, `bin_size`
  - Add `firing_rate()` method
  - Use `@dataclass(frozen=True)` for immutability
  - Follow NumPy docstring format

- [x] **M1.3**: Implement `PopulationPeriEventResult` dataclass in `_core.py`
  - Fields: `bin_centers`, `histograms`, `sem`, `mean_histogram`, `n_events`, `n_units`, `window`, `bin_size`
  - Add `firing_rates()` method
  - Use `@dataclass(frozen=True)`

- [x] **M1.4**: Implement `validate_events_dataframe()` in `_core.py`
  - Check: is DataFrame, has timestamp column, timestamp is numeric
  - Support `required_columns` parameter
  - Use diagnostic error messages (WHAT/WHY/HOW pattern from existing code)

- [x] **M1.5**: Implement `validate_spatial_columns()` in `_core.py`
  - Check for `x`, `y` columns
  - Support `require_positions` parameter to optionally raise
  - No `env` parameter needed (just checks column presence)

- [x] **M1.6**: Create `tests/test_events_core.py`
  - Test dataclass creation and methods
  - Test validation functions with valid/invalid inputs
  - Test edge cases: empty DataFrame, missing columns, wrong types

- [x] **M1.7**: Verify milestone completion
  - Run `uv run pytest tests/test_events_core.py -v`
  - Run `uv run mypy src/neurospatial/events/`
  - Run `uv run ruff check src/neurospatial/events/`

---

## Milestone 2: Temporal GLM Regressors

**Goal**: Implement time-based regressors for GLM design matrices.

**Dependencies**: Milestone 1

### Tasks

- [x] **M2.1**: Implement `time_to_nearest_event()` in `regressors.py`
  - Refactored from separate `time_since_event()` / `time_to_event()` to unified function
  - Parameters: `sample_times`, `event_times`, `signed`, `max_time`
  - Returns signed peri-event time (negative before, positive after) - matches PSTH convention
  - Use `np.searchsorted` for efficient nearest-event lookup
  - Handle edge cases: empty arrays, NaN/Inf validation, midpoint tie-breaking

- [x] **M2.2**: Implement `event_count_in_window()` in `regressors.py`
  - Parameters: `sample_times`, `event_times`, `window`
  - Return type: `NDArray[np.int64]`
  - Handle: empty events → return zeros

- [x] **M2.3**: Implement `event_indicator()` in `regressors.py`
  - Parameters: `sample_times`, `event_times`, `window`
  - Return type: `NDArray[np.bool_]`
  - Handle: empty events → return all False

- [x] **M2.4**: ~~Implement `exponential_kernel()` in `regressors.py`~~ **DEFERRED**
  - **Reason**: nemos/patsy provide superior GLM basis functions; avoid duplication
  - **Alternative**: Users can combine `time_to_nearest_event()` with `np.exp()` directly

- [x] **M2.5**: Verify milestone completion
  - Run `uv run pytest tests/test_events_regressors.py -v` ✓ (63 tests pass)
  - Run `uv run mypy src/neurospatial/events/regressors.py` ✓

---

## Milestone 3: Event Detection Functions **DEFERRED**

**Status**: Deferred - detection functions can be implemented when needed

**Reason**:
- `extract_region_crossing_events()` wraps existing `segmentation.detect_region_crossings()`
- `extract_threshold_crossing_events()` and `extract_movement_onset_events()` are general signal processing
- Users can use scipy/numpy directly for threshold detection
- Focus on spatial event utilities (M3.4-M3.6) which provide unique value

### Deferred Tasks

- [ ] **M3.1**: `extract_region_crossing_events()` - deferred
- [ ] **M3.2**: `extract_threshold_crossing_events()` - deferred
- [ ] **M3.3**: `extract_movement_onset_events()` - deferred
- [ ] **M3.5**: `events_in_region()` - deferred (can use Environment.contains() directly)
- [ ] **M3.6**: `spatial_event_rate()` - deferred (can use existing place field methods)
- [ ] **M3.7**: Additional detection tests - deferred
- [ ] **M3.8**: Milestone verification - deferred

### Completed Tasks

- [x] **M3.4**: Implement `add_positions()` in `detection.py`
  - Parameters: `events`, `positions`, `times`, `timestamp_column`
  - Interpolate positions at event times using scipy.interpolate.interp1d
  - Supports 1D, 2D, 3D trajectories with linear interpolation/extrapolation
  - Only adds `x`, `y` (and `z` if 3D) - no derived columns
  - Return new DataFrame (don't modify input)
  - 20 tests in `tests/test_events_detection.py`

Milestone 3 Status: PARTIALLY COMPLETE

- `add_positions()` implemented (20 tests)
- Remaining spatial utilities deferred - users can combine `add_positions()` with existing Environment methods

---

## Milestone 4: Interval Utilities

**Goal**: Implement utilities for converting between point events and intervals.

**Dependencies**: Milestone 1

### Tasks

- [x] **M4.1**: Implement `intervals_to_events()` in `intervals.py`
  - Parameters: `intervals`, `which`, `start_column`, `stop_column`, `preserve_columns`
  - Support `which`: "start", "stop", "both"
  - When "both", add `boundary` column

- [x] **M4.2**: Implement `events_to_intervals()` in `intervals.py`
  - Parameters: `start_events`, `stop_events`, `match_by`, `max_duration`
  - Sequential pairing when `match_by` is None
  - Match by column value when `match_by` specified
  - Return DataFrame with: `start_time`, `stop_time`, `duration`

- [x] **M4.3**: Implement `filter_by_intervals()` in `intervals.py`
  - Parameters: `events`, `intervals`, `include`, `timestamp_column`, `start_column`, `stop_column`
  - Use efficient interval overlap detection
  - Support both inclusion and exclusion filtering

- [x] **M4.4**: Create `tests/test_events_intervals.py`
  - Test round-trip: intervals → events → intervals
  - Test filtering with overlapping intervals
  - Test edge cases: empty intervals, single interval, adjacent intervals

- [x] **M4.5**: Verify milestone completion
  - Run `uv run pytest tests/test_events_intervals.py -v` ✓ (45 tests pass)
  - Run `uv run mypy src/neurospatial/events/intervals.py` ✓

**Milestone 4 Status: COMPLETE**

---

## Milestone 5: Spatial Distance Regressors

**Goal**: Implement spatial distance-based regressors.

**Dependencies**: Milestones 1, 3 (for `add_positions`)

### Tasks

- [ ] **M5.1**: Implement `distance_to_event_at_time()` in `regressors.py`
  - Parameters: `sample_times`, `sample_positions`, `events`, `env`, `metric`, `which`
  - Support `which`: "last", "next"
  - Find temporally relevant event, then compute spatial distance
  - Support `metric`: "euclidean", "geodesic"

- [ ] **M5.2**: Add tests for spatial regressors in `tests/test_events_regressors.py`
  - Test euclidean vs geodesic distance
  - Test with 1D and 2D environments
  - Test edge cases: no events, single event location

- [ ] **M5.3**: Verify milestone completion
  - Run `uv run pytest tests/test_events_regressors.py -v`

---

## Milestone 6: Peri-Event Analysis

**Goal**: Implement PSTH and event alignment functions.

**Dependencies**: Milestones 1, 2

### Tasks

- [ ] **M6.1**: Implement `align_spikes_to_events()` in `alignment.py`
  - Parameters: `spike_times`, `event_times`, `window`
  - Return list of arrays (one per event) with relative spike times
  - Low-level function for raster plots and custom analyses
  - Used internally by `peri_event_histogram()`

- [ ] **M6.2**: Implement `peri_event_histogram()` in `alignment.py`
  - Parameters: `spike_times`, `event_times`, `window`, `bin_size`, `baseline_window`
  - Use `align_spikes_to_events()` internally
  - Compute histogram with mean ± SEM across events
  - Handle edge case: single event → SEM is NaN with warning
  - Return `PeriEventResult`

- [ ] **M6.3**: Implement `population_peri_event_histogram()` in `alignment.py`
  - Parameters: `spike_trains`, `event_times`, `window`, `bin_size`, `baseline_window`
  - Process multiple spike trains
  - Compute per-unit and population statistics
  - Return `PopulationPeriEventResult`

- [ ] **M6.4**: Implement `align_events()` in `alignment.py`
  - Parameters: `events_df`, `reference_events`, `window`, `reference_column`, `event_column`
  - Extract events within window of each reference
  - Add `relative_time` and `reference_index` columns
  - Handle edge case: empty reference events → empty DataFrame

- [ ] **M6.5**: Implement `plot_peri_event_histogram()` in `_core.py`
  - Parameters: `result`, `ax`, `show_sem`, `color`, `as_rate`, `title`, `xlabel`, `ylabel`
  - Create matplotlib figure if `ax` not provided
  - Show shaded SEM region
  - Return axes for further customization

- [ ] **M6.6**: Create `tests/test_events_alignment.py`
  - Test `align_spikes_to_events()` returns correct per-trial data
  - Test PSTH with known spike patterns
  - Test against reference implementation (scipy.ndimage or manual calculation)
  - Test population PSTH
  - Test edge cases: no spikes in window, overlapping events

- [ ] **M6.7**: Verify milestone completion
  - Run `uv run pytest tests/test_events_alignment.py -v`
  - Run `uv run mypy src/neurospatial/events/alignment.py`

---

## Milestone 7: NWB Integration

**Goal**: Add generic event writing to NWB module.

**Dependencies**: Milestones 1, 3

### Tasks

- [ ] **M7.1**: Implement `write_events()` in `nwb/_events.py`
  - Parameters: `nwbfile`, `events`, `name`, `description`, `processing_module`, `overwrite`
  - Convert DataFrame columns to EventsTable columns
  - Handle spatial columns (x, y, bin_index, region)
  - Support overwrite mode

- [ ] **M7.2**: Implement `dataframe_to_events_table()` in `nwb/_events.py`
  - Convert standard DataFrame to ndx-events EventsTable
  - Map column types appropriately
  - Handle optional columns

- [ ] **M7.3**: Update `nwb/__init__.py` exports
  - Export `write_events` and `dataframe_to_events_table`
  - Maintain lazy import pattern

- [ ] **M7.4**: Create `tests/test_events_nwb.py`
  - Test round-trip: DataFrame → NWB → DataFrame
  - Test with spatial columns
  - Test with custom columns
  - Test overwrite behavior

- [ ] **M7.5**: Verify milestone completion
  - Run `uv run pytest tests/test_events_nwb.py -v`

---

## Milestone 8: Top-Level Exports and Documentation

**Goal**: Export all functions and update documentation.

**Dependencies**: All previous milestones

### Tasks

- [ ] **M8.1**: Update `src/neurospatial/__init__.py`
  - Add events module imports (lazy)
  - Export: `add_positions`, `events_in_region`, `spatial_event_rate`, etc.
  - Note: `add_spatial_columns` renamed to `add_positions`

- [ ] **M8.2**: Update `src/neurospatial/events/__init__.py`
  - Create `__all__` list with all public exports
  - Implement lazy imports for optional dependencies

- [ ] **M8.3**: Update `.claude/QUICKSTART.md`
  - Add "Events" section with common patterns
  - Include GLM regressor example
  - Include PSTH example

- [ ] **M8.4**: Update `.claude/API_REFERENCE.md`
  - Add events module imports section
  - List all functions with signatures

- [ ] **M8.5**: Update `CLAUDE.md`
  - Add events to "Quick Navigation" tables
  - Add common event patterns to "Most Common Patterns"

- [ ] **M8.6**: Final verification
  - Run full test suite: `uv run pytest`
  - Run type checking: `uv run mypy src/neurospatial/`
  - Run linting: `uv run ruff check . && uv run ruff format .`
  - Verify imports work: `uv run python -c "from neurospatial.events import peri_event_histogram"`

---

## Quick Reference: File Locations

| File | Purpose |
|------|---------|
| `src/neurospatial/events/__init__.py` | Public API, lazy imports |
| `src/neurospatial/events/_core.py` | Dataclasses, validation, plotting |
| `src/neurospatial/events/detection.py` | Event detection from data |
| `src/neurospatial/events/intervals.py` | Interval conversion utilities |
| `src/neurospatial/events/regressors.py` | GLM regressor generation |
| `src/neurospatial/events/alignment.py` | Peri-event analysis (PSTH) |
| `src/neurospatial/nwb/_events.py` | NWB read/write (extend existing) |
| `tests/test_events_core.py` | Core tests |
| `tests/test_events_detection.py` | Detection tests |
| `tests/test_events_intervals.py` | Interval tests |
| `tests/test_events_regressors.py` | Regressor tests |
| `tests/test_events_alignment.py` | Alignment tests |
| `tests/test_events_nwb.py` | NWB integration tests |

---

## Notes

- All functions must follow NumPy docstring format
- Use diagnostic error messages (WHAT/WHY/HOW pattern)
- Frozen dataclasses for immutable result objects
- Validate inputs early with clear error messages
- Handle edge cases gracefully (see PLAN.md "Edge Case Handling")

## Design Philosophy: Spatial Events

**Store coordinates, compute everything else.**

- `add_positions()` only adds `x`, `y` columns (no `bin_index`, no `region`)
- `events_in_region()` computes region membership from coordinates at query time
- `spatial_event_rate()` requires pre-computed occupancy as explicit input

**Why?** Derived data (`bin_index`, `region`) depends on Environment configuration and can become stale. By storing only coordinates and computing on demand, events remain portable across different Environment configurations.

**Exception**: If `region` is an intrinsic event property (e.g., opto stim trigger zone), users add it as a domain-specific column themselves.
