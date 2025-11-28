# Directional Place Fields - Implementation Tasks

This document breaks down the implementation plan into actionable tasks organized by milestone. Each task includes success criteria and dependencies.

---

## Milestone 1: Core Infrastructure

Foundation for directional place field computation. No external dependencies.

### Task 1.1: Add DirectionalPlaceFields dataclass

**File**: `src/neurospatial/spike_field.py`

**What to do**:

1. Add imports: `from dataclasses import dataclass` and `from collections.abc import Mapping`
2. Add frozen dataclass after existing imports, before function definitions:

   ```python
   @dataclass(frozen=True)
   class DirectionalPlaceFields:
       fields: Mapping[str, NDArray[np.float64]]
       labels: tuple[str, ...]
   ```

3. Add NumPy-style docstring documenting both attributes

**Success criteria**:

- [x] Dataclass is frozen (immutable)
- [x] `fields` maps string labels to 1D firing rate arrays
- [x] `labels` is a tuple preserving iteration order
- [x] Docstring follows NumPy format with Attributes section

**Dependencies**: None

---

### Task 1.2: Add _subset_spikes_by_time_mask helper

**File**: `src/neurospatial/spike_field.py`

**What to do**:

1. Add private helper function after imports, before public functions:

   ```python
   def _subset_spikes_by_time_mask(
       times: NDArray[np.float64],
       spike_times: NDArray[np.float64],
       mask: NDArray[np.bool_],
   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
   ```

2. Implementation:
   - Find contiguous True segments in mask using `np.diff` on mask indices
   - For each segment, get `t_start = times[seg_start]`, `t_end = times[seg_end]`
   - Use `np.searchsorted(spike_times, t_start, side="left")` and `np.searchsorted(spike_times, t_end, side="right")` to slice spikes
   - Concatenate all spike slices
   - Return `(times[mask], concatenated_spikes)`

**Success criteria**:

- [x] Correctly identifies contiguous segments
- [x] Uses searchsorted for O(log n) spike slicing per segment
- [x] Returns times_sub with same length as `np.sum(mask)`
- [x] Returns spike_times_sub containing only spikes within masked time ranges
- [x] Handles edge cases: empty mask, no spikes, single segment

**Dependencies**: None

---

### Task 1.3: Add compute_directional_place_fields function

**File**: `src/neurospatial/spike_field.py`

**What to do**:

1. Add function with signature from PLAN.md Section 1.2
2. Implementation:
   - Validate `len(direction_labels) == len(times)`
   - Convert labels: `labels_arr = np.asarray(direction_labels, dtype=object)`
   - Get unique labels: `unique_labels = [l for l in np.unique(labels_arr) if l != "other"]`
   - For each label:
     - Build mask: `mask = labels_arr == label`
     - Get subsets: `times_sub, spike_times_sub = _subset_spikes_by_time_mask(times, spike_times, mask)`
     - Get positions: `positions_sub = positions[mask]`
     - Call: `field = compute_place_field(env, spike_times_sub, times_sub, positions_sub, method=method, bandwidth=bandwidth, min_occupancy_seconds=min_occupancy_seconds)`
     - Store in dict
   - Return `DirectionalPlaceFields(fields=fields_dict, labels=tuple(unique_labels))`
3. Add comprehensive NumPy-style docstring

**Success criteria**:

- [ ] Signature matches PLAN.md exactly
- [ ] Reuses `compute_place_field` for each direction (no duplication)
- [ ] Excludes "other" label from results
- [ ] Returns DirectionalPlaceFields with correct structure
- [ ] Docstring includes Parameters, Returns, Raises, Examples sections

**Dependencies**: Task 1.1, Task 1.2

---

## Milestone 2: Direction Label Helpers

Functions to generate direction labels for common use cases.

### Task 2.1: Add goal_pair_direction_labels function

**File**: `src/neurospatial/behavioral.py`

**What to do**:

1. Add import: `from neurospatial.segmentation import Trial` (use TYPE_CHECKING guard)
2. Add function:

   ```python
   def goal_pair_direction_labels(
       times: NDArray[np.float64],
       trials: list[Trial],
   ) -> NDArray[object]:
   ```

3. Implementation:
   - Initialize: `labels = np.full(len(times), "other", dtype=object)`
   - Loop over trials:
     - Skip if `trial.end_region is None` (failed trial)
     - Create label: `f"{trial.start_region}→{trial.end_region}"`
     - Create mask: `mask = (times >= trial.start_time) & (times <= trial.end_time)`
     - Assign: `labels[mask] = label`
   - Return labels
4. Add NumPy-style docstring

**Success criteria**:

- [ ] Returns array of same length as `times`
- [ ] Failed trials (end_region=None) are labeled "other"
- [ ] Labels use arrow notation: "start→end"
- [ ] Later trials overwrite earlier if overlapping
- [ ] Docstring documents overlap behavior

**Dependencies**: None (uses existing Trial class)

---

### Task 2.2: Add heading_direction_labels function

**File**: `src/neurospatial/behavioral.py`

**What to do**:

1. Add function with signature from PLAN.md Section 3.2
2. Implementation:
   - Validate inputs: require either (positions, times) or (speed, heading)
   - If speed/heading not provided:
     - Compute velocity: `velocity = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]`
     - Compute speed: `speed_computed = np.linalg.norm(velocity, axis=1)`
     - Compute heading: `heading_computed = np.arctan2(velocity[:, 1], velocity[:, 0])`
     - Pad first element: `speed = np.concatenate([[0], speed_computed])`
     - Pad first element: `heading = np.concatenate([[0], heading_computed])`
   - Validate speed and heading have same length
   - Create labels array
   - Compute bin edges: `edges = np.linspace(-np.pi, np.pi, n_directions + 1)`
   - For each timepoint:
     - If `speed[i] < min_speed`: label = "stationary"
     - Else: bin heading into sector, create label like "0–45°"
   - Return labels
3. Add NumPy-style docstring

**Success criteria**:

- [ ] Accepts either (positions, times) or (speed, heading)
- [ ] Raises ValueError if neither provided
- [ ] Precomputed values take precedence
- [ ] Labels are strings: "stationary", "0–45°", "45–90°", etc.
- [ ] Bin boundaries are correct (test at edges)
- [ ] First timepoint handled correctly when computing from positions

**Dependencies**: None

---

## Milestone 3: Directional Index Metric

Metric for comparing directional place fields.

### Task 3.1: Add directional_field_index function

**File**: `src/neurospatial/metrics/place_fields.py`

**What to do**:

1. Add function:

   ```python
   def directional_field_index(
       field_forward: NDArray[np.float64],
       field_reverse: NDArray[np.float64],
       *,
       eps: float = 1e-9,
   ) -> NDArray[np.float64]:
   ```

2. Implementation:
   - Validate shapes match
   - Compute: `index = (field_forward - field_reverse) / (field_forward + field_reverse + eps)`
   - Return index
3. Add NumPy-style docstring

**Success criteria**:

- [ ] Returns array of same shape as inputs
- [ ] Values in range [-1, 1]
- [ ] eps prevents division by zero
- [ ] NaN inputs produce NaN outputs
- [ ] No environment dependency

**Dependencies**: None

---

## Milestone 4: Public API Exports

Make new functions accessible from top-level imports.

### Task 4.1: Update spike_field exports

**File**: `src/neurospatial/__init__.py`

**What to do**:

1. Add to existing spike_field import line:

   ```python
   from neurospatial.spike_field import (
       compute_place_field,
       compute_directional_place_fields,
       DirectionalPlaceFields,
       spikes_to_field,
   )
   ```

2. Add to `__all__` list:
   - `"compute_directional_place_fields"`
   - `"DirectionalPlaceFields"`

**Success criteria**:

- [ ] `from neurospatial import compute_directional_place_fields` works
- [ ] `from neurospatial import DirectionalPlaceFields` works

**Dependencies**: Task 1.1, Task 1.3

---

### Task 4.2: Update behavioral exports

**File**: `src/neurospatial/__init__.py`

**What to do**:

1. Add to existing behavioral import:

   ```python
   from neurospatial.behavioral import (
       # ... existing imports ...
       goal_pair_direction_labels,
       heading_direction_labels,
   )
   ```

2. Add to `__all__` list:
   - `"goal_pair_direction_labels"`
   - `"heading_direction_labels"`

**Success criteria**:

- [ ] `from neurospatial import goal_pair_direction_labels` works
- [ ] `from neurospatial import heading_direction_labels` works

**Dependencies**: Task 2.1, Task 2.2

---

### Task 4.3: Update metrics exports

**File**: `src/neurospatial/metrics/__init__.py`

**What to do**:

1. Add import:

   ```python
   from neurospatial.metrics.place_fields import directional_field_index
   ```

2. Add to `__all__` list:
   - `"directional_field_index"`

**Success criteria**:

- [ ] `from neurospatial.metrics import directional_field_index` works

**Dependencies**: Task 3.1

---

## Milestone 5: Tests

Comprehensive test coverage for all new functionality.

### Task 5.1: Add tests for compute_directional_place_fields

**File**: `tests/test_directional_place_fields.py`

**What to do**:

1. Create new test file
2. Add fixtures for sample environment, trajectory, spikes
3. Add tests:
   - `test_constant_labels_equals_compute_place_field`: All same label → result matches compute_place_field
   - `test_two_directions_partition`: Split session in half with different labels → verify each field computed correctly
   - `test_other_label_excluded`: "other" labels not in result
   - `test_no_spikes`: Empty spike train → fields are all zero/NaN
   - `test_label_with_few_samples`: Label with <3 samples → handled gracefully
   - `test_nan_handling`: Verify NaN behavior matches compute_place_field
   - `test_result_structure`: DirectionalPlaceFields has correct fields and labels

**Success criteria**:

- [ ] All tests pass with `uv run pytest tests/test_directional_place_fields.py -v`
- [ ] Tests cover happy path and edge cases
- [ ] Tests are independent (no shared mutable state)

**Dependencies**: Task 1.3

---

### Task 5.2: Add tests for direction label helpers

**File**: `tests/test_direction_labels.py`

**What to do**:

1. Create new test file
2. Add tests for `goal_pair_direction_labels`:
   - `test_basic_trials`: Two trials → correct labels assigned
   - `test_outside_trials`: Timepoints outside trials → "other"
   - `test_failed_trial`: Trial with end_region=None → "other"
   - `test_overlapping_trials`: Later trial overwrites earlier
3. Add tests for `heading_direction_labels`:
   - `test_straight_path_x`: Movement in +x → single direction label
   - `test_stationary`: Low speed → "stationary"
   - `test_bin_boundaries`: Test heading at exactly 0°, 45°, 90°
   - `test_precomputed_matches_computed`: Same result from both input modes
   - `test_error_no_inputs`: Raises ValueError if no inputs provided
   - `test_n_directions`: Different n_directions values produce correct bins

**Success criteria**:

- [ ] All tests pass with `uv run pytest tests/test_direction_labels.py -v`
- [ ] Tests for both functions
- [ ] Edge cases covered

**Dependencies**: Task 2.1, Task 2.2

---

### Task 5.3: Add tests for directional_field_index

**File**: `tests/metrics/test_directional_index.py`

**What to do**:

1. Create new test file in tests/metrics/
2. Add tests:
   - `test_all_forward`: field_forward >> field_reverse → index ≈ +1
   - `test_all_reverse`: field_reverse >> field_forward → index ≈ -1
   - `test_equal_fields`: Equal fields → index ≈ 0
   - `test_nan_propagation`: NaN in input → NaN in output at that position
   - `test_eps_prevents_division_by_zero`: Both fields zero → finite result
   - `test_shape_preserved`: Output shape matches input shape

**Success criteria**:

- [ ] All tests pass with `uv run pytest tests/metrics/test_directional_index.py -v`
- [ ] Numerical edge cases covered

**Dependencies**: Task 3.1

---

## Milestone 6: Documentation

Example scripts and user guide updates.

### Task 6.1: Add example script

**File**: `docs/examples/20_directional_place_fields.py`

**What to do**:

1. Create example script with two sections:
   - **Section 1: Linear track with goal_pair_direction_labels**
     - Create simple 1D environment
     - Simulate trajectory with outbound/inbound runs
     - Segment into trials
     - Compute directional place fields
     - Plot forward vs reverse fields
   - **Section 2: Open field with heading_direction_labels**
     - Create 2D environment
     - Simulate trajectory with various headings
     - Compute heading-binned place fields
     - Plot fields for different heading sectors
2. Use matplotlib for visualization
3. Add comments explaining each step

**Success criteria**:

- [ ] Script runs without errors: `uv run python docs/examples/20_directional_place_fields.py`
- [ ] Produces interpretable visualizations
- [ ] Comments explain the workflow
- [ ] Follows existing example script patterns

**Dependencies**: All implementation tasks (1.1-4.3)

---

### Task 6.2: Update CLAUDE.md quick reference

**File**: `CLAUDE.md`

**What to do**:

1. Add to Quick Reference section under "Most Common Patterns":

   ```python
   # Compute directional place fields (v0.10.0+)
   from neurospatial import (
       compute_directional_place_fields,
       goal_pair_direction_labels,
       heading_direction_labels,
   )
   from neurospatial.metrics import directional_field_index

   # For trialized tasks (T-maze, Y-maze)
   trials = segment_trials(trajectory_bins, times, env, ...)
   labels = goal_pair_direction_labels(times, trials)
   result = compute_directional_place_fields(
       env, spike_times, times, positions, labels, bandwidth=5.0
   )
   forward_field = result.fields["home→goal"]
   reverse_field = result.fields["goal→home"]

   # For open fields (heading-based)
   labels = heading_direction_labels(positions, times, n_directions=8)
   result = compute_directional_place_fields(
       env, spike_times, times, positions, labels, bandwidth=5.0
   )

   # Compare directionality
   index = directional_field_index(forward_field, reverse_field)
   ```

**Success criteria**:

- [ ] Examples are correct and follow existing CLAUDE.md patterns
- [ ] Version number updated appropriately

**Dependencies**: All implementation tasks

---

## Verification Checklist

Run after all tasks complete:

```bash
# 1. All tests pass
uv run pytest tests/test_directional_place_fields.py tests/test_direction_labels.py tests/metrics/test_directional_index.py -v

# 2. Linting passes
uv run ruff check src/neurospatial/spike_field.py src/neurospatial/behavioral.py src/neurospatial/metrics/place_fields.py

# 3. Type checking passes
uv run mypy src/neurospatial/spike_field.py src/neurospatial/behavioral.py src/neurospatial/metrics/place_fields.py

# 4. Example script runs
uv run python docs/examples/20_directional_place_fields.py

# 5. Imports work
uv run python -c "from neurospatial import compute_directional_place_fields, DirectionalPlaceFields, goal_pair_direction_labels, heading_direction_labels; from neurospatial.metrics import directional_field_index; print('All imports successful')"
```

---

## Task Summary

| Milestone | Task | Status |
|-----------|------|--------|
| 1. Core | 1.1 DirectionalPlaceFields dataclass | [x] |
| 1. Core | 1.2 _subset_spikes_by_time_mask helper | [x] |
| 1. Core | 1.3 compute_directional_place_fields | [ ] |
| 2. Labels | 2.1 goal_pair_direction_labels | [ ] |
| 2. Labels | 2.2 heading_direction_labels | [ ] |
| 3. Metric | 3.1 directional_field_index | [ ] |
| 4. Exports | 4.1 spike_field exports | [ ] |
| 4. Exports | 4.2 behavioral exports | [ ] |
| 4. Exports | 4.3 metrics exports | [ ] |
| 5. Tests | 5.1 directional_place_fields tests | [ ] |
| 5. Tests | 5.2 direction_labels tests | [ ] |
| 5. Tests | 5.3 directional_index tests | [ ] |
| 6. Docs | 6.1 Example script | [ ] |
| 6. Docs | 6.2 CLAUDE.md update | [ ] |
