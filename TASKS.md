# Tasks: Start/End Region for Trial Segmentation

Reference: [PLAN.md](PLAN.md) for detailed specifications.

---

## Milestone 1: Update Segmentation Module

**Goal**: Update `Trial` dataclass and `segment_trials()` to include `start_region` and rename `outcome` to `end_region`.

### Tasks

- [x] **1.1** Update `Trial` dataclass in `src/neurospatial/segmentation/trials.py`
  - Add `start_region: str` field after `end_time`
  - Rename `outcome: str | None` to `end_region: str | None`
  - Update docstring

- [x] **1.2** Update `segment_trials()` in `src/neurospatial/segmentation/trials.py`
  - Change all `outcome=` to `end_region=` in `Trial()` calls
  - Add `start_region=start_region` to all `Trial()` calls
  - Update docstring Returns section

- [x] **1.3** Update tests in `tests/segmentation/test_trials.py`
  - Replace `.outcome` with `.end_region` in all assertions
  - Add assertions for `.start_region` field
  - Verify field values match expected regions

- [x] **1.4** Update integration tests in `tests/segmentation/test_integration.py`
  - Update any `Trial` field references

### Verify

```bash
uv run pytest tests/segmentation/test_trials.py -v
uv run pytest tests/segmentation/test_integration.py -v
```

---

## Milestone 2: Add `write_trials()` and `read_trials()`

**Goal**: Add NWB functions to write/read trial data with start/end regions.

### Tasks

- [x] **2.1** Add `write_trials()` in `src/neurospatial/nwb/_events.py`
  - Implement function signature per PLAN.md Section 3
  - Handle `list[Trial]` input (extract fields)
  - Handle raw array input (validate lengths)
  - Raise `ValueError` if both provided
  - Raise `ValueError` if required arrays missing
  - Implement overwrite logic (clear existing trials)
  - Use `nwbfile.add_trial_column()` and `nwbfile.add_trial()`

- [x] **2.2** Add `read_trials()` in `src/neurospatial/nwb/_events.py`
  - Implement as wrapper around `read_intervals("trials")`
  - Add docstring

- [x] **2.3** Create `tests/nwb/test_trials.py`
  - `test_write_trials_from_trial_objects`
  - `test_write_trials_from_arrays`
  - `test_write_trials_mixed_args_error`
  - `test_write_trials_missing_required_error`
  - `test_write_trials_length_mismatch_error`
  - `test_write_trials_overwrite_false_error`
  - `test_write_trials_overwrite_true`
  - `test_write_trials_roundtrip`
  - `test_read_trials_not_found`
  - `test_read_trials_with_custom_columns`

### Verify

```bash
uv run pytest tests/nwb/test_trials.py -v
```

---

## Milestone 3: Extend `write_laps()`

**Goal**: Add optional `start_regions`, `end_regions`, `stop_times` parameters to `write_laps()`.

### Tasks

- [ ] **3.1** Extend `write_laps()` in `src/neurospatial/nwb/_events.py`
  - Add parameters: `start_regions`, `end_regions`, `stop_times`
  - Validate lengths match `lap_times`
  - Add columns to EventsTable when provided
  - Maintain backwards compatibility (all new params optional)

- [ ] **3.2** Add tests in `tests/nwb/test_events.py`
  - `test_write_laps_with_start_regions`
  - `test_write_laps_with_end_regions`
  - `test_write_laps_with_stop_times`
  - `test_write_laps_with_all_optional`
  - `test_write_laps_region_length_mismatch`

### Verify

```bash
uv run pytest tests/nwb/test_events.py -v
```

---

## Milestone 4: Update Exports

**Goal**: Export new functions from `neurospatial.nwb`.

### Tasks

- [x] **4.1** Update `src/neurospatial/nwb/__init__.py`
  - Add `write_trials` and `read_trials` to `_LAZY_IMPORTS`
  - Add to `__all__`
  - Update module docstring (Reading/Writing Functions lists)

### Verify

```bash
uv run python -c "from neurospatial.nwb import write_trials, read_trials; print('OK')"
```

---

## Milestone 5: Documentation

**Goal**: Update all documentation to reflect changes.

### Tasks

- [ ] **5.1** Update `CLAUDE.md`
  - Add trial segmentation example to Quick Reference
  - Add `write_trials`/`read_trials` to NWB Integration section
  - Update NWB Data Locations table
  - Add to Import Patterns section

- [ ] **5.2** Update `src/neurospatial/segmentation/__init__.py` docstring
  - Reflect `Trial` field changes if referenced

### Verify

```bash
uv run pytest --doctest-modules src/neurospatial/segmentation/trials.py
uv run pytest --doctest-modules src/neurospatial/nwb/_events.py
```

---

## Milestone 6: Final Verification

**Goal**: Run full test suite and verify no regressions.

### Tasks

- [ ] **6.1** Run full test suite

  ```bash
  uv run pytest
  ```

- [ ] **6.2** Run type checking

  ```bash
  uv run mypy src/neurospatial/segmentation/trials.py
  uv run mypy src/neurospatial/nwb/_events.py
  ```

- [ ] **6.3** Run linting

  ```bash
  uv run ruff check src/neurospatial/segmentation/trials.py src/neurospatial/nwb/_events.py
  uv run ruff format src/neurospatial/segmentation/trials.py src/neurospatial/nwb/_events.py
  ```

---

## Dependency Graph

```
M1 (Segmentation) ─────┬─────> M4 (Exports) ────> M5 (Docs) ────> M6 (Final)
                       │
M2 (write/read_trials) ┘
                       │
M3 (write_laps)  ──────┘
```

- M1 must complete first (Trial dataclass used by M2)
- M2 and M3 can run in parallel after M1
- M4 depends on M2 and M3
- M5 depends on M4
- M6 is final verification

---

## Breaking Change Notice

`Trial.outcome` renamed to `Trial.end_region`. Update downstream code:

```python
# Before
trial.outcome

# After
trial.end_region
```
