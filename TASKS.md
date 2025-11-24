# TASKS.md - NWB Integration Implementation

**Status**: In Progress
**Plan Reference**: [PLAN.md](PLAN.md)
**Command**: `/project:freshstart`

---

## Overview

This checklist implements the NWB Integration Plan for neurospatial. Each milestone follows TDD workflow:

1. Write tests first
2. Verify tests fail
3. Implement feature
4. Verify tests pass
5. Code review and refactor

---

## Milestone 0: Project Setup

- [x] **M0.1** Create module structure
  - [x] Create `src/neurospatial/nwb/` directory
  - [x] Create `src/neurospatial/nwb/__init__.py` with public API exports
  - [x] Create `src/neurospatial/nwb/_core.py` for discovery utilities
  - [x] Create placeholder files for each submodule

- [x] **M0.2** Add optional dependencies
  - [x] Add `nwb` extra to `pyproject.toml` (pynwb>=2.5.0, hdmf>=3.0.0)
  - [x] Add `nwb-pose` extra (pynwb, ndx-pose>=0.2.0)
  - [x] Add `nwb-events` extra (pynwb, ndx-events>=0.4.0)
  - [x] Add `nwb-full` extra combining all NWB deps
  - [x] Run `uv sync` to verify dependencies resolve

- [x] **M0.3** Create test infrastructure
  - [x] Create `tests/nwb/` directory
  - [x] Create `tests/nwb/__init__.py`
  - [x] Create `tests/nwb/conftest.py` with NWB fixtures
  - [x] Implement `sample_nwb_with_position` fixture
  - [x] Implement `sample_nwb_with_pose` fixture (requires ndx-pose)
  - [x] Implement `sample_nwb_with_events` fixture (requires ndx-events)
  - [x] Implement `create_empty_nwb` helper

---

## Milestone 1: Core NWB Reading (Phase 1)

### M1.1 Position/Trajectory Reading

- [x] **M1.1.1** Core discovery utilities
  - [x] Write tests for `_find_containers_by_type()` in `tests/nwb/test_core.py`
  - [x] Implement `_find_containers_by_type()` in `_core.py`
  - [x] Test search order: processing/behavior > processing/* > acquisition
  - [x] Verify logging behavior for auto-discovery

- [x] **M1.1.2** Position reading function
  - [x] Write tests for `read_position()` in `tests/nwb/test_behavior.py`
    - [x] Test basic position reading
    - [x] Test with explicit processing_module parameter
    - [x] Test with explicit position_name parameter
    - [x] Test error when no Position found (KeyError)
    - [x] Test error when named Position not found (KeyError with available list)
    - [x] Test multiple SpatialSeries selection (alphabetical, INFO log)
  - [x] Implement `read_position()` in `_behavior.py`
  - [x] Return tuple: (positions: NDArray, timestamps: NDArray)
  - [x] Verify shapes: positions (n_samples, n_dims), timestamps (n_samples,)

- [x] **M1.1.3** Head direction reading function
  - [x] Write tests for `read_head_direction()` in `tests/nwb/test_behavior.py`
    - [x] Test basic CompassDirection reading
    - [x] Test error when no CompassDirection found
  - [x] Implement `read_head_direction()` in `_behavior.py`
  - [x] Return tuple: (angles: NDArray, timestamps: NDArray)

### M1.2 Pose/Skeleton Reading

- [x] **M1.2.1** Pose reading function
  - [x] Write tests for `read_pose()` in `tests/nwb/test_pose.py`
    - [x] Test basic PoseEstimation reading
    - [x] Test with explicit pose_estimation_name parameter
    - [x] Test error when no PoseEstimation found (KeyError)
    - [x] Test error when named pose not found (KeyError with available list)
    - [x] Test ImportError when ndx-pose not installed
    - [x] Test Skeleton extraction from PoseEstimation
    - [x] Test multiple bodyparts conversion to dict
  - [x] Implement `read_pose()` in `_pose.py`
  - [x] Return tuple: (bodyparts: dict[str, NDArray], timestamps: NDArray, skeleton: Skeleton)
  - [x] Verify lazy import pattern for ndx-pose

- [x] **M1.2.2** Skeleton integration
  - [x] Verify existing `Skeleton.from_ndx_pose()` works with read data
  - [x] Add tests for round-trip: ndx-pose Skeleton → neurospatial Skeleton

### M1.3 Events Reading

- [x] **M1.3.1** Events reading function (ndx-events EventsTable)
  - [x] Write tests for `read_events()` in `tests/nwb/test_events.py`
    - [x] Test basic EventsTable reading
    - [x] Test with explicit processing_module parameter
    - [x] Test error when EventsTable not found (KeyError)
    - [x] Test ImportError when ndx-events not installed
    - [x] Test DataFrame output with timestamp column
    - [x] Test additional columns preserved
  - [x] Implement `read_events()` in `_events.py`
  - [x] Return DataFrame with timestamps and event columns
  - [x] Verify lazy import pattern for ndx-events

- [x] **M1.3.2** TimeIntervals reading function (built-in NWB)
  - [x] Write tests for `read_intervals()` in `tests/nwb/test_events.py`
    - [x] Test reading trials table
    - [x] Test reading epochs table
    - [x] Test reading custom intervals
    - [x] Test start_time/stop_time columns in output
    - [x] Test error when interval not found (KeyError)
    - [x] Test additional columns preserved
  - [x] Implement `read_intervals()` in `_events.py`
  - [x] Return DataFrame with start_time, stop_time, and event columns
  - [x] No extension dependency (uses pynwb built-in TimeIntervals)

### M1.4 Overlay Factory Functions

- [x] **M1.4.1** Position overlay factory
  - [x] Write tests for `position_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [x] Test PositionOverlay creation from Position
    - [x] Test color, trail_length parameters passed through
    - [x] Test times parameter populated from timestamps
  - [x] Implement `position_overlay_from_nwb()` in `_overlays.py`
  - [x] Return PositionOverlay with data and times

- [x] **M1.4.2** Bodypart overlay factory
  - [x] Write tests for `bodypart_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [x] Test BodypartOverlay creation from PoseEstimation
    - [x] Test skeleton parameter auto-extraction
    - [x] Test colors, skeleton_color parameters passed through
  - [x] Implement `bodypart_overlay_from_nwb()` in `_overlays.py`
  - [x] Return BodypartOverlay with bodypart dict and skeleton

- [x] **M1.4.3** Head direction overlay factory
  - [x] Write tests for `head_direction_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [x] Test HeadDirectionOverlay creation from CompassDirection
    - [x] Test color, length parameters passed through
  - [x] Implement `head_direction_overlay_from_nwb()` in `_overlays.py`
  - [x] Return HeadDirectionOverlay with angles and times

- [x] **M1.4.4** Environment from position factory
  - [x] Write tests for `environment_from_position()` in `tests/nwb/test_environment.py`
    - [x] Test Environment creation from Position with bin_size
    - [x] Test units parameter propagation
    - [x] Test infer_active_bins parameter
    - [x] Test kwargs forwarded to Environment.from_samples()
    - [x] Test error when Position not found
  - [x] Implement `environment_from_position()` in `_environment.py`
  - [x] Return Environment.from_samples() result

---

## Milestone 2: NWB Writing (Phase 2)

### M2.1 Writing Spatial Fields

- [x] **M2.1.1** Place field writing
  - [x] Write tests for `write_place_field()` in `tests/nwb/test_fields.py`
    - [x] Test basic field writing to analysis/
    - [x] Test field metadata (description, units)
    - [x] Test shape validation (field.shape[0] == env.n_bins)
    - [x] Test error on duplicate name without overwrite=True
    - [x] Test overwrite=True replaces existing
    - [x] Test bin_centers reference stored
  - [x] Implement `write_place_field()` in `_fields.py`
  - [x] Create analysis/ processing module if needed
  - [x] Store as TimeSeries with bin_centers reference

- [x] **M2.1.2** Occupancy map writing
  - [x] Write tests for `write_occupancy()` in `tests/nwb/test_fields.py`
    - [x] Test basic occupancy writing to analysis/
    - [x] Test normalization options (counts vs. probability)
  - [x] Implement `write_occupancy()` in `_fields.py`
  - [x] Store with appropriate units metadata

### M2.2 Writing Events

- [x] **M2.2.1** Lap events writing
  - [x] Write tests for `write_laps()` in `tests/nwb/test_events.py`
    - [x] Test basic lap times writing
    - [x] Test with lap_types (direction) column
    - [x] Test EventsTable creation in processing/behavior/
    - [x] Test ImportError when ndx-events not installed
  - [x] Implement `write_laps()` in `_events.py`
  - [x] Create processing/behavior module if needed
  - [x] Add EventsTable with timestamp and optional columns

- [x] **M2.2.2** Region crossings writing
  - [x] Write tests for `write_region_crossings()` in `tests/nwb/test_events.py`
    - [x] Test basic crossing times writing
    - [x] Test region name column
    - [x] Test event_type column (enter/exit)
  - [x] Implement `write_region_crossings()` in `_events.py`
  - [x] Add EventsTable with region and event_type columns

---

## Milestone 3: Environment Round-trip (Phase 3)

### M3.1 Environment to Scratch

- [x] **M3.1.1** Environment writing function
  - [x] Write tests for `write_environment()` in `tests/nwb/test_environment.py`
    - [x] Test basic Environment writing to scratch/
    - [x] Test bin_centers dataset (n_bins, n_dims)
    - [x] Test edges dataset as edge list (n_edges, 2)
    - [x] Test edge_weights dataset (optional)
    - [x] Test dimension_ranges dataset (n_dims, 2)
    - [x] Test group attributes (units, frame, n_dims, layout_type)
    - [x] Test regions DynamicTable creation
    - [x] Test point regions stored correctly
    - [x] Test polygon regions stored correctly (ragged vertices)
    - [x] Test metadata.json for extra attributes
  - [x] Implement `write_environment()` in `_environment.py`
  - [x] Create scratch group structure per spec
  - [x] Handle edge extraction from nx.Graph

- [x] **M3.1.2** Regions to DynamicTable
  - [x] Write tests for regions DynamicTable conversion
    - [x] Test point region serialization
    - [x] Test polygon region serialization
    - [x] Test multiple regions in single table
  - [x] Implement `_regions_to_json()` helper (JSON approach instead of ragged arrays)
  - [x] Handle polygon vertices via JSON serialization

### M3.2 Environment from Scratch

- [x] **M3.2.1** Environment reading function
  - [x] Write tests for `read_environment()` in `tests/nwb/test_environment.py`
    - [x] Test basic Environment reading from scratch/
    - [x] Test bin_centers reconstruction
    - [x] Test connectivity graph reconstruction from edge list
    - [x] Test edge weights applied to graph
    - [x] Test dimension_ranges reconstruction
    - [x] Test units/frame attributes restored
    - [x] Test regions DynamicTable reading
    - [x] Test point regions restored
    - [x] Test polygon regions restored
    - [x] Test error when environment not found (KeyError)
  - [x] Implement `read_environment()` in `_environment.py`
  - [x] Reconstruct nx.Graph from edge list
  - [x] Restore Environment with all attributes

- [x] **M3.2.2** DynamicTable to Regions
  - [x] Write tests for DynamicTable → Regions conversion
    - [x] Test point region deserialization
    - [x] Test polygon region deserialization
    - [x] Test empty regions table handling
  - [x] Implement `_json_to_regions()` helper (JSON approach instead of DynamicTable ragged arrays)

### M3.3 Environment Class Methods

- [x] **M3.3.1** Environment.from_nwb() classmethod
  - [x] Write tests for `Environment.from_nwb()` in `tests/nwb/test_environment.py`
    - [x] Test loading from scratch (scratch_name parameter)
    - [x] Test creating from position (bin_size parameter)
    - [x] Test error when neither parameter provided
    - [x] Test kwargs forwarded to environment_from_position
  - [x] Implement `Environment.from_nwb()` in environment/factories.py
  - [x] Lazy import neurospatial.nwb functions

- [x] **M3.3.2** Environment.to_nwb() method
  - [x] Write tests for `Environment.to_nwb()` in `tests/nwb/test_environment.py`
    - [x] Test writing to NWB file
    - [x] Test custom name parameter
  - [x] Implement `Environment.to_nwb()` in environment/serialization.py
  - [x] Lazy import neurospatial.nwb functions

### M3.4 Round-trip Integration Tests

- [x] **M3.4.1** Full round-trip tests (implemented in test_environment.py)
  - [x] Write integration tests in `tests/nwb/test_environment.py` (TestEnvironmentRoundTrip + TestAllLayoutsRoundTrip)
    - [x] Test Environment survives NWB write/read cycle
    - [x] Test bin_centers exactly preserved
    - [x] Test connectivity graph structure preserved
    - [x] Test edge weights preserved
    - [x] Test regions preserved (points and polygons)
    - [x] Test units and frame metadata preserved
  - [x] Test with different layout types (grid, polygon, masked) - 8 layout types tested
  - [x] Test with 1D and 2D environments - Graph and 3D layouts tested

---

## Milestone 4: Documentation and Polish

- [x] **M4.1** Update CLAUDE.md
  - [x] Add NWB integration section to Quick Reference
  - [x] Add NWB-specific import patterns
  - [x] Add NWB troubleshooting section
  - [x] Document optional dependency installation

- [x] **M4.2** Add docstrings
  - [x] Add NumPy docstrings to all public functions
  - [x] Add usage examples to module docstrings
  - [x] Verify doctests pass

- [x] **M4.3** Type annotations
  - [x] Add complete type annotations to all functions
  - [x] Run mypy and fix any errors
  - [x] Verify py.typed marker works with NWB types

- [ ] **M4.4** Final verification
  - [ ] Run full test suite (`uv run pytest`)
  - [ ] Run linter (`uv run ruff check .`)
  - [ ] Run formatter (`uv run ruff format .`)
  - [ ] Run type checker (`uv run mypy src/neurospatial/nwb/`)
  - [ ] Verify all tests pass

---

## Notes

- Phase 4 (ndx-spatial-environment extension) is out of scope - lives in separate repo
- All NWB dependencies are optional via lazy imports
- Follow TDD strictly: test first, then implement
- Update SCRATCHPAD.md with decisions and blockers
- Commit frequently with conventional commit messages

---

## Progress Tracking

| Milestone | Status | Started | Completed |
|-----------|--------|---------|-----------|
| M0: Setup | Complete | 2025-11-23 | 2025-11-23 |
| M1: Reading | Complete | 2025-11-23 | 2025-11-23 |
| M2: Writing | Complete | 2025-11-23 | 2025-11-23 |
| M3: Round-trip | Complete | 2025-11-23 | 2025-11-24 |
| M4: Docs | Not Started | - | - |
