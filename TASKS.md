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

- [x] **M1.3.1** Events reading function
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

### M1.4 Overlay Factory Functions

- [ ] **M1.4.1** Position overlay factory
  - [ ] Write tests for `position_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [ ] Test PositionOverlay creation from Position
    - [ ] Test color, trail_length parameters passed through
    - [ ] Test times parameter populated from timestamps
  - [ ] Implement `position_overlay_from_nwb()` in `_overlays.py`
  - [ ] Return PositionOverlay with data and times

- [ ] **M1.4.2** Bodypart overlay factory
  - [ ] Write tests for `bodypart_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [ ] Test BodypartOverlay creation from PoseEstimation
    - [ ] Test skeleton parameter auto-extraction
    - [ ] Test colors, skeleton_color parameters passed through
  - [ ] Implement `bodypart_overlay_from_nwb()` in `_overlays.py`
  - [ ] Return BodypartOverlay with bodypart dict and skeleton

- [ ] **M1.4.3** Head direction overlay factory
  - [ ] Write tests for `head_direction_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
    - [ ] Test HeadDirectionOverlay creation from CompassDirection
    - [ ] Test color, length parameters passed through
  - [ ] Implement `head_direction_overlay_from_nwb()` in `_overlays.py`
  - [ ] Return HeadDirectionOverlay with angles and times

- [ ] **M1.4.4** Environment from position factory
  - [ ] Write tests for `environment_from_position()` in `tests/nwb/test_overlays.py`
    - [ ] Test Environment creation from Position with bin_size
    - [ ] Test units parameter propagation
    - [ ] Test infer_active_bins parameter
  - [ ] Implement `environment_from_position()` in `_overlays.py`
  - [ ] Return Environment.from_samples() result

---

## Milestone 2: NWB Writing (Phase 2)

### M2.1 Writing Spatial Fields

- [ ] **M2.1.1** Place field writing
  - [ ] Write tests for `write_place_field()` in `tests/nwb/test_fields.py`
    - [ ] Test basic field writing to analysis/
    - [ ] Test field metadata (description, units)
    - [ ] Test shape validation (field.shape[0] == env.n_bins)
    - [ ] Test error on duplicate name without overwrite=True
    - [ ] Test overwrite=True replaces existing
    - [ ] Test bin_centers reference stored
  - [ ] Implement `write_place_field()` in `_fields.py`
  - [ ] Create analysis/ processing module if needed
  - [ ] Store as TimeSeries with bin_centers reference

- [ ] **M2.1.2** Occupancy map writing
  - [ ] Write tests for `write_occupancy()` in `tests/nwb/test_fields.py`
    - [ ] Test basic occupancy writing to analysis/
    - [ ] Test normalization options (counts vs. probability)
  - [ ] Implement `write_occupancy()` in `_fields.py`
  - [ ] Store with appropriate units metadata

### M2.2 Writing Events

- [ ] **M2.2.1** Lap events writing
  - [ ] Write tests for `write_laps()` in `tests/nwb/test_events.py`
    - [ ] Test basic lap times writing
    - [ ] Test with lap_types (direction) column
    - [ ] Test EventsTable creation in processing/behavior/
    - [ ] Test ImportError when ndx-events not installed
  - [ ] Implement `write_laps()` in `_events.py`
  - [ ] Create processing/behavior module if needed
  - [ ] Add EventsTable with timestamp and optional columns

- [ ] **M2.2.2** Region crossings writing
  - [ ] Write tests for `write_region_crossings()` in `tests/nwb/test_events.py`
    - [ ] Test basic crossing times writing
    - [ ] Test region name column
    - [ ] Test event_type column (enter/exit)
  - [ ] Implement `write_region_crossings()` in `_events.py`
  - [ ] Add EventsTable with region and event_type columns

---

## Milestone 3: Environment Round-trip (Phase 3)

### M3.1 Environment to Scratch

- [ ] **M3.1.1** Environment writing function
  - [ ] Write tests for `write_environment()` in `tests/nwb/test_environment.py`
    - [ ] Test basic Environment writing to scratch/
    - [ ] Test bin_centers dataset (n_bins, n_dims)
    - [ ] Test edges dataset as edge list (n_edges, 2)
    - [ ] Test edge_weights dataset (optional)
    - [ ] Test dimension_ranges dataset (n_dims, 2)
    - [ ] Test group attributes (units, frame, n_dims, layout_type)
    - [ ] Test regions DynamicTable creation
    - [ ] Test point regions stored correctly
    - [ ] Test polygon regions stored correctly (ragged vertices)
    - [ ] Test metadata.json for extra attributes
  - [ ] Implement `write_environment()` in `_environment.py`
  - [ ] Create scratch group structure per spec
  - [ ] Handle edge extraction from nx.Graph

- [ ] **M3.1.2** Regions to DynamicTable
  - [ ] Write tests for regions DynamicTable conversion
    - [ ] Test point region serialization
    - [ ] Test polygon region serialization
    - [ ] Test multiple regions in single table
  - [ ] Implement `_regions_to_dynamic_table()` helper
  - [ ] Handle ragged arrays for polygon vertices

### M3.2 Environment from Scratch

- [ ] **M3.2.1** Environment reading function
  - [ ] Write tests for `read_environment()` in `tests/nwb/test_environment.py`
    - [ ] Test basic Environment reading from scratch/
    - [ ] Test bin_centers reconstruction
    - [ ] Test connectivity graph reconstruction from edge list
    - [ ] Test edge weights applied to graph
    - [ ] Test dimension_ranges reconstruction
    - [ ] Test units/frame attributes restored
    - [ ] Test regions DynamicTable reading
    - [ ] Test point regions restored
    - [ ] Test polygon regions restored
    - [ ] Test error when environment not found (KeyError)
  - [ ] Implement `read_environment()` in `_environment.py`
  - [ ] Reconstruct nx.Graph from edge list
  - [ ] Restore Environment with all attributes

- [ ] **M3.2.2** DynamicTable to Regions
  - [ ] Write tests for DynamicTable → Regions conversion
    - [ ] Test point region deserialization
    - [ ] Test polygon region deserialization
    - [ ] Test empty regions table handling
  - [ ] Implement `_dynamic_table_to_regions()` helper

### M3.3 Environment Class Methods

- [ ] **M3.3.1** Environment.from_nwb() classmethod
  - [ ] Write tests for `Environment.from_nwb()` in `tests/nwb/test_environment.py`
    - [ ] Test loading from scratch (scratch_name parameter)
    - [ ] Test creating from position (bin_size parameter)
    - [ ] Test error when neither parameter provided
    - [ ] Test kwargs forwarded to environment_from_position
  - [ ] Implement `Environment.from_nwb()` in environment/factories.py
  - [ ] Lazy import neurospatial.nwb functions

- [ ] **M3.3.2** Environment.to_nwb() method
  - [ ] Write tests for `Environment.to_nwb()` in `tests/nwb/test_environment.py`
    - [ ] Test writing to NWB file
    - [ ] Test custom name parameter
  - [ ] Implement `Environment.to_nwb()` in environment/serialization.py
  - [ ] Lazy import neurospatial.nwb functions

### M3.4 Round-trip Integration Tests

- [ ] **M3.4.1** Full round-trip tests
  - [ ] Write integration tests in `tests/nwb/test_roundtrip.py`
    - [ ] Test Environment survives NWB write/read cycle
    - [ ] Test bin_centers exactly preserved
    - [ ] Test connectivity graph structure preserved
    - [ ] Test edge weights preserved
    - [ ] Test regions preserved (points and polygons)
    - [ ] Test units and frame metadata preserved
  - [ ] Test with different layout types (grid, polygon, masked)
  - [ ] Test with 1D and 2D environments

---

## Milestone 4: Documentation and Polish

- [ ] **M4.1** Update CLAUDE.md
  - [ ] Add NWB integration section to Quick Reference
  - [ ] Add NWB-specific import patterns
  - [ ] Add NWB troubleshooting section
  - [ ] Document optional dependency installation

- [ ] **M4.2** Add docstrings
  - [ ] Add NumPy docstrings to all public functions
  - [ ] Add usage examples to module docstrings
  - [ ] Verify doctests pass

- [ ] **M4.3** Type annotations
  - [ ] Add complete type annotations to all functions
  - [ ] Run mypy and fix any errors
  - [ ] Verify py.typed marker works with NWB types

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
| M1: Reading | In Progress | 2025-11-23 | - |
| M2: Writing | Not Started | - | - |
| M3: Round-trip | Not Started | - | - |
| M4: Docs | Not Started | - | - |
