# SCRATCHPAD.md - NWB Integration Notes

**Last Updated**: 2025-11-24

---

## Current Status

- **Active Task**: Code review fixes complete
- **Blocker**: None
- **Next Action**: M4.1 - Update CLAUDE.md with NWB integration documentation

---

## Session Log

### 2025-11-24: NWB Module Code Review Fixes

Based on comprehensive code review feedback, implemented the following fixes:

**High Priority Fixes:**

1. **ndx-events import order** (`_events.py`) - Changed import pattern to call `_require_ndx_events()` BEFORE importing `EventsTable` to ensure users see friendly error messages. Applied to `read_events()`, `write_laps()`, `write_region_crossings()`.

**Medium Priority Fixes:**

1. **Schema versioning** (`_environment.py`) - Added `schema_version: "1.0"` to metadata JSON for future format migrations. Added version check in `read_environment()` with warning for unknown versions.
2. **2D place field time axis semantics** (`_fields.py`) - Updated docstring to document that 2D time-varying fields use abstract indices `[0, 1, 2, ...]` by default.
3. **Optional timestamps parameter** (`_fields.py`) - Added `timestamps: NDArray[np.float64] | None = None` parameter to `write_place_field()` for physical timestamps on time-varying fields. Includes validation for 1D shape and length matching.

**Low Priority Fixes:**

1. **Logger imports** - Centralized logger imports from `_core.py` in `_fields.py` and `_environment.py` instead of local definitions.
2. **bin_sizes docstring** (`_environment.py`) - Clarified that `_ReconstructedLayout.bin_sizes()` returns volume estimates (spacing^n_dims), not linear sizes.
3. **Type annotation** (`_environment.py`) - Fixed `point_to_bin_index()` return type from `np.int_` to `np.intp` for consistency.
4. **is_1d discrepancy documentation** (`_environment.py`) - Added Notes section to `_ReconstructedLayout` docstring explaining that `env.layout.is_1d` may differ from `env.is_1d` for reconstructed 1D layouts.

**New Tests Added:**

- `test_time_varying_field_with_physical_timestamps` - Verifies physical timestamps stored correctly
- `test_time_varying_field_without_timestamps_uses_indices` - Verifies default behavior
- `test_static_field_ignores_timestamps_parameter` - Verifies 1D fields ignore timestamps
- `test_timestamps_length_mismatch_raises_error` - Verifies validation
- `test_timestamps_must_be_1d` - Verifies 2D timestamps rejected

**Results:**

- 359 tests pass (354 original + 5 new)
- ruff check clean
- mypy passes

### 2025-11-23: M3.2.1 Complete - Environment Reading Function

- Created 15 tests for `read_environment()` in `tests/nwb/test_environment.py`
- Tests cover:
  - Basic Environment reading from scratch/
  - bin_centers reconstruction (exact match)
  - connectivity graph reconstruction from edge list
  - edge weights (distances) applied to graph
  - dimension_ranges reconstruction
  - units/frame attributes restoration
  - Point regions restored correctly
  - Polygon regions restored correctly
  - Error handling: environment not found (KeyError)
  - Custom name parameter
  - Environment is fitted and ready to use
  - Graph node attributes (pos) match bin_centers
  - Empty regions handled correctly
  - Name attribute restored from metadata
- Implemented `read_environment()` in `_environment.py` with helper functions:
  - `_reconstruct_graph()` - rebuilds connectivity graph from edge list with all required node/edge attributes
  - `_json_to_regions()` - deserializes regions from JSON with warnings for skipped regions
  - `_estimate_bin_size()` - estimates bin size from bin_centers using median nearest-neighbor distance
- Code review: APPROVED after addressing feedback
  - Added proper imports for `NDArray`, `nx`, `warnings`
  - Added `Regions` to TYPE_CHECKING imports
  - Added warnings for skipped regions with None data
  - Added Notes section to docstring documenting limitations
  - Added Raises section to `_reconstruct_graph()` docstring
- All 51 tests pass (36 write + 15 read), ruff check clean, mypy passes
- **Round-trip capability now functional**: Environment can be written to and read from NWB

### 2025-11-23: M3.1.2 Complete - Regions to DynamicTable (JSON approach)

- Marked complete as regions are already serialized via `_regions_to_json()` helper
- JSON approach chosen over ragged arrays for simplicity
- Point and polygon regions both supported

### 2025-11-23: M3.1.1 Complete - Environment Writing Function

- Created 21 tests for `write_environment()` in `tests/nwb/test_environment.py`
- Tests cover:
  - Basic Environment writing to scratch/
  - bin_centers dataset with correct shape (n_bins, n_dims)
  - edges dataset as edge list (n_edges, 2)
  - edge_weights dataset (n_edges,)
  - dimension_ranges dataset (n_dims, 2)
  - Group attributes (units, frame, n_dims, layout_type)
  - Regions stored as JSON in DynamicTable column
  - Point regions stored correctly
  - Polygon regions stored correctly
  - metadata.json for extra attributes (including n_edges for deserialization)
  - Default name ("spatial_environment")
  - Custom name parameter
  - Duplicate name error handling
  - overwrite=True replaces existing
  - Data integrity verification for bin_centers and edges
  - Error on unfitted environment (RuntimeError)
  - Alternative 2D environment test
- Implemented `write_environment()` in `_environment.py`:
  - Uses DynamicTable in scratch space (required by HDMF for uniform row counts)
  - Arrays are padded to uniform row count, with actual counts stored in metadata JSON
  - Extracts edge list and weights from connectivity graph
  - Serializes regions to JSON via `_regions_to_json()` helper
  - Stores metadata JSON with n_bins, n_edges, n_dims for proper deserialization
  - Validates environment is fitted before writing
  - Comprehensive NumPy-style docstring with Notes section explaining padding
- Code review: Addressed critical feedback
  - Added fitted state validation (RuntimeError if not fitted)
  - Added n_edges to metadata JSON for deserialization
  - Updated docstring to clarify padded storage format
  - Added test for unfitted environment
  - Fixed misleading test name (test_1d_environment → test_alternative_2d_environment)
- All 21 tests pass, ruff check clean, mypy passes
- **Milestone M3: Round-trip is now in progress**

### 2025-11-23: M2.2.2 Complete - Region Crossings Writing

- Created 18 tests for `write_region_crossings()` in `tests/nwb/test_events.py`
- Tests cover:
  - Basic crossing times writing to processing/behavior/ module
  - Region name column (string values)
  - Event type column (enter/exit/arbitrary strings)
  - EventsTable creation and verification
  - Custom description and name parameters
  - Default description verification
  - Duplicate name error handling
  - overwrite=True replaces existing table
  - Empty crossing times array handling
  - Shape validation (1D only, 2D rejected)
  - region_names length mismatch validation
  - event_types length mismatch validation
  - NaN value rejection in timestamps
  - Inf value rejection in timestamps
  - Negative timestamp rejection
  - Data integrity verification via round-trip with read_events()
  - Behavior module reuse with existing data
- Implemented `write_region_crossings()` in `_events.py`:
  - Uses `_get_or_create_processing_module()` for behavior module
  - Uses `_require_ndx_events()` for dependency validation
  - Creates ndx-events EventsTable with timestamp column
  - Adds region and event_type columns for crossing metadata
  - Validates 1D shape, finite values, non-negative timestamps
  - Validates matching lengths for all parallel arrays
  - Supports name parameter for custom table names
  - Supports overwrite parameter for replacement
  - Comprehensive NumPy-style docstring with examples
- Code review: APPROVED
  - Updated docstring to clarify event_types accepts arbitrary strings
  - Not just "enter"/"exit" - supports custom event types like "approach", "dwell"
- All 18 tests pass, ruff check clean, mypy passes
- **Milestone M2: Writing is now complete**

### 2025-11-23: M2.2.1 Complete - Lap Events Writing

- Created 16 tests for `write_laps()` in `tests/nwb/test_events.py`
- Tests cover:
  - Basic lap times writing to processing/behavior/ module
  - Direction/lap_types column support
  - EventsTable creation and verification
  - Custom description and name parameters
  - Default description verification
  - Duplicate name error handling
  - overwrite=True replaces existing table
  - Empty lap times array handling
  - Shape validation (1D only, 2D rejected)
  - lap_types length mismatch validation
  - NaN value rejection in timestamps
  - Inf value rejection in timestamps
  - Negative timestamp rejection
  - Data integrity verification via round-trip with read_events()
  - Behavior module reuse with existing data
  - Import error handling for ndx-events
- Implemented `write_laps()` in `_events.py`:
  - Uses `_get_or_create_processing_module()` for behavior module
  - Uses `_require_ndx_events()` for dependency validation
  - Creates ndx-events EventsTable with timestamp column
  - Optional direction column for lap types
  - Validates 1D shape, finite values, non-negative timestamps
  - Supports name parameter for custom table names
  - Supports overwrite parameter for replacement
  - Comprehensive NumPy-style docstring with examples
- Code review: APPROVED
  - Added NaN/Inf/negative timestamp validation per reviewer feedback
  - Removed unused sample_env fixture from tests
- All 16 tests pass, ruff check clean, mypy passes

### 2025-11-23: M2.1.2 Complete - Occupancy Map Writing

- Created 18 tests for `write_occupancy()` in `tests/nwb/test_fields.py`
- Tests cover:
  - Basic occupancy writing to analysis/ module
  - Occupancy metadata (description, unit)
  - Data integrity verification
  - Shape validation (1D only, 2D rejected)
  - Duplicate name handling without overwrite
  - overwrite=True replaces existing occupancy
  - bin_centers reference stored and deduplicated
  - Multiple spatial fields share single bin_centers dataset
  - Default name ("occupancy")
  - Empty description allowed
  - Analysis module reuse
  - NaN value preservation
  - Zero value preservation
  - Unit parameter variations (seconds, probability, counts)
- Implemented `write_occupancy()` in `_fields.py`:
  - Uses `_validate_1d_field_shape()` helper for 1D-only validation
  - Reuses `_ensure_bin_centers()` helper for bin_centers deduplication
  - Stores occupancy as TimeSeries in analysis/ processing module
  - 1D occupancy stored as (1, n_bins) for NWB TimeSeries compatibility
  - Default unit is "seconds" (common for time-in-bin occupancy)
  - Includes comprehensive docstring with Notes and Examples
- Code review: APPROVED with feedback
  - Renamed `units` parameter to `unit` for consistency with `write_place_field()` and NWB
  - Updated tests from `test_units_*` to `test_unit_*`
- All 18 tests pass, ruff check clean, mypy passes

### 2025-11-23: M2.1.1 Complete - Place Field Writing

- Created 19 tests for `write_place_field()` in `tests/nwb/test_fields.py`
- Tests cover:
  - Basic field writing to analysis/ module
  - Field metadata (description, units)
  - Data integrity verification
  - Shape validation (1D, 2D, and 3D rejection)
  - Duplicate name handling without overwrite
  - overwrite=True replaces existing fields
  - bin_centers reference stored and deduplicated
  - Multiple fields share single bin_centers dataset
  - Default name ("place_field")
  - Empty description allowed
  - Analysis module reuse
  - NaN value preservation
  - Custom unit parameter
- Implemented `write_place_field()` in `_fields.py`:
  - Uses `_validate_field_shape()` helper for shape validation
  - Uses `_ensure_bin_centers()` helper for bin_centers deduplication
  - Stores fields as TimeSeries in analysis/ processing module
  - 1D fields stored as (1, n_bins) for NWB TimeSeries compatibility
  - Supports custom unit parameter (default: "Hz")
  - Includes comprehensive docstring with Notes section
- Code review: APPROVED after addressing feedback
  - Fixed mypy type error with timestamps variable
  - Added unit parameter for customizable field units
  - Added Notes section to docstring
  - Added test for 3D field rejection
- All 19 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.4.4 Complete - Environment from Position Factory

- Architectural decision: Moved `environment_from_position()` from `_overlays.py` to `_environment.py`
  - Function creates Environment, not an overlay
  - Better cohesion with environment-related functions
- Created 15 tests for `environment_from_position()` in `tests/nwb/test_environment.py`
- Tests cover:
  - Basic Environment creation from NWB Position data
  - Position bounds matching (environment extent matches position data)
  - units parameter propagation (explicit and auto-detection from SpatialSeries)
  - frame parameter propagation
  - infer_active_bins parameter forwarding
  - kwargs forwarded to Environment.from_samples()
  - processing_module parameter forwarding
  - position_name parameter for multiple SpatialSeries
  - Error handling: Position not found (KeyError)
  - Error handling: processing_module not found (KeyError)
  - bin_size required (TypeError)
  - Different bin sizes produce different grid resolutions
  - Environment is fitted and ready to use
  - Connectivity graph properly created
- Implemented `environment_from_position()` in `_environment.py`:
  - Reads position data via `read_position()`
  - Auto-detects units from SpatialSeries via `_get_position_units()` helper
  - Creates Environment via `Environment.from_samples()`
  - Sets metadata (units, frame)
  - Calls `_require_pynwb()` for consistency with other NWB functions
- Updated `__init__.py` to import from `_environment.py` instead of `_overlays.py`
- Removed placeholder from `_overlays.py`
- Code review: APPROVED
- All 15 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.4.3 Complete - Head Direction Overlay Factory

- Created 11 tests for `head_direction_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
- Tests cover:
  - Basic HeadDirectionOverlay creation from NWB CompassDirection data
  - Data integrity verification (matches `read_head_direction()` output)
  - color parameter passed through (default: "yellow")
  - length parameter passed through (default: 15.0)
  - Default parameters verification
  - processing_module forwarding to `read_head_direction()`
  - compass_name forwarding to `read_head_direction()`
  - Additional kwargs passthrough (e.g., `interp`)
  - Combined multiple parameters usage
  - Error handling: CompassDirection not found (KeyError)
  - Error handling: named compass series not found (KeyError with available list)
- Implemented `head_direction_overlay_from_nwb()` in `_overlays.py`:
  - Uses lazy imports for `HeadDirectionOverlay` and `read_head_direction`
  - Delegates NWB reading to `read_head_direction()`
  - Returns `HeadDirectionOverlay` with data, times, color, and length
- Code review: APPROVED
- All 11 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.4.2 Complete - Bodypart Overlay Factory

- Created 11 tests for `bodypart_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
- Tests cover:
  - Basic BodypartOverlay creation from NWB PoseEstimation data
  - Data integrity verification (matches `read_pose()` output)
  - Skeleton auto-extraction from PoseEstimation
  - Skeleton matches `read_pose()` output
  - colors parameter passed through
  - Default colors is None (defer to skeleton colors)
  - pose_estimation_name forwarding to `read_pose()`
  - Additional kwargs passthrough (e.g., `interp`)
  - Combined multiple parameters usage
  - Error handling: PoseEstimation not found (KeyError)
  - Error handling: named PoseEstimation not found (KeyError with available list)
- Implemented `bodypart_overlay_from_nwb()` in `_overlays.py`:
  - Uses lazy imports for `BodypartOverlay` and `read_pose`
  - Delegates NWB reading to `read_pose()`
  - Returns `BodypartOverlay` with data, times, skeleton, and colors
- Code review: APPROVED
- All 11 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.4.1 Complete - Position Overlay Factory

- Created 11 tests for `position_overlay_from_nwb()` in `tests/nwb/test_overlays.py`
- Tests cover:
  - Basic PositionOverlay creation from NWB Position data
  - Data integrity verification (matches `read_position()` output)
  - color, size, trail_length parameters passed through
  - Default parameters verification
  - processing_module and position_name forwarding to `read_position()`
  - Additional kwargs passthrough (e.g., `interp`)
  - Combined multiple parameters usage
  - Error handling: Position not found (KeyError)
- Implemented `position_overlay_from_nwb()` in `_overlays.py`:
  - Uses lazy imports for `PositionOverlay` and `read_position`
  - Delegates NWB reading to `read_position()`
  - Returns `PositionOverlay` with data, times, and styling parameters
- Code review: APPROVED
- All 11 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.3.2 Complete - TimeIntervals Reading Function

- Created 8 tests for `read_intervals()` in `tests/nwb/test_events.py`
- Tests cover:
  - Reading trials table (predefined NWB table)
  - Reading epochs table (predefined NWB table)
  - Reading custom intervals (user-defined TimeIntervals)
  - start_time/stop_time columns in output
  - Error handling: interval not found (KeyError)
  - Additional columns preserved
  - Empty intervals table handling
  - Data integrity verification
- Implemented `read_intervals()` in `_events.py`:
  - No extension dependency (uses built-in pynwb TimeIntervals)
  - Supports predefined tables: trials, epochs, invalid_times
  - Supports custom intervals via nwbfile.intervals
  - Returns pandas DataFrame with start_time, stop_time, and all columns
- All 8 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.3.1 Complete - Events Reading Function

- Created 11 tests for `read_events()` in `tests/nwb/test_events.py`
- Tests cover:
  - Basic EventsTable reading (data and timestamps)
  - Data integrity verification against original NWB data
  - Explicit processing_module parameter
  - Error handling: EventsTable not found, module not found, wrong type (TypeError)
  - DataFrame output with timestamp column
  - Additional columns preserved (direction, duration)
  - Empty EventsTable handling
  - String columns preservation
  - ImportError for missing ndx-events
- Implemented `read_events()` in `_events.py`:
  - Explicit table naming (no auto-discovery per design decision)
  - Type validation using `isinstance()` for EventsTable
  - Debug logging for consistency with other NWB modules
  - Returns pandas DataFrame with timestamp and all columns
- Code review: APPROVED
- All 11 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.2.2 Complete - Skeleton Integration

- Added 5 round-trip tests for ndx-pose Skeleton → neurospatial Skeleton conversion
- Tests verify:
  - Skeleton name preservation
  - All nodes preserved
  - All edges preserved (including edge canonicalization)
  - Skeleton with no edges (isolated nodes)
  - Complex graph structures (star topology)
- Refactored tests to use `_create_nwb_with_pose()` helper function for reduced duplication
- All 18 tests pass in `test_pose.py` (13 original + 5 round-trip)

### 2025-11-23: M1.2.1 Complete - Pose Reading Function

- Created 13 tests for `read_pose()` in `tests/nwb/test_pose.py`
- Tests cover:
  - Basic PoseEstimation reading (data, timestamps, skeleton)
  - Data integrity verification against original NWB data
  - Explicit pose_estimation_name parameter
  - Error handling: no PoseEstimation found, named pose not found
  - Skeleton extraction from PoseEstimation (using `Skeleton.from_ndx_pose()`)
  - Multiple bodyparts conversion to dict
  - Multiple PoseEstimation containers selection (alphabetical, INFO log)
  - Priority behavior (behavior module over others)
  - Rate-based timestamp computation (with and without starting_time offset)
  - 3D pose data support
- Implemented `read_pose()` in `_pose.py`:
  - Uses type-based container discovery via `_find_containers_by_type()`
  - Converts ndx-pose Skeleton to neurospatial Skeleton
  - Returns tuple: (bodyparts dict, timestamps, skeleton)
  - Proper error handling with ValueError for empty containers
- Code review: APPROVED
- All 13 tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.1.3 Complete - Head Direction Reading Function

- Created 14 tests for `read_head_direction()` in `tests/nwb/test_behavior.py`
- Tests cover:
  - Basic CompassDirection reading (2 tests - data and match original)
  - Explicit processing_module parameter
  - Explicit compass_name parameter
  - Error handling: no CompassDirection found, named compass not found, module not found, empty container
  - Multiple SpatialSeries selection (alphabetical, INFO log)
  - Priority behavior (behavior module over others)
  - Acquisition fallback
  - Rate-based timestamp computation (with and without starting_time offset)
  - 2D column vector input (edge case for different storage formats)
- Refactored helper functions for code reuse:
  - Renamed `_get_position_container()` to `_get_behavior_container()` with `type_name` parameter
  - Updated `_get_spatial_series()` with `container_type_name` parameter
- Implemented `read_head_direction()` in `_behavior.py`:
  - Uses `.ravel()` to handle both 1D and 2D column vector angle data
  - Same auto-discovery and error handling patterns as `read_position()`
- Code review: APPROVED
- All 29 tests pass (15 position + 14 head direction), ruff check clean, mypy passes

### 2025-11-23: M1.1.2 Complete - Position Reading Function

- Created `tests/nwb/test_behavior.py` with 15 tests for `read_position()`
- Tests cover:
  - Basic position reading (2D data)
  - Explicit processing_module parameter
  - Explicit position_name parameter
  - Error handling: no Position found, named Position not found, module not found
  - Multiple SpatialSeries selection (alphabetical, INFO log)
  - Priority behavior (behavior module over others, processing over acquisition)
  - 1D and 3D position data
  - Rate-based timestamp computation (with and without starting_time offset)
  - Empty Position container handling
- Implemented `read_position()` in `_behavior.py` with helper functions:
  - `_get_position_container()` - finds Position container with priority search
  - `_get_spatial_series()` - extracts SpatialSeries with auto-selection
  - `_get_timestamps()` - handles both explicit and rate-computed timestamps
- Code review: APPROVED
- All tests pass, ruff check clean, mypy passes

### 2025-11-23: M1.1.1 Complete - Core Discovery Utilities

- Created `tests/nwb/test_core.py` with 16 tests for core discovery utilities
- Tests cover:
  - `_find_containers_by_type()` - finding NWB containers by type
  - Search order priority: processing/behavior > processing/* > acquisition
  - Multiple containers in same module
  - Empty file handling
  - Type not present handling
  - Helper functions (`_require_pynwb`, `_require_ndx_pose`, `_require_ndx_events`)
  - `_get_or_create_processing_module()`
  - Logger setup verification
- Code review: APPROVED
- All tests pass, linting clean

### 2025-11-23: M0.3 Complete - Test Infrastructure Created

- Created `tests/nwb/` directory with `__init__.py` and `conftest.py`
- Implemented fixtures: `sample_nwb_with_position`, `sample_nwb_with_pose`, `sample_nwb_with_events`
- Additional fixtures: `sample_nwb_with_head_direction`, `sample_nwb_with_position_multiple_series`
- Helper functions: `create_empty_nwb`, `_get_pynwb`, `_get_ndx_pose`, `_get_ndx_events`
- Uses `pytest.importorskip()` for graceful skipping when NWB deps not installed
- Verified fixtures work with pynwb 3.1.2

### 2025-11-23: M0.2 Complete - Optional Dependencies Added

- Added optional dependency extras to pyproject.toml:
  - `nwb`: pynwb>=2.5.0, hdmf>=3.0.0
  - `nwb-pose`: pynwb, ndx-pose>=0.2.0
  - `nwb-events`: pynwb, ndx-events>=0.4.0
  - `nwb-full`: all NWB dependencies combined
- Added NWB modules to mypy ignore_missing_imports
- Verified: `uv sync --extra nwb-full` installs all dependencies

### 2025-11-23: M0.1 Complete - Module Structure Created

- Created `src/neurospatial/nwb/` directory structure
- Created `__init__.py` with lazy import pattern for public API
- Created `_core.py` with discovery utilities (`_find_containers_by_type`, `_require_pynwb`, etc.)
- Created placeholder files: `_behavior.py`, `_pose.py`, `_events.py`, `_fields.py`, `_environment.py`, `_overlays.py`
- All files have NumPy-style docstrings and proper type annotations
- Fixed import path for overlay classes (`neurospatial.animation.overlays` not `neurospatial.animation`)
- Verified: ruff check passes, mypy passes, module imports correctly

### 2025-11-23: Project Initialized

- Created TASKS.md with full implementation checklist
- Created SCRATCHPAD.md for session notes
- Plan reference: PLAN.md (NWB Integration Plan v1.1)

---

## Design Decisions

### DD-001: Edge List as Canonical Graph Format

**Decision**: Use edge list (n_edges, 2) instead of CSR for NWB storage

**Rationale**:
- Simple, self-explanatory format
- Works with any language (MATLAB, Julia, R)
- CSR can be reconstructed on load (cost is negligible)

**Reference**: PLAN.md - Design Principles, Section 3

### DD-002: Standard NWB Types Only (Phase 1)

**Decision**: Use only standard NWB types, no custom extension initially

**Rationale**:
- Immediate usability without installing custom extensions
- Broad compatibility with existing NWB tools
- Lower barrier for adoption
- Future extension is a thin wrapper (separate repo)

**Reference**: PLAN.md - Design Principles, Section 1

### DD-003: Data Location Strategy

**Decision**: Use appropriate standard NWB locations

| Data Type | NWB Location |
|-----------|--------------|
| Lap events | processing/behavior/ |
| Region crossings | processing/behavior/ |
| Place fields | analysis/ |
| Occupancy maps | analysis/ |
| Environment | scratch/ |

**Reference**: PLAN.md - Design Principles, Section 2

---

## Technical Notes

### Lazy Import Pattern

All NWB dependencies use lazy imports to keep them optional:

```python
def read_position(nwbfile):
    try:
        import pynwb
    except ImportError:
        raise ImportError(
            "pynwb is required for NWB integration. "
            "Install with: pip install pynwb"
        )
    # ... implementation
```

### Container Discovery Priority

When multiple containers exist, use this search order:
1. `processing/behavior/`
2. `processing/*` (other modules, alphabetically)
3. `acquisition/`

Use `isinstance()` checks for robust discovery.

---

## Open Questions

- [ ] Q1: Should we support streaming/lazy loading for large NWB files?
  - Deferred to future enhancement

- [ ] Q2: Cloud storage support (DANDI URLs)?
  - Deferred to future enhancement

---

## Blockers

*None currently*

---

## Testing Notes

### Test Fixtures Required

1. `sample_nwb_with_position` - NWB file with Position data
2. `sample_nwb_with_pose` - NWB file with PoseEstimation (requires ndx-pose)
3. `sample_nwb_with_events` - NWB file with EventsTable (requires ndx-events)
4. `create_empty_nwb()` - Helper to create minimal NWB file

### Skip Markers

Tests requiring optional dependencies should use:
```python
pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")
pytest.importorskip("ndx_events")
```

---

## References

- [PLAN.md](PLAN.md) - Full NWB Integration Plan
- [PyNWB Documentation](https://pynwb.readthedocs.io/en/stable/)
- [ndx-pose Repository](https://github.com/rly/ndx-pose)
- [ndx-events Repository](https://github.com/rly/ndx-events)
