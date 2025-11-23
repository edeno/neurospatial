# SCRATCHPAD.md - NWB Integration Notes

**Last Updated**: 2025-11-23

---

## Current Status

- **Active Task**: M1.2.2 - Skeleton integration
- **Blocker**: None
- **Next Action**: Verify existing `Skeleton.from_ndx_pose()` works with read data

---

## Session Log

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
