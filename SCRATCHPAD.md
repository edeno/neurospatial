# SCRATCHPAD.md

**Current Focus**: Milestone 1 - Egocentric Reference Frames

## Session: 2025-12-05

### Completed: M1.1 Core Reference Frame Module

**Files Created:**

- `src/neurospatial/reference_frames.py` - Core module with all functions
- `tests/test_reference_frames.py` - 35 comprehensive tests

**Implemented:**

- [x] `EgocentricFrame` dataclass with `to_egocentric()` / `to_allocentric()`
- [x] `allocentric_to_egocentric()` - batch transform (n_time, n_points, 2)
- [x] `egocentric_to_allocentric()` - inverse batch transform
- [x] `compute_egocentric_bearing()` - angle to targets relative to heading
- [x] `compute_egocentric_distance()` - Euclidean and geodesic metrics
- [x] `heading_from_velocity()` - compute heading from position timeseries
- [x] `heading_from_body_orientation()` - heading from pose keypoints
- [x] `_wrap_angle()` - wrap angles to (-π, π]
- [x] `_interpolate_heading_circular()` - circular interpolation for NaN gaps

**Key Design Decisions:**

- Allocentric: 0=East, π/2=North (standard mathematical convention)
- Egocentric: 0=ahead, π/2=left, -π/2=right, ±π=behind
- Used `np.einsum('tij,tpj->tpi', rot, centered)` for vectorized rotation
- Circular interpolation via unit vectors for heading gaps (avoids ±π discontinuity)
- Geodesic distance uses `neurospatial.distance.distance_field()`

**Tests:**

- 35/35 passing
- Module structure tests
- Dataclass tests
- Batch transform tests
- Bearing computation tests
- Distance computation tests (Euclidean and geodesic)
- Heading computation tests with edge cases

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after type safety fixes

### Completed: M1.3 Egocentric Polar Environment

**Files Modified/Created:**

- `src/neurospatial/environment/factories.py` - Added `from_polar_egocentric()` factory method
- `tests/test_polar_egocentric.py` - 21 comprehensive tests

**Implemented:**

- [x] `from_polar_egocentric()` classmethod with parameters:
  - `distance_range`, `angle_range` - define the polar space bounds
  - `distance_bin_size`, `angle_bin_size` - control bin resolution
  - `circular_angle=True` - enables periodic connectivity for full-circle angles
- [x] `_add_circular_connectivity()` helper - connects first/last angle bins at each distance ring
- [x] Comprehensive parameter validation with clear error messages
- [x] NumPy-style docstring with coordinate convention and examples

**Key Design Decisions:**

- Reuses existing `from_mask()` infrastructure (MaskedGridLayout)
- Coordinate convention matches `reference_frames`: angle 0=ahead, π/2=left, -π/2=right
- Circular edges include proper `edge_id` attribute (critical for graph consistency)
- `bin_centers[:, 0]` = distances, `bin_centers[:, 1]` = angles

**Tests:**

- 21/21 passing
- Basic creation tests (n_bins, dimensions, value ranges)
- Circular connectivity tests (wrapping, edge attributes)
- Parameter validation tests (bin sizes, ranges)
- Edge case tests (single angle bin, equal range bounds)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved after edge_id fix

### Completed: M1.5 Integration and Documentation

**Files Modified:**

- `.claude/QUICKSTART.md` - Added "Egocentric Reference Frames" section with:
  - Heading computation examples (from velocity and pose keypoints)
  - Allocentric to egocentric transform examples
  - Bearing and distance computation examples
  - Coordinate convention documentation
  - Egocentric polar environment creation example
- `.claude/API_REFERENCE.md` - Added "Egocentric Reference Frames (v0.17.0+)" section with:
  - All exports from `neurospatial` top-level
  - Full `neurospatial.reference_frames` module API
  - `Environment.from_polar_egocentric()` factory example

**Milestone 1 Complete!** All tasks in M1 (Egocentric Reference Frames) are now done.

### Next Task

- **M2.1**: Object-Vector Cell Model (Simulation)
  - Create `src/neurospatial/simulation/models/object_vector_cells.py`
  - Implement `ObjectVectorCellModel` dataclass

### Blockers

None currently.

---
