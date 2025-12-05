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

### Next Task
- **M1.2**: Heading Computation Utilities (some already implemented in M1.1)
- **M1.3**: Egocentric Polar Environment (`from_polar_egocentric()` factory)

### Blockers
None currently.

---
