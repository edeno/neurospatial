# SCRATCHPAD.md

**Current Focus**: Milestone 2 - Object-Vector Cells

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

### Completed: M2.1 Object-Vector Cell Model (Simulation)

**Files Created:**

- `src/neurospatial/simulation/models/object_vector_cells.py` - Main implementation
- `tests/simulation/models/test_object_vector_cells.py` - 32 comprehensive tests

**Implemented:**

- [x] `ObjectVectorCellModel` class with parameters:
  - `env`, `object_positions` - required spatial setup
  - `preferred_distance`, `distance_width` - Gaussian distance tuning
  - `preferred_direction`, `direction_kappa` - optional von Mises direction tuning
  - `object_selectivity` - "any", "nearest", or "specific" modes
  - `max_rate`, `baseline_rate` - firing rate parameters
  - `distance_metric` - "euclidean" or "geodesic"
- [x] Parameter validation in `__init__` with clear error messages
- [x] Warning for objects outside environment bounds
- [x] Geodesic distance field precomputation
- [x] `firing_rate()` method with:
  - Gaussian distance tuning: `exp(-0.5 * ((d - pref) / width)^2)`
  - von Mises direction tuning: `exp(κ * cos(θ - θ_pref)) / exp(κ)`
  - Uses `compute_egocentric_bearing()` from M1
  - Three aggregation modes: any/nearest/specific
- [x] `ground_truth` property returning all model parameters

**Key Design Decisions:**

- Class-based (not dataclass) following `PlaceCellModel` pattern
- Direction tuning requires explicit `headings` parameter (not from positions)
- Von Mises normalization: divide by `exp(κ)` so peak response = 1
- Geodesic mode precomputes distance field per object for O(1) lookup
- Euclidean is default (faster, sufficient for open fields)

**Tests:**

- 32/32 passing
- Module structure tests (imports, docstring)
- Parameter validation tests (10 tests for error messages)
- Distance tuning tests (Gaussian shape verified)
- Direction tuning tests (von Mises decreasing away from preferred)
- Object selectivity tests (any/nearest/specific modes)
- Distance metric tests (euclidean/geodesic)
- Ground truth tests (completeness, immutability)
- Protocol compliance (implements NeuralModel)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - high quality scientific code
- 38% documentation ratio (522 LOC, ~200 lines of docstrings)
- Complete type annotations

**Module Updated:**

- `src/neurospatial/simulation/models/__init__.py` - Added `ObjectVectorCellModel` export

### Completed: M2.2 Object-Vector Metrics

**Files Created:**

- `src/neurospatial/metrics/object_vector_cells.py` - Main implementation (520 LOC)
- `tests/metrics/test_object_vector_cells.py` - 53 comprehensive tests

**Implemented:**

- [x] `ObjectVectorMetrics` frozen dataclass with fields:
  - `preferred_distance`, `preferred_direction` - peak tuning location
  - `distance_selectivity`, `direction_selectivity` - individual selectivity metrics
  - `object_vector_score` - combined selectivity score [0, 1]
  - `peak_rate`, `mean_rate` - firing rate statistics
  - `tuning_curve`, `distance_bins`, `direction_bins` - full 2D tuning data
  - `interpretation()` method returning human-readable string
  - `__str__()` returns interpretation
- [x] `compute_object_vector_tuning()` function:
  - Bins spikes by egocentric (distance, direction) to nearest object
  - Uses `compute_egocentric_bearing()` from M1
  - Computes occupancy-normalized firing rates
  - Applies `min_occupancy_seconds` threshold (default 0.1s)
  - Returns `ObjectVectorMetrics` dataclass
- [x] `object_vector_score()` function:
  - Distance selectivity: `s_d = peak / mean`
  - Direction selectivity: uses `_mean_resultant_length` from `metrics.circular`
  - Combined: `s_OV = ((s_d - 1) / (s_d* - 1)) * s_θ`
  - Validates `max_distance_selectivity > 1`
- [x] `is_object_vector_cell()` classifier:
  - Checks `score >= threshold` and `peak_rate >= min_rate`
  - Default thresholds: score=0.3, min_peak=5.0 Hz
- [x] `plot_object_vector_tuning()` visualization:
  - Polar heatmap (distance=radial, direction=angular)
  - Optional peak location marker
  - Optional colorbar
  - Configurable colormap

**Key Design Decisions:**

- API pattern mirrors `head_direction.py` (dataclass + compute + classifier + plot)
- Uses frozen dataclass for immutable metrics
- Egocentric coordinate convention: 0=ahead, π/2=left, -π/2=right
- Nearest object used by default (consistent with biological OVCs)
- Combined score formula from Hoydal et al. (2019)

**Tests:**

- 53/53 passing
- Module structure tests
- Dataclass tests (fields, frozen, interpretation)
- Tuning computation tests (binning, occupancy, normalization)
- Score computation tests (distance, direction, combined)
- Classifier tests (thresholds, edge cases)
- Visualization tests (polar projection, peak marking)
- Ground truth recovery tests (simulated OVC → metrics → classification)

**Code Quality:**

- ruff: All checks pass
- mypy: No issues found
- Code review: Approved - excellent scientific code
- NumPy docstrings with full examples
- Complete type annotations

**Module Updated:**

- `src/neurospatial/metrics/__init__.py` - Added all exports:
  - `ObjectVectorMetrics`, `compute_object_vector_tuning`
  - `object_vector_score`, `is_object_vector_cell`, `plot_object_vector_tuning`

### Next Task

- **M2.3**: Object-Vector Field Computation
  - Create `src/neurospatial/object_vector_field.py`
  - Implement `ObjectVectorFieldResult` dataclass
  - Implement `compute_object_vector_field()` function

### Blockers

None currently.

---
