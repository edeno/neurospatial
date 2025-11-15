# neurospatial Code Review Implementation Plan

**Generated**: 2025-11-14
**Based on**: Comprehensive code review by 8 specialized reviewers

This document outlines a prioritized plan to address issues found in the comprehensive code review. Issues are organized by priority with estimated effort and clear acceptance criteria.

---

## Summary Statistics

- **Critical Issues**: 8 (must fix before v1.0)
- **High Priority**: 12 (should fix for v1.1)
- **Medium Priority**: 10 (target for v1.2)
- **Low Priority / Nice to Have**: 10+ (future enhancements)

**Estimated Total Effort**: ~6-8 weeks for critical + high priority items

---

## Phase 1: Critical Fixes (Target: 1-2 weeks)

These issues must be addressed before any production deployment or v1.0 release.

### 1.1 Security: Path Traversal Vulnerability

**Priority**: CRITICAL
**Effort**: 2 hours
**Files**: `src/neurospatial/io.py:141-146`

**Issue**: `to_file()` accepts arbitrary user paths without validation, enabling path traversal attacks.

**Tasks**:
- [ ] Add path validation to prevent `..` in path components
- [ ] Add test case for path traversal attempt
- [ ] Update docstring to document allowed paths

**Implementation**:
```python
def to_file(env: Environment, path: str | Path) -> None:
    path_obj = Path(path).resolve()

    # Validate path is safe (no parent directory traversal)
    if ".." in path_obj.parts:
        raise ValueError(
            f"Path traversal detected in path: {path}. "
            f"Use absolute paths or paths without '..' components."
        )

    json_path = path_obj.with_suffix(".json")
    npz_path = path_obj.with_suffix(".npz")

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # ... rest of implementation
```

**Test**:
```python
def test_to_file_rejects_path_traversal(tmp_path, simple_env):
    with pytest.raises(ValueError, match="Path traversal detected"):
        to_file(simple_env, tmp_path / "../../../etc/passwd")
```

**Acceptance Criteria**:
- Path traversal attempts raise ValueError
- Test coverage for attack vectors
- Documentation updated

---

### 1.2 Numerical Stability: Division by Zero in Trajectory Analysis

**Priority**: CRITICAL
**Effort**: 3 hours
**Files**: `src/neurospatial/environment/trajectory.py:1171`

**Issue**: Ray-grid intersection can divide by very small numbers (near epsilon threshold), producing inf/nan.

**Tasks**:
- [ ] Add epsilon check before division in ray-grid intersection
- [ ] Add test for ray parallel to grid edge (ray_dir[dim] ≈ 0)
- [ ] Add test for very small ray direction components

**Implementation**:
```python
# In _compute_occupancy_linear_allocation
EPSILON = 1e-12

for dim in range(n_dims):
    # Skip if ray is parallel to this dimension's grid lines
    if abs(ray_dir[dim]) < EPSILON:
        continue

    # Safe to divide now
    t = (edge_pos - start_pos[dim]) / ray_dir[dim]
```

**Test**:
```python
def test_occupancy_ray_parallel_to_edge(simple_env):
    """Test trajectory segment parallel to grid edge."""
    times = np.array([0.0, 1.0])
    # Movement only in x (parallel to y-edges)
    positions = np.array([[10.0, 10.0], [20.0, 10.0]])

    occupancy = simple_env.occupancy(times, positions, method="linear")
    assert np.all(np.isfinite(occupancy))
    assert np.sum(occupancy) > 0
```

**Acceptance Criteria**:
- No inf/nan in occupancy calculations
- Test coverage for parallel rays
- Performance not significantly impacted

---

### 1.3 Numerical Stability: Floating Point Comparison in Hexagonal Layout

**Priority**: CRITICAL
**Effort**: 2 hours
**Files**: `src/neurospatial/layout/helpers/hexagonal.py:205, 212-214`

**Issue**: Direct equality check on floating point `hex_radius` without tolerance.

**Tasks**:
- [ ] Replace `hex_radius == 0` with `np.isclose(hex_radius, 0.0, atol=1e-12)`
- [ ] Add validation for minimum hex_radius to prevent near-zero values
- [ ] Add test for very small hexagon_width

**Implementation**:
```python
def _points_to_hex_coords(points_x, points_y, hex_radius):
    """Convert Cartesian to hexagonal axial coordinates."""
    # Validate hex_radius is not too small
    MIN_HEX_RADIUS = 1e-10
    if abs(hex_radius) < MIN_HEX_RADIUS:
        raise ValueError(
            f"hex_radius too small ({hex_radius}). "
            f"Minimum supported radius: {MIN_HEX_RADIUS}"
        )

    if np.isclose(hex_radius, 0.0, atol=1e-12):
        zero_coords = np.zeros_like(points_x)
        return zero_coords, zero_coords, zero_coords

    # Safe to divide
    q_frac = (np.sqrt(3.0) / 3.0 * points_x - 1.0 / 3.0 * points_y) / hex_radius
    r_frac = (2.0 / 3.0 * points_y) / hex_radius
```

**Test**:
```python
def test_hexagonal_layout_very_small_width():
    """Test hex layout with width near numerical precision limits."""
    layout = HexagonalLayout()

    # Should raise error for too-small width
    with pytest.raises(ValueError, match="hex_radius too small"):
        layout.build(hexagon_width=1e-12, dimension_ranges=[(0, 1e-6), (0, 1e-6)])

    # Should work for small but valid width
    layout.build(hexagon_width=1e-8, dimension_ranges=[(0, 1e-6), (0, 1e-6)])
    assert np.all(np.isfinite(layout.bin_centers))
```

**Acceptance Criteria**:
- No direct float equality checks
- Clear error for too-small hex_radius
- Test coverage for edge cases

---

### 1.4 Correctness: Metadata Mutability Leak in Region

**Priority**: CRITICAL
**Effort**: 2 hours
**Files**: `src/neurospatial/regions/core.py:63-65`

**Issue**: Region class uses shallow copy of metadata, allowing mutation through nested structures.

**Tasks**:
- [ ] Replace `dict(self.metadata)` with `copy.deepcopy(dict(self.metadata))`
- [ ] Add test verifying nested metadata is immutable
- [ ] Document immutability guarantee in docstring

**Implementation**:
```python
import copy

@dataclass(frozen=True, slots=True)
class Region:
    """Immutable region of interest (ROI) in an environment."""

    def __post_init__(self) -> None:
        # Deep freeze metadata to prevent mutation through nested structures
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
```

**Test**:
```python
def test_region_metadata_deeply_immutable():
    """Test that nested metadata cannot be mutated."""
    metadata = {"color": "red", "tags": ["important", "goal"]}
    region = Region("test", "point", (0.0, 0.0), metadata)

    # External mutation should not affect region
    metadata["tags"].append("new_tag")
    assert region.metadata["tags"] == ["important", "goal"]

    # Direct mutation should also fail
    with pytest.raises((AttributeError, TypeError)):
        region.metadata["color"] = "blue"
```

**Acceptance Criteria**:
- Nested structures in metadata are deeply copied
- Test verifies complete immutability
- No performance regression for typical metadata sizes

---

### 1.5 Testing: Add 3D Environment Test Coverage

**Priority**: CRITICAL
**Effort**: 4 hours
**Files**: `tests/conftest.py`, `tests/test_environment.py`

**Issue**: All test fixtures use 2D data despite 3D support existing. Critical gap in test coverage.

**Tasks**:
- [ ] Add `simple_3d_env` fixture to conftest.py
- [ ] Add test class `TestEnvironment3D` with core operations
- [ ] Test 3D-specific edge cases (26-neighbor connectivity)
- [ ] Add 3D serialization roundtrip test

**Implementation**:
```python
# In tests/conftest.py
@pytest.fixture
def simple_3d_env() -> Environment:
    """3D environment for testing volumetric operations."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((2000, 3)) * 20  # 3D data
    return Environment.from_samples(
        data,
        bin_size=5.0,
        name="test_3d",
        connect_diagonal_neighbors=True
    )

# In tests/test_environment.py
class TestEnvironment3D:
    """Tests for 3D environments."""

    def test_creation_3d(self, simple_3d_env):
        assert simple_3d_env.n_dims == 3
        assert simple_3d_env.bin_centers.shape[1] == 3
        assert simple_3d_env.n_bins > 0

    def test_bin_at_3d(self, simple_3d_env):
        points = np.array([[10, 20, 30], [0, 0, 0], [-5, 10, 15]])
        bin_indices = simple_3d_env.bin_at(points)
        assert bin_indices.shape == (3,)
        assert np.all(bin_indices >= -1)

    def test_neighbors_3d_connectivity(self, simple_3d_env):
        # 3D grid with diagonals: up to 26 neighbors (3^3 - 1)
        neighbors = simple_3d_env.neighbors(10)
        assert len(neighbors) <= 26
        assert len(neighbors) >= 6  # At least orthogonal neighbors

    def test_serialization_roundtrip_3d(self, simple_3d_env, tmp_path):
        simple_3d_env.to_file(tmp_path / "env_3d")
        loaded = Environment.from_file(tmp_path / "env_3d")

        assert loaded.n_dims == 3
        np.testing.assert_array_equal(loaded.bin_centers, simple_3d_env.bin_centers)
```

**Acceptance Criteria**:
- 3D environment fixture available
- Core operations tested in 3D (bin_at, neighbors, distance_between)
- Serialization tested for 3D
- No failures in existing 2D tests

---

### 1.6 UX: Add Progress Feedback for Long Operations

**Priority**: CRITICAL
**Effort**: 6 hours
**Files**: `src/neurospatial/spike_field.py`, `src/neurospatial/environment/trajectory.py`

**Issue**: Operations like `compute_place_field` can take minutes with no progress indication.

**Tasks**:
- [ ] Add optional `tqdm` dependency
- [ ] Add progress bars to `compute_place_field` (KDE iterations)
- [ ] Add progress to `occupancy` (linear allocation loop)
- [ ] Add progress to large environment creation
- [ ] Add `show_progress` parameter (default True)

**Implementation**:
```python
# In pyproject.toml
[project.optional-dependencies]
progress = ["tqdm>=4.66.0"]

# In spike_field.py
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

def compute_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    bandwidth: float = 5.0,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    show_progress: bool = True,
) -> NDArray[np.float64]:
    """Compute place field from spike data.

    Parameters
    ----------
    show_progress : bool, default True
        If True, display progress bar for long computations.
        Requires tqdm package (pip install neurospatial[progress]).
    """
    # ... validation ...

    # Show progress if requested and available
    progress_bar = None
    if show_progress and HAS_TQDM and tqdm is not None:
        progress_bar = tqdm(total=n_iterations, desc="Computing place field")

    try:
        for i in range(n_iterations):
            # ... computation ...
            if progress_bar:
                progress_bar.update(1)
    finally:
        if progress_bar:
            progress_bar.close()
```

**Test**:
```python
def test_compute_place_field_with_progress(simple_env, capsys):
    """Test that progress bar appears when requested."""
    pytest.importorskip("tqdm")  # Skip if tqdm not installed

    spike_times = np.array([1.0, 2.0, 3.0])
    times = np.linspace(0, 10, 1000)
    positions = np.random.randn(1000, 2) * 10

    field = compute_place_field(
        simple_env, spike_times, times, positions,
        show_progress=True
    )
    # Progress bar should have appeared (visible in tqdm.auto)
```

**Acceptance Criteria**:
- Progress bars appear for operations >5 seconds
- `show_progress=False` disables all progress output
- Graceful fallback if tqdm not installed
- Progress bars work in Jupyter notebooks (tqdm.auto)

---

### 1.7 UX: Add Units Validation

**Priority**: CRITICAL
**Effort**: 4 hours
**Files**: `src/neurospatial/environment/factories.py`, `src/neurospatial/environment/core.py`

**Issue**: `bin_size=5.0` accepted without asking what units data is in. Silent errors possible.

**Tasks**:
- [ ] Add `units` parameter to `from_samples()` and other factory methods
- [ ] Warn if units not provided
- [ ] Add `validate_environment()` check for missing units
- [ ] Update examples to always show units

**Implementation**:
```python
@classmethod
def from_samples(
    cls,
    data_samples: NDArray[np.float64],
    bin_size: float | Sequence[float],
    name: str = "",
    units: str | None = None,  # NEW PARAMETER
    frame: str | None = None,
    # ... other params
) -> Environment:
    """Create environment from sample data.

    Parameters
    ----------
    units : str, optional
        Spatial units of the data and bin_size (e.g., 'cm', 'meters', 'pixels').
        HIGHLY RECOMMENDED to specify for reproducibility and to prevent
        unit confusion errors.
    """
    env = cls.from_layout(...)

    if units is None:
        warnings.warn(
            "Creating environment without units. Consider setting units "
            "(e.g., 'cm', 'meters', 'pixels') for clarity and reproducibility. "
            "Example: Environment.from_samples(data, bin_size=5.0, units='cm')",
            UserWarning,
            stacklevel=2
        )
    else:
        env.units = units

    if frame:
        env.frame = frame

    return env
```

**Documentation Update**:
```python
# Update all examples in docstrings
>>> env = Environment.from_samples(data, bin_size=2.0, units='cm')
>>> env.units
'cm'
```

**Acceptance Criteria**:
- Warning issued if units not provided
- `units` parameter documented in all factory methods
- Examples updated to show units
- CLAUDE.md updated with units guidance

---

### 1.8 API: Fix Import Inconsistencies

**Priority**: CRITICAL
**Effort**: 2 hours
**Files**: `src/neurospatial/__init__.py`

**Issue**: CLAUDE.md documents imports that don't work (e.g., `from neurospatial import to_file`).

**Tasks**:
- [ ] Add `to_file`, `from_file`, `to_dict`, `from_dict` to top-level exports
- [ ] Add `Region`, `Regions` to top-level exports
- [ ] Add `clear_kdtree_cache` to top-level exports
- [ ] Update `__all__` and verify against CLAUDE.md

**Implementation**:
```python
# In src/neurospatial/__init__.py

# Add to imports section
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.regions import Region, Regions
from neurospatial.spatial import clear_kdtree_cache

# Update __all__ (keep alphabetically sorted within groups)
__all__ = [
    # Core classes
    "CompositeEnvironment",
    "Environment",
    "Region",
    "Regions",

    # Enums
    "LayoutType",
    "TieBreakStrategy",

    # I/O functions
    "from_dict",
    "from_file",
    "to_dict",
    "to_file",

    # Spatial queries
    "clear_kdtree_cache",
    "distance_field",
    "map_points_to_bins",
    # ... rest alphabetically
]
```

**Test**:
```python
def test_top_level_imports():
    """Verify all documented top-level imports work."""
    # Should all work without error
    from neurospatial import (
        Environment, Region, Regions,
        to_file, from_file, to_dict, from_dict,
        map_points_to_bins, clear_kdtree_cache,
    )
```

**Acceptance Criteria**:
- All imports in CLAUDE.md work
- `__all__` matches documented API
- No breaking changes to existing imports

---

## Phase 2: High Priority (Target: 2-3 weeks)

These issues significantly impact code quality and user experience.

### 2.1 Code Quality: Extract Graph Connectivity Helper

**Priority**: HIGH
**Effort**: 8 hours
**Files**: `src/neurospatial/layout/helpers/regular_grid.py:38-200`, `src/neurospatial/layout/helpers/hexagonal.py:508-630`

**Issue**: ~70% code duplication between graph connectivity creation functions.

**Tasks**:
- [ ] Create `_create_connectivity_graph_generic()` helper
- [ ] Refactor regular_grid to use new helper
- [ ] Refactor hexagonal to use new helper
- [ ] Verify all tests still pass
- [ ] Add tests for new helper function

---

### 2.2 Performance: Optimize region_membership()

**Priority**: HIGH
**Effort**: 3 hours
**Files**: `src/neurospatial/environment/regions.py:366-394`

**Issue**: Creates shapely Points array N times (once per region) instead of once total.

---

### 2.3 Testing: Add Performance Regression Tests

**Priority**: HIGH
**Effort**: 6 hours
**Files**: `tests/test_performance.py` (new)

**Tasks**:
- [ ] Create `tests/test_performance.py` module
- [ ] Add pytest marker for slow tests
- [ ] Add large environment creation benchmark
- [ ] Add KDTree batch query benchmark
- [ ] Add shortest path benchmark
- [ ] Configure CI to track performance over time

---

### 2.4 Testing: Add Property-Based Tests

**Priority**: HIGH
**Effort**: 8 hours
**Files**: `tests/test_properties.py` (new)

**Tasks**:
- [ ] Add hypothesis dependency
- [ ] Create property-based test module
- [ ] Add tests for mathematical properties (triangle inequality, etc.)
- [ ] Add tests for transformation properties
- [ ] Configure hypothesis settings

---

### 2.5 UX: Standardize Parameter Naming

**Priority**: HIGH
**Effort**: 6 hours
**Files**: Multiple files using `data_samples` vs `positions`

**Tasks**:
- [ ] Replace `data_samples` with `positions` in `from_samples()`
- [ ] Add deprecation warning for `data_samples` (keep as alias)
- [ ] Update all docstrings and examples
- [ ] Update CLAUDE.md

---

### 2.6 Documentation: Add Error Codes

**Priority**: HIGH
**Effort**: 4 hours
**Files**: Error-raising code throughout, `docs/errors.md` (new)

**Tasks**:
- [ ] Create error code system (E1001, E1002, etc.)
- [ ] Add error codes to common errors
- [ ] Create `docs/errors.md` with solutions
- [ ] Link error messages to documentation

---

### 2.7 Refactoring: Split Long Methods

**Priority**: MEDIUM
**Effort**: 4 hours
**Files**: `src/neurospatial/layout/helpers/regular_grid.py:301-483`

**Tasks**:
- [ ] Split `_create_regular_grid()` into smaller functions
- [ ] Extract validation logic
- [ ] Extract range inference logic
- [ ] Verify tests still pass

---

### 2.8 Type Safety: Fix SubsetLayout Annotations

**Priority**: MEDIUM
**Effort**: 2 hours
**Files**: `src/neurospatial/environment/transforms.py:562-612`

**Tasks**:
- [ ] Remove incorrect `self: SelfEnv` from SubsetLayout
- [ ] Add proper type hints to SubsetLayout methods
- [ ] Run mypy to verify fixes

---

### 2.9-2.12 Additional High Priority Items

**2.9 Documentation: Add Module-Level API Overview**
Effort: 2 hours
Add comprehensive docstring to `src/neurospatial/__init__.py`

**2.10 Consistency: Standardize Scale Parameter Naming**
Effort: 2 hours
Unify `scale`, `scale_factor`, `sx`/`sy` naming

**2.11 Memory: Add KDTree Cache Management Helpers**
Effort: 3 hours
Add `env.clear_cache()` method, memory usage warnings

**2.12 Code Cleanup: Remove Dead Code**
Effort: 1 hour
Remove commented code in `multi_index_to_flat()`

---

## Phase 3: Medium Priority (Target: v1.2, 4-6 weeks)

### 3.1 Feature: Add env.info() Method

**Priority**: MEDIUM
**Effort**: 3 hours

Add formatted summary of environment properties for easy inspection.

---

### 3.2 Feature: Workflow Convenience Functions

**Priority**: MEDIUM
**Effort**: 6 hours

Create `src/neurospatial/workflows.py` with common analysis patterns packaged into single functions.

---

### 3.3-3.10 Additional Medium Priority Items

**3.3 Feature: Transform Visualization Helper**
**3.4 Feature: Multi-Session Alignment**
**3.5 Feature: Builder Pattern for Complex Configs**
**3.6 Testing: Add Integration/Workflow Tests**
**3.7 Testing: Add Visual Regression Tests (pytest-mpl)**
**3.8 Documentation: Add Mathematical Formulations**
**3.9 Performance: Add Caching for Distance Fields**
**3.10 Architecture: Decouple GraphLayout from track_linearization**

---

## Phase 4: Low Priority / Nice to Have (Future)

- Add `__version__` attribute
- Schema migration framework
- Atomic file writes
- Replace magic numbers with named constants
- Add fuzzing tests
- Add mutation testing (mutmut)
- Contract tests for LayoutEngine protocol
- Parallel test execution optimization
- Benchmark suite with tracking
- Test data extraction to fixtures/data/
- Session-scoped fixtures for expensive setup

---

## Success Metrics

### Phase 1 (Critical) Completion Criteria
- [ ] All security vulnerabilities fixed
- [ ] All numerical stability issues resolved
- [ ] 3D test coverage >80%
- [ ] Progress bars on all long operations
- [ ] Units handling improved with warnings
- [ ] Import patterns match documentation
- [ ] All critical tests passing

### Phase 2 (High Priority) Completion Criteria
- [ ] Code duplication <10%
- [ ] Performance regression tests in CI
- [ ] Property-based tests covering key invariants
- [ ] API consistency improved (positions, error codes)
- [ ] User-facing documentation complete

### Overall Quality Metrics
- Test coverage >85%
- mypy passes with no errors
- ruff/black formatting consistent
- Documentation coverage >90%
- Performance within 10% of baseline
- No critical or high severity issues open

---

## Timeline

**Week 1-2**: Phase 1 (Critical) - Items 1.1-1.8
**Week 3-5**: Phase 2 (High Priority) - Items 2.1-2.12
**Week 6-10**: Phase 3 (Medium Priority) - Items 3.1-3.10
**Ongoing**: Phase 4 (Low Priority) as time permits

**First Milestone**: v1.0 (Phases 1-2 complete) - Target 5 weeks
**Second Milestone**: v1.1 (Phase 3 partial) - Target 8 weeks
**Third Milestone**: v1.2 (Phase 3 complete) - Target 12 weeks

---

## Implementation Notes

1. **Branch Strategy**: Create feature branches for each major item (e.g., `fix/path-traversal`, `feature/progress-bars`)

2. **Testing**: Every fix must include tests demonstrating the issue and verifying the solution

3. **Documentation**: Update CLAUDE.md, docstrings, and examples for all user-facing changes

4. **Backward Compatibility**: Use deprecation warnings for API changes, maintain for at least one minor version

5. **Performance**: Run benchmarks before/after for any performance-related changes

6. **Code Review**: All changes require review focusing on correctness, usability, and maintainability

---

## Questions for Consideration

1. Should units be required (error if not provided) or optional (warning only)?
2. Should progress bars default to on or off?
3. What's the deprecation timeline for `data_samples` → `positions`?
4. Should we add a `neurospatial.experimental` module for new features?
5. What's the minimum supported Python version going forward?

---

**Last Updated**: 2025-11-14
**Status**: Draft - Pending Review
**Next Review**: After Phase 1 completion
