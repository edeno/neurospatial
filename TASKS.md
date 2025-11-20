# Animation Overlays Implementation Tasks (v0.4.0)

**Status:** Ready for implementation
**Target Version:** v0.4.0
**Reference:** See [ANIMATION_IMPLEMENTATION_PLAN.md](ANIMATION_IMPLEMENTATION_PLAN.md)

---

## Milestone 1: Core Infrastructure & Validation (Weeks 1-2)

### 1.1 Public API Dataclasses

- [x] Create `src/neurospatial/animation/overlays.py` module
- [x] Implement `PositionOverlay` dataclass with NumPy docstrings
  - [x] `data: NDArray[np.float64]` with shape `(n_samples, n_dims)`
  - [x] `times: NDArray[np.float64] | None`
  - [x] `color: str = "red"`
  - [x] `size: float = 10.0`
  - [x] `trail_length: int | None = None`
- [x] Implement `BodypartOverlay` dataclass with NumPy docstrings
  - [x] `data: dict[str, NDArray[np.float64]]` for multi-keypoint poses
  - [x] `times: NDArray[np.float64] | None`
  - [x] `skeleton: list[tuple[str, str]] | None`
  - [x] `colors: dict[str, str] | None`
  - [x] `skeleton_color: str = "white"`
  - [x] `skeleton_width: float = 2.0`
- [x] Implement `HeadDirectionOverlay` dataclass with NumPy docstrings
  - [x] `data: NDArray[np.float64]` for angles (rad) or unit vectors
  - [x] `times: NDArray[np.float64] | None`
  - [x] `color: str = "yellow"`
  - [x] `length: float = 20.0`
- [x] Export overlay dataclasses in `src/neurospatial/__init__.py`
- [x] Add type annotations to satisfy mypy (use `EnvironmentProtocol` pattern)

### 1.2 Internal Data Model

- [x] Implement `PositionData` internal dataclass in `overlays.py`
- [x] Implement `BodypartData` internal dataclass in `overlays.py`
- [x] Implement `HeadDirectionData` internal dataclass in `overlays.py`
- [x] Implement `OverlayData` container dataclass with:
  - [x] `positions: list[PositionData]`
  - [x] `bodypart_sets: list[BodypartData]`
  - [x] `head_directions: list[HeadDirectionData]`
  - [x] `regions: list[str] | dict[int, list[str]] | None`
  - [x] `__post_init__()` pickle-ability check with actionable error messages

### 1.3 Timeline & Interpolation Helpers

- [x] Implement private `_build_frame_times()` function
  - [x] Accept `frame_times` (preferred) or synthesize from `fps` and `n_frames`
  - [x] Validate monotonicity
- [x] Implement `_interp_linear(t_src, x_src, t_frame)` vectorized interpolation
- [x] Implement `_interp_nearest(t_src, x_src, t_frame)` vectorized interpolation
- [x] Handle edge cases: extrapolate as NaN for out-of-bounds

### 1.4 Validation Functions (WHAT/WHY/HOW)

- [x] Implement `_validate_monotonic_time()` with actionable error
  - [x] WHAT: Non-monotonic times detected
  - [x] WHY: Interpolation requires increasing timestamps
  - [x] HOW: Sort or call `fix_monotonic_timestamps()`
- [x] Implement `_validate_finite_values()` with count and first index
  - [x] WHAT: Found NaN/Inf in overlay arrays
  - [x] WHY: Rendering cannot place invalid coordinates
  - [x] HOW: Clean or mask; suggest interpolation over gaps
- [x] Implement `_validate_shape()` with expected vs actual
  - [x] WHAT: Shape mismatch
  - [x] WHY: Coordinate dimensionality must match environment
  - [x] HOW: Project/reformat to match `env.n_dims`
- [x] Implement `_validate_temporal_alignment()` with overlap percentage
  - [x] ERROR: No overlap between `overlay.times` and `frame_times`
  - [x] WARN: Partial overlap <50% (report percentage)
  - [x] HOW: Provide overlapping time ranges or resample
- [x] Implement `_validate_bounds()` warning with percentage stats
  - [x] WARN: >X% points outside `env.dimension_ranges`
  - [x] Show min/max vs env ranges
  - [x] HOW: Confirm coordinate system and units
- [x] Implement `_validate_skeleton_consistency()` with name suggestions
  - [x] WHAT: Skeleton references missing part(s)
  - [x] WHY: Cannot draw edges without endpoints
  - [x] HOW: Fix names; suggest nearest matches (fuzzy)
- [x] Implement `_validate_pickle_ability()` with attribute details
  - [x] WHAT: OverlayData not pickle-able
  - [x] WHY: Parallel video rendering requires pickling
  - [x] HOW: Remove unpickleable obj or `n_workers=1`; call `env.clear_cache()`

### 1.5 Conversion Funnel

- [x] Implement `_convert_overlays_to_data()` function signature
  - [x] Parameters: `overlays, frame_times, n_frames, env`
  - [x] Returns: `OverlayData`
- [x] Align each `PositionOverlay` to `n_frames` using Timeline
- [x] Align each `BodypartOverlay` (per keypoint) to `n_frames`
- [x] Align each `HeadDirectionOverlay` to `n_frames`
- [x] Ensure coordinate dimension matches `env.n_dims`
- [x] Run all validation functions during conversion
- [x] Return pickle-safe `OverlayData` instance

### 1.6 Unit Tests (Core & Validation)

- [x] Create `tests/animation/test_overlays.py`
- [x] Test monotonicity validation (error case)
- [x] Test finite values validation (NaN/Inf detection)
- [x] Test shape validation (dimension mismatch)
- [x] Test temporal alignment error (no overlap)
- [x] Test temporal alignment warning (partial overlap <50%)
- [x] Test bounds warning (>X% out of bounds)
- [x] Test skeleton consistency (missing parts, name suggestions)
- [x] Test pickle-ability validation (actionable messages)
- [x] Test `_interp_linear()` with edge cases
- [x] Test `_interp_nearest()` with edge cases
- [x] Test `_convert_overlays_to_data()` end-to-end
- [x] Run tests with `uv run pytest tests/animation/test_overlays.py -v`

---

## Milestone 2: Protocol Update & Core Dispatcher (Week 2)

### 2.1 Environment Protocol Update

- [ ] Update `EnvironmentProtocol` in `src/neurospatial/environment/_protocols.py`
  - [ ] Add `overlays` parameter to `animate_fields()` signature
  - [ ] Add `frame_times: NDArray[np.float64] | None` parameter
  - [ ] Add `show_regions: bool | list[str] = False` parameter
  - [ ] Add `region_alpha: float = 0.3` parameter
  - [ ] Ensure type annotations satisfy mypy
- [ ] Update docstring with NumPy format for new parameters

### 2.2 Core Dispatcher Updates

- [ ] Update `src/neurospatial/animation/core.py` dispatcher
- [ ] Compute `n_frames` from `fields` shape
- [ ] Build/verify `frame_times` using `_build_frame_times()`
- [ ] Call `overlay_data = _convert_overlays_to_data(...)` if overlays provided
- [ ] Pass `overlay_data` to selected backend
- [ ] Update all backend routing calls to accept `overlay_data` parameter

### 2.3 Integration Tests (Core)

- [ ] Create `tests/animation/test_core_dispatcher.py`
- [ ] Test dispatcher with no overlays (backward compatibility)
- [ ] Test dispatcher with `frame_times` provided
- [ ] Test dispatcher with synthesized `frame_times` from fps
- [ ] Test `overlay_data` correctly passed to backend mock
- [ ] Run tests with `uv run pytest tests/animation/test_core_dispatcher.py -v`

---

## Milestone 3: Napari Backend (Full Overlays) (Weeks 3-4)

### 3.1 Napari Overlay Rendering

- [ ] Update `src/neurospatial/animation/backends/napari_backend.py`
- [ ] Accept `overlay_data: OverlayData | None` parameter
- [ ] Implement position overlay rendering
  - [ ] Use `add_tracks` for trail if `trail_length` specified
  - [ ] Use `add_points` for current position marker
  - [ ] Apply color and size from `PositionData`
- [ ] Implement bodypart overlay rendering
  - [ ] Use `add_points` per bodypart set with per-part colors
  - [ ] Use `add_shapes` with `shape_type="line"` for skeleton
  - [ ] Apply skeleton color and width from `BodypartData`
- [ ] Implement head direction overlay rendering
  - [ ] Use `add_vectors` for arrow display
  - [ ] Apply color and length from `HeadDirectionData`
- [ ] Implement region overlay rendering
  - [ ] Use `add_shapes` with polygons if `show_regions` is True
  - [ ] Filter regions by list if `show_regions` is a list
  - [ ] Apply `region_alpha` transparency

### 3.2 Napari Coordinate Transformation

- [ ] Implement axis order conversion: `(x, y)` → `(y, x)` for all overlays
- [ ] Ensure transformation applies to positions, bodyparts, and head directions
- [ ] Test coordinate transformation with known reference points

### 3.3 Napari Batched Updates

- [ ] Bind single callback to `viewer.dims.events.current_step`
- [ ] Batch update all overlay layers in single callback
- [ ] Update position markers and trails
- [ ] Update bodypart points and skeleton lines
- [ ] Update head direction vectors
- [ ] Target update latency < 50 ms per frame

### 3.4 Napari Tests

- [ ] Create `tests/animation/test_napari_overlays.py`
- [ ] Test position layer creation (`add_tracks` + `add_points`)
- [ ] Test bodypart layer creation (`add_points` + `add_shapes`)
- [ ] Test head direction layer creation (`add_vectors`)
- [ ] Test region layer creation (`add_shapes` with polygons)
- [ ] Test axis order conversion `(x, y)` → `(y, x)`
- [ ] Test batched update callback mechanism
- [ ] Test multiple overlays (multi-animal scenario)
- [ ] Run tests with `uv run pytest tests/animation/test_napari_overlays.py -v`

### 3.5 Napari Performance Tests

- [ ] Create `tests/animation/test_napari_performance.py` (mark as `@pytest.mark.slow`)
- [ ] Benchmark update latency with realistic pose + trail data
- [ ] Ensure update < 50 ms on standard hardware
- [ ] Profile batched vs individual layer updates
- [ ] Run with `uv run pytest -m slow tests/animation/test_napari_performance.py -v -s`

---

## Milestone 4: Video Backend (Full Overlays) (Weeks 5-6)

### 4.1 Video Overlay Rendering (Matplotlib)

- [ ] Update `src/neurospatial/animation/backends/video_backend.py`
- [ ] Accept `overlay_data: OverlayData | None` parameter in render function
- [ ] Implement position overlay rendering
  - [ ] Render trails as polylines with decaying alpha
  - [ ] Use ring buffer by `trail_length` for efficiency
  - [ ] Render current position marker with specified color/size
- [ ] Implement bodypart overlay rendering
  - [ ] Render skeleton via `LineCollection` (single call per frame)
  - [ ] Render bodypart points with per-part colors
  - [ ] Avoid per-edge loops (use vectorized approach)
- [ ] Implement head direction overlay rendering
  - [ ] Render arrows using vectorized matplotlib approach
  - [ ] Apply color and length from `HeadDirectionData`
- [ ] Implement region overlay rendering
  - [ ] Use `PathPatch` for region polygons
  - [ ] Apply alpha transparency and filtering

### 4.2 Video Parallel Safety

- [ ] Validate `OverlayData` pickle-ability before parallel rendering
- [ ] Emit clear error if pickle check fails with `n_workers > 1`
- [ ] Document requirement to call `env.clear_cache()` before parallel rendering
- [ ] Test pickle-ability of all overlay data structures

### 4.3 Video Rendering Optimization

- [ ] Use single allocations per frame for trails
- [ ] Reuse matplotlib artists where possible
- [ ] Minimize per-frame object creation
- [ ] Profile rendering overhead vs no-overlay baseline

### 4.4 Video Tests

- [ ] Create `tests/animation/test_video_overlays.py`
- [ ] Test position rendering (trail + marker)
- [ ] Test bodypart rendering (skeleton as `LineCollection`)
- [ ] Test head direction rendering (vectorized arrows)
- [ ] Test region rendering (`PathPatch`)
- [ ] Test multiple overlays (multi-animal)
- [ ] Test parallel rendering with `n_workers > 1`
- [ ] Test pickle-ability validation errors
- [ ] Run tests with `uv run pytest tests/animation/test_video_overlays.py -v`

### 4.5 Video Performance Tests

- [ ] Create `tests/animation/test_video_performance.py` (mark as `@pytest.mark.slow`)
- [ ] Benchmark rendering overhead: overlays vs no overlays
- [ ] Target overhead < 2× for typical overlay configurations
- [ ] Test parallel rendering speedup with multiple workers
- [ ] Run with `uv run pytest -m slow tests/animation/test_video_performance.py -v -s`

---

## Milestone 5: HTML & Widget Backends (Partial Overlays) (Week 7)

### 5.1 HTML Backend (Positions + Regions Only)

- [ ] Update `src/neurospatial/animation/backends/html_backend.py`
- [ ] Accept `overlay_data: OverlayData | None` parameter
- [ ] Implement client-side canvas rendering for positions
  - [ ] Serialize position data to compact JSON
  - [ ] Render position markers in JavaScript canvas
  - [ ] Implement trails with decaying opacity
- [ ] Implement region rendering in HTML canvas
  - [ ] Serialize region polygons to JSON
  - [ ] Render with alpha transparency
- [ ] Emit capability warnings when bodyparts/head direction provided
  - [ ] Clear message: "HTML backend supports positions and regions only"
  - [ ] Suggest using video or napari backend for full features
- [ ] Enforce `max_html_frames` limit (default 500)
- [ ] Auto-disable oversized overlays with user-facing warning

### 5.2 HTML File Size Guardrails

- [ ] Calculate estimated JSON size for overlay data
- [ ] Warn if overlay data exceeds reasonable size (e.g., 5MB)
- [ ] Provide suggestions for subsampling or using video backend
- [ ] Test file size limits with large datasets

### 5.3 Widget Backend (Reuse Video Renderer)

- [ ] Update `src/neurospatial/animation/backends/widget_backend.py`
- [ ] Reuse video renderer to produce PNG frames with overlays
- [ ] Implement LRU cache for responsive scrubbing
- [ ] Pass `overlay_data` to video renderer
- [ ] Test widget in notebook environment

### 5.4 HTML & Widget Tests

- [ ] Create `tests/animation/test_html_overlays.py`
- [ ] Test position rendering in HTML backend
- [ ] Test region rendering in HTML backend
- [ ] Test capability warnings for bodyparts/head direction
- [ ] Test `max_html_frames` enforcement
- [ ] Test file size warnings
- [ ] Run tests with `uv run pytest tests/animation/test_html_overlays.py -v`
- [ ] Create `tests/animation/test_widget_overlays.py`
- [ ] Test widget reuses video renderer correctly
- [ ] Test LRU cache behavior
- [ ] Run tests with `uv run pytest tests/animation/test_widget_overlays.py -v`

---

## Milestone 6: Integration & Cross-Backend Tests (Week 7)

### 6.1 Integration Tests

- [ ] Create `tests/animation/test_animation_with_overlays.py`
- [ ] Test Napari backend end-to-end with all overlay types
- [ ] Test video backend end-to-end with all overlay types
- [ ] Test HTML backend with positions + regions
- [ ] Test widget backend with overlays
- [ ] Test cross-backend consistency (same config, different backends)
- [ ] Test multi-animal scenarios (multiple overlays of same type)
- [ ] Test mixed overlay types in single animation
- [ ] Run tests with `uv run pytest tests/animation/test_animation_with_overlays.py -v`

### 6.2 Visual Regression Tests (pytest-mpl)

- [ ] Create `tests/animation/test_overlay_visual_regression.py`
- [ ] Generate golden image: position with trail
- [ ] Generate golden image: bodypart with skeleton
- [ ] Generate golden image: head direction arrows
- [ ] Generate golden image: regions with alpha
- [ ] Generate golden image: mixed overlays
- [ ] Configure pytest-mpl in `tests/conftest.py`
- [ ] Run with `uv run pytest tests/animation/test_overlay_visual_regression.py --mpl-generate-path=baseline`
- [ ] Verify visual consistency on subsequent runs

### 6.3 Backend Capability Matrix Tests

- [ ] Test each backend reports capabilities correctly
- [ ] Verify Napari supports: positions, bodyparts, head direction, regions
- [ ] Verify video supports: positions, bodyparts, head direction, regions
- [ ] Verify HTML supports: positions, regions only
- [ ] Verify widget supports: positions, bodyparts, head direction, regions
- [ ] Test warnings when unsupported overlay used with backend

---

## Milestone 7: Documentation (Week 8)

### 7.1 Overlay API Documentation

- [ ] Create `docs/animation_overlays.md`
- [ ] Add quickstart section with three overlay types
- [ ] Document multi-animal support (multiple overlay instances)
- [ ] Create backend capability comparison table
- [ ] Document HTML backend limitations clearly
- [ ] Add common errors section (WHAT/WHY/HOW format)
- [ ] Add troubleshooting guide with validation error fixes

### 7.2 Docstring Updates

- [ ] Add comprehensive NumPy docstrings to `PositionOverlay`
- [ ] Add comprehensive NumPy docstrings to `BodypartOverlay`
- [ ] Add comprehensive NumPy docstrings to `HeadDirectionOverlay`
- [ ] Add comprehensive NumPy docstrings to `animate_fields()` updated signature
- [ ] Add examples to all public overlay API docstrings
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/animation/overlays.py`

### 7.3 Example Notebook

- [ ] Create `examples/08_animation_with_overlays.ipynb`
- [ ] Example 1: Trajectory with trail overlay
- [ ] Example 2: Pose tracking with skeleton
- [ ] Example 3: Head direction visualization
- [ ] Example 4: Multi-animal tracking (multiple overlays)
- [ ] Example 5: Regions overlay with spatial fields
- [ ] Example 6: Mixed-rate alignment using `frame_times`
- [ ] Example 7: Backend comparison (same data, different backends)
- [ ] Test notebook executes without errors

### 7.4 CLAUDE.md Updates

- [ ] Add overlay examples to "Most Common Patterns" section
- [ ] Document `frame_times` parameter usage
- [ ] Document `show_regions` parameter
- [ ] Add multi-animal pattern
- [ ] Update "Key Implementation Notes" with overlay architecture
- [ ] Add overlay troubleshooting to "Common Gotchas"
- [ ] Update "Troubleshooting" section with validation errors

### 7.5 CHANGELOG Update

- [ ] Create v0.4.0 section in CHANGELOG.md
- [ ] Document new overlay dataclasses API
- [ ] Document `animate_fields()` parameter additions
- [ ] List backend-specific overlay support
- [ ] Note HTML backend limitations
- [ ] Credit contributors

---

## Milestone 8: Final Quality Assurance (Week 8)

### 8.1 Type Checking

- [ ] Run mypy on overlay module: `uv run mypy src/neurospatial/animation/overlays.py`
- [ ] Run mypy on core module: `uv run mypy src/neurospatial/animation/core.py`
- [ ] Run mypy on all backends: `uv run mypy src/neurospatial/animation/backends/`
- [ ] Fix all mypy errors and warnings
- [ ] Ensure no `type: ignore` comments without justification

### 8.2 Code Quality

- [ ] Run ruff check: `uv run ruff check src/neurospatial/animation/`
- [ ] Run ruff format: `uv run ruff format src/neurospatial/animation/`
- [ ] Review all new code for NumPy docstring compliance
- [ ] Ensure all validation errors follow WHAT/WHY/HOW format
- [ ] Check for proper error messages with actionable fixes

### 8.3 Test Coverage

- [ ] Run full test suite: `uv run pytest`
- [ ] Run with coverage: `uv run pytest --cov=src/neurospatial/animation/`
- [ ] Ensure coverage >80% for overlay module
- [ ] Ensure coverage >80% for backend overlay rendering
- [ ] Run slow tests: `uv run pytest -m slow -v -s`

### 8.4 Performance Benchmarks

- [ ] Create `tests/animation/test_performance.py` baseline
- [ ] Benchmark Napari update latency (<50ms target)
- [ ] Benchmark video rendering overhead (<2× target)
- [ ] Document performance results in `docs/animation_overlays.md`
- [ ] Profile memory usage with large overlay datasets

### 8.5 Pre-commit Hooks

- [ ] Test pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Fix any ruff violations
- [ ] Fix any mypy violations
- [ ] Ensure conventional commit messages used

### 8.6 Final Integration Test

- [ ] Create real-world test scenario with all features
- [ ] Test complete workflow: create env → compute fields → animate with overlays
- [ ] Test all four backends with identical overlay configuration
- [ ] Verify visual consistency across backends (where supported)
- [ ] Test with large dataset (1000+ frames)
- [ ] Test with multi-animal scenario (3+ animals)

---

## Milestone 9: Release Preparation (Post-Implementation)

### 9.1 Version Updates

- [ ] Update version in `pyproject.toml` to 0.4.0
- [ ] Update version in `src/neurospatial/__init__.py` if present
- [ ] Update "Last Updated" date in CLAUDE.md

### 9.2 Documentation Review

- [ ] Review all documentation for accuracy
- [ ] Ensure examples run successfully
- [ ] Check all links in documentation
- [ ] Verify API reference is complete
- [ ] Proofread CHANGELOG entry

### 9.3 Git & Release

- [ ] Create feature branch: `git checkout -b feat/animation-overlays-v0.4.0`
- [ ] Commit with conventional commits format
- [ ] Push branch and create pull request
- [ ] Address code review feedback
- [ ] Merge to main after approval
- [ ] Tag release: `git tag v0.4.0`
- [ ] Push tag: `git push origin v0.4.0`

---

## Notes for Implementers

### Development Workflow

1. **Always use `uv run`** for all commands (pytest, python, mypy, ruff)
2. **Run from project root** for all commands
3. **Use NumPy docstring format** for all new code
4. **Follow conventional commits**: `feat(animation): add PositionOverlay dataclass`
5. **Test incrementally** as you implement each component
6. **Use `@pytest.mark.slow`** for performance benchmarks

### Key Design Principles

- **Protocol-based**: Overlay dataclasses are the user API (no inheritance)
- **Immutable where possible**: Dataclasses should use `frozen=True` where appropriate
- **Validation is helpful**: All errors follow WHAT/WHY/HOW format
- **Backends are independent**: Each backend imports only `OverlayData`
- **Timeline is single source of truth**: All alignment goes through Timeline helpers
- **Pickle-safety required**: For parallel video rendering

### Testing Strategy

- **Unit tests first**: Validate each component in isolation
- **Integration tests**: Test full workflows across backends
- **Visual regression**: Use pytest-mpl for rendering consistency
- **Performance tests**: Separate from unit tests, marked as `@pytest.mark.slow`
- **Always test multi-animal**: Multiple overlays of same type

### Common Pitfalls to Avoid

1. Don't introduce another public overlay protocol
2. Don't modify regions in v0.4.0 (keep as flag)
3. Don't capture closures in parallel worker payloads
4. Don't skip validation - it saves users time
5. Don't forget axis conversion for Napari: `(x, y)` → `(y, x)`
6. Don't use loops for skeleton rendering - use `LineCollection`

### Performance Targets

- Napari update latency: **< 50 ms/frame**
- Video rendering overhead: **< 2× vs no overlays**
- HTML file size: **< 10 MB for 500 frames**
- Memory usage: **Reasonable for 10,000+ frames**

---

## Progress Tracking

**Milestone Completion:**

- [x] Milestone 1: Core Infrastructure & Validation (Weeks 1-2)
- [ ] Milestone 2: Protocol Update & Core Dispatcher (Week 2)
- [ ] Milestone 3: Napari Backend (Full Overlays) (Weeks 3-4)
- [ ] Milestone 4: Video Backend (Full Overlays) (Weeks 5-6)
- [ ] Milestone 5: HTML & Widget Backends (Partial Overlays) (Week 7)
- [ ] Milestone 6: Integration & Cross-Backend Tests (Week 7)
- [ ] Milestone 7: Documentation (Week 8)
- [ ] Milestone 8: Final Quality Assurance (Week 8)
- [ ] Milestone 9: Release Preparation (Post-Implementation)

**Overall Progress:** 11% (1/9 milestones completed)

---

*Last Updated:* 2025-11-20
*Target Release:* v0.4.0 Animation Overlays Feature
