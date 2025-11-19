# Animation Feature Implementation Tasks

**Feature:** Multi-backend spatial field animation for neuroscience data
**Plan Document:** [ANIMATION_IMPLEMENTATION_PLAN.md](ANIMATION_IMPLEMENTATION_PLAN.md)
**Estimated Timeline:** 4-5 weeks

---

## Overview

Implement animation capabilities supporting four backends:

1. **Napari** - GPU-accelerated interactive viewer (900K+ frames)
2. **Video (MP4)** - Parallel rendering with ffmpeg
3. **HTML** - Standalone player with instant scrubbing
4. **Jupyter Widget** - Notebook integration

**Key Challenges:**

- Memory-efficient handling of large-scale sessions (900K frames at 250 Hz)
- Parallel frame rendering with matplotlib constraints
- Colormap normalization across frames
- Pickle-ability validation for parallel workers

---

## Milestone 1: Core Infrastructure

**Goal:** Build foundation - rendering utilities, core dispatcher, and subsample function
**Dependencies:** None
**Estimated Time:** 3-4 days

### Module Structure

- [x] Create `src/neurospatial/animation/` directory
- [x] Create `src/neurospatial/animation/__init__.py`
- [x] Create `src/neurospatial/animation/backends/` directory
- [x] Create `src/neurospatial/animation/backends/__init__.py`

### Rendering Utilities (`rendering.py`)

- [x] Create `src/neurospatial/animation/rendering.py`
- [x] Implement `compute_global_colormap_range()` (single-pass optimization)
- [x] Implement `render_field_to_rgb()` (matplotlib → RGB array)
- [x] Implement `render_field_to_png_bytes()` (for HTML embedding)
- [x] Implement `field_to_rgb_for_napari()` (fast colormap lookup)
- [x] Add NumPy docstrings for all functions
- [x] Write unit tests for rendering utilities
  - [x] Test colormap range with edge cases (all same values, partial overrides)
  - [x] Test RGB conversion accuracy
  - [x] Test PNG byte encoding
  - [x] Test field shape validation (added in code review)

### Core Dispatcher (`core.py`)

- [x] Create `src/neurospatial/animation/core.py`
- [x] Implement `animate_fields()` dispatcher function
  - [x] Add environment fitted state validation
  - [x] Add field shape validation (must match env.n_bins)
  - [x] Add early ffmpeg availability check (fail fast)
  - [x] Add pickle-ability validation for parallel rendering
- [x] Implement `_select_backend()` with transparent logging
  - [x] Add file extension detection (.mp4, .html)
  - [x] Add large dataset detection (>10K frames → Napari)
  - [x] Add Jupyter detection (use widget backend)
  - [x] Add clear error messages when no backend available
- [x] Implement `subsample_frames()` utility function
  - [x] Support both ndarray and list inputs
  - [x] Preserve input type in output
  - [x] Add validation (target_fps ≤ source_fps)
- [x] Add unit tests for core functions
  - [x] Test backend auto-selection logic
  - [x] Test subsample_frames with memory-mapped arrays
  - [x] Test pickle validation error messages

### Public API Export

- [x] Update `src/neurospatial/animation/__init__.py`
  - [x] Export `subsample_frames`
  - [x] Add `__all__` list
- [x] Verify import works: `from neurospatial.animation import subsample_frames`

### Dependencies

- [x] Update `pyproject.toml` with optional dependencies

  ```toml
  [project.optional-dependencies]
  animation = [
      "napari[all]>=0.4.18,<0.6",
      "ipywidgets>=8.0,<9.0",
  ]
  ```

- [x] Document ffmpeg as system dependency (user installs)

### Type Checking

- [x] Run `uv run mypy src/neurospatial/animation/core.py`
- [x] Run `uv run mypy src/neurospatial/animation/rendering.py`
- [x] Fix any type errors (all passing)

---

## Milestone 2: HTML Backend (MVP)

**Goal:** First complete end-to-end backend (simplest, no dependencies)
**Dependencies:** Milestone 1
**Estimated Time:** 2-3 days

### Implementation

- [ ] Create `src/neurospatial/animation/backends/html_backend.py`
- [ ] Implement `render_html()` function
  - [ ] Add file size estimation BEFORE rendering
  - [ ] Add hard limit check (500 frames max)
  - [ ] Add warning for files >50MB
  - [ ] Add base64 frame encoding with progress bar
  - [ ] Add frame label generation
- [ ] Implement `_generate_html_player()` function
  - [ ] Embed frames as base64 data URLs
  - [ ] Add JavaScript play/pause controls
  - [ ] Add slider for scrubbing
  - [ ] Add keyboard shortcuts (space, arrows)
  - [ ] Add ARIA labels for accessibility
  - [ ] Add frame label display

### Testing

- [ ] Write unit tests (`tests/animation/test_html_backend.py`)
  - [ ] Test file size limit enforcement
  - [ ] Test warning for large files
  - [ ] Test HTML generation (valid markup)
  - [ ] Test with 10-50 frames (realistic size)
- [ ] Manual test: Create 20-frame HTML, open in browser
- [ ] Verify controls work (play, pause, scrub, keyboard)

### Integration Test

- [ ] Create simple end-to-end test

  ```python
  env = Environment.from_samples(positions, bin_size=5.0)
  fields = [np.random.rand(env.n_bins) for _ in range(10)]
  path = env.animate_fields(fields, save_path='test.html')
  assert path.exists()
  ```

---

## Milestone 3: Video Backend (Parallel Rendering)

**Goal:** Implement parallel video export with ffmpeg
**Dependencies:** Milestone 1
**Estimated Time:** 3-4 days

### Implementation

- [ ] Create `src/neurospatial/animation/backends/video_backend.py`
- [ ] Implement `check_ffmpeg_available()` function
- [ ] Implement `render_video()` function
  - [ ] Add dry_run mode (estimate time/size without rendering)
  - [ ] Add progress estimates
  - [ ] Add codec selection (h264, vp9, mpeg4)
  - [ ] Add temporary directory creation/cleanup
  - [ ] Add ffmpeg encoding command
- [ ] Create `src/neurospatial/animation/_parallel.py`
- [ ] Implement `parallel_render_frames()` function
  - [ ] Partition frames across workers
  - [ ] Create worker task dictionaries
  - [ ] Use ProcessPoolExecutor for parallelism
  - [ ] Add progress bar (tqdm)
- [ ] Implement `_render_worker_frames()` function
  - [ ] Create matplotlib figure per worker
  - [ ] Render frames to PNG files
  - [ ] Add frame labels to titles
  - [ ] Add finally block for cleanup (prevent memory leaks)

### Testing

- [ ] Write unit tests (`tests/animation/test_video_backend.py`)
  - [ ] Test ffmpeg availability check
  - [ ] Test dry_run mode
  - [ ] Test frame partitioning logic
  - [ ] Mock ProcessPoolExecutor for unit tests
- [ ] Integration test with actual ffmpeg
  - [ ] Skip if ffmpeg not available
  - [ ] Test with n_workers=1 (serial)
  - [ ] Test with n_workers=2 (parallel)
  - [ ] Verify output video plays

### Pickle Validation

- [ ] Test environment pickle-ability

  ```python
  env = Environment.from_samples(positions, bin_size=5.0)
  env.clear_cache()  # Ensure pickle-able
  import pickle
  pickle.dumps(env)  # Should succeed
  ```

- [ ] Test error message when pickle fails
- [ ] Document in CLAUDE.md: "Call env.clear_cache() before parallel rendering"

---

## Milestone 4: Napari Backend (Interactive Viewer)

**Goal:** GPU-accelerated viewer with lazy loading for large datasets
**Dependencies:** Milestone 1
**Estimated Time:** 2-3 days

### Implementation

- [ ] Create `src/neurospatial/animation/backends/napari_backend.py`
- [ ] Add napari availability check

  ```python
  try:
      import napari
      NAPARI_AVAILABLE = True
  except ImportError:
      NAPARI_AVAILABLE = False
  ```

- [ ] Implement `render_napari()` function
  - [ ] Compute global colormap range
  - [ ] Pre-compute colormap lookup table (256 RGB values)
  - [ ] Create LazyFieldRenderer class with true LRU cache
  - [ ] Add napari.Viewer creation
  - [ ] Add image layer with RGB data
  - [ ] Add trajectory overlay support (2D tracks)
- [ ] Implement `LazyFieldRenderer` class
  - [ ] Use OrderedDict for LRU cache
  - [ ] Implement `__getitem__` with cache check
  - [ ] Implement `move_to_end()` for LRU updates
  - [ ] Implement `popitem(last=False)` for eviction
  - [ ] Add `shape` and `dtype` properties

### Testing

- [ ] Write unit tests (`tests/animation/test_napari_backend.py`)
  - [ ] Mock napari if not available
  - [ ] Test LazyFieldRenderer cache behavior
  - [ ] Test LRU eviction (access order matters)
  - [ ] Skip napari viewer tests in CI (no display)
- [ ] Manual test: Launch viewer with 1000 frames
- [ ] Verify seeking performance (<100ms)
- [ ] Test with memory-mapped arrays

### CI Configuration

- [ ] Update `.github/workflows/tests.yml`

  ```yaml
  - name: Run tests (skip napari)
    run: uv run pytest -m "not napari"
  ```

- [ ] Add `@pytest.mark.napari` to napari tests

---

## Milestone 5: Jupyter Widget Backend

**Goal:** Notebook integration with play/pause controls
**Dependencies:** Milestone 1
**Estimated Time:** 1-2 days

### Implementation

- [ ] Create `src/neurospatial/animation/backends/widget_backend.py`
- [ ] Add ipywidgets availability check
- [ ] Implement `render_widget()` function
  - [ ] Pre-render subset of frames (first 500)
  - [ ] Create on-demand rendering for remaining frames
  - [ ] Implement `show_frame()` callback
  - [ ] Create ipywidgets.IntSlider
  - [ ] Add HTML display with base64 images
  - [ ] Return interactive widget

### Testing

- [ ] Write unit tests (`tests/animation/test_widget_backend.py`)
  - [ ] Mock ipywidgets if not available
  - [ ] Test frame caching logic
  - [ ] Test on-demand rendering fallback
- [ ] Manual test in Jupyter notebook
  - [ ] Verify slider works
  - [ ] Verify frame labels display
  - [ ] Test with 50-100 frames

---

## Milestone 6: Environment Integration

**Goal:** Add animate_fields() method to Environment
**Dependencies:** Milestones 1-5
**Estimated Time:** 1 day

### Mixin Implementation

- [ ] Open `src/neurospatial/environment/visualization.py`
- [ ] Add `animate_fields()` method to EnvironmentVisualization mixin
  - [ ] Use `self: EnvironmentProtocol` type annotation
  - [ ] Add complete parameter list (all backends)
  - [ ] Add comprehensive NumPy docstring
  - [ ] Delegate to `neurospatial.animation.core.animate_fields()`
- [ ] Add method signature to EnvironmentProtocol
  - [ ] Open `src/neurospatial/environment/_protocols.py`
  - [ ] Add `animate_fields` method stub

### Type Checking

- [ ] Run `uv run mypy src/neurospatial/environment/visualization.py`
- [ ] Fix any type errors
- [ ] Verify mixin pattern works with Protocol

### Testing

- [ ] Test method exists on Environment

  ```python
  env = Environment.from_samples(positions, bin_size=5.0)
  assert hasattr(env, 'animate_fields')
  ```

- [ ] Test delegation works (call through Environment)

---

## Milestone 7: Examples and Documentation

**Goal:** User-facing examples and documentation
**Dependencies:** Milestone 6
**Estimated Time:** 2 days

### Example Script

- [ ] Create `examples/08_field_animation.py`
- [ ] Add Example 1: Napari interactive viewer
- [ ] Add Example 2: Video export (MP4)
- [ ] Add Example 3: HTML player
- [ ] Add Example 4: Jupyter widget
- [ ] Add Example 5: Large-scale session (900K frames)
  - [ ] Memory-mapped array example
  - [ ] Napari lazy loading
  - [ ] Subsample for video export
  - [ ] Dry run estimation
- [ ] Run all examples: `uv run python examples/08_field_animation.py`
- [ ] Verify all outputs generated

### Documentation Updates

- [ ] Update `CLAUDE.md`
  - [ ] Add animation section to "Import Patterns"
  - [ ] Document `subsample_frames` import
  - [ ] Add pickle-ability requirement
  - [ ] Add example usage patterns
- [ ] Create `docs/user-guide/animation.md`
  - [ ] Quick start (5 lines)
  - [ ] Backend comparison table
  - [ ] Remote server workflow
  - [ ] Large-scale data guide (memory-mapped arrays)
  - [ ] Troubleshooting section

### README Updates

- [ ] Add animation feature to main README
- [ ] Create example GIF/video demonstrating each backend
- [ ] Add installation instructions for optional dependencies

---

## Milestone 8: Testing and Polish

**Goal:** Comprehensive tests, benchmarks, error handling
**Dependencies:** Milestone 7
**Estimated Time:** 2-3 days

### Unit Tests

- [ ] Test `subsample_frames()` with arrays and lists
- [ ] Test pickle-ability validation
- [ ] Test field shape validation
- [ ] Test HTML file size limits
- [ ] Test dry_run mode
- [ ] Run all tests: `uv run pytest tests/animation/`
- [ ] Achieve >90% coverage: `uv run pytest --cov=src/neurospatial/animation`

### Integration Tests

- [ ] Test with memory-mapped arrays (large-scale)

  ```python
  fields = np.memmap('test.dat', dtype='float32', mode='w+',
                     shape=(100000, env.n_bins))
  ```

- [ ] Test all backends with same data
- [ ] Test backend auto-selection logic
- [ ] Test error messages (missing dependencies)

### Performance Benchmarks

- [ ] Create `tests/animation/test_benchmarks.py`
- [ ] Benchmark Napari seek performance
  - [ ] Target: <100ms for 100K frames
- [ ] Benchmark parallel rendering scalability
  - [ ] Test 1, 2, 4, 8 workers
- [ ] Benchmark HTML generation
  - [ ] Target: <20s for 100 frames

### Memory Profiling

- [ ] Profile memory usage for large datasets
- [ ] Verify Napari lazy loading doesn't load all frames
- [ ] Verify parallel rendering cleans up properly
- [ ] Document memory requirements in docs

### Error Message Review

- [ ] Review all error messages for clarity
- [ ] Ensure all suggest actionable solutions
- [ ] Test missing dependency errors
  - [ ] napari not installed
  - [ ] ipywidgets not installed
  - [ ] ffmpeg not installed
- [ ] Test validation errors
  - [ ] Environment not fitted
  - [ ] Field shape mismatch
  - [ ] Pickle failure

### Type Checking (Final)

- [ ] Run `uv run mypy src/neurospatial/animation/`
- [ ] Run `uv run mypy src/neurospatial/environment/visualization.py`
- [ ] Fix all type errors
- [ ] Verify no `type: ignore` comments needed

### Code Quality

- [ ] Run `uv run ruff check src/neurospatial/animation/`
- [ ] Run `uv run ruff format src/neurospatial/animation/`
- [ ] Fix all linting issues
- [ ] Review all docstrings (NumPy format)

---

## Milestone 9: Final Review and Release Prep

**Goal:** Code review, documentation polish, prepare for merge
**Dependencies:** Milestone 8
**Estimated Time:** 1-2 days

### Code Review

- [ ] Self-review all code changes
- [ ] Check for TODO/FIXME comments
- [ ] Verify all functions have docstrings
- [ ] Verify all tests pass: `uv run pytest`
- [ ] Verify type checking passes: `uv run mypy src/neurospatial/`

### Documentation Review

- [ ] Proofread all documentation
- [ ] Verify all code examples run
- [ ] Check all links work
- [ ] Review API reference completeness

### User Testing

- [ ] Test on clean environment

  ```bash
  uv venv test-env
  source test-env/bin/activate
  uv pip install -e .
  ```

- [ ] Run examples without optional dependencies
- [ ] Verify error messages are helpful
- [ ] Install optional dependencies one-by-one

  ```bash
  uv add "napari[all]>=0.4.18,<0.6"
  uv add "ipywidgets>=8.0,<9.0"
  ```

### Git and Commits

- [ ] Review git status: `git status`
- [ ] Stage changes: `git add src/neurospatial/animation/ tests/animation/`
- [ ] Create commit with proper message:

  ```
  feat(animation): add multi-backend field animation

  Implements four animation backends for visualizing spatial fields over time:
  - Napari: GPU-accelerated interactive viewer (900K+ frames)
  - Video: Parallel MP4 export with ffmpeg
  - HTML: Standalone player with instant scrubbing
  - Jupyter Widget: Notebook integration

  Includes subsample_frames() utility for large-scale sessions.

  🤖 Generated with [Claude Code](https://claude.com/claude-code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

### Success Criteria Verification

- [ ] ✅ All four backends functional
- [ ] ✅ Napari handles 100K+ frames efficiently
- [ ] ✅ Video export uses parallel rendering correctly
- [ ] ✅ HTML player works in all modern browsers
- [ ] ✅ Clear error messages for missing dependencies
- [ ] ✅ Napari seek time <100ms for 100K frames
- [ ] ✅ Video renders 100 frames in <30s (4-core)
- [ ] ✅ HTML generates 100-frame player in <20s

### Final Checklist

- [ ] All tests passing
- [ ] All type checks passing
- [ ] All examples running
- [ ] Documentation complete
- [ ] No regression in existing functionality
- [ ] Ready for pull request/merge

---

## Notes

**Testing Commands:**

```bash
# Run all tests
uv run pytest

# Run animation tests only
uv run pytest tests/animation/

# Run with coverage
uv run pytest --cov=src/neurospatial/animation

# Run type checking
uv run mypy src/neurospatial/animation/

# Run linting
uv run ruff check src/neurospatial/animation/

# Run formatting
uv run ruff format src/neurospatial/animation/
```

**Common Issues:**

- **Pickle errors:** Call `env.clear_cache()` before parallel rendering
- **Napari not showing:** Requires Qt/display; skip in CI
- **ffmpeg missing:** Video backend requires system install
- **Memory issues:** Use memory-mapped arrays for >100K frames

**Reference Documents:**

- [ANIMATION_IMPLEMENTATION_PLAN.md](ANIMATION_IMPLEMENTATION_PLAN.md) - Complete implementation specification
- [ANIMATION_PLAN_REVISIONS.md](ANIMATION_PLAN_REVISIONS.md) - Review feedback and fixes
- [CLAUDE.md](CLAUDE.md) - Project conventions and patterns
