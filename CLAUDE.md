# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2025-12-05 (v0.3.0 - Added egocentric frames, object-vector cells, spatial view cells)

---

## 🎯 Critical Rules (MUST Follow)

When working with this codebase, you MUST follow these rules:

1. **ALWAYS use `uv run`** before Python commands - never use bare `python`, `pip`, or `pytest`
2. **NEVER create bare `Environment()`** - always use factory methods like `Environment.from_samples()`
3. **bin_size is REQUIRED** - all Environment creation needs explicit bin_size parameter
4. **NumPy docstring format** - all docstrings must follow NumPy style (not Google or reST)
5. **Check `is_1d` before linearization** - only 1D environments have `to_linear()` method
6. **Regions are immutable** - use `env.regions.update_region()`, never modify in place
7. **Use `@check_fitted` decorator** - methods requiring fitted state must use this decorator
8. **Egocentric angles use animal-centered convention** - 0=ahead, π/2=left, -π/2=right (NOT allocentric 0=East)

---

## 📦 Package Management

**This project uses `uv` (not pip or conda).**

- Python version: 3.13 (specified in `.python-version`)
- Virtual environment: Automatically managed by `uv`
- Dependencies: Defined in `pyproject.toml`

### Essential Commands

```bash
# Run all tests (most common)
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_function_name -v

# Lint and format
uv run ruff check . && uv run ruff format .

# Type checking
uv run mypy src/neurospatial/

# Add dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

**Why uv?** Automatically manages virtual environment without manual activation.

---

## 🚀 Most Common Patterns (90% of tasks)

### 1. Create Environment from Data

```python
from neurospatial import Environment
import numpy as np

# Generate sample position data
positions = np.random.rand(100, 2) * 100  # 100 points in 2D

# Create environment (bin_size is REQUIRED)
env = Environment.from_samples(positions, bin_size=2.0)

# Set metadata (recommended)
env.units = "cm"
env.frame = "session1"

# Query the environment
bin_idx = env.bin_at([50.0, 50.0])
neighbors = env.neighbors(bin_idx)
```

**Need different layout?** See [QUICKSTART.md - Environment Creation](.claude/QUICKSTART.md#environment-creation)

### 2. Compute Place Fields

```python
from neurospatial import compute_place_field

# Compute place field for one neuron
firing_rate = compute_place_field(
    env, spike_times, times, positions,
    method="diffusion_kde",  # Default: graph-based boundary-aware KDE
    bandwidth=5.0  # Smoothing bandwidth (cm)
)
# Methods: "diffusion_kde" (default), "gaussian_kde", "binned" (legacy)
```

**Need decoding?** See [QUICKSTART.md - Bayesian Decoding](.claude/QUICKSTART.md#neural-analysis)

### 3. Animate Spatial Fields

```python
# IMPORTANT: frame_times is REQUIRED
frame_times = np.arange(len(fields)) / 30.0  # 30 Hz timestamps

# Interactive napari viewer
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Video export with parallel rendering (requires ffmpeg)
env.clear_cache()  # Required before parallel rendering
env.animate_fields(
    fields, frame_times=frame_times, speed=1.0,
    backend="video", save_path="animation.mp4", n_workers=4
)
```

**Need overlays?** See [QUICKSTART.md - Animation](.claude/QUICKSTART.md#visualization--animation)

### 4. Add Trajectory Overlays

```python
from neurospatial import PositionOverlay

# Position overlay with trail
position_overlay = PositionOverlay(
    data=trajectory,  # Shape: (n_frames, n_dims) in environment (x, y) coordinates
    color="red",
    size=12.0,
    trail_length=10  # Show last 10 frames as decaying trail
)
env.animate_fields(fields, frame_times=frame_times, overlays=[position_overlay])
```

**Need pose tracking or events?** See [QUICKSTART.md - Overlays](.claude/QUICKSTART.md#visualization--animation)

### 5. Compute Peri-Event Histogram (PSTH)

```python
from neurospatial import peri_event_histogram

# Compute PSTH around reward events
result = peri_event_histogram(
    spike_times, reward_times,
    window=(-1.0, 2.0),  # -1s before to 2s after event
    bin_size=0.025,      # 25 ms bins
)

# Access results
firing_rates = result.firing_rate()  # Convert to Hz
print(f"Peak: {firing_rates.max():.1f} Hz at {result.bin_centers[firing_rates.argmax()]:.2f}s")
```

**Need GLM regressors?** See [QUICKSTART.md - Events](.claude/QUICKSTART.md#events-and-peri-event-analysis)

### 6. Save and Load Environments

```python
# Save environment (creates .json + .npz files)
env.to_file("my_environment")

# Load environment
from neurospatial import Environment
loaded_env = Environment.from_file("my_environment")
```

**Need NWB integration?** See [ADVANCED.md - NWB Integration](.claude/ADVANCED.md#nwb-integration-v070)

### 7. Compute Egocentric Bearing and Distance

```python
from neurospatial import (
    heading_from_velocity,
    compute_egocentric_bearing,
    compute_egocentric_distance,
)

# Compute heading from movement direction (min_speed in cm/s)
headings = heading_from_velocity(positions, times, min_speed=5.0)  # cm/s

# Compute egocentric bearing to objects (0=ahead, π/2=left, -π/2=right)
object_positions = np.array([[50, 50], [75, 25]])  # 2 objects, coordinates in cm
bearings = compute_egocentric_bearing(positions, headings, object_positions)

# Compute egocentric distance (Euclidean or geodesic)
distances = compute_egocentric_distance(
    positions, object_positions, metric="euclidean"
)
```

**Need body orientation?** Use `heading_from_body_orientation(nose_pos, tail_pos)`

### 8. Compute Object-Vector Field

```python
from neurospatial import compute_object_vector_field

# Compute firing field in egocentric polar coordinates
result = compute_object_vector_field(
    spike_times, times, positions, headings, object_positions,
    distance_range=(0, 50),  # cm from object
    angle_range=(-np.pi, np.pi),  # full circle
    distance_bin_size=5.0,  # 5 cm bins
    angle_bin_size=np.pi/12,  # 15° bins
)
# result.field: firing rate in egocentric polar bins
# result.ego_env: the egocentric polar environment
# result.occupancy: time spent in each bin
```

**Need metrics?** See [QUICKSTART.md - Object-Vector Cells](.claude/QUICKSTART.md#object-vector-cells)

### 9. Compute Spatial View Field

```python
from neurospatial import compute_spatial_view_field

# Compute firing field indexed by VIEWED location (not animal position)
result = compute_spatial_view_field(
    env, spike_times, times, positions, headings,
    view_distance=20.0,  # cm ahead
    gaze_model="fixed_distance",  # or "ray_cast", "boundary"
)
# result.field: firing rate at each spatial bin (indexed by where animal looked)
# result.view_occupancy: time spent viewing each bin
```

**Need classification?** See [QUICKSTART.md - Spatial View Cells](.claude/QUICKSTART.md#spatial-view-cells)

---

## ⚠️ Common Gotchas (Fix These First)

### Gotcha 1: Always use `uv run`

❌ **Wrong:**

```bash
python script.py
pytest
pip install package
```

✅ **Right:**

```bash
uv run python script.py
uv run pytest
uv add package
```

### Gotcha 2: Check `_is_fitted` state

❌ **Wrong:**

```python
env = Environment()  # Not fitted!
env.bin_at([10.0, 5.0])  # RuntimeError
```

✅ **Right:**

```python
env = Environment.from_samples(positions, bin_size=2.0)
env.bin_at([10.0, 5.0])  # Works
```

### Gotcha 3: bin_size is required

❌ **Wrong:**

```python
env = Environment.from_samples(data)  # TypeError
```

✅ **Right:**

```python
env = Environment.from_samples(positions, bin_size=2.0)
```

### Gotcha 4: Regions are immutable

❌ **Wrong:**

```python
env.regions['goal'].point = new_point  # AttributeError
```

✅ **Right:**

```python
env.regions.update_region('goal', point=new_point)  # No warning
```

### Gotcha 5: Check `is_1d` before linearization

❌ **Wrong:**

```python
env = Environment.from_samples(positions, bin_size=2.0)  # 2D grid
linear_pos = env.to_linear(position)  # AttributeError
```

✅ **Right:**

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    bin_idx = env.bin_at(position)
```

**More gotchas?** See [TROUBLESHOOTING.md - Common Gotchas](.claude/TROUBLESHOOTING.md#common-gotchas)

---

## 🔧 When Things Break

### Error: `ModuleNotFoundError: No module named 'neurospatial'`

```bash
uv sync  # From project root
uv run python -c "import neurospatial; print(neurospatial.__file__)"
```

### Error: `RuntimeError: Environment must be fitted before calling this method`

Use factory methods:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Not: Environment()
```

### Error: `ValueError: No active bins found`

Read the detailed error message - it provides diagnostics. Common fixes:

```python
# Reduce bin_size
env = Environment.from_samples(positions, bin_size=1.0)  # Was 10.0

# Lower threshold
env = Environment.from_samples(positions, bin_size=2.0, bin_count_threshold=1)

# Enable morphological operations
env = Environment.from_samples(positions, bin_size=2.0, dilate=True, fill_holes=True)
```

**More errors?** See [TROUBLESHOOTING.md - Error Messages](.claude/TROUBLESHOOTING.md#error-messages)

---

## 🏗️ Architecture Overview

**neurospatial** uses a three-layer architecture:

1. **Layout Engines** (`src/neurospatial/layout/`)
   - Protocol-based design with `LayoutEngine` interface
   - Available engines: RegularGrid, Hexagonal, Graph (1D), Masked, ImageMask, ShapelyPolygon, TriangularMesh
   - All engines produce: `bin_centers`, `connectivity` graph, `dimension_ranges`

2. **Environment** (`src/neurospatial/environment/`)
   - Main user-facing class using **mixin pattern** for 6,000+ lines of functionality
   - Mixins: core, factories, queries, trajectory, transforms, fields, metrics, regions, serialization, visualization
   - Factory methods: `from_samples()`, `from_graph()`, `from_polygon()`, `from_mask()`, `from_image()`, `from_polar_egocentric()`

3. **Regions** (`src/neurospatial/regions/`)
   - Immutable `Region` dataclass (points or polygons)
   - `Regions` container with dict-like interface
   - JSON serialization with versioned schema

4. **Cell Type Modules**
   - **Reference Frames** (`reference_frames.py`): Allocentric↔egocentric transforms, heading computation
   - **Object-Vector Cells** (`object_vector_field.py`, `metrics/object_vector_cells.py`): Simulation, tuning analysis
   - **Spatial View Cells** (`spatial_view_field.py`, `visibility.py`, `metrics/spatial_view_cells.py`): Gaze, visibility, view fields

**Need architecture details?** See [ARCHITECTURE.md](.claude/ARCHITECTURE.md)

---

## 📚 Documentation Structure

This documentation is organized into focused modules:

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[QUICKSTART.md](.claude/QUICKSTART.md)** | Essential patterns | Start here - copy-paste examples |
| **[API_REFERENCE.md](.claude/API_REFERENCE.md)** | Import patterns | When you need an import statement |
| **[DEVELOPMENT.md](.claude/DEVELOPMENT.md)** | Commands & workflow | When running tests or committing |
| **[PATTERNS.md](.claude/PATTERNS.md)** | Design patterns | When extending the codebase |
| **[TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md)** | Errors & fixes | When something breaks |
| **[ADVANCED.md](.claude/ADVANCED.md)** | NWB, video overlays | When using advanced features |
| **[ARCHITECTURE.md](.claude/ARCHITECTURE.md)** | Core design | When understanding internals |
| **[PROFILING.md](.claude/PROFILING.md)** | Performance profiling | When optimizing napari animations |

**Total documentation:** ~2,400 lines across 8 files
**Token reduction:** ~60-70% per conversation (only load what's needed)

---

## 🎓 Learning Path

### Beginner (first time using neurospatial)

1. Read "Most Common Patterns" above
2. Try [QUICKSTART.md - Your First Environment](.claude/QUICKSTART.md#your-first-environment)
3. Reference [TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md) when stuck

### Intermediate (extending or modifying code)

1. Understand [ARCHITECTURE.md - Three-Layer Design](.claude/ARCHITECTURE.md#three-layer-design)
2. Learn [PATTERNS.md - Mixin Pattern](.claude/PATTERNS.md#mixin-pattern-for-environment)
3. Follow [DEVELOPMENT.md - Testing](.claude/DEVELOPMENT.md#testing)

### Advanced (architecting features)

1. Master [PATTERNS.md - All Patterns](.claude/PATTERNS.md)
2. Read [ADVANCED.md - NWB Integration](.claude/ADVANCED.md#nwb-integration-v070)
3. Study [DEVELOPMENT.md - Full Workflow](.claude/DEVELOPMENT.md)

---

## 🔍 Quick Navigation

### By Task Type

| I want to... | Go to... |
|--------------|----------|
| Create first environment | "Most Common Patterns" above |
| Compute place fields | "Most Common Patterns" above |
| Animate fields | "Most Common Patterns" above |
| Compute PSTH | "Most Common Patterns" above |
| Compute egocentric bearing/distance | "Most Common Patterns" above |
| Analyze object-vector cells | "Most Common Patterns" above |
| Analyze spatial view cells | "Most Common Patterns" above |
| Create GLM regressors from events | [QUICKSTART.md - Events](.claude/QUICKSTART.md#events-and-peri-event-analysis) |
| Transform allocentric↔egocentric | [QUICKSTART.md - Egocentric Frames](.claude/QUICKSTART.md#egocentric-reference-frames) |
| Simulate object-vector cells | [QUICKSTART.md - Object-Vector Cells](.claude/QUICKSTART.md#object-vector-cells) |
| Compute visibility/viewshed | [ADVANCED.md - Gaze Analysis](.claude/ADVANCED.md#gaze-analysis-and-visibility-v0190) |
| Classify spatial view cells | [QUICKSTART.md - Spatial View Cells](.claude/QUICKSTART.md#spatial-view-cells) |
| Find import statement | [API_REFERENCE.md](.claude/API_REFERENCE.md) |
| Run tests | "Essential Commands" above |
| Fix error | "When Things Break" above |
| Understand mixins | [PATTERNS.md - Mixin Pattern](.claude/PATTERNS.md#mixin-pattern-for-environment) |
| Work with NWB files | [ADVANCED.md - NWB](.claude/ADVANCED.md#nwb-integration-v070) |
| Add video overlay | [ADVANCED.md - Video Overlay](.claude/ADVANCED.md#video-overlay-v050) |
| Annotate track graphs | [ADVANCED.md - Track Graph](.claude/ADVANCED.md#track-graph-annotation-v090) |
| Profile napari performance | [PROFILING.md](.claude/PROFILING.md) |

### By Problem Type

| Problem | Solution |
|---------|----------|
| Import error | "When Things Break" above |
| RuntimeError: not fitted | "When Things Break" above |
| ValueError: no active bins | "When Things Break" above |
| Tests fail | [DEVELOPMENT.md - Testing](.claude/DEVELOPMENT.md#testing) |
| Pre-commit hooks fail | [TROUBLESHOOTING.md - Pre-commit](.claude/TROUBLESHOOTING.md#pre-commit-hooks-fail-on-commit) |
| Memory warning | [TROUBLESHOOTING.md - ResourceWarning](.claude/TROUBLESHOOTING.md#resourcewarning-creating-large-grid-v021) |
| Type errors | [PATTERNS.md - Mypy](.claude/PATTERNS.md#mypy-type-checking-requirements) |
| Slow napari animations | [PROFILING.md - Common Performance Issues](.claude/PROFILING.md#common-performance-issues) |

---

## 📝 Development Quick Reference

### Git Commit Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): description
fix(scope): description
docs(scope): description
test(scope): description
chore(scope): description
```

**Examples:**

- `feat(M3): add .info() method`
- `fix: correct version reference`
- `docs(M8): update CLAUDE.md with speed-based animation API`

### NumPy Docstring Format (Required)

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
```

**More details:** [DEVELOPMENT.md - Documentation Style](.claude/DEVELOPMENT.md#documentation-style)

---

## 🧪 Testing Quick Reference

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_function_name -v

# Run with coverage
uv run pytest --cov=src/neurospatial

# Run doctests
uv run pytest --doctest-modules src/neurospatial/

# Skip slow tests
uv run pytest -m "not slow"
```

**More testing options:** [DEVELOPMENT.md - Testing](.claude/DEVELOPMENT.md#testing)

---

## 📖 Project Context

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides tools for spatial analysis, particularly for neuroscience applications involving place fields, position tracking, and spatial navigation.

**Key Features:**

- Flexible discretization (regular grids, hexagonal, triangular, masked, polygon-bounded)
- 1D linearization (track-based environments like T-maze, linear track)
- Neural analysis (place fields, Bayesian decoding, trajectory analysis)
- Egocentric reference frames (allocentric↔egocentric transforms, heading computation)
- Object-vector cells (simulation, metrics, field computation, visualization)
- Spatial view cells (visibility, gaze models, view fields, classification)
- Visualization (interactive animation with napari, video export, HTML players)
- NWB integration (read/write NeurodataWithoutBorders files - optional)

**Current Version:** v0.3.0 (Spatial view cells, object-vector cells, egocentric frames)

---

## ❓ Can't Find What You Need?

1. **Check "Most Common Patterns" above** - covers 90% of tasks
2. **Search "Quick Navigation" tables** - organized by task and problem type
3. **Read the relevant guide**:
   - Patterns/examples → [QUICKSTART.md](.claude/QUICKSTART.md)
   - Imports → [API_REFERENCE.md](.claude/API_REFERENCE.md)
   - Errors → [TROUBLESHOOTING.md](.claude/TROUBLESHOOTING.md)
   - Design → [PATTERNS.md](.claude/PATTERNS.md)
   - Advanced → [ADVANCED.md](.claude/ADVANCED.md)
   - Performance → [PROFILING.md](.claude/PROFILING.md)
4. **Search across files** - Use Ctrl+F in editor across `.claude/` directory

**For questions or issues:** <https://github.com/anthropics/claude-code/issues>
