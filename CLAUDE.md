# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2026-01-08 (Encoding API updated to use result classes)

---

## 🎯 Critical Rules (MUST Follow)

When working with this codebase, you MUST follow these rules:

1. **ALWAYS use `uv run`** before Python commands - never use bare `python`, `pip`, or `pytest`
2. **NEVER create bare `Environment()`** - always use factory methods like `Environment.from_samples()`
3. **bin-size argument is REQUIRED for grid-inferring factories** — `from_samples`, `from_polygon`, `from_graph` need an explicit ``bin_size``; `from_pixel_mask` needs ``pixel_size``; `from_grid_mask` reads bin geometry from explicit ``grid_edges`` and accepts neither. (No bare ``Environment()`` — always go through a factory.)
4. **NumPy docstring format** - all docstrings must follow NumPy style (not Google or reST)
5. **Check `is_linearized_track` before linearization** - only 1D environments have `to_linear()` method
6. **Regions are immutable** - use `env.regions.update_region()`, never modify in place
7. **Use `@check_fitted` decorator** - methods requiring fitted state must use this decorator
8. **Egocentric angles use animal-centered convention** - 0=ahead, π/2=left, -π/2=right (NOT allocentric 0=East)

### Coordinate Convention Diagram

```
Allocentric (world):              Egocentric (animal-centered):
      North                              Left
       π/2                                π/2
        |                                  |
West----+----East                 Back----+----Ahead
  π     |     0                    ±π     |      0
        |                                  |
      South                              Right
      -π/2                               -π/2
```

**Example:** Animal at (0,0) facing East (heading=0), object at (10, 10):
- Allocentric bearing to object: π/4 (45° from East toward North)
- Egocentric bearing to object: π/4 (45° left of ahead)

### Canonical Argument Order

All public functions follow a consistent argument order pattern:

```python
# Neural encoding functions (place fields, object-vector, spatial view, etc.)
func(
    env,                    # 1. Environment (spatial context)
    spike_times,            # 2. Neural data (what fired)
    times,                  # 3. Timestamps (when sampled)
    positions,              # 4. Position coordinates (where animal was)
    headings,               # 5. Head direction (which way facing) - if egocentric
    object_positions,       # 6. External targets - if relevant
    *,                      # 7. Keyword-only separator
    method_params,          # 8. Algorithm parameters (bandwidth, smoothing_method, etc.)
)

# Egocentric operations (bearing, distance to targets)
func(
    positions,              # 1. Animal positions (where animal is)
    headings,               # 2. Animal headings (which way facing)
    targets,                # 3. Target locations (what animal is relating to)
)

# Behavioral segmentation (laps, trials, crossings)
func(
    position_bins,          # 1. Discretized position indices
    times,                  # 2. Timestamps
    env,                    # 3. Environment (for graph/region lookups)
    *,                      # 4. Keyword-only separator
    region_params,          # 5. Region specifications (start_region, end_regions, etc.)
)
```

**Key principles:**

- **Environment first** for encoding functions (establishes spatial context)
- **Animal state before targets** for egocentric ops (positions, headings, then targets)
- **Data before metadata** (spike_times before times, positions before headings)
- **Use `positions`** not `trajectory` for coordinate arrays (consistency)
- **Use `position_bins`** not `trajectory_bins` for discretized indices

**Documented exception — directional encoding.** `compute_directional_rate`,
`compute_directional_rates`, and `is_head_direction_cell` operate on
heading (a circular angular variable, not a spatial position) and so
take `(spike_times, times, headings, *, ...)` with no `env` parameter.
This is intentional and is called out in each function's docstring;
sister spatial classifiers (`is_object_vector_cell`,
`is_spatial_view_cell`, etc.) keep the env-first signature.

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
from neurospatial.encoding import compute_spatial_rate

# Compute place field for one neuron (returns SpatialRateResult)
result = compute_spatial_rate(
    env, spike_times, times, positions,
    smoothing_method="diffusion_kde",  # Default: graph-based boundary-aware KDE
    bandwidth=5.0,  # Smoothing bandwidth (cm)
    min_occupancy=0.5,  # Exclude bins with <0.5s occupancy (sets them NaN)
    fill_value=0.0,  # Replace those NaN bins with 0 Hz for the decoding golden path
)
firing_rate = result.firing_rate  # Access firing rate from result object

# Methods: "diffusion_kde" (default), "gaussian_kde", "binned" (legacy)
# Result also has: result.occupancy, result.env, result.spatial_information(), etc.
#
# fill_value default is None: low-occupancy bins stay NaN (no behavior change
# for existing callers). Pass fill_value=0.0 when feeding decode_position() so
# the model is explicitly zero-rate there -- the documented encode->decode
# golden path then composes with no manual np.nan_to_num. decode_position()
# also tolerates residual NaN bins (treats them as zero-rate, warns once).
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
from neurospatial.animation import PositionOverlay

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
from neurospatial.events import peri_event_histogram

# Compute PSTH around reward events
result = peri_event_histogram(
    spike_times, reward_times,
    window=(-1.0, 2.0),  # -1s before to 2s after event
    bin_size=0.025,      # 25 ms bins
)

# Access results (firing_rate is a cached attribute, not a method)
firing_rates = result.firing_rate
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
from neurospatial.ops.egocentric import (
    heading_from_velocity,
    compute_egocentric_bearing,
    compute_egocentric_distance,
)

# Compute heading from movement direction (min_speed in cm/s)
dt = times[1] - times[0]  # time step in seconds
headings = heading_from_velocity(positions, dt, min_speed=5.0)  # cm/s

# Compute egocentric bearing to objects (0=ahead, π/2=left, -π/2=right)
object_positions = np.array([[50, 50], [75, 25]])  # 2 objects, coordinates in cm
bearings = compute_egocentric_bearing(positions, headings, object_positions)

# Compute egocentric distance (Euclidean or geodesic)
distances = compute_egocentric_distance(
    positions, headings, object_positions, metric="euclidean"
)
```

**Need body orientation?** Use `heading_from_body_orientation(nose_pos, tail_pos)`

### 8. Compute Object-Vector Field

```python
from neurospatial.encoding import compute_egocentric_rate

# Compute firing field in egocentric polar coordinates (returns EgocentricRateResult)
result = compute_egocentric_rate(
    env, spike_times, times, positions, headings, object_positions,
    distance_range=(0.0, 50.0),  # min/max distance to object (cm)
    n_distance_bins=10,          # radial resolution
    n_direction_bins=12,         # angular resolution (full circle)
)
# result.firing_rate: firing rate in egocentric polar bins
# result.env: the egocentric polar environment
# result.occupancy: time spent in each bin
# result.preferred_distance(), result.preferred_direction(): peak location
```

**Egocentric polar environments are a DISTINCT type.**
`Environment.from_polar_egocentric(...)` returns an
`EgocentricPolarEnvironment` (in `neurospatial.environment.polar`), **not** an
`Environment` — it is a sibling type, not a subclass, so
`isinstance(polar_env, Environment)` is `False`. Its `bin_centers[:, 0]` is
distance and `bin_centers[:, 1]` is angle in radians, and its connectivity
edges carry physically correct polar lengths (arc `r·Δθ`, radial `Δr`,
diagonal `sqrt(Δr² + (r·Δθ)²)`). Cartesian-only methods (`bin_at`, `contains`,
`distance_between`, `distance_to(metric="euclidean")`, `apply_transform`) raise
`NotImplementedError`; use graph operations (`neighbors`, `path_between`,
`reachable_from`, `distance_to(metric="geodesic")`, `smooth`) instead.

**Need metrics?** See [QUICKSTART.md - Object-Vector Cells](.claude/QUICKSTART.md#object-vector-cells)

### 9. Compute Spatial View Field

```python
from neurospatial.encoding import compute_view_rate

# Compute firing field indexed by VIEWED location (returns ViewRateResult)
result = compute_view_rate(
    env, spike_times, times, positions, headings,
    gaze_model="fixed_distance",  # or "ray_cast", "boundary"
    view_distance=20.0,  # cm ahead
    smoothing_method="diffusion_kde",
    bandwidth=5.0,
)
# result.firing_rate: firing rate at each spatial bin (indexed by where animal looked)
# result.occupancy: time spent viewing each bin
# result.is_spatial_view_cell(): classification method
# result.view_spatial_information(): spatial information metric
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

### Gotcha 5: Check `is_linearized_track` before linearization

❌ **Wrong:**

```python
env = Environment.from_samples(positions, bin_size=2.0)  # 2D grid
linear_pos = env.to_linear(position)  # AttributeError
```

✅ **Right:**

```python
if env.is_linearized_track:
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

**neurospatial** uses a domain-centric architecture with clear dependency tiers:

### Tier 1 - Foundation (Zero internal deps)

- **`layout/`** - Layout engines: RegularGrid, Hexagonal, Graph (1D), Masked, ImageMask, ShapelyPolygon, TriangularMesh
- **`regions/`** - Immutable `Region` dataclass, `Regions` container with dict-like interface
- **`stats/`** - Statistical methods: circular statistics, shuffle controls, surrogates

### Tier 2 - Core

- **`environment/`** - Main user-facing class using **mixin pattern** for 6,000+ lines of functionality
  - Factory methods: `from_samples()`, `from_graph()`, `from_polygon()`, `from_grid_mask()`, `from_pixel_mask()`, `from_polar_egocentric()`

### Tier 3 - Primitives

- **`ops/`** - Low-level operations (power users)
  - `binning.py` - Point-to-bin mapping (`map_points_to_bins`)
  - `distance.py` - Distance fields and pairwise distances
  - `egocentric.py` - Allocentric↔egocentric transforms, heading computation
  - `visibility.py` - Viewshed, gaze, line-of-sight
  - `transforms.py`, `smoothing.py`, `graph.py`, `calculus.py`, `basis.py`

### Tier 4 - Domains

- **`encoding/`** - Neural encoding (how neurons represent space)
  - `spatial.py`, `grid.py`, `directional.py`, `border.py`
  - `egocentric.py`, `view.py`, `phase_precession.py`, `population.py`
- **`decoding/`** - Neural decoding (read out from population)
- **`behavior/`** - Behavioral analysis
  - `trajectory.py`, `segmentation.py`, `navigation.py`, `decisions.py`, `reward.py`
- **`events/`** - Peri-event analysis, GLM regressors

### Tier 5 - Leaf Nodes

- **`animation/`** - Napari viewer, video export, overlays
- **`simulation/`** - Neural and trajectory simulation
- **`io/`** - File I/O, NWB integration

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

**Current Version:** v0.3.x (Domain-centric package reorganization)

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
