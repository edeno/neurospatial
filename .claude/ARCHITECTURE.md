# Architecture Overview

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs.

---

## Three-Layer Design

The codebase follows a three-layer architecture:

### 1. Layout Engines (`src/neurospatial/layout/`)

Protocol-based design with `LayoutEngine` interface. Available engines:

| Engine | Description | Use Case |
|--------|-------------|----------|
| `RegularGridLayout` | Rectangular grids | Standard 2D/3D spatial binning |
| `HexagonalLayout` | Hexagonal tessellation | Equal-distance neighbors |
| `GraphLayout` | 1D linearized tracks | T-maze, linear track, Y-maze |
| `MaskedGridLayout` | Grid with active/inactive bins | Arbitrary active regions |
| `ImageMaskLayout` | Binary image-based | Video annotation masks |
| `ShapelyPolygonLayout` | Polygon-bounded grid | Irregular arena shapes |
| `TriangularMeshLayout` | Triangular tessellation | Alternative to hexagonal |

**Key properties:**

- All engines produce: `bin_centers`, `connectivity` graph, `dimension_ranges`
- Graph has mandatory node/edge attributes (see [PATTERNS.md - Graph Metadata](PATTERNS.md#graph-metadata-requirements))
- Factory pattern via `create_layout()` in [layout/factories.py](../src/neurospatial/layout/factories.py)

### 2. Environment (`src/neurospatial/environment/`)

Main user-facing class wrapping a `LayoutEngine`. Uses **mixin pattern** for organization:

| Mixin Module | Responsibility | Lines |
|--------------|----------------|-------|
| `core.py` | Dataclass definition, state, properties | 1,023 |
| `factories.py` | Factory classmethods (`from_samples`, etc.) | 630 |
| `queries.py` | Spatial queries (`bin_at`, `neighbors`, etc.) | 897 |
| `trajectory.py` | Occupancy, bin sequences, transitions | 1,222 |
| `transforms.py` | Rebin, subset operations | 634 |
| `fields.py` | Kernel smoothing, interpolation | 564 |
| `metrics.py` | Boundary bins, linearization | 469 |
| `regions.py` | Region operations | 398 |
| `serialization.py` | Save/load methods | 315 |
| `visualization.py` | Plotting methods | 211 |

**Mixin pattern constraints:**

- **ONLY** `Environment` is a `@dataclass` - mixins MUST be plain classes
- Mixins use `TYPE_CHECKING` guards to avoid circular imports
- All mixin methods annotate `self: "Environment"` for type checking

**See [PATTERNS.md - Mixin Pattern](PATTERNS.md#mixin-pattern-for-environment) for details.**

### 3. Regions (`src/neurospatial/regions/`)

- **Immutable** `Region` dataclass (points or polygons)
- `Regions` container with dict-like interface
- JSON serialization with versioned schema

---

## Key Features

- **Flexible discretization**: Regular grids, hexagonal, triangular, masked, polygon-bounded
- **1D linearization**: Track-based environments (T-maze, linear track, etc.)
- **Neural analysis**: Place fields, Bayesian decoding, trajectory analysis
- **Visualization**: Interactive animation with napari, video export, HTML players
- **NWB integration**: Read/write NeurodataWithoutBorders files (optional)

---

## Package Management with uv

**CRITICAL: This project uses `uv` for package management.**

- Python version: 3.13 (specified in `.python-version`)
- **ALWAYS** use `uv run` to execute Python commands
- **NEVER** use bare `python`, `pip`, or `pytest` commands

**Why uv?** Automatically manages virtual environment and ensures correct Python environment without manual activation.

---

## Dependencies

**Core:**

- `numpy` - Array operations
- `pandas` - Data structures
- `matplotlib` - Visualization
- `networkx` - Graph structures
- `scipy` - KDTree, morphological operations
- `scikit-learn` - ML utilities
- `shapely` - Geometric operations
- `track-linearization` - 1D track linearization

**Development:**

- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `ruff` - Linter and formatter
- `mypy` - Type checking
- `ipython` - Interactive shell

**Optional:**

- `pynwb`, `hdmf`, `ndx-pose`, `ndx-events` - NWB support (install with `[nwb-full]`)
- `scikit-image` - Radon transform for trajectory detection (install with `[trajectory]`)
- `alphashape` - Concave hull for boundary inference

---

## Design Philosophy

### Methods vs Free Functions

**Environment methods** - Questions about that environment or local transforms:

- Examples: `env.bin_at()`, `env.neighbors()`, `env.distance_between()`, `env.rebin()`
- Use when: Working with single environment's structure

**Free functions** - Cross-environment operations, neural/behavioral analysis:

- Examples: `distance_field()`, `map_points_to_bins()`, `compute_place_field()`
- Use when: Neural analysis, batch processing, multi-environment operations

**Principle:** Separation keeps `Environment` focused on spatial structure while providing specialized domain functions.

### 1D vs N-D Environments

- **1D**: `GraphLayout` with `is_1d=True`, provides `to_linear()` and `linear_to_nd()`
- **N-D**: Grid-based layouts with spatial queries in original coordinates

**Always check before linearization:**

```python
if env.is_1d:
    linear_pos = env.to_linear(nd_position)
else:
    bin_idx = env.bin_at(position)  # Use N-D queries
```

---

## Testing Structure

Tests mirror source structure:

- `tests/test_environment.py` - Core `Environment` tests
- `tests/test_composite.py` - `CompositeEnvironment` tests
- `tests/test_alignment.py` - Alignment/transformation tests
- `tests/layout/` - Layout engine-specific tests
- `tests/regions/` - Region functionality tests
- `tests/nwb/` - NWB integration tests (requires nwb-full extra)
- `tests/conftest.py` - Shared fixtures

**NWB fixtures** use `pytest.importorskip()` for graceful skipping when dependencies not installed.

---

## Animation System Architecture

### Overlay System

**Public overlay dataclasses:**

- `PositionOverlay` - Trajectory tracking with trails
- `BodypartOverlay` - Pose tracking with skeleton
- `HeadDirectionOverlay` - Orientation arrows
- `EventOverlay` - Spikes, licks, rewards
- `TimeSeriesOverlay` - Continuous variables
- `VideoOverlay` - Recorded video behind/above fields

**Conversion funnel:**

1. User creates overlay dataclasses with behavioral data
2. `_convert_overlays_to_data()` validates and aligns overlays to field frames
3. Temporal interpolation handles mixed sampling rates (e.g., 120 Hz → 10 Hz)
4. Outputs pickle-safe `OverlayData` for backend rendering

**Backend support:**

| Backend | Position | Bodypart | Head Dir | Event | TimeSeries | Video | Regions |
|---------|----------|----------|----------|-------|------------|-------|---------|
| Napari  | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ |
| Video   | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ |
| Widget  | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ |
| HTML    | ✓ | ⚠️ | ⚠️ | ✓ (instant) | ✗ | ✗ | ✓ |

(⚠️ = Skipped with warning, ✗ = Not supported)

### Playback Control

Speed-based API separates data sample rate from playback speed:

- **`frame_times`** (required): Timestamps for each frame in seconds
- **`speed`** (default 1.0): Playback multiplier (1.0=real-time, 0.1=slow motion, 2.0=fast forward)

**How playback fps is computed:**

```python
sample_rate_hz = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
playback_fps = min(sample_rate_hz * speed, MAX_PLAYBACK_FPS)  # Capped at 60 fps
```

### Coordinate Systems

Environment and napari use different conventions:

| System | X-axis | Y-axis | Origin |
|--------|--------|--------|--------|
| Environment | Horizontal (columns) | Vertical (rows), up is positive | Bottom-left |
| Napari | Column index | Row index, down is positive | Top-left |

**For overlays:**

- Use **environment coordinates** (same as your position data)
- Animation system automatically transforms to napari pixel space
- Transformations: (x,y) to (row,col) swap and Y-axis inversion

---

## Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` - New features
- `fix(scope): description` - Bug fixes
- `docs(scope): description` - Documentation changes
- `test(scope): description` - Test additions/fixes
- `chore(scope): description` - Maintenance tasks

Examples:

- `feat(M3): add .info() method`
- `fix: correct version reference`
- `docs(M8): update CLAUDE.md with speed-based animation API`
