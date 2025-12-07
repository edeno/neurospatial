# Architecture Overview

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs.

**Updated** - Domain-centric package reorganization.

---

## Domain-Centric Architecture

The codebase follows a tiered, domain-centric architecture:

### Dependency Tiers

```text
Tier 1 (Foundation):  layout/, regions/, stats/
                           ↓
Tier 2 (Core):        environment/
                           ↓
Tier 3 (Primitives):  ops/
                           ↓
Tier 4 (Domains):     encoding/, decoding/, behavior/, events/
                           ↓
Tier 5 (Leaf Nodes):  animation/, simulation/, annotation/, io/
```

**Key principle**: Lower tiers never import from higher tiers.

---

## Module Overview

### layout/ - Layout Engines

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

### regions/ - Spatial Regions

- **Immutable** `Region` dataclass (points or polygons)
- `Regions` container with dict-like interface
- JSON serialization with versioned schema

### stats/ - Statistical Methods

- `circular.py` - Circular statistics, basis functions for GLMs
- `shuffle.py` - Shuffle controls for null distributions
- `surrogates.py` - Surrogate data generation

### Tier 2: Core

### environment/ - Main User-Facing Class

Main user-facing class wrapping a `LayoutEngine`. Uses **mixin pattern** for organization:

| Mixin Module | Responsibility |
|--------------|----------------|
| `core.py` | Dataclass definition, state, properties |
| `factories.py` | Factory classmethods (`from_samples`, etc.) |
| `queries.py` | Spatial queries (`bin_at`, `neighbors`, etc.) |
| `trajectory.py` | Occupancy, bin sequences, transitions |
| `transforms.py` | Rebin, subset operations |
| `fields.py` | Kernel smoothing, interpolation |
| `metrics.py` | Boundary bins, linearization |
| `regions.py` | Region operations |
| `serialization.py` | Save/load methods |
| `visualization.py` | Plotting methods |

**Mixin pattern constraints:**

- **ONLY** `Environment` is a `@dataclass` - mixins MUST be plain classes
- Mixins use `TYPE_CHECKING` guards to avoid circular imports
- All mixin methods annotate `self: "Environment"` for type checking

**See [PATTERNS.md - Mixin Pattern](PATTERNS.md#mixin-pattern-for-environment) for details.**

### Tier 3: Primitives

### ops/ - Low-Level Operations

Power-user utilities for spatial operations:

| Module | Purpose |
|--------|---------|
| `binning.py` | Point-to-bin mapping (`map_points_to_bins`) |
| `distance.py` | Distance fields, pairwise distances |
| `egocentric.py` | Allocentric↔egocentric transforms, heading |
| `visibility.py` | Viewshed, gaze, line-of-sight |
| `transforms.py` | Affine transforms, calibration |
| `smoothing.py` | Diffusion kernels, apply kernel |
| `normalize.py` | Field normalization, clamping |
| `graph.py` | Graph convolution, neighbor reduce |
| `calculus.py` | Spatial gradient, divergence |
| `alignment.py` | Cross-environment alignment |
| `basis.py` | Spatial basis functions for GLMs |

### Tier 4: Domains

### encoding/ - Neural Encoding

How neurons represent space:

| Module | Purpose |
|--------|---------|
| `place.py` | Place field computation and metrics |
| `grid.py` | Grid cell analysis |
| `head_direction.py` | Head direction cell analysis |
| `border.py` | Boundary/border cell analysis |
| `object_vector.py` | Object-vector cell fields and metrics |
| `spatial_view.py` | Spatial view cell fields and metrics |
| `phase_precession.py` | Theta phase precession |
| `population.py` | Population-level metrics |

### decoding/ - Neural Decoding

Read out from population activity:

- Bayesian decoding, likelihood, posterior
- Trajectory analysis (isotonic, linear, Radon)
- Quality metrics, shuffle tests

### behavior/ - Behavioral Analysis

| Module | Purpose |
|--------|---------|
| `trajectory.py` | Step lengths, turn angles, MSD, home range, curvature |
| `segmentation.py` | Laps, trials, region crossings, runs |
| `navigation.py` | Path efficiency, goal-directed metrics, path progress |
| `decisions.py` | Decision analysis, choice points, Voronoi boundaries |
| `vte.py` | VTE (Vicarious Trial and Error) detection, head sweeping |
| `reward.py` | Reward field computation |

### events/ - Peri-Event Analysis

- PSTH computation and plotting
- GLM regressors (temporal and spatial)
- Interval utilities

### Tier 5: Leaf Nodes

### animation/ - Visualization

- Overlay system (Position, Bodypart, HeadDirection, Event, Video)
- Napari viewer integration
- Video export with parallel rendering

### simulation/ - Neural and Trajectory Simulation

| Module | Purpose |
|--------|---------|
| `models/` | Cell models (PlaceCell, GridCell, BoundaryCell, ObjectVector, SpatialView) |
| `spikes.py` | Poisson spike generation, population spikes, modulation |
| `trajectory.py` | Trajectory simulation (OU, laps, sinusoidal) |
| `session.py` | High-level session API |
| `examples.py` | Pre-configured example sessions |
| `validation.py` | Simulation validation and plotting |
| `mazes/` | Pre-built maze environments |

### annotation/ - Video Annotation

- `core.py` - Interactive video annotation (`annotate_video`)
- `track_graph.py` - 1D track graph annotation
- `io.py` - Import from LabelMe, CVAT formats
- `_boundary_inference.py` - Auto-infer boundaries from positions

### io/ - File I/O

- `files.py` - Environment serialization
- `nwb/` - NWB integration (optional)

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

Tests mirror the source structure:

| Test Directory | What It Tests |
|----------------|---------------|
| `tests/environment/` | Core `Environment` tests |
| `tests/layout/` | Layout engine-specific tests |
| `tests/regions/` | Region functionality tests |
| `tests/ops/` | Operations module tests |
| `tests/encoding/` | Neural encoding module tests |
| `tests/decoding/` | Neural decoding tests |
| `tests/behavior/` | Behavioral analysis tests |
| `tests/events/` | Peri-event analysis tests |
| `tests/stats/` | Statistical methods tests |
| `tests/simulation/` | Simulation module tests |
| `tests/animation/` | Animation and overlay tests |
| `tests/annotation/` | Video annotation tests |
| `tests/nwb/` | NWB integration tests (requires nwb-full extra) |
| `tests/io_tests/` | File I/O tests |
| `tests/conftest.py` | Shared fixtures |

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
- `ObjectVectorOverlay` - Vectors from animal to objects (for OVC visualization)

**Conversion funnel:**

1. User creates overlay dataclasses with behavioral data
2. `_convert_overlays_to_data()` validates and aligns overlays to field frames
3. Temporal interpolation handles mixed sampling rates (e.g., 120 Hz → 10 Hz)
4. Outputs pickle-safe `OverlayData` for backend rendering

**Backend support:**

| Backend | Position | Bodypart | Head Dir | Event | TimeSeries | Video | ObjVector | Regions |
|---------|----------|----------|----------|-------|------------|-------|-----------|---------|
| Napari  | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ | ✓ |
| Video   | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ | ✓ |
| Widget  | ✓ | ✓ | ✓ | ✓ (decay) | ✓ | ✓ | ✓ | ✓ |
| HTML    | ✓ | ⚠️ | ⚠️ | ✓ (instant) | ✗ | ✗ | ⚠️ | ✓ |

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
