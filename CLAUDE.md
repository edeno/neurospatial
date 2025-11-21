# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2025-11-20 (v0.4.0 - Animation overlays feature)

## Table of Contents

- [Quick Reference](#quick-reference)
- [Project Overview](#project-overview)
- [Package Management with uv](#package-management-with-uv)
- [Core Architecture](#core-architecture)
- [Import Patterns](#import-patterns)
- [Important Patterns & Constraints](#important-patterns--constraints)
- [Development Commands](#development-commands)
- [Key Implementation Notes](#key-implementation-notes)
- [Testing Structure](#testing-structure)
- [Documentation Style](#documentation-style)
- [Common Gotchas](#common-gotchas)
- [Troubleshooting](#troubleshooting)

## Quick Reference

**Most Common Commands**:

```bash
# Run all tests (from project root)
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_function_name -v

# Run with coverage
uv run pytest --cov=src/neurospatial

# Lint and format
uv run ruff check . && uv run ruff format .

# Run doctests
uv run pytest --doctest-modules src/neurospatial/

# Run performance benchmarks (slow tests)
uv run pytest -m slow -v -s

# Run all tests except performance benchmarks
uv run pytest -m "not slow"
```

**Most Common Patterns**:

```python
# Create environment from data
env = Environment.from_samples(positions, bin_size=2.0)  # bin_size is required

# Add units and coordinate frame (v0.1.0+)
env.units = "cm"
env.frame = "session1"

# Save and load environments (v0.1.0+)
env.to_file("my_environment")  # Creates .json + .npz files
loaded_env = Environment.from_file("my_environment")

# Map points to bins with KDTree caching (v0.1.0+)
from neurospatial import map_points_to_bins
bin_indices = map_points_to_bins(points, env, tie_break="lowest_index")

# Estimate transform from corresponding points (v0.1.0+)
from neurospatial import estimate_transform, apply_transform_to_environment
transform = estimate_transform(src_landmarks, dst_landmarks, kind="rigid")
aligned_env = apply_transform_to_environment(env, T)

# Compute distance fields (v0.1.0+)
from neurospatial import distance_field
distances = distance_field(env.connectivity, sources=[goal_bin_id])

# Compute place fields from spike data (v0.2.0+)
from neurospatial import compute_place_field
firing_rate = compute_place_field(
    env, spike_times, times, positions,
    method="diffusion_kde",  # Default: graph-based boundary-aware KDE
    bandwidth=5.0  # Smoothing bandwidth (cm)
)
# Methods: "diffusion_kde" (default), "gaussian_kde", "binned" (legacy)

# Validate environment (v0.1.0+)
from neurospatial import validate_environment
validate_environment(env, strict=True)  # Warns if units/frame missing

# Update a region (don't modify in place)
env.regions.update_region("goal", point=new_point)

# Check if 1D before using linearization
if env.is_1d:
    linear_pos = env.to_linear(nd_position)

# Always use factory methods, not bare Environment()
env = Environment.from_samples(...)  # ✓ Correct
env = Environment()  # ✗ Wrong - won't be fitted

# Memory safety: automatic warnings for large grids (v0.2.1+)
# Warns at 100MB estimated memory (but creation still proceeds)
positions = np.random.uniform(0, 10000, (1000, 2))
env = Environment.from_samples(positions, bin_size=1.0)  # May warn, but will succeed

# Disable warning if intentional (v0.2.1+)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))

# Animate spatial fields over time (v0.3.0+)
from neurospatial.animation import subsample_frames

# Interactive Napari viewer (best for 100K+ frames)
env.animate_fields(fields, backend="napari")

# Video export with parallel rendering (requires ffmpeg)
env.animate_fields(fields, backend="video", save_path="animation.mp4", fps=30, n_workers=4)

# HTML standalone player (max 500 frames)
env.animate_fields(fields, backend="html", save_path="animation.html")

# Jupyter widget (notebook integration)
env.animate_fields(fields, backend="widget")

# Auto-select backend based on save_path or context
env.animate_fields(fields, save_path="animation.mp4")  # Detects .mp4 → video backend

# Subsample large datasets for video export (e.g., 250 Hz → 30 fps)
subsampled_fields = subsample_frames(fields, source_fps=250, target_fps=30)

# IMPORTANT: Clear caches before parallel rendering (pickle-ability requirement)
env.clear_cache()  # Makes environment pickle-able for multiprocessing
env.animate_fields(fields, backend="video", n_workers=4, save_path="output.mp4")

# Animation overlays (v0.4.0+)
from neurospatial import PositionOverlay, BodypartOverlay, HeadDirectionOverlay

# Position overlay with trail
position_overlay = PositionOverlay(
    data=trajectory,  # Shape: (n_frames, n_dims)
    color="red",
    size=12.0,
    trail_length=10  # Show last 10 frames as decaying trail
)
env.animate_fields(fields, overlays=[position_overlay], backend="napari")

# Pose tracking with skeleton
bodypart_overlay = BodypartOverlay(
    data={"nose": nose_traj, "body": body_traj, "tail": tail_traj},
    skeleton=[("tail", "body"), ("body", "nose")],
    colors={"nose": "yellow", "body": "red", "tail": "blue"},
    skeleton_color="white",
    skeleton_width=2.0
)
env.animate_fields(fields, overlays=[bodypart_overlay], backend="napari")

# Head direction arrows
head_direction_overlay = HeadDirectionOverlay(
    data=head_angles,  # Shape: (n_frames,) radians OR (n_frames, n_dims) unit vectors
    color="yellow",
    length=15.0  # Arrow length in environment units
)
env.animate_fields(fields, overlays=[head_direction_overlay], backend="napari")

# Multi-animal tracking (multiple overlays)
animal1 = PositionOverlay(data=traj1, color="red", trail_length=10)
animal2 = PositionOverlay(data=traj2, color="blue", trail_length=10)
env.animate_fields(fields, overlays=[animal1, animal2], backend="napari")

# Mixed-rate temporal alignment (120 Hz position → 10 Hz fields)
overlay = PositionOverlay(
    data=trajectory_120hz,
    times=timestamps_120hz,  # Overlay timestamps
    color="red",
    trail_length=15
)
env.animate_fields(
    fields_10hz,
    overlays=[overlay],
    frame_times=timestamps_10hz,  # Field timestamps - automatic interpolation
    backend="napari"
)

# Show regions with overlays
env.animate_fields(
    fields,
    overlays=[position_overlay],
    show_regions=True,  # Show all regions, or ["region1", "region2"] for specific
    region_alpha=0.3,   # 30% transparent
    backend="napari"
)

# Backend capabilities: Napari/Video/Widget support all overlays; HTML supports position+regions only
```

**Type Checking Support (v0.2.1+)**:

This package now includes a `py.typed` marker, enabling type checkers like mypy to use the library's type annotations:

```python
# Your IDE and mypy will now see neurospatial types!
from neurospatial import Environment
import numpy as np

positions: np.ndarray = np.random.rand(100, 2)
env: Environment = Environment.from_samples(positions, bin_size=5.0)
# mypy will validate types ✓
```

**Commit Message Format**:

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` - New features
- `fix(scope): description` - Bug fixes
- `docs(scope): description` - Documentation changes
- `test(scope): description` - Test additions/fixes
- `chore(scope): description` - Maintenance tasks

Examples: `feat(M3): add .info() method`, `fix: correct version reference`

## Project Overview

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides tools for spatial analysis, particularly for neuroscience applications involving place fields, position tracking, and spatial navigation.

## Package Management with uv

**CRITICAL: This project uses `uv` for package management and virtual environment handling.**

- Python version: 3.13 (specified in `.python-version`)
- **ALWAYS** use `uv run` to execute Python commands in the correct virtual environment
- **NEVER** use bare `python`, `pip`, or `pytest` commands - always prefix with `uv run`

### Why uv?

`uv` automatically manages the virtual environment and ensures all commands run in the correct Python environment without manual activation. It reads `.python-version` and handles environment setup transparently.

## Core Architecture

### Three-Layer Design

The codebase follows a three-layer architecture:

1. **Layout Engines** (`src/neurospatial/layout/`)
   - Protocol-based design with `LayoutEngine` interface ([layout/base.py](src/neurospatial/layout/base.py))
   - Multiple concrete implementations in `layout/engines/`:
     - `RegularGridLayout` - Standard rectangular grids
     - `HexagonalLayout` - Hexagonal tessellations
     - `GraphLayout` - 1D linearized track representations (requires `track-linearization` package)
     - `MaskedGridLayout` - Grids with arbitrary active/inactive regions
     - `ImageMaskLayout` - Binary image-based layouts
     - `ShapelyPolygonLayout` - Polygon-bounded grids
     - `TriangularMeshLayout` - Triangular tessellations
   - Factory pattern via `create_layout()` in [layout/factories.py](src/neurospatial/layout/factories.py:126-177)
   - All engines produce: `bin_centers`, `connectivity` graph, `dimension_ranges`, and optional grid metadata

2. **Environment** (`src/neurospatial/environment/`)
   - Main user-facing class wrapping a `LayoutEngine` instance
   - **Modular package structure** (as of v0.2.1) using mixin pattern:
     - [core.py](src/neurospatial/environment/core.py) - Core `Environment` dataclass with state and properties (1,023 lines)
     - [factories.py](src/neurospatial/environment/factories.py) - Factory classmethods for creating instances (630 lines)
     - [queries.py](src/neurospatial/environment/queries.py) - Spatial query methods (897 lines)
     - [trajectory.py](src/neurospatial/environment/trajectory.py) - Trajectory analysis (occupancy, bin_sequence, transitions) (1,222 lines)
     - [transforms.py](src/neurospatial/environment/transforms.py) - Rebin/subset operations (634 lines)
     - [fields.py](src/neurospatial/environment/fields.py) - Spatial field operations (compute_kernel, smooth, interpolate) (564 lines)
     - [metrics.py](src/neurospatial/environment/metrics.py) - Environment metrics and properties (boundary_bins, bin_attributes, linearization) (469 lines)
     - [regions.py](src/neurospatial/environment/regions.py) - Region operations (398 lines)
     - [serialization.py](src/neurospatial/environment/serialization.py) - Save/load methods (315 lines)
     - [visualization.py](src/neurospatial/environment/visualization.py) - Plotting methods (211 lines)
     - [decorators.py](src/neurospatial/environment/decorators.py) - `@check_fitted` decorator (77 lines)
   - Factory methods for common use cases:
     - `Environment.from_samples()` - Discretize point data into bins
     - `Environment.from_graph()` - Create 1D track-based environments
     - `Environment.from_polygon()` - Grid masked by Shapely polygon
     - `Environment.from_mask()` - Pre-defined N-D boolean mask
     - `Environment.from_image()` - Binary image mask
     - `Environment.from_layout()` - Direct layout specification
   - Provides spatial queries: `bin_at()`, `contains()`, `neighbors()`, `distance_between()`, `path_between()`
   - Integrates `Regions` for defining named ROIs (regions of interest)
   - Uses `@check_fitted` decorator ([environment/decorators.py](src/neurospatial/environment/decorators.py)) to ensure methods are only called after initialization

3. **Regions** (`src/neurospatial/regions/`)
   - Immutable `Region` dataclass for points or polygons ([regions/core.py:36-125](src/neurospatial/regions/core.py#L36-L125))
   - `Regions` mapping container (dict-like interface)
   - JSON serialization support with versioned schema
   - Operations: `add()`, `buffer()`, `area()`, `region_center()`

### Supporting Modules

- **CompositeEnvironment** ([composite.py](src/neurospatial/composite.py)) - Merges multiple `Environment` instances with automatic mutual-nearest-neighbor bridge inference
- **Alignment** ([alignment.py](src/neurospatial/alignment.py)) - Transforms and maps probability distributions between environments (rotation, scaling, translation)
- **Transforms** ([transforms.py](src/neurospatial/transforms.py)) - 2D affine transformations with composable API

## Import Patterns

Standard import patterns for this package:

```python
# Main classes
from neurospatial import Environment
from neurospatial.regions import Region, Regions

# v0.1.0+ Public API functions
from neurospatial import (
    validate_environment,          # Validate environment structure
    map_points_to_bins,            # Batch point-to-bin mapping with KDTree
    estimate_transform,             # Estimate transform from point pairs
    apply_transform_to_environment, # Transform entire environment
    distance_field,                # Multi-source geodesic distances
    pairwise_distances,            # Distances between node subsets
)

# Serialization (v0.1.0+)
from neurospatial.io import to_file, from_file, to_dict, from_dict

# Animation (v0.3.0+)
from neurospatial.animation import subsample_frames

# Cache management (v0.3.0+)
# Use env.clear_cache() for all cache clearing operations
# Example: env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
# IMPORTANT: Call env.clear_cache() before parallel rendering (makes environment pickle-able)

# Layout engines and factories
from neurospatial.layout.factories import create_layout, list_available_layouts
from neurospatial.layout.engines.regular_grid import RegularGridLayout

# Utility functions
from neurospatial.alignment import get_2d_rotation_matrix, map_probabilities
from neurospatial.transforms import Affine2D, translate, scale_2d
```

## Important Patterns & Constraints

### Graph Metadata Requirements

The connectivity graph (`nx.Graph`) has **mandatory node and edge attributes**:

**Node attributes** (enforced by layout engines):

- `'pos'`: Tuple[float, ...] - N-D coordinates
- `'source_grid_flat_index'`: int - Flat index in original grid
- `'original_grid_nd_index'`: Tuple[int, ...] - N-D grid index

**Edge attributes** (enforced by layout engines):

- `'distance'`: float - Euclidean distance between bin centers
- `'vector'`: Tuple[float, ...] - Displacement vector
- `'edge_id'`: int - Unique edge ID
- `'angle_2d'`: Optional[float] - Angle for 2D layouts

### Mixin Pattern for Environment

The `Environment` class uses **mixin inheritance** to organize its 6,000+ lines of functionality into focused modules:

```python
# In src/neurospatial/environment/core.py
@dataclass  # Only Environment is a dataclass
class Environment(
    EnvironmentFactories,      # Factory classmethods
    EnvironmentQueries,         # Spatial query methods
    EnvironmentSerialization,   # Save/load methods
    EnvironmentRegions,         # Region operations
    EnvironmentVisualization,   # Plotting methods
    EnvironmentMetrics,         # Metrics and spatial properties
    EnvironmentFields,          # Spatial field operations
    EnvironmentTrajectory,      # Trajectory analysis
    EnvironmentTransforms,      # Rebin/subset operations
):
    """Main Environment class assembled from mixins."""
    name: str = ""
    layout: LayoutEngine | None = None
    # ... rest of dataclass fields
```

**Key constraints:**

- **ONLY `Environment` is a `@dataclass`** - All mixins MUST be plain classes
- Mixins use `TYPE_CHECKING` guards to avoid circular imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from neurospatial.environment.core import Environment

  class EnvironmentQueries:
      def bin_at(self: "Environment", points) -> int:
          # Use string annotation for type hint
          ...
  ```
- All mixin methods have access to `self` attributes (from Environment dataclass)
- Public API unchanged: `from neurospatial import Environment` still works

### Mypy Type Checking Requirements

**IMPORTANT: Mypy runs in pre-commit hooks and should pass without errors.**

This project uses mypy for type checking with a **pragmatic configuration** suited for scientific Python code. The configuration (in `pyproject.toml`) balances type safety with practicality:

- **Type annotations encouraged**: Public APIs and mixin methods should have type hints
- **Lenient for scientific code**: Allows untyped defs, untyped calls, and incomplete defs (common in scientific libraries)
- **Pre-commit enforcement**: Mypy runs automatically on commit with basic checks
- **Key settings** (see `[tool.mypy]` in `pyproject.toml`):
  - `disallow_untyped_defs = false` - Allows functions without type annotations
  - `check_untyped_defs = false` - Doesn't check inside untyped functions
  - `allow_untyped_calls = true` - Allows calling untyped library functions
  - `warn_unused_ignores = true` - Warns about unnecessary `type: ignore` comments

**Guidelines:**

1. **Prefer proper typing over suppressions** - Add type hints when possible rather than using `type: ignore`
2. **Mixin methods should be typed** - Use proper type annotations for all public mixin methods (see pattern below)
3. **Avoid skipping mypy** - Let pre-commit run mypy normally; only skip if absolutely necessary

**Mixin Type Annotation Pattern:**

The mixin pattern requires special care for mypy. Since mixins access Environment attributes that don't exist on the mixin class itself, use these patterns:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

class EnvironmentMixin:
    """Mixin providing methods for Environment class.

    Methods in this mixin expect to be called on an Environment instance
    and will have access to all Environment attributes.
    """

    def method_name(self: "Environment", param: int) -> NDArray[np.float64]:
        """Use string annotation 'Environment' for self parameter."""
        # Mypy understands self has Environment attributes
        return self.bin_centers  # ✓ No error
```

**Common mypy issues and fixes:**

- **Missing attribute errors**: Ensure `self: "Environment"` annotation is present
- **Type mismatches**: Ensure return types match exactly between mixin and Environment
- **Cache type annotations**: Use precise Literal types for cache keys
- **Import TYPE_CHECKING**: Always use `if TYPE_CHECKING:` guard for Environment import

**Pre-commit mypy configuration** (`.pre-commit-config.yaml`):

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.13.0
  hooks:
    - id: mypy
      args: [--ignore-missing-imports, --warn-unused-ignores]
```

**Using Protocol for Mixin Type Safety:**

This project uses the `Protocol` pattern (PEP 544) to enable proper type checking for mixins. See [mypy docs on mixins](https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes).

**Pattern** (`src/neurospatial/environment/_protocols.py`):

```python
from typing import Protocol

class EnvironmentProtocol(Protocol):
    """Interface that Environment provides to mixins."""
    name: str
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    # ... all attributes mixins need

    def bin_at(self, points: NDArray[np.float64]) -> NDArray[np.int_]: ...
```

**Mixin Usage:**

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol

class EnvironmentFields:
    def smooth(self: EnvironmentProtocol, field: NDArray, ...) -> NDArray:
        # Mypy validates Protocol at call sites, not definition sites
        return self.compute_kernel(...) @ field
```

**Why this works:** Mypy checks that `Environment` satisfies `EnvironmentProtocol` at usage sites. Mixins can safely access Protocol-defined attributes without "erased type" errors.

**Running mypy manually:**

```bash
# Check all files (same args as pre-commit)
uv run mypy --ignore-missing-imports --warn-unused-ignores src/neurospatial/

# Check specific file
uv run mypy --ignore-missing-imports --warn-unused-ignores src/neurospatial/environment/fields.py

# Check with pyproject.toml config (recommended)
uv run mypy src/neurospatial/
```

### Protocol-Based Design

Layout engines implement the `LayoutEngine` Protocol ([layout/base.py:10-166](src/neurospatial/layout/base.py#L10-L166)), not inheritance. When creating new engines:

- Implement required attributes: `bin_centers`, `connectivity`, `dimension_ranges`, `is_1d`, `_layout_type_tag`, `_build_params_used`
- Implement required methods: `build()`, `point_to_bin_index()`, `bin_sizes()`, `plot()`
- Optionally provide grid-specific attributes: `grid_edges`, `grid_shape`, `active_mask`

### Fitted State Pattern

`Environment` uses a `_is_fitted` flag set after `_setup_from_layout()` completes. Methods requiring fitted state use the `@check_fitted` decorator. This prevents calling spatial query methods before the environment is properly initialized.

### Regions are Immutable

`Region` objects are immutable dataclasses - create new instances rather than modifying existing ones. The `Regions` container uses dict-like semantics:

- Use `regions.add()` to create and insert (raises `KeyError` if name already exists)
- Use `del regions[name]` or `regions.remove(name)` to delete
- Assignment to existing keys succeeds but emits a `UserWarning` to prevent accidental overwrites
- Use `regions.update_region()` to update regions without warnings

### 1D vs N-D Environments

Environments can be 1D (linearized tracks) or N-D (grids):

- 1D: `GraphLayout` with `is_1d=True`, provides `to_linear()` and `linear_to_nd()` methods
- N-D: Grid-based layouts with spatial queries in original coordinate space

Check `env.is_1d` before calling linearization methods.

### Methods vs Free Functions

**When to use Environment methods vs module-level functions:**

- **Methods on Environment** answer questions about that environment or perform local transforms.
  - Examples: `env.bin_at()`, `env.neighbors()`, `env.distance_between()`, `env.rebin()`
  - Use when: Working with a single environment's structure and properties

- **Free functions** take environments/graphs/fields as input and perform higher-level analysis (neural metrics, segmentation, alignment).
  - Examples: `distance_field()`, `map_points_to_bins()`, `estimate_transform()`, `compute_place_field()`
  - Use when: Cross-environment operations, neural/behavioral analysis, or batch processing

**If you're unsure:** Start from the object you have (Environment, field array, graph) and use autocomplete. If it's about cross-environment, neural, or behavioral analysis, look under the free functions in `neurospatial.__init__`.

**Design principle:** This separation keeps the `Environment` class focused on spatial structure while providing specialized functions for domain-specific analyses (neuroscience, navigation, etc.).

## Development Commands

**IMPORTANT: All commands below MUST be prefixed with `uv run` to ensure they execute in the correct virtual environment. Run all commands from the project root directory.**

### Environment Setup

```bash
# Initialize/sync the virtual environment (uv handles this automatically)
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Install the package in editable mode (usually not needed with uv)
uv pip install -e .
```

### Testing

```bash
# Run all tests (most common - use this for verification)
uv run pytest

# Run specific test module
uv run pytest tests/test_environment.py

# Run tests for a specific layout engine
uv run pytest tests/layout/test_regular_grid_utils.py

# Run with verbose output (helpful for debugging)
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src/neurospatial

# Run a specific test function
uv run pytest tests/test_environment.py::test_function_name -v

# Run doctests (validate docstring examples)
uv run pytest --doctest-modules src/neurospatial/

# Run tests matching a pattern
uv run pytest -k "test_bin_size"
```

### Running the Package

```bash
# Run any Python script (from project root)
uv run python path/to/script.py

# Import in interactive session
uv run python -c "from neurospatial import Environment; print(Environment)"
```

### Python REPL

```bash
# Start interactive Python session with package available
uv run python

# Or use IPython if installed (recommended for exploration)
uv run ipython
```

### Code Quality

```bash
# Run ruff linter (check for issues)
uv run ruff check .

# Run ruff formatter (auto-format code)
uv run ruff format .

# Run both check and format (common workflow)
uv run ruff check . && uv run ruff format .

# Check specific file
uv run ruff check src/neurospatial/environment.py
```

## Key Implementation Notes

### Creating New Layout Engines

1. Implement the `LayoutEngine` protocol in `src/neurospatial/layout/engines/`
2. Populate required attributes in your `build()` method
3. Add to `_LAYOUT_MAP` in [layout/factories.py:17-25](src/neurospatial/layout/factories.py#L17-L25)
4. Ensure graph nodes/edges have mandatory metadata
5. Test boundary detection works with your layout in `layout/helpers/utils.py`
6. Add tests in `tests/layout/` following existing patterns

### Dependencies

Core dependencies:

- `numpy` - Array operations and numerical computing
- `pandas` - Data structures and analysis
- `matplotlib` - Plotting and visualization
- `networkx` - Graph data structures for connectivity
- `scipy` - Scientific computing (KDTree, morphological operations)
- `scikit-learn` - Machine learning utilities (KDTree)
- `shapely` - Geometric operations and polygon support
- `track-linearization` - 1D track linearization for GraphLayout

Development dependencies:

- `pytest` - Testing framework
- `pytest-cov` - Test coverage reporting
- `ruff` - Fast Python linter and formatter
- `ipython` - Enhanced interactive Python shell

### Animation Overlay Architecture (v0.4.0+)

The overlay system provides three public dataclasses for visualizing animal behavior alongside spatial fields:

**Public API** (`src/neurospatial/animation/overlays.py`):
- `PositionOverlay` - Trajectory tracking with decaying trails
- `BodypartOverlay` - Pose tracking with skeleton rendering (dict of bodypart trajectories)
- `HeadDirectionOverlay` - Orientation arrows (angles or unit vectors)

**Conversion funnel**:
1. User creates overlay dataclasses with behavioral data
2. `_convert_overlays_to_data()` validates and aligns overlays to field frames
3. Temporal interpolation handles mixed sampling rates (e.g., 120 Hz → 10 Hz)
4. Outputs pickle-safe `OverlayData` for backend rendering

**Backend support**:
- Napari: Full support (all overlay types + regions)
- Video: Full support (all overlay types + regions)
- HTML: Partial support (position + regions only; warns for bodypart/head direction)
- Widget: Full support (reuses video backend renderer)

**Key patterns**:
- Overlays accept optional `times` parameter for temporal alignment
- `frame_times` parameter on `animate_fields()` enables mixed-rate synchronization
- Multi-animal: Pass multiple overlay instances in a list
- Regions: Use `show_regions=True` or `show_regions=["region1", "region2"]`
- Parallel rendering: Call `env.clear_cache()` before `n_workers > 1`

## Testing Structure

Tests mirror source structure:

- `tests/test_environment.py` - Core `Environment` class tests
- `tests/test_composite.py` - `CompositeEnvironment` tests
- `tests/test_alignment.py` - Alignment/transformation tests
- `tests/layout/` - Layout engine-specific tests
- `tests/regions/` - Region functionality tests
- `tests/conftest.py` - Shared fixtures (plus maze, sample environments)

Fixtures in `conftest.py` provide common test environments (plus maze graphs, sample data).

## Documentation Style

### Docstring Format

**All docstrings MUST follow NumPy docstring format.** This is the standard for scientific Python projects and ensures consistency with the broader ecosystem.

#### NumPy Docstring Structure

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Optional longer description providing more context about what the
    function does, its behavior, and any important implementation details.

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

    Raises
    ------
    ValueError
        Description of when ValueError is raised.
    TypeError
        Description of when TypeError is raised.

    See Also
    --------
    related_function : Brief description of relation.

    Notes
    -----
    Additional technical information, implementation notes, or mathematical
    details.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
```

#### Key NumPy Docstring Guidelines

1. **Section Headers**: Use underlines with dashes (`---`) matching section name length
2. **Type Annotations**: Include types after parameter names with colon separator
3. **Section Order**: Parameters → Returns → Raises → See Also → Notes → Examples
4. **Blank Lines**: One blank line between sections
5. **Examples**: Use `>>>` for interactive examples, show expected output
6. **Cross-references**: Use backticks for code elements: `Environment`, `bin_centers`
7. **Math**: Use LaTeX notation in Notes section when needed
8. **Arrays**: Specify shape in type: `NDArray[np.float64], shape (n_samples, n_dims)`

#### Common Patterns in This Codebase

**Class docstrings**:

```python
class Environment:
    """
    Short summary of the class.

    Longer description of purpose, key features, and design patterns.

    Attributes
    ----------
    name : str
        Description of name attribute.
    layout : LayoutEngine
        Description of layout attribute.

    See Also
    --------
    CompositeEnvironment : Related class.

    Examples
    --------
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> print(env.n_bins)
    100
    """
```

**Protocol/Interface docstrings**: Document expected behavior, not implementation
**Property docstrings**: Focus on what the property represents, include type info
**Factory method docstrings**: Emphasize parameters and typical usage patterns

#### Resources

- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [numpydoc package](https://github.com/numpy/numpydoc) for validation

## Common Gotchas

### 1. Always use `uv run`

**Problem**: Running Python commands directly uses the wrong environment.

❌ Wrong:

```bash
python script.py
pytest
pip install package
```

✅ Right:

```bash
uv run python script.py
uv run pytest
uv add package
```

### 2. Check `_is_fitted` state

**Problem**: Calling spatial query methods on unfitted Environment raises error.

❌ Wrong:

```python
env = Environment()  # Not fitted!
env.bin_at([10.0, 5.0])  # RuntimeError
```

✅ Right:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Factory methods fit automatically
env.bin_at([10.0, 5.0])  # Works
```

### 3. Graph metadata is mandatory

**Problem**: Missing node/edge attributes cause failures in spatial queries.

**Required node attributes**: `'pos'`, `'source_grid_flat_index'`, `'original_grid_nd_index'`
**Required edge attributes**: `'distance'`, `'vector'`, `'edge_id'`, `'angle_2d'` (optional)

All layout engines must populate these. If creating custom graphs, ensure all attributes present.

### 4. Regions are immutable

**Problem**: Trying to modify Region objects in place fails.

❌ Wrong:

```python
env.regions['goal'].point = new_point  # AttributeError - immutable
```

⚠️ Discouraged (emits warning):

```python
env.regions['goal'] = new_region  # UserWarning - overwriting existing region
```

✅ Right:

```python
env.regions.update_region('goal', point=new_point)  # Creates new Region, no warning
env.regions.add('new_goal', point=point)  # Add new region
del env.regions['old_goal']  # Delete existing
```

### 5. Check `is_1d` before linearization

**Problem**: Calling `to_linear()` on N-D environments fails.

❌ Wrong:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Creates 2D grid
linear_pos = env.to_linear(position)  # AttributeError
```

✅ Right:

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    # Use N-D spatial queries instead
    bin_idx = env.bin_at(position)
```

### 6. Protocol, not inheritance

**Problem**: Layout engines don't inherit from a base class.

❌ Wrong:

```python
class MyLayout(LayoutEngine):  # LayoutEngine is a Protocol, not a class
    pass
```

✅ Right:

```python
class MyLayout:
    """Implements LayoutEngine protocol."""
    def build(self, ...): ...
    def point_to_bin_index(self, ...): ...
    # Implement all required methods and attributes
```

### 7. NumPy docstrings required

**Problem**: Using Google or reStructuredText style docstrings inconsistent with codebase.

❌ Wrong:

```python
def foo(x, y):
    """Does foo.

    Args:
        x: First parameter
        y: Second parameter
    """
```

✅ Right:

```python
def foo(x, y):
    """Does foo.

    Parameters
    ----------
    x : type
        First parameter
    y : type
        Second parameter
    """
```

### 8. bin_size is required

**Problem**: Forgetting bin_size parameter causes TypeError.

❌ Wrong:

```python
env = Environment.from_samples(data)  # TypeError: missing required argument
```

✅ Right:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Explicit is better
```

**Tip**: Choose bin_size based on your data's spatial scale and units (cm, meters, pixels, etc.)

### 9. Error messages show diagnostics

**What this means**: When validation fails, error messages include the actual invalid values to help debugging. Use these diagnostics to understand what went wrong.

Example:

```
ValueError: bin_size must be positive (got -2.0)
ValueError: No active bins found. Data range: [0.0, 100.0], bin_size: 200.0
```

The diagnostic values help identify the problem immediately.

### 10. Memory safety checks (v0.2.1+)

**Problem**: Creating very large grids can cause unexpected memory usage.

**Solution**: Grid creation now includes automatic memory estimation and warnings:

- **Warning at 100MB**: Large grid detected, creation proceeds but you're informed of memory usage

⚠️ Creates large grid (will warn but succeed):

```python
positions = np.random.uniform(0, 100000, (1000, 2))
env = Environment.from_samples(positions, bin_size=1.0)  # ResourceWarning, but succeeds
```

✅ Better (reduce grid size to avoid warning):

```python
# Option 1: Increase bin_size
env = Environment.from_samples(positions, bin_size=10.0)  # Smaller grid, no warning

# Option 2: Filter active bins
env = Environment.from_samples(positions, bin_size=1.0, infer_active_bins=True)

# Option 3: Disable warning (if intentional and you have RAM)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

**Tip**: Warning messages include estimated memory and suggestions for reducing usage.

### 11. Overlay temporal alignment (v0.4.0+)

**Problem**: Overlays and fields at different sampling rates need explicit timestamps.

❌ Wrong (no timestamps for mixed-rate data):

```python
# Position tracked at 120 Hz, fields at 10 Hz - without timestamps
position_overlay = PositionOverlay(data=trajectory_120hz)  # No times!
env.animate_fields(fields_10hz, overlays=[position_overlay])  # Mismatch!
```

✅ Right (provide timestamps for alignment):

```python
# Position tracked at 120 Hz, fields at 10 Hz - with timestamps
position_overlay = PositionOverlay(
    data=trajectory_120hz,
    times=timestamps_120hz  # Overlay timestamps
)
env.animate_fields(
    fields_10hz,
    overlays=[position_overlay],
    frame_times=timestamps_10hz  # Field timestamps - auto interpolation
)
```

**Tip**: Linear interpolation automatically aligns overlay to field frames.

### 12. HTML backend overlay limitations (v0.4.0+)

**Problem**: HTML backend only supports position and region overlays.

⚠️ Will warn (HTML doesn't support bodypart/head direction):

```python
env.animate_fields(
    fields,
    overlays=[bodypart_overlay, head_direction_overlay],  # Not supported in HTML!
    backend="html"
)
# UserWarning: HTML backend does not support bodypart overlays. Use video or napari backend.
```

✅ Right (use supported overlays or different backend):

```python
# Option 1: Use only position + regions with HTML
env.animate_fields(
    fields,
    overlays=[position_overlay],
    show_regions=True,
    backend="html"
)

# Option 2: Use video/napari for full overlay support
env.animate_fields(
    fields,
    overlays=[bodypart_overlay, head_direction_overlay],
    backend="napari"  # or "video" or "widget"
)
```

**Backend capability matrix**:
- Napari: All overlays ✓
- Video: All overlays ✓
- Widget: All overlays ✓
- HTML: Position + regions only ⚠️

## Troubleshooting

### `ModuleNotFoundError: No module named 'neurospatial'`

**Cause**: Dependencies not installed or wrong Python environment.

**Solution**:

```bash
# Sync dependencies (run from project root)
uv sync

# Verify environment
uv run python -c "import neurospatial; print(neurospatial.__file__)"
```

### Tests fail with import errors

**Cause**: Running pytest without `uv run` prefix.

**Solution**:

```bash
# Wrong
pytest

# Right
uv run pytest
```

### `RuntimeError: Environment must be fitted before calling this method`

**Cause**: Calling spatial query methods on unfitted Environment.

**Solution**: Use factory methods, not bare `Environment()`:

```python
# Wrong
env = Environment()
env.bin_at([10, 5])

# Right
env = Environment.from_samples(positions, bin_size=2.0)
env.bin_at([10, 5])
```

### UserWarning when overwriting a region

**Cause**: Using assignment to overwrite an existing region.

**Solution**: Assignment works but emits a `UserWarning`. Use `update_region()` to suppress the warning:

```python
# Works but emits UserWarning
env.regions['goal'] = new_region  # UserWarning: Overwriting existing region 'goal'

# Preferred - no warning
env.regions.update_region('goal', point=new_point)
```

**Note**: This warning follows standard dict semantics while preventing accidental overwrites. To suppress the warning without using `update_region()`, use Python's warnings filter.

### `AttributeError: 'Environment' object has no attribute 'to_linear'`

**Cause**: Calling `to_linear()` on N-D environment. Only 1D (GraphLayout) environments support linearization.

**Solution**: Check `is_1d` first:

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    bin_idx = env.bin_at(position)  # Use this for N-D
```

### `ValueError: No active bins found`

**Cause**: bin_size too large, threshold too high, or data too sparse.

**Solution**: Read the detailed error message - it provides diagnostics:

- Data range and extent
- Grid shape and bin_size used
- Suggested fixes (reduce bin_size, lower threshold, enable morphological operations)

Example fix:

```python
# If bin_size is too large
env = Environment.from_samples(positions, bin_size=1.0)  # Reduce from 10.0

# If threshold is too high
env = Environment.from_samples(positions, bin_size=2.0, bin_count_threshold=1)

# If data is sparse
env = Environment.from_samples(positions, bin_size=2.0, dilate=True, fill_holes=True)
```

### Pre-commit hooks fail on commit

**Cause**: Linting or formatting issues in code.

**Solution**: Let hooks auto-fix, then commit again:

```bash
git commit -m "message"
# Hooks run and fix files
git add .  # Stage the fixes
git commit -m "message"  # Commit again
```

Or manually run checks before committing:

```bash
uv run ruff check . && uv run ruff format .
git add .
git commit -m "message"
```

### Slow test execution

**Cause**: Running tests without parallelization.

**Solution**: Install pytest-xdist and use parallel execution:

```bash
uv add --dev pytest-xdist
uv run pytest -n auto  # Use all CPU cores
```

### `ResourceWarning: Creating large grid` (v0.2.1+)

**Cause**: Grid estimated to use >100MB memory (warning threshold).

**Solution**: This is a warning, not an error. Grid creation will proceed, but you're being informed about memory usage. Consider:

- Is this grid size intentional?
- Can you increase `bin_size` to reduce resolution?
- Would `infer_active_bins=True` help filter unused bins?

**Common fixes:**

```python
# Fix 1: Increase bin_size (most common)
env = Environment.from_samples(positions, bin_size=10.0)  # Instead of 1.0

# Fix 2: Enable active bin filtering
env = Environment.from_samples(
    positions,
    bin_size=1.0,
    infer_active_bins=True,
    bin_count_threshold=1
)

# Fix 3: Disable warning (if intentional and you have sufficient RAM)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

**To suppress the warning globally:**

```python
import warnings
warnings.filterwarnings('ignore', category=ResourceWarning)
```

**Note**: The memory estimate is conservative but approximate. Actual usage may vary by ±20%.

### Type errors despite correct code

**Cause**: May be using outdated type stubs or IDE not recognizing runtime checks.

**Note**: This project includes a `py.typed` marker (v0.2.1+) for type checking support. If you encounter type errors, ensure you're using the latest version. IDE warnings may be false positives that can be ignored if tests pass.
