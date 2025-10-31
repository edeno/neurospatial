# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

2. **Environment** (`src/neurospatial/environment.py`)
   - Main user-facing class wrapping a `LayoutEngine` instance
   - Factory methods for common use cases:
     - `Environment.from_samples()` - Discretize point data into bins
     - `Environment.from_graph()` - Create 1D track-based environments
     - `Environment.from_polygon()` - Grid masked by Shapely polygon
     - `Environment.from_mask()` - Pre-defined N-D boolean mask
     - `Environment.from_image()` - Binary image mask
     - `Environment.from_layout()` - Direct layout specification
   - Provides spatial queries: `bin_at()`, `contains()`, `neighbors()`, `distance_between()`, `shortest_path()`
   - Integrates `Regions` for defining named ROIs (regions of interest)
   - Uses `@check_fitted` decorator ([environment.py:42-63](src/neurospatial/environment.py#L42-L63)) to ensure methods are only called after initialization

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

# Layout engines and factories
from neurospatial.layout.factories import create_layout, list_available_layouts
from neurospatial.layout.engines.regular_grid import RegularGridLayout

# Utility functions
from neurospatial.alignment import get_2d_rotation_matrix, map_probabilities_to_nearest_target_bin
from neurospatial.transforms import Affine2D
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

### Protocol-Based Design

Layout engines implement the `LayoutEngine` Protocol ([layout/base.py:10-166](src/neurospatial/layout/base.py#L10-L166)), not inheritance. When creating new engines:
- Implement required attributes: `bin_centers`, `connectivity`, `dimension_ranges`, `is_1d`, `_layout_type_tag`, `_build_params_used`
- Implement required methods: `build()`, `point_to_bin_index()`, `bin_sizes()`, `plot()`
- Optionally provide grid-specific attributes: `grid_edges`, `grid_shape`, `active_mask`

### Fitted State Pattern

`Environment` uses a `_is_fitted` flag set after `_setup_from_layout()` completes. Methods requiring fitted state use the `@check_fitted` decorator. This prevents calling spatial query methods before the environment is properly initialized.

### Regions are Immutable

`Region` objects are immutable dataclasses - create new instances rather than modifying existing ones. The `Regions` container uses dict-like semantics:
- Use `regions.add()` to create and insert
- Use `del regions[name]` or `regions.remove(name)` to delete
- Trying to set an existing key raises `KeyError` - use explicit update patterns

### 1D vs N-D Environments

Environments can be 1D (linearized tracks) or N-D (grids):
- 1D: `GraphLayout` with `is_1d=True`, provides `to_linear()` and `linear_to_nd()` methods
- N-D: Grid-based layouts with spatial queries in original coordinate space

Check `env.is_1d` before calling linearization methods.

## Development Commands

**IMPORTANT: All commands below MUST be prefixed with `uv run` to ensure they execute in the correct virtual environment.**

### Environment Setup
```bash
# Initialize/sync the virtual environment (uv handles this automatically)
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Install the package in editable mode
uv pip install -e .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test module
uv run pytest tests/test_environment.py

# Run tests for a specific layout engine
uv run pytest tests/layout/test_regular_grid_utils.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/neurospatial

# Run a specific test
uv run pytest tests/test_environment.py::test_function_name -v
```

### Running the Package
```bash
# Run main entry point
uv run python main.py

# Run any Python script
uv run python path/to/script.py
```

### Python REPL
```bash
# Start interactive Python session with package available
uv run python

# Or use IPython if installed
uv run ipython
```

### Code Quality
```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Run both check and format
uv run ruff check . && uv run ruff format .
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
    >>> env = Environment.from_samples(data, bin_size=2.0)
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

1. **Always use `uv run`** - Don't run Python commands directly
2. **Check `_is_fitted`** - Many Environment methods require fitted state
3. **Graph metadata is mandatory** - All nodes/edges must have required attributes
4. **Regions are immutable** - Don't try to modify Region objects in place
5. **Check `is_1d`** - Only GraphLayout supports linearization methods
6. **Protocol, not inheritance** - Layout engines implement protocol, don't inherit from base class
7. **NumPy docstrings** - Use NumPy docstring format for all documentation, not Google or reStructuredText style
