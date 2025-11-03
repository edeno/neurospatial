# API Reference

Complete API documentation for neurospatial, automatically generated from source code docstrings.

## Core Modules

### [neurospatial.environment](neurospatial/environment.md)

The main `Environment` class and related functionality.

**Key Classes:**

- `Environment`: Main class for discretized spatial environments

### [neurospatial.composite](neurospatial/composite.md)

Merge multiple environments into composite structures.

**Key Classes:**

- `CompositeEnvironment`: Combine multiple environments with automatic bridge inference

### [neurospatial.regions](neurospatial/regions/index.md)

Define and manage named regions of interest (ROIs).

**Key Classes:**

- `Region`: Immutable point or polygon region
- `Regions`: Container for managing multiple regions

### [neurospatial.layout](neurospatial/layout/index.md)

Layout engines for discretizing continuous space.

**Key Modules:**

- `layout.base`: `LayoutEngine` protocol definition
- `layout.engines.*`: Concrete layout implementations
- `layout.factories`: Factory functions for creating layouts

### [neurospatial.alignment](neurospatial/alignment.md)

Transform and align spatial representations.

**Key Functions:**

- `map_probabilities_to_nearest_target_bin()`: Align probability distributions between environments
- `get_2d_rotation_matrix()`: Create 2D rotation matrices

### [neurospatial.transforms](neurospatial/transforms.md)

2D affine transformations.

**Key Classes:**

- `Affine2D`: Composable 2D affine transformations

## Layout Engines

Detailed documentation for each layout engine:

- [RegularGridLayout](neurospatial/layout/engines/regular_grid.md)
- [HexagonalLayout](neurospatial/layout/engines/hexagonal.md)
- [GraphLayout](neurospatial/layout/engines/graph.md)
- [MaskedGridLayout](neurospatial/layout/engines/masked_grid.md)
- [ShapelyPolygonLayout](neurospatial/layout/engines/shapely_polygon.md)
- [TriangularMeshLayout](neurospatial/layout/engines/triangular_mesh.md)
- [ImageMaskLayout](neurospatial/layout/engines/image_mask.md)

## Navigation

Use the sidebar to browse the complete API reference, or search for specific functions, classes, or methods.

## Docstring Format

All docstrings follow [NumPy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency with the scientific Python ecosystem.
