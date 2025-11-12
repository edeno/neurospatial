# API Reference

Complete API documentation for neurospatial, automatically generated from source code docstrings.

## Core Modules

### [neurospatial.environment](neurospatial/environment/index.md)

The main `Environment` class and related functionality (modular package).

**Key Classes:**

- `Environment`: Main class for discretized spatial environments

**Modules:**

- `environment.core`: Core dataclass with state and properties
- `environment.factories`: Factory classmethods for creating environments
- `environment.queries`: Spatial query methods
- `environment.trajectory`: Trajectory analysis methods
- `environment.fields`: Spatial field operations
- `environment.metrics`: Environment metrics and properties

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

### [neurospatial.simulation](neurospatial/simulation/index.md) <span style="color: #4CAF50;">v0.2.0+</span>

Generate synthetic spatial data, neural activity, and spike trains for testing and validation.

**Key Modules:**

- `simulation.trajectory`: Trajectory generation (OU process, structured laps)
- `simulation.models`: Neural models (place cells, boundary cells, grid cells)
- `simulation.spikes`: Spike generation (Poisson process, refractory periods)
- `simulation.session`: High-level session simulation API
- `simulation.validation`: Automated validation against ground truth
- `simulation.examples`: Pre-configured example sessions

**Key Classes:**

- `PlaceCellModel`: Gaussian place field model with ground truth
- `BoundaryCellModel`: Distance-tuned boundary/border cell model
- `GridCellModel`: Hexagonal grid cell model (2D only)
- `SimulationSession`: Complete simulation session dataclass

**Key Functions:**

- `simulate_trajectory_ou()`: Ornstein-Uhlenbeck process for realistic exploration
- `simulate_trajectory_sinusoidal()`: Sinusoidal movement for 1D tracks
- `simulate_trajectory_laps()`: Structured lap-based trajectories
- `generate_poisson_spikes()`: Generate spikes from firing rates
- `generate_population_spikes()`: Generate spikes for neuron populations
- `simulate_session()`: One-call workflow for complete sessions
- `validate_simulation()`: Compare detected fields to ground truth
- `open_field_session()`, `linear_track_session()`, etc.: Pre-configured examples

**See Also:**

- [Simulation Workflows Tutorial](../examples/15_simulation_workflows.ipynb): Comprehensive examples
- [README Simulation Section](../README.md#simulation-v020): Quick start guide

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
