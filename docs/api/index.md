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

### [neurospatial.ops.alignment](neurospatial/ops/alignment.md)

Transform and align spatial representations.

**Key Functions:**

- `map_probabilities()`: Align probability distributions between environments
- `get_2d_rotation_matrix()`: Create 2D rotation matrices

### [neurospatial.ops.transforms](neurospatial/ops/transforms.md)

2D affine transformations.

**Key Classes:**

- `Affine2D`: Composable 2D affine transformations

### [neurospatial.encoding](neurospatial/encoding/index.md)

Neural encoding analyses: how neurons represent space, direction,
egocentric targets, and gaze.

**Key Modules:**

- `encoding.spatial`: Place fields and spatial firing-rate maps
- `encoding.grid`: Grid cells (autocorrelation, gridness score)
- `encoding.directional`: Head-direction cells and tuning curves
- `encoding.border`: Border / boundary cell metrics
- `encoding.egocentric`: Object-vector cells and egocentric tuning
- `encoding.view`: Spatial-view cells and gaze-based fields
- `encoding.phase_precession`: Theta-phase precession
- `encoding.population`: Population-level metrics

**Key Functions:**

- `compute_spatial_rate()` / `compute_spatial_rates()`: Build firing-rate maps
- `compute_directional_rate()`: Head-direction tuning curve
- `compute_egocentric_rate()`: Object-vector tuning
- `compute_view_rate()`: Spatial-view tuning
- `detect_place_fields()`: Threshold-and-cluster on a rate map
- `spatial_information()`, `sparsity()`, `selectivity()`, `border_score()`,
  `grid_score()`: Classic place / boundary / grid metrics

### [neurospatial.decoding](neurospatial/decoding/index.md)

Bayesian decoding of position from spike counts.

**Key Functions:**

- `decode_position()`: Single-step or sequential Bayesian decoder
- `decoding_error()`, `median_decoding_error()`: Accuracy metrics
- `fit_isotonic_trajectory()`, `fit_linear_trajectory()`: Detect
  trajectory structure in posteriors

**Key Classes:**

- `DecodingResult`: Posterior + helpers (MAP, mean, entropy)

### [neurospatial.behavior](neurospatial/behavior/index.md)

Behavioral analysis built on trajectory + environment.

**Key Modules:**

- `behavior.trajectory`: Step lengths, MSD, curvature, home range
- `behavior.navigation`: Path efficiency, goal-directed metrics,
  graph turn sequences
- `behavior.segmentation`: Laps, trials, region crossings,
  velocity-based segmentation
- `behavior.decisions` / `behavior.vte`: Decision-point analysis,
  vicarious trial-and-error
- `behavior.reward`: Reward-field construction

### [neurospatial.events](neurospatial/events/index.md)

Peri-event spike alignment and GLM regressors.

**Key Functions:**

- `peri_event_histogram()`: PSTH around discrete events
- `align_spikes_to_events()`: Per-trial spike rasters
- `time_to_nearest_event()`, `distance_to_reward()`: GLM regressors

### [neurospatial.ops.egocentric](neurospatial/ops/egocentric.md)

Allocentric ↔ egocentric coordinate transforms.

**Key Functions:**

- `heading_from_velocity()`, `heading_from_body_orientation()`:
  Derive head direction from tracking data
- `allocentric_to_egocentric()`, `egocentric_to_allocentric()`:
  Frame conversions
- `compute_egocentric_bearing()`, `compute_egocentric_distance()`:
  Relate animal to external targets

### [neurospatial.ops.visibility](neurospatial/ops/visibility.md)

Visibility / gaze / viewshed computations.

**Key Classes:**

- `FieldOfView`: Symmetric / rat / primate FOV presets
- `ViewshedResult`: Visible bins + visibility fraction

**Key Functions:**

- `compute_viewshed()`, `compute_view_field()`,
  `compute_viewshed_trajectory()`: Viewshed from positions / headings
- `compute_viewed_location()`: Gaze-direction projection
- `visibility_occupancy()`, `visible_cues()`: View-time aggregates

### [neurospatial.ops.basis](neurospatial/ops/basis.md)

Spatial basis functions for regression / kernel methods.

**Key Functions:**

- `select_basis_centers()`: K-means / random / farthest-point placement
- `geodesic_rbf_basis()`, `heat_kernel_wavelet_basis()`,
  `chebyshev_filter_basis()`: Graph-aware kernels
- `spatial_basis()`, `plot_basis_functions()`: Unified spatial-basis API

### [neurospatial.stats](neurospatial/stats/index.md)

Circular statistics, surrogates, and shuffle controls.

**Key Modules:**

- `stats.circular`: Rayleigh test, circular mean / variance / R,
  circular-circular and circular-linear correlations, circular basis
- `stats.shuffle`: Position / cell / posterior shuffles with
  reproducibility helpers
- `stats.surrogates`: Poisson and inhomogeneous-Poisson surrogates,
  jittered-spike controls

### [neurospatial.animation](neurospatial/animation/index.md)

Field animation backends (napari / video / HTML / widget).

**Key Function:**

- `animate_fields()`: Single entry point; dispatches across backends

**Key Classes:**

- `PositionOverlay`, `EventOverlay`, `SpikeOverlay`,
  `HeadDirectionOverlay`, `BodypartOverlay`, `VideoOverlay`:
  Composable overlays for animations

### [neurospatial.io.nwb](neurospatial/io/nwb/index.md)

NWB (Neurodata Without Borders) read / write integration.

**Key Functions:**

- `read_environment()`, `read_position()`, `read_pose()`,
  `read_events()`, `read_intervals()`, `read_trials()`: Read NWB
  components into neurospatial types
- `write_environment()`, `write_place_field()`,
  `write_occupancy()`, `write_events()`, `write_laps()`,
  `write_region_crossings()`, `write_trials()`: Persist neurospatial
  results back into an NWBFile
- `environment_from_position()`: Build an environment directly from
  the position channel of an NWB file

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

- [Simulation Workflows Tutorial](../examples/15_simulation_workflows.ipynb): Comprehensive examples and quick start guide

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
