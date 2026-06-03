# API Reference

Import patterns organized by feature area.

**Updated** - Domain-centric package reorganization.

---

## Function Argument Conventions

### Canonical Argument Names

Use these standardized names consistently across all modules:

| Concept | Canonical Name | NOT | Notes |
|---------|----------------|-----|-------|
| Continuous coordinates | `positions` | `trajectory`, `coords`, `xy` | Shape: (n_time, n_dims) |
| Discretized bin indices | `position_bins` | `trajectory_bins`, `bins`, `bin_idx` | Shape: (n_time,), dtype=int64 |
| Sample timestamps | `times` | `timestamps`, `t`, `time_array` | Shape: (n_time,), in seconds |
| Spike event times | `spike_times` | `spikes`, `spike_train` | Shape: (n_spikes,), in seconds |
| Animal heading | `headings` | `heading`, `head_direction`, `hd` | Shape: (n_time,), in radians |
| Target locations | `targets` | `target_positions`, `objects` | For egocentric operations |
| Object locations | `object_positions` | `objects`, `landmarks` | For encoding functions |
| Smoothing kernel size | `bandwidth` | `sigma`, `smoothing` | In physical units (e.g., cm) |
| Estimation algorithm | `smoothing_method` | `method`, `estimator` | For place field computation |
| Distance algorithm | `metric` | `distance_metric`, `distance_type`, `use_geodesic` | "euclidean" or "geodesic" |

### Canonical Argument Order by Function Type

#### Neural Encoding Functions (place fields, object-vector, spatial view)

```python
func(
    env,                    # 1. Environment (spatial context)
    spike_times,            # 2. Neural data (what fired)
    times,                  # 3. Timestamps (when sampled)
    positions,              # 4. Position coordinates (where animal was)
    headings,               # 5. Head direction (which way facing) - if egocentric
    object_positions,       # 6. External targets - if relevant
    *,                      # 7. Keyword-only separator
    smoothing_method=...,   # 8. Algorithm parameters
    bandwidth=...,
    min_occupancy=...,
)
```

**Examples:**

- `compute_spatial_rate(env, spike_times, times, positions, *, smoothing_method=...)`
- `compute_egocentric_rate(env, spike_times, times, positions, headings, object_positions, *, ...)`
- `compute_view_rate(env, spike_times, times, positions, headings, *, ...)`
- `compute_directional_rate(spike_times, times, headings, *, ...)`

#### Egocentric Operations (bearing, distance to targets)

```python
func(
    positions,              # 1. Animal positions (where animal is)
    headings,               # 2. Animal headings (which way facing)
    targets,                # 3. Target locations (what animal relates to)
)
```

**Examples:**

- `compute_egocentric_bearing(positions, headings, targets)`
- `allocentric_to_egocentric(positions, headings, points)`

#### Behavioral Segmentation (laps, trials, crossings)

```python
func(
    position_bins,          # 1. Discretized position indices
    times,                  # 2. Timestamps
    env,                    # 3. Environment (for graph/region lookups)
    *,                      # 4. Keyword-only separator
    region_params...,       # 5. Region specifications
)
```

**Alternative for functions requiring continuous positions:**

```python
func(
    positions,              # 1. Continuous position coordinates
    times,                  # 2. Timestamps
    env,                    # 3. Environment
    *,                      # 4. Keyword-only separator
    source=...,             # 5. Region parameters
    target=...,
)
```

**Examples:**

- `segment_trials(position_bins, times, env, *, start_region=..., end_regions=...)`
- `detect_laps(position_bins, times, env, *, method=..., min_overlap=...)`
- `detect_runs_between_regions(positions, times, env, *, source=..., target=...)`

#### Head Direction Functions

Head direction functions are an exception - they don't require `env` because head direction is independent of spatial discretization:

```python
func(
    spike_times,            # 1. Spike times
    times,                  # 2. Sample timestamps
    head_directions,        # 3. HD at each timepoint
    *,                      # 4. Keyword-only separator
    bin_size=...,           # 5. Angular bin size
    angle_unit=...,
)
```

**Examples:**

- `compute_directional_rate(spike_times, times, headings, *, bin_size=...)`
- `is_head_direction_cell(spike_times, times, headings, **kwargs)`

#### Events/Peri-Event Functions

Event functions analyze temporal alignment without spatial context:

```python
func(
    spike_times,            # 1. Spike times
    event_times,            # 2. Event timestamps
    window,                 # 3. Time window around events
    *,                      # 4. Keyword-only separator
    bin_size=...,           # 5. Temporal bin size
)
```

**Examples:**

- `peri_event_histogram(spike_times, event_times, window, *, bin_size=...)`
- `align_spikes_to_events(spike_times, event_times, window)`

### Keyword-Only Separator (`*`) Guidelines

Use `*` to force keyword-only arguments when:

1. Function has more than 3-4 positional arguments
2. Optional parameters follow required ones
3. Parameters control algorithm behavior (smoothing_method, bandwidth)
4. Parameters specify thresholds or ranges

**Good:**

```python
def compute_spatial_rate(
    env, spike_times, times, positions,
    *,  # Force keyword-only after core data
    smoothing_method="diffusion_kde",
    bandwidth=5.0,
): ...
```

**Avoid:**

```python
def compute_spatial_rate(
    env, spike_times, times, positions,
    smoothing_method="diffusion_kde",  # Can be passed positionally - error-prone
    bandwidth=5.0,
): ...
```

---

## Core Classes

```python
from neurospatial import Environment, EnvironmentNotFittedError
from neurospatial.regions import Region, Regions
from neurospatial import CompositeEnvironment
```

---

## Spatial Operations (ops/)

```python
from neurospatial.ops import (
    # Binning
    map_points_to_bins,             # Batch point-to-bin mapping with KDTree
    regions_to_mask,                # Convert regions to binary mask
    resample_field,                 # Resample field to different grid
    TieBreakStrategy,               # Enum for tie-breaking
    clear_kdtree_cache,             # Clear KDTree cache

    # Distance
    distance_field,                 # Multi-source geodesic distances
    pairwise_distances,             # Distances between node subsets
    euclidean_distance_matrix,      # Euclidean distance matrix
    geodesic_distance_matrix,       # Geodesic distance matrix
    geodesic_distance_between_points,  # Point-to-point geodesic
    neighbors_within,               # Neighbors within range

    # Transforms - Core classes
    Affine2D,                       # 2D affine transform
    Affine3D,                       # 3D affine transform
    AffineND,                       # ND affine transform
    SpatialTransform,               # Protocol for composable transforms
    VideoCalibration,               # Video calibration transform

    # Transforms - 2D factories
    translate,                      # Translation transform
    scale_2d,                       # 2D scale transform
    flip_y,                         # Flip Y axis
    identity,                       # Identity transform

    # Transforms - 3D factories
    translate_3d,                   # 3D translation
    scale_3d,                       # 3D scale transform
    from_rotation_matrix,           # From rotation matrix
    identity_nd,                    # ND identity transform

    # Transforms - Calibration
    estimate_transform,             # Estimate transform from point pairs
    apply_transform_to_environment, # Transform entire environment
    calibrate_from_landmarks,       # Calibrate from point pairs
    calibrate_from_scale_bar,       # Calibrate from scale bar
    simple_scale,                   # Simple scale factor

    # Transforms - Helpers
    convert_to_cm,                  # Convert to cm units
    convert_to_pixels,              # Convert to pixel units
    flip_y_data,                    # Flip Y in data array

    # Alignment
    get_2d_rotation_matrix,         # 2D rotation matrix
    map_probabilities,              # Map probabilities between envs
    apply_similarity_transform,     # Apply similarity transform
    ProbabilityMappingParams,       # Alignment parameters dataclass

    # Smoothing
    compute_diffusion_kernels,      # Diffusion kernels
    apply_kernel,                   # Apply kernel to field

    # Normalization
    normalize_field,                # Normalize field to sum=1
    clamp,                          # Clamp to range
    combine_fields,                 # Weighted combination

    # Graph operations
    graph_convolve,                 # Graph convolution
    neighbor_reduce,                # Neighborhood reduce

    # Calculus
    gradient,                       # Spatial gradient
    divergence,                     # Spatial divergence
    compute_differential_operator,  # Build differential operator
)
```

---

## Neural Encoding (encoding/)

### Canonical Rate API

Encoding uses consistent `compute_*_rate` names and returns rich result objects:

```python
from neurospatial.encoding import (
    # Spatial Rate (Place/Grid/Border Cells)
    compute_spatial_rate,                   # Single-neuron spatial rate map
    compute_spatial_rates,                  # Population spatial rate maps
    SpatialRateResult,                      # Result with firing_rate, occupancy, metrics
    SpatialRatesResult,                     # Population result

    # Directional Rate (Head Direction Cells)
    compute_directional_rate,               # Single-neuron HD tuning
    compute_directional_rates,              # Population HD tuning
    DirectionalRateResult,                  # Result with preferred_direction, MRL, etc.
    DirectionalRatesResult,                 # Population result

    # View Rate (Spatial View Cells)
    compute_view_rate,                      # Single-neuron view field
    compute_view_rates,                     # Population view fields
    ViewRateResult,                         # Result with occupancy, is_spatial_view_cell()
    ViewRatesResult,                        # Population result

    # Egocentric Rate (Object-Vector Cells)
    compute_egocentric_rate,                # Single-neuron egocentric polar field
    compute_egocentric_rates,               # Population egocentric fields
    EgocentricRateResult,                   # Result with preferred_distance(), preferred_direction()
    EgocentricRatesResult,                  # Population result

    # Metrics (available on result objects or standalone)
    spatial_information,                    # Spatial info (bits/spike)
    sparsity,                               # Spatial sparsity
    detect_place_fields,                    # Detect fields from rate map
    field_size,                             # Field size
    rate_map_centroid,                      # Field centroid
    field_stability,                        # Temporal stability
    field_shift_distance,                   # Shift between sessions

    # Helper functions
    compute_viewed_location,                # Compute viewed location from gaze
    compute_viewshed,                       # Compute visible bins
)
```

### Backend Parameter

All canonical encoding functions accept a `backend` parameter for computation:

```python
# Backend options
backend="numpy"   # Default: Works everywhere, including Windows
backend="jax"     # Requires JAX (Linux/macOS); raises ImportError if unavailable
backend="auto"    # Uses JAX if available, falls back to NumPy silently

# Example usage
result = compute_spatial_rate(
    env, spike_times, times, positions,
    backend="numpy",  # or "jax" or "auto"
)
```

**When to use each backend:**
- `"numpy"`: Default choice, works everywhere
- `"jax"`: For GPU acceleration or JAX-based pipelines (Linux/macOS only)
- `"auto"`: For portable code that uses JAX when available

### Result Class Methods

Result objects from the new API provide convenient methods:

**SpatialRateResult** (from `compute_spatial_rate`):
- `.firing_rate` - Firing rate map (n_bins,) in Hz
- `.occupancy` - Time in each bin (n_bins,) in seconds
- `.env` - Environment used for computation
- `.peak_location()` - Coordinates of peak firing (n_dims,)
- `.peak_firing_rate()` - Maximum firing rate (scalar)

**DirectionalRateResult** (from `compute_directional_rate`):
- `.firing_rate` - Tuning curve (n_bins,) in Hz
- `.occupancy` - Time at each direction (n_bins,) in seconds
- `.bin_centers` - Angular bin centers (n_bins,) in radians
- `.preferred_direction()` - Circular mean weighted by rate (radians)
- `.preferred_direction_deg()` - Same in degrees
- `.peak_firing_rate()` - Maximum firing rate
- `.mean_vector_length()` - Mean resultant length (tuning strength)
- `.tuning_width()` - Half-width at half-maximum (radians)
- `.plot(ax=None, polar=True)` - Plot tuning curve

**ViewRateResult** (from `compute_view_rate`):
- `.firing_rate` - View field (n_bins,) in Hz
- `.occupancy` - Time *viewing* each bin (n_bins,) in seconds
- `.env` - Environment used for computation
- `.gaze_model` - Gaze model used ("fixed_distance", "ray_cast", "boundary")
- `.view_distance` - Distance parameter for gaze model

**EgocentricRateResult** (from `compute_egocentric_rate`):
- `.firing_rate` - Egocentric polar field (n_bins,) in Hz
- `.occupancy` - Time in each egocentric bin (n_bins,) in seconds
- `.env` - Egocentric polar environment
- `.distance_range` - Distance range (min, max)
- `.n_distance_bins` - Number of distance bins
- `.n_direction_bins` - Number of direction bins

### Place Field Metrics

Use `compute_spatial_rate()` for rate-map estimation. Use the metric helpers
below when you already have a firing-rate map and need standalone analysis:

```python
from neurospatial.encoding import (
    detect_place_fields,
    spatial_information,
    sparsity,
    rate_map_centroid,
    field_size,
)
```

### Grid Cells

```python
from neurospatial.encoding.grid import (
    grid_score,                             # Grid cell grid score
    spatial_autocorrelation,                # 2D FFT autocorrelogram (regular grids)
    spatial_autocorrelation_radial,         # 1D distance profile (irregular topologies)
    grid_scale,                             # Grid spacing
    grid_properties,                        # Comprehensive properties (incl. orientation)
    periodicity_score,                      # Periodicity measure
    GridProperties,                         # Properties dataclass
)
```

### Head Direction Cells

```python
from neurospatial.encoding import (
    compute_directional_rate,               # HD tuning curve (returns DirectionalRateResult)
    compute_directional_rates,              # Population HD tuning
    DirectionalRateResult,                  # Result with preferred_direction(), mrl(), etc.
    DirectionalRatesResult,                 # Population result
)
```

For standalone circular statistics, import from `neurospatial.stats.circular`.

### Border/Boundary Cells

```python
from neurospatial.encoding.border import (
    border_score,                           # Border score
    compute_region_coverage,                # Region coverage stats
)
# Note: border_score parameter order changed to (env, firing_rate, ...) for consistency
```

### Object-Vector Cells

```python
from neurospatial.encoding import (
    compute_egocentric_rate,                # Egocentric polar field (returns EgocentricRateResult)
    compute_egocentric_rates,               # Population egocentric fields
    EgocentricRateResult,                   # Result with preferred_distance(), preferred_direction(), etc.
    EgocentricRatesResult,                  # Population result
)
```

Use result methods such as `preferred_distance()`, `preferred_direction()`, and
`is_object_vector_cell()` for classification workflows.

### Spatial View Cells

```python
from neurospatial.encoding import (
    compute_view_rate,                      # View field (returns ViewRateResult)
    compute_view_rates,                     # Population view fields
    ViewRateResult,                         # Result with occupancy, is_spatial_view_cell(), etc.
    ViewRatesResult,                        # Population result
    compute_viewed_location,                # Compute viewed location from gaze
    compute_viewshed,                       # Compute visible bins
)
```

Use `compute_view_rate()`/`compute_view_rates()` for view-field estimation and
the returned result methods for classification and summaries.

### Phase Precession

```python
from neurospatial.encoding.phase_precession import (
    phase_precession,                       # Phase precession analysis
    has_phase_precession,                   # Significance test
    plot_phase_precession,                  # Phase-position plot
    PhasePrecessionResult,                  # Result dataclass
)
```

### Population Metrics

```python
from neurospatial.encoding.population import (
    population_vector_correlation,          # PVC metric
    population_coverage,                    # Coverage by population
    count_place_cells,                      # Count by threshold
    field_density_map,                      # Field center density
    field_overlap,                          # Pairwise field overlap
    plot_population_coverage,               # Coverage visualization
    PopulationCoverageResult,               # Result dataclass
)
```

---

## Neural Decoding (decoding/)

```python
from neurospatial.decoding import (
    # Main entry point
    DecodingResult,                         # Result container class
    decode_position,                        # Main entry point
    decoding_error,                         # Per-time-bin position error
    median_decoding_error,                  # Summary statistic

    # Likelihood
    log_poisson_likelihood,
    poisson_likelihood,

    # Posterior
    normalize_to_posterior,

    # Point estimates
    posterior_mode,
    map_position,
    mean_position,
    posterior_entropy,
    credible_region,

    # Trajectory analysis
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    detect_trajectory_radon,                # Requires scikit-image
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,

    # Quality metrics
    confusion_matrix,
    decoding_correlation,
)

# Shuffle controls and surrogates live under stats/ — import from their canonical
# locations rather than from neurospatial.decoding (the re-exports were removed in v0.4).
from neurospatial.stats.shuffle import (
    ShuffleTestResult,
    compute_shuffle_pvalue,
    shuffle_cell_identity,
    shuffle_time_bins,
)
from neurospatial.stats.surrogates import generate_poisson_surrogates
```

---

## Basis Functions for GLMs (ops/basis)

```python
from neurospatial.ops.basis import (
    # Convenience function (recommended starting point)
    spatial_basis,              # Automatic parameter selection

    # Center selection
    select_basis_centers,       # kmeans, farthest_point, random methods

    # Basis types
    geodesic_rbf_basis,         # RBF using shortest-path distances
    heat_kernel_wavelet_basis,  # Diffusion-based multi-scale
    chebyshev_filter_basis,     # Polynomial filters with k-bin locality

    # Visualization
    plot_basis_functions,       # Visualize selected basis functions
)
```

---

## Statistical Methods (stats/)

### Circular Statistics

```python
from neurospatial.stats.circular import (
    # Core statistics
    rayleigh_test,                  # Rayleigh test for uniformity
    circular_mean,                  # Circular mean direction
    circular_variance,              # Circular variance
    mean_resultant_length,          # Mean resultant length

    # Correlations
    circular_linear_correlation,    # Circular-linear correlation
    circular_circular_correlation,  # Circular-circular correlation
    phase_position_correlation,     # Phase-position correlation

    # Utilities
    wrap_angle,                     # Wrap angle to [-π, π]

    # GLM circular basis
    circular_basis,                 # sin/cos basis for circular predictors
    CircularBasisResult,            # Result container with design_matrix
    circular_basis_metrics,         # Extract amplitude, phase, p-value from GLM
    is_modulated,                   # Quick significance check
    plot_circular_basis_tuning,     # Polar or linear tuning curve plots
)
```

### Shuffle Controls

```python
from neurospatial.stats.shuffle import (
    # Result dataclass
    ShuffleTestResult,

    # Shuffle methods
    shuffle_time_bins,              # Shuffle temporal order
    shuffle_time_bins_coherent,     # Coherent temporal shuffle
    shuffle_cell_identity,          # Shuffle cell labels
    shuffle_place_fields_circular,  # 1D circular place field shuffle
    shuffle_place_fields_circular_2d,  # 2D circular place field shuffle
    shuffle_posterior_circular,     # Posterior circular shuffle
    shuffle_posterior_weighted_circular,  # Weighted circular posterior
    shuffle_trials,                 # Shuffle trial labels
    shuffle_spikes_isi,             # Shuffle inter-spike intervals

    # P-value computation
    compute_shuffle_pvalue,         # P-value from null distribution
    compute_shuffle_zscore,         # Z-score from null distribution
)
```

### Surrogate Data Generation

```python
from neurospatial.stats.surrogates import (
    generate_poisson_surrogates,              # Homogeneous Poisson surrogates
    generate_inhomogeneous_poisson_surrogates,  # Inhomogeneous Poisson
    generate_jittered_spikes,                 # Temporal jitter surrogates
)
```

---

## Egocentric Reference Frames (ops/egocentric)

```python
from neurospatial.ops.egocentric import (
    # Core transforms
    EgocentricFrame,              # Single-timepoint frame dataclass
    allocentric_to_egocentric,    # Batch transform to egocentric
    egocentric_to_allocentric,    # Batch transform to allocentric

    # Bearing and distance
    compute_egocentric_bearing,   # Angle to targets relative to heading
    compute_egocentric_distance,  # Distance to targets (Euclidean or geodesic)

    # Heading computation
    heading_from_velocity,        # Heading from position timeseries
    heading_from_body_orientation, # Heading from nose/tail keypoints
)
```

**Egocentric polar environment:**

```python
from neurospatial import Environment

# Create polar grid in egocentric space (for object-vector cells).
# Returns EgocentricPolarEnvironment — a DISTINCT type, NOT a subclass of
# Environment. Cartesian-only methods (bin_at, contains, distance_between,
# distance_to(metric="euclidean"), apply_transform) raise NotImplementedError;
# graph ops (neighbors, path_between, reachable_from,
# distance_to(metric="geodesic"), smooth) work and use physical polar geometry.
env = Environment.from_polar_egocentric(
    distance_range=(0, 50),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=5.0,
    angle_bin_size=np.pi / 8,
    circular_angle=True,
)

# The concrete class lives at:
from neurospatial.environment.polar import EgocentricPolarEnvironment
assert isinstance(env, EgocentricPolarEnvironment)
assert not isinstance(env, Environment)
```

---

## Visibility and Gaze (ops/visibility)

```python
from neurospatial.ops.visibility import (
    # Field of view
    FieldOfView,                   # Species-specific FOV (rat, primate, etc.)
    ViewshedResult,                # Visible bins from position

    # Gaze computation
    compute_viewed_location,       # Where is animal looking?
    compute_viewshed,              # What bins are visible?
    compute_view_field,            # Binary visibility mask
    compute_viewshed_trajectory,   # Viewshed along trajectory
    visibility_occupancy,          # Time each bin was visible
    visible_cues,                  # Check cue visibility
)
```

---

## Simulation (simulation/)

```python
from neurospatial.simulation import (
    # Submodule
    mazes,                          # Pre-built maze environments

    # Cell models
    NeuralModel,                    # Protocol for neural models
    PlaceCellModel,                 # Place cell model
    BoundaryCellModel,              # Boundary/border cell model
    GridCellModel,                  # Grid cell model
    ObjectVectorCellModel,          # OVC model with Gaussian/von Mises tuning
    SpatialViewCellModel,           # Gaze-based firing model

    # Spike generation
    generate_poisson_spikes,        # Generate Poisson spike train
    generate_population_spikes,     # Generate population spike trains
    add_modulation,                 # Add temporal modulation to rates

    # Trajectory simulation
    simulate_trajectory_ou,         # Ornstein-Uhlenbeck random walk
    simulate_trajectory_laps,       # Lap-based trajectory
    simulate_trajectory_sinusoidal, # Sinusoidal motion

    # Session API
    SimulationSession,              # Session container class
    simulate_session,               # High-level session generation

    # Pre-configured example sessions
    open_field_session,             # Open field with place cells
    linear_track_session,           # Linear track session
    tmaze_alternation_session,      # T-maze alternation task
    boundary_cell_session,          # Boundary cell session
    grid_cell_session,              # Grid cell session

    # Validation and visualization
    validate_simulation,            # Validate simulation output
    plot_session_summary,           # Plot session summary
)

# Animation overlays for simulation
from neurospatial.animation import (
    ObjectVectorOverlay,           # Vectors from animal to objects
    ObjectVectorData,              # Internal data container
)
```

---

## Behavioral Analysis (behavior/)

### Trajectory Metrics

```python
from neurospatial.behavior.trajectory import (
    compute_step_lengths,           # Step lengths
    compute_turn_angles,            # Turn angles
    compute_home_range,             # MCP home range
    mean_square_displacement,       # MSD vs lag
    compute_trajectory_curvature,   # Continuous curvature
)
```

### Segmentation

```python
from neurospatial.behavior.segmentation import (
    # Dataclasses
    Trial,                          # Trial dataclass
    Crossing,                       # Crossing dataclass
    Lap,                            # Lap dataclass
    Run,                            # Run dataclass

    # Segmentation functions
    detect_laps,                    # Lap detection
    detect_region_crossings,        # Entry/exit events
    detect_runs_between_regions,    # Source→target runs
    detect_goal_directed_runs,      # Goal-directed segments
    segment_by_velocity,            # Movement/rest
    segment_trials,                 # Behavioral trials
    trajectory_similarity,          # Path similarity
)
```

### Navigation

```python
from neurospatial.behavior.navigation import (
    # Navigation functions
    path_progress,                  # Normalized progress (0→1) along path
    distance_to_region,             # Distance to goal over time
    cost_to_goal,                   # RL cost with terrain/avoidance
    time_to_goal,                   # Time until goal arrival
    graph_turn_sequence,            # Discrete turn labels
    trials_to_region_arrays,        # Helper for trial arrays
    goal_pair_direction_labels,     # Trial-based direction labels
    heading_direction_labels,       # Heading-based direction labels

    # Path efficiency
    PathEfficiencyResult,           # Path efficiency metrics
    SubgoalEfficiencyResult,        # Multi-waypoint efficiency
    traveled_path_length,           # Total distance traveled
    shortest_path_length,           # Geodesic/Euclidean distance
    time_efficiency,                # Ratio: T_optimal / T_actual
    angular_efficiency,             # 1 - mean(|delta_theta|) / pi
    subgoal_efficiency,             # Per-segment efficiency
    compute_path_efficiency,        # All metrics in one call

    # Goal-directed metrics
    GoalDirectedMetrics,            # Goal-directed navigation metrics
    goal_vector,                    # Vector from position to goal
    goal_direction,                 # Angle to goal
    instantaneous_goal_alignment,   # Cos(velocity, goal direction)
    goal_bias,                      # Mean alignment over trajectory
    approach_rate,                  # d/dt of distance to goal
    compute_goal_directed_metrics,  # All metrics in one call
)
```

### Decisions and VTE

```python
from neurospatial.behavior.decisions import (
    # Decision analysis dataclasses
    PreDecisionMetrics,             # Kinematics before decision region
    DecisionBoundaryMetrics,        # Boundary crossing metrics
    DecisionAnalysisResult,         # Complete trial analysis

    # Pre-decision functions
    decision_region_entry_time,     # First entry to region
    extract_pre_decision_window,    # Slice trajectory before entry
    pre_decision_heading_stats,     # Circular mean, variance, MRL
    pre_decision_speed_stats,       # Mean and min speed
    compute_pre_decision_metrics,   # Combined pre-decision metrics

    # Decision boundary functions
    geodesic_voronoi_labels,        # Label bins by nearest goal
    distance_to_decision_boundary,  # Distance to Voronoi edge
    detect_boundary_crossings,      # Find boundary crossing events
    compute_decision_analysis,      # Full decision analysis
)
```

### VTE (Vicarious Trial and Error)

```python
from neurospatial.behavior.vte import (
    # VTE dataclasses
    VTETrialResult,                 # Single trial VTE metrics
    VTESessionResult,               # Session-level VTE analysis

    # VTE functions
    head_sweep_magnitude,           # Sum of |delta_theta| (IdPhi)
    integrated_absolute_rotation,   # Alias for head_sweep_magnitude
    head_sweep_from_positions,      # IdPhi from trajectory
    normalize_vte_scores,           # Z-score across trials
    compute_vte_index,              # Combined VTE index
    classify_vte,                   # VTE if index > threshold
    compute_vte_trial,              # Single trial analysis
    compute_vte_session,            # Full session analysis
)
```

### Reward

```python
from neurospatial.behavior.reward import (
    goal_reward_field,              # Point goal reward
    region_reward_field,            # Region reward
)
```

---

## Events and Peri-Event Analysis (events/)

```python
from neurospatial.events import (
    # Result dataclasses
    PeriEventResult,                  # PSTH result container
    PopulationPeriEventResult,        # Population PSTH result

    # Peri-event analysis
    peri_event_histogram,             # Compute PSTH
    population_peri_event_histogram,  # Population PSTH
    align_spikes_to_events,           # Get per-trial spike times
    align_events,                     # Align events to reference events
    plot_peri_event_histogram,        # Plot PSTH results

    # GLM regressors (temporal)
    time_to_nearest_event,            # Signed time to nearest event
    event_count_in_window,            # Count events in window
    event_indicator,                  # Binary presence indicator

    # GLM regressors (spatial)
    distance_to_reward,               # Distance to reward location
    distance_to_boundary,             # Distance to walls/obstacles/regions

    # Spatial utilities
    add_positions,                    # Add x, y columns to events

    # Interval utilities
    intervals_to_events,              # Convert intervals to point events
    events_to_intervals,              # Pair start/stop events
    filter_by_intervals,              # Filter events by intervals

    # Validation
    validate_events_dataframe,        # Validate events DataFrame
    validate_spatial_columns,         # Check for spatial columns
)
```

---

## Animation & Visualization (animation/)

```python
from neurospatial.animation import (
    # Overlays
    PositionOverlay,                        # Trajectory with trail
    BodypartOverlay,                        # Pose tracking with skeleton
    HeadDirectionOverlay,                   # Orientation arrows
    EventOverlay,                           # Spikes, licks, rewards
    SpikeOverlay,                           # Alias for EventOverlay
    TimeSeriesOverlay,                      # Continuous variables
    ObjectVectorOverlay,                    # Vectors from animal to objects

    # Video
    VideoOverlay,                           # Video behind/above fields
    calibrate_video,                        # Video calibration helper

    # Configuration
    ScaleBarConfig,                         # Scale bar configuration
    estimate_colormap_range_from_subset,    # Large session helper
    large_session_napari_config,            # Napari config for large data

    # Utilities
    subsample_frames,                       # Subsample frame arrays
)

# VideoCalibration and calibrate_from_landmarks also available from neurospatial.ops
```

---

## Video Annotation (annotation/)

```python
from neurospatial.annotation import (
    annotate_video,          # Interactive annotation
    regions_from_labelme,    # Import from LabelMe
    regions_from_cvat,       # Import from CVAT
    annotate_track_graph,    # 1D track annotation
    TrackGraphResult,        # Track graph annotation result
    BoundaryConfig,          # Boundary inference config
    boundary_from_positions, # Auto-infer boundary from positions
)
```

---

## I/O and Serialization (io/)

```python
from neurospatial.io import (
    to_file,                 # Save environment to file
    from_file,               # Load environment from file
    to_dict,                 # Convert to dictionary
    from_dict,               # Load from dictionary
)
```

### NWB Integration

**Requires:** `uv add neurospatial[nwb-full]`

```python
from neurospatial.io.nwb import (
    # Reading
    read_position,           # Position → (positions, timestamps)
    read_head_direction,     # CompassDirection → (angles, timestamps)
    read_pose,               # PoseEstimation → (bodyparts, timestamps, skeleton)
    read_events,             # EventsTable → DataFrame
    read_intervals,          # TimeIntervals → DataFrame
    read_trials,             # Trials table → DataFrame
    read_environment,        # scratch/ → Environment

    # Writing
    write_place_field,       # Write to analysis/
    write_occupancy,         # Write to analysis/
    write_events,            # Write DataFrame to EventsTable
    write_trials,            # Write to intervals/trials/
    write_laps,              # Write to processing/behavior/
    write_region_crossings,  # Write to processing/behavior/
    write_environment,       # Write to scratch/

    # Factories
    environment_from_position,
    position_overlay_from_nwb,
    bodypart_overlay_from_nwb,
    head_direction_overlay_from_nwb,
)
```

---

## Layout Engines (layout/)

```python
from neurospatial.layout.factories import create_layout, list_available_layouts
from neurospatial.layout.engines.regular_grid import RegularGridLayout

# Alignment and transforms also available from neurospatial.ops:
# get_2d_rotation_matrix, map_probabilities, Affine2D, translate, scale_2d
```

---

## Utilities

```python
from neurospatial.animation.skeleton import Skeleton, SIMPLE_SKELETON
```
