# API Reference

Import patterns organized by feature area.

**Updated** - Domain-centric package reorganization.

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

    # Distance
    distance_field,                 # Multi-source geodesic distances
    pairwise_distances,             # Distances between node subsets
    euclidean_distance_matrix,      # Euclidean distance matrix
    geodesic_distance_matrix,       # Geodesic distance matrix
    geodesic_distance_between_points,  # Point-to-point geodesic
    neighbors_within,               # Neighbors within range

    # Transforms
    estimate_transform,             # Estimate transform from point pairs
    apply_transform_to_environment, # Transform entire environment
    Affine2D,                       # 2D affine transform
    AffineND,                       # ND affine transform
    translate,                      # Translation transform
    scale_2d,                       # 2D scale transform

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

### Place Cells

```python
from neurospatial.encoding.place import (
    # Field computation
    compute_place_field,                    # Place field estimation
    compute_directional_place_fields,       # Directional tuning
    DirectionalPlaceFields,                 # Container for directional fields

    # Place field metrics
    detect_place_fields,                    # Detect fields from rate map
    skaggs_information,                     # Spatial info (bits/spike)
    sparsity,                               # Spatial sparsity
    selectivity,                            # Place field selectivity
    rate_map_centroid,                      # Field centroid
    field_size,                             # Field size
    field_stability,                        # Temporal stability
    field_shape_metrics,                    # Shape metrics
    field_shift_distance,                   # Shift between sessions
    in_out_field_ratio,                     # In/out firing ratio
    information_per_second,                 # Bits/second
    mutual_information,                     # MI between fields
    rate_map_coherence,                     # Spatial coherence
    spatial_coverage_single_cell,           # Coverage fraction
    compute_field_emd,                      # Earth mover distance
)
```

### Grid Cells

```python
from neurospatial.encoding.grid import (
    grid_score,                             # Grid cell grid score
    spatial_autocorrelation,                # Autocorrelation of field
    grid_orientation,                       # Grid orientation
    grid_scale,                             # Grid spacing
    grid_properties,                        # Comprehensive properties
    periodicity_score,                      # Periodicity measure
    GridProperties,                         # Properties dataclass
)
```

### Head Direction Cells

```python
from neurospatial.encoding.head_direction import (
    head_direction_tuning_curve,            # HD tuning curve
    head_direction_metrics,                 # Comprehensive HD metrics
    is_head_direction_cell,                 # HD cell classification
    plot_head_direction_tuning,             # Polar plot
    HeadDirectionMetrics,                   # Metrics dataclass

    # Re-exported from stats.circular
    rayleigh_test,
    mean_resultant_length,
    circular_mean,
)
```

### Border/Boundary Cells

```python
from neurospatial.encoding.border import (
    border_score,                           # Border score
    compute_region_coverage,                # Region coverage stats
)
```

### Object-Vector Cells

```python
from neurospatial.encoding.object_vector import (
    # Field computation
    ObjectVectorFieldResult,                # Result container
    compute_object_vector_field,            # Compute egocentric polar field

    # Metrics and classification
    ObjectVectorMetrics,                    # Frozen dataclass with tuning metrics
    compute_object_vector_tuning,           # Compute tuning from spikes
    object_vector_score,                    # Distance × direction selectivity
    is_object_vector_cell,                  # Classifier (score + peak rate)
    plot_object_vector_tuning,              # Polar heatmap visualization
)
```

### Spatial View Cells

```python
from neurospatial.encoding.spatial_view import (
    # Field computation
    SpatialViewFieldResult,                 # Result container
    compute_spatial_view_field,             # Compute view field

    # Metrics and classification
    SpatialViewMetrics,                     # Frozen dataclass with metrics
    spatial_view_cell_metrics,              # Compute view vs place metrics
    is_spatial_view_cell,                   # Quick classifier

    # Re-exported from ops.visibility
    compute_viewed_location,
    compute_viewshed,
    visibility_occupancy,
    FieldOfView,
)
```

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

    # Shuffle testing (re-exported from stats)
    shuffle_time_bins,
    shuffle_cell_identity,
    compute_shuffle_pvalue,
    ShuffleTestResult,
    generate_poisson_surrogates,
)
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

# Create polar grid in egocentric space (for object-vector cells)
ego_env = Environment.from_polar_egocentric(
    distance_range=(0, 50),
    angle_range=(-np.pi, np.pi),
    distance_bin_size=5.0,
    angle_bin_size=np.pi / 8,
    circular_angle=True,
)
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

## Simulation

```python
from neurospatial.simulation import (
    # Cell models
    ObjectVectorCellModel,         # OVC model with Gaussian/von Mises tuning
    SpatialViewCellModel,          # Gaze-based firing model
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
    path_efficiency,                # Ratio: shortest / traveled
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

from neurospatial.ops.transforms import (
    VideoCalibration,                       # Video calibration transform
    calibrate_from_landmarks,               # Calibrate from point pairs
)
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

from neurospatial.ops.alignment import get_2d_rotation_matrix, map_probabilities
from neurospatial.ops.transforms import Affine2D, translate, scale_2d
```

---

## Utilities

```python
from neurospatial.animation.skeleton import Skeleton, SIMPLE_SKELETON
```
