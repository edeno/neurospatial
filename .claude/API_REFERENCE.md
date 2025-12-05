# API Reference

Import patterns organized by feature area.

---

## Core Classes

```python
from neurospatial import Environment
from neurospatial.regions import Region, Regions
```

---

## Spatial Analysis

```python
from neurospatial import (
    validate_environment,           # Validate environment structure
    map_points_to_bins,             # Batch point-to-bin mapping with KDTree
    estimate_transform,             # Estimate transform from point pairs
    apply_transform_to_environment, # Transform entire environment
    distance_field,                 # Multi-source geodesic distances
    pairwise_distances,             # Distances between node subsets
)
```

---

## Neural Analysis

```python
from neurospatial import (
    compute_place_field,                    # Place field estimation
    compute_directional_place_fields,       # Directional tuning
    goal_pair_direction_labels,             # Trial-based direction labels
    heading_direction_labels,               # Heading-based direction labels
)
```

---

## Bayesian Decoding

```python
from neurospatial import (
    DecodingResult,          # Result container class
    decode_position,         # Main entry point
    decoding_error,          # Per-time-bin position error
    median_decoding_error,   # Summary statistic
)

# Full decoding API
from neurospatial.decoding import (
    # Likelihood
    log_poisson_likelihood,
    poisson_likelihood,

    # Posterior
    normalize_to_posterior,

    # Point estimates
    map_estimate,
    map_position,
    mean_position,
    entropy,
    credible_region,

    # Trajectory analysis
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    detect_trajectory_radon,  # Requires scikit-image
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,

    # Quality metrics
    confusion_matrix,
    decoding_correlation,

    # Shuffle testing
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    generate_poisson_surrogates,
    generate_inhomogeneous_poisson_surrogates,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    ShuffleTestResult,
)
```

---

## Basis Functions for GLMs

```python
from neurospatial import (
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

## Circular Basis Functions for GLMs

```python
from neurospatial.metrics import (
    # Design matrix creation
    circular_basis,             # sin/cos basis for circular predictors
    CircularBasisResult,        # Result container with design_matrix property

    # Interpretation helpers
    circular_basis_metrics,     # Extract amplitude, phase, p-value from GLM
    is_modulated,               # Quick significance check (True/False)

    # Visualization
    plot_circular_basis_tuning, # Polar or linear tuning curve plots
)
```

**Related circular metrics:**

```python
from neurospatial.metrics import (
    # Head direction analysis
    head_direction_tuning_curve,
    head_direction_metrics,
    is_head_direction_cell,
    plot_head_direction_tuning,
    HeadDirectionMetrics,

    # Phase precession
    phase_precession,
    has_phase_precession,
    plot_phase_precession,
    PhasePrecessionResult,

    # Circular statistics
    rayleigh_test,
    circular_linear_correlation,
    circular_circular_correlation,
    phase_position_correlation,
)
```

---

## Egocentric Reference Frames

```python
from neurospatial import (
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

# Full reference frames API
from neurospatial.reference_frames import (
    EgocentricFrame,
    allocentric_to_egocentric,
    egocentric_to_allocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    heading_from_velocity,
    heading_from_body_orientation,
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

## Object-Vector Cells

```python
from neurospatial import (
    # Field computation
    ObjectVectorFieldResult,       # Result container
    compute_object_vector_field,   # Compute egocentric polar field
)

# Metrics and classification
from neurospatial.metrics import (
    ObjectVectorMetrics,           # Frozen dataclass with tuning metrics
    compute_object_vector_tuning,  # Compute tuning from spikes
    object_vector_score,           # Distance × direction selectivity
    is_object_vector_cell,         # Classifier (score + peak rate)
    plot_object_vector_tuning,     # Polar heatmap visualization
)

# Simulation
from neurospatial.simulation import (
    ObjectVectorCellModel,         # OVC model with Gaussian/von Mises tuning
)

# Animation overlay
from neurospatial.animation import (
    ObjectVectorOverlay,           # Vectors from animal to objects
    ObjectVectorData,              # Internal data container
)
```

---

## Spatial View Cells

```python
from neurospatial import (
    # Field computation
    SpatialViewFieldResult,        # Result container
    compute_spatial_view_field,    # Compute view field (by viewed location)

    # Metrics and classification
    SpatialViewMetrics,            # Frozen dataclass with metrics
    spatial_view_cell_metrics,     # Compute view vs place metrics
    is_spatial_view_cell,          # Quick classifier

    # Simulation
    SpatialViewCellModel,          # Gaze-based firing model
)
```

**Visibility and gaze analysis:**

```python
from neurospatial import (
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

**Full visibility module API:**

```python
from neurospatial.visibility import (
    FieldOfView,
    ViewshedResult,
    compute_viewed_location,
    compute_viewshed,
    compute_view_field,
    compute_viewshed_trajectory,
    visibility_occupancy,
    visible_cues,
)
```

---

## Behavioral Analysis

```python
from neurospatial.segmentation import segment_trials

from neurospatial import (
    path_progress,                  # Normalized progress (0→1) along path
    distance_to_region,             # Distance to goal over time
    cost_to_goal,                   # RL cost with terrain/avoidance
    time_to_goal,                   # Time until goal arrival
    compute_trajectory_curvature,   # Continuous curvature
    graph_turn_sequence,            # Discrete turn labels
    trials_to_region_arrays,        # Helper for trial arrays
)
```

---

## Behavioral Trajectory Metrics

**Path efficiency:**

```python
from neurospatial.metrics import (
    # Result dataclasses
    PathEfficiencyResult,       # Path efficiency metrics
    SubgoalEfficiencyResult,    # Multi-waypoint efficiency

    # Core functions
    traveled_path_length,       # Total distance traveled
    shortest_path_length,       # Geodesic/Euclidean distance
    path_efficiency,            # Ratio: shortest / traveled
    time_efficiency,            # Ratio: T_optimal / T_actual
    angular_efficiency,         # 1 - mean(|delta_theta|) / pi
    subgoal_efficiency,         # Per-segment efficiency

    # Combined analysis
    compute_path_efficiency,    # All metrics in one call
)
```

**Goal-directed metrics:**

```python
from neurospatial.metrics import (
    # Result dataclass
    GoalDirectedMetrics,        # Goal-directed navigation metrics

    # Core functions
    goal_vector,                # Vector from position to goal
    goal_direction,             # Angle to goal
    instantaneous_goal_alignment,  # Cos(velocity, goal direction)
    goal_bias,                  # Mean alignment over trajectory
    approach_rate,              # d/dt of distance to goal

    # Combined analysis
    compute_goal_directed_metrics,  # All metrics in one call
)
```

**Decision point analysis:**

```python
from neurospatial.metrics import (
    # Result dataclasses
    PreDecisionMetrics,         # Kinematics before decision region
    DecisionBoundaryMetrics,    # Boundary crossing metrics
    DecisionAnalysisResult,     # Complete trial analysis

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

    # Combined analysis
    compute_decision_analysis,      # Full decision analysis
)
```

**VTE (Vicarious Trial and Error) metrics:**

```python
from neurospatial.metrics import (
    # Result dataclasses
    VTETrialResult,             # Single trial VTE metrics
    VTESessionResult,           # Session-level VTE analysis

    # Core functions
    wrap_angle,                 # Wrap angle to (-pi, pi]
    head_sweep_magnitude,       # Sum of |delta_theta| (IdPhi)
    integrated_absolute_rotation,  # Alias for head_sweep_magnitude
    head_sweep_from_positions,  # IdPhi from trajectory

    # Z-scoring and classification
    normalize_vte_scores,       # Z-score across trials
    compute_vte_index,          # Combined VTE index
    classify_vte,               # VTE if index > threshold

    # Combined analysis
    compute_vte_trial,          # Single trial analysis
    compute_vte_session,        # Full session analysis
)
```

---

## Events and Peri-Event Analysis

```python
from neurospatial import (
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

**Full events module API:**

```python
from neurospatial.events import (
    # All of the above, plus:
    PeriEventResult,
    PopulationPeriEventResult,
)
```

---

## Animation & Visualization

```python
from neurospatial.animation import subsample_frames

from neurospatial import (
    PositionOverlay,        # Trajectory with trail
    BodypartOverlay,        # Pose tracking with skeleton
    HeadDirectionOverlay,   # Orientation arrows
    EventOverlay,           # Spikes, licks, rewards
    SpikeOverlay,           # Alias for EventOverlay
    TimeSeriesOverlay,      # Continuous variables
    ScaleBarConfig,         # Scale bar configuration
)

from neurospatial.animation import (
    VideoOverlay,                           # Video behind/above fields
    calibrate_video,                        # Video calibration helper
    estimate_colormap_range_from_subset,    # Large session helper
    large_session_napari_config,            # Napari config for large data
)

from neurospatial.transforms import VideoCalibration, calibrate_from_landmarks
```

---

## Video Annotation

```python
from neurospatial import (
    annotate_video,          # Interactive annotation
    regions_from_labelme,    # Import from LabelMe
    regions_from_cvat,       # Import from CVAT
)

from neurospatial.annotation import (
    annotate_track_graph,    # 1D track annotation
    TrackGraphResult,        # Track graph annotation result
    BoundaryConfig,          # Boundary inference config
    boundary_from_positions, # Auto-infer boundary from positions
)
```

---

## NWB Integration

**Requires:** `pip install neurospatial[nwb-full]`

```python
from neurospatial.nwb import (
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

## Serialization

```python
from neurospatial.io import to_file, from_file, to_dict, from_dict
```

---

## Layout Engines & Transforms

```python
from neurospatial.layout.factories import create_layout, list_available_layouts
from neurospatial.layout.engines.regular_grid import RegularGridLayout

from neurospatial.alignment import get_2d_rotation_matrix, map_probabilities
from neurospatial.transforms import Affine2D, translate, scale_2d
```

---

## Composite Environments

```python
from neurospatial import CompositeEnvironment
```

---

## Utilities

```python
from neurospatial.animation.skeleton import Skeleton, SIMPLE_SKELETON
```
