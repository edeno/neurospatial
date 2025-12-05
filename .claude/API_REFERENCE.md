# API Reference

Import patterns organized by feature area.

---

## Core Classes

```python
from neurospatial import Environment
from neurospatial.regions import Region, Regions
```

---

## Spatial Analysis (v0.1.0+)

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

## Neural Analysis (v0.2.0+)

```python
from neurospatial import (
    compute_place_field,                    # Place field estimation
    compute_directional_place_fields,       # Directional tuning (v0.10.0+)
    goal_pair_direction_labels,             # Trial-based direction labels
    heading_direction_labels,               # Heading-based direction labels
)
```

---

## Bayesian Decoding (v0.12.0+)

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

## Basis Functions for GLMs (v0.15.0+)

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

## Behavioral Analysis (v0.7.0+)

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

## Events and Peri-Event Analysis (v0.16.0+)

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

## Animation & Visualization (v0.3.0+)

```python
from neurospatial.animation import subsample_frames

from neurospatial import (
    PositionOverlay,        # Trajectory with trail
    BodypartOverlay,        # Pose tracking with skeleton
    HeadDirectionOverlay,   # Orientation arrows
    EventOverlay,           # Spikes, licks, rewards (v0.13.0+)
    SpikeOverlay,           # Alias for EventOverlay
    TimeSeriesOverlay,      # Continuous variables (v0.14.0+)
    ScaleBarConfig,         # Scale bar configuration (v0.11.0+)
)

from neurospatial.animation import (
    VideoOverlay,                           # Video behind/above fields (v0.5.0+)
    calibrate_video,                        # Video calibration helper
    estimate_colormap_range_from_subset,    # Large session helper
    large_session_napari_config,            # Napari config for large data
)

from neurospatial.transforms import VideoCalibration, calibrate_from_landmarks
```

---

## Video Annotation (v0.6.0+)

```python
from neurospatial import (
    annotate_video,          # Interactive annotation
    regions_from_labelme,    # Import from LabelMe
    regions_from_cvat,       # Import from CVAT
)

from neurospatial.annotation import (
    annotate_track_graph,    # 1D track annotation (v0.9.0+)
    TrackGraphResult,        # Track graph annotation result
    BoundaryConfig,          # Boundary inference config
    boundary_from_positions, # Auto-infer boundary from positions
)
```

---

## NWB Integration (v0.7.0+)

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

## Serialization (v0.1.0+)

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
