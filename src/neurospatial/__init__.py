"""Spatial discretization and analysis for neuroscience.

**neurospatial** provides tools for discretizing continuous N-dimensional spatial
environments into bins/nodes with connectivity graphs. It enables spatial analysis
for neuroscience applications including place fields, position tracking, and
spatial navigation.

Core Classes
------------
Environment : Main spatial discretization class
    Discretizes continuous space into bins with connectivity graph.
    Factory methods: from_samples, from_polygon, from_graph, from_mask, from_image.
Region : Immutable region of interest (ROI)
    Point or polygon-based spatial region with metadata.
Regions : Container for multiple named regions
    Dict-like interface for managing collections of ROIs.
CompositeEnvironment : Multi-environment composition
    Merges multiple environments with automatic bridge inference.

Key Functions by Category
--------------------------

Spatial Queries and Mapping:
    map_points_to_bins : Batch point-to-bin mapping with KDTree caching
    distance_field : Multi-source geodesic distance computation
    pairwise_distances : Distances between node subsets
    neighbors_within : Find nodes within distance threshold

Trajectory Analysis:
    compute_place_field : Place field estimation from spike data
    spikes_to_field : Convert spike times to spatial firing rate
    Environment.occupancy : Compute spatial occupancy from trajectory
    Environment.bin_sequence : Extract bin sequence from positions

Neuroscience Metrics:
    detect_place_fields : Detect place fields from firing rate map
    skaggs_information : Spatial information content (bits/spike)
    sparsity : Measure of spatial selectivity
    selectivity : Place field selectivity metric
    border_score : Boundary cell border score
    grid_score : Grid cell grid score
    population_vector_correlation : Correlation between population vectors

Behavioral Segmentation:
    detect_laps : Detect laps on circular tracks
    segment_trials : Segment trajectory into behavioral trials
    detect_region_crossings : Detect region entry/exit events

Events and Peri-Event Analysis:
    peri_event_histogram : Peri-stimulus time histogram (PSTH)
    population_peri_event_histogram : Population PSTH across units
    align_spikes_to_events : Align spike times to event times
    time_to_nearest_event : Time to nearest event (GLM regressor)
    add_positions : Add spatial positions to events DataFrame

Field Operations:
    normalize_field : Normalize field to sum to 1
    combine_fields : Weighted combination of multiple fields
    clamp : Clamp field values to range
    Environment.smooth : Graph-based field smoothing
    Environment.interpolate : Interpolate field values

Transforms and Alignment:
    estimate_transform : Estimate affine transform from point pairs
    apply_transform_to_environment : Transform entire environment
    get_2d_rotation_matrix : Create 2D rotation matrix
    map_probabilities : Align probability distributions

Regions:
    Environment.region_membership : Compute region membership for bins
    regions_to_mask : Convert regions to boolean mask
    goal_reward_field : Gaussian reward field centered at goal
    region_reward_field : Reward field for region

Kernels and Convolution:
    compute_diffusion_kernels : Graph-based diffusion kernels
    apply_kernel : Apply kernel to field
    convolve : Graph-based convolution
    neighbor_reduce : Reduce over graph neighborhoods

Graph Operations:
    gradient : Spatial gradient on graph
    divergence : Spatial divergence on graph
    Environment.path_between : Find shortest path between bins
    Environment.neighbors : Get neighboring bins

I/O and Serialization:
    to_file : Save environment to .json + .npz files
    from_file : Load environment from files
    to_dict : Serialize to dictionary
    from_dict : Deserialize from dictionary

Validation and Utilities:
    validate_environment : Validate environment structure
    list_available_layouts : List all available layout types
    get_layout_parameters : Get parameters for layout type

Import Patterns
---------------
Import core classes and functions::

    from neurospatial import Environment, Region, Regions, CompositeEnvironment
    from neurospatial import (
        map_points_to_bins,
        distance_field,
        compute_place_field,
        to_file,
        from_file,
        validate_environment,
    )

Common Usage
------------
Create environment from position data::

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0, units='cm')
    >>> env.n_bins
    400

Map trajectory to bins::

    >>> times = np.linspace(0, 10, 100)
    >>> trajectory = np.random.uniform(0, 100, (100, 2))
    >>> bin_sequence = env.bin_sequence(trajectory)
    >>> occupancy = env.occupancy(times, trajectory)

Compute place field from spikes::

    >>> from neurospatial import compute_place_field
    >>> spike_times = np.array([1.2, 2.5, 3.7, 5.1])
    >>> firing_rate = compute_place_field(
    ...     env, spike_times, times, trajectory,
    ...     method='diffusion_kde', bandwidth=5.0
    ... )

Add and query regions::

    >>> env.regions.add('goal', point=[50, 50])
    >>> env.regions.add('start', point=[10, 10])
    >>> membership = env.region_membership(env.bin_centers)

Save and load::

    >>> from neurospatial import to_file, from_file
    >>> to_file(env, 'my_environment')
    >>> loaded = from_file('my_environment')

See Also
--------
Environment : Core environment class with detailed documentation
Region : Region of interest documentation
compute_place_field : Place field computation methods
distance_field : Graph-based distance computation

Notes
-----
This package uses graph-based representations to handle arbitrary spatial
topologies including regular grids, hexagonal tessellations, 1D tracks,
polygon-bounded regions, and custom connectivity patterns.

For detailed documentation, see https://neurospatial.readthedocs.io

Examples
--------
Create 2D environment and compute shortest path::

    >>> env = Environment.from_samples(
    ...     positions, bin_size=5.0, units='cm',
    ...     connect_diagonal_neighbors=True
    ... )
    >>> path = env.path_between(start_bin=0, goal_bin=100)
    >>> distance = env.distance_between(0, 100)

Create 3D environment::

    >>> positions_3d = np.random.uniform(0, 100, (1000, 3))
    >>> env_3d = Environment.from_samples(
    ...     positions_3d, bin_size=5.0, units='cm'
    ... )
    >>> env_3d.n_dims
    3

Create environment from polygon::

    >>> from shapely.geometry import box
    >>> polygon = box(0, 0, 100, 100)
    >>> env = Environment.from_polygon(
    ...     polygon, bin_size=5.0, units='cm'
    ... )
"""

import logging

from neurospatial.alignment import (
    get_2d_rotation_matrix,
    map_probabilities,
)
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    SpikeOverlay,
    VideoOverlay,
)
from neurospatial.annotation import (
    AnnotationResult,
    annotate_video,
    regions_from_cvat,
    regions_from_labelme,
)

# Basis functions for spatial regression (GLMs)
from neurospatial.basis import (
    chebyshev_filter_basis,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    plot_basis_functions,
    select_basis_centers,
    spatial_basis,
)
from neurospatial.behavioral import (
    compute_trajectory_curvature,
    cost_to_goal,
    distance_to_region,
    goal_pair_direction_labels,
    graph_turn_sequence,
    heading_direction_labels,
    path_progress,
    time_to_goal,
    trials_to_region_arrays,
)
from neurospatial.composite import CompositeEnvironment
from neurospatial.decoding import (
    DecodingResult,
    decode_position,
    decoding_error,
    median_decoding_error,
)
from neurospatial.differential import divergence, gradient
from neurospatial.distance import distance_field, neighbors_within, pairwise_distances
from neurospatial.environment import Environment, EnvironmentNotFittedError

# Events and peri-event analysis
from neurospatial.events import (
    PeriEventResult,
    PopulationPeriEventResult,
    add_positions,
    align_events,
    align_spikes_to_events,
    event_count_in_window,
    event_indicator,
    events_to_intervals,
    filter_by_intervals,
    intervals_to_events,
    peri_event_histogram,
    plot_peri_event_histogram,
    population_peri_event_histogram,
    time_to_nearest_event,
    validate_events_dataframe,
    validate_spatial_columns,
)
from neurospatial.field_ops import (
    clamp,
    combine_fields,
    normalize_field,
)
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.kernels import apply_kernel, compute_diffusion_kernels
from neurospatial.layout.factories import (
    LayoutType,
    get_layout_parameters,
    list_available_layouts,
)
from neurospatial.layout.validation import validate_environment

# Neuroscience metrics and behavioral analysis
from neurospatial.metrics import (
    SpatialViewMetrics,
    border_score,
    detect_place_fields,
    grid_score,
    is_spatial_view_cell,
    population_vector_correlation,
    selectivity,
    skaggs_information,
    sparsity,
    spatial_view_cell_metrics,
)

# Object-vector field analysis
from neurospatial.object_vector_field import (
    ObjectVectorFieldResult,
    compute_object_vector_field,
)
from neurospatial.primitives import convolve, neighbor_reduce

# Reference frame transformations (egocentric/allocentric)
from neurospatial.reference_frames import (
    EgocentricFrame,
    allocentric_to_egocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    egocentric_to_allocentric,
    heading_from_body_orientation,
    heading_from_velocity,
)
from neurospatial.regions import Region, Regions
from neurospatial.reward import goal_reward_field, region_reward_field
from neurospatial.segmentation import (
    detect_goal_directed_runs,
    detect_laps,
    detect_region_crossings,
    detect_runs_between_regions,
    segment_by_velocity,
    segment_trials,
)

# Spatial view cell simulation model
from neurospatial.simulation import SpatialViewCellModel
from neurospatial.spatial import (
    TieBreakStrategy,
    map_points_to_bins,
    regions_to_mask,
    resample_field,
)

# Spatial view field analysis
from neurospatial.spatial_view_field import (
    SpatialViewFieldResult,
    compute_spatial_view_field,
)
from neurospatial.spike_field import (
    DirectionalPlaceFields,
    compute_directional_place_fields,
    compute_place_field,
    spikes_to_field,
)
from neurospatial.transforms import (
    apply_transform_to_environment,
    estimate_transform,
)

# Visibility and gaze analysis
from neurospatial.visibility import (
    FieldOfView,
    ViewshedResult,
    compute_view_field,
    compute_viewed_location,
    compute_viewshed,
    compute_viewshed_trajectory,
    visibility_occupancy,
    visible_cues,
)
from neurospatial.visualization.scale_bar import ScaleBarConfig

# Add NullHandler to prevent "No handler found" warnings if user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Core classes
    "CompositeEnvironment",
    "Environment",
    "EnvironmentNotFittedError",
    "Region",
    "Regions",
    # Decoding (Bayesian population analysis)
    "DecodingResult",
    "decode_position",
    "decoding_error",
    "median_decoding_error",
    # Animation overlays
    "BodypartOverlay",
    "EventOverlay",
    "HeadDirectionOverlay",
    "PositionOverlay",
    "SpikeOverlay",
    "VideoOverlay",
    # Visualization config
    "ScaleBarConfig",
    # Annotation tools
    "AnnotationResult",
    "annotate_video",
    "regions_from_cvat",
    "regions_from_labelme",
    # Enums and types
    "LayoutType",
    "TieBreakStrategy",
    # I/O functions
    "from_dict",
    "from_file",
    "to_dict",
    "to_file",
    # Neuroscience metrics
    "SpatialViewMetrics",
    "border_score",
    "detect_place_fields",
    "grid_score",
    "is_spatial_view_cell",
    "population_vector_correlation",
    "selectivity",
    "skaggs_information",
    "sparsity",
    "spatial_view_cell_metrics",
    # Basis functions for GLMs
    "chebyshev_filter_basis",
    "geodesic_rbf_basis",
    "heat_kernel_wavelet_basis",
    "plot_basis_functions",
    "select_basis_centers",
    "spatial_basis",
    # Behavioral segmentation
    "detect_goal_directed_runs",
    "detect_laps",
    "detect_region_crossings",
    "detect_runs_between_regions",
    "segment_by_velocity",
    "segment_trials",
    # Behavioral analysis
    "compute_trajectory_curvature",
    "cost_to_goal",
    "distance_to_region",
    "goal_pair_direction_labels",
    "graph_turn_sequence",
    "heading_direction_labels",
    "path_progress",
    "time_to_goal",
    "trials_to_region_arrays",
    # Events and peri-event analysis
    "PeriEventResult",
    "PopulationPeriEventResult",
    "add_positions",
    "align_events",
    "align_spikes_to_events",
    "event_count_in_window",
    "event_indicator",
    "events_to_intervals",
    "filter_by_intervals",
    "intervals_to_events",
    "peri_event_histogram",
    "plot_peri_event_histogram",
    "population_peri_event_histogram",
    "time_to_nearest_event",
    "validate_events_dataframe",
    "validate_spatial_columns",
    # Spatial operations and queries
    "apply_kernel",
    "apply_transform_to_environment",
    "clamp",
    "combine_fields",
    "compute_diffusion_kernels",
    "compute_directional_place_fields",
    "compute_place_field",
    "convolve",
    "DirectionalPlaceFields",
    "distance_field",
    "divergence",
    "estimate_transform",
    "get_2d_rotation_matrix",
    "get_layout_parameters",
    "goal_reward_field",
    "gradient",
    "list_available_layouts",
    "map_points_to_bins",
    "map_probabilities",
    "neighbor_reduce",
    "neighbors_within",
    "normalize_field",
    "pairwise_distances",
    "region_reward_field",
    "regions_to_mask",
    "resample_field",
    "spikes_to_field",
    "validate_environment",
    # Reference frame transformations (egocentric/allocentric)
    "EgocentricFrame",
    "allocentric_to_egocentric",
    "compute_egocentric_bearing",
    "compute_egocentric_distance",
    "egocentric_to_allocentric",
    "heading_from_body_orientation",
    "heading_from_velocity",
    # Object-vector field analysis
    "ObjectVectorFieldResult",
    "compute_object_vector_field",
    # Spatial view cell simulation
    "SpatialViewCellModel",
    # Spatial view field analysis
    "SpatialViewFieldResult",
    "compute_spatial_view_field",
    # Visibility and gaze analysis
    "FieldOfView",
    "ViewshedResult",
    "compute_view_field",
    "compute_viewed_location",
    "compute_viewshed",
    "compute_viewshed_trajectory",
    "visibility_occupancy",
    "visible_cues",
]
