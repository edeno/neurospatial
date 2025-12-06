"""Spatial discretization and analysis for neuroscience.

**neurospatial** provides tools for discretizing continuous N-dimensional spatial
environments into bins/nodes with connectivity graphs. It enables spatial analysis
for neuroscience applications including place fields, position tracking, and
spatial navigation.

Core Classes (Top-Level Exports)
--------------------------------
Environment : Main spatial discretization class
    Discretizes continuous space into bins with connectivity graph.
    Factory methods: from_samples, from_polygon, from_graph, from_mask, from_image.
EnvironmentNotFittedError : Exception for unfitted environments
    Raised when methods requiring fitted state are called on unfitted environment.
Region : Immutable region of interest (ROI)
    Point or polygon-based spatial region with metadata.
Regions : Container for multiple named regions
    Dict-like interface for managing collections of ROIs.
CompositeEnvironment : Multi-environment composition
    Merges multiple environments with automatic bridge inference.

Submodule Organization
----------------------
All other functionality is accessed via explicit submodule imports. This design
follows Raymond Hettinger's "sparse top-level" principle for better autocomplete
and clearer import patterns.

encoding : Neural encoding analysis
    Place cells, grid cells, head direction cells, border cells, object-vector
    cells, spatial view cells, phase precession, and population metrics.

    >>> from neurospatial.encoding import place, grid, head_direction
    >>> from neurospatial.encoding.place import compute_place_field, detect_place_fields

decoding : Neural decoding
    Bayesian position decoding, trajectory detection, cell assemblies.

    >>> from neurospatial.decoding import decode_position, DecodingResult

behavior : Behavioral analysis
    Trajectory metrics, segmentation (laps, trials), navigation metrics,
    decision analysis (VTE), reward fields.

    >>> from neurospatial.behavior import segmentation, navigation, trajectory
    >>> from neurospatial.behavior.segmentation import detect_laps, segment_trials

events : Peri-event analysis
    PSTH, event alignment, GLM regressors.

    >>> from neurospatial.events import peri_event_histogram, align_spikes_to_events

ops : Low-level operations
    Binning, distance, smoothing, graph operations, calculus (gradient/divergence),
    transforms, alignment, egocentric reference frames, visibility, basis functions.

    >>> from neurospatial.ops import (
    ...     distance_field,
    ...     normalize_field,
    ...     heading_from_velocity,
    ... )

stats : Statistical methods
    Circular statistics, shuffle controls, surrogate generation.

    >>> from neurospatial.stats import rayleigh_test, shuffle_time_bins

io : File I/O and NWB integration
    Save/load environments, NWB file integration.

    >>> from neurospatial.io import to_file, from_file

animation : Visualization
    Napari viewer, video export, overlays (position, spike, event, HD, bodypart).

    >>> from neurospatial.animation import PositionOverlay, SpikeOverlay

simulation : Neural and trajectory simulation
    Cell models (place, grid, HD, border, OVC, SVC), trajectory generation.

    >>> from neurospatial.simulation import PlaceCellModel, simulate_session

layout : Layout engines
    Regular grid, hexagonal, graph-based, masked, polygon, triangular mesh.

    >>> from neurospatial.layout import list_available_layouts, get_layout_parameters

annotation : Video annotation tools
    Environment annotation from video, CVAT/LabelMe format support.

    >>> from neurospatial.annotation import annotate_video

Import Patterns
---------------
Core classes only at top level::

    from neurospatial import Environment, Region, Regions, CompositeEnvironment

Explicit submodule imports for all else (recommended)::

    # Neural encoding
    from neurospatial.encoding.place import compute_place_field, skaggs_information
    from neurospatial.encoding.grid import grid_score

    # Neural decoding
    from neurospatial.decoding import decode_position

    # Behavioral analysis
    from neurospatial.behavior.segmentation import detect_laps
    from neurospatial.behavior.navigation import path_efficiency

    # Events
    from neurospatial.events import peri_event_histogram

    # Operations
    from neurospatial.ops.distance import distance_field
    from neurospatial.ops.egocentric import heading_from_velocity

    # I/O
    from neurospatial.io import to_file, from_file

    # Animation
    from neurospatial.animation import PositionOverlay

Common Usage
------------
Create environment from position data::

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> positions = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> env.units = 'cm'
    >>> env.n_bins > 0  # Number of bins depends on data coverage
    True

Map trajectory to bins::

    >>> times = np.linspace(0, 10, 100)
    >>> trajectory = np.random.uniform(0, 100, (100, 2))
    >>> bin_sequence = env.bin_sequence(trajectory)
    >>> occupancy = env.occupancy(times, trajectory)

Compute place field from spikes::

    >>> from neurospatial.encoding.place import compute_place_field
    >>> spike_times = np.array([1.2, 2.5, 3.7, 5.1])
    >>> firing_rate = compute_place_field(
    ...     env, spike_times, times, trajectory,
    ...     smoothing_method='diffusion_kde', bandwidth=5.0
    ... )

Add and query regions::

    >>> env.regions.add('goal', point=[50, 50])
    >>> env.regions.add('start', point=[10, 10])
    >>> membership = env.region_membership(env.bin_centers)

Save and load::

    >>> from neurospatial.io import to_file, from_file
    >>> to_file(env, 'my_environment')
    >>> loaded = from_file('my_environment')

See Also
--------
Environment : Core environment class with detailed documentation
Region : Region of interest documentation
neurospatial.encoding : Neural encoding analysis
neurospatial.decoding : Neural decoding
neurospatial.behavior : Behavioral analysis

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

from neurospatial.composite import CompositeEnvironment
from neurospatial.environment import Environment, EnvironmentNotFittedError
from neurospatial.regions import Region, Regions

# Add NullHandler to prevent "No handler found" warnings if user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CompositeEnvironment",
    "Environment",
    "EnvironmentNotFittedError",
    "Region",
    "Regions",
]
