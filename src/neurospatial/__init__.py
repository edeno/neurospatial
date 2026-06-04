"""Spatial discretization and analysis for neuroscience.

**neurospatial** provides tools for discretizing continuous N-dimensional spatial
environments into bins/nodes with connectivity graphs. It enables spatial analysis
for neuroscience applications including place fields, position tracking, and
spatial navigation.

Core Classes (Top-Level Exports)
--------------------------------
Environment : Main spatial discretization class
    Discretizes continuous space into bins with connectivity graph.
    Factory methods: from_samples, from_polygon, from_graph, from_grid_mask, from_pixel_mask.
EnvironmentNotFittedError : Exception for unfitted environments
    Raised when methods requiring fitted state are called on unfitted environment.
Region : Immutable region of interest (ROI)
    Point or polygon-based spatial region with metadata.
Regions : Container for multiple named regions
    Dict-like interface for managing collections of ROIs.
CompositeEnvironment : Multi-environment composition
    Merges multiple environments with automatic bridge inference.
bin_spikes_in_time : Spike-time -> count-matrix binner
    Bins per-neuron spike-time arrays onto a regular time grid, producing the
    count matrix ``decode_position`` (or the assembly functions) consume.
    Re-exported from :mod:`neurospatial.decoding` as a primary entry point.

Submodule Organization
----------------------
All other functionality is accessed via explicit submodule imports. This design
follows Raymond Hettinger's "sparse top-level" principle for better autocomplete
and clearer import patterns.

encoding : Neural encoding analysis
    Place cells, grid cells, head direction cells, border cells, object-vector
    cells, spatial view cells, phase precession, and population metrics.

    >>> from neurospatial.encoding import compute_spatial_rate, detect_place_fields
    >>> from neurospatial.encoding import compute_directional_rate, compute_view_rate

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
    from neurospatial.encoding import compute_spatial_rate, spatial_information
    from neurospatial.encoding.grid import grid_score

    # Neural decoding
    from neurospatial.decoding import decode_position

    # Behavioral analysis
    from neurospatial.behavior.segmentation import detect_laps
    from neurospatial.behavior.navigation import compute_path_efficiency

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
    >>> assert env.n_bins > 0  # Number of bins depends on data coverage

Map trajectory to bins::

    >>> times = np.linspace(0, 10, 100)  # doctest: +SKIP
    >>> trajectory = np.random.uniform(0, 100, (100, 2))  # doctest: +SKIP
    >>> # bin_sequence and occupancy take (times, positions). Reversing the
    >>> # arguments raises ValueError because the first argument must be a
    >>> # 1-D `times` array, not the 2-D positions array.
    >>> bin_sequence = env.bin_sequence(times, trajectory)  # doctest: +SKIP
    >>> occupancy = env.occupancy(times, trajectory)  # doctest: +SKIP

Compute a spatial firing-rate map from spikes::

    >>> from neurospatial.encoding import compute_spatial_rate  # doctest: +SKIP
    >>> spike_times = np.array([1.2, 2.5, 3.7, 5.1])  # doctest: +SKIP
    >>> result = compute_spatial_rate(  # doctest: +SKIP
    ...     env, spike_times, times, trajectory,
    ...     smoothing_method='diffusion_kde', bandwidth=5.0
    ... )
    >>> firing_rate = result.firing_rate  # doctest: +SKIP

Add and query regions::

    >>> env.regions.add('goal', point=[50, 50])  # doctest: +SKIP
    >>> env.regions.add('start', point=[10, 10])  # doctest: +SKIP
    >>> membership = env.region_membership(env.bin_centers)  # doctest: +SKIP

Save and load::

    >>> from neurospatial.io import to_file, from_file  # doctest: +SKIP
    >>> to_file(env, 'my_environment')  # doctest: +SKIP
    >>> loaded = from_file('my_environment')  # doctest: +SKIP

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

    >>> env = Environment.from_samples(  # doctest: +SKIP
    ...     positions, bin_size=5.0,
    ...     connect_diagonal_neighbors=True,
    ... )
    >>> env.units = 'cm'  # doctest: +SKIP
    >>> path = env.path_between(0, 100)  # doctest: +SKIP
    >>> # distance_between takes coordinates; for graph distance between bin
    >>> # indices use distance_to([target_bin]) and index by the source bin.
    >>> distance = float(env.distance_to([100])[0])  # doctest: +SKIP

Create 3D environment::

    >>> positions_3d = np.random.uniform(0, 100, (1000, 3))  # doctest: +SKIP
    >>> env_3d = Environment.from_samples(  # doctest: +SKIP
    ...     positions_3d, bin_size=5.0,
    ... )
    >>> env_3d.units = 'cm'  # doctest: +SKIP
    >>> env_3d.n_dims  # doctest: +SKIP
    3

Create environment from polygon::

    >>> from shapely.geometry import box  # doctest: +SKIP
    >>> polygon = box(0, 0, 100, 100)  # doctest: +SKIP
    >>> env = Environment.from_polygon(polygon, bin_size=5.0)  # doctest: +SKIP
    >>> env.units = 'cm'  # doctest: +SKIP
"""

import logging
from importlib import import_module
from typing import Any

from neurospatial._exceptions import (
    BinIndexOutOfRangeError,
    EnvironmentNotFittedError,
    GraphValidationError,
    IncompatibleEnvironmentError,
    LayoutNotBuiltError,
    RegionNotFoundError,
)
from neurospatial.composite import CompositeEnvironment
from neurospatial.decoding import bin_spikes_in_time, decode_session
from neurospatial.environment import Environment
from neurospatial.regions import Region, Regions

# Add NullHandler to prevent "No handler found" warnings if user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Submodules exposed via lazy (PEP 562) attribute access. Accessing, e.g.,
# ``neurospatial.encoding`` imports ``neurospatial.encoding`` on first use and
# caches it in the package globals; the submodule itself is *not* imported when
# the package loads. This keeps the top-level namespace cheap to import while
# letting autocomplete and ``dir(neurospatial)`` reveal every domain.
_LAZY_SUBMODULES: tuple[str, ...] = (
    "encoding",
    "decoding",
    "behavior",
    "events",
    "ops",
    "layout",
    "regions",
    "stats",
    "simulation",
    "annotation",
    "animation",
    "io",
)


def __getattr__(name: str) -> Any:
    """Lazily import a known submodule on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = import_module(f"neurospatial.{name}")
        # Cache in package globals so subsequent access skips this hook.
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List eager exports plus lazily importable submodules for autocomplete."""
    return sorted(set(__all__) | set(_LAZY_SUBMODULES))


__all__ = [
    "BinIndexOutOfRangeError",
    "CompositeEnvironment",
    "Environment",
    "EnvironmentNotFittedError",
    "GraphValidationError",
    "IncompatibleEnvironmentError",
    "LayoutNotBuiltError",
    "Region",
    "RegionNotFoundError",
    "Regions",
    "animation",
    "annotation",
    "behavior",
    "bin_spikes_in_time",
    "decode_session",
    "decoding",
    "encoding",
    "events",
    "io",
    "layout",
    "ops",
    "regions",
    "simulation",
    "stats",
]
