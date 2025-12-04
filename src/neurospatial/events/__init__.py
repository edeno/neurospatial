"""
Events module for neurospatial.

This module provides tools for event detection, temporal alignment, and
GLM regressor generation for neural data analysis.

Public API
----------
Result Dataclasses:
    PeriEventResult : Result from peri-event histogram analysis
    PopulationPeriEventResult : Result from population peri-event analysis

Validation Helpers:
    validate_events_dataframe : Validate events DataFrame structure
    validate_spatial_columns : Check for spatial columns (x, y)

Detection Functions:
    extract_region_crossing_events : Detect region entry/exit events
    extract_threshold_crossing_events : Detect signal threshold crossings
    extract_movement_onset_events : Detect movement onset from position

Spatial Utilities:
    add_positions : Add x, y columns to events by interpolation
    events_in_region : Filter events to those within a region
    spatial_event_rate : Compute spatial rate map of events

Interval Utilities:
    intervals_to_events : Convert intervals to point events
    events_to_intervals : Pair start/stop events into intervals
    filter_by_intervals : Filter events by time intervals

GLM Regressors:
    time_since_event : Time since most recent event
    time_to_event : Time until next event
    event_count_in_window : Count events in time window
    event_indicator : Binary indicator of event presence
    exponential_kernel : Convolve events with exponential kernel
    distance_to_event_at_time : Distance to last/next event location

Peri-Event Analysis:
    align_spikes_to_events : Get per-trial spike times
    peri_event_histogram : Compute PSTH
    population_peri_event_histogram : Compute population PSTH
    align_events : Align events to reference events

Visualization:
    plot_peri_event_histogram : Plot PSTH results

Examples
--------
Computing a peri-event time histogram (PSTH):

>>> from neurospatial.events import peri_event_histogram
>>> result = peri_event_histogram(  # doctest: +SKIP
...     spike_times,
...     reward_times,
...     window=(-1.0, 2.0),
...     bin_size=0.025,
... )
>>> print(f"Peak at {result.bin_centers[result.histogram.argmax()]:.2f}s")

Creating GLM regressors:

>>> from neurospatial.events import time_since_event, exponential_kernel
>>> time_since_reward = time_since_event(  # doctest: +SKIP
...     sample_times, reward_times, max_time=10.0
... )
>>> reward_signal = exponential_kernel(  # doctest: +SKIP
...     sample_times, reward_times, tau=2.0
... )
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Lazy imports - functions are imported when first accessed
# This keeps the module lightweight at import time

# Mapping of public API names to their module paths
# Format: "name": "module_path:attribute_name"
_LAZY_IMPORTS: dict[str, str] = {
    # Result dataclasses
    "PeriEventResult": "neurospatial.events._core:PeriEventResult",
    "PopulationPeriEventResult": "neurospatial.events._core:PopulationPeriEventResult",
    # Validation helpers
    "validate_events_dataframe": "neurospatial.events._core:validate_events_dataframe",
    "validate_spatial_columns": "neurospatial.events._core:validate_spatial_columns",
    # Visualization
    "plot_peri_event_histogram": "neurospatial.events._core:plot_peri_event_histogram",
    # Detection functions
    "extract_region_crossing_events": "neurospatial.events.detection:extract_region_crossing_events",
    "extract_threshold_crossing_events": "neurospatial.events.detection:extract_threshold_crossing_events",
    "extract_movement_onset_events": "neurospatial.events.detection:extract_movement_onset_events",
    "add_positions": "neurospatial.events.detection:add_positions",
    "events_in_region": "neurospatial.events.detection:events_in_region",
    "spatial_event_rate": "neurospatial.events.detection:spatial_event_rate",
    # Interval utilities
    "intervals_to_events": "neurospatial.events.intervals:intervals_to_events",
    "events_to_intervals": "neurospatial.events.intervals:events_to_intervals",
    "filter_by_intervals": "neurospatial.events.intervals:filter_by_intervals",
    # GLM regressors
    "time_since_event": "neurospatial.events.regressors:time_since_event",
    "time_to_event": "neurospatial.events.regressors:time_to_event",
    "event_count_in_window": "neurospatial.events.regressors:event_count_in_window",
    "event_indicator": "neurospatial.events.regressors:event_indicator",
    "exponential_kernel": "neurospatial.events.regressors:exponential_kernel",
    "distance_to_event_at_time": "neurospatial.events.regressors:distance_to_event_at_time",
    # Peri-event analysis
    "align_spikes_to_events": "neurospatial.events.alignment:align_spikes_to_events",
    "peri_event_histogram": "neurospatial.events.alignment:peri_event_histogram",
    "population_peri_event_histogram": "neurospatial.events.alignment:population_peri_event_histogram",
    "align_events": "neurospatial.events.alignment:align_events",
}


def __getattr__(name: str) -> Any:
    """Lazy import public API functions."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name].split(":")
    module = import_module(module_path)
    value = getattr(module, attr_name)

    # Cache in module globals for subsequent access
    globals()[name] = value
    return value


__all__ = [
    "PeriEventResult",
    "PopulationPeriEventResult",
    "add_positions",
    "align_events",
    "align_spikes_to_events",
    "distance_to_event_at_time",
    "event_count_in_window",
    "event_indicator",
    "events_in_region",
    "events_to_intervals",
    "exponential_kernel",
    "extract_movement_onset_events",
    "extract_region_crossing_events",
    "extract_threshold_crossing_events",
    "filter_by_intervals",
    "intervals_to_events",
    "peri_event_histogram",
    "plot_peri_event_histogram",
    "population_peri_event_histogram",
    "spatial_event_rate",
    "time_since_event",
    "time_to_event",
    "validate_events_dataframe",
    "validate_spatial_columns",
]
