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

Spatial Utilities:
    add_positions : Add x, y columns to events by interpolation

Interval Utilities:
    intervals_to_events : Convert intervals to point events
    events_to_intervals : Pair start/stop events into intervals
    filter_by_intervals : Filter events by time intervals

GLM Regressors:
    time_to_nearest_event : Signed time to nearest event (peri-event time)
    event_count_in_window : Count events in time window
    event_indicator : Binary indicator of event presence
    distance_to_reward : Distance to reward location (spatial regressor)
    distance_to_boundary : Distance to environment boundaries (spatial regressor)

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
>>> print(
...     f"Peak at {result.bin_centers[result.histogram.argmax()]:.2f}s"
... )  # doctest: +SKIP

Creating GLM regressors:

>>> from neurospatial.events import time_to_nearest_event, event_indicator
>>> peri_event_time = time_to_nearest_event(  # doctest: +SKIP
...     sample_times, reward_times, max_time=2.0
... )
>>> is_near_event = event_indicator(  # doctest: +SKIP
...     sample_times, reward_times, window=(-0.5, 1.0)
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
    # Spatial utilities
    "add_positions": "neurospatial.events.detection:add_positions",
    # Interval utilities
    "intervals_to_events": "neurospatial.events.intervals:intervals_to_events",
    "events_to_intervals": "neurospatial.events.intervals:events_to_intervals",
    "filter_by_intervals": "neurospatial.events.intervals:filter_by_intervals",
    # GLM regressors (temporal)
    "time_to_nearest_event": "neurospatial.events.regressors:time_to_nearest_event",
    "event_count_in_window": "neurospatial.events.regressors:event_count_in_window",
    "event_indicator": "neurospatial.events.regressors:event_indicator",
    # GLM regressors (spatial)
    "distance_to_reward": "neurospatial.events.regressors:distance_to_reward",
    "distance_to_boundary": "neurospatial.events.regressors:distance_to_boundary",
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


# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Result dataclasses
    "PeriEventResult",
    "PopulationPeriEventResult",
    # Validation helpers
    "validate_events_dataframe",
    "validate_spatial_columns",
    # Spatial utilities
    "add_positions",
    # Interval utilities
    "events_to_intervals",
    "filter_by_intervals",
    "intervals_to_events",
    # GLM regressors (temporal)
    "event_count_in_window",
    "event_indicator",
    "time_to_nearest_event",
    # GLM regressors (spatial)
    "distance_to_boundary",
    "distance_to_reward",
    # Peri-event analysis
    "align_events",
    "align_spikes_to_events",
    "peri_event_histogram",
    "population_peri_event_histogram",
    # Visualization
    "plot_peri_event_histogram",
]
