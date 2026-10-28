"""
Events module for neurospatial.

This module aligns spikes, continuous signals and event tables to events,
converts between point events and intervals, and builds GLM regressors.

Public API
----------
Result Dataclasses:
    PeriEventResult : Result from peri-event histogram analysis
    PopulationPeriEventResult : Result from population peri-event analysis

Spatial Utilities:
    add_positions : Add x, y (and z) columns to events by interpolation

Interval Utilities:
    intervals_to_events : Convert intervals to point events
    events_to_intervals : Pair start/stop events into intervals
    filter_by_intervals : Filter events by time intervals

GLM Regressors:
    time_to_nearest_event : Signed time to nearest event (peri-event time)
    event_count_in_window : Count events in time window
    event_indicator : Binary indicator of event presence

    Per-sample spatial covariates (``distance_to_reward``,
    ``distance_to_boundary``) are in ``neurospatial.behavior``.

Peri-Event Analysis:
    align_spikes_to_events : Get per-trial spike times
    peri_event_histogram : Compute PSTH
    population_peri_event_histogram : Compute population PSTH
    align_events : Align events to reference events

Continuous signals:
    align_continuous_to_events : Sample speed, heading or zone around events
    ContinuousEventResult : Result from continuous event alignment

Visualization:
    plot_peri_event_histogram : Plot PSTH results
    plot_peri_event_response : Plot a raster and PSTH for the same retained events

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

from neurospatial.events._core import (
    PeriEventResult,
    PopulationPeriEventResult,
    plot_peri_event_histogram,
)
from neurospatial.events._plotting import plot_peri_event_response
from neurospatial.events.alignment import (
    align_events,
    align_spikes_to_events,
    peri_event_histogram,
    population_peri_event_histogram,
)
from neurospatial.events.continuous import (
    ContinuousEventResult,
    align_continuous_to_events,
)
from neurospatial.events.detection import add_positions
from neurospatial.events.intervals import (
    events_to_intervals,
    filter_by_intervals,
    intervals_to_events,
)
from neurospatial.events.regressors import (
    event_count_in_window,
    event_indicator,
    time_to_nearest_event,
)

# ruff: noqa: RUF022  - Intentionally organized into groups with comments
__all__ = [
    # Result dataclasses
    "PeriEventResult",
    "PopulationPeriEventResult",
    # Spatial utilities
    "add_positions",
    # Interval utilities
    "events_to_intervals",
    "filter_by_intervals",
    "intervals_to_events",
    # GLM regressors
    "event_count_in_window",
    "event_indicator",
    "time_to_nearest_event",
    # Peri-event analysis
    "align_events",
    "align_spikes_to_events",
    "peri_event_histogram",
    "population_peri_event_histogram",
    # Continuous signals
    "ContinuousEventResult",
    "align_continuous_to_events",
    # Visualization
    "plot_peri_event_histogram",
    "plot_peri_event_response",
]
