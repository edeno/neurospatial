"""
GLM regressor generation for neurospatial events.

This module provides functions to generate regressors for GLM design matrices:

Temporal regressors:
- time_since_event: Time since most recent event
- time_to_event: Time until next event
- event_count_in_window: Count events in time window
- event_indicator: Binary indicator of event presence
- exponential_kernel: Convolve events with exponential kernel

Spatial regressors:
- distance_to_event_at_time: Distance to last/next event location
"""

from __future__ import annotations

# Implementations will be added in Milestones 2 and 5
