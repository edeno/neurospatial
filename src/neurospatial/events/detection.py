"""
Event detection functions for neurospatial.

This module provides functions to detect events from trajectories and signals:
- extract_region_crossing_events: Detect region entry/exit events
- extract_threshold_crossing_events: Detect signal threshold crossings
- extract_movement_onset_events: Detect movement onset from position

Spatial utilities:
- add_positions: Add x, y columns to events by interpolation
- events_in_region: Filter events to those within a region
- spatial_event_rate: Compute spatial rate map of events
"""

from __future__ import annotations

# Implementations will be added in Milestone 3
