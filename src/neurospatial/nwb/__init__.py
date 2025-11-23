"""
NWB integration for neurospatial.

This module provides functions for reading and writing neurospatial data
to/from NWB (Neurodata Without Borders) files.

All NWB dependencies are optional and loaded lazily. Install with:
    pip install neurospatial[nwb]       # Basic NWB support
    pip install neurospatial[nwb-full]  # Full NWB support with extensions

Public API
----------
Reading Functions:
    read_position : Read position data from NWB Position container
    read_head_direction : Read head direction from NWB CompassDirection
    read_pose : Read pose estimation data from ndx-pose PoseEstimation
    read_events : Read events from ndx-events EventsTable
    read_intervals : Read intervals from NWB TimeIntervals (trials, epochs, etc.)
    read_environment : Read Environment from NWB scratch space

Writing Functions:
    write_place_field : Write place field to NWB analysis/
    write_occupancy : Write occupancy map to NWB analysis/
    write_laps : Write lap events to NWB processing/behavior/
    write_region_crossings : Write region crossing events
    write_environment : Write Environment to NWB scratch space

Factory Functions:
    environment_from_position : Create Environment from NWB Position data
    position_overlay_from_nwb : Create PositionOverlay from NWB Position
    bodypart_overlay_from_nwb : Create BodypartOverlay from ndx-pose
    head_direction_overlay_from_nwb : Create HeadDirectionOverlay from NWB
"""

from __future__ import annotations

# Lazy imports - functions are imported when first accessed
# This keeps NWB dependencies optional


def __getattr__(name: str):
    """Lazy import public API functions."""
    # Reading functions from _behavior.py
    if name == "read_position":
        from neurospatial.nwb._behavior import read_position

        return read_position
    if name == "read_head_direction":
        from neurospatial.nwb._behavior import read_head_direction

        return read_head_direction

    # Reading functions from _pose.py
    if name == "read_pose":
        from neurospatial.nwb._pose import read_pose

        return read_pose

    # Reading functions from _events.py
    if name == "read_events":
        from neurospatial.nwb._events import read_events

        return read_events
    if name == "read_intervals":
        from neurospatial.nwb._events import read_intervals

        return read_intervals

    # Reading functions from _environment.py
    if name == "read_environment":
        from neurospatial.nwb._environment import read_environment

        return read_environment

    # Writing functions from _fields.py
    if name == "write_place_field":
        from neurospatial.nwb._fields import write_place_field

        return write_place_field
    if name == "write_occupancy":
        from neurospatial.nwb._fields import write_occupancy

        return write_occupancy

    # Writing functions from _events.py
    if name == "write_laps":
        from neurospatial.nwb._events import write_laps

        return write_laps
    if name == "write_region_crossings":
        from neurospatial.nwb._events import write_region_crossings

        return write_region_crossings

    # Writing functions from _environment.py
    if name == "write_environment":
        from neurospatial.nwb._environment import write_environment

        return write_environment

    # Factory functions
    if name == "environment_from_position":
        from neurospatial.nwb._environment import environment_from_position

        return environment_from_position

    # Overlay factory functions from _overlays.py
    if name == "position_overlay_from_nwb":
        from neurospatial.nwb._overlays import position_overlay_from_nwb

        return position_overlay_from_nwb
    if name == "bodypart_overlay_from_nwb":
        from neurospatial.nwb._overlays import bodypart_overlay_from_nwb

        return bodypart_overlay_from_nwb
    if name == "head_direction_overlay_from_nwb":
        from neurospatial.nwb._overlays import head_direction_overlay_from_nwb

        return head_direction_overlay_from_nwb

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "bodypart_overlay_from_nwb",
    "environment_from_position",
    "head_direction_overlay_from_nwb",
    "position_overlay_from_nwb",
    "read_environment",
    "read_events",
    "read_head_direction",
    "read_intervals",
    "read_pose",
    "read_position",
    "write_environment",
    "write_laps",
    "write_occupancy",
    "write_place_field",
    "write_region_crossings",
]
