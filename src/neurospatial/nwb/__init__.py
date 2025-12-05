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
    write_events : Write generic events DataFrame to NWB EventsTable
    write_laps : Write lap events to NWB processing/behavior/
    write_region_crossings : Write region crossing events
    write_environment : Write Environment to NWB scratch space

Factory Functions:
    environment_from_position : Create Environment from NWB Position data
    position_overlay_from_nwb : Create PositionOverlay from NWB Position
    bodypart_overlay_from_nwb : Create BodypartOverlay from ndx-pose
    head_direction_overlay_from_nwb : Create HeadDirectionOverlay from NWB

Examples
--------
Reading position data:

>>> from pynwb import NWBHDF5IO
>>> from neurospatial.nwb import read_position
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     positions, timestamps = read_position(nwbfile)

Creating an environment from NWB position data:

>>> from neurospatial.nwb import environment_from_position
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     env = environment_from_position(nwbfile, bin_size=2.0, units="cm")

Writing analysis results:

>>> from neurospatial.nwb import write_place_field, write_occupancy
>>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     write_place_field(nwbfile, env, place_field, name="cell_001")
...     write_occupancy(nwbfile, env, occupancy)
...     io.write(nwbfile)

Environment round-trip (save and reload):

>>> from neurospatial.nwb import write_environment, read_environment
>>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     write_environment(nwbfile, env, name="linear_track")
...     io.write(nwbfile)
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     loaded_env = read_environment(nwbfile, name="linear_track")

Creating animation overlays from NWB:

>>> from neurospatial.nwb import position_overlay_from_nwb
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     overlay = position_overlay_from_nwb(nwbfile, trail_length=10)
>>> env.animate_fields(fields, overlays=[overlay])  # doctest: +SKIP
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Lazy imports - functions are imported when first accessed
# This keeps NWB dependencies optional

# Mapping of public API names to their module paths
# Format: "name": "module_path:attribute_name"
_LAZY_IMPORTS: dict[str, str] = {
    # Reading functions
    "read_position": "neurospatial.nwb._behavior:read_position",
    "read_head_direction": "neurospatial.nwb._behavior:read_head_direction",
    "read_pose": "neurospatial.nwb._pose:read_pose",
    "read_events": "neurospatial.nwb._events:read_events",
    "read_intervals": "neurospatial.nwb._events:read_intervals",
    "read_environment": "neurospatial.nwb._environment:read_environment",
    "read_trials": "neurospatial.nwb._events:read_trials",
    # Writing functions
    "write_place_field": "neurospatial.nwb._fields:write_place_field",
    "write_occupancy": "neurospatial.nwb._fields:write_occupancy",
    "write_events": "neurospatial.nwb._events:write_events",
    "write_laps": "neurospatial.nwb._events:write_laps",
    "write_region_crossings": "neurospatial.nwb._events:write_region_crossings",
    "dataframe_to_events_table": "neurospatial.nwb._events:dataframe_to_events_table",
    "write_environment": "neurospatial.nwb._environment:write_environment",
    "write_trials": "neurospatial.nwb._events:write_trials",
    # Factory functions
    "environment_from_position": "neurospatial.nwb._environment:environment_from_position",
    "position_overlay_from_nwb": "neurospatial.nwb._overlays:position_overlay_from_nwb",
    "bodypart_overlay_from_nwb": "neurospatial.nwb._overlays:bodypart_overlay_from_nwb",
    "head_direction_overlay_from_nwb": "neurospatial.nwb._overlays:head_direction_overlay_from_nwb",
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
    "bodypart_overlay_from_nwb",
    "dataframe_to_events_table",
    "environment_from_position",
    "head_direction_overlay_from_nwb",
    "position_overlay_from_nwb",
    "read_environment",
    "read_events",
    "read_head_direction",
    "read_intervals",
    "read_pose",
    "read_position",
    "read_trials",
    "write_environment",
    "write_events",
    "write_laps",
    "write_occupancy",
    "write_place_field",
    "write_region_crossings",
    "write_trials",
]
