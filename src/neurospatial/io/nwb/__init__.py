"""
NWB integration for neurospatial.

This module provides functions for reading and writing neurospatial data
to/from NWB (Neurodata Without Borders) files.

All NWB dependencies are optional and loaded lazily. Install with:
    pip install neurospatial[nwb]  # pynwb + hdmf + ndx-pose / ndx-events extensions

Public API
----------
Reading Functions:
    read_position : Read position data from NWB Position container
    read_head_direction : Read head direction from NWB CompassDirection
    read_pose : Read pose estimation data from ndx-pose PoseEstimation
    read_events : Read events from ndx-events EventsTable
    read_intervals : Read intervals from NWB TimeIntervals (trials, epochs, etc.)
    read_units : Read spike-time arrays from an NWB units table
    read_spatial_rates : Read a SpatialRatesResult saved by write_spatial_rates
    read_environment : Read Environment from NWB scratch space

Writing Functions:
    write_spatial_rates : Write a SpatialRatesResult with its occupancy and environment
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
>>> from neurospatial.io.nwb import read_position
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     pos = read_position(nwbfile)

Creating an environment from NWB position data:

>>> from neurospatial.io.nwb import environment_from_position
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     env = environment_from_position(nwbfile, bin_size=2.0, units="cm")

Writing analysis results:

>>> from neurospatial.io.nwb import write_spatial_rates
>>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     write_spatial_rates(nwbfile, rates, name="ca1_place_fields")
...     io.write(nwbfile)

Environment round-trip (save and reload):

>>> from neurospatial.io.nwb import write_environment, read_environment
>>> with NWBHDF5IO("session.nwb", "r+") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     write_environment(nwbfile, env, name="linear_track")
...     io.write(nwbfile)
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     loaded_env = read_environment(nwbfile, name="linear_track")

Creating animation overlays from NWB:

>>> from neurospatial.io.nwb import position_overlay_from_nwb
>>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
...     nwbfile = io.read()
...     overlay = position_overlay_from_nwb(nwbfile, trail_length=10)
>>> env.animate_fields(
...     fields,
...     frame_times=frame_times,
...     overlays=[overlay],
... )  # doctest: +SKIP
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Type checkers see the real signatures; at runtime the names load on first
# access, so the NWB dependencies stay optional.
if TYPE_CHECKING:
    from neurospatial.io.nwb._behavior import (
        read_head_direction as read_head_direction,
    )
    from neurospatial.io.nwb._behavior import (
        read_position as read_position,
    )
    from neurospatial.io.nwb._environment import (
        environment_from_position as environment_from_position,
    )
    from neurospatial.io.nwb._environment import (
        read_environment as read_environment,
    )
    from neurospatial.io.nwb._environment import (
        write_environment as write_environment,
    )
    from neurospatial.io.nwb._events import (
        dataframe_to_events_table as dataframe_to_events_table,
    )
    from neurospatial.io.nwb._events import (
        read_events as read_events,
    )
    from neurospatial.io.nwb._events import (
        read_intervals as read_intervals,
    )
    from neurospatial.io.nwb._events import (
        read_trials as read_trials,
    )
    from neurospatial.io.nwb._events import (
        write_events as write_events,
    )
    from neurospatial.io.nwb._events import (
        write_laps as write_laps,
    )
    from neurospatial.io.nwb._events import (
        write_region_crossings as write_region_crossings,
    )
    from neurospatial.io.nwb._events import (
        write_trials as write_trials,
    )
    from neurospatial.io.nwb._fields import (
        read_spatial_rates as read_spatial_rates,
    )
    from neurospatial.io.nwb._fields import (
        write_spatial_rates as write_spatial_rates,
    )
    from neurospatial.io.nwb._holders import (
        NWBHeadDirection as NWBHeadDirection,
    )
    from neurospatial.io.nwb._holders import (
        NWBPosition as NWBPosition,
    )
    from neurospatial.io.nwb._holders import (
        NWBUnits as NWBUnits,
    )
    from neurospatial.io.nwb._overlays import (
        bodypart_overlay_from_nwb as bodypart_overlay_from_nwb,
    )
    from neurospatial.io.nwb._overlays import (
        head_direction_overlay_from_nwb as head_direction_overlay_from_nwb,
    )
    from neurospatial.io.nwb._overlays import (
        position_overlay_from_nwb as position_overlay_from_nwb,
    )
    from neurospatial.io.nwb._pose import (
        read_pose as read_pose,
    )
    from neurospatial.io.nwb._units import (
        read_units as read_units,
    )

# Lazy imports - functions are imported when first accessed
# This keeps NWB dependencies optional

# Mapping of public API names to their module paths
# Format: "name": "module_path:attribute_name"
_LAZY_IMPORTS: dict[str, str] = {
    # Named array holders (no optional dependency imports)
    "NWBPosition": "neurospatial.io.nwb._holders:NWBPosition",
    "NWBHeadDirection": "neurospatial.io.nwb._holders:NWBHeadDirection",
    "NWBUnits": "neurospatial.io.nwb._holders:NWBUnits",
    # Reading functions
    "read_position": "neurospatial.io.nwb._behavior:read_position",
    "read_head_direction": "neurospatial.io.nwb._behavior:read_head_direction",
    "read_pose": "neurospatial.io.nwb._pose:read_pose",
    "read_events": "neurospatial.io.nwb._events:read_events",
    "read_intervals": "neurospatial.io.nwb._events:read_intervals",
    "read_units": "neurospatial.io.nwb._units:read_units",
    "read_spatial_rates": "neurospatial.io.nwb._fields:read_spatial_rates",
    "read_environment": "neurospatial.io.nwb._environment:read_environment",
    "read_trials": "neurospatial.io.nwb._events:read_trials",
    # Writing functions
    "write_spatial_rates": "neurospatial.io.nwb._fields:write_spatial_rates",
    "write_events": "neurospatial.io.nwb._events:write_events",
    "write_laps": "neurospatial.io.nwb._events:write_laps",
    "write_region_crossings": "neurospatial.io.nwb._events:write_region_crossings",
    "dataframe_to_events_table": "neurospatial.io.nwb._events:dataframe_to_events_table",
    "write_environment": "neurospatial.io.nwb._environment:write_environment",
    "write_trials": "neurospatial.io.nwb._events:write_trials",
    # Factory functions
    "environment_from_position": "neurospatial.io.nwb._environment:environment_from_position",
    "position_overlay_from_nwb": "neurospatial.io.nwb._overlays:position_overlay_from_nwb",
    "bodypart_overlay_from_nwb": "neurospatial.io.nwb._overlays:bodypart_overlay_from_nwb",
    "head_direction_overlay_from_nwb": "neurospatial.io.nwb._overlays:head_direction_overlay_from_nwb",
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
    "NWBHeadDirection",
    "NWBPosition",
    "NWBUnits",
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
    "read_spatial_rates",
    "read_trials",
    "read_units",
    "write_environment",
    "write_events",
    "write_laps",
    "write_region_crossings",
    "write_spatial_rates",
    "write_trials",
]
