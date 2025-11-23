"""
Core utilities for NWB integration.

This module provides common utilities used across NWB reading/writing functions,
including container discovery, import helpers, and validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynwb import NWBFile

logger = logging.getLogger("neurospatial.nwb")


def _require_pynwb():
    """
    Lazily import pynwb, raising ImportError with helpful message if not installed.

    Returns
    -------
    module
        The pynwb module.

    Raises
    ------
    ImportError
        If pynwb is not installed.
    """
    try:
        import pynwb

        return pynwb
    except ImportError as e:
        raise ImportError(
            "pynwb is required for NWB integration. "
            "Install with: pip install neurospatial[nwb]"
        ) from e


def _require_ndx_pose():
    """
    Lazily import ndx_pose, raising ImportError with helpful message if not installed.

    Returns
    -------
    module
        The ndx_pose module.

    Raises
    ------
    ImportError
        If ndx-pose is not installed.
    """
    try:
        import ndx_pose

        return ndx_pose
    except ImportError as e:
        raise ImportError(
            "ndx-pose is required for pose data. "
            "Install with: pip install neurospatial[nwb-pose]"
        ) from e


def _require_ndx_events():
    """
    Lazily import ndx_events, raising ImportError with helpful message if not installed.

    Returns
    -------
    module
        The ndx_events module.

    Raises
    ------
    ImportError
        If ndx-events is not installed.
    """
    try:
        import ndx_events

        return ndx_events
    except ImportError as e:
        raise ImportError(
            "ndx-events is required for event data. "
            "Install with: pip install neurospatial[nwb-events]"
        ) from e


def _find_containers_by_type(
    nwbfile: NWBFile, target_type: type
) -> list[tuple[str, Any]]:
    """
    Find all containers of a given type anywhere in the NWB file.

    Searches through processing modules and acquisition in a deterministic order.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to search.
    target_type : type
        The type of container to find (e.g., Position, CompassDirection).

    Returns
    -------
    list[tuple[str, Any]]
        List of (path, container) tuples, sorted alphabetically by path.
        Path format is "processing/{module_name}/{obj_name}" or "acquisition/{obj_name}".

    Notes
    -----
    Search order priority:
    1. processing/behavior/ (if exists)
    2. processing/* (other modules, alphabetically)
    3. acquisition/

    Examples
    --------
    >>> from pynwb.behavior import Position
    >>> containers = _find_containers_by_type(nwbfile, Position)
    >>> for path, container in containers:
    ...     print(f"Found Position at {path}")
    """
    found = []

    # Search processing modules
    for mod_name in sorted(nwbfile.processing.keys()):
        module = nwbfile.processing[mod_name]
        for obj_name in sorted(module.data_interfaces.keys()):
            obj = module.data_interfaces[obj_name]
            if isinstance(obj, target_type):
                found.append((f"processing/{mod_name}/{obj_name}", obj))

    # Search acquisition
    for obj_name in sorted(nwbfile.acquisition.keys()):
        obj = nwbfile.acquisition[obj_name]
        if isinstance(obj, target_type):
            found.append((f"acquisition/{obj_name}", obj))

    # Sort with priority: processing/behavior first, then alphabetically
    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        path = item[0]
        if path.startswith("processing/behavior/"):
            return (0, path)
        elif path.startswith("processing/"):
            return (1, path)
        else:
            return (2, path)

    return sorted(found, key=sort_key)


def _get_or_create_processing_module(
    nwbfile: NWBFile, name: str, description: str
) -> Any:
    """
    Get existing processing module or create a new one.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file.
    name : str
        Name of the processing module.
    description : str
        Description for the module (used only if creating new).

    Returns
    -------
    ProcessingModule
        The processing module.
    """
    if name in nwbfile.processing:
        return nwbfile.processing[name]
    return nwbfile.create_processing_module(name=name, description=description)
