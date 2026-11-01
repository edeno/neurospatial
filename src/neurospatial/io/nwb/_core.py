"""
Core utilities for NWB integration.

This module provides common utilities used across NWB reading/writing functions,
including container discovery, import helpers, and validation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neurospatial._optional import require_optional

if TYPE_CHECKING:
    from pynwb import NWBFile

logger = logging.getLogger("neurospatial.nwb")


def _refuse_in_place_replacement(nwbfile: NWBFile, what: str) -> None:
    """Refuse to replace an object in a file opened for in-place writing.

    pynwb cannot delete an object that is already written to disk, so removing
    one from a file opened with ``"r+"`` or ``"a"`` makes the next
    ``io.write`` fail. Replacing it in a file opened with ``"r"`` and exporting
    to a new path works, as does replacing it in an in-memory ``NWBFile``.

    Parameters
    ----------
    nwbfile : NWBFile
        The file about to have an object replaced.
    what : str
        Description of the object, e.g. ``"events table 'rewards'"``.

    Raises
    ------
    NotImplementedError
        If ``nwbfile`` was read from disk by an IO opened with ``"r+"`` or
        ``"a"``.
    """
    read_io = getattr(nwbfile, "read_io", None)
    mode = getattr(read_io, "mode", None)
    if mode in (None, "r"):
        return
    msg = (
        f"Cannot replace the {what}: the file was opened with mode={mode!r}, "
        "and pynwb cannot delete an object that is already written to disk. "
        "Open the file with mode='r', call this function with overwrite=True, "
        "then save a copy with "
        "NWBHDF5IO(new_path, 'w').export(src_io=io, nwbfile=nwbfile). "
        "Or write under a different name."
    )
    raise NotImplementedError(msg)


def _clear_for_write(
    nwbfile: NWBFile, container: Any, name: str, *, overwrite: bool, what: str
) -> bool:
    """Make room for ``name`` in ``container``; return whether one was removed.

    Call it after the replacement is built and checked, so a bad replacement
    never costs the original.

    Parameters
    ----------
    nwbfile : NWBFile
        The file that holds ``container``.
    container : mapping
        Where the object lives, e.g. ``nwbfile.analysis``,
        ``nwbfile.intervals`` or a processing module's ``data_interfaces``.
    name : str
        Name of the object about to be written.
    overwrite : bool
        Whether an existing object may be replaced.
    what : str
        Description of the object, e.g. ``"events table 'rewards'"``.

    Raises
    ------
    ValueError
        If ``name`` exists and ``overwrite=False``.
    NotImplementedError
        If ``name`` exists in a file opened for in-place writing.
    """
    if name not in container:
        return False
    if not overwrite:
        msg = f"The {what} already exists. Pass overwrite=True to replace it."
        raise ValueError(msg)
    _refuse_in_place_replacement(nwbfile, what)
    del container[name]
    logger.info("Overwriting the existing %s", what)
    return True


class _MissingObjectError(KeyError):
    """A ``KeyError`` that prints its message as written.

    ``str(KeyError)`` shows the message's repr, which quotes it and prints
    line breaks as ``\\n``.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _require_pynwb() -> Any:
    """Import pynwb, or raise an ImportError naming the ``nwb`` extra."""
    return require_optional("pynwb", extra="nwb", purpose="NWB integration")


def _require_ndx_pose() -> Any:
    """Import ndx_pose, or raise an ImportError naming the ``nwb`` extra."""
    return require_optional("ndx_pose", extra="nwb", purpose="NWB pose data")


def _require_ndx_events() -> Any:
    """Import ndx_events, or raise an ImportError naming the ``nwb`` extra."""
    return require_optional("ndx_events", extra="nwb", purpose="NWB event data")


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
        List of (path, container) tuples in the search order below.
        Path format is "processing/{module_name}/{obj_name}" or "acquisition/{obj_name}".

    Notes
    -----
    Search order priority:
    1. processing/behavior/ (if exists)
    2. processing/* (other modules, alphabetically)
    3. acquisition/

    Examples
    --------
    >>> from pynwb.behavior import Position  # doctest: +SKIP
    >>> containers = _find_containers_by_type(nwbfile, Position)  # doctest: +SKIP
    >>> for path, container in containers:  # doctest: +SKIP
    ...     print(f"Found Position at {path}")
    """
    found = []

    # Search processing modules
    for mod_name, module in nwbfile.processing.items():
        for obj_name, obj in module.data_interfaces.items():
            if isinstance(obj, target_type):
                found.append((f"processing/{mod_name}/{obj_name}", obj))

    # Search acquisition
    for obj_name, obj in nwbfile.acquisition.items():
        if isinstance(obj, target_type):
            found.append((f"acquisition/{obj_name}", obj))

    # Paths are unique, so this one sort fixes the order: processing/behavior
    # first, then other processing modules, then acquisition, each by path.
    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        path = item[0]
        if path.startswith("processing/behavior/"):
            return (0, path)
        if path.startswith("processing/"):
            return (1, path)
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
