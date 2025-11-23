"""
Spatial fields writing to NWB analysis containers.

This module provides functions for writing spatial analysis results
(place fields, occupancy maps) to NWB analysis/ containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment


def write_place_field(
    nwbfile: NWBFile,
    env: Environment,
    field: NDArray[np.float64],
    name: str = "place_field",
    description: str = "",
    *,
    overwrite: bool = False,
) -> None:
    """
    Write spatial field to NWB file.

    Stores field values aligned with environment bin centers in analysis/.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment providing bin structure.
    field : NDArray[np.float64], shape (n_bins,) or (n_time, n_bins)
        Spatial field values. First dimension must match env.n_bins
        (or second dimension for time-varying fields).
    name : str, default "place_field"
        Name for the place field in NWB.
    description : str, default ""
        Description of the place field.
    overwrite : bool, default False
        If True, replace existing field with same name.
        If False, raise ValueError on duplicate name.

    Raises
    ------
    ValueError
        If field with same name exists and overwrite=False.
        If field shape doesn't match env.n_bins.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> from neurospatial import compute_place_field
    >>> place_field = compute_place_field(env, spike_times, timestamps, positions)
    >>> with NWBHDF5IO("session.nwb", "r+") as io:
    ...     nwbfile = io.read()
    ...     write_place_field(nwbfile, env, place_field, name="cell_001")
    ...     io.write(nwbfile)
    """
    raise NotImplementedError("write_place_field not yet implemented")


def write_occupancy(
    nwbfile: NWBFile,
    env: Environment,
    occupancy: NDArray[np.float64],
    name: str = "occupancy",
    description: str = "",
    *,
    units: str = "seconds",
    overwrite: bool = False,
) -> None:
    """
    Write occupancy map to NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment providing bin structure.
    occupancy : NDArray[np.float64], shape (n_bins,)
        Occupancy values per bin.
    name : str, default "occupancy"
        Name for the occupancy map in NWB.
    description : str, default ""
        Description of the occupancy map.
    units : str, default "seconds"
        Units for occupancy values (e.g., "seconds", "probability").
    overwrite : bool, default False
        If True, replace existing occupancy with same name.

    Raises
    ------
    ValueError
        If occupancy shape doesn't match env.n_bins.
    ImportError
        If pynwb is not installed.
    """
    raise NotImplementedError("write_occupancy not yet implemented")
