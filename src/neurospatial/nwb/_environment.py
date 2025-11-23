"""
Environment serialization to/from NWB scratch space.

This module provides functions for writing and reading Environment objects
to NWB scratch/ space using standard NWB types (no custom extension required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment


def write_environment(
    nwbfile: NWBFile,
    env: Environment,
    name: str = "spatial_environment",
) -> None:
    """
    Write Environment to NWB scratch space using standard types.

    Creates structure in scratch/:
        scratch/{name}/
            bin_centers       # Dataset (n_bins, n_dims)
            edges             # Dataset (n_edges, 2) - edge list
            edge_weights      # Dataset (n_edges,) - optional
            dimension_ranges  # Dataset (n_dims, 2)
            regions           # DynamicTable with point/polygon data
            metadata.json     # Dataset (string) - JSON blob for extras

    Group attributes:
        units, frame, n_dims, layout_type

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to write to.
    env : Environment
        The Environment to serialize.
    name : str, default "spatial_environment"
        Name for the environment group in scratch/.

    Raises
    ------
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r+") as io:
    ...     nwbfile = io.read()
    ...     write_environment(nwbfile, env, name="linear_track")
    ...     io.write(nwbfile)
    """
    raise NotImplementedError("write_environment not yet implemented")


def read_environment(
    nwbfile: NWBFile,
    name: str = "spatial_environment",
) -> Environment:
    """
    Read Environment from NWB scratch space.

    Reconstructs Environment from stored bin_centers and edge list.
    Rebuilds connectivity graph and regions.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    name : str, default "spatial_environment"
        Name of the environment group in scratch/.

    Returns
    -------
    Environment
        Reconstructed Environment with all attributes.

    Raises
    ------
    KeyError
        If environment not found in scratch/{name}.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = read_environment(nwbfile, name="linear_track")
    """
    raise NotImplementedError("read_environment not yet implemented")
