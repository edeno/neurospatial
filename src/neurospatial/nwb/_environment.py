"""
Environment serialization to/from NWB scratch space.

This module provides functions for writing and reading Environment objects
to NWB scratch/ space using standard NWB types (no custom extension required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def environment_from_position(
    nwbfile: NWBFile,
    bin_size: float,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    units: str | None = None,
    frame: str | None = None,
    **kwargs: Any,
) -> Environment:
    """
    Create Environment from NWB Position data.

    Convenience function that reads position data and creates an Environment
    using Environment.from_samples().

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    bin_size : float
        Bin size for environment discretization.
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.
    units : str, optional
        Spatial units for the environment. If None, auto-detected from
        the SpatialSeries unit attribute.
    frame : str, optional
        Coordinate frame identifier for the environment.
    **kwargs
        Additional arguments passed to Environment.from_samples().
        Common kwargs include:
        - infer_active_bins : bool
            Whether to only include bins visited by the animal.
        - bin_count_threshold : int
            Minimum samples per bin when infer_active_bins=True.

    Returns
    -------
    Environment
        Environment discretized from position data.

    Raises
    ------
    KeyError
        If no Position container found, or if specified position_name not found.
    ImportError
        If pynwb is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = environment_from_position(nwbfile, bin_size=2.0)

    With explicit units and active bin inference:

    >>> env = environment_from_position(
    ...     nwbfile,
    ...     bin_size=5.0,
    ...     units="cm",
    ...     infer_active_bins=True,
    ...     bin_count_threshold=5,
    ... )
    """
    from neurospatial.nwb._core import _require_pynwb

    _require_pynwb()

    from neurospatial import Environment
    from neurospatial.nwb._behavior import read_position

    # Read position data from NWB
    positions, _timestamps = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )

    # Auto-detect units from SpatialSeries if not provided
    if units is None:
        units = _get_position_units(
            nwbfile,
            processing_module=processing_module,
            position_name=position_name,
        )

    # Create Environment from position samples
    env = Environment.from_samples(positions, bin_size=bin_size, **kwargs)

    # Set metadata
    env.units = units
    if frame is not None:
        env.frame = frame

    return env


def _get_position_units(
    nwbfile: NWBFile,
    processing_module: str | None = None,
    position_name: str | None = None,
) -> str:
    """
    Get the units from the Position SpatialSeries.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.

    Returns
    -------
    str
        The units from the SpatialSeries, or "cm" as fallback.
    """
    from pynwb.behavior import Position as PositionType

    from neurospatial.nwb._behavior import (
        _get_behavior_container,
        _get_spatial_series,
    )

    # Get Position container
    position_container = _get_behavior_container(
        nwbfile, PositionType, "Position", processing_module
    )

    # Get SpatialSeries
    spatial_series = _get_spatial_series(
        position_container, position_name, "SpatialSeries", "Position"
    )

    # Return units (default to "cm" if not set)
    return str(spatial_series.unit) if spatial_series.unit else "cm"
