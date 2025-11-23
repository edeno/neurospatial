"""
Overlay factory functions for creating animation overlays from NWB data.

This module provides convenience functions for creating PositionOverlay,
BodypartOverlay, and HeadDirectionOverlay directly from NWB containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial import Environment
    from neurospatial.animation.overlays import (
        BodypartOverlay,
        HeadDirectionOverlay,
        PositionOverlay,
    )


def position_overlay_from_nwb(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    color: str = "red",
    size: float = 12.0,
    trail_length: int = 0,
    **kwargs: Any,
) -> PositionOverlay:
    """
    Create PositionOverlay from NWB Position data.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of processing module containing Position.
    position_name : str, optional
        Name of specific SpatialSeries within Position.
    color : str, default "red"
        Color for the position marker.
    size : float, default 12.0
        Size of the position marker.
    trail_length : int, default 0
        Number of previous positions to show as trail.
    **kwargs
        Additional arguments passed to PositionOverlay.

    Returns
    -------
    PositionOverlay
        Overlay with position data and timestamps from NWB.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     overlay = position_overlay_from_nwb(nwbfile, trail_length=10)
    >>> env.animate_fields(fields, overlays=[overlay])
    """
    raise NotImplementedError("position_overlay_from_nwb not yet implemented")


def bodypart_overlay_from_nwb(
    nwbfile: NWBFile,
    *,
    pose_estimation_name: str | None = None,
    colors: dict[str, str] | None = None,
    skeleton_color: str = "white",
    skeleton_width: float = 2.0,
    **kwargs: Any,
) -> BodypartOverlay:
    """
    Create BodypartOverlay from NWB PoseEstimation data.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    pose_estimation_name : str, optional
        Name of specific PoseEstimation container.
    colors : dict[str, str], optional
        Mapping from bodypart name to color.
    skeleton_color : str, default "white"
        Color for skeleton edges.
    skeleton_width : float, default 2.0
        Width of skeleton edges.
    **kwargs
        Additional arguments passed to BodypartOverlay.

    Returns
    -------
    BodypartOverlay
        Overlay with bodypart trajectories and skeleton from NWB.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     overlay = bodypart_overlay_from_nwb(nwbfile, skeleton_color="yellow")
    >>> env.animate_fields(fields, overlays=[overlay])
    """
    raise NotImplementedError("bodypart_overlay_from_nwb not yet implemented")


def head_direction_overlay_from_nwb(
    nwbfile: NWBFile,
    *,
    processing_module: str | None = None,
    compass_name: str | None = None,
    color: str = "yellow",
    length: float = 15.0,
    **kwargs: Any,
) -> HeadDirectionOverlay:
    """
    Create HeadDirectionOverlay from NWB CompassDirection data.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    processing_module : str, optional
        Name of processing module containing CompassDirection.
    compass_name : str, optional
        Name of specific SpatialSeries within CompassDirection.
    color : str, default "yellow"
        Color for the head direction arrow.
    length : float, default 15.0
        Length of the arrow in environment units.
    **kwargs
        Additional arguments passed to HeadDirectionOverlay.

    Returns
    -------
    HeadDirectionOverlay
        Overlay with head direction angles and timestamps from NWB.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     overlay = head_direction_overlay_from_nwb(nwbfile, color="cyan")
    >>> env.animate_fields(fields, overlays=[overlay])
    """
    raise NotImplementedError("head_direction_overlay_from_nwb not yet implemented")


def environment_from_position(
    nwbfile: NWBFile,
    bin_size: float,
    *,
    processing_module: str | None = None,
    position_name: str | None = None,
    units: str = "cm",
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
    units : str, default "cm"
        Spatial units for the environment.
    **kwargs
        Additional arguments passed to Environment.from_samples().

    Returns
    -------
    Environment
        Environment discretized from position data.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     env = environment_from_position(nwbfile, bin_size=2.0)
    """
    raise NotImplementedError("environment_from_position not yet implemented")
