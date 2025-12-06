"""
Overlay factory functions for creating animation overlays from NWB data.

This module provides convenience functions for creating PositionOverlay,
BodypartOverlay, and HeadDirectionOverlay directly from NWB containers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pynwb import NWBFile

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
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     overlay = position_overlay_from_nwb(nwbfile, trail_length=10)
    >>> env.animate_fields(fields, overlays=[overlay])  # doctest: +SKIP
    """
    from neurospatial.animation.overlays import PositionOverlay
    from neurospatial.io.nwb._behavior import read_position

    # Read position data from NWB
    positions, timestamps = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )

    # Create and return PositionOverlay
    return PositionOverlay(
        data=positions,
        times=timestamps,
        color=color,
        size=size,
        trail_length=trail_length,
        **kwargs,
    )


def bodypart_overlay_from_nwb(
    nwbfile: NWBFile,
    *,
    pose_estimation_name: str | None = None,
    colors: dict[str, str] | None = None,
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
        Mapping from bodypart name to color. If None, uses skeleton's
        node_colors or default colors.
    **kwargs
        Additional arguments passed to BodypartOverlay. Supported kwargs:

        - interp : {"linear", "nearest"}, optional
            Interpolation method for temporal alignment. Default is "linear".

    Returns
    -------
    BodypartOverlay
        Overlay with bodypart trajectories and skeleton from NWB.

    Raises
    ------
    KeyError
        If no PoseEstimation container found, or if specified name not found.
    ImportError
        If ndx-pose is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     overlay = bodypart_overlay_from_nwb(nwbfile)
    >>> env.animate_fields(fields, overlays=[overlay])  # doctest: +SKIP

    With custom colors:

    >>> overlay = bodypart_overlay_from_nwb(  # doctest: +SKIP
    ...     nwbfile,
    ...     colors={"nose": "yellow", "body": "red", "tail": "blue"},
    ... )
    """
    from neurospatial.animation.overlays import BodypartOverlay
    from neurospatial.io.nwb._pose import read_pose

    # Read pose data from NWB
    bodyparts, timestamps, skeleton = read_pose(
        nwbfile,
        pose_estimation_name=pose_estimation_name,
    )

    # Create and return BodypartOverlay
    return BodypartOverlay(
        data=bodyparts,
        times=timestamps,
        skeleton=skeleton,  # Auto-extracted from PoseEstimation.skeleton
        colors=colors,
        **kwargs,
    )


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
        Additional arguments passed to HeadDirectionOverlay. Supported kwargs:

        - interp : {"linear", "nearest"}, optional
            Interpolation method for temporal alignment. Default is "linear".

    Returns
    -------
    HeadDirectionOverlay
        Overlay with head direction angles and timestamps from NWB.

    Raises
    ------
    KeyError
        If no CompassDirection container found, or if specified name not found.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     overlay = head_direction_overlay_from_nwb(nwbfile, color="cyan")
    >>> env.animate_fields(fields, overlays=[overlay])  # doctest: +SKIP
    """
    from neurospatial.animation.overlays import HeadDirectionOverlay
    from neurospatial.io.nwb._behavior import read_head_direction

    # Read head direction data from NWB
    angles, timestamps = read_head_direction(
        nwbfile,
        processing_module=processing_module,
        compass_name=compass_name,
    )

    # Create and return HeadDirectionOverlay
    return HeadDirectionOverlay(
        data=angles,
        times=timestamps,
        color=color,
        length=length,
        **kwargs,
    )
