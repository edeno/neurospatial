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
        Additional arguments passed to PositionOverlay. ``units`` and ``frame``
        may supply undeclared coordinate metadata; known NWB declarations
        cannot be erased or overridden with conflicting declarations.

    Returns
    -------
    PositionOverlay
        Overlay with position data and timestamps from NWB.

    Raises
    ------
    ValueError
        If a keyword coordinate declaration conflicts with NWB metadata.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     overlay = position_overlay_from_nwb(nwbfile, trail_length=10)
    >>> env.animate_fields(
    ...     fields,
    ...     frame_times=frame_times,
    ...     overlays=[overlay],
    ... )  # doctest: +SKIP
    """
    from neurospatial.animation.overlays import PositionOverlay
    from neurospatial.io.nwb._adapters import merge_overlay_coordinate_metadata
    from neurospatial.io.nwb._behavior import read_position

    # Read position data from NWB
    position_data = read_position(
        nwbfile,
        processing_module=processing_module,
        position_name=position_name,
    )
    positions = position_data.positions
    timestamps = position_data.times
    declarations = merge_overlay_coordinate_metadata(
        position_data.units,
        position_data.frame,
        kwargs,
        context="PositionOverlay",
        source="position",
    )

    # Create and return PositionOverlay
    return PositionOverlay(
        positions=positions,
        times=timestamps,
        units=declarations["units"],
        frame=declarations["frame"],
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
        - units, frame : str, optional
            Coordinate declarations when absent from NWB. Known NWB metadata
            cannot be overridden with conflicting declarations.

    Returns
    -------
    BodypartOverlay
        Overlay with bodypart trajectories and skeleton from NWB.

    Raises
    ------
    KeyError
        If no PoseEstimation container found, or if specified name not found.
    ValueError
        If bodypart clocks or known coordinate declarations disagree, or a
        keyword declaration conflicts with the NWB metadata.
    ImportError
        If ndx-pose is not installed.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     overlay = bodypart_overlay_from_nwb(nwbfile)
    >>> env.animate_fields(
    ...     fields,
    ...     frame_times=frame_times,
    ...     overlays=[overlay],
    ... )  # doctest: +SKIP

    With custom colors:

    >>> overlay = bodypart_overlay_from_nwb(  # doctest: +SKIP
    ...     nwbfile,
    ...     colors={"nose": "yellow", "body": "red", "tail": "blue"},
    ... )
    """
    from neurospatial.animation.overlays import BodypartOverlay
    from neurospatial.io.nwb._adapters import merge_overlay_coordinate_metadata
    from neurospatial.io.nwb._pose import _read_pose_with_metadata

    # Read pose data from NWB
    bodyparts, timestamps, skeleton, units, frame = _read_pose_with_metadata(
        nwbfile,
        pose_estimation_name=pose_estimation_name,
    )

    declarations = merge_overlay_coordinate_metadata(
        units, frame, kwargs, context="BodypartOverlay", source="pose"
    )

    # Create and return BodypartOverlay
    return BodypartOverlay(
        data=bodyparts,
        times=timestamps,
        skeleton=skeleton,  # Auto-extracted from PoseEstimation.skeleton
        colors=colors,
        units=declarations["units"],
        frame=declarations["frame"],
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
    >>> env.animate_fields(
    ...     fields,
    ...     frame_times=frame_times,
    ...     overlays=[overlay],
    ... )  # doctest: +SKIP
    """
    from neurospatial.animation.overlays import HeadDirectionOverlay
    from neurospatial.io.nwb._behavior import read_head_direction

    # Read head direction data from NWB
    head_direction_data = read_head_direction(
        nwbfile,
        processing_module=processing_module,
        compass_name=compass_name,
    )
    angles = head_direction_data.headings
    timestamps = head_direction_data.times

    # Create and return HeadDirectionOverlay
    return HeadDirectionOverlay(
        headings=angles,
        times=timestamps,
        color=color,
        length=length,
        **kwargs,
    )
