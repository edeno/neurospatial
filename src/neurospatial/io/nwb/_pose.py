"""
Pose estimation reading from NWB ndx-pose containers.

This module provides functions for reading PoseEstimation data from
ndx-pose containers, including bodypart trajectories and skeleton definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.io.nwb._adapters import timestamps_from_series
from neurospatial.io.nwb._core import (
    _find_containers_by_type,
    _require_ndx_pose,
    logger,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial.animation.skeleton import Skeleton


def read_pose(
    nwbfile: NWBFile,
    pose_estimation_name: str | None = None,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], Skeleton]:
    """
    Read pose estimation data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    pose_estimation_name : str, optional
        Name of the specific PoseEstimation container.
        If None, auto-discovers using priority order:
        processing/behavior > processing/*.

    Returns
    -------
    bodyparts : dict[str, NDArray[np.float64]]
        Mapping from bodypart name to coordinates, each shape (n_samples, n_dims).
    timestamps : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds (shared across all bodyparts).
    skeleton : Skeleton
        Skeleton definition from the PoseEstimation container.

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
    ...     bodyparts, timestamps, skeleton = read_pose(nwbfile)
    ...     print(f"Found {len(bodyparts)} bodyparts")
    """
    _require_ndx_pose()
    from ndx_pose import PoseEstimation as PoseEstimationType

    from neurospatial.animation.skeleton import Skeleton as NSSkeleton

    # Find PoseEstimation container
    pose_estimation = _get_pose_estimation_container(
        nwbfile, PoseEstimationType, pose_estimation_name
    )

    # Extract bodyparts as dict
    bodyparts: dict[str, NDArray[np.float64]] = {}
    timestamps: NDArray[np.float64] | None = None

    for series_name in sorted(pose_estimation.pose_estimation_series.keys()):
        series = pose_estimation.pose_estimation_series[series_name]
        bodyparts[series_name] = np.asarray(series.data[:], dtype=np.float64)

        # Get timestamps from the first series (they should all be the same)
        if timestamps is None:
            timestamps = _get_timestamps(series)

    if timestamps is None:
        raise ValueError("PoseEstimation container has no pose estimation series")

    # Convert ndx-pose Skeleton to neurospatial Skeleton
    skeleton = NSSkeleton.from_ndx_pose(pose_estimation.skeleton)

    return bodyparts, timestamps, skeleton


def _get_pose_estimation_container(
    nwbfile: NWBFile,
    target_type: type,
    pose_estimation_name: str | None,
) -> Any:
    """
    Get a PoseEstimation container from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to search.
    target_type : type
        The PoseEstimation type class.
    pose_estimation_name : str, optional
        If provided, look for this specific PoseEstimation by name.

    Returns
    -------
    PoseEstimation
        The PoseEstimation container.

    Raises
    ------
    KeyError
        If no container found or if specified name doesn't exist.
    """
    # Find all PoseEstimation containers using type-based search
    found = _find_containers_by_type(nwbfile, target_type)

    if not found:
        searched_locations = ["processing/*"]
        raise KeyError(
            f"No PoseEstimation data found in NWB file. Searched: {searched_locations}"
        )

    if pose_estimation_name is not None:
        # Look for specific name
        for path, container in found:
            if container.name == pose_estimation_name:
                logger.debug(
                    "Found PoseEstimation '%s' at %s", pose_estimation_name, path
                )
                return container
        # Not found - raise with available names
        available_names = [container.name for _, container in found]
        raise KeyError(
            f"PoseEstimation '{pose_estimation_name}' not found. "
            f"Available: {sorted(available_names)}"
        )

    # Auto-select: use first by priority (behavior module first, then alphabetically)
    path, container = found[0]
    if len(found) > 1:
        all_names = [c.name for _, c in found]
        logger.info(
            "Multiple PoseEstimation containers found: %s. Using '%s'",
            all_names,
            container.name,
        )
    else:
        logger.debug("Found PoseEstimation at %s", path)

    return container


def _get_timestamps(series: Any) -> NDArray[np.float64]:
    """
    Get timestamps from a PoseEstimationSeries.

    If explicit timestamps are not available, computes them from rate and starting_time.

    Parameters
    ----------
    series : PoseEstimationSeries
        The pose estimation series object.

    Returns
    -------
    NDArray[np.float64]
        Timestamps in seconds.
    """
    return timestamps_from_series(series)
