"""
Pose estimation reading from NWB ndx-pose containers.

This module provides functions for reading PoseEstimation data from
ndx-pose containers, including bodypart trajectories and skeleton definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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
    >>> from pynwb import NWBHDF5IO
    >>> with NWBHDF5IO("session.nwb", "r") as io:
    ...     nwbfile = io.read()
    ...     bodyparts, timestamps, skeleton = read_pose(nwbfile)
    ...     print(f"Found {len(bodyparts)} bodyparts")
    """
    raise NotImplementedError("read_pose not yet implemented")
