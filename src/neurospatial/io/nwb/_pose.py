"""
Pose estimation reading from NWB ndx-pose containers.

This module provides functions for reading PoseEstimation data from
ndx-pose containers, including bodypart trajectories and skeleton definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._keyword_hint import keyword_hint
from neurospatial._units import same_coordinate_declaration
from neurospatial.io.nwb._adapters import (
    data_from_series,
    missing_error,
    position_frame_from_series,
    position_units_from_series,
    require_unscaled_for_lazy,
    select_one,
    timestamps_from_series,
    timestamps_handle_from_series,
    validate_handle_lengths,
)
from neurospatial.io.nwb._core import (
    _find_containers_by_type,
    _require_ndx_pose,
    logger,
)

if TYPE_CHECKING:
    from pynwb import NWBFile

    from neurospatial.animation.skeleton import Skeleton


@keyword_hint("pose_estimation_name")
def read_pose(
    nwbfile: NWBFile,
    *,
    pose_estimation_name: str | None = None,
    lazy: bool = False,
) -> tuple[dict[str, NDArray[np.float64] | Any], NDArray[np.float64] | Any, Skeleton]:
    """
    Read pose estimation data from NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        The NWB file to read from.
    pose_estimation_name : str, optional
        Name of the PoseEstimation container, or its full path such as
        ``"processing/behavior/PoseEstimation"``. Required when the file
        holds more than one PoseEstimation container.
    lazy : bool, default False
        If ``False`` (default), each bodypart trajectory and the timestamps are
        fully materialized into ``NDArray[np.float64]``. If ``True``, the
        h5py-backed ``PoseEstimationSeries.data`` handles (and the timestamps
        handle) are returned **without** copying, so they materialize only when
        sliced or ``np.asarray``-ed. Values are in each series' ``unit``:
        stored × ``conversion`` + ``offset``; ``lazy=True`` raises when that map
        is not the identity.

    Returns
    -------
    bodyparts : dict[str, NDArray[np.float64] or h5py.Dataset]
        Mapping from bodypart name to coordinates, each shape
        (n_samples, n_dims), in the series' ``unit``: stored × ``conversion``
        + ``offset``. Values are materialized arrays when ``lazy=False`` and
        lazy handles when ``lazy=True``.
    timestamps : NDArray[np.float64] or h5py.Dataset, shape (n_samples,)
        Timestamps in seconds (shared across all bodyparts). A lazy handle when
        ``lazy=True`` and the series carries explicit timestamps.
    skeleton : Skeleton
        Skeleton definition from the PoseEstimation container.

    Raises
    ------
    KeyError
        If no PoseEstimation container found, or if specified name not found.
    ValueError
        If more than one PoseEstimation container matches, ``lazy=True`` and
        a series declares a non-identity ``conversion`` or ``offset``,
        bodypart timestamps differ, or bodyparts declare
        incompatible units or reference frames.
    ImportError
        If ndx-pose is not installed.

    Notes
    -----
    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Slice or ``np.asarray`` them inside the ``with NWBHDF5IO(...)``
    block. ``lazy=False`` returns fully-materialized arrays as before.

    All bodyparts must have the same timestamps on both paths (within float64
    rounding precision); no resampling is performed. Lazy reads inspect
    timestamps in bounded chunks to check alignment while preserving the
    original coordinate and timestamp handles.
    Units and reference frames must also agree across bodyparts, including
    whether each declaration is present. Unit aliases such as ``"meters"``
    and ``"m"`` are equivalent; coordinates are
    never rescaled between units. Reference-frame descriptions are compared
    verbatim, without interpreting their meaning.

    ``lazy=True`` returns the **stored** dtype (whatever each
    ``PoseEstimationSeries`` was written with, e.g. ``float32``), whereas
    ``lazy=False`` casts to ``float64``. The values are equal; only the dtype may
    differ.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     nwbfile = io.read()
    ...     bodyparts, timestamps, skeleton = read_pose(nwbfile)
    ...     print(f"Found {len(bodyparts)} bodyparts")
    """
    bodyparts, timestamps, skeleton, _, _ = _read_pose_with_metadata(
        nwbfile, pose_estimation_name=pose_estimation_name, lazy=lazy
    )
    return bodyparts, timestamps, skeleton


def _read_pose_with_metadata(
    nwbfile: NWBFile,
    *,
    pose_estimation_name: str | None = None,
    lazy: bool = False,
) -> tuple[dict[str, Any], Any, Skeleton, str | None, str | None]:
    """Extract a shared clock and coordinate declarations in one traversal."""
    _require_ndx_pose()
    from ndx_pose import PoseEstimation as PoseEstimationType

    from neurospatial.animation.skeleton import Skeleton as NSSkeleton

    # Find PoseEstimation container
    pose_estimation = _get_pose_estimation_container(
        nwbfile, PoseEstimationType, pose_estimation_name
    )

    # Extract bodyparts as dict
    bodyparts: dict[str, NDArray[np.float64] | Any] = {}
    timestamps: NDArray[np.float64] | Any | None = None
    units: str | None = None
    frame: str | None = None
    clock_name: str | None = None

    for series_name in sorted(pose_estimation.pose_estimation_series.keys()):
        series = pose_estimation.pose_estimation_series[series_name]
        if lazy:
            require_unscaled_for_lazy(series, context="read_pose")
            # h5py-backed handle without copy; materializes on slice.
            bodyparts[series_name] = series.data
            series_times = timestamps_handle_from_series(series)
        else:
            bodyparts[series_name] = data_from_series(series)
            series_times = _get_timestamps(series)

        if timestamps is None:
            timestamps = series_times
            clock_name = series_name
        validate_handle_lengths(
            {
                f"bodypart[{series_name}]": len(series.data),
                "timestamps": len(timestamps),
                f"timestamps[{series_name}]": len(series_times),
            }
        )
        # Only timestamp slices are materialized; coordinate handles stay lazy.
        previous_time: float | None = None
        for start in range(0, len(timestamps), 65536):
            stop = start + 65536
            clock_chunk = np.asarray(series_times[start:stop], dtype=np.float64)
            if (
                not np.all(np.isfinite(clock_chunk))
                or np.any(np.diff(clock_chunk) <= 0)
                or (previous_time is not None and clock_chunk[0] <= previous_time)
            ):
                raise ValueError(
                    _format_error(
                        f"Pose bodypart {series_name!r} timestamps must be finite "
                        "and strictly increasing.",
                        fix="correct the source time axis before reading.",
                    )
                )
            previous_time = float(clock_chunk[-1])
            # Allow only floating-point roundoff, including clocks computed by
            # equivalent explicit and rate-based expressions. A tolerance tied
            # to sampling intervals would hide actual acquisition-clock shifts.
            precision = 4 * np.finfo(np.float64).eps
            if not np.allclose(
                timestamps[start:stop],
                clock_chunk,
                rtol=precision,
                atol=precision,
            ):
                raise ValueError(
                    _format_error(
                        f"Pose bodypart {series_name!r} timestamps do not match "
                        f"bodypart {clock_name!r}.",
                        why="Why: all bodyparts must share identical timestamps.",
                        fix="explicitly align the bodypart series to a common "
                        "clock before reading; no resampling is performed.",
                    )
                )

        series_units = position_units_from_series(series)
        series_frame = position_frame_from_series(series)
        for label, declared, shared in (
            ("units", series_units, units),
            ("reference_frame", series_frame, frame),
        ):
            if series_name != clock_name and not same_coordinate_declaration(
                label, declared, shared, undeclared_matches=False
            ):
                raise ValueError(
                    _format_error(
                        f"Pose bodypart {series_name!r} {label} {declared!r} does "
                        f"not match the other bodyparts' {label} {shared!r}.",
                        fix="explicitly declare common units and reference frame "
                        "for every bodypart, converting coordinates where needed "
                        "before reading; coordinates are never silently rescaled.",
                    )
                )
        if series_units is not None:
            units = series_units
        if series_frame is not None:
            frame = series_frame

    if timestamps is None:
        raise ValueError("PoseEstimation container has no pose estimation series")

    # Convert ndx-pose Skeleton to neurospatial Skeleton
    skeleton = NSSkeleton.from_ndx_pose(pose_estimation.skeleton)

    return bodyparts, timestamps, skeleton, units, frame


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
        Container name or full path. Required when there is more than one.

    Returns
    -------
    PoseEstimation
        The PoseEstimation container.

    Raises
    ------
    KeyError
        If no container found or if specified name doesn't exist.
    ValueError
        If more than one container matches.
    """
    found = _find_containers_by_type(nwbfile, target_type)
    if not found:
        raise missing_error(
            "No PoseEstimation data found in NWB file. Searched: "
            "processing/* and acquisition.",
            fix="add an ndx-pose PoseEstimation container to processing/behavior.",
        )
    container = select_one(
        found,
        pose_estimation_name,
        what="PoseEstimation container",
        keyword="pose_estimation_name",
    )
    logger.debug("Reading PoseEstimation '%s'", container.name)
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
