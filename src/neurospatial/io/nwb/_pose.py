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
from neurospatial.io.nwb._holders import NWBPose

if TYPE_CHECKING:
    from pynwb import NWBFile


@keyword_hint("pose_estimation_name")
def read_pose(
    nwbfile: NWBFile,
    *,
    pose_estimation_name: str | None = None,
    lazy: bool = False,
) -> NWBPose:
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
        If ``False`` (default), each bodypart trajectory, confidence and the
        timestamps are fully materialized into read-only
        ``NDArray[np.float64]``. If ``True``, the h5py-backed
        ``PoseEstimationSeries.data`` and ``confidence`` handles (and the
        timestamps handle) are returned **without** copying, so they
        materialize only when sliced or ``np.asarray``-ed. Values are in each
        series' ``unit``: stored × ``conversion`` + ``offset``; ``lazy=True``
        raises when that map is not the identity.

    Returns
    -------
    NWBPose
        Frozen holder with ``bodyparts`` (series name to coordinates, shape
        (n_samples, n_dims)), shared ``times`` in seconds, ``skeleton`` (None
        when the container links none), the shared ``units`` and ``frame``
        declarations, and per-bodypart ``confidence`` (None when no series
        stores one). Untracked keypoints stay NaN.

    Raises
    ------
    KeyError
        If no PoseEstimation container found, or if specified name not found.
    ValueError
        If more than one PoseEstimation container matches; the container has
        no series; a bodypart's timestamps are not finite and strictly
        increasing, differ in length from or do not match the other
        bodyparts' timestamps; a bodypart's data or confidence does not have
        one row per timestamp; bodyparts declare incompatible units or
        reference frames; or ``lazy=True`` and a series declares a
        non-identity ``conversion`` or ``offset``.
    ImportError
        If ndx-pose is not installed.

    Notes
    -----
    Lazy handles are **only valid while the backing ``NWBFile`` / ``NWBHDF5IO``
    is open**. Slice or ``np.asarray`` them inside the ``with NWBHDF5IO(...)``
    block. ``lazy=False`` returns arrays that remain valid after the file
    closes.

    All bodyparts must have the same timestamps on both paths (within float64
    rounding precision); no resampling is performed. Lazy reads inspect
    timestamps in bounded chunks to check alignment while preserving the
    original coordinate and timestamp handles.
    Units and reference frames must also agree across bodyparts, including
    whether each declaration is present. Unit aliases such as ``"meters"``
    and ``"m"`` are equivalent; coordinates are
    never rescaled between units. Reference-frame descriptions are compared
    verbatim, without interpreting their meaning.

    Keypoints are not masked by ``confidence``: no threshold suits every
    tracker. Choose one and set low-confidence samples to NaN yourself.

    ``bodyparts`` keys are the ``PoseEstimationSeries`` names. ndx-pose
    expects these to equal the skeleton's node names but does not link them;
    files whose series and node names differ are read unchanged.

    ``lazy=True`` returns the **stored** dtype (whatever each
    ``PoseEstimationSeries`` was written with, e.g. ``float32``), whereas
    ``lazy=False`` casts to ``float64``. The values are equal; only the dtype may
    differ.

    Examples
    --------
    >>> from pynwb import NWBHDF5IO  # doctest: +SKIP
    >>> with NWBHDF5IO("session.nwb", "r") as io:  # doctest: +SKIP
    ...     pose = read_pose(io.read())
    >>> sorted(pose.bodyparts), pose.units  # doctest: +SKIP
    (['body', 'nose', 'tail'], 'cm')
    """
    _require_ndx_pose()
    from ndx_pose import PoseEstimation as PoseEstimationType

    from neurospatial.animation.skeleton import Skeleton

    pose_estimation = _get_pose_estimation_container(
        nwbfile, PoseEstimationType, pose_estimation_name
    )

    bodyparts: dict[str, NDArray[np.float64] | Any] = {}
    confidence: dict[str, NDArray[np.float64] | Any] = {}
    timestamps: NDArray[np.float64] | Any | None = None
    units: str | None = None
    frame: str | None = None
    clock_name: str | None = None

    for series_name in sorted(pose_estimation.pose_estimation_series.keys()):
        series = pose_estimation.pose_estimation_series[series_name]
        if lazy:
            require_unscaled_for_lazy(series, context="read_pose")
            # h5py-backed handles without copy; they materialize on slice.
            bodyparts[series_name] = series.data
            series_times = timestamps_handle_from_series(series)
        else:
            bodyparts[series_name] = data_from_series(series)
            series_times = timestamps_from_series(series)
        if series.confidence is not None:
            confidence[series_name] = (
                series.confidence
                if lazy
                else np.asarray(series.confidence[:], dtype=np.float64)
            )

        if timestamps is None:
            timestamps = series_times
            clock_name = series_name
        validate_handle_lengths(
            {
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
        raise ValueError(
            _format_error(
                f"PoseEstimation '{pose_estimation.name}' has no PoseEstimationSeries.",
                fix="add one PoseEstimationSeries per tracked bodypart.",
            )
        )

    ndx_skeleton = pose_estimation.skeleton
    return NWBPose(
        bodyparts,
        timestamps,
        None if ndx_skeleton is None else Skeleton.from_ndx_pose(ndx_skeleton),
        units=units,
        frame=frame,
        confidence=confidence or None,
    )


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
