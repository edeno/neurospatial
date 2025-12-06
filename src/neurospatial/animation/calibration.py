"""Video calibration convenience functions.

This module provides high-level functions for calibrating video coordinates
to environment space. It wraps lower-level calibration functions from
neurospatial.transforms with additional validation and bounds checking.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.ops.transforms import (
    VideoCalibration,
    calibrate_from_landmarks,
    scale_2d,
)
from neurospatial.ops.transforms import (
    flip_y as flip_y_transform,
)

if TYPE_CHECKING:
    from neurospatial import Environment


def calibrate_video(
    video_path: str | Path,
    env: Environment,
    *,
    scale_bar: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    landmarks_px: NDArray[np.float64] | None = None,
    landmarks_env: NDArray[np.float64] | None = None,
    cm_per_px: float | None = None,
    flip_y: bool = True,
) -> VideoCalibration:
    """Calibrate video coordinates to environment space.

    Creates a VideoCalibration object that maps video pixel coordinates to
    environment centimeter coordinates. Supports three calibration methods:
    scale bar, landmark correspondences, or direct scale factor.

    Parameters
    ----------
    video_path : str or Path
        Path to video file for extracting frame size.
    env : Environment
        Environment to calibrate against (used for bounds validation).
    scale_bar : tuple, optional
        Scale bar calibration as ((x1, y1), (x2, y2), length_cm).
        The two points define the endpoints of a known-length bar in pixels.
    landmarks_px : ndarray of shape (n_points, 2), optional
        Landmark coordinates in video pixels as (x, y) = (column, row).
        Must be provided together with landmarks_env.
    landmarks_env : ndarray of shape (n_points, 2), optional
        Corresponding landmark coordinates in environment space as (x, y) in cm.
        Must be provided together with landmarks_px.
    cm_per_px : float, optional
        Direct scale factor if known (cm per pixel).
    flip_y : bool, default=True
        Whether to flip the Y-axis when transforming coordinates.

        - ``True`` (default): Environment uses scientific/Cartesian coordinates
          (Y increases upward). This is the common case for tracking data in
          cm or meters from DeepLabCut, SLEAP, etc.
        - ``False``: Environment uses image/pixel coordinates (Y increases
          downward). Use this when your environment is already in pixel space.

        This parameter only applies to ``cm_per_px`` and ``scale_bar`` methods.
        For ``landmarks``, the Y-flip is implicit in the correspondences.

        If ``env.units`` is set, the default behavior is inferred:
        - ``env.units == "pixels"``: Assumes ``flip_y=False`` (warns if True)
        - Otherwise: Assumes ``flip_y=True`` (scientific convention)

    Returns
    -------
    VideoCalibration
        Calibration object containing the pixel→cm transform and frame metadata.
        Can be passed to VideoOverlay for spatial alignment.

    Raises
    ------
    ValueError
        If no calibration method is specified, multiple methods are specified,
        or landmark arrays have mismatched lengths.
    FileNotFoundError
        If video_path does not exist.

    Warns
    -----
    UserWarning
        If environment bounds extend beyond calibrated video coverage, or if
        ``env.units='pixels'`` but ``cm_per_px`` is provided (unusual combination).

    Notes
    -----
    **Coordinate Conventions**

    Video coordinates use image convention: origin at top-left, Y increases
    downward. Scientific data typically uses Cartesian convention: origin at
    bottom-left, Y increases upward.

    The ``flip_y`` parameter controls whether to flip the Y-axis:

    - ``flip_y=True``: Converts video Y-down to environment Y-up
    - ``flip_y=False``: No Y-flip (both use same convention)

    The video overlay itself serves as visual validation - if the overlay
    appears inverted, toggle ``flip_y``.

    Examples
    --------
    >>> from neurospatial.animation import calibrate_video, VideoOverlay
    >>>
    >>> # Using scale bar method (scientific coordinates, Y-up)
    >>> calibration = calibrate_video(
    ...     "session.mp4",
    ...     env,
    ...     scale_bar=((100, 200), (300, 200), 50.0),  # 200px = 50cm
    ... )
    >>>
    >>> # Using landmark correspondences (Y-flip implicit in correspondences)
    >>> corners_px = np.array([[50, 50], [590, 50], [590, 430], [50, 430]])
    >>> corners_env = np.array([[0, 0], [100, 0], [100, 80], [0, 80]])
    >>> calibration = calibrate_video(
    ...     "session.mp4",
    ...     env,
    ...     landmarks_px=corners_px,
    ...     landmarks_env=corners_env,
    ... )
    >>>
    >>> # Using direct scale factor (most common)
    >>> calibration = calibrate_video(
    ...     "session.mp4",
    ...     env,
    ...     cm_per_px=0.25,
    ... )
    >>>
    >>> # If overlay appears inverted, toggle flip_y
    >>> calibration = calibrate_video(
    ...     "session.mp4",
    ...     env,
    ...     cm_per_px=0.25,
    ...     flip_y=False,  # For pixel-space environments
    ... )
    >>>
    >>> # Use calibration with VideoOverlay
    >>> video = VideoOverlay(source="session.mp4", calibration=calibration)
    >>> env.animate_fields(fields, overlays=[video], backend="napari")
    """
    # Validate video path exists
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(
            f"WHAT: Video file not found at '{video_path}'.\n"
            f"WHY: The video_path must point to an existing file.\n"
            f"HOW: Check the file path and ensure the video file exists."
        )

    # Get frame size from video
    frame_size_px = _get_video_frame_size(video_path)

    # Validate calibration method specification
    methods_provided = []
    if scale_bar is not None:
        methods_provided.append("scale_bar")
    if landmarks_px is not None or landmarks_env is not None:
        methods_provided.append("landmarks")
    if cm_per_px is not None:
        methods_provided.append("cm_per_px")

    if len(methods_provided) == 0:
        raise ValueError(
            "WHAT: No calibration method specified.\n"
            "WHY: calibrate_video() requires one of: scale_bar, landmarks, or cm_per_px.\n"
            "HOW: Provide exactly one calibration parameter, e.g.:\n"
            "     - scale_bar=((x1, y1), (x2, y2), length_cm)\n"
            "     - landmarks_px=array, landmarks_env=array\n"
            "     - cm_per_px=0.25"
        )

    if len(methods_provided) > 1:
        raise ValueError(
            f"WHAT: Multiple calibration methods specified: {methods_provided}.\n"
            f"WHY: scale_bar, landmarks, and cm_per_px are mutually exclusive.\n"
            f"HOW: Provide exactly one calibration method."
        )

    # Check for unusual env.units='pixels' + cm_per_px combination
    env_units = getattr(env, "units", None)
    if env_units == "pixels" and cm_per_px is not None:
        warnings.warn(
            f"WHAT: env.units='pixels' but cm_per_px={cm_per_px} provided.\n"
            f"WHY: This is an unusual combination. If your environment uses pixel "
            f"coordinates, you typically don't need a scale factor.\n"
            f"HOW: If converting from pixels to another unit, set env.units to that "
            f"unit (e.g., 'cm'). Using flip_y=False since pixel coordinates typically "
            f"use image convention (Y-down).",
            UserWarning,
            stacklevel=2,
        )

    # Build transform based on method
    if scale_bar is not None:
        (p1_px, p2_px, known_length_cm) = scale_bar

        # Validate inputs
        if known_length_cm <= 0:
            raise ValueError(
                f"WHAT: known_length_cm must be positive (got {known_length_cm}).\n"
                f"WHY: A scale bar must have positive real-world length.\n"
                f"HOW: Provide a positive value for known_length_cm."
            )

        # Compute pixel distance between endpoints
        dx = p2_px[0] - p1_px[0]
        dy = p2_px[1] - p1_px[1]
        px_distance = np.sqrt(dx * dx + dy * dy)

        if px_distance == 0:
            raise ValueError(
                f"WHAT: Scale bar has zero pixel length (p1={p1_px}, p2={p2_px}).\n"
                f"WHY: Cannot compute scale from coincident endpoints.\n"
                f"HOW: Provide two distinct points for the scale bar."
            )

        # Compute scale factor
        cm_per_px_computed = known_length_cm / px_distance
        _, frame_height = frame_size_px

        # Build transform with conditional Y-flip
        if flip_y:
            transform = scale_2d(cm_per_px_computed) @ flip_y_transform(frame_height)
        else:
            transform = scale_2d(cm_per_px_computed)

    elif landmarks_px is not None or landmarks_env is not None:
        # Validate both landmarks are provided
        if landmarks_px is None:
            raise ValueError(
                "WHAT: landmarks_px is required when landmarks_env is provided.\n"
                "WHY: Landmark calibration needs corresponding pixel and environment points.\n"
                "HOW: Provide landmarks_px array of shape (n_points, 2)."
            )
        if landmarks_env is None:
            raise ValueError(
                "WHAT: landmarks_env is required when landmarks_px is provided.\n"
                "WHY: Landmark calibration needs corresponding pixel and environment points.\n"
                "HOW: Provide landmarks_env array of shape (n_points, 2)."
            )

        # Ensure arrays
        landmarks_px = np.asarray(landmarks_px, dtype=np.float64)
        landmarks_env = np.asarray(landmarks_env, dtype=np.float64)

        # Validate same length
        if len(landmarks_px) != len(landmarks_env):
            raise ValueError(
                f"WHAT: landmarks_px has {len(landmarks_px)} points but landmarks_env "
                f"has {len(landmarks_env)} points.\n"
                f"WHY: Each pixel landmark must have a corresponding environment point.\n"
                f"HOW: Ensure landmarks_px and landmarks_env have the same number of points."
            )

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_env,
            frame_size_px=frame_size_px,
            kind="similarity",  # Default to similarity transform
        )

    else:  # cm_per_px is not None
        # Direct scale factor with conditional Y-flip
        assert cm_per_px is not None  # Asserted by method validation above
        if cm_per_px <= 0:
            raise ValueError(
                f"WHAT: cm_per_px must be positive (got {cm_per_px}).\n"
                f"WHY: Scale factor must be positive to produce valid transforms.\n"
                f"HOW: Provide a positive value, e.g., cm_per_px=0.25."
            )
        _, frame_height = frame_size_px

        # Build transform with conditional Y-flip
        if flip_y:
            transform = scale_2d(cm_per_px) @ flip_y_transform(frame_height)
        else:
            transform = scale_2d(cm_per_px)

    # Create calibration object
    calibration = VideoCalibration(
        transform_px_to_cm=transform,
        frame_size_px=frame_size_px,
    )

    # Validate bounds coverage
    _validate_calibration_coverage(calibration, env)

    return calibration


def _get_video_frame_size(video_path: Path) -> tuple[int, int]:
    """Extract frame size from video file.

    Parameters
    ----------
    video_path : Path
        Path to video file.

    Returns
    -------
    tuple[int, int]
        Frame size as (width, height) in pixels.

    Raises
    ------
    ValueError
        If video cannot be opened or frame size cannot be determined.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid frame size ({width}, {height}) from {video_path}"
            )

        return (width, height)

    except ImportError:
        # Fallback to imageio if cv2 not available
        try:
            import imageio.v3 as iio

            meta = iio.immeta(video_path)
            # imageio returns size differently depending on backend
            if "size" in meta:
                return tuple(meta["size"])
            elif "shape" in meta:
                # Shape is (height, width, channels) or (frames, height, width, channels)
                shape = meta["shape"]
                if len(shape) == 3:
                    return (shape[1], shape[0])
                elif len(shape) == 4:
                    return (shape[2], shape[1])
            raise ValueError(f"Cannot determine frame size from {video_path}")
        except ImportError as exc:
            raise ImportError(
                "Neither cv2 (opencv-python) nor imageio is available. "
                "Install one of them to read video files."
            ) from exc


def _validate_calibration_coverage(
    calibration: VideoCalibration,
    env: Environment,
) -> None:
    """Warn if environment bounds exceed calibrated video coverage.

    Parameters
    ----------
    calibration : VideoCalibration
        The calibration to validate.
    env : Environment
        Environment to check bounds against.

    Warns
    -----
    UserWarning
        If environment bounds extend beyond calibrated video coverage.
    """
    # Transform video corners to environment coordinates
    w, h = calibration.frame_size_px
    corners_px = np.array(
        [
            [0.0, 0.0],
            [float(w), 0.0],
            [float(w), float(h)],
            [0.0, float(h)],
        ]
    )
    corners_cm = calibration.transform_px_to_cm(corners_px)

    # Compute video coverage bounds
    video_x_min, video_x_max = corners_cm[:, 0].min(), corners_cm[:, 0].max()
    video_y_min, video_y_max = corners_cm[:, 1].min(), corners_cm[:, 1].max()

    # Get environment bounds
    if env.dimension_ranges is None:
        return  # Cannot validate without dimension_ranges
    (env_x_min, env_x_max), (env_y_min, env_y_max) = env.dimension_ranges

    # Check if environment extends beyond video
    exceeds = (
        env_x_min < video_x_min - 1e-6
        or env_x_max > video_x_max + 1e-6
        or env_y_min < video_y_min - 1e-6
        or env_y_max > video_y_max + 1e-6
    )

    if exceeds:
        warnings.warn(
            f"WHAT: Environment bounds extend beyond calibrated video coverage.\n"
            f"WHY: Environment range x=[{env_x_min:.1f}, {env_x_max:.1f}], "
            f"y=[{env_y_min:.1f}, {env_y_max:.1f}] exceeds "
            f"video range x=[{video_x_min:.1f}, {video_x_max:.1f}], "
            f"y=[{video_y_min:.1f}, {video_y_max:.1f}].\n"
            f"HOW: Regions outside video coverage will appear blank during rendering. "
            f"Consider adjusting calibration or cropping the environment.",
            UserWarning,
            stacklevel=3,  # Point to caller of calibrate_video
        )
