"""Animation overlay dataclasses and conversion pipeline.

This module provides the public API for adding dynamic overlays to spatial field
animations. Overlays include trajectories, pose tracking, head direction, and
regions of interest.

The module contains:
- Public dataclasses: PositionOverlay, BodypartOverlay, HeadDirectionOverlay
- Internal data containers: PositionData, BodypartData, HeadDirectionData
- Conversion pipeline: Aligns overlay data to animation frame times
- Validation functions: WHAT/WHY/HOW error messages for common issues
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Public API: User-facing overlay dataclasses
# =============================================================================


@dataclass
class PositionOverlay:
    """Single trajectory with optional trail visualization.

    Represents position data for a single entity (e.g., animal, object) over time.
    Can be rendered with a trail showing recent history. For multi-animal tracking,
    create multiple PositionOverlay instances with different colors.

    Parameters
    ----------
    data : NDArray[np.float64]
        Position coordinates with shape (n_samples, n_dims). Each row is a
        position at a time point. Dimensionality must match the environment.
    times : NDArray[np.float64] | None, optional
        Timestamps for each position sample with shape (n_samples,). If None,
        samples are assumed uniformly spaced. Must be monotonically increasing.
        Default is None.
    color : str, optional
        Color for the position marker and trail (matplotlib color string).
        Default is "red".
    size : float, optional
        Size of the position marker in points. Default is 10.0.
    trail_length : int | None, optional
        Number of recent frames to show as a trail. If None, no trail is rendered.
        Trail opacity decays over the length. Default is None.

    Attributes
    ----------
    data : NDArray[np.float64]
        Position coordinates.
    times : NDArray[np.float64] | None
        Optional timestamps.
    color : str
        Marker and trail color.
    size : float
        Marker size.
    trail_length : int | None
        Trail length in frames.

    See Also
    --------
    BodypartOverlay : Multi-keypoint pose tracking
    HeadDirectionOverlay : Directional heading visualization

    Notes
    -----
    For multi-animal tracking, create separate PositionOverlay instances:

    - Use distinct colors for each animal
    - Ensure all have consistent temporal coverage
    - Trail lengths can differ per animal

    Examples
    --------
    Basic trajectory without trail:

    >>> import numpy as np
    >>> positions = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    >>> overlay = PositionOverlay(data=positions)
    >>> overlay.color
    'red'
    >>> overlay.trail_length is None
    True

    Trajectory with timestamps and trail:

    >>> positions = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    >>> times = np.array([0.0, 0.5, 1.0])
    >>> overlay = PositionOverlay(
    ...     data=positions, times=times, color="blue", trail_length=10
    ... )
    >>> overlay.trail_length
    10

    Multi-animal tracking:

    >>> positions1 = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> positions2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> animal1 = PositionOverlay(data=positions1, color="red")
    >>> animal2 = PositionOverlay(data=positions2, color="blue")
    >>> overlays = [animal1, animal2]
    >>> len(overlays)
    2
    """

    data: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    color: str = "red"
    size: float = 10.0
    trail_length: int | None = None


@dataclass
class BodypartOverlay:
    """Multi-keypoint pose tracking with optional skeleton visualization.

    Represents pose data with multiple body parts (keypoints) and optional
    skeleton connections. Each body part is tracked independently over time.
    Skeletons are rendered as lines connecting specified keypoint pairs.

    Parameters
    ----------
    data : dict[str, NDArray[np.float64]]
        Dictionary mapping body part names to position arrays. Each array has
        shape (n_samples, n_dims). All body parts must have the same number of
        samples and dimensionality matching the environment.
    times : NDArray[np.float64] | None, optional
        Timestamps for each sample with shape (n_samples,). If None, samples
        are assumed uniformly spaced. Must be monotonically increasing.
        Default is None.
    skeleton : list[tuple[str, str]] | None, optional
        List of body part name pairs defining skeleton edges. Each tuple
        specifies (start_part, end_part). Part names must exist in `data`.
        If None, no skeleton is rendered. Default is None.
    colors : dict[str, str] | None, optional
        Dictionary mapping body part names to colors (matplotlib color strings).
        If None, all parts use default colors. Default is None.
    skeleton_color : str, optional
        Color for skeleton lines (matplotlib color string). Default is "white".
    skeleton_width : float, optional
        Width of skeleton lines in points. Default is 2.0.

    Attributes
    ----------
    data : dict[str, NDArray[np.float64]]
        Body part positions.
    times : NDArray[np.float64] | None
        Optional timestamps.
    skeleton : list[tuple[str, str]] | None
        Skeleton edge definitions.
    colors : dict[str, str] | None
        Per-part colors.
    skeleton_color : str
        Skeleton line color.
    skeleton_width : float
        Skeleton line width.

    See Also
    --------
    PositionOverlay : Single trajectory tracking
    HeadDirectionOverlay : Directional heading visualization

    Notes
    -----
    Skeleton validation occurs during conversion to internal representation:

    - All skeleton part names must exist in `data`
    - Invalid names trigger errors with suggestions
    - Missing data (NaN) breaks skeleton rendering at that frame

    For multi-animal pose tracking, create multiple BodypartOverlay instances.

    Examples
    --------
    Basic pose without skeleton:

    >>> import numpy as np
    >>> data = {
    ...     "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
    ...     "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
    ... }
    >>> overlay = BodypartOverlay(data=data)
    >>> "head" in overlay.data
    True
    >>> overlay.skeleton is None
    True

    Pose with skeleton connections:

    >>> data = {
    ...     "head": np.array([[0.0, 1.0]]),
    ...     "body": np.array([[1.0, 2.0]]),
    ...     "tail": np.array([[2.0, 3.0]]),
    ... }
    >>> skeleton = [("head", "body"), ("body", "tail")]
    >>> overlay = BodypartOverlay(data=data, skeleton=skeleton)
    >>> len(overlay.skeleton)
    2

    Custom colors per body part:

    >>> colors = {"head": "red", "body": "blue", "tail": "green"}
    >>> overlay = BodypartOverlay(
    ...     data=data, skeleton=skeleton, colors=colors, skeleton_color="white"
    ... )
    >>> overlay.colors["head"]
    'red'
    """

    data: dict[str, NDArray[np.float64]]
    times: NDArray[np.float64] | None = None
    skeleton: list[tuple[str, str]] | None = None
    colors: dict[str, str] | None = None
    skeleton_color: str = "white"
    skeleton_width: float = 2.0


@dataclass
class HeadDirectionOverlay:
    """Heading direction visualization rendered as arrows.

    Represents directional heading data as either angles (in radians) or unit
    vectors. Rendered as arrows at the corresponding position in the field.
    Commonly used for visualizing animal orientation during navigation.

    Parameters
    ----------
    data : NDArray[np.float64]
        Heading data in one of two formats:

        - Angles: shape (n_samples,) in radians, where 0 is right (east),
          π/2 is up (north), etc. (standard mathematical convention)
        - Unit vectors: shape (n_samples, n_dims) with direction vectors
          in environment coordinates

        Dimensionality must match the environment when using vectors.
    times : NDArray[np.float64] | None, optional
        Timestamps for each sample with shape (n_samples,). If None, samples
        are assumed uniformly spaced. Must be monotonically increasing.
        Default is None.
    color : str, optional
        Color for arrows (matplotlib color string). Default is "yellow".
    length : float, optional
        Arrow length in environment coordinate units. Default is 20.0.

    Attributes
    ----------
    data : NDArray[np.float64]
        Heading angles or unit vectors.
    times : NDArray[np.float64] | None
        Optional timestamps.
    color : str
        Arrow color.
    length : float
        Arrow length in environment units.

    See Also
    --------
    PositionOverlay : Single trajectory tracking
    BodypartOverlay : Multi-keypoint pose tracking

    Notes
    -----
    Angle convention (for 2D environments):

    - 0 radians: rightward (east, +x direction)
    - π/2 radians: upward (north, +y direction)
    - π radians: leftward (west, -x direction)
    - 3π/2 radians: downward (south, -y direction)

    For 3D environments, use unit vectors instead of angles.

    Examples
    --------
    Head direction as angles (2D):

    >>> import numpy as np
    >>> angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    >>> overlay = HeadDirectionOverlay(data=angles)
    >>> overlay.data.shape
    (4,)
    >>> overlay.color
    'yellow'

    Head direction as unit vectors (2D):

    >>> vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    >>> overlay = HeadDirectionOverlay(data=vectors, color="red")
    >>> overlay.data.shape
    (3, 2)

    Custom arrow styling:

    >>> overlay = HeadDirectionOverlay(data=angles, color="cyan", length=30.0)
    >>> overlay.length
    30.0

    With timestamps for temporal alignment:

    >>> times = np.array([0.0, 0.5, 1.0, 1.5])
    >>> overlay = HeadDirectionOverlay(data=angles, times=times)
    >>> overlay.times.shape
    (4,)
    """

    data: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    color: str = "yellow"
    length: float = 20.0


# =============================================================================
# Internal data model: Backend-facing containers (to be implemented)
# =============================================================================


@dataclass
class PositionData:
    """Internal container for position overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from PositionOverlay instances.

    Parameters
    ----------
    data : NDArray[np.float64]
        Position coordinates aligned to animation frames, shape (n_frames, n_dims).
    color : str
        Marker and trail color.
    size : float
        Marker size.
    trail_length : int | None
        Trail length in frames.

    Attributes
    ----------
    data : NDArray[np.float64]
        Position coordinates aligned to frames.
    color : str
        Marker and trail color.
    size : float
        Marker size.
    trail_length : int | None
        Trail length in frames.

    See Also
    --------
    PositionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    size: float
    trail_length: int | None


@dataclass
class BodypartData:
    """Internal container for bodypart overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from BodypartOverlay instances.

    Parameters
    ----------
    bodyparts : dict[str, NDArray[np.float64]]
        Body part positions aligned to frames, each shape (n_frames, n_dims).
    skeleton : list[tuple[str, str]] | None
        Skeleton edge definitions.
    colors : dict[str, str] | None
        Per-part colors.
    skeleton_color : str
        Skeleton line color.
    skeleton_width : float
        Skeleton line width.

    Attributes
    ----------
    bodyparts : dict[str, NDArray[np.float64]]
        Body part positions aligned to frames.
    skeleton : list[tuple[str, str]] | None
        Skeleton edge definitions.
    colors : dict[str, str] | None
        Per-part colors.
    skeleton_color : str
        Skeleton line color.
    skeleton_width : float
        Skeleton line width.

    See Also
    --------
    BodypartOverlay : User-facing overlay configuration
    """

    bodyparts: dict[str, NDArray[np.float64]]
    skeleton: list[tuple[str, str]] | None
    colors: dict[str, str] | None
    skeleton_color: str
    skeleton_width: float


@dataclass
class HeadDirectionData:
    """Internal container for head direction overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from HeadDirectionOverlay instances.

    Parameters
    ----------
    data : NDArray[np.float64]
        Heading data aligned to frames, shape (n_frames,) or (n_frames, n_dims).
    color : str
        Arrow color.
    length : float
        Arrow length in environment units.

    Attributes
    ----------
    data : NDArray[np.float64]
        Heading data aligned to frames.
    color : str
        Arrow color.
    length : float
        Arrow length in environment units.

    See Also
    --------
    HeadDirectionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    length: float


@dataclass
class OverlayData:
    """Container for all overlay data passed to animation backends.

    Aggregates all overlay types into a single pickle-safe container. Backends
    receive this object and render the appropriate overlay types.

    Parameters
    ----------
    positions : list[PositionData], optional
        List of position overlays. Default is empty list.
    bodypart_sets : list[BodypartData], optional
        List of bodypart overlays. Default is empty list.
    head_directions : list[HeadDirectionData], optional
        List of head direction overlays. Default is empty list.
    regions : list[str] | dict[int, list[str]] | None, optional
        Region names or per-frame region lists. When dict, keys are frame
        indices (0-based). Default is None.

    Attributes
    ----------
    positions : list[PositionData]
        List of position overlays.
    bodypart_sets : list[BodypartData]
        List of bodypart overlays.
    head_directions : list[HeadDirectionData]
        List of head direction overlays.
    regions : list[str] | dict[int, list[str]] | None
        Region names or per-frame region lists.

    Notes
    -----
    This container is created automatically by the conversion pipeline when
    `Environment.animate_fields()` is called with overlays. Users should not
    instantiate OverlayData directly.

    All data is aligned to animation frame times and validated for:

    - Pickle-ability (required for parallel video rendering)
    - Shape consistency with environment dimensions
    - Finite values (no NaN/Inf)

    Backends access this object to render overlays without needing to handle
    temporal alignment or validation.

    Raises
    ------
    ValueError
        If the OverlayData instance is not pickle-able (required for parallel
        video rendering). This will be validated during conversion pipeline
        implementation.
    """

    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    regions: list[str] | dict[int, list[str]] | None = None

    def __post_init__(self) -> None:
        """Placeholder for future pickle-ability validation.

        Notes
        -----
        Pickle-ability validation will be implemented in the validation module
        during conversion pipeline implementation (Milestone 1.4). This method is
        reserved for future validation logic to ensure parallel rendering
        compatibility.

        Once implemented, this will raise ValueError if any attribute contains
        unpickleable objects (lambdas, closures, etc.).
        """
        # Validation deferred to conversion pipeline implementation
        pass


# =============================================================================
# Timeline and interpolation helpers (private)
# =============================================================================


def _build_frame_times(
    frame_times: NDArray[np.float64] | None,
    fps: int | None,
    n_frames: int,
) -> NDArray[np.float64]:
    """Build or validate frame times array for animation.

    Creates frame times either from provided array or by synthesizing from fps.
    Validates that times are monotonically increasing.

    Parameters
    ----------
    frame_times : NDArray[np.float64] | None
        Explicit frame times array with shape (n_frames,). If provided, this
        takes precedence over fps. Must be monotonically increasing.
    fps : int | None
        Frames per second for synthesizing frame times. Only used if frame_times
        is None. Creates uniformly spaced times: [0, 1/fps, 2/fps, ...].
    n_frames : int
        Number of frames in the animation. Used to validate frame_times length
        or to synthesize times from fps.

    Returns
    -------
    NDArray[np.float64]
        Frame times array with shape (n_frames,), monotonically increasing.

    Raises
    ------
    ValueError
        If neither frame_times nor fps is provided.
    ValueError
        If frame_times length does not match n_frames.
    ValueError
        If frame_times is not monotonically increasing.

    Notes
    -----
    Frame times define the temporal sampling of the animation. They are used
    to align overlay data (which may have different sampling rates) to animation
    frames via interpolation.

    When both frame_times and fps are provided, frame_times takes precedence.

    Examples
    --------
    Synthesize from fps:

    >>> times = _build_frame_times(frame_times=None, fps=30, n_frames=90)
    >>> times.shape
    (90,)
    >>> times[0], times[1]
    (0.0, 0.03333333333333333)

    Use provided frame times:

    >>> custom_times = np.array([0.0, 0.5, 1.0, 1.5])
    >>> times = _build_frame_times(frame_times=custom_times, fps=None, n_frames=4)
    >>> times[0], times[-1]
    (0.0, 1.5)
    """
    if frame_times is not None:
        # Validate provided frame_times
        if len(frame_times) != n_frames:
            raise ValueError(
                f"frame_times length ({len(frame_times)}) does not match "
                f"n_frames ({n_frames})"
            )

        # Check monotonicity
        if not np.all(np.diff(frame_times) > 0):
            raise ValueError(
                "frame_times must be monotonically increasing. "
                "Found non-increasing values. "
                "HOW: Sort frame_times or check for duplicate values."
            )

        return frame_times

    if fps is not None:
        # Synthesize from fps
        return np.arange(n_frames, dtype=np.float64) / fps

    raise ValueError(
        "Either frame_times or fps must be provided. "
        "Got both as None. "
        "HOW: Provide explicit frame_times array or specify fps."
    )


def _interp_linear(
    t_src: NDArray[np.float64],
    x_src: NDArray[np.float64],
    t_frame: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized linear interpolation with NaN extrapolation.

    Interpolates source data at frame times using linear interpolation. Points
    outside the source time range are filled with NaN.

    Parameters
    ----------
    t_src : NDArray[np.float64]
        Source timestamps with shape (n_samples,). Must be monotonically increasing.
    x_src : NDArray[np.float64]
        Source data with shape (n_samples,) for 1D or (n_samples, n_dims) for N-D.
    t_frame : NDArray[np.float64]
        Frame times to interpolate at, shape (n_frames,).

    Returns
    -------
    NDArray[np.float64]
        Interpolated data at frame times. Shape matches input: (n_frames,) for 1D
        or (n_frames, n_dims) for N-D. Values outside [t_src.min(), t_src.max()]
        are NaN.

    Notes
    -----
    Uses numpy.interp for 1D data and applies along columns for N-D data.
    Extrapolation beyond source time range returns NaN to indicate missing data.

    NaN values in source data will propagate through interpolation.

    Examples
    --------
    1D interpolation:

    >>> t_src = np.array([0.0, 1.0, 2.0])
    >>> x_src = np.array([0.0, 10.0, 20.0])
    >>> t_frame = np.array([0.5, 1.5])
    >>> _interp_linear(t_src, x_src, t_frame)
    array([ 5., 15.])

    2D interpolation:

    >>> x_src_2d = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
    >>> _interp_linear(t_src, x_src_2d, t_frame)
    array([[ 5. ,  2.5],
           [15. ,  7.5]])

    Extrapolation returns NaN:

    >>> t_frame_extrap = np.array([-1.0, 0.5, 3.0])
    >>> result = _interp_linear(t_src, x_src, t_frame_extrap)
    >>> np.isnan(result[0]), result[1], np.isnan(result[2])
    (True, 5.0, True)
    """
    # Handle empty frame times
    if len(t_frame) == 0:
        return np.array([], dtype=np.float64)

    # Determine output shape
    if x_src.ndim == 1:
        output_shape: tuple[int, ...] = (len(t_frame),)
    else:
        output_shape = (len(t_frame), int(x_src.shape[1]))

    result = np.full(output_shape, np.nan, dtype=np.float64)

    # Find frame times within source range
    t_min, t_max = t_src.min(), t_src.max()
    valid_mask = (t_frame >= t_min) & (t_frame <= t_max)

    if not np.any(valid_mask):
        # No valid interpolation points
        return result

    # Interpolate valid points
    t_valid = t_frame[valid_mask]

    if x_src.ndim == 1:
        # 1D data
        result[valid_mask] = np.interp(t_valid, t_src, x_src)
    else:
        # N-D data - interpolate each dimension
        for dim in range(x_src.shape[1]):
            result[valid_mask, dim] = np.interp(t_valid, t_src, x_src[:, dim])

    return result


def _interp_nearest(
    t_src: NDArray[np.float64],
    x_src: NDArray[np.float64],
    t_frame: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized nearest neighbor interpolation with NaN extrapolation.

    Interpolates source data at frame times using nearest neighbor. Points
    outside the source time range are filled with NaN.

    Parameters
    ----------
    t_src : NDArray[np.float64]
        Source timestamps with shape (n_samples,). Must be monotonically increasing.
    x_src : NDArray[np.float64]
        Source data with shape (n_samples,) for 1D or (n_samples, n_dims) for N-D.
    t_frame : NDArray[np.float64]
        Frame times to interpolate at, shape (n_frames,).

    Returns
    -------
    NDArray[np.float64]
        Interpolated data at frame times. Shape matches input: (n_frames,) for 1D
        or (n_frames, n_dims) for N-D. Values outside [t_src.min(), t_src.max()]
        are NaN.

    Notes
    -----
    For each frame time, finds the closest source time and returns that value.
    Extrapolation beyond source time range returns NaN to indicate missing data.

    For frame times exactly at the midpoint between two source times, the
    behavior depends on numpy.searchsorted (typically picks the right neighbor).

    NaN values in source data will be returned when the nearest source point is NaN.

    Examples
    --------
    1D nearest neighbor:

    >>> t_src = np.array([0.0, 1.0, 2.0])
    >>> x_src = np.array([0.0, 10.0, 20.0])
    >>> t_frame = np.array([0.4, 1.4])
    >>> _interp_nearest(t_src, x_src, t_frame)
    array([ 0., 10.])

    2D nearest neighbor:

    >>> x_src_2d = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])
    >>> _interp_nearest(t_src, x_src_2d, t_frame)
    array([[ 0.,  0.],
           [10.,  5.]])

    Extrapolation returns NaN:

    >>> t_frame_extrap = np.array([-1.0, 0.4, 3.0])
    >>> result = _interp_nearest(t_src, x_src, t_frame_extrap)
    >>> np.isnan(result[0]), result[1], np.isnan(result[2])
    (True, 0.0, True)
    """
    # Handle empty frame times
    if len(t_frame) == 0:
        return np.array([], dtype=np.float64)

    # Determine output shape
    if x_src.ndim == 1:
        output_shape: tuple[int, ...] = (len(t_frame),)
    else:
        output_shape = (len(t_frame), int(x_src.shape[1]))

    result = np.full(output_shape, np.nan, dtype=np.float64)

    # Find frame times within source range
    t_min, t_max = t_src.min(), t_src.max()
    valid_mask = (t_frame >= t_min) & (t_frame <= t_max)

    if not np.any(valid_mask):
        # No valid interpolation points
        return result

    # For each valid frame time, find nearest source index
    t_valid = t_frame[valid_mask]

    # Compute distances to all source times
    # Shape: (n_valid_frames, n_src_samples)
    distances = np.abs(t_valid[:, np.newaxis] - t_src[np.newaxis, :])

    # Find index of minimum distance for each frame
    nearest_indices = np.argmin(distances, axis=1)

    # Index into source data
    if x_src.ndim == 1:
        result[valid_mask] = x_src[nearest_indices]
    else:
        result[valid_mask, :] = x_src[nearest_indices, :]

    return result
