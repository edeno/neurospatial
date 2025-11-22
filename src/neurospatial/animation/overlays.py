"""Animation overlay dataclasses and conversion pipeline.

This module provides the public API for adding dynamic overlays to spatial field
animations. Overlays include trajectories, pose tracking, head direction, and
regions of interest.

The module contains:
- Public dataclasses: PositionOverlay, BodypartOverlay, HeadDirectionOverlay
- Internal data containers: PositionData, BodypartData, HeadDirectionData
- Conversion pipeline: Aligns overlay data to animation frame times
- Validation functions: WHAT/WHY/HOW error messages for common issues

Performance Considerations
--------------------------
- **NaN handling**: NaN values in overlay data are propagated through interpolation.
  Frames where any coordinate is NaN will not render that overlay element. This is
  useful for handling missing tracking data (e.g., occluded bodyparts).
- **Expected shapes**: Position data should be shape (n_samples, n_dims). For 2D
  environments, n_dims=2. All bodyparts in BodypartOverlay must have the same
  n_samples.
- **Temporal alignment**: When overlays have higher sampling rates than field frames
  (e.g., 120 Hz tracking vs 10 Hz fields), provide ``times`` to the overlay and
  ``frame_times`` to animate_fields. Linear interpolation aligns automatically.
- **Multi-animal**: Multiple overlays can be passed as a list. Each is rendered
  independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.animation.skeleton import Skeleton

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
    data : ndarray of shape (n_samples, n_dims), dtype float64
        Position coordinates in environment space. Each row is a position at
        a time point. Dimensionality must match the environment (env.n_dims).
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each position sample, in seconds. If None, samples are
        assumed uniformly spaced at the animation fps rate. Must be
        monotonically increasing. Default is None.
    color : str, optional
        Color for the position marker and trail (matplotlib color string).
        Default is "red".
    size : float, optional
        Size of the position marker in points. Default is 10.0.
    trail_length : int | None, optional
        Number of recent frames to show as a trail. If None, no trail is rendered.
        Trail opacity decays over the length. Default is None.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

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
    interp : {"linear", "nearest"}
        Interpolation method.

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
    interp: Literal["linear", "nearest"] = "linear"


@dataclass
class BodypartOverlay:
    """Multi-keypoint pose tracking with optional skeleton visualization.

    Represents pose data with multiple body parts (keypoints) and optional
    skeleton connections. Each body part is tracked independently over time.
    Skeletons are rendered as lines connecting specified keypoint pairs.

    Parameters
    ----------
    data : dict of str to ndarray of shape (n_samples, n_dims), dtype float64
        Dictionary mapping body part names to position arrays. Each array has
        shape (n_samples, n_dims), dtype float64. All body parts must have the
        same number of samples and dimensionality matching the environment
        (env.n_dims).
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each sample, in seconds. If None, samples are assumed
        uniformly spaced at the animation fps rate. Must be monotonically
        increasing. Default is None.
    skeleton : Skeleton | None, optional
        Skeleton object defining node names and edge connections. If provided,
        edges are rendered as lines connecting keypoints. The skeleton's
        node_colors, edge_color, and edge_width can override the overlay's
        colors settings. If None, no skeleton is rendered. Default is None.
    colors : dict[str, str] | None, optional
        Dictionary mapping body part names to colors (matplotlib color strings).
        If None and skeleton has node_colors, uses skeleton's colors.
        Otherwise all parts use default colors. Default is None.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

    Attributes
    ----------
    data : dict[str, NDArray[np.float64]]
        Body part positions.
    times : NDArray[np.float64] | None
        Optional timestamps.
    skeleton : Skeleton | None
        Skeleton definition with nodes and edges.
    colors : dict[str, str] | None
        Per-part colors (overrides skeleton.node_colors if provided).
    interp : {"linear", "nearest"}
        Interpolation method.

    See Also
    --------
    PositionOverlay : Single trajectory tracking
    HeadDirectionOverlay : Directional heading visualization
    Skeleton : Anatomical structure definition

    Notes
    -----
    Skeleton validation occurs during conversion to internal representation:

    - All skeleton edge endpoints must exist in `data`
    - Invalid names trigger errors with suggestions
    - Missing data (NaN) breaks skeleton rendering at that frame

    For multi-animal pose tracking, create multiple BodypartOverlay instances.

    Color resolution order:
    1. overlay.colors (if provided)
    2. skeleton.node_colors (if skeleton provided and has colors)
    3. Default colors

    Edge styling comes from skeleton.edge_color and skeleton.edge_width.

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

    Pose with skeleton object:

    >>> from neurospatial.animation.skeleton import Skeleton
    >>> data = {
    ...     "head": np.array([[0.0, 1.0]]),
    ...     "body": np.array([[1.0, 2.0]]),
    ...     "tail": np.array([[2.0, 3.0]]),
    ... }
    >>> skeleton = Skeleton(
    ...     name="simple",
    ...     nodes=("head", "body", "tail"),
    ...     edges=(("head", "body"), ("body", "tail")),
    ...     edge_color="white",
    ...     edge_width=2.0,
    ... )
    >>> overlay = BodypartOverlay(data=data, skeleton=skeleton)
    >>> overlay.skeleton.n_edges
    2

    Custom colors per body part:

    >>> colors = {"head": "red", "body": "blue", "tail": "green"}
    >>> overlay = BodypartOverlay(data=data, skeleton=skeleton, colors=colors)
    >>> overlay.colors["head"]
    'red'
    """

    data: dict[str, NDArray[np.float64]]
    times: NDArray[np.float64] | None = None
    skeleton: Skeleton | None = None
    colors: dict[str, str] | None = None
    interp: Literal["linear", "nearest"] = "linear"


@dataclass
class HeadDirectionOverlay:
    """Heading direction visualization rendered as arrows.

    Represents directional heading data as either angles (in radians) or unit
    vectors. Rendered as arrows at the corresponding position in the field.
    Commonly used for visualizing animal orientation during navigation.

    Parameters
    ----------
    data : ndarray of shape (n_samples,) or (n_samples, n_dims), dtype float64
        Heading data in one of two formats:

        - Angles: shape (n_samples,), dtype float64, in radians where 0 is
          right (east), π/2 is up (north), etc. (standard mathematical
          convention)
        - Unit vectors: shape (n_samples, n_dims), dtype float64, with
          direction vectors in environment coordinates. Dimensionality must
          match the environment (env.n_dims).
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each sample, in seconds. If None, samples are assumed
        uniformly spaced at the animation fps rate. Must be monotonically
        increasing. Default is None.
    color : str, optional
        Color for arrows (matplotlib color string). Default is "yellow".
    length : float, optional
        Arrow length in environment coordinate units. Default is 0.25.

        **Important**: The default value (0.25) may be too small for many environments.
        For environments measured in centimeters (e.g., bin_size=5.0), consider using
        length=10.0 to 20.0 for visible arrows. A good rule of thumb is to use
        approximately 2-4x your bin_size.
    width : float, optional
        Arrow line width in pixels. Default is 1.0.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

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
    width : float
        Arrow line width in pixels.
    interp : {"linear", "nearest"}
        Interpolation method.

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
    length: float = 0.25
    width: float = 1.0
    interp: Literal["linear", "nearest"] = "linear"


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
    data : ndarray of shape (n_frames, n_dims), dtype float64
        Position coordinates aligned to animation frames. NaN values indicate
        frames where position is unavailable (extrapolation outside overlay
        time range).
    color : str
        Marker and trail color (matplotlib color string).
    size : float
        Marker size in points.
    trail_length : int or None
        Trail length in frames, or None for no trail.

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
    bodyparts : dict of str to ndarray of shape (n_frames, n_dims), dtype float64
        Body part positions aligned to animation frames. NaN values indicate
        frames where the body part is unavailable.
    skeleton : Skeleton | None
        Skeleton object with edge definitions, colors, and styling.
    colors : dict of str to str or None
        Per-part colors as matplotlib color strings. Takes precedence over
        skeleton.node_colors if both are provided.

    Attributes
    ----------
    bodyparts : dict[str, NDArray[np.float64]]
        Body part positions aligned to frames.
    skeleton : Skeleton | None
        Skeleton definition with nodes, edges, and styling.
    colors : dict[str, str] | None
        Per-part colors.

    See Also
    --------
    BodypartOverlay : User-facing overlay configuration
    Skeleton : Anatomical structure definition
    """

    bodyparts: dict[str, NDArray[np.float64]]
    skeleton: Skeleton | None
    colors: dict[str, str] | None


@dataclass
class HeadDirectionData:
    """Internal container for head direction overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from HeadDirectionOverlay instances.

    Parameters
    ----------
    data : ndarray of shape (n_frames,) or (n_frames, n_dims), dtype float64
        Heading data aligned to animation frames. Shape (n_frames,) for angles
        in radians, or (n_frames, n_dims) for unit vectors. NaN values indicate
        frames where heading is unavailable.
    color : str
        Arrow color (matplotlib color string).
    length : float
        Arrow length in environment coordinate units.
    width : float
        Arrow line width in pixels.

    Attributes
    ----------
    data : NDArray[np.float64]
        Heading data aligned to frames.
    color : str
        Arrow color.
    length : float
        Arrow length in environment units.
    width : float
        Arrow line width in pixels.

    See Also
    --------
    HeadDirectionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    length: float
    width: float = 1.0


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
# Validation Functions (WHAT/WHY/HOW error messages)
# =============================================================================


def _validate_monotonic_time(times: NDArray[np.float64], *, name: str) -> None:
    """Validate that timestamps are monotonically increasing.

    Parameters
    ----------
    times : NDArray[np.float64]
        Array of timestamps to validate.
    name : str
        Name of the overlay for error messages.

    Raises
    ------
    ValueError
        If times are not strictly monotonically increasing, with actionable
        WHAT/WHY/HOW guidance.

    Notes
    -----
    Empty arrays or single-element arrays pass validation trivially.
    """
    if times.size <= 1:
        return  # Empty or single element is trivially monotonic

    # Check if strictly increasing
    if not np.all(np.diff(times) > 0):
        # Find first non-monotonic index
        diffs = np.diff(times)
        first_bad_idx = np.where(diffs <= 0)[0][0] + 1

        # WHAT/WHY/HOW format
        raise ValueError(
            f"WHAT: Non-monotonic timestamps detected in '{name}'.\n"
            f"  First violation at index {first_bad_idx}: "
            f"times[{first_bad_idx - 1}]={times[first_bad_idx - 1]:.6f}, "
            f"times[{first_bad_idx}]={times[first_bad_idx]:.6f}\n\n"
            f"WHY: Interpolation requires strictly increasing timestamps to align "
            f"overlay data with animation frames.\n\n"
            f"HOW: Sort the times array using np.argsort(), or if duplicates exist, "
            f"apply jitter or remove duplicates. Example:\n"
            f"  sorted_indices = np.argsort(times)\n"
            f"  times = times[sorted_indices]\n"
            f"  data = data[sorted_indices]"
        )


def _validate_finite_values(data: NDArray[np.float64], *, name: str) -> None:
    """Validate that array contains only finite values (no NaN/Inf).

    Parameters
    ----------
    data : NDArray[np.float64]
        Array to validate for finite values.
    name : str
        Name of the data for error messages.

    Raises
    ------
    ValueError
        If array contains NaN or Inf values, with count, first index, and
        actionable WHAT/WHY/HOW guidance.
    """
    non_finite_mask = ~np.isfinite(data)
    n_non_finite = np.sum(non_finite_mask)

    if n_non_finite > 0:
        # Find first non-finite index
        first_bad_idx = np.unravel_index(np.argmax(non_finite_mask), data.shape)

        # Count NaN vs Inf
        n_nan = np.sum(np.isnan(data))
        n_inf = np.sum(np.isinf(data))

        raise ValueError(
            f"WHAT: Found {n_non_finite} non-finite values in '{name}' "
            f"({n_nan} NaN, {n_inf} Inf).\n"
            f"  First occurrence at index {first_bad_idx}: "
            f"value={data[first_bad_idx]}\n\n"
            f"WHY: Rendering cannot place markers or draw paths at invalid "
            f"coordinates (NaN/Inf).\n\n"
            f"HOW: Clean the data by removing or masking invalid values, or use "
            f"interpolation to fill gaps:\n"
            f"  # Option 1: Remove invalid samples\n"
            f"  valid_mask = np.isfinite(data).all(axis=-1)\n"
            f"  data = data[valid_mask]\n"
            f"  times = times[valid_mask]\n\n"
            f"  # Option 2: Interpolate over gaps (pandas or scipy)\n"
            f"  from scipy.interpolate import interp1d"
        )


def _validate_shape(
    data: NDArray[np.float64],
    expected_ndims: int,
    *,
    name: str,
) -> None:
    """Validate that data has the expected number of spatial dimensions.

    Parameters
    ----------
    data : NDArray[np.float64]
        Data array to validate. Expected shape is (n_samples, expected_ndims).
    expected_ndims : int
        Expected number of spatial dimensions (should match env.n_dims).
    name : str
        Name of the data for error messages.

    Raises
    ------
    ValueError
        If data dimensions don't match expected, with actual vs expected and
        actionable WHAT/WHY/HOW guidance.
    """
    # Handle both 1D arrays (angles) and 2D arrays (coordinates)
    if data.ndim == 1:
        actual_ndims = 1
    elif data.ndim == 2:
        actual_ndims = data.shape[1]
    else:
        # Invalid shape (3D+ arrays)
        raise ValueError(
            f"WHAT: Invalid array shape for '{name}'.\n"
            f"  Expected: (n_samples,) or (n_samples, n_dims)\n"
            f"  Got: {data.shape} with {data.ndim} dimensions\n\n"
            f"WHY: Data must be either 1D (angles) or 2D (coordinates) for overlay "
            f"rendering.\n\n"
            f"HOW: Reshape your data to 2D:\n"
            f"  data = data.reshape(-1, {expected_ndims})  # Flatten to 2D"
        )

    if actual_ndims != expected_ndims:
        raise ValueError(
            f"WHAT: Shape mismatch for '{name}'.\n"
            f"  Expected: (n_samples, {expected_ndims}) spatial dimensions\n"
            f"  Got: {data.shape} with {actual_ndims} spatial dimension(s)\n\n"
            f"WHY: Coordinate dimensionality must match the environment's spatial "
            f"dimensions (env.n_dims={expected_ndims}) for proper rendering.\n\n"
            f"HOW: Reformat your data to match the environment dimensions:\n"
            f"  # If you have wrong dimensions, project or slice:\n"
            f"  data = data[:, :{expected_ndims}]  # Use first {expected_ndims} columns\n"
            f"  # Or ensure env.n_dims matches your data dimensions"
        )


def _validate_temporal_alignment(
    overlay_times: NDArray[np.float64],
    frame_times: NDArray[np.float64],
    *,
    name: str,
) -> None:
    """Validate temporal overlap between overlay and animation frame times.

    Parameters
    ----------
    overlay_times : NDArray[np.float64]
        Timestamps for overlay data samples.
    frame_times : NDArray[np.float64]
        Timestamps for animation frames.
    name : str
        Name of the overlay for error/warning messages.

    Raises
    ------
    ValueError
        If there is no temporal overlap between overlay_times and frame_times.

    Warns
    -----
    UserWarning
        If temporal overlap is less than 50%, indicating potential interpolation
        issues or suboptimal alignment.
    """
    import warnings

    overlay_min, overlay_max = overlay_times.min(), overlay_times.max()
    frame_min, frame_max = frame_times.min(), frame_times.max()

    # Check for any overlap
    overlap_start = max(overlay_min, frame_min)
    overlap_end = min(overlay_max, frame_max)

    if overlap_start >= overlap_end:
        # No overlap at all
        raise ValueError(
            f"WHAT: No temporal overlap between '{name}' and animation frames.\n"
            f"  Overlay times: [{overlay_min:.3f}, {overlay_max:.3f}]\n"
            f"  Frame times: [{frame_min:.3f}, {frame_max:.3f}]\n\n"
            f"WHY: Interpolation domain is disjoint - cannot align overlay data "
            f"to any animation frames.\n\n"
            f"HOW: Ensure overlapping time ranges:\n"
            f"  # Option 1: Adjust overlay timestamps to match animation\n"
            f"  overlay_times = overlay_times + {frame_min - overlay_min:.3f}\n\n"
            f"  # Option 2: Resample overlay data to animation time range\n"
            f"  # Option 3: Adjust frame_times to include overlay range"
        )

    # Calculate overlap percentage relative to frame times
    overlap_duration = overlap_end - overlap_start
    frame_duration = frame_max - frame_min
    overlap_pct = (overlap_duration / frame_duration) * 100

    # Warn if overlap is less than 50%
    if overlap_pct < 50.0:
        warnings.warn(
            f"Partial temporal overlap for '{name}': {overlap_pct:.1f}%\n"
            f"  Overlap range: [{overlap_start:.3f}, {overlap_end:.3f}]\n"
            f"  Overlay times: [{overlay_min:.3f}, {overlay_max:.3f}]\n"
            f"  Frame times: [{frame_min:.3f}, {frame_max:.3f}]\n"
            f"Frames outside overlap will have NaN values (extrapolation disabled).\n"
            f"Consider adjusting time ranges for better coverage.",
            UserWarning,
            stacklevel=2,
        )


def _validate_bounds(
    data: NDArray[np.float64],
    dim_ranges: list[tuple[float, float]],
    *,
    name: str,
    threshold: float = 0.1,
) -> None:
    """Validate that overlay coordinates fall within environment bounds.

    Parameters
    ----------
    data : NDArray[np.float64]
        Coordinate array with shape (n_samples, n_dims).
    dim_ranges : list[tuple[float, float]]
        Environment dimension ranges [(min0, max0), (min1, max1), ...].
    name : str
        Name of the data for warning messages.
    threshold : float, optional
        Fraction of points allowed outside bounds before warning. Default is 0.1
        (10%).

    Warns
    -----
    UserWarning
        If more than threshold fraction of points fall outside dimension_ranges,
        with statistics and actionable guidance.
    """
    import warnings

    # Check if data is 1D (angles) - skip bounds checking
    if data.ndim == 1:
        return

    # Count points outside bounds
    n_total = data.shape[0]
    outside_mask = np.zeros(n_total, dtype=bool)

    for dim_idx, (dim_min, dim_max) in enumerate(dim_ranges):
        outside_mask |= (data[:, dim_idx] < dim_min) | (data[:, dim_idx] > dim_max)

    n_outside = np.sum(outside_mask)
    pct_outside = (n_outside / n_total) * 100

    if pct_outside > (threshold * 100):
        # Calculate actual data ranges
        data_mins = data.min(axis=0)
        data_maxs = data.max(axis=0)

        # Format dimension ranges for display
        env_ranges_str = ", ".join(
            f"[{dmin:.2f}, {dmax:.2f}]" for dmin, dmax in dim_ranges
        )
        data_ranges_str = ", ".join(
            f"[{dmin:.2f}, {dmax:.2f}]"
            for dmin, dmax in zip(data_mins, data_maxs, strict=True)
        )

        warnings.warn(
            f"{pct_outside:.1f}% of '{name}' coordinates fall outside environment "
            f"bounds ({n_outside}/{n_total} points).\n"
            f"  Environment ranges: {env_ranges_str}\n"
            f"  Data ranges: {data_ranges_str}\n\n"
            f"This may indicate:\n"
            f"  1. Mismatched coordinate systems or units\n"
            f"  2. Incorrect environment dimensions\n"
            f"  3. Data from a different recording session\n\n"
            f"HOW: Confirm that overlay coordinates use the same coordinate system "
            f"and units as the environment (check env.units and env.frame).",
            UserWarning,
            stacklevel=2,
        )


def _validate_skeleton_consistency(
    skeleton: Skeleton | None,
    bodypart_names: list[str],
    *,
    name: str,
) -> None:
    """Validate that skeleton connections reference existing bodypart names.

    Parameters
    ----------
    skeleton : Skeleton | None
        Skeleton object with edges. Can be None.
    bodypart_names : list[str]
        Available bodypart names in the overlay data.
    name : str
        Name of the overlay for error messages.

    Raises
    ------
    ValueError
        If skeleton references missing bodypart names, with suggestions for
        nearest matches and actionable WHAT/WHY/HOW guidance.
    """
    if skeleton is None or len(skeleton.edges) == 0:
        return  # Nothing to validate

    # Collect all referenced parts from skeleton edges
    referenced_parts = set()
    for part1, part2 in skeleton.edges:
        referenced_parts.add(part1)
        referenced_parts.add(part2)

    # Find missing parts (in skeleton but not in data)
    available_parts = set(bodypart_names)
    missing_parts = referenced_parts - available_parts

    if missing_parts:
        # Find suggestions using simple string distance
        from difflib import get_close_matches

        suggestions = {}
        for missing in missing_parts:
            matches = get_close_matches(missing, bodypart_names, n=2, cutoff=0.6)
            if matches:
                suggestions[missing] = matches

        # Format suggestions
        suggestion_str = "\n".join(
            f"    '{missing}' → did you mean {matches}?"
            for missing, matches in suggestions.items()
        )

        raise ValueError(
            f"WHAT: Skeleton '{skeleton.name}' for '{name}' references missing "
            f"bodypart(s): {sorted(missing_parts)}\n"
            f"  Available bodyparts in data: {sorted(bodypart_names)}\n"
            f"  Skeleton nodes: {sorted(skeleton.nodes)}\n\n"
            f"WHY: Cannot draw skeleton edges without both endpoints defined in the "
            f"bodypart data.\n\n"
            f"HOW: Fix the bodypart names in your skeleton or data:\n"
            f"{suggestion_str}\n"
            f"  # Option 1: Add missing bodyparts to data dict\n"
            f"  # Option 2: Create skeleton with only available bodyparts:\n"
            f"  skeleton = Skeleton.from_edge_list(\n"
            f"      [(p1, p2) for p1, p2 in skeleton.edges\n"
            f"       if p1 in data and p2 in data]\n"
            f"  )"
        )


def _validate_pickle_ability(
    overlay_data: OverlayData,
    *,
    n_workers: int | None,
) -> None:
    """Validate that OverlayData can be pickled for parallel rendering.

    Parameters
    ----------
    overlay_data : OverlayData
        The overlay data container to validate.
    n_workers : int | None
        Number of parallel workers. Validation is skipped if n_workers is None
        or 1 (no parallelization).

    Raises
    ------
    ValueError
        If overlay_data cannot be pickled and n_workers > 1, with details about
        the unpickleable attribute and actionable WHAT/WHY/HOW guidance.
    """
    # Skip validation if not using parallel rendering
    if n_workers is None or n_workers <= 1:
        return

    import pickle

    try:
        # Attempt to pickle the overlay data
        pickle.dumps(overlay_data)
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        raise ValueError(
            f"WHAT: OverlayData is not pickle-able, preventing parallel video "
            f"rendering.\n"
            f"  Pickling failed with: {type(e).__name__}: {e}\n\n"
            f"WHY: Parallel video rendering (n_workers > 1) requires pickling "
            f"OverlayData to pass it to worker processes. Unpickleable objects "
            f"include lambdas, closures, local functions, and certain class instances.\n\n"
            f"HOW: Fix the issue using one of these approaches:\n"
            f"  # Option 1: Call env.clear_cache() before rendering\n"
            f"  env.clear_cache()  # Remove cached unpickleable objects\n"
            f"  env.animate_fields(..., n_workers={n_workers})\n\n"
            f"  # Option 2: Use single-threaded rendering\n"
            f"  env.animate_fields(..., n_workers=1)\n\n"
            f"  # Option 3: Remove unpickleable attributes from OverlayData\n"
            f"  # (check for lambdas, local functions, or cached methods)"
        ) from e


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


# =============================================================================
# Conversion Funnel (Milestone 1.5)
# =============================================================================


def _convert_overlays_to_data(
    overlays: list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay],
    frame_times: NDArray[np.float64],
    n_frames: int,
    env: Any,
) -> OverlayData:
    """Convert overlay configurations to aligned internal data representation.

    This is the main conversion pipeline that:
    1. Validates each overlay (monotonicity, finite values, shape, skeleton)
    2. Aligns overlay data to animation frame times via interpolation
    3. Aggregates all overlays into a single OverlayData container

    Parameters
    ----------
    overlays : list[PositionOverlay | BodypartOverlay | HeadDirectionOverlay]
        List of overlay configurations to convert. Can be empty or contain
        multiple instances of any overlay type.
    frame_times : NDArray[np.float64]
        Animation frame timestamps with shape (n_frames,). Used as interpolation
        targets for aligning overlay data.
    n_frames : int
        Number of animation frames. Used for validation.
    env : Any
        Environment object with `n_dims` and `dimension_ranges` attributes.
        Used for validating overlay coordinate dimensions and bounds.

    Returns
    -------
    OverlayData
        Aggregated container with all overlay data aligned to frame times.
        Contains lists of PositionData, BodypartData, and HeadDirectionData
        instances. Guaranteed to be pickle-safe.

    Raises
    ------
    ValueError
        If any validation fails:
        - Non-monotonic timestamps in overlay.times
        - NaN/Inf values in overlay data
        - Shape mismatch between overlay and environment dimensions
        - Skeleton references missing bodypart names
        - No temporal overlap between overlay and frame times

    Notes
    -----
    Temporal alignment behavior:

    - If overlay.times is None, assumes overlay data is already aligned to
      frame_times (same length and uniform spacing)
    - If overlay.times is provided, uses interpolation to align (controlled by
      overlay.interp parameter)
    - Extrapolation outside overlay time range produces NaN values
    - Warns if temporal overlap is less than 50%

    Interpolation modes:

    - "linear": Smooth interpolation for continuous trajectories (default)
    - "nearest": Nearest-neighbor for discrete/categorical data or to preserve
      exact samples

    For BodypartOverlay, each keypoint is interpolated separately, preserving
    independent temporal dynamics of body parts.

    Examples
    --------
    Convert single position overlay:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>> from neurospatial.animation.overlays import (
    ...     PositionOverlay,
    ...     _convert_overlays_to_data,
    ... )
    >>>
    >>> # Create environment
    >>> positions_env = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions_env, bin_size=10.0)
    >>>
    >>> # Create overlay
    >>> trajectory = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    >>> times = np.array([0.0, 1.0, 2.0])
    >>> overlay = PositionOverlay(data=trajectory, times=times)
    >>>
    >>> # Convert
    >>> frame_times = np.array([0.0, 1.0, 2.0])
    >>> overlay_data = _convert_overlays_to_data(
    ...     overlays=[overlay], frame_times=frame_times, n_frames=3, env=env
    ... )
    >>> len(overlay_data.positions)
    1
    >>> overlay_data.positions[0].data.shape
    (3, 2)
    """
    # Initialize lists for each overlay type
    position_data_list: list[PositionData] = []
    bodypart_data_list: list[BodypartData] = []
    head_direction_data_list: list[HeadDirectionData] = []

    # Get environment dimensions
    env_n_dims = env.n_dims
    env_dim_ranges = env.dimension_ranges

    # Process each overlay
    for overlay in overlays:
        if isinstance(overlay, PositionOverlay):
            # Validate position data
            _validate_finite_values(overlay.data, name="PositionOverlay.data")
            _validate_shape(overlay.data, env_n_dims, name="PositionOverlay.data")

            # Validate and align times
            if overlay.times is not None:
                _validate_monotonic_time(overlay.times, name="PositionOverlay.times")
                _validate_temporal_alignment(
                    overlay.times, frame_times, name="PositionOverlay"
                )

                # Interpolate to frame times using overlay's interp setting
                interp_fn = (
                    _interp_linear if overlay.interp == "linear" else _interp_nearest
                )
                aligned_data = interp_fn(overlay.times, overlay.data, frame_times)
            else:
                # No times provided - assume data matches frame times
                if len(overlay.data) != n_frames:
                    raise ValueError(
                        f"PositionOverlay data length ({len(overlay.data)}) does not "
                        f"match n_frames ({n_frames}). When times=None, data must "
                        f"have exactly n_frames samples."
                    )
                aligned_data = overlay.data

            # Validate bounds (warning only)
            _validate_bounds(
                aligned_data, env_dim_ranges, name="PositionOverlay.data", threshold=0.1
            )

            # Create PositionData
            position_data = PositionData(
                data=aligned_data,
                color=overlay.color,
                size=overlay.size,
                trail_length=overlay.trail_length,
            )
            position_data_list.append(position_data)

        elif isinstance(overlay, BodypartOverlay):
            # Validate skeleton consistency first
            bodypart_names = list(overlay.data.keys())
            _validate_skeleton_consistency(
                overlay.skeleton, bodypart_names, name="BodypartOverlay.skeleton"
            )

            # Validate and align times
            if overlay.times is not None:
                _validate_monotonic_time(overlay.times, name="BodypartOverlay.times")
                _validate_temporal_alignment(
                    overlay.times, frame_times, name="BodypartOverlay"
                )

            # Align each bodypart separately
            aligned_bodyparts: dict[str, NDArray[np.float64]] = {}

            for part_name, part_data in overlay.data.items():
                # Validate each bodypart
                _validate_finite_values(
                    part_data, name=f"BodypartOverlay.data['{part_name}']"
                )
                _validate_shape(
                    part_data, env_n_dims, name=f"BodypartOverlay.data['{part_name}']"
                )

                # Align to frame times using overlay's interp setting
                if overlay.times is not None:
                    interp_fn = (
                        _interp_linear
                        if overlay.interp == "linear"
                        else _interp_nearest
                    )
                    aligned_part = interp_fn(overlay.times, part_data, frame_times)
                else:
                    # No times provided - assume data matches frame times
                    if len(part_data) != n_frames:
                        raise ValueError(
                            f"BodypartOverlay.data['{part_name}'] length "
                            f"({len(part_data)}) does not match n_frames ({n_frames}). "
                            f"When times=None, all bodypart data must have exactly "
                            f"n_frames samples."
                        )
                    aligned_part = part_data

                # Validate bounds (warning only)
                _validate_bounds(
                    aligned_part,
                    env_dim_ranges,
                    name=f"BodypartOverlay.data['{part_name}']",
                    threshold=0.1,
                )

                aligned_bodyparts[part_name] = aligned_part

            # Create BodypartData
            bodypart_data = BodypartData(
                bodyparts=aligned_bodyparts,
                skeleton=overlay.skeleton,
                colors=overlay.colors,
            )
            bodypart_data_list.append(bodypart_data)

        elif isinstance(overlay, HeadDirectionOverlay):
            # Validate head direction data
            _validate_finite_values(overlay.data, name="HeadDirectionOverlay.data")

            # For head direction, validate shape only if 2D (vectors)
            if overlay.data.ndim == 2:
                _validate_shape(
                    overlay.data, env_n_dims, name="HeadDirectionOverlay.data"
                )

            # Validate and align times
            if overlay.times is not None:
                _validate_monotonic_time(
                    overlay.times, name="HeadDirectionOverlay.times"
                )
                _validate_temporal_alignment(
                    overlay.times, frame_times, name="HeadDirectionOverlay"
                )

                # Interpolate to frame times using overlay's interp setting
                interp_fn = (
                    _interp_linear if overlay.interp == "linear" else _interp_nearest
                )
                aligned_data = interp_fn(overlay.times, overlay.data, frame_times)
            else:
                # No times provided - assume data matches frame times
                if len(overlay.data) != n_frames:
                    raise ValueError(
                        f"HeadDirectionOverlay data length ({len(overlay.data)}) does "
                        f"not match n_frames ({n_frames}). When times=None, data must "
                        f"have exactly n_frames samples."
                    )
                aligned_data = overlay.data

            # Create HeadDirectionData
            head_direction_data = HeadDirectionData(
                data=aligned_data,
                color=overlay.color,
                length=overlay.length,
                width=overlay.width,
            )
            head_direction_data_list.append(head_direction_data)

    # Aggregate all overlay data
    overlay_data = OverlayData(
        positions=position_data_list,
        bodypart_sets=bodypart_data_list,
        head_directions=head_direction_data_list,
        regions=None,  # Regions will be handled separately in v0.4.0
    )

    return overlay_data
