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

import copy
import dataclasses
import warnings
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._units import same_coordinate_declaration
from neurospatial._validation import timestamps_of, validate_finite
from neurospatial.animation._resources import _OwnedReaders
from neurospatial.animation._sampling import (
    observed_support,
    resample,
    resolve_support,
    sample_at_times,
)
from neurospatial.geometry import require_cartesian

if TYPE_CHECKING:
    from neurospatial._video_io import VideoReaderProtocol
    from neurospatial.animation.skeleton import Skeleton
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.ops.transforms import AffineND, VideoCalibration


# =============================================================================
# Overlay Protocol: Enables custom overlay extensions
# =============================================================================


@runtime_checkable
class OverlayProtocol(Protocol):
    """Input adapter protocol for the supported animation renderable types.

    This protocol adapts custom inputs to the closed set of internal data
    containers listed by ``convert_to_data``. It does not register a new
    renderer or introduce a new renderable type.

    ``convert_to_data()`` is called once, when an ``Animation`` is built,
    with the scene's clock. Sampled data (positions, pose, headings, object
    vectors, video) keeps its own samples and states their times in
    ``times``; views sample it at the times they show. Omitted
    ``sample_valid`` and ``interval_valid`` masks accept every finite row
    and every interval between finite rows; a row with a NaN is missing and
    is sampled across as if it were deleted.

    See Also
    --------
    PositionOverlay : Built-in position trajectory overlay
    BodypartOverlay : Built-in pose tracking overlay
    HeadDirectionOverlay : Built-in heading visualization overlay
    VideoOverlay : Built-in video background overlay

    Examples
    --------
    Creating a custom overlay::

        from dataclasses import dataclass
        import numpy as np
        from neurospatial.animation.overlays import PositionData


        @dataclass
        class MyCustomOverlay:
            positions: NDArray[np.float64]
            times: NDArray[np.float64]

            def convert_to_data(
                self,
                frame_times: NDArray[np.float64],
                n_frames: int,
                env: Any,
            ) -> PositionData:
                # The samples keep their own times; the scene samples them.
                return PositionData(
                    self.positions, "green", 10.0, None, times=self.times
                )
    """

    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> (
        PositionData
        | BodypartData
        | HeadDirectionData
        | VideoData
        | EventData
        | TimeSeriesData
        | SpikeRasterData
        | EpochData
        | ObjectVectorData
    ):
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            The scene's clock, in seconds, with shape (n_frames,). Overlays
            without times of their own are sampled on it.
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        PositionData | BodypartData | HeadDirectionData | VideoData | EventData | TimeSeriesData | SpikeRasterData | EpochData | ObjectVectorData
            Internal data container.

        Notes
        -----
        Implementations should:
        1. Validate overlay data (NaN marks missing data; reject Inf and wrong shapes)
        2. Keep sampled data on its own clock, stated in ``times``
        3. Return appropriate internal data type
        """
        ...


# =============================================================================
# Public API: User-facing overlay dataclasses
# =============================================================================


_OverlayT = TypeVar("_OverlayT", bound="_Overlay")
_SampledT = TypeVar("_SampledT", bound="_SampledData")
_ItemT = TypeVar("_ItemT")


class _Overlay:
    """A built-in overlay, which an ``Animation`` keeps a snapshot of.

    The snapshot owns read-only copies of the metadata arrays named in
    ``_owned_arrays``, deep copies of ``_owned_objects`` and shallow copies of
    the ``_owned_mappings``; large observed arrays stay caller-owned.
    """

    _owned_arrays: ClassVar[tuple[str, ...]] = ("times", "break_before")
    _owned_objects: ClassVar[tuple[str, ...]] = ()
    _owned_mappings: ClassVar[tuple[str, ...]] = ()

    def _snapshot(self: _OverlayT) -> _OverlayT:
        from neurospatial.animation.scene import _readonly

        result = copy.copy(self)
        for name in self._owned_arrays:
            value = getattr(result, name)
            if value is not None:
                setattr(result, name, _readonly(value))
        for name in self._owned_objects:
            setattr(result, name, copy.deepcopy(getattr(result, name)))
        for name in self._owned_mappings:
            value = getattr(result, name)
            if isinstance(value, dict):
                setattr(result, name, dict(value))
        return result


@dataclass
class PositionOverlay(_Overlay):
    """Single trajectory with optional trail visualization.

    **IMPORTANT - Coordinate System**:
      Provide coordinates in your tracking/environment format (x, y).
      DO NOT pre-convert to napari (row, col) - handled automatically.

      ✓ CORRECT:  positions=trajectory  # Your tracking coordinates
      ✗ INCORRECT: positions=trajectory[:, ::-1]  # Don't swap axes manually

    Represents position data for a single entity (e.g., animal, object) over time.
    Can be rendered with a trail showing recent history. For multi-animal tracking,
    create multiple PositionOverlay instances with different colors.

    Parameters
    ----------
    positions : ndarray of shape (n_samples, n_dims), dtype float64
        Position coordinates in **environment (x, y) format**. Each row is a
        position at a time point. Dimensionality must match the environment
        (env.n_dims). The animation system handles coordinate transformation
        automatically, including axis swap and Y-axis inversion for proper display.
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each position sample, in seconds. If None, samples are
        aligned to animation frames by sample index. Must be monotonically
        increasing. Default is None.
    color : str, optional
        Color for the position marker and trail (matplotlib color string).
        Default is "red".
    size : float, optional
        Position marker diameter in logical screen pixels, independent of
        coordinate units and zoom. Default is 10.0.
    trail_length : int | None, optional
        Length of the trail in animation frames: the path through this
        overlay's own samples over the last ``trail_length`` typical frame
        intervals of the animation, so it spans the same time whatever the
        tracking rate and in every viewer. Older segments fade. If None, no
        trail is rendered. Default is None.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, that the overlay
        bridges, as the trajectory analyses do: a missing row inside a shorter
        gap is drawn interpolated between the rows around it, and nothing is
        drawn across a longer gap. ``None`` bridges every gap; declared
        recording breaks are never bridged.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    break_before : ndarray of bool, shape (n_samples,), optional
        True at sample k forbids interpolation between samples k-1 and k.
        Entry zero is ignored. NaN denotes missing tracking; Inf is rejected.
    units, frame : str, optional
        Declared coordinate units and frame. When the environment declares these,
        they must match (known length-unit aliases are normalized). Convert your
        coordinates explicitly before animating different units or frames.

    Attributes
    ----------
    positions : NDArray[np.float64]
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
    >>> trajectory = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    >>> overlay = PositionOverlay(positions=trajectory)
    >>> overlay.color
    'red'
    >>> overlay.trail_length is None
    True

    Trajectory with timestamps and trail:

    >>> trajectory = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    >>> times = np.array([0.0, 0.5, 1.0])
    >>> overlay = PositionOverlay(
    ...     positions=trajectory, times=times, color="blue", trail_length=10
    ... )
    >>> overlay.trail_length
    10

    Multi-animal tracking:

    >>> positions1 = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> positions2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    >>> animal1 = PositionOverlay(positions=positions1, color="red")
    >>> animal2 = PositionOverlay(positions=positions2, color="blue")
    >>> overlays = [animal1, animal2]
    >>> len(overlays)
    2
    """

    positions: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    color: str = "red"
    size: float = 10.0
    trail_length: int | None = None
    interp: Literal["linear", "nearest"] = "linear"
    _: KW_ONLY
    max_gap: float | None = DEFAULT_MAX_GAP
    break_before: NDArray[np.bool_] | None = None
    units: str | None = None
    frame: str | None = None

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> PositionData:
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        PositionData
            The positions on their own clock: ``times``, or ``frame_times``
            when the overlay has none.
        """
        _validate_position_metadata(self, env)
        validate_finite(
            self.positions, name="PositionOverlay.positions", allow_nan=True
        )
        _validate_shape(self.positions, env.n_dims, name="PositionOverlay.positions")
        times = _sample_clock(
            self.times,
            frame_times,
            n_frames,
            len(self.positions),
            name="PositionOverlay",
        )
        _validate_bounds(
            self.positions,
            env.dimension_ranges,
            name="PositionOverlay.positions",
            threshold=0.1,
        )
        sample_valid, interval_valid = observed_support(
            times,
            self.positions,
            max_gap=self.max_gap,
            break_before=self.break_before,
        )
        return PositionData(
            self.positions,
            self.color,
            self.size,
            self.trail_length,
            times=times,
            interp=self.interp,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class BodypartOverlay(_Overlay):
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

        **Important**: Provide coordinates in **environment (x, y) format** -
        the same coordinate system as your tracking data and Environment.
        Do NOT pre-convert to napari (row, col) format. The animation system
        handles coordinate transformation automatically.
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each sample, in seconds. If None, samples are assumed
        aligned to animation frames by sample index. Must be monotonically
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

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, that the overlay
        bridges, as the trajectory analyses do: a missing row inside a shorter
        gap is drawn interpolated between the rows around it, and nothing is
        drawn across a longer gap. ``None`` bridges every gap; declared
        recording breaks are never bridged.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    break_before : ndarray of bool, shape (n_samples,), optional
        True at sample k forbids interpolation between samples k-1 and k.
        Entry zero is ignored. NaN denotes missing tracking; Inf is rejected.
    units, frame : str, optional
        Coordinate units and reference frame shared by every bodypart. Declared
        values must match the environment's declarations (known length-unit
        aliases are normalized). Convert coordinates explicitly before using
        different units or frames; these labels never rescale the data.

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
    _: KW_ONLY
    max_gap: float | None = DEFAULT_MAX_GAP
    break_before: NDArray[np.bool_] | None = None
    units: str | None = None
    frame: str | None = None

    _owned_objects: ClassVar[tuple[str, ...]] = ("colors", "skeleton")
    _owned_mappings: ClassVar[tuple[str, ...]] = ("data",)

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> BodypartData:
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        BodypartData
            The bodyparts on their own clock: ``times``, or ``frame_times``
            when the overlay has none. Each part has its final color: from
            ``colors``, else the skeleton's ``node_colors``, else red.
        """
        _require_allocentric(
            env,
            "BodypartOverlay",
            why="It draws tracked body parts at their arena coordinates.",
        )
        _validate_position_metadata(self, env)
        bodypart_names = list(self.data.keys())
        _validate_skeleton_consistency(
            self.skeleton, bodypart_names, name="BodypartOverlay.skeleton"
        )
        times: NDArray[np.float64] | None = None
        parts: dict[str, NDArray[np.float64]] = {}
        sample_valid: dict[str, NDArray[np.bool_]] = {}
        interval_valid: dict[str, NDArray[np.bool_]] = {}
        for part_name, part_data in self.data.items():
            name = f"BodypartOverlay.data['{part_name}']"
            validate_finite(part_data, name=name, allow_nan=True)
            _validate_shape(part_data, env.n_dims, name=name)
            if times is None:
                times = _sample_clock(
                    self.times, frame_times, n_frames, len(part_data), name=name
                )
            elif len(part_data) != len(times):
                raise ValueError(
                    f"{name} has {len(part_data)} samples; the other bodyparts "
                    f"have {len(times)}. Fix: give every bodypart one row per "
                    "sample time."
                )
            _validate_bounds(part_data, env.dimension_ranges, name=name, threshold=0.1)
            # Each keypoint has its own missing observations and support.
            parts[part_name] = part_data
            sample_valid[part_name], interval_valid[part_name] = observed_support(
                times,
                part_data,
                max_gap=self.max_gap,
                break_before=self.break_before,
            )
        if times is None:
            times = _sample_clock(
                self.times, frame_times, n_frames, n_frames, name="BodypartOverlay"
            )
        return BodypartData(
            parts,
            self.skeleton,
            self.colors or {},
            times=times,
            interp=self.interp,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class HeadDirectionOverlay(_Overlay):
    """Heading direction visualization rendered as a line from head to indicator.

    Represents directional heading data as either angles (in radians) or unit
    vectors. Rendered as a line segment from head position to direction indicator.
    Commonly used for visualizing animal orientation during navigation.

    When paired with a PositionOverlay, creates a clear visualization:
    - Head position marker (from PositionOverlay)
    - Direction line extending from head in heading direction

    Without a PositionOverlay, the line is a compass arrow drawn from the
    arena center. With several PositionOverlays the start is ambiguous, and
    conversion raises ``ValueError``.

    Parameters
    ----------
    headings : ndarray of shape (n_samples,) or (n_samples, n_dims), dtype float64
        Heading data in one of two formats:

        - Angles: shape (n_samples,), dtype float64, in radians where 0 is
          right (east), π/2 is up (north), etc. (standard mathematical
          convention)
        - Unit vectors: shape (n_samples, n_dims), dtype float64, with
          direction vectors in **environment (x, y) format**. Dimensionality
          must match the environment (env.n_dims).

        **Important**: For unit vectors, use the same coordinate system as
        your tracking data and Environment. Do NOT pre-convert to napari
        (row, col) format. The animation system handles coordinate
        transformation automatically.
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each sample, in seconds. If None, samples are assumed
        aligned to animation frames by sample index. Must be monotonically
        increasing. Default is None.
    color : str, optional
        Colormap name for the direction line. Use colormaps like "hsv",
        "twilight", "gray", or "purple" for best visibility against dark
        backgrounds. Default is "hsv".
    length : float, optional
        Length of the direction line in environment coordinate units.
        The line extends from head position to:
        ``head_position + length * unit_direction``. Default is 15.0.

        **Tip**: A good rule of thumb is to use approximately 2-4x your bin_size.
        For environments measured in centimeters with bin_size=5.0, use
        length=10.0 to 20.0.
    width : float, optional
        Line width in pixels. Default is 3.0.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, that the overlay
        bridges, as the trajectory analyses do: a missing row inside a shorter
        gap is drawn interpolated between the rows around it, and nothing is
        drawn across a longer gap. ``None`` bridges every gap; declared
        recording breaks are never bridged.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    break_before : ndarray of bool, shape (n_samples,), optional
        True at sample k forbids interpolation between samples k-1 and k.
        Entry zero is ignored. NaN denotes missing tracking; Inf is rejected.

    Attributes
    ----------
    headings : NDArray[np.float64]
        Heading angles or unit vectors.
    times : NDArray[np.float64] | None
        Optional timestamps.
    color : str
        Colormap name for direction line.
    length : float
        Line length in environment units.
    width : float
        Line width in pixels.
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
    >>> overlay = HeadDirectionOverlay(headings=angles)
    >>> overlay.headings.shape
    (4,)
    >>> overlay.color
    'hsv'

    Head direction as unit vectors (2D):

    >>> vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    >>> overlay = HeadDirectionOverlay(headings=vectors, color="twilight")
    >>> overlay.headings.shape
    (3, 2)

    Custom styling with visible offset and size:

    >>> overlay = HeadDirectionOverlay(
    ...     headings=angles, color="purple", length=20.0, width=4.0
    ... )
    >>> overlay.length
    20.0
    >>> overlay.width
    4.0

    With timestamps for temporal alignment:

    >>> times = np.array([0.0, 0.5, 1.0, 1.5])
    >>> overlay = HeadDirectionOverlay(headings=angles, times=times)
    >>> overlay.times.shape
    (4,)
    """

    headings: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    color: str = "hsv"
    length: float = 15.0
    width: float = 3.0
    interp: Literal["linear", "nearest"] = "linear"
    _: KW_ONLY
    max_gap: float | None = DEFAULT_MAX_GAP
    break_before: NDArray[np.bool_] | None = None

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> HeadDirectionData:
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        HeadDirectionData
            The headings on their own clock: ``times``, or ``frame_times``
            when the overlay has none.

        Notes
        -----
        The conversion pipeline checks the headings against the environment's
        dimensions, for this overlay and custom adapters alike.
        """
        _require_allocentric(
            env,
            "HeadDirectionOverlay",
            why="It draws allocentric heading arrows in arena coordinates.",
        )
        validate_finite(
            self.headings, name="HeadDirectionOverlay.headings", allow_nan=True
        )
        times = _sample_clock(
            self.times,
            frame_times,
            n_frames,
            len(self.headings),
            name="HeadDirectionOverlay",
        )
        sample_valid, interval_valid = observed_support(
            times,
            self.headings,
            max_gap=self.max_gap,
            break_before=self.break_before,
        )
        return HeadDirectionData(
            self.headings,
            self.color,
            self.length,
            self.width,
            times=times,
            interp=self.interp,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class EventOverlay(_Overlay):
    """Overlay for discrete timestamped events at specified spatial positions.

    Displays events (spikes, licks, rewards, zone entries, etc.) as markers at
    positions that can be either:

    - Explicitly provided per event (for rewards, stimuli, zone events)
    - Interpolated from animal trajectory (for spikes, licks, animal-centric events)

    Supports multiple event types with distinct colors and optional temporal
    persistence (decay) to show recent event history.

    Parameters
    ----------
    event_times : NDArray[np.float64] | dict[str, NDArray[np.float64]]
        Event timestamps in seconds. Either:

        - Single event type: 1D array of event times
        - Multiple event types: dict mapping event names to time arrays

        A pynapple ``Ts`` or ``Tsd`` is read by its timestamps.

    event_positions : NDArray[np.float64] | dict[str, NDArray[np.float64]] | None
        **Position Mode A - Explicit positions** (for rewards, stimuli, zone events):
        Explicit event positions in environment coordinates.

        - Shape: (n_events, n_dims) for per-event positions
        - Shape: (1, n_dims) to broadcast single position to all events
        - dict: Mapping event names to position arrays

        If provided, positions are used directly (no interpolation).
        Mutually exclusive with ``positions``/``position_times``.
        Default is None.
    positions : NDArray[np.float64] | None
        **Position Mode B - Trajectory interpolation** (for spikes, licks):
        Animal position trajectory with shape (n_samples, n_dims) in environment
        (x, y) coordinates. Used to interpolate position at each event time.
        Mutually exclusive with ``event_positions``. Default is None.
    position_times : NDArray[np.float64] | None
        Timestamps for position samples in seconds. Must be monotonically
        increasing. Required when using ``positions`` parameter. Default is None.
    interp : {"linear", "nearest"}, optional
        Interpolation for position lookup at event times. "linear" for smooth
        trajectory-based positions, "nearest" to snap to exact samples.
        Only used with ``positions``/``position_times`` mode. Default is "linear".
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, across which an
        event's position is interpolated from the trajectory; an event inside
        a longer gap has no position. ``None`` bridges every gap; recording
        breaks are never bridged.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    break_before : ndarray of bool, shape (n_samples,), optional
        True at sample k forbids interpolation between samples k-1 and k.
        Exact finite observations remain available beside gaps and breaks.
        Events without supported tracking retain their times and have NaN positions.
    colors : str | dict[str, str] | None, optional
        Colors for event markers:

        - str: Single color for all event types (e.g., "red")
        - dict: Mapping event names to colors
        - None: Auto-assign from perceptually distinct colormap (tab10)

        Default is None.
    size : float, optional
        Marker diameter in logical screen pixels. Default is 8.0.
    decay_frames : int | None, optional
        Number of frames over which event markers persist and fade.

        - None: Events appear only on their exact frame (instant)
        - 0: Same as None (instant)
        - >0: Events persist for N frames with decaying opacity

        Default is None (instant, no decay).
    markers : str | dict[str, str] | None, optional
        Matplotlib marker code(s) ('o', 's', '^', 'v', 'd'), drawn by every
        backend.

        - str: Single marker for all event types
        - dict: Mapping event names to markers
        - None: Use 'o' (circle) for all

        Default is None.
    border_color : str, optional
        Border color for markers. Default is "white".
    border_width : float, optional
        Border width in pixels. Default is 0.5.

    Attributes
    ----------
    event_times : NDArray[np.float64] | dict[str, NDArray[np.float64]]
        Event timestamps (normalized to dict in __post_init__).
    event_positions : NDArray[np.float64] | dict[str, NDArray[np.float64]] | None
        Explicit event positions (Mode A).
    positions : NDArray[np.float64] | None
        Animal trajectory positions (Mode B).
    position_times : NDArray[np.float64] | None
        Trajectory timestamps (Mode B).
    interp : str
        Interpolation method.
    colors : str | dict[str, str] | None
        Event colors.
    size : float
        Marker size.
    decay_frames : int | None
        Decay frame count.
    markers : str | dict[str, str] | None
        Marker styles.
    border_color : str
        Border color.
    border_width : float
        Border width.

    See Also
    --------
    EventOverlay.from_fixed_positions : Create overlay with events at fixed locations (Mode A).
    EventOverlay.from_trajectory : Create overlay with events at animal position (Mode B).
    PositionOverlay : Trajectory visualization overlay.

    Notes
    -----
    **Position Modes (mutually exclusive):**

    EventOverlay supports two mutually exclusive modes for determining where events
    appear in space:

    **Mode A - Fixed Positions** (use ``event_positions``):
      Events occur at known fixed locations, independent of animal position.

      - Use case: Reward delivery at feeder, zone entry at boundary, stimulus at screen
      - Factory: ``EventOverlay.from_fixed_positions()``
      - Parameters: ``event_positions`` (provide), ``positions`` (omit), ``position_times`` (omit)
      - Position shape: (n_events, n_dims) or (1, n_dims) to broadcast

    **Mode B - Trajectory Interpolation** (use ``positions`` + ``position_times``):
      Events occur "at the animal", position interpolated from trajectory at event time.

      - Use case: Neural spikes, licks, lever presses at animal location
      - Factory: ``EventOverlay.from_trajectory()``
      - Parameters: ``positions`` (provide), ``position_times`` (provide), ``event_positions`` (omit)
      - Requires both trajectory and timestamps for temporal interpolation

    **ERROR if both provided:** You cannot provide ``event_positions`` AND
    (``positions`` + ``position_times``) simultaneously - choose one mode.

    **Coordinate Convention:**

    Event positions use environment coordinates (x, y), the same as tracking data.
    The animation system automatically transforms to napari pixel space.

    Examples
    --------
    **RECOMMENDED: Use factory methods for clearer intent:**

    Reward delivery at fixed feeder location::

        events = EventOverlay.from_fixed_positions(
            event_times=reward_times,
            event_positions=np.array([[50.0, 25.0]]),  # Feeder location (broadcast)
            colors="gold",
            markers="s",
        )

    Neural spikes at animal position::

        events = EventOverlay.from_trajectory(
            event_times=spike_times,
            positions=trajectory,  # Shape: (n_samples, 2)
            position_times=timestamps,  # Shape: (n_samples,)
            colors="red",
            size=10.0,
        )

    **Direct constructor usage (advanced):**

    Zone entry events at zone boundaries (explicit positions per event)::

        events = EventOverlay(
            event_times=zone_entry_times,
            event_positions=zone_entry_locations,  # (n_events, 2)
            colors="cyan",
        )

    Multiple neurons with auto-colors (trajectory interpolation)::

        events = EventOverlay(
            event_times={
                "cell_001": spikes_1,
                "cell_002": spikes_2,
                "cell_003": spikes_3,
            },
            positions=trajectory,
            position_times=timestamps,
            # colors=None -> auto-assign from tab10
        )

    Behavioral events with different markers::

        events = EventOverlay(
            event_times={
                "lick": lick_times,
                "reward": reward_times,
                "lever_press": press_times,
            },
            positions=trajectory,
            position_times=timestamps,
            colors={"lick": "cyan", "reward": "gold", "lever_press": "magenta"},
            markers={"lick": "o", "reward": "s", "lever_press": "^"},
        )

    With temporal decay (events visible for 5 frames)::

        events = EventOverlay(
            event_times=event_times,
            positions=trajectory,
            position_times=timestamps,
            decay_frames=5,  # Recent events fade over 5 frames
        )
    """

    # Event times (required)
    event_times: NDArray[np.float64] | dict[str, NDArray[np.float64]]

    # Position Mode A: Explicit positions
    event_positions: NDArray[np.float64] | dict[str, NDArray[np.float64]] | None = None

    # Position Mode B: Trajectory interpolation
    positions: NDArray[np.float64] | None = None
    position_times: NDArray[np.float64] | None = None
    interp: Literal["linear", "nearest"] = "linear"

    # Appearance
    colors: str | dict[str, str] | None = None
    size: float = 8.0
    opacity: float = 0.7  # Base opacity for visible events (0.0-1.0)
    decay_frames: int | None = None
    markers: str | dict[str, str] | None = None
    border_color: str = "white"
    border_width: float = 0.05  # Fraction of point size (napari convention)

    _: KW_ONLY
    max_gap: float | None = DEFAULT_MAX_GAP
    break_before: NDArray[np.bool_] | None = None

    _owned_arrays: ClassVar[tuple[str, ...]] = ("position_times", "break_before")
    _owned_objects: ClassVar[tuple[str, ...]] = ("colors", "markers")
    _owned_mappings: ClassVar[tuple[str, ...]] = ("event_positions",)

    def _snapshot(self) -> EventOverlay:
        from neurospatial.animation.scene import _readonly

        result = super()._snapshot()
        assert isinstance(result.event_times, dict)  # normalized in __post_init__
        result.event_times = {
            name: _readonly(times) for name, times in result.event_times.items()
        }
        return result

    def __post_init__(self) -> None:
        """Validate inputs and normalize event_times to dict format."""
        # Normalize event_times to a dict of float arrays. A pynapple Ts/Tsd
        # is read by its timestamps, not its values.
        if isinstance(self.event_times, Mapping):
            event_times = {
                name: np.asarray(
                    timestamps_of(times, name=f"event_times[{name!r}]"),
                    dtype=np.float64,
                )
                for name, times in self.event_times.items()
            }
        else:
            event_times = {
                "event": np.asarray(
                    timestamps_of(self.event_times, name="event_times"),
                    dtype=np.float64,
                )
            }
        object.__setattr__(self, "event_times", event_times)

        # Normalize event_positions to dict format if provided
        if self.event_positions is not None and isinstance(
            self.event_positions, np.ndarray
        ):
            object.__setattr__(self, "event_positions", {"event": self.event_positions})

        # Validate position mode (mutual exclusion)
        has_explicit = self.event_positions is not None
        has_trajectory = self.positions is not None or self.position_times is not None

        if has_explicit and has_trajectory:
            raise ValueError(
                "WHAT: Both event_positions and positions/position_times provided.\n\n"
                "WHY: EventOverlay supports two mutually exclusive position modes:\n"
                "  - Mode A (explicit): Use event_positions for fixed-location events\n"
                "  - Mode B (trajectory): Use positions + position_times for "
                "events at animal location\n\n"
                "Fix: Provide EITHER event_positions OR (positions AND position_times), "
                "not both."
            )

        if not has_explicit and not has_trajectory:
            raise ValueError(
                "WHAT: No position data provided for events.\n\n"
                "WHY: EventOverlay needs to know where to display each event.\n\n"
                "Fix: Provide either:\n"
                "  - event_positions: for events at fixed locations "
                "(rewards, stimuli, zone entries)\n"
                "  - positions + position_times: for events at animal location "
                "(spikes, licks)"
            )

        # Validate trajectory mode has both required fields
        if has_trajectory:
            if self.positions is None:
                raise ValueError(
                    "WHAT: position_times provided without positions.\n\n"
                    "WHY: Trajectory mode requires both the trajectory array (positions) "
                    "and its timestamps (position_times).\n\n"
                    "Fix: Provide positions array with shape (n_samples, n_dims)."
                )
            if self.position_times is None:
                raise ValueError(
                    "WHAT: positions provided without position_times.\n\n"
                    "WHY: Trajectory mode requires both the trajectory array (positions) "
                    "and its timestamps (position_times) for temporal interpolation.\n\n"
                    "Fix: Provide position_times array with shape (n_samples,)."
                )

        # Validate event_positions shape compatibility (if explicit positions mode)
        if self.event_positions is not None:
            assert isinstance(self.event_positions, dict)  # Type narrowing for mypy
            assert isinstance(self.event_times, dict)  # Type narrowing for mypy
            for name, pos_arr in self.event_positions.items():
                # Get matching event times (avoid `or` on numpy arrays)
                times_arr = self.event_times.get(name)
                if times_arr is None:
                    times_arr = self.event_times.get("event")
                if times_arr is not None and pos_arr.shape[0] not in (
                    1,
                    len(times_arr),
                ):
                    raise ValueError(
                        f"WHAT: event_positions for '{name}' has incompatible shape.\n\n"
                        f"WHY: Got {pos_arr.shape[0]} positions for {len(times_arr)} "
                        f"events. Must be either 1 (broadcast) or {len(times_arr)} "
                        "(per-event).\n\n"
                        "Fix: Provide either:\n"
                        f"  - Single position (1, n_dims) to broadcast to all events\n"
                        f"  - One position per event ({len(times_arr)}, n_dims)"
                    )

        # Validate event times arrays (now guaranteed to be dict after normalization)
        assert isinstance(self.event_times, dict)  # Type narrowing for mypy
        for name, times_arr in self.event_times.items():
            if times_arr.ndim != 1:
                raise ValueError(
                    f"WHAT: Event times for '{name}' is not 1D "
                    f"(got shape {times_arr.shape}).\n\n"
                    "WHY: Event times must be a 1D array of timestamps.\n\n"
                    "Fix: Flatten the array or ensure shape is (n_events,)."
                )
            if not np.all(np.isfinite(times_arr)):
                n_invalid = np.sum(~np.isfinite(times_arr))
                raise ValueError(
                    f"WHAT: Event times for '{name}' contains {n_invalid} "
                    "non-finite values (NaN or Inf).\n\n"
                    "WHY: Event timestamps must be finite numbers for temporal "
                    "alignment.\n\n"
                    "Fix: Remove or interpolate NaN/Inf values before creating overlay."
                )

    @classmethod
    def from_fixed_positions(
        cls,
        event_times: NDArray[np.float64] | dict[str, NDArray[np.float64]],
        event_positions: NDArray[np.float64] | dict[str, NDArray[np.float64]],
        *,
        colors: str | dict[str, str] | None = None,
        size: float = 8.0,
        opacity: float = 0.7,
        decay_frames: int | None = None,
        markers: str | dict[str, str] | None = None,
        border_color: str = "white",
        border_width: float = 0.05,
    ) -> EventOverlay:
        """Create EventOverlay with events at fixed spatial locations.

        Use this factory when events occur at known fixed locations, independent
        of animal position. Examples: reward delivery at feeder, zone entry at
        boundary, stimulus presentation at screen location.

        Parameters
        ----------
        event_times : NDArray[np.float64] | dict[str, NDArray[np.float64]]
            Event timestamps in seconds. Either:

            - Single event type: 1D array of event times
            - Multiple event types: dict mapping event names to time arrays

        event_positions : NDArray[np.float64] | dict[str, NDArray[np.float64]]
            Explicit event positions in environment coordinates.

            - Shape: (n_events, n_dims) for per-event positions
            - Shape: (1, n_dims) to broadcast single position to all events
            - dict: Mapping event names to position arrays

        colors : str | dict[str, str] | None, optional
            Colors for event markers. See EventOverlay for details. Default is None.
        size : float, optional
            Marker diameter in logical screen pixels. Default is 8.0.
        opacity : float, optional
            Base opacity for visible events (0.0-1.0). Default is 0.7.
        decay_frames : int | None, optional
            Number of frames over which event markers persist and fade.
            Default is None (instant).
        markers : str | dict[str, str] | None, optional
            Marker style(s). Default is None.
        border_color : str, optional
            Border color for markers. Default is "white".
        border_width : float, optional
            Border width as fraction of point size. Default is 0.05.

        Returns
        -------
        EventOverlay
            Configured overlay with fixed event positions.

        Examples
        --------
        Reward delivery at fixed feeder location::

            overlay = EventOverlay.from_fixed_positions(
                event_times=reward_times,
                event_positions=np.array([[50.0, 25.0]]),  # Feeder location (broadcast)
                colors="gold",
                markers="s",
            )

        Zone entry events at zone boundaries::

            overlay = EventOverlay.from_fixed_positions(
                event_times=zone_entry_times,
                event_positions=zone_entry_locations,  # (n_events, 2)
                colors="cyan",
            )

        Multiple zones with different colors::

            overlay = EventOverlay.from_fixed_positions(
                event_times={"start": start_times, "goal": goal_times},
                event_positions={"start": start_pos, "goal": goal_pos},
                colors={"start": "green", "goal": "red"},
            )
        """
        return cls(
            event_times=event_times,
            event_positions=event_positions,
            colors=colors,
            size=size,
            opacity=opacity,
            decay_frames=decay_frames,
            markers=markers,
            border_color=border_color,
            border_width=border_width,
        )

    @classmethod
    @reads_pynapple_samples
    def from_trajectory(
        cls,
        event_times: NDArray[np.float64] | dict[str, NDArray[np.float64]],
        positions: NDArray[np.float64],
        position_times: NDArray[np.float64],
        *,
        interp: Literal["linear", "nearest"] = "linear",
        max_gap: float | None = DEFAULT_MAX_GAP,
        break_before: NDArray[np.bool_] | None = None,
        colors: str | dict[str, str] | None = None,
        size: float = 8.0,
        opacity: float = 0.7,
        decay_frames: int | None = None,
        markers: str | dict[str, str] | None = None,
        border_color: str = "white",
        border_width: float = 0.05,
    ) -> EventOverlay:
        """Create EventOverlay with events at animal position (interpolated from trajectory).

        Use this factory when events occur "at the animal", and position should be
        interpolated from the animal's trajectory at event time. Examples: neural
        spikes, licks, lever presses.

        Parameters
        ----------
        event_times : NDArray[np.float64] | dict[str, NDArray[np.float64]]
            Event timestamps in seconds. Either:

            - Single event type: 1D array of event times
            - Multiple event types: dict mapping event names to time arrays

        positions : NDArray[np.float64]
            Animal position trajectory with shape (n_samples, n_dims) in environment
            (x, y) coordinates.
        position_times : NDArray[np.float64]
            Timestamps for position samples in seconds. Must be monotonically
            increasing. Shape: (n_samples,).
        interp : {"linear", "nearest"}, optional
            Interpolation for position lookup at event times. "linear" for smooth
            trajectory-based positions, "nearest" to snap to exact samples.
            Default is "linear".
        max_gap : float or None, default=1.0
            Longest stretch of missing tracking, in seconds, across which an
            event's position is interpolated from the trajectory; an event inside
            a longer gap has no position. ``None`` bridges every gap; recording
            breaks are never bridged.
            The default is calibrated for typical video and DeepLabCut tracking;
            choose a value longer than routine dropped or low-confidence frames and
            shorter than real tracking losses or pickups.
        break_before : ndarray of bool, shape (n_samples,), optional
            True at sample k forbids interpolation from sample k-1 to k.
            Exact finite samples survive; unsupported event positions are NaN.
        colors : str | dict[str, str] | None, optional
            Colors for event markers. See EventOverlay for details. Default is None.
        size : float, optional
            Marker diameter in logical screen pixels. Default is 8.0.
        opacity : float, optional
            Base opacity for visible events (0.0-1.0). Default is 0.7.
        decay_frames : int | None, optional
            Number of frames over which event markers persist and fade.
            Default is None (instant).
        markers : str | dict[str, str] | None, optional
            Marker style(s). Default is None.
        border_color : str, optional
            Border color for markers. Default is "white".
        border_width : float, optional
            Border width as fraction of point size. Default is 0.05.

        Returns
        -------
        EventOverlay
            Configured overlay with trajectory-interpolated positions.

        Examples
        --------
        Neural spikes at animal position::

            overlay = EventOverlay.from_trajectory(
                event_times=spike_times,
                positions=trajectory,  # Shape: (n_samples, 2)
                position_times=timestamps,  # Shape: (n_samples,)
                colors="red",
                size=10.0,
            )

        Multiple neurons with auto-colors::

            overlay = EventOverlay.from_trajectory(
                event_times={
                    "cell_001": spikes_1,
                    "cell_002": spikes_2,
                    "cell_003": spikes_3,
                },
                positions=trajectory,
                position_times=timestamps,
                # colors=None -> auto-assign from tab10
            )

        Behavioral events with different markers::

            overlay = EventOverlay.from_trajectory(
                event_times={
                    "lick": lick_times,
                    "reward": reward_times,
                    "lever_press": press_times,
                },
                positions=trajectory,
                position_times=timestamps,
                colors={"lick": "cyan", "reward": "gold", "lever_press": "magenta"},
                markers={"lick": "o", "reward": "s", "lever_press": "^"},
            )
        """
        return cls(
            event_times=event_times,
            positions=positions,
            position_times=position_times,
            interp=interp,
            max_gap=max_gap,
            break_before=break_before,
            colors=colors,
            size=size,
            opacity=opacity,
            decay_frames=decay_frames,
            markers=markers,
            border_color=border_color,
            border_width=border_width,
        )

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> EventData:
        """Convert EventOverlay to EventData aligned to animation frames.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        EventData
            Internal data container with event positions and frame indices.
        """
        frame_times = _validate_frame_times(frame_times, n_frames)
        support = None
        if self.positions is not None:
            if self.positions.ndim != 2:
                raise ValueError(
                    "EventOverlay.positions must have shape (n_samples, n_dims)."
                )
            _validate_shape(self.positions, env.n_dims, name="EventOverlay.positions")
            validate_finite(
                self.positions, name="EventOverlay.positions", allow_nan=True
            )
            assert self.position_times is not None
            _validate_monotonic_time(
                self.position_times, name="EventOverlay.position_times"
            )
            if not len(self.position_times) or len(self.position_times) != len(
                self.positions
            ):
                raise ValueError(
                    "EventOverlay.position_times must have one timestamp per "
                    "nonempty position sample."
                )
            support = observed_support(
                self.position_times,
                self.positions,
                max_gap=self.max_gap,
                break_before=self.break_before,
            )
        else:
            assert isinstance(self.event_positions, dict)
            for name, positions in self.event_positions.items():
                if positions.ndim != 2:
                    raise ValueError(
                        "EventOverlay.event_positions must have shape (n_events, n_dims)."
                    )
                _validate_shape(
                    positions,
                    env.n_dims,
                    name=f"EventOverlay.event_positions[{name!r}]",
                )
                validate_finite(
                    positions,
                    name=f"EventOverlay.event_positions[{name!r}]",
                    allow_nan=True,
                )

        # Type narrowing: __post_init__ normalizes these to dict format
        assert isinstance(self.event_times, dict)  # Always dict after __post_init__

        event_positions_out: dict[str, NDArray[np.float64]] = {}
        event_frame_indices_out: dict[str, NDArray[np.int_]] = {}
        event_times_out: dict[str, NDArray[np.float64]] = {}

        # Get frame time range for filtering
        frame_t_min = frame_times[0] if len(frame_times) > 0 else 0.0
        frame_t_max = frame_times[-1] if len(frame_times) > 0 else 0.0

        # Process each event type
        for event_name, event_times_arr in self.event_times.items():
            # Filter events to those within frame time range
            in_range_mask = (event_times_arr >= frame_t_min) & (
                event_times_arr <= frame_t_max
            )

            # Events retain their identity when tracking is unavailable. Only
            # the display clock filters events; source support controls location.
            n_outside = np.sum(~in_range_mask)
            if n_outside > 0:
                warnings.warn(
                    f"EventOverlay '{event_name}': {n_outside} events are outside "
                    f"the frame time range [{frame_t_min:.3f}, {frame_t_max:.3f}] "
                    "and will be excluded.",
                    UserWarning,
                    stacklevel=2,
                )

            # Get filtered event times
            filtered_times = event_times_arr[in_range_mask]
            event_times_out[event_name] = filtered_times

            if len(filtered_times) == 0:
                # No events in range - use empty arrays
                event_positions_out[event_name] = np.array([]).reshape(0, env.n_dims)
                event_frame_indices_out[event_name] = np.array([], dtype=np.int_)
                continue

            # Compute positions
            if self.event_positions is not None:
                # Mode A: Explicit positions
                # Type narrowing: __post_init__ normalizes arrays to dict format
                assert isinstance(self.event_positions, dict)
                event_pos = self.event_positions.get(event_name)
                if event_pos is None:
                    # Single array normalized to dict with "event" key
                    event_pos = self.event_positions.get("event")

                if event_pos is not None:
                    # Handle broadcast: single position to all events
                    if event_pos.shape[0] == 1:
                        # Broadcast single position to all filtered events
                        positions_for_events = np.broadcast_to(
                            event_pos, (len(filtered_times), event_pos.shape[1])
                        ).copy()
                    else:
                        # Use positions for events in range
                        positions_for_events = event_pos[in_range_mask]
                else:
                    # Fallback - should not happen if validation passed
                    positions_for_events = np.zeros((len(filtered_times), env.n_dims))
            else:
                # Mode B: Interpolate from trajectory
                # Type narrowing: validation ensures both are not None in this mode
                assert self.positions is not None
                assert self.position_times is not None
                assert support is not None
                rows_valid, rows_intervals = support
                positions_for_events = sample_at_times(
                    self.position_times,
                    self.positions,
                    filtered_times,
                    interp=self.interp,
                    sample_valid=rows_valid,
                    interval_valid=rows_intervals,
                )

            # Compute frame indices (nearest frame for each event)
            frame_indices = np.searchsorted(frame_times, filtered_times)
            frame_indices = np.clip(frame_indices, 0, n_frames - 1)

            event_positions_out[event_name] = positions_for_events
            event_frame_indices_out[event_name] = frame_indices.astype(np.int_)

        # Resolve colors
        colors_out = self._resolve_colors(list(self.event_times.keys()))

        # Resolve markers
        markers_out = self._resolve_markers(list(self.event_times.keys()))

        return EventData(
            event_positions=event_positions_out,
            event_frame_indices=event_frame_indices_out,
            event_times=event_times_out,
            colors=colors_out,
            markers=markers_out,
            size=self.size,
            decay_frames=self.decay_frames,  # Preserve None for cumulative mode
            border_color=self.border_color,
            border_width=self.border_width,
            opacity=self.opacity,
        )

    def _resolve_colors(self, event_names: list[str]) -> dict[str, str]:
        """Resolve colors to dict[str, str] format.

        Parameters
        ----------
        event_names : list[str]
            Names of event types.

        Returns
        -------
        dict[str, str]
            Mapping from event name to color string.
        """
        if self.colors is None:
            # Auto-assign from tab10 colormap
            tab10 = colormaps["tab10"]
            return {name: to_hex(tab10(i % 10)) for i, name in enumerate(event_names)}
        elif isinstance(self.colors, str):
            # Single color for all
            return dict.fromkeys(event_names, self.colors)
        else:
            # Dict provided
            return self.colors

    def _resolve_markers(self, event_names: list[str]) -> dict[str, str]:
        """Resolve markers to dict[str, str] format.

        Parameters
        ----------
        event_names : list[str]
            Names of event types.

        Returns
        -------
        dict[str, str]
            Mapping from event name to marker style.
        """
        if self.markers is None:
            # Default 'o' for all
            return dict.fromkeys(event_names, "o")
        elif isinstance(self.markers, str):
            # Single marker for all
            return dict.fromkeys(event_names, self.markers)
        else:
            # Dict provided
            return self.markers


# Convenience alias for neural spike visualization
"""Convenience alias for EventOverlay for neural spike visualization.

See EventOverlay for full documentation.
"""


@dataclass
class TimeSeriesOverlay(_Overlay):
    """Time series visualization in right column during animation.

    Displays continuous variables (speed, acceleration, LFP, etc.) as scrolling
    time series plots alongside spatial field animations. Multiple overlays can be
    stacked as rows or overlaid on the same plot using the ``group`` parameter.

    Parameters
    ----------
    data : ndarray of shape (n_samples,), dtype float64
        Time series values. No downsampling is applied - full resolution is
        preserved. NaN values create gaps in the line. Inf values are not allowed.
    times : ndarray of shape (n_samples,), dtype float64
        Timestamps for each sample, in seconds. Must be monotonically increasing.
        Required (no default).
    label : str, optional
        Label for Y-axis or legend. Default is "".
    color : str, optional
        Line color (matplotlib color string). Default is "white".
    linewidth : float, optional
        Line width in points. Default is 1.0.
    alpha : float, optional
        Line opacity (0.0-1.0). Default is 1.0.
    group : str | None, optional
        Group name for overlaying multiple variables on same plot.
        Variables with the same group share axes and are overlaid.
        Variables with different groups (or None) get separate rows.
        Default is None (separate row for each).
    normalize : bool, optional
        If True, normalize to 0-1 range for overlaying variables with
        different scales. Only meaningful when group is set.
        **Note**: Constant finite data (min == max) is normalized to zero;
        missing values remain NaN.
        Default is False.
    show_cursor : bool, optional
        Show vertical line at current time. Default is True.
    cursor_color : str, optional
        Color for cursor line. Default is "red".
    vmin : float | None, optional
        Minimum Y-axis value. If None, auto-computed from data.
        Default is None.
    vmax : float | None, optional
        Maximum Y-axis value. If None, auto-computed from data.
        Default is None.

    Attributes
    ----------
    data : NDArray[np.float64]
        Time series values.
    times : NDArray[np.float64]
        Sample timestamps.
    label : str
        Y-axis or legend label.
    color : str
        Line color.
    linewidth : float
        Line width in points.
    alpha : float
        Line opacity.
    group : str | None
        Grouping key for overlay layout.
    normalize : bool
        Whether to normalize data to [0, 1].
    show_cursor : bool
        Whether to show cursor line.
    cursor_color : str
        Cursor line color.
    vmin : float | None
        Explicit Y-axis minimum.
    vmax : float | None
        Explicit Y-axis maximum.

    See Also
    --------
    PositionOverlay : Trajectory visualization
    EventOverlay : Discrete event visualization

    Notes
    -----
    **NaN handling**: NaN values in ``data`` create gaps in the plotted line.
    This is useful for handling periods of missing data.

    **Inf handling**: Inf values are NOT allowed and will raise an error.
    Use NaN for missing data instead.

    **Time window**: every signal panel spans the animation's
    ``window_seconds`` around the current time, so the panels line up.

    **Groups**: signals in one ``group`` share a panel, whose y-axis runs from
    the lowest of their limits to the highest, and its cursor follows the
    first signal's ``show_cursor`` and ``cursor_color``.

    **Y-axis limits**: By default, global limits (min/max across all data) are
    used for stable scales during animation. Use ``vmin``/``vmax`` to override.

    Examples
    --------
    Single time series (speed):

    >>> import numpy as np
    >>> from neurospatial.animation import TimeSeriesOverlay
    >>> speed = np.random.rand(1000) * 100  # cm/s
    >>> times = np.linspace(0, 100, 1000)  # 100 seconds at 10 Hz
    >>> overlay = TimeSeriesOverlay(
    ...     data=speed,
    ...     times=times,
    ...     label="Speed (cm/s)",
    ...     color="cyan",
    ... )
    >>> overlay.label
    'Speed (cm/s)'

    Multiple stacked rows (different groups):

    >>> speed_overlay = TimeSeriesOverlay(data=speed, times=times, label="Speed")
    >>> lfp = np.random.randn(10000)  # 1 kHz LFP
    >>> lfp_times = np.linspace(0, 100, 10000)
    >>> lfp_overlay = TimeSeriesOverlay(
    ...     data=lfp, times=lfp_times, label="LFP", color="yellow"
    ... )
    >>> overlays = [speed_overlay, lfp_overlay]
    >>> len(overlays)
    2

    Overlaid variables (same group, normalized):

    >>> accel = np.random.randn(1000) * 50  # cm/s^2
    >>> speed_overlay = TimeSeriesOverlay(
    ...     data=speed,
    ...     times=times,
    ...     label="Speed",
    ...     group="kinematics",
    ...     normalize=True,
    ...     color="cyan",
    ... )
    >>> accel_overlay = TimeSeriesOverlay(
    ...     data=accel,
    ...     times=times,
    ...     label="Accel",
    ...     group="kinematics",
    ...     normalize=True,
    ...     color="orange",
    ... )
    >>> speed_overlay.group == accel_overlay.group
    True
    """

    # Required fields
    data: NDArray[np.float64]
    times: NDArray[np.float64]

    # Display settings
    label: str = ""
    color: str = "white"
    linewidth: float = 1.0
    alpha: float = 1.0

    # Grouping
    group: str | None = None
    normalize: bool = False

    # Cursor settings
    show_cursor: bool = True
    cursor_color: str = "red"

    # Y-axis limits
    vmin: float | None = None
    vmax: float | None = None

    _owned_arrays: ClassVar[tuple[str, ...]] = ("times",)

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        # Validate data is 1D
        if self.data.ndim != 1:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data must be 1D "
                f"(got shape {self.data.shape}).\n\n"
                "WHY: Time series visualization expects a single variable over time.\n\n"
                "Fix: Flatten the array or select a single column/variable."
            )

        # Validate times is 1D
        if self.times.ndim != 1:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.times must be 1D "
                f"(got shape {self.times.shape}).\n\n"
                "WHY: Timestamps must be a 1D array of time values.\n\n"
                "Fix: Flatten the times array or ensure shape is (n_samples,)."
            )

        # Validate data and times have same length
        if len(self.data) != len(self.times):
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data and times have different lengths "
                f"({len(self.data)} vs {len(self.times)}).\n\n"
                "WHY: Each data point must have a corresponding timestamp.\n\n"
                "Fix: Ensure both arrays have the same number of elements."
            )

        # Validate data has at least one sample
        if len(self.data) == 0:
            raise ValueError(
                "WHAT: TimeSeriesOverlay.data must have at least 1 sample (got 0).\n\n"
                "WHY: Time series visualization requires at least one data point.\n\n"
                "Fix: Provide non-empty data array."
            )

        # Validate times are finite (no NaN or Inf)
        if not np.all(np.isfinite(self.times)):
            n_invalid = np.sum(~np.isfinite(self.times))
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.times contains {n_invalid} "
                "non-finite values (NaN or Inf).\n\n"
                "WHY: Timestamps must be finite numbers for window extraction.\n\n"
                "Fix: Remove or fix NaN/Inf values in times array."
            )

        # Validate times are monotonically increasing
        if len(self.times) > 1 and not np.all(np.diff(self.times) > 0):
            diffs = np.diff(self.times)
            first_bad_idx = np.where(diffs <= 0)[0][0] + 1
            raise ValueError(
                f"WHAT: Non-monotonic timestamps in TimeSeriesOverlay.times.\n"
                f"  First violation at index {first_bad_idx}: "
                f"times[{first_bad_idx - 1}]={self.times[first_bad_idx - 1]:.6f}, "
                f"times[{first_bad_idx}]={self.times[first_bad_idx]:.6f}\n\n"
                "WHY: Window extraction requires strictly increasing timestamps.\n\n"
                "Fix: Sort the times array or remove duplicates."
            )

        # Validate alpha in [0, 1]
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.alpha must be in [0, 1] "
                f"(got {self.alpha}).\n\n"
                "WHY: Alpha is an opacity value between fully transparent (0) "
                "and fully opaque (1).\n\n"
                "Fix: Use a value between 0.0 and 1.0."
            )

        # Validate linewidth is positive
        if self.linewidth <= 0:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.linewidth must be positive "
                f"(got {self.linewidth}).\n\n"
                "WHY: Line width must be a positive value for rendering.\n\n"
                "Fix: Use a positive value like linewidth=1.0."
            )

        # Validate no Inf in data (NaN is allowed for gaps)
        if np.any(np.isinf(self.data)):
            n_inf = np.sum(np.isinf(self.data))
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data contains {n_inf} Inf values.\n\n"
                "WHY: Inf values cannot be rendered. NaN is allowed for gaps, "
                "but Inf is not supported.\n\n"
                "Fix: Replace Inf values with NaN or finite values."
            )

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> TimeSeriesData:
        """Convert overlay to internal data representation.

        Computes the y-axis limits and applies normalization.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object (not used for time series, but required by protocol).

        Returns
        -------
        TimeSeriesData
            Internal data container with the display settings and limits.
        """
        # Compute global limits for stable y-axis (default behavior)
        finite_mask = np.isfinite(self.data)
        if finite_mask.any():
            global_vmin = float(np.min(self.data[finite_mask]))
            global_vmax = float(np.max(self.data[finite_mask]))
        else:
            global_vmin, global_vmax = 0.0, 1.0

        # Override with explicit limits if provided
        if self.vmin is not None:
            global_vmin = self.vmin
        if self.vmax is not None:
            global_vmax = self.vmax

        # Apply normalization if requested
        output_data = self.data.copy()
        if self.normalize:
            range_val = global_vmax - global_vmin
            if range_val > 0:
                output_data = (self.data - global_vmin) / range_val
            else:
                # Constant finite observations normalize to zero; missing
                # observations remain absent from the signal and its plots.
                output_data = np.where(finite_mask, 0.0, np.nan)
            # After normalization, limits become [0, 1]
            global_vmin, global_vmax = 0.0, 1.0

        return TimeSeriesData(
            data=output_data,
            times=self.times,
            label=self.label,
            color=self.color,
            linewidth=self.linewidth,
            alpha=self.alpha,
            group=self.group,
            normalize=self.normalize,
            show_cursor=self.show_cursor,
            cursor_color=self.cursor_color,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
        )


@dataclass
class SpikeRasterOverlay(_Overlay):
    """Spikes of many units as a raster beside the field, on the same clock.

    Each unit is a row of ticks at its spike times, and the panel shows the
    animation's ``window_seconds`` around the current time, lined up with the
    other panels drawn against time, so replay or theta sequences
    can be read against a decoded posterior or the animal's position. Every
    view and export draws it. In ``backend="fastplotlib"``, hovering over a
    spike names its unit; with ``rates``, clicking a unit's row shows its rate
    map under the field and highlights its spikes.

    Parameters
    ----------
    spike_times : sequence of ndarray, ndarray, or spike group
        Spike times of each unit, in seconds, on the clock of
        ``frame_times``: one array per unit, a 2-D NaN-padded array, or a
        group such as a pynapple ``TsGroup``.
    order : array-like of int, shape (n_units,), optional
        The units from the bottom row up, as indices into ``spike_times``.
        On a track, sort by place-field peak with
        ``np.argsort(env.to_linear(rates.peak_locations()))``, or
        ``np.argsort(rates.peak_locations()[:, 0])`` on a 1-D environment.
        Default: the order of ``spike_times``.
    unit_ids : array-like, shape (n_units,), optional
        Unit labels, in the order of ``spike_times``. Default: the labels a
        spike group carries, else ``0 .. n_units - 1``.
    rates : SpatialRatesResult or ndarray of shape (n_units, n_bins), optional
        A rate result validates bin geometry and matches rows by unit ID before
        applying display order. A raw array must already follow spike unit order.
        Each unit's firing rate map, in spikes per second, on the animated
        environment's bins and in the order of ``spike_times``, such as
        ``compute_spatial_rates(...).firing_rates``. NaN marks bins without
        an estimate.
    color : str, default="white"
        Tick color.
    linewidth : float, default=1.5
        Tick width, in screen pixels.

    Raises
    ------
    ValueError
        If there are no units, ``order`` is not a permutation of the units,
        ``unit_ids`` has the wrong length or repeats a label, or a spike time
        is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.animation import SpikeRasterOverlay
    >>> spike_times = [np.array([0.1, 0.5]), np.array([0.3])]
    >>> raster = SpikeRasterOverlay(spike_times, order=[1, 0])
    >>> raster.color
    'white'
    """

    spike_times: Any
    _: KW_ONLY
    order: ArrayLike | None = None
    unit_ids: ArrayLike | None = None
    rates: SpatialRatesResult | NDArray[np.float64] | None = None
    color: str = "white"
    linewidth: float = 1.5
    # The units in row order, bottom row first: the order of spike_times.
    _order: NDArray[np.intp] = field(init=False, repr=False, compare=False)
    _rows: tuple[list[NDArray[np.float64]], NDArray[Any]] = field(
        init=False, repr=False, compare=False
    )

    _owned_arrays: ClassVar[tuple[str, ...]] = ("order", "_order")

    def _snapshot(self) -> SpikeRasterOverlay:
        from neurospatial._results import as_label_array
        from neurospatial.animation.scene import _readonly

        result = super()._snapshot()
        if result.unit_ids is not None:
            result.unit_ids = _readonly(as_label_array(result.unit_ids))
        rows, ids = result._rows
        result._rows = (list(rows), _readonly(ids))
        return result

    def __post_init__(self) -> None:
        """Check the spikes, the order and the labels."""
        from neurospatial._spikes import as_labelled_spike_trains

        trains, ids, _ = as_labelled_spike_trains(
            self.spike_times,
            cast("Any", self.unit_ids),
            context="SpikeRasterOverlay",
        )
        if not trains:
            raise ValueError(
                "SpikeRasterOverlay needs the spikes of at least one unit."
            )
        order = (
            np.arange(len(trains))
            if self.order is None
            else np.asarray(self.order).astype(np.intp, casting="same_kind")
        )
        if not np.array_equal(np.sort(order), np.arange(len(trains))):
            raise ValueError(
                f"order must list each of the {len(trains)} units once, as "
                f"indices into spike_times; got {order.tolist()}. Why: each "
                "row of the raster is one unit. Fix: pass a permutation, such "
                "as np.argsort of each unit's place-field peak."
            )
        self._order = order
        self._rows = [trains[i] for i in order], ids[order]

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> SpikeRasterData:
        """The raster's rows, bottom row first.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,); unused, the
            spikes keep their own times.
        n_frames : int
            Number of animation frames.
        env : Any
            The environment ``rates`` must have one value per bin of.

        Returns
        -------
        SpikeRasterData

        Raises
        ------
        ValueError
            If ``rates`` is not one rate map per unit over ``env``'s bins.
        """
        spike_times, unit_ids = self._rows
        rates = None
        if self.rates is not None:
            from neurospatial.encoding.spatial import SpatialRatesResult

            if isinstance(self.rates, SpatialRatesResult):
                from neurospatial._exceptions import IncompatibleEnvironmentError
                from neurospatial.environment import _environment_differences

                # A rate map drawn under the field must mean the same places;
                # connectivity does not change what a drawn bin shows.
                mismatch = _environment_differences(self.rates.env, env).bins
                if mismatch is not None:
                    raise IncompatibleEnvironmentError(
                        "SpikeRasterOverlay rates were computed on an environment "
                        f"that differs from the animated environment in {mismatch}.",
                        fix="compute rates on the animated environment.",
                        first=self.rates.env,
                        second=env,
                    )
                original_ids = unit_ids[np.argsort(self._order)]
                lookup = {
                    label: i
                    for i, label in enumerate(np.asarray(self.rates.unit_ids).tolist())
                }
                missing = [
                    label for label in original_ids.tolist() if label not in lookup
                ]
                if missing:
                    raise ValueError(
                        _format_error(
                            f"rates is missing raster unit IDs {missing!r}.",
                            why="Why: each raster row shows the rate map of its unit.",
                            fix="provide rates for every spike unit.",
                        )
                    )
                rates = np.asarray(self.rates.firing_rates)[
                    [lookup[label] for label in original_ids.tolist()]
                ]
            else:
                rates = np.asarray(self.rates, dtype=np.float64)
            expected = (len(spike_times), env.n_bins)
            if rates.shape != expected:
                raise ValueError(
                    f"rates has shape {rates.shape}, expected {expected}: one "
                    "rate map per unit, in the order of spike_times, over the "
                    "animated environment's bins. Fix: pass rate maps computed "
                    "on this environment, e.g. "
                    "compute_spatial_rates(env, ...).firing_rates."
                )
            rates = rates[self._order]
        return SpikeRasterData(
            spike_times=spike_times,
            unit_ids=unit_ids,
            rates=rates,
            color=self.color,
            linewidth=self.linewidth,
        )


@dataclass
class EpochOverlay(_Overlay):
    """Intervals of time, such as ripples or run bouts, on every time panel.

    Napari, the notebook widget and exported files draw each overlay as a
    labeled row of shaded intervals in a panel against time. In
    ``backend="fastplotlib"`` each epoch is shaded on the panels drawn against
    time (a track's fields over time, spike rasters and time series), the line
    above the panels names the epoch the current time is in, and N and Shift+N
    jump to the start of the next and previous epoch.

    Parameters
    ----------
    epochs : (start, stop), array-like of shape (n_epochs, 2), or IntervalSet
        Start and stop of each epoch, in seconds, on the clock of
        ``frame_times``. Overlapping or touching epochs are merged.
    label : str, default="epoch"
        What an epoch is, such as ``"ripple"``, for the status line.
    color : str, default="gold"
        Shading color.
    alpha : float, default=0.45
        Shading opacity, from 0 to 1, as it looks in Matplotlib. Over a
        track's fields over time, which shading would tint, epochs are a
        solid strip above them instead.

    Raises
    ------
    ValueError
        If an epoch is not finite or does not stop after it starts, or
        ``alpha`` is outside [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.animation import EpochOverlay
    >>> ripples = EpochOverlay([[1.0, 1.1], [3.2, 3.35]], label="ripple")
    >>> ripples.label
    'ripple'
    """

    epochs: Any
    _: KW_ONLY
    label: str = "epoch"
    color: str = "gold"
    alpha: float = 0.45
    _intervals: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    _owned_arrays: ClassVar[tuple[str, ...]] = ("_intervals",)
    _owned_objects: ClassVar[tuple[str, ...]] = ("epochs",)

    def __post_init__(self) -> None:
        """Check the epochs and the opacity."""
        from neurospatial._intervals import as_intervals

        intervals = as_intervals(self.epochs, name="epochs")
        assert intervals is not None  # epochs is required
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}.")
        self._intervals = intervals

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> EpochData:
        """The epochs, sorted and merged.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps; unused, epochs keep their own times.
        n_frames : int
            Number of animation frames.
        env : Any
            The environment; unused.

        Returns
        -------
        EpochData
        """
        return EpochData(
            intervals=self._intervals,
            label=self.label,
            color=self.color,
            alpha=self.alpha,
        )


@dataclass
class ObjectVectorOverlay(_Overlay):
    """Vector overlay showing lines from animal to objects for object-vector cell analysis.

    Visualizes the spatial relationship between an animal and one or more objects
    in the environment. Renders lines from the animal's position to each object,
    useful for understanding object-vector cell responses.

    The overlay draws vectors pointing from the animal position to object positions
    at each animation frame. Optionally, firing rate modulation can be shown via
    line opacity or color intensity.

    Parameters
    ----------
    object_positions : ndarray of shape (n_objects, n_dims), dtype float64
        Static positions of objects in **environment (x, y) format**. These positions
        remain fixed throughout the animation. Dimensionality must match the
        environment (env.n_dims).
    animal_positions : ndarray of shape (n_samples, n_dims), dtype float64
        Animal position trajectory in **environment (x, y) format**. Each row is a
        position at a time point. Dimensionality must match the environment.
    times : ndarray of shape (n_samples,), dtype float64, optional
        Timestamps for each position sample, in seconds. If None, samples are
        aligned to animation frames by sample index. Must be monotonically
        increasing. Default is None.
    firing_rates : ndarray of shape (n_samples,), dtype float64, optional
        Firing rates at each time point for modulating line appearance (e.g.,
        opacity or color intensity). Must match length of animal_positions.
        If None, uniform appearance is used. Default is None.
    color : str, optional
        Color for the vector lines (matplotlib color string). Default is "white".
    linewidth : float, optional
        Width of the vector lines in pixels. Default is 2.0.
    show_objects : bool, optional
        If True, render object positions as markers. Default is True.
    object_marker : str, optional
        Marker style for objects ('o', 's', '^', etc.). Default is "o".
    object_size : float, optional
        Object marker diameter in logical screen pixels. Default is 15.0.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, that the overlay
        bridges, as the trajectory analyses do: a missing row inside a shorter
        gap is drawn interpolated between the rows around it, and nothing is
        drawn across a longer gap. ``None`` bridges every gap; declared
        recording breaks are never bridged.
        The default is calibrated for typical video and DeepLabCut tracking;
        choose a value longer than routine dropped or low-confidence frames and
        shorter than real tracking losses or pickups.
    break_before : ndarray of bool, shape (n_samples,), optional
        True at sample k forbids interpolation between samples k-1 and k.
        Entry zero is ignored. NaN denotes missing tracking; Inf is rejected.

    Attributes
    ----------
    object_positions : NDArray[np.float64]
        Static object positions.
    animal_positions : NDArray[np.float64]
        Animal position trajectory.
    times : NDArray[np.float64] | None
        Optional timestamps.
    firing_rates : NDArray[np.float64] | None
        Optional firing rate modulation.
    color : str
        Line color.
    linewidth : float
        Line width.
    show_objects : bool
        Whether to show object markers.
    object_marker : str
        Object marker style.
    object_size : float
        Object marker size.
    interp : {"linear", "nearest"}
        Interpolation method.

    See Also
    --------
    PositionOverlay : Single trajectory visualization
    HeadDirectionOverlay : Heading direction visualization
    ObjectVectorCellModel : Simulation model for object-vector cells
    compute_egocentric_rate : Egocentric rate computation for object-vector cells

    Notes
    -----
    **Coordinate conventions**: Use environment (x, y) coordinates, matching
    your tracking data and Environment. Do NOT pre-convert to napari (row, col)
    format - the animation system handles coordinate transformation automatically.

    **Multiple objects**: When multiple objects are provided, vectors are drawn
    from the animal position to each object at every frame.

    **Firing rate modulation**: When firing_rates is provided, it can be used
    by backends to modulate line opacity or color intensity, showing stronger
    responses as brighter/more opaque lines.

    Examples
    --------
    Basic vector overlay with two objects:

    >>> import numpy as np
    >>> from neurospatial.animation import ObjectVectorOverlay
    >>> objects = np.array([[25.0, 75.0], [75.0, 25.0]])
    >>> trajectory = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
    >>> overlay = ObjectVectorOverlay(
    ...     object_positions=objects,
    ...     animal_positions=trajectory,
    ... )
    >>> overlay.show_objects
    True

    With timestamps and firing rate modulation:

    >>> times = np.array([0.0, 0.5, 1.0])
    >>> firing_rates = np.array([1.0, 10.0, 3.0])
    >>> overlay = ObjectVectorOverlay(
    ...     object_positions=objects,
    ...     animal_positions=trajectory,
    ...     times=times,
    ...     firing_rates=firing_rates,
    ...     color="cyan",
    ... )
    >>> overlay.firing_rates is not None
    True
    """

    object_positions: NDArray[np.float64]
    animal_positions: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    firing_rates: NDArray[np.float64] | None = None
    color: str = "white"
    linewidth: float = 2.0
    show_objects: bool = True
    object_marker: str = "o"
    object_size: float = 15.0
    interp: Literal["linear", "nearest"] = "linear"
    _: KW_ONLY
    max_gap: float | None = DEFAULT_MAX_GAP
    break_before: NDArray[np.bool_] | None = None

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> ObjectVectorData:
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        ObjectVectorData
            The animal positions and firing rates on their own clock:
            ``times``, or ``frame_times`` when the overlay has none.

        Raises
        ------
        ValueError
            If shapes don't match or firing_rates length mismatches.
        """
        _require_allocentric(
            env,
            "ObjectVectorOverlay",
            why="It draws lines from the animal to objects in arena coordinates.",
        )
        # Validate object positions shape
        validate_finite(
            self.object_positions,
            name="ObjectVectorOverlay.object_positions",
            allow_nan=True,
        )
        _validate_shape(
            self.object_positions,
            env.n_dims,
            name="ObjectVectorOverlay.object_positions",
        )

        # Validate animal positions
        validate_finite(
            self.animal_positions,
            name="ObjectVectorOverlay.animal_positions",
            allow_nan=True,
        )
        _validate_shape(
            self.animal_positions,
            env.n_dims,
            name="ObjectVectorOverlay.animal_positions",
        )

        # Validate firing rates if provided
        if self.firing_rates is not None:
            validate_finite(
                self.firing_rates,
                name="ObjectVectorOverlay.firing_rates",
                allow_nan=True,
            )
            if len(self.firing_rates) != len(self.animal_positions):
                raise ValueError(
                    f"WHAT: ObjectVectorOverlay.firing_rates length "
                    f"({len(self.firing_rates)}) does not match "
                    f"animal_positions length ({len(self.animal_positions)}).\n\n"
                    "WHY: Each firing rate value corresponds to an animal position "
                    "sample, so they must have the same length.\n\n"
                    "Fix: Ensure firing_rates has shape (n_samples,) matching "
                    "animal_positions shape (n_samples, n_dims)."
                )

        times = _sample_clock(
            self.times,
            frame_times,
            n_frames,
            len(self.animal_positions),
            name="ObjectVectorOverlay.animal_positions",
        )
        _validate_bounds(
            self.animal_positions,
            env.dimension_ranges,
            name="ObjectVectorOverlay.animal_positions",
            threshold=0.1,
        )
        sample_valid, interval_valid = observed_support(
            times,
            self.animal_positions,
            max_gap=self.max_gap,
            break_before=self.break_before,
        )
        # A rate is observed where it and the animal's position are finite;
        # its own gaps are bridged or not like the positions'.
        rate_support = (
            None
            if self.firing_rates is None
            else observed_support(
                times,
                np.column_stack([self.animal_positions, self.firing_rates]),
                max_gap=self.max_gap,
                break_before=self.break_before,
            )
        )
        return ObjectVectorData(
            self.object_positions,
            self.animal_positions,
            self.firing_rates,
            self.color,
            self.linewidth,
            self.show_objects,
            self.object_marker,
            self.object_size,
            times=times,
            interp=self.interp,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
            rate_support=rate_support,
        )


class _DownsampledFrames:
    """Resize in-memory video frames when read, as file readers do.

    Resizing the whole array up front would copy the video on every
    conversion.
    """

    def __init__(self, frames: NDArray[np.uint8], downsample: int) -> None:
        self.frames = frames
        self.n_frames = len(frames)
        self.frame_size_px = (
            frames.shape[2] // downsample,
            frames.shape[1] // downsample,
        )

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        import cv2

        return cast(
            "NDArray[np.uint8]",
            cv2.resize(np.ascontiguousarray(self.frames[idx]), self.frame_size_px),
        )


@dataclass
class VideoOverlay(_Overlay):
    """Video background overlay for displaying recorded footage behind or above fields.

    Renders video frames aligned to animation frame times, optionally transformed
    from video pixel coordinates to environment coordinates using a calibration.
    Useful for visualizing spatial fields overlaid on the original experimental video.

    Parameters
    ----------
    source : str | Path | NDArray[np.uint8]
        Video source. Can be:

        - File path (str or Path): Path to video file (.mp4, .avi, etc.).
          File existence is validated during playback, not at construction.
        - Pre-loaded array: shape (n_frames, height, width, 3), dtype uint8.
          RGB video frames pre-loaded into memory.
    calibration : VideoCalibration | None, optional
        Coordinate transform from video pixels to environment cm.
        If None, the video frame is stretched over ``env.dimension_ranges``,
        the extent of the environment's active cells. On a subset, or a grid
        whose border cells are inactive, that is smaller than the arena the
        camera saw, so a video of the whole arena needs a calibration there.
        Created using calibrate_from_scale_bar() or calibrate_from_landmarks().
        Default is None.
    times : ndarray of shape (n_frames,), dtype float64, optional
        Timestamps for each video frame, in seconds. If None, video frames
        are aligned to animation frames by sample index. Must be monotonically
        increasing. Default is None.
    alpha : float, optional
        Opacity of the video layer (0.0 = transparent, 1.0 = opaque).
        At alpha=0.5, video and field are equally visible (balanced blend).
        Default is 0.5.
    z_order : {"below", "above"}, optional
        Rendering order relative to the spatial field layer.

        - "above" (default): Video blends on top of the field. Use when
          field is opaque (most common case).
        - "below": Video appears behind the field. Only visible if field
          has transparency (e.g., NaN masking, transparent colormap).

        Default is "above".
    crop : tuple[int, int, int, int] | None, optional
        Crop region as (x, y, width, height) in video pixel coordinates.
        If None, full frame is used. Default is None.
    downsample : int, optional
        Spatial downsampling factor. 1 = full resolution, 2 = half resolution, etc.
        Reduces memory and rendering time for large videos. Default is 1.
    interp : {"linear", "nearest"}, optional
        Temporal interpolation method for aligning video to animation frames.

        - "nearest" (default): Use nearest video frame (no interpolation)
        - "linear": **Not yet implemented** - emits warning and uses nearest

        Currently only "nearest" is supported. Linear interpolation for video
        would require blending adjacent frames, which is expensive. A warning
        is emitted if "linear" is requested. Default is "nearest".
    cache_size : int, optional
        Number of video frames to keep in memory cache for file-based video.
        Larger values reduce disk I/O but use more memory. Only applies to
        file-based video sources (ignored for pre-loaded arrays).
        Default is 100.
    prefetch_ahead : int, optional
        Number of frames to prefetch in background when a frame is accessed.
        Set to 0 (default) to disable prefetching. A value of 5 will load
        frames [current+1, current+5] in a background thread after each access.
        This can improve playback smoothness for file-based video by hiding
        disk I/O latency. Only applies to file-based video sources.
        Default is 0.

    Attributes
    ----------
    source : str | Path | NDArray[np.uint8]
        Video source (file path or array).
    calibration : VideoCalibration | None
        Pixel-to-cm coordinate transform.
    times : NDArray[np.float64] | None
        Video frame timestamps.
    alpha : float
        Video opacity.
    z_order : {"below", "above"}
        Rendering order.
    crop : tuple[int, int, int, int] | None
        Crop region in pixels.
    downsample : int
        Spatial downsampling factor.
    interp : {"linear", "nearest"}
        Temporal interpolation method (currently only "nearest" is implemented).
    cache_size : int
        Frame cache size for file-based video.
    prefetch_ahead : int
        Number of frames to prefetch ahead.

    See Also
    --------
    PositionOverlay : Trajectory visualization
    neurospatial.ops.transforms.calibrate_from_scale_bar : Create calibration from scale bar
    neurospatial.ops.transforms.calibrate_from_landmarks : Create calibration from landmarks
    neurospatial.ops.transforms.VideoCalibration : Calibration data container

    Notes
    -----
    Video calibration workflow:

    1. Identify landmarks in video pixels (corners, rulers, etc.)
    2. Measure corresponding coordinates in environment cm
    3. Create calibration using calibrate_from_scale_bar() or calibrate_from_landmarks()
    4. Pass calibration to VideoOverlay

    Memory considerations:

    - File path source: Video is read frame-by-frame during playback (low memory)
    - Pre-loaded array: Video is stored in memory (fast but memory-intensive)

    For large videos, use downsample > 1 to reduce memory and improve performance.

    Examples
    --------
    Video overlay with calibration:

    >>> import numpy as np
    >>> from neurospatial.ops.transforms import (
    ...     calibrate_from_scale_bar,
    ...     VideoCalibration,
    ... )
    >>> # Create calibration from scale bar in video
    >>> transform = calibrate_from_scale_bar(
    ...     p1_px=(100.0, 200.0),
    ...     p2_px=(300.0, 200.0),
    ...     known_length_cm=50.0,
    ...     frame_size_px=(640, 480),
    ... )
    >>> calib = VideoCalibration(transform, frame_size_px=(640, 480))
    >>> # Create video overlay
    >>> overlay = VideoOverlay(source="experiment.mp4", calibration=calib)
    >>> overlay.alpha
    0.5
    >>> overlay.z_order
    'above'

    Pre-loaded video array:

    >>> # 10 frames, 480x640 pixels, RGB
    >>> frames = np.zeros((10, 480, 640, 3), dtype=np.uint8)
    >>> overlay = VideoOverlay(source=frames, alpha=0.5, z_order="above")
    >>> overlay.source.shape
    (10, 480, 640, 3)

    With temporal alignment:

    >>> times = np.linspace(0.0, 10.0, 300)  # 30 Hz video over 10 seconds
    >>> overlay = VideoOverlay(source="video.mp4", times=times)
    >>> overlay.times.shape
    (300,)
    """

    source: str | Path | NDArray[np.uint8]
    calibration: VideoCalibration | None = None
    times: NDArray[np.float64] | None = None
    alpha: float = 0.5
    z_order: Literal["below", "above"] = "above"
    crop: tuple[int, int, int, int] | None = None
    downsample: int = 1
    interp: Literal["linear", "nearest"] = "nearest"
    cache_size: int = 100
    prefetch_ahead: int = 0

    _owned_arrays: ClassVar[tuple[str, ...]] = ("times",)
    _owned_objects: ClassVar[tuple[str, ...]] = ("calibration",)

    def __post_init__(self) -> None:
        """Validate VideoOverlay parameters.

        Raises
        ------
        ValueError
            If alpha is not in [0.0, 1.0].
            If source is an array with wrong shape, dtype, or channels.
            If downsample is not a positive integer.
        """
        # Validate alpha bounds
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                f"WHAT: alpha must be between 0.0 and 1.0, got {self.alpha}.\n"
                f"WHY: Alpha controls transparency (0=invisible, 1=opaque).\n"
                f"Fix: Use alpha=0.5 (default) for semi-transparent overlay."
            )

        # Validate downsample
        if not isinstance(self.downsample, int) or self.downsample < 1:
            raise ValueError(
                f"WHAT: downsample must be a positive integer >= 1, got {self.downsample}.\n"
                f"WHY: Downsample factor controls spatial resolution reduction.\n"
                f"Fix: Use downsample=1 (full resolution) or downsample=2 (half resolution)."
            )

        # Validate cache_size
        if not isinstance(self.cache_size, int) or self.cache_size < 1:
            raise ValueError(
                f"WHAT: cache_size must be a positive integer >= 1, got {self.cache_size}.\n"
                f"WHY: cache_size controls how many video frames are kept in memory.\n"
                f"Fix: Use cache_size=100 (default) or increase for faster playback "
                f"with more memory usage."
            )

        # Validate prefetch_ahead
        if not isinstance(self.prefetch_ahead, int) or self.prefetch_ahead < 0:
            raise ValueError(
                f"WHAT: prefetch_ahead must be a non-negative integer, "
                f"got {self.prefetch_ahead}.\n"
                f"WHY: prefetch_ahead controls how many frames to load in background.\n"
                f"Fix: Use prefetch_ahead=0 (default) to disable, or e.g. 5 to preload "
                f"5 upcoming frames during playback."
            )

        # Validate source array if it's a numpy array
        if isinstance(self.source, np.ndarray):
            self._validate_source_array(self.source)

        # Warn if linear interpolation is requested (not implemented for video)
        if self.interp == "linear":
            import warnings

            warnings.warn(
                "WHAT: VideoOverlay interp='linear' is not yet implemented.\n"
                "WHY: Linear interpolation for video would require blending adjacent "
                "frames, which is computationally expensive and rarely needed.\n"
                "Fix: Using nearest-neighbor frame selection instead. Set interp='nearest' "
                "explicitly to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

    def _validate_source_array(self, arr: NDArray[np.uint8]) -> None:
        """Validate source array has correct shape and dtype.

        Parameters
        ----------
        arr : ndarray
            Array to validate.

        Raises
        ------
        ValueError
            If array has wrong shape, dtype, or number of channels.
        """
        # Check ndim (must be 4D: n_frames, height, width, channels)
        if arr.ndim != 4:
            raise ValueError(
                f"WHAT: Video array must be 4D (n_frames, height, width, channels), "
                f"got shape {arr.shape} ({arr.ndim}D).\n"
                f"WHY: Video data requires frames, height, width, and color channels.\n"
                f"Fix: Reshape array to (n_frames, height, width, 3) for RGB video."
            )

        # Check channels (must be 3 for RGB)
        if arr.shape[3] != 3:
            raise ValueError(
                f"WHAT: Video array must have 3 RGB channels, got {arr.shape[3]} channels.\n"
                f"WHY: VideoOverlay requires RGB format for rendering.\n"
                f"Fix: Convert to RGB (drop alpha channel if RGBA, or convert grayscale)."
            )

        # Check dtype (must be uint8)
        if arr.dtype != np.uint8:
            raise ValueError(
                f"WHAT: Video array must have dtype uint8, got {arr.dtype}.\n"
                f"WHY: Video pixels are 0-255 values stored as unsigned 8-bit integers.\n"
                f"Fix: Convert with arr.astype(np.uint8) if values are in 0-255 range."
            )

    @reads_pynapple_samples
    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> VideoData:
        """Convert overlay to internal data representation.

        Parameters
        ----------
        frame_times : NDArray[np.float64]
            Animation frame timestamps with shape (n_frames,).
        n_frames : int
            Number of animation frames.
        env : Any
            Environment object with ``n_dims`` and ``dimension_ranges``.

        Returns
        -------
        VideoData
            The video frames on their own clock: ``times``, or
            ``frame_times`` when the overlay has none. A file source is read
            by a reader the caller closes (``data.reader.close()``); an
            ``Animation`` closes the readers its overlays open.
        """
        frame_times = _validate_frame_times(frame_times, n_frames)
        _require_allocentric(
            env,
            "VideoOverlay",
            why="It places camera frames in arena coordinates.",
        )
        import warnings

        # Validate environment supports video overlays
        _validate_video_env(env)

        # Warn if no calibration provided
        if self.calibration is None:
            warnings.warn(
                "VideoOverlay has no calibration. Video will be scaled to fit "
                "environment bounds but may not align accurately with spatial data. "
                "Consider providing a VideoCalibration via calibrate_from_scale_bar() "
                "or calibrate_from_landmarks().",
                UserWarning,
                stacklevel=2,
            )

        with ExitStack() as cleanup:
            # Determine video source: array or file path
            if isinstance(self.source, np.ndarray):
                original_size = (self.source.shape[2], self.source.shape[1])
                _validate_video_processing(original_size, self.crop, self.downsample)
                reader: NDArray[np.uint8] | Any = self.source
                if self.crop is not None:
                    x, y, width, height = self.crop
                    reader = reader[:, y : y + height, x : x + width]
                if self.downsample > 1:
                    reader = _DownsampledFrames(reader, self.downsample)
                    processed_size = reader.frame_size_px
                else:
                    processed_size = (reader.shape[2], reader.shape[1])
                n_video_frames = self.source.shape[0]
            else:
                # File path - create _VideoReader
                from neurospatial._video_io import _VideoReader

                reader = _VideoReader(
                    self.source,
                    cache_size=self.cache_size,
                    downsample=self.downsample,
                    crop=self.crop,
                    prefetch_ahead=self.prefetch_ahead,
                )
                cleanup.callback(reader.close)
                n_video_frames = reader.n_frames
                original_size = reader.original_size_px
                _validate_video_processing(original_size, self.crop, self.downsample)
                processed_size = reader.frame_size_px

            # Each video frame is a sample at its own time.
            if self.times is not None:
                times = np.asarray(self.times, dtype=np.float64)
                _validate_monotonic_time(times, name="VideoOverlay.times")
                if len(times) != n_video_frames:
                    raise ValueError(
                        f"VideoOverlay.times length ({len(times)}) does not "
                        f"match number of video frames ({n_video_frames})."
                    )
                _validate_temporal_alignment(times, frame_times, name="VideoOverlay")
            else:
                if n_video_frames != n_frames:
                    raise ValueError(
                        f"VideoOverlay source has {n_video_frames} frames but "
                        f"animation has {n_frames} frames. When times=None, "
                        f"video frames must match animation frames."
                    )
                times = frame_times

            # Compute environment bounds for transform
            env_dim_ranges = env.dimension_ranges
            x_range = env_dim_ranges[0]
            y_range = env_dim_ranges[1]
            env_bounds = (
                float(x_range[0]),
                float(x_range[1]),
                float(y_range[0]),
                float(y_range[1]),
            )

            # Transform from calibration (to be used by backends)
            transform_to_env = None
            if self.calibration is not None:
                from neurospatial.animation.calibration import _calibration_units_per_cm
                from neurospatial.ops.transforms import AffineND, scale_2d

                # cv2.resize maps output pixel centers to input pixel centers:
                # old = (new + 0.5) * (input_size / output_size) - 0.5.
                # Use actual dimensions; floor division can make scale != downsample.
                crop_x, crop_y = (0, 0) if self.crop is None else self.crop[:2]
                width, height = original_size if self.crop is None else self.crop[2:]
                scale_x, scale_y = width / processed_size[0], height / processed_size[1]
                processed_to_original = AffineND(
                    np.array(
                        [
                            [scale_x, 0, crop_x + (scale_x - 1) / 2],
                            [0, scale_y, crop_y + (scale_y - 1) / 2],
                            [0, 0, 1],
                        ],
                        dtype=np.float64,
                    )
                )
                transform_to_env = (
                    scale_2d(_calibration_units_per_cm(env))
                    @ self.calibration.transform_px_to_cm
                    @ processed_to_original
                )

            result = VideoData(
                np.arange(n_video_frames, dtype=np.int_),
                reader,
                transform_to_env,
                env_bounds,
                self.alpha,
                self.z_order,
                times=times,
            )
            cleanup.pop_all()
            return result


# =============================================================================
# Internal data model: Backend-facing containers
# =============================================================================


def _check_rows(
    data: Any,
    rows: dict[str, Any],
    masks: dict[str, tuple[Any, Any]],
) -> None:
    """Check that observed rows and their masks match the sample clock."""
    kind = type(data).__name__
    n_times = len(data.times)
    for name, values in rows.items():
        if np.ndim(values) not in (1, 2) or len(values) != n_times:
            raise ValueError(
                f"{kind}.{name} must have one row per sample time ({n_times})."
            )
    for name, pair in masks.items():
        for label, mask, expected in zip(
            ("sample_valid", "interval_valid"),
            pair,
            (n_times, max(n_times - 1, 0)),
            strict=True,
        ):
            if mask is not None and (
                np.shape(mask) != (expected,) or np.asarray(mask).dtype != np.bool_
            ):
                raise ValueError(
                    f"{kind}.{label} ({name}) must be a boolean array with "
                    f"shape ({expected},)."
                )


def _moved(item: _ItemT, **changes: Any) -> _ItemT:
    """A shallow copy of ``item`` with ``changes``: rows already resolved."""
    moved = copy.copy(item)
    for name, value in changes.items():
        setattr(moved, name, value)
    return moved


class _SampledData:
    """Observed rows of an overlay at their own ``times``.

    Position, pose, heading and object-vector data keep each overlay's
    samples. Their support masks are resolved when the data is made: a
    sample is valid when its row is finite and the given ``sample_valid``
    allows it, an interval when a valid sample bounds its gap on each side
    and the given ``interval_valid`` allows it. Views sample across the gap
    from those two samples, so a rejected row behaves as a deleted one.
    Rejected rows are NaN. Omitted masks accept every finite row and every
    interval between finite rows.
    """

    times: NDArray[np.float64]
    interp: Literal["linear", "nearest"]

    def support(
        self, name: str | None = None
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """``(sample_valid, interval_valid)`` of one stream of rows.

        Parameters
        ----------
        name : str, optional
            The stream: a bodypart, or ``"firing_rates"``. None is the
            overlay's positions or headings.
        """
        raise NotImplementedError

    def on_clock(self: _SampledT, clock: NDArray[np.float64]) -> _SampledT:
        """The rows and their support sampled at ``clock``."""
        raise NotImplementedError


@dataclass
class PositionData(_SampledData):
    """Internal container for position overlay data on its own clock.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from PositionOverlay instances.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_dims), dtype float64
        Position coordinates at ``times``. NaN marks a missing sample.
    color : str
        Marker and trail color (matplotlib color string).
    size : float
        Marker diameter in logical screen pixels.
    trail_length : int or None
        Trail length in frames, or None for no trail.
    times : ndarray of shape (n_samples,), dtype float64
        Times of the rows, in seconds.
    interp : {"linear", "nearest"}
        How views sample the rows between observations.
    sample_valid, interval_valid : ndarray of bool, optional
        Samples and intervals that may be used; see ``_SampledData``.

    See Also
    --------
    PositionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    size: float
    trail_length: int | None
    _: KW_ONLY
    times: NDArray[np.float64]
    interp: Literal["linear", "nearest"] = "linear"
    # None accepts every finite row; __post_init__ always stores an array.
    sample_valid: NDArray[np.bool_] = None  # type: ignore[assignment]
    interval_valid: NDArray[np.bool_] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Resolve the rows' support."""
        _check_rows(
            self,
            {"data": self.data},
            {"data": (self.sample_valid, self.interval_valid)},
        )
        self.data, self.sample_valid, self.interval_valid = resolve_support(
            self.data, self.sample_valid, self.interval_valid
        )

    def support(
        self, name: str | None = None
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """``(sample_valid, interval_valid)`` of the positions."""
        return self.sample_valid, self.interval_valid

    def on_clock(self, clock: NDArray[np.float64]) -> PositionData:
        """The positions and their support sampled at ``clock``."""
        data, sample_valid, interval_valid = resample(
            self.times,
            self.data,
            self.sample_valid,
            self.interval_valid,
            clock,
            interp=self.interp,
        )
        return _moved(
            self,
            times=clock,
            data=data,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class BodypartData(_SampledData):
    """Internal container for bodypart overlay data on its own clock.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from BodypartOverlay instances.

    Parameters
    ----------
    bodyparts : dict of str to ndarray of shape (n_samples, n_dims), dtype float64
        Body part positions at ``times``. NaN marks a missing sample.
    skeleton : Skeleton | None
        Skeleton object with edge definitions, colors, and styling.
    colors : dict of str to str
        Each body part's color, as a matplotlib color string. Parts it omits
        (all, for None) take the skeleton's node_colors, else red.
    times : ndarray of shape (n_samples,), dtype float64
        Times of the rows, in seconds.
    interp : {"linear", "nearest"}
        How views sample the rows between observations.
    sample_valid, interval_valid : dict of str to ndarray of bool, optional
        Each body part's support; see ``_SampledData``. A missing entry
        accepts every finite row of that part.

    See Also
    --------
    BodypartOverlay : User-facing overlay configuration
    Skeleton : Anatomical structure definition
    """

    bodyparts: dict[str, NDArray[np.float64]]
    skeleton: Skeleton | None
    colors: dict[str, str]
    _: KW_ONLY
    times: NDArray[np.float64]
    interp: Literal["linear", "nearest"] = "linear"
    # None accepts every finite row; __post_init__ always stores every part.
    sample_valid: dict[str, NDArray[np.bool_]] = None  # type: ignore[assignment]
    interval_valid: dict[str, NDArray[np.bool_]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Resolve each body part's support and color."""
        given = {
            label: {} if masks is None else masks
            for label, masks in (
                ("sample_valid", self.sample_valid),
                ("interval_valid", self.interval_valid),
            )
        }
        for label, masks in given.items():
            if not isinstance(masks, dict):
                raise ValueError(
                    f"BodypartData.{label} must map bodypart names to boolean masks."
                )
            unknown = masks.keys() - self.bodyparts.keys()
            if unknown:
                raise ValueError(
                    f"BodypartData.{label} references unknown bodyparts: "
                    f"{sorted(unknown, key=str)}."
                )
        # Every view draws a part in the same color: given, else the
        # skeleton's node color, else red.
        node_colors = {} if self.skeleton is None else self.skeleton.node_colors or {}
        self.colors = {
            name: (self.colors or {}).get(name, node_colors.get(name, "red"))
            for name in self.bodyparts
        }
        _check_rows(
            self,
            self.bodyparts,
            {
                name: (
                    given["sample_valid"].get(name),
                    given["interval_valid"].get(name),
                )
                for name in self.bodyparts
            },
        )
        bodyparts, sample_valid, interval_valid = {}, {}, {}
        for name, values in self.bodyparts.items():
            bodyparts[name], sample_valid[name], interval_valid[name] = resolve_support(
                values,
                given["sample_valid"].get(name),
                given["interval_valid"].get(name),
            )
        self.bodyparts = bodyparts
        self.sample_valid, self.interval_valid = sample_valid, interval_valid

    def support(
        self, name: str | None = None
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """``(sample_valid, interval_valid)`` of the body part ``name``."""
        if name is None:
            raise ValueError("Name the body part whose support to return.")
        return self.sample_valid[name], self.interval_valid[name]

    def on_clock(self, clock: NDArray[np.float64]) -> BodypartData:
        """Each body part and its support sampled at ``clock``."""
        bodyparts, sample_valid, interval_valid = {}, {}, {}
        for name, values in self.bodyparts.items():
            bodyparts[name], sample_valid[name], interval_valid[name] = resample(
                self.times,
                values,
                self.sample_valid[name],
                self.interval_valid[name],
                clock,
                interp=self.interp,
            )
        return _moved(
            self,
            times=clock,
            bodyparts=bodyparts,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class HeadDirectionData(_SampledData):
    """Internal container for head direction overlay data on its own clock.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from HeadDirectionOverlay instances.

    Parameters
    ----------
    data : ndarray of shape (n_samples,) or (n_samples, n_dims), dtype float64
        Headings at ``times``: angles in radians, shape (n_samples,), or
        unit vectors, shape (n_samples, n_dims). NaN marks a missing sample.
    color : str
        Colormap name for the direction line.
    length : float
        Length of direction line in environment coordinate units.
    width : float
        Line width in pixels.
    times : ndarray of shape (n_samples,), dtype float64
        Times of the rows, in seconds.
    interp : {"linear", "nearest"}
        How views sample the rows between observations; angles interpolate
        on the circle.
    sample_valid, interval_valid : ndarray of bool, optional
        Samples and intervals that may be used; see ``_SampledData``.

    See Also
    --------
    HeadDirectionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    length: float
    width: float = 3.0
    _: KW_ONLY
    times: NDArray[np.float64]
    interp: Literal["linear", "nearest"] = "linear"
    # None accepts every finite row; __post_init__ always stores an array.
    sample_valid: NDArray[np.bool_] = None  # type: ignore[assignment]
    interval_valid: NDArray[np.bool_] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Resolve the rows' support."""
        _check_rows(
            self,
            {"data": self.data},
            {"data": (self.sample_valid, self.interval_valid)},
        )
        self.data, self.sample_valid, self.interval_valid = resolve_support(
            self.data, self.sample_valid, self.interval_valid
        )

    def support(
        self, name: str | None = None
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """``(sample_valid, interval_valid)`` of the headings."""
        return self.sample_valid, self.interval_valid

    def on_clock(self, clock: NDArray[np.float64]) -> HeadDirectionData:
        """The headings and their support sampled at ``clock``."""
        data, sample_valid, interval_valid = resample(
            self.times,
            self.data,
            self.sample_valid,
            self.interval_valid,
            clock,
            interp=self.interp,
            circular=self.data.ndim == 1,
        )
        return _moved(
            self,
            times=clock,
            data=data,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
        )


@dataclass
class EventData:
    """Internal container for event overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from EventOverlay instances.

    Parameters
    ----------
    event_positions : dict[str, NDArray[np.float64]]
        Dict mapping event type names to position arrays with shape (n_events, n_dims).
        Positions are in environment coordinates (x, y).
    event_frame_indices : dict[str, NDArray[np.int_]]
        Dict mapping event type names to frame index arrays with shape (n_events,).
        Each value is the animation frame index where the event should appear.
    event_times : dict[str, ndarray of shape (n_events,)], optional
        Times of the events within the animation, in seconds, by type.
    colors : dict[str, str]
        Dict mapping event type names to color strings.
    markers : dict[str, str]
        Dict mapping event type names to marker style strings.
    size : float
        Marker diameter in logical screen pixels.
    decay_frames : int | None
        Number of frames over which events persist:

        - None: Cumulative mode - events stay visible permanently once they appear
        - 0: Instant mode - events visible only on their exact frame
        - >0: Decay mode - events visible for N frames then hidden
    border_color : str
        Border color for markers.
    border_width : float
        Border width in pixels.
    opacity : float
        Base opacity for visible events (0.0-1.0).

    Attributes
    ----------
    event_positions : dict[str, NDArray[np.float64]]
        Event positions by type.
    event_frame_indices : dict[str, NDArray[np.int_]]
        Frame indices by type.
    event_times : dict[str, NDArray[np.float64]]
        Times of the events in the animation, in seconds, by type.
    colors : dict[str, str]
        Colors by type.
    markers : dict[str, str]
        Markers by type.
    size : float
        Marker size.
    decay_frames : int | None
        Decay frame count (None = cumulative, 0 = instant).
    border_color : str
        Border color.
    border_width : float
        Border width.
    opacity : float
        Base opacity for visible events.

    See Also
    --------
    EventOverlay : User-facing overlay configuration
    """

    event_positions: dict[str, NDArray[np.float64]]
    event_frame_indices: dict[str, NDArray[np.int_]]
    colors: dict[str, str]
    markers: dict[str, str]
    size: float
    decay_frames: int | None
    border_color: str
    border_width: float
    opacity: float
    event_times: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass
class VideoData:
    """Internal container for video overlay data on its own clock.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from VideoOverlay instances.

    Parameters
    ----------
    frame_indices : ndarray of shape (n_samples,), dtype int
        The reader frame shown from each of ``times``; -1 for none.
    reader : NDArray[np.uint8] or VideoReaderProtocol
        Video source. Either a pre-loaded array of shape (n_video_frames, height,
        width, 3) with dtype uint8, or a VideoReaderProtocol-compliant reader.
        Copies of the data share the reader; it is closed by its owner.
    transform_to_env : AffineND or None
        Transform from video pixel coordinates to environment coordinates.
        If None, the frame is stretched over ``env_bounds``.
    env_bounds : tuple of float
        Environment bounding box as (xmin, xmax, ymin, ymax) for positioning
        the video in the visualization.
    alpha : float
        Opacity of the video layer (0.0-1.0).
    z_order : {"below", "above"}
        Rendering order relative to the spatial field layer.
    times : ndarray of shape (n_samples,), dtype float64
        Times of the samples, in seconds.

    See Also
    --------
    VideoOverlay : User-facing overlay configuration
    VideoReaderProtocol : Interface for video readers

    Notes
    -----
    For pickle-safety with parallel rendering:

    - Pre-loaded arrays (NDArray[np.uint8]) are pickle-safe
    - _VideoReader instances implement pickle protocol (drops cache on pickle)
    """

    frame_indices: NDArray[np.int_]
    reader: NDArray[np.uint8] | VideoReaderProtocol
    transform_to_env: AffineND | None
    env_bounds: tuple[float, float, float, float]
    alpha: float
    z_order: Literal["below", "above"]
    _: KW_ONLY
    times: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Check the mapping from samples to reader frames."""
        indices = np.asarray(self.frame_indices)
        if indices.shape != (len(self.times),) or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError(
                "VideoData.frame_indices must be a 1-D integer array with one "
                "index per sample time."
            )
        # Adapter readers need not report their length.
        count = (
            len(self.reader)
            if isinstance(self.reader, np.ndarray)
            else getattr(self.reader, "n_frames", None)
        )
        if (indices < -1).any() or (count is not None and (indices >= count).any()):
            raise ValueError(
                "VideoData.frame_indices must contain valid reader frame indices "
                "or -1 for missing frames."
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> VideoData:
        # Readers may own decoder handles; copies share the reader.
        result = copy.copy(self)
        memo[id(self)] = result
        for item in dataclasses.fields(self):
            if item.name != "reader":
                setattr(
                    result, item.name, copy.deepcopy(getattr(self, item.name), memo)
                )
        return result

    def on_clock(self, clock: NDArray[np.float64]) -> VideoData:
        """The nearest video frame at each of ``clock``; -1 outside the video."""
        nearest = _find_nearest_indices(self.times, clock)
        indices = np.full(len(clock), -1, dtype=np.int_)
        indices[nearest >= 0] = self.frame_indices[nearest[nearest >= 0]]
        return _moved(self, times=clock, frame_indices=indices)

    def pixel_to_env(self, width: int, height: int) -> NDArray[np.float64]:
        """The matrix taking a video pixel ``(column, row, 1)`` to ``(x, y, 1)``.

        The first pixel's center is ``(0, 0)``. With a calibration this is its
        complete pixel-to-environment transform, rotation and reflection
        included; without one the frame is stretched upright over
        ``env_bounds``.

        Parameters
        ----------
        width, height : int
            Frame size, in pixels.

        Returns
        -------
        ndarray of float64, shape (3, 3)
        """
        if self.transform_to_env is not None:
            return np.asarray(self.transform_to_env.A, dtype=np.float64)
        xmin, xmax, ymin, ymax = self.env_bounds
        sx, sy = (xmax - xmin) / width, (ymax - ymin) / height
        # The first row at the top, and the frame's edges on the extent.
        return np.array(
            [[sx, 0.0, xmin + sx / 2], [0.0, -sy, ymax - sy / 2], [0.0, 0.0, 1.0]]
        )

    def get_frame(self, sample: int) -> NDArray[np.uint8] | None:
        """Get the video frame of one sample.

        Parameters
        ----------
        sample : int
            Index into ``times`` (0-based).

        Returns
        -------
        NDArray[np.uint8] | None
            RGB video frame of shape (height, width, 3) with dtype uint8,
            or None for a sample index out of bounds or mapped to -1.
        """
        if sample < 0 or sample >= len(self.frame_indices):
            return None
        video_frame_idx = self.frame_indices[sample]
        if video_frame_idx < 0:
            return None
        frame: NDArray[np.uint8] = self.reader[video_frame_idx]
        return frame


@dataclass
class TimeSeriesData:
    """Internal container for time series overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from TimeSeriesOverlay instances.

    Parameters
    ----------
    data : ndarray of shape (n_samples,), dtype float64
        Full-resolution time series values (no downsampling). NaN values
        create gaps in the rendered line.
    times : ndarray of shape (n_samples,), dtype float64
        Timestamps for each sample, in seconds.
    label : str
        Label for Y-axis or legend.
    color : str
        Line color (matplotlib color string).
    linewidth : float
        Line width in points.
    alpha : float
        Line opacity (0.0-1.0).
    group : str | None
        Group name for overlaying multiple variables.
    normalize : bool
        Whether data was normalized to [0, 1].
    show_cursor : bool
        Whether to show cursor line.
    cursor_color : str
        Cursor line color.
    global_vmin : float
        Minimum Y-axis value (global across all data).
    global_vmax : float
        Maximum Y-axis value (global across all data).

    Attributes
    ----------
    data : NDArray[np.float64]
        Time series values.
    times : NDArray[np.float64]
        Sample timestamps.
    label : str
        Display label.
    color : str
        Line color.
    linewidth : float
        Line width.
    alpha : float
        Line opacity.
    group : str | None
        Grouping key.
    normalize : bool
        Normalization flag.
    show_cursor : bool
        Cursor visibility.
    cursor_color : str
        Cursor color.
    global_vmin : float
        Y-axis minimum.
    global_vmax : float
        Y-axis maximum.

    See Also
    --------
    TimeSeriesOverlay : User-facing overlay configuration
    """

    # Full resolution data (no downsampling)
    data: NDArray[np.float64]
    times: NDArray[np.float64]

    # Display settings
    label: str
    color: str
    linewidth: float
    alpha: float
    group: str | None
    normalize: bool
    show_cursor: bool
    cursor_color: str

    # Y-axis limits
    global_vmin: float
    global_vmax: float


@dataclass
class SpikeRasterData:
    """Internal container for a spike raster, one row per unit.

    Created by the conversion pipeline from a SpikeRasterOverlay.

    Attributes
    ----------
    spike_times : list of ndarray of float64
        Each row's spike times, in seconds, bottom row first.
    unit_ids : ndarray, shape (n_units,)
        Each row's unit label.
    rates : ndarray of float64, shape (n_units, n_bins), or None
        Each row's rate map, in spikes per second.
    color : str
        Tick color.
    linewidth : float
        Tick width, in screen pixels.
    """

    spike_times: list[NDArray[np.float64]]
    unit_ids: NDArray[Any]
    rates: NDArray[np.float64] | None
    color: str
    linewidth: float


@dataclass
class EpochData:
    """Internal container for epochs: intervals of time.

    Created by the conversion pipeline from an EpochOverlay.

    Attributes
    ----------
    intervals : ndarray of float64, shape (n_epochs, 2)
        Start and stop of each epoch, in seconds, sorted and merged.
    label : str
        What an epoch is.
    color : str
        Shading color.
    alpha : float
        Shading opacity.
    """

    intervals: NDArray[np.float64]
    label: str
    color: str
    alpha: float


@dataclass
class ObjectVectorData(_SampledData):
    """Internal container for object-vector overlay data on its own clock.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from ObjectVectorOverlay instances.

    Parameters
    ----------
    object_positions : ndarray of shape (n_objects, n_dims), dtype float64
        Static positions of objects in environment coordinates. These positions
        remain fixed throughout the animation.
    animal_positions : ndarray of shape (n_samples, n_dims), dtype float64
        Animal positions at ``times``. NaN marks a missing sample.
    firing_rates : ndarray of shape (n_samples,), dtype float64, optional
        Firing rates at ``times`` for modulating line appearance. None if no
        modulation was specified. A rate is used only where the animal's
        position is valid.
    color : str
        Line color (matplotlib color string).
    linewidth : float
        Line width in pixels.
    show_objects : bool
        Whether to render object markers.
    object_marker : str
        Marker style for objects.
    object_size : float
        Object marker diameter in logical screen pixels.
    times : ndarray of shape (n_samples,), dtype float64
        Times of the rows, in seconds.
    interp : {"linear", "nearest"}
        How views sample the rows between observations.
    sample_valid, interval_valid : ndarray of bool, optional
        Support of the animal positions; see ``_SampledData``.
    rate_support : tuple of ndarray of bool, optional
        ``(sample_valid, interval_valid)`` of ``firing_rates``. A rate is
        valid only where the animal position is too. None accepts every
        finite rate at a valid position and every interval between them.

    See Also
    --------
    ObjectVectorOverlay : User-facing overlay configuration
    """

    object_positions: NDArray[np.float64]
    animal_positions: NDArray[np.float64]
    firing_rates: NDArray[np.float64] | None
    color: str
    linewidth: float
    show_objects: bool
    object_marker: str
    object_size: float
    _: KW_ONLY
    times: NDArray[np.float64]
    interp: Literal["linear", "nearest"] = "linear"
    # None accepts every finite row; __post_init__ always stores an array.
    sample_valid: NDArray[np.bool_] = None  # type: ignore[assignment]
    interval_valid: NDArray[np.bool_] = None  # type: ignore[assignment]

    # The full recording's finite peak rate; 0 until computed, then positive.
    _rate_peak: float = field(default=0.0, repr=False, compare=False)
    rate_support: tuple[NDArray[np.bool_], NDArray[np.bool_]] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Resolve the rows' support and the full-recording rate summary."""
        rows = {"animal_positions": self.animal_positions}
        masks: dict[str, tuple[Any, Any]] = {
            "animal_positions": (self.sample_valid, self.interval_valid)
        }
        if self.firing_rates is not None:
            rows["firing_rates"] = self.firing_rates
            masks["firing_rates"] = self.rate_support or (None, None)
        _check_rows(self, rows, masks)
        self.animal_positions, self.sample_valid, self.interval_valid = resolve_support(
            self.animal_positions, self.sample_valid, self.interval_valid
        )
        if self.firing_rates is not None:
            rate_valid, rate_intervals = self.rate_support or (None, None)
            _, *rate_support = resolve_support(
                self.firing_rates,
                self.sample_valid
                if rate_valid is None
                else rate_valid & self.sample_valid,
                rate_intervals,
            )
            self.rate_support = (rate_support[0], rate_support[1])
        if not self._rate_peak:
            from neurospatial.animation._style import _finite_rate_peak

            # One finite full-recording peak keeps the opacity scale stable
            # when a view selects a window.
            self._rate_peak = _finite_rate_peak(self.firing_rates)

    def support(
        self, name: str | None = None
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """``(sample_valid, interval_valid)`` of the positions or ``"firing_rates"``."""
        if name == "firing_rates":
            assert self.rate_support is not None
            return self.rate_support
        return self.sample_valid, self.interval_valid

    def on_clock(self, clock: NDArray[np.float64]) -> ObjectVectorData:
        """The animal positions, rates and their support sampled at ``clock``."""
        positions, sample_valid, interval_valid = resample(
            self.times,
            self.animal_positions,
            self.sample_valid,
            self.interval_valid,
            clock,
            interp=self.interp,
        )
        rates, rate_support = None, None
        if self.firing_rates is not None:
            assert self.rate_support is not None
            rates, *moved = resample(
                self.times,
                self.firing_rates,
                *self.rate_support,
                clock,
                interp=self.interp,
            )
            rate_support = (moved[0], moved[1])
        return _moved(
            self,
            times=clock,
            animal_positions=positions,
            firing_rates=rates,
            sample_valid=sample_valid,
            interval_valid=interval_valid,
            rate_support=rate_support,
        )


@dataclass
class OverlayData:
    """Container for all overlay data passed to animation backends.

    Aggregates all overlay types into a single container. Backends receive this
    object and render the appropriate overlay types. Pickle-ability is validated
    separately in the animation pipeline when needed (e.g., for parallel video
    rendering with n_workers > 1).

    Parameters
    ----------
    positions : list[PositionData], optional
        List of position overlays. Default is empty list.
    bodypart_sets : list[BodypartData], optional
        List of bodypart overlays. Default is empty list.
    head_directions : list[HeadDirectionData], optional
        List of head direction overlays. Default is empty list.
    videos : list[VideoData], optional
        List of video overlays. Default is empty list.
    events : list[EventData], optional
        List of event overlays. Default is empty list.
    timeseries : list[TimeSeriesData], optional
        List of time series overlays. Default is empty list.
    spike_rasters : list[SpikeRasterData], optional
        List of spike rasters. Default is empty list.
    epochs : list[EpochData], optional
        List of epoch overlays. Default is empty list.
    object_vectors : list[ObjectVectorData], optional
        List of object-vector overlays. Default is empty list.

    Notes
    -----
    This container is created by the conversion pipeline when an
    ``Animation`` is built with overlays. Users should not instantiate
    OverlayData directly.

    Observed values are checked for shape and for infinite values; NaN marks
    a missing observation and is kept. Sampled overlays keep their own
    ``times`` and carry the sample and interval support that backends use,
    so a backend never interpolates across missing tracking or a recording
    break.
    """

    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    videos: list[VideoData] = field(default_factory=list)
    events: list[EventData] = field(default_factory=list)
    timeseries: list[TimeSeriesData] = field(default_factory=list)
    spike_rasters: list[SpikeRasterData] = field(default_factory=list)
    epochs: list[EpochData] = field(default_factory=list)
    object_vectors: list[ObjectVectorData] = field(default_factory=list)

    # The list each internal data type goes in.
    _LISTS: ClassVar[dict[type, str]] = {}
    # Lists whose items keep samples at their own times, which a view
    # samples at the times it shows.
    _SAMPLED: ClassVar[tuple[str, ...]] = (
        "positions",
        "bodypart_sets",
        "head_directions",
        "object_vectors",
        "videos",
    )

    def add(self, item: Any) -> None:
        """Append ``item`` to the list of its type."""
        for kind, name in self._LISTS.items():
            if isinstance(item, kind):
                getattr(self, name).append(item)
                return
        raise TypeError(
            "convert_to_data() must return PositionData, BodypartData, "
            "HeadDirectionData, VideoData, EventData, TimeSeriesData, "
            "SpikeRasterData, EpochData or ObjectVectorData, got "
            f"{type(item).__name__}."
        )

    def sampled(self) -> Iterator[_SampledData | VideoData]:
        """Every item that keeps samples at its own ``times``."""
        for name in self._SAMPLED:
            yield from getattr(self, name)


OverlayData._LISTS.update(
    {
        PositionData: "positions",
        BodypartData: "bodypart_sets",
        HeadDirectionData: "head_directions",
        VideoData: "videos",
        EventData: "events",
        TimeSeriesData: "timeseries",
        SpikeRasterData: "spike_rasters",
        EpochData: "epochs",
        ObjectVectorData: "object_vectors",
    }
)


# =============================================================================
# Validation Functions (WHAT/WHY/HOW error messages)
# =============================================================================


def _validate_video_processing(
    original_size: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    downsample: int,
) -> None:
    width, height = original_size
    if crop is not None:
        if (
            len(crop) != 4
            or any(not isinstance(value, (int, np.integer)) for value in crop)
            or crop[0] < 0
            or crop[1] < 0
            or crop[2] <= 0
            or crop[3] <= 0
            or crop[0] + crop[2] > width
            or crop[1] + crop[3] > height
        ):
            raise ValueError(
                "crop must be (x, y, width, height) with positive size inside the original video frame."
            )
        width, height = crop[2:]
    if width // downsample < 1 or height // downsample < 1:
        raise ValueError(
            _format_error(
                "downsample leaves no video pixels.",
                fix="use a smaller downsample factor or a larger crop.",
            )
        )


def _validate_position_metadata(
    overlay: PositionOverlay | BodypartOverlay, env: Any
) -> None:
    for name in ("units", "frame"):
        declared, expected = getattr(overlay, name), getattr(env, name, None)
        # An overlay without a declaration is in the environment's coordinates.
        if not same_coordinate_declaration(
            name, declared, expected, undeclared_matches=True
        ):
            raise ValueError(
                _format_error(
                    f"{type(overlay).__name__} {name} {declared!r} does not match "
                    f"environment {name} {expected!r}.",
                    fix="explicitly convert the coordinates to the environment's "
                    "units and frame before constructing the overlay; "
                    "coordinates are never silently rescaled.",
                )
            )


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
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{name} must be a finite 1-D array of timestamps in seconds.")
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
            f"Fix: Sort the times array using np.argsort(), or if duplicates exist, "
            f"apply jitter or remove duplicates. Example:\n"
            f"  sorted_indices = np.argsort(times)\n"
            f"  times = times[sorted_indices]\n"
            f"  data = data[sorted_indices]"
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
            f"Fix: Reshape your data to 2D:\n"
            f"  data = data.reshape(-1, {expected_ndims})  # Flatten to 2D"
        )

    if actual_ndims != expected_ndims:
        raise ValueError(
            f"WHAT: Shape mismatch for '{name}'.\n"
            f"  Expected: (n_samples, {expected_ndims}) spatial dimensions\n"
            f"  Got: {data.shape} with {actual_ndims} spatial dimension(s)\n\n"
            f"WHY: Coordinate dimensionality must match the environment's spatial "
            f"dimensions (env.n_dims={expected_ndims}) for proper rendering.\n\n"
            f"Fix: Reformat your data to match the environment dimensions:\n"
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

    if overlap_start > overlap_end:
        # No overlap at all
        raise ValueError(
            f"WHAT: No temporal overlap between '{name}' and animation frames.\n"
            f"  Overlay times: [{overlay_min:.3f}, {overlay_max:.3f}]\n"
            f"  Frame times: [{frame_min:.3f}, {frame_max:.3f}]\n\n"
            f"WHY: Interpolation domain is disjoint - cannot align overlay data "
            f"to any animation frames.\n\n"
            f"Fix: Ensure overlapping time ranges:\n"
            f"  # Option 1: Adjust overlay timestamps to match animation\n"
            f"  overlay_times = overlay_times + {frame_min - overlay_min:.3f}\n\n"
            f"  # Option 2: Resample overlay data to animation time range\n"
            f"  # Option 3: Adjust frame_times to include overlay range"
        )

    # Calculate overlap percentage relative to frame times
    overlap_duration = overlap_end - overlap_start
    frame_duration = frame_max - frame_min
    overlap_pct = (overlap_duration / frame_duration) * 100 if frame_duration else 100.0

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
        # Calculate actual data ranges, ignoring NaN gaps from temporal
        # extrapolation (which would otherwise make every range read [nan, nan]).
        with warnings.catch_warnings():
            # An all-NaN column yields nan from nanmin/nanmax and a RuntimeWarning;
            # suppress it here (this is a diagnostic message, not a computation).
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data_mins = np.nanmin(data, axis=0)
            data_maxs = np.nanmax(data, axis=0)

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
            f"Fix: Confirm that overlay coordinates use the same coordinate system "
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
            f"Fix: Fix the bodypart names in your skeleton or data:\n"
            f"{suggestion_str}\n"
            f"  # Option 1: Add missing bodyparts to data dict\n"
            f"  # Option 2: Create skeleton with only available bodyparts:\n"
            f"  skeleton = Skeleton.from_edge_list(\n"
            f"      [(p1, p2) for p1, p2 in skeleton.edges\n"
            f"       if p1 in data and p2 in data]\n"
            f"  )"
        )


def _require_allocentric(env: Any, overlay: str, *, why: str) -> None:
    """Refuse an overlay drawn in arena coordinates on a polar environment.

    A polar environment's axes are ``(distance, angle)`` relative to the
    animal, so arena positions, headings and frames have no place on them.
    """
    require_cartesian(
        env.coordinate_system,
        overlay,
        why=why,
        fix="animate on the allocentric environment the animal moves in, or "
        "draw positions as (distance, angle) rows with PositionOverlay",
    )


def _validate_video_env(env: Any) -> None:
    """Validate that environment supports video overlay rendering.

    Video overlays require a 2D environment with finite dimension ranges
    to compute proper coordinate transforms.

    Parameters
    ----------
    env : Any
        Environment object with n_dims and dimension_ranges attributes.

    Raises
    ------
    ValueError
        If environment is not 2D or has non-finite dimension ranges.

    Notes
    -----
    WHAT: Video overlays require 2D environments with finite bounds.

    WHY: Video frames are 2D images that need to be transformed into
    environment coordinates. Non-finite bounds make it impossible to
    compute a valid affine transform from pixel space to environment space.

    Fix: Ensure your environment is 2D (e.g., from_samples with 2D positions)
    and has finite dimension_ranges (no infinite extents).

    Examples
    --------
    >>> env = Environment.from_samples(
    ...     np.random.rand(100, 2) * 100, bin_size=5
    ... )  # doctest: +SKIP
    >>> _validate_video_env(env)  # Should not raise  # doctest: +SKIP
    """
    # Check 2D requirement
    if env.n_dims != 2:
        raise ValueError(
            f"VideoOverlay requires a 2D environment.\n\n"
            f"WHAT: Environment has n_dims={env.n_dims}, expected n_dims=2.\n\n"
            f"WHY: Video frames are 2D images that must be transformed into "
            f"environment coordinates. This requires a 2D coordinate system.\n\n"
            f"Fix: Create a 2D environment:\n"
            f"  env = Environment.from_samples(positions_2d, bin_size=5.0)"
        )

    # Check finite dimension ranges
    dim_ranges = env.dimension_ranges
    if not np.all(np.isfinite(dim_ranges)):
        raise ValueError(
            f"VideoOverlay requires finite dimension ranges.\n\n"
            f"WHAT: Environment dimension_ranges contains non-finite values: "
            f"{dim_ranges}.\n\n"
            f"WHY: Computing the video-to-environment transform requires "
            f"finite bounds to establish the coordinate mapping.\n\n"
            f"Fix: Ensure your environment data has finite coordinates. "
            f"Check for infinite or NaN values in your position data."
        )


# =============================================================================
# Timeline and interpolation helpers (private)
# =============================================================================


def _validate_frame_times(
    frame_times: NDArray[np.float64],
    n_frames: int,
) -> NDArray[np.float64]:
    """Validate frame times array for animation.

    Validates that frame_times has the correct length and is strictly
    monotonically increasing. This function only validates - it does NOT
    synthesize frame_times from fps.

    Parameters
    ----------
    frame_times : NDArray[np.float64]
        Frame times array with shape (n_frames,). Must be strictly
        monotonically increasing.
    n_frames : int
        Expected number of frames. Used to validate frame_times length.

    Returns
    -------
    NDArray[np.float64]
        Validated frame times array (same as input).

    Raises
    ------
    ValueError
        If frame_times length does not match n_frames.
    ValueError
        If frame_times is not strictly monotonically increasing.

    Notes
    -----
    Frame times define the temporal sampling of the animation. They are used
    to align overlay data (which may have different sampling rates) to animation
    frames via interpolation.

    Examples
    --------
    Validate provided frame times:

    >>> custom_times = np.array([0.0, 0.5, 1.0, 1.5])
    >>> times = _validate_frame_times(custom_times, n_frames=4)
    >>> float(times[0]), float(times[-1])
    (0.0, 1.5)

    Single frame passes validation:

    >>> single = np.array([5.0])
    >>> _validate_frame_times(single, n_frames=1)
    array([5.])
    """
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if frame_times.ndim != 1 or not np.isfinite(frame_times).all():
        raise ValueError(
            "frame_times must be a finite 1-D array of timestamps in seconds."
        )
    # Validate length
    if len(frame_times) != n_frames:
        raise ValueError(
            f"frame_times length ({len(frame_times)}) must match n_frames ({n_frames})"
        )

    # Check strict monotonicity (only if more than one frame)
    if len(frame_times) > 1:
        diffs = np.diff(frame_times)
        if not np.all(diffs > 0):
            n_bad = int(np.sum(diffs <= 0))
            raise ValueError(
                "frame_times must be strictly monotonically increasing. "
                f"Found {n_bad} non-increasing intervals."
            )

    return frame_times


def _sample_clock(
    times: NDArray[np.float64] | None,
    frame_times: NDArray[np.float64],
    n_frames: int,
    n_samples: int,
    *,
    name: str,
) -> NDArray[np.float64]:
    """The clock of an overlay's own samples: ``times``, or the frame clock.

    Parameters
    ----------
    times : ndarray of shape (n_samples,) or None
        The overlay's timestamps, in seconds. None means one sample per
        animation frame.
    frame_times : ndarray of shape (n_frames,)
        The animation clock.
    n_frames : int
        Number of animation frames.
    n_samples : int
        Number of the overlay's samples.
    name : str
        Overlay name for error messages (e.g., "PositionOverlay").

    Raises
    ------
    ValueError
        If ``times`` is not strictly increasing, does not have one timestamp
        per sample or does not overlap ``frame_times``; or, without
        ``times``, if there is not one sample per frame.
    """
    frame_times = _validate_frame_times(frame_times, n_frames)
    if times is None:
        if n_samples != n_frames:
            raise ValueError(
                f"{name} data length ({n_samples}) does not match n_frames "
                f"({n_frames}). When times=None, data must have exactly n_frames samples."
            )
        return frame_times
    times = np.asarray(times, dtype=np.float64)
    _validate_monotonic_time(times, name=f"{name}.times")
    if len(times) != n_samples or not len(times):
        raise ValueError(
            f"{name}.times must have one timestamp per nonempty data sample."
        )
    _validate_temporal_alignment(times, frame_times, name=name)
    return times


def _find_nearest_indices(
    t_src: NDArray[np.float64],
    t_query: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Find nearest source indices for query times.

    For each query time, finds the index of the nearest source time.
    Returns -1 for query times outside the source time range.

    Parameters
    ----------
    t_src : NDArray[np.float64]
        Source timestamps with shape (n_samples,). Must be monotonically increasing.
    t_query : NDArray[np.float64]
        Query timestamps with shape (n_queries,).

    Returns
    -------
    NDArray[np.int_]
        Array of indices into t_src for each query time. Shape (n_queries,).
        Values are -1 for query times outside [t_src.min(), t_src.max()].

    Notes
    -----
    For query times exactly at the midpoint between two source times, the
    earlier (left) neighbor is chosen.

    Uses searchsorted for O(n_query log n_source) complexity.

    Examples
    --------
    >>> t_src = np.array([0.0, 1.0, 2.0])
    >>> t_query = np.array([0.4, 1.4, 3.0])
    >>> _find_nearest_indices(t_src, t_query)
    array([ 0,  1, -1])
    """
    # Handle empty query
    if len(t_query) == 0:
        return np.array([], dtype=np.int_)

    # Handle empty source (no valid indices possible)
    if len(t_src) == 0:
        return np.full(len(t_query), -1, dtype=np.int_)

    # Initialize result with -1 (out-of-range marker)
    result = np.full(len(t_query), -1, dtype=np.int_)

    # Restrict to query times within source range
    t_min = float(t_src.min())
    t_max = float(t_src.max())
    valid_mask = (t_query >= t_min) & (t_query <= t_max)

    if not np.any(valid_mask):
        return result

    t_valid = t_query[valid_mask]

    # Handle single-point source case
    if len(t_src) == 1:
        result[valid_mask] = 0  # Only one index possible
        return result

    # Use searchsorted to find nearest indices
    idx_right = np.searchsorted(t_src, t_valid, side="left")
    idx_right = np.clip(idx_right, 1, len(t_src) - 1)
    idx_left = idx_right - 1

    t_left = t_src[idx_left]
    t_right = t_src[idx_right]

    # Choose neighbor with smaller absolute time difference
    # For ties (exactly at midpoint), choose left (earlier) neighbor
    choose_right = (t_valid - t_left) > (t_right - t_valid)
    nearest_indices = np.where(choose_right, idx_right, idx_left)

    result[valid_mask] = nearest_indices
    return result


# =============================================================================
# Conversion Funnel
# =============================================================================


# Overlays with samples of their own, which a backend can show at their own
# rate.
_OWN_SAMPLE_TYPES = (
    PositionData,
    BodypartData,
    HeadDirectionData,
    VideoData,
    ObjectVectorData,
)


def _reindex_events(events: EventData, clock: NDArray[np.float64]) -> None:
    """Clip owned event records and remap their indices onto a validated clock."""
    for name, times in events.event_times.items():
        keep = (times >= clock[0]) & (times <= clock[-1])
        events.event_times[name] = times[keep].copy()
        events.event_positions[name] = events.event_positions[name][keep].copy()
        events.event_frame_indices[name] = np.clip(
            np.searchsorted(clock, times[keep]), 0, len(clock) - 1
        ).astype(np.int_)


def _check_adapter_clock(
    item: Any, frame_times: NDArray[np.float64], *, name: str
) -> None:
    """Check the sample clock a custom adapter's sampled data states.

    Built-in overlays check their own ``times`` when they convert.
    """
    if isinstance(item, (_SampledData, VideoData)):
        times = np.asarray(item.times, dtype=np.float64)
        if not len(times):
            raise ValueError(f"{type(item).__name__}.times must be nonempty.")
        _validate_frame_times(times, len(times))
        _validate_temporal_alignment(times, frame_times, name=name)


def _check_spatial_dimensions(item: Any, env: Any, *, name: str) -> None:
    """Check converted data against the environment's dimensions.

    Built-in overlays and custom adapters share this rule: heading vectors
    have one coordinate per environment dimension, and an angle names a
    direction only in a plane.
    """
    if not isinstance(item, HeadDirectionData):
        return
    if item.data.ndim == 2:
        _validate_shape(item.data, env.n_dims, name=f"{name}.headings")
    elif env.n_dims == 3:
        raise ValueError(
            _format_error(
                "3D head directions require (n_samples, 3) unit vectors.",
                why="Why: angles alone do not specify a direction in 3D.",
                fix="pass each heading as a unit vector in the environment's "
                "(x, y, z) coordinates.",
            )
        )


def _convert_overlays_to_data(
    overlays: list[OverlayProtocol],
    frame_times: NDArray[np.float64],
    n_frames: int,
    env: Any,
) -> tuple[OverlayData, _OwnedReaders]:
    """Convert overlay configurations to internal data on their own clocks.

    This is the main conversion pipeline that:
    1. Calls each overlay's ``convert_to_data()`` method with the scene clock
    2. Checks a custom adapter's sample clock against the scene clock
    3. Checks every result's headings against the environment's dimensions
    4. Collects the results, which it owns, into one OverlayData container
    5. Refuses overlays the scene renderer cannot draw in the environment's
       dimensions (headings and video in 1D)

    Parameters
    ----------
    overlays : list[OverlayProtocol]
        List of overlay configurations to convert. Can be empty or contain
        multiple instances of any overlay type implementing ``OverlayProtocol``.
        Custom overlays must implement the ``convert_to_data()`` method.
    frame_times : NDArray[np.float64]
        The scene clock, in seconds, with shape (n_frames,). Overlays
        without ``times`` have one sample per entry.
    n_frames : int
        Number of animation frames. Used for validation.
    env : Any
        Environment object with ``n_dims`` and ``dimension_ranges`` attributes.
        Used for validating overlay coordinate dimensions and bounds.

    Returns
    -------
    OverlayData
        One list per internal data type. Sampled data keeps each overlay's
        own samples, with their times in ``times``; events are indexed on
        ``frame_times``.
    _OwnedReaders
        The video readers the built-in overlays opened, which the caller
        closes. Readers a custom adapter returns stay the adapter's.

    Raises
    ------
    ValueError
        If any validation fails:
        - Non-monotonic timestamps in overlay.times
        - Inf values in observed overlay data (NaN marks missing observations)
        - Shape mismatch between overlay and environment dimensions
        - Skeleton references missing bodypart names
        - No temporal overlap between overlay and frame times
        - Headings or video in a 1D environment
    TypeError
        If overlay does not implement ``OverlayProtocol``, or if
        ``convert_to_data()`` returns a type other than the nine internal
        data types listed by ``OverlayProtocol.convert_to_data``.

    Notes
    -----
    An overlay without ``times`` has one sample per entry of
    ``frame_times``, including nonuniform frame intervals. Samples outside
    the overlay's own time range are missing (NaN) wherever they are shown,
    and a temporal overlap below 50% warns.

    Custom overlays can be created by implementing ``OverlayProtocol``; its
    ``convert_to_data()`` must return one of the internal data types.

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
    >>> overlay = PositionOverlay(positions=trajectory, times=times)
    >>>
    >>> # Convert
    >>> frame_times = np.array([0.0, 2.0])
    >>> overlay_data, readers = _convert_overlays_to_data(
    ...     overlays=[overlay], frame_times=frame_times, n_frames=2, env=env
    ... )
    >>> overlay_data.positions[0].data.shape
    (3, 2)
    >>> readers.close()
    """
    frame_times = _validate_frame_times(frame_times, n_frames)
    overlay_data = OverlayData()
    with ExitStack() as cleanup:
        readers = _OwnedReaders()
        cleanup.callback(readers.close)
        for overlay in overlays:
            if not isinstance(overlay, OverlayProtocol):
                raise TypeError(
                    f"Overlay must implement OverlayProtocol, got {type(overlay).__name__}. "
                    "Ensure your overlay has a 'convert_to_data()' method."
                )
            item = overlay.convert_to_data(frame_times, n_frames, env)
            # A video overlay with a file source opened its reader for us.
            if isinstance(overlay, VideoOverlay) and not isinstance(
                overlay.source, np.ndarray
            ):
                readers.add(cast("VideoData", item).reader)
            if not isinstance(overlay, _Overlay):
                _check_adapter_clock(item, frame_times, name=type(overlay).__name__)
            _check_spatial_dimensions(item, env, name=type(overlay).__name__)
            # Own the converted metadata; copies share video readers.
            item = copy.deepcopy(item)
            if isinstance(item, EventData):
                # Events keep their own times but are indexed on the scene clock.
                _reindex_events(item, frame_times)
            overlay_data.add(item)

        if overlay_data.head_directions and len(overlay_data.positions) > 1:
            raise ValueError(
                "HeadDirectionOverlay cannot tell which of the "
                f"{len(overlay_data.positions)} PositionOverlays it starts from. "
                "Heading arrows are drawn from the animal's position, and "
                "several position overlays give no single position. Pass exactly "
                "one PositionOverlay with heading overlays (the tracked point "
                "whose heading you measured), or omit PositionOverlays to draw "
                "the heading as a compass arrow at the arena center."
            )
        from neurospatial.animation._capabilities import require_scene_overlays

        require_scene_overlays(overlay_data, env.n_dims)
        cleanup.pop_all()
    return overlay_data, readers


def _overlay_arrays(data: Any) -> Iterator[np.ndarray]:
    """Every array held by overlay data, through dataclasses and containers."""
    if isinstance(data, np.ndarray):
        yield data
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        for item in dataclasses.fields(data):
            if item.name != "reader":
                yield from _overlay_arrays(getattr(data, item.name))
    elif isinstance(data, dict):
        for value in data.values():
            yield from _overlay_arrays(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _overlay_arrays(value)


def _freeze_overlay_arrays(data: OverlayData) -> None:
    """Make prepared overlay arrays read-only so realigned views can share them."""
    for array in _overlay_arrays(data):
        array.flags.writeable = False


def _realign_prepared_overlays(
    prepared: OverlayData,
    frame_times: NDArray[np.float64],
) -> OverlayData:
    """Sample prepared overlays at ``frame_times``, borrowing video readers.

    Adapters run only when the scene is prepared. Sampling their resolved
    data keeps the original observation support, and copies share the
    prepared readers, which stay owned by the scene. Sampled data gets
    ``times = frame_times``; read-only prepared arrays are shared and every
    container is independent.
    """
    clock = _validate_frame_times(frame_times, len(frame_times))
    memo: dict[int, Any] = {
        id(array): array
        for array in _overlay_arrays(prepared)
        if not array.flags.writeable
    }
    result = copy.deepcopy(prepared, memo)
    for name in OverlayData._SAMPLED:
        setattr(result, name, [item.on_clock(clock) for item in getattr(result, name)])
    for events in result.events:
        _reindex_events(events, clock)
    return result
