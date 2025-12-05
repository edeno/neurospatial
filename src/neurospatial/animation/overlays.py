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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.animation._video_io import VideoReaderProtocol
    from neurospatial.animation.skeleton import Skeleton
    from neurospatial.transforms import Affine2D, VideoCalibration


# =============================================================================
# Overlay Protocol: Enables custom overlay extensions
# =============================================================================


@runtime_checkable
class OverlayProtocol(Protocol):
    """Protocol for animation overlays, enabling custom extensions.

    Users can create custom overlays by implementing this protocol.
    The ``convert_to_data()`` method is called during animation to transform
    user-facing overlay configuration into internal data representation.

    Attributes
    ----------
    times : NDArray[np.float64] | None
        Timestamps for overlay data samples, in seconds. If None, samples are
        assumed uniformly spaced at the animation fps rate.
    interp : {"linear", "nearest"}
        Interpolation method for aligning overlay to animation frames.

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
        from neurospatial.animation import PositionData


        @dataclass
        class MyCustomOverlay:
            data: NDArray[np.float64]
            times: NDArray[np.float64] | None = None
            interp: Literal["linear", "nearest"] = "linear"
            custom_attr: str = "default"

            def convert_to_data(
                self,
                frame_times: NDArray[np.float64],
                n_frames: int,
                env: Any,
            ) -> PositionData:
                # If times provided, interpolate to frame times
                if self.times is not None:
                    # Use numpy.interp for each dimension
                    aligned = np.column_stack(
                        [
                            np.interp(frame_times, self.times, self.data[:, d])
                            for d in range(self.data.shape[1])
                        ]
                    )
                else:
                    aligned = self.data

                return PositionData(
                    data=aligned,
                    color="green",
                    size=10.0,
                    trail_length=None,
                )
    """

    times: NDArray[np.float64] | None
    interp: Literal["linear", "nearest"]

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
        | ObjectVectorData
    ):
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
        PositionData | BodypartData | HeadDirectionData | VideoData | EventData | TimeSeriesData | ObjectVectorData
            Internal data container aligned to frame times.

        Notes
        -----
        Implementations should:
        1. Validate overlay data (finite values, correct shape)
        2. Align data to frame_times via interpolation if times provided
        3. Return appropriate internal data type
        """
        ...


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
        Position coordinates in **environment (x, y) format**. Each row is a
        position at a time point. Dimensionality must match the environment
        (env.n_dims).

        **Important**: Use the same coordinate system as your tracking data
        and Environment. Do NOT pre-convert to napari (row, col) format - the
        animation system handles coordinate transformation automatically,
        including axis swap and Y-axis inversion for proper display.
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
            Internal data container aligned to frame times.
        """
        # Validate position data
        _validate_finite_values(self.data, name="PositionOverlay.data")
        _validate_shape(self.data, env.n_dims, name="PositionOverlay.data")

        # Align to frame times (validates times if provided)
        aligned_data = _align_to_frame_times(
            self.data,
            self.times,
            frame_times,
            n_frames,
            self.interp,
            name="PositionOverlay",
        )

        # Validate bounds (warning only)
        _validate_bounds(
            aligned_data,
            env.dimension_ranges,
            name="PositionOverlay.data",
            threshold=0.1,
        )

        return PositionData(
            data=aligned_data,
            color=self.color,
            size=self.size,
            trail_length=self.trail_length,
        )


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

        **Important**: Provide coordinates in **environment (x, y) format** -
        the same coordinate system as your tracking data and Environment.
        Do NOT pre-convert to napari (row, col) format. The animation system
        handles coordinate transformation automatically.
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
            Internal data container aligned to frame times.
        """
        # Validate skeleton consistency first
        bodypart_names = list(self.data.keys())
        _validate_skeleton_consistency(
            self.skeleton, bodypart_names, name="BodypartOverlay.skeleton"
        )

        # Align each bodypart separately
        aligned_bodyparts: dict[str, NDArray[np.float64]] = {}

        for part_name, part_data in self.data.items():
            # Validate each bodypart
            _validate_finite_values(
                part_data, name=f"BodypartOverlay.data['{part_name}']"
            )
            _validate_shape(
                part_data, env.n_dims, name=f"BodypartOverlay.data['{part_name}']"
            )

            # Align to frame times (validates times if provided)
            aligned_part = _align_to_frame_times(
                part_data,
                self.times,
                frame_times,
                n_frames,
                self.interp,
                name=f"BodypartOverlay.data['{part_name}']",
            )

            # Validate bounds (warning only)
            _validate_bounds(
                aligned_part,
                env.dimension_ranges,
                name=f"BodypartOverlay.data['{part_name}']",
                threshold=0.1,
            )

            aligned_bodyparts[part_name] = aligned_part

        return BodypartData(
            bodyparts=aligned_bodyparts,
            skeleton=self.skeleton,
            colors=self.colors,
        )


@dataclass
class HeadDirectionOverlay:
    """Heading direction visualization rendered as a line from head to indicator.

    Represents directional heading data as either angles (in radians) or unit
    vectors. Rendered as a line segment from head position to direction indicator.
    Commonly used for visualizing animal orientation during navigation.

    When paired with a PositionOverlay, creates a clear visualization:
    - Head position marker (from PositionOverlay)
    - Direction line extending from head in heading direction

    Parameters
    ----------
    data : ndarray of shape (n_samples,) or (n_samples, n_dims), dtype float64
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
        uniformly spaced at the animation fps rate. Must be monotonically
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

    Attributes
    ----------
    data : NDArray[np.float64]
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

    **Performance Note**: This overlay uses napari's Tracks layer for efficient
    rendering with time-based filtering, achieving smooth 30 FPS playback even
    with 40k+ frames.

    Examples
    --------
    Head direction as angles (2D):

    >>> import numpy as np
    >>> angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    >>> overlay = HeadDirectionOverlay(data=angles)
    >>> overlay.data.shape
    (4,)
    >>> overlay.color
    'hsv'

    Head direction as unit vectors (2D):

    >>> vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    >>> overlay = HeadDirectionOverlay(data=vectors, color="twilight")
    >>> overlay.data.shape
    (3, 2)

    Custom styling with visible offset and size:

    >>> overlay = HeadDirectionOverlay(
    ...     data=angles, color="purple", length=20.0, width=4.0
    ... )
    >>> overlay.length
    20.0
    >>> overlay.width
    4.0

    With timestamps for temporal alignment:

    >>> times = np.array([0.0, 0.5, 1.0, 1.5])
    >>> overlay = HeadDirectionOverlay(data=angles, times=times)
    >>> overlay.times.shape
    (4,)
    """

    data: NDArray[np.float64]
    times: NDArray[np.float64] | None = None
    color: str = "hsv"
    length: float = 15.0
    width: float = 3.0
    interp: Literal["linear", "nearest"] = "linear"

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
            Internal data container aligned to frame times.
        """
        # Validate head direction data
        _validate_finite_values(self.data, name="HeadDirectionOverlay.data")

        # For head direction, validate shape only if 2D (vectors)
        if self.data.ndim == 2:
            _validate_shape(self.data, env.n_dims, name="HeadDirectionOverlay.data")

        # Align to frame times (validates times if provided)
        aligned_data = _align_to_frame_times(
            self.data,
            self.times,
            frame_times,
            n_frames,
            self.interp,
            name="HeadDirectionOverlay",
        )

        return HeadDirectionData(
            data=aligned_data,
            color=self.color,
            length=self.length,
            width=self.width,
        )


@dataclass
class EventOverlay:
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
    colors : str | dict[str, str] | None, optional
        Colors for event markers:

        - str: Single color for all event types (e.g., "red")
        - dict: Mapping event names to colors
        - None: Auto-assign from perceptually distinct colormap (tab10)

        Default is None.
    size : float, optional
        Marker size in points. Default is 8.0.
    decay_frames : int | None, optional
        Number of frames over which event markers persist and fade.

        - None: Events appear only on their exact frame (instant)
        - 0: Same as None (instant)
        - >0: Events persist for N frames with decaying opacity

        Default is None (instant, no decay).
    markers : str | dict[str, str] | None, optional
        Marker style(s) for matplotlib/video backend ('o', 's', '^', 'v', 'd').

        - str: Single marker for all event types
        - dict: Mapping event names to markers
        - None: Use 'o' (circle) for all

        Napari uses circles regardless. Default is None.
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
    SpikeOverlay : Convenience alias for EventOverlay for neural spike visualization.
    PositionOverlay : Trajectory visualization overlay.

    Notes
    -----
    **Position Modes:**

    Mode A (explicit positions) is used when events occur at known fixed locations,
    independent of animal position. Examples: reward delivery at feeder, zone entry
    at boundary, stimulus presentation at screen location.

    Mode B (trajectory interpolation) is used when events occur "at the animal",
    and position should be interpolated from the animal's trajectory at event time.
    Examples: neural spikes, licks, lever presses.

    The two modes are mutually exclusive - provide either ``event_positions`` OR
    ``positions`` + ``position_times``, but not both.

    **Coordinate Convention:**

    Event positions use environment coordinates (x, y), the same as tracking data.
    The animation system automatically transforms to napari pixel space.

    Examples
    --------
    Reward delivery at fixed feeder location (explicit positions)::

        events = EventOverlay(
            event_times=reward_times,
            event_positions=np.array([[50.0, 25.0]]),  # Feeder location (broadcast)
            colors="gold",
            markers="s",
        )

    Zone entry events at zone boundaries (explicit positions per event)::

        events = EventOverlay(
            event_times=zone_entry_times,
            event_positions=zone_entry_locations,  # (n_events, 2)
            colors="cyan",
        )

    Neural spikes at animal position (trajectory interpolation)::

        events = EventOverlay(
            event_times=spike_times,  # Shape: (n_spikes,)
            positions=trajectory,  # Shape: (n_samples, 2)
            position_times=timestamps,  # Shape: (n_samples,)
            colors="red",
            size=10.0,
        )
        env.animate_fields(fields, overlays=[events])

    Multiple neurons with auto-colors::

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

    # Protocol attributes (set to None - not used by EventOverlay directly)
    times: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and normalize event_times to dict format."""
        # Normalize event_times to dict format
        if isinstance(self.event_times, np.ndarray):
            object.__setattr__(self, "event_times", {"event": self.event_times})

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
                "HOW: Provide EITHER event_positions OR (positions AND position_times), "
                "not both."
            )

        if not has_explicit and not has_trajectory:
            raise ValueError(
                "WHAT: No position data provided for events.\n\n"
                "WHY: EventOverlay needs to know where to display each event.\n\n"
                "HOW: Provide either:\n"
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
                    "HOW: Provide positions array with shape (n_samples, n_dims)."
                )
            if self.position_times is None:
                raise ValueError(
                    "WHAT: positions provided without position_times.\n\n"
                    "WHY: Trajectory mode requires both the trajectory array (positions) "
                    "and its timestamps (position_times) for temporal interpolation.\n\n"
                    "HOW: Provide position_times array with shape (n_samples,)."
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
                        "HOW: Provide either:\n"
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
                    "HOW: Flatten the array or ensure shape is (n_events,)."
                )
            if not np.all(np.isfinite(times_arr)):
                n_invalid = np.sum(~np.isfinite(times_arr))
                raise ValueError(
                    f"WHAT: Event times for '{name}' contains {n_invalid} "
                    "non-finite values (NaN or Inf).\n\n"
                    "WHY: Event timestamps must be finite numbers for temporal "
                    "alignment.\n\n"
                    "HOW: Remove or interpolate NaN/Inf values before creating overlay."
                )

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
        # Type narrowing: __post_init__ normalizes these to dict format
        assert isinstance(self.event_times, dict)  # Always dict after __post_init__

        event_positions_out: dict[str, NDArray[np.float64]] = {}
        event_frame_indices_out: dict[str, NDArray[np.int_]] = {}

        # Get frame time range for filtering
        frame_t_min = frame_times[0] if len(frame_times) > 0 else 0.0
        frame_t_max = frame_times[-1] if len(frame_times) > 0 else 0.0

        # Process each event type
        for event_name, event_times_arr in self.event_times.items():
            # Filter events to those within frame time range
            in_range_mask = (event_times_arr >= frame_t_min) & (
                event_times_arr <= frame_t_max
            )

            # Also check position_times range if in trajectory mode
            if self.positions is not None and self.position_times is not None:
                pos_t_min = self.position_times[0]
                pos_t_max = self.position_times[-1]
                pos_range_mask = (event_times_arr >= pos_t_min) & (
                    event_times_arr <= pos_t_max
                )
                combined_mask = in_range_mask & pos_range_mask

                # Warn if events are outside range
                n_outside = np.sum(~combined_mask)
                if n_outside > 0:
                    import warnings

                    warnings.warn(
                        f"EventOverlay '{event_name}': {n_outside} events are outside "
                        f"the valid time range and will be excluded. "
                        f"Frame range: [{frame_t_min:.3f}, {frame_t_max:.3f}], "
                        f"Position range: [{pos_t_min:.3f}, {pos_t_max:.3f}].",
                        UserWarning,
                        stacklevel=2,
                    )
                in_range_mask = combined_mask
            else:
                # Explicit positions mode - only check frame range
                n_outside = np.sum(~in_range_mask)
                if n_outside > 0:
                    import warnings

                    warnings.warn(
                        f"EventOverlay '{event_name}': {n_outside} events are outside "
                        f"the frame time range [{frame_t_min:.3f}, {frame_t_max:.3f}] "
                        "and will be excluded.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Get filtered event times
            filtered_times = event_times_arr[in_range_mask]

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
                if self.interp == "linear":
                    positions_for_events = np.column_stack(
                        [
                            np.interp(
                                filtered_times,
                                self.position_times,
                                self.positions[:, d],
                            )
                            for d in range(self.positions.shape[1])
                        ]
                    )
                else:  # nearest
                    # Find nearest position sample for each event
                    indices = np.searchsorted(self.position_times, filtered_times)
                    indices = np.clip(indices, 0, len(self.position_times) - 1)

                    # Check if previous index is closer
                    prev_indices = np.maximum(indices - 1, 0)
                    dist_to_current = np.abs(
                        filtered_times - self.position_times[indices]
                    )
                    dist_to_prev = np.abs(
                        filtered_times - self.position_times[prev_indices]
                    )
                    use_prev = dist_to_prev < dist_to_current
                    final_indices = np.where(use_prev, prev_indices, indices)

                    positions_for_events = self.positions[final_indices]

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
SpikeOverlay = EventOverlay
"""Convenience alias for EventOverlay for neural spike visualization.

See EventOverlay for full documentation.
"""


@dataclass
class TimeSeriesOverlay:
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
    window_seconds : float, optional
        Total time window to display, centered on current frame.
        E.g., 2.0 shows ±1 second. Must be positive. Default is 2.0.
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
        **Note**: Constant data (min == max) is normalized to all zeros.
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
    interp : {"linear", "nearest"}, optional
        Interpolation method for computing the value at the current cursor time.
        Used when displaying cursor value (e.g., "Speed: 45.3 cm/s" tooltip).

        - "linear": Linearly interpolate between neighboring samples
        - "nearest": Use nearest sample value

        Default is "linear".
    update_mode : {"live", "on_pause", "manual"}, optional
        Controls when the time series dock widget updates during playback.

        - "live" (default): Update on every frame change, throttled to 20 Hz max.
          Best for real-time visualization during playback.
        - "on_pause": Only update when playback pauses. Best for performance
          when the time series isn't needed during playback.
        - "manual": Never auto-update. Updates only via explicit API call.
          Useful for custom update logic or debugging.

        Default is "live".
    playback_throttle_hz : float, optional
        Maximum update frequency (Hz) during playback. Lower values improve
        performance but reduce visual responsiveness. Default is 10.0.
    scrub_throttle_hz : float, optional
        Maximum update frequency (Hz) when scrubbing (paused). Higher values
        make scrubbing feel more responsive. Default is 20.0.

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
    window_seconds : float
        Time window width in seconds.
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
    interp : {"linear", "nearest"}
        Interpolation method for cursor value.
    update_mode : {"live", "on_pause", "manual"}
        Update mode for dock widget during playback.
    playback_throttle_hz : float
        Maximum update frequency during playback.
    scrub_throttle_hz : float
        Maximum update frequency when scrubbing.

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

    **Group conflicts**: When multiple overlays share the same ``group`` but have
    different ``window_seconds``, ``vmin``, or ``vmax``, a warning is emitted and
    the first overlay's values are used ("first wins" strategy).

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
    ...     window_seconds=3.0,
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
    window_seconds: float = 2.0
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

    # Interpolation for cursor value
    interp: Literal["linear", "nearest"] = "linear"

    # Update mode for dock widget during playback
    update_mode: Literal["live", "on_pause", "manual"] = "live"

    # Throttle settings for performance optimization
    playback_throttle_hz: float = 10.0
    scrub_throttle_hz: float = 20.0

    def __post_init__(self) -> None:
        """Validate inputs after initialization."""
        # Validate data is 1D
        if self.data.ndim != 1:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data must be 1D "
                f"(got shape {self.data.shape}).\n\n"
                "WHY: Time series visualization expects a single variable over time.\n\n"
                "HOW: Flatten the array or select a single column/variable."
            )

        # Validate times is 1D
        if self.times.ndim != 1:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.times must be 1D "
                f"(got shape {self.times.shape}).\n\n"
                "WHY: Timestamps must be a 1D array of time values.\n\n"
                "HOW: Flatten the times array or ensure shape is (n_samples,)."
            )

        # Validate data and times have same length
        if len(self.data) != len(self.times):
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data and times have different lengths "
                f"({len(self.data)} vs {len(self.times)}).\n\n"
                "WHY: Each data point must have a corresponding timestamp.\n\n"
                "HOW: Ensure both arrays have the same number of elements."
            )

        # Validate data has at least one sample
        if len(self.data) == 0:
            raise ValueError(
                "WHAT: TimeSeriesOverlay.data must have at least 1 sample (got 0).\n\n"
                "WHY: Time series visualization requires at least one data point.\n\n"
                "HOW: Provide non-empty data array."
            )

        # Validate times are finite (no NaN or Inf)
        if not np.all(np.isfinite(self.times)):
            n_invalid = np.sum(~np.isfinite(self.times))
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.times contains {n_invalid} "
                "non-finite values (NaN or Inf).\n\n"
                "WHY: Timestamps must be finite numbers for window extraction.\n\n"
                "HOW: Remove or fix NaN/Inf values in times array."
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
                "HOW: Sort the times array or remove duplicates."
            )

        # Validate window_seconds is positive
        if self.window_seconds <= 0:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.window_seconds must be positive "
                f"(got {self.window_seconds}).\n\n"
                "WHY: The time window must have a positive duration.\n\n"
                "HOW: Use a positive value like window_seconds=2.0."
            )

        # Validate alpha in [0, 1]
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.alpha must be in [0, 1] "
                f"(got {self.alpha}).\n\n"
                "WHY: Alpha is an opacity value between fully transparent (0) "
                "and fully opaque (1).\n\n"
                "HOW: Use a value between 0.0 and 1.0."
            )

        # Validate linewidth is positive
        if self.linewidth <= 0:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.linewidth must be positive "
                f"(got {self.linewidth}).\n\n"
                "WHY: Line width must be a positive value for rendering.\n\n"
                "HOW: Use a positive value like linewidth=1.0."
            )

        # Validate no Inf in data (NaN is allowed for gaps)
        if np.any(np.isinf(self.data)):
            n_inf = np.sum(np.isinf(self.data))
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.data contains {n_inf} Inf values.\n\n"
                "WHY: Inf values cannot be rendered. NaN is allowed for gaps, "
                "but Inf is not supported.\n\n"
                "HOW: Replace Inf values with NaN or finite values."
            )

        # Validate update_mode
        valid_modes = ("live", "on_pause", "manual")
        if self.update_mode not in valid_modes:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.update_mode must be one of "
                f"{valid_modes} (got '{self.update_mode}').\n\n"
                "WHY: update_mode controls when the time series dock updates:\n"
                "  - 'live': Update on every frame change (throttled to 20 Hz)\n"
                "  - 'on_pause': Only update when playback pauses\n"
                "  - 'manual': Never auto-update, only via explicit API call\n\n"
                "HOW: Use one of: 'live', 'on_pause', or 'manual'."
            )

        # Validate playback_throttle_hz is positive
        if self.playback_throttle_hz <= 0:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.playback_throttle_hz must be positive "
                f"(got {self.playback_throttle_hz}).\n\n"
                "WHY: The throttle frequency controls how often the time series "
                "updates during playback. A positive value is required.\n\n"
                "HOW: Use a positive value like playback_throttle_hz=10.0."
            )

        # Validate scrub_throttle_hz is positive
        if self.scrub_throttle_hz <= 0:
            raise ValueError(
                f"WHAT: TimeSeriesOverlay.scrub_throttle_hz must be positive "
                f"(got {self.scrub_throttle_hz}).\n\n"
                "WHY: The throttle frequency controls how often the time series "
                "updates during scrubbing. A positive value is required.\n\n"
                "HOW: Use a positive value like scrub_throttle_hz=20.0."
            )

    def convert_to_data(
        self,
        frame_times: NDArray[np.float64],
        n_frames: int,
        env: Any,
    ) -> TimeSeriesData:
        """Convert overlay to internal data representation.

        Precomputes window indices for O(1) per-frame extraction during rendering.

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
            Internal data container with precomputed window indices.
        """
        # Vectorized precomputation of window indices for all frames
        half_window = self.window_seconds / 2
        start_indices = np.searchsorted(self.times, frame_times - half_window)
        end_indices = np.searchsorted(self.times, frame_times + half_window)

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
                output_data = np.zeros_like(self.data)  # Constant data -> 0
            # After normalization, limits become [0, 1]
            global_vmin, global_vmax = 0.0, 1.0

        return TimeSeriesData(
            data=output_data,
            times=self.times,
            start_indices=start_indices,
            end_indices=end_indices,
            label=self.label,
            color=self.color,
            window_seconds=self.window_seconds,
            linewidth=self.linewidth,
            alpha=self.alpha,
            group=self.group,
            normalize=self.normalize,
            show_cursor=self.show_cursor,
            cursor_color=self.cursor_color,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
            use_global_limits=True,
            interp=self.interp,
            update_mode=self.update_mode,
            playback_throttle_hz=self.playback_throttle_hz,
            scrub_throttle_hz=self.scrub_throttle_hz,
        )


@dataclass
class ObjectVectorOverlay:
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
        assumed uniformly spaced at the animation fps rate. Must be monotonically
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
        Size of object markers in points. Default is 15.0.
    interp : {"linear", "nearest"}, optional
        Interpolation method for aligning overlay to animation frames.
        "linear" (default) for smooth trajectories.
        "nearest" for discrete/categorical data or to preserve exact samples.

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
    compute_object_vector_field : Field computation for object-vector cells

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
            Internal data container aligned to frame times.

        Raises
        ------
        ValueError
            If shapes don't match or firing_rates length mismatches.
        """
        # Validate object positions shape
        _validate_finite_values(
            self.object_positions, name="ObjectVectorOverlay.object_positions"
        )
        _validate_shape(
            self.object_positions,
            env.n_dims,
            name="ObjectVectorOverlay.object_positions",
        )

        # Validate animal positions
        _validate_finite_values(
            self.animal_positions, name="ObjectVectorOverlay.animal_positions"
        )
        _validate_shape(
            self.animal_positions,
            env.n_dims,
            name="ObjectVectorOverlay.animal_positions",
        )

        # Validate firing rates if provided
        if self.firing_rates is not None:
            _validate_finite_values(
                self.firing_rates, name="ObjectVectorOverlay.firing_rates"
            )
            if len(self.firing_rates) != len(self.animal_positions):
                raise ValueError(
                    f"WHAT: ObjectVectorOverlay.firing_rates length "
                    f"({len(self.firing_rates)}) does not match "
                    f"animal_positions length ({len(self.animal_positions)}).\n\n"
                    "WHY: Each firing rate value corresponds to an animal position "
                    "sample, so they must have the same length.\n\n"
                    "HOW: Ensure firing_rates has shape (n_samples,) matching "
                    "animal_positions shape (n_samples, n_dims)."
                )

        # Align animal positions to frame times
        aligned_positions = _align_to_frame_times(
            self.animal_positions,
            self.times,
            frame_times,
            n_frames,
            self.interp,
            name="ObjectVectorOverlay.animal_positions",
        )

        # Validate bounds (warning only)
        _validate_bounds(
            aligned_positions,
            env.dimension_ranges,
            name="ObjectVectorOverlay.animal_positions",
            threshold=0.1,
        )

        # Align firing rates if provided
        aligned_firing_rates: NDArray[np.float64] | None = None
        if self.firing_rates is not None:
            # Reshape to 2D for alignment function, then squeeze back
            rates_2d = self.firing_rates.reshape(-1, 1)
            aligned_rates_2d = _align_to_frame_times(
                rates_2d,
                self.times,
                frame_times,
                n_frames,
                self.interp,
                name="ObjectVectorOverlay.firing_rates",
            )
            aligned_firing_rates = aligned_rates_2d.squeeze()

        return ObjectVectorData(
            object_positions=self.object_positions,
            animal_positions=aligned_positions,
            firing_rates=aligned_firing_rates,
            color=self.color,
            linewidth=self.linewidth,
            show_objects=self.show_objects,
            object_marker=self.object_marker,
            object_size=self.object_size,
        )


@dataclass
class VideoOverlay:
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
        If None, video is displayed in pixel coordinates without transform.
        Created using calibrate_from_scale_bar() or calibrate_from_landmarks().
        Default is None.
    times : ndarray of shape (n_frames,), dtype float64, optional
        Timestamps for each video frame, in seconds. If None, video frames
        are assumed uniformly spaced at the animation fps rate. Must be
        monotonically increasing. Default is None.
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
    neurospatial.transforms.calibrate_from_scale_bar : Create calibration from scale bar
    neurospatial.transforms.calibrate_from_landmarks : Create calibration from landmarks
    neurospatial.transforms.VideoCalibration : Calibration data container

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
    >>> from neurospatial.transforms import calibrate_from_scale_bar, VideoCalibration
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
                f"HOW: Use alpha=0.5 (default) for semi-transparent overlay."
            )

        # Validate downsample
        if not isinstance(self.downsample, int) or self.downsample < 1:
            raise ValueError(
                f"WHAT: downsample must be a positive integer >= 1, got {self.downsample}.\n"
                f"WHY: Downsample factor controls spatial resolution reduction.\n"
                f"HOW: Use downsample=1 (full resolution) or downsample=2 (half resolution)."
            )

        # Validate cache_size
        if not isinstance(self.cache_size, int) or self.cache_size < 1:
            raise ValueError(
                f"WHAT: cache_size must be a positive integer >= 1, got {self.cache_size}.\n"
                f"WHY: cache_size controls how many video frames are kept in memory.\n"
                f"HOW: Use cache_size=100 (default) or increase for faster playback "
                f"with more memory usage."
            )

        # Validate prefetch_ahead
        if not isinstance(self.prefetch_ahead, int) or self.prefetch_ahead < 0:
            raise ValueError(
                f"WHAT: prefetch_ahead must be a non-negative integer, "
                f"got {self.prefetch_ahead}.\n"
                f"WHY: prefetch_ahead controls how many frames to load in background.\n"
                f"HOW: Use prefetch_ahead=0 (default) to disable, or e.g. 5 to preload "
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
                "HOW: Using nearest-neighbor frame selection instead. Set interp='nearest' "
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
                f"HOW: Reshape array to (n_frames, height, width, 3) for RGB video."
            )

        # Check channels (must be 3 for RGB)
        if arr.shape[3] != 3:
            raise ValueError(
                f"WHAT: Video array must have 3 RGB channels, got {arr.shape[3]} channels.\n"
                f"WHY: VideoOverlay requires RGB format for rendering.\n"
                f"HOW: Convert to RGB (drop alpha channel if RGBA, or convert grayscale)."
            )

        # Check dtype (must be uint8)
        if arr.dtype != np.uint8:
            raise ValueError(
                f"WHAT: Video array must have dtype uint8, got {arr.dtype}.\n"
                f"WHY: Video pixels are 0-255 values stored as unsigned 8-bit integers.\n"
                f"HOW: Convert with arr.astype(np.uint8) if values are in 0-255 range."
            )

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
            Internal data container aligned to frame times.
        """
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

        # Determine video source: array or file path
        if isinstance(self.source, np.ndarray):
            # Array source - use directly as reader
            reader: NDArray[np.uint8] | Any = self.source
            n_video_frames = self.source.shape[0]
        else:
            # File path - create VideoReader
            from neurospatial.animation._video_io import VideoReader

            reader = VideoReader(
                self.source,
                cache_size=self.cache_size,
                downsample=self.downsample,
                crop=self.crop,
                prefetch_ahead=self.prefetch_ahead,
            )
            n_video_frames = reader.n_frames

        # Compute frame index mapping
        if self.times is not None:
            # Video has explicit times - map animation frames to video frames
            _validate_monotonic_time(self.times, name="VideoOverlay.times")

            if len(self.times) != n_video_frames:
                raise ValueError(
                    f"VideoOverlay.times length ({len(self.times)}) does not "
                    f"match number of video frames ({n_video_frames})."
                )

            # Find nearest video frame for each animation frame
            frame_indices = _find_nearest_indices(self.times, frame_times)
        else:
            # No times provided - assume 1:1 mapping
            if n_video_frames != n_frames:
                raise ValueError(
                    f"VideoOverlay source has {n_video_frames} frames but "
                    f"animation has {n_frames} frames. When times=None, "
                    f"video frames must match animation frames."
                )
            frame_indices = np.arange(n_frames, dtype=np.int_)

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
        transform_to_env = (
            self.calibration.transform_px_to_cm if self.calibration else None
        )

        return VideoData(
            frame_indices=frame_indices,
            reader=reader,
            transform_to_env=transform_to_env,
            env_bounds=env_bounds,
            alpha=self.alpha,
            z_order=self.z_order,
        )


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
        Colormap name for the direction line.
    length : float
        Length of direction line in environment coordinate units.
    width : float
        Line width in pixels.

    Attributes
    ----------
    data : NDArray[np.float64]
        Heading data aligned to frames.
    color : str
        Colormap name for direction line.
    length : float
        Line length in environment units.
    width : float
        Line width in pixels.

    See Also
    --------
    HeadDirectionOverlay : User-facing overlay configuration
    """

    data: NDArray[np.float64]
    color: str
    length: float
    width: float = 3.0


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
    colors : dict[str, str]
        Dict mapping event type names to color strings.
    markers : dict[str, str]
        Dict mapping event type names to marker style strings.
    size : float
        Marker size in points.
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


@dataclass
class VideoData:
    """Internal container for video overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from VideoOverlay instances.

    Parameters
    ----------
    frame_indices : ndarray of shape (n_animation_frames,), dtype int
        Mapping from animation frame indices to video frame indices.
        Value of -1 indicates no video frame available (out of range).
    reader : NDArray[np.uint8] or VideoReaderProtocol
        Video source. Either a pre-loaded array of shape (n_video_frames, height,
        width, 3) with dtype uint8, or a VideoReaderProtocol-compliant reader.
    transform_to_env : Affine2D or None
        Transform from video pixel coordinates to environment coordinates.
        If None, video is displayed in pixel coordinates.
    env_bounds : tuple of float
        Environment bounding box as (xmin, xmax, ymin, ymax) for positioning
        the video in the visualization.
    alpha : float
        Opacity of the video layer (0.0-1.0).
    z_order : {"below", "above"}
        Rendering order relative to the spatial field layer.

    Attributes
    ----------
    frame_indices : NDArray[np.int_]
        Animation-to-video frame mapping.
    reader : NDArray[np.uint8] | VideoReaderProtocol
        Video source. Either a pre-loaded array of shape (n_video_frames, height,
        width, 3) with dtype uint8, or a VideoReaderProtocol-compliant reader.
    transform_to_env : Affine2D | None
        Coordinate transform.
    env_bounds : tuple[float, float, float, float]
        Environment bounds (xmin, xmax, ymin, ymax).
    alpha : float
        Video opacity.
    z_order : {"below", "above"}
        Rendering order.

    See Also
    --------
    VideoOverlay : User-facing overlay configuration
    VideoReaderProtocol : Interface for video readers

    Notes
    -----
    For pickle-safety with parallel rendering:

    - Pre-loaded arrays (NDArray[np.uint8]) are pickle-safe
    - VideoReader instances implement pickle protocol (drops cache on pickle)
    """

    frame_indices: NDArray[np.int_]
    reader: NDArray[np.uint8] | VideoReaderProtocol
    transform_to_env: Affine2D | None
    env_bounds: tuple[float, float, float, float]
    alpha: float
    z_order: Literal["below", "above"]

    def get_frame(self, anim_frame_idx: int) -> NDArray[np.uint8] | None:
        """Get the video frame for a given animation frame index.

        Parameters
        ----------
        anim_frame_idx : int
            Animation frame index (0-based).

        Returns
        -------
        NDArray[np.uint8] | None
            RGB video frame of shape (height, width, 3) with dtype uint8,
            or None if no video frame is available for this animation frame.
            Returns None for:
            - Animation index out of bounds (>= len(frame_indices))
            - Video frame index is -1 (out of range in frame_indices)
        """
        # Check if animation index is out of bounds
        if anim_frame_idx < 0 or anim_frame_idx >= len(self.frame_indices):
            return None

        # Get the video frame index
        video_frame_idx = self.frame_indices[anim_frame_idx]

        # Check if video frame index is out of range (indicated by -1)
        if video_frame_idx < 0:
            return None

        # Return the video frame from the reader
        if isinstance(self.reader, np.ndarray):
            frame: NDArray[np.uint8] = self.reader[video_frame_idx]
            return frame
        else:
            # VideoReader case - uses subscript access
            result: NDArray[np.uint8] = self.reader[video_frame_idx]
            return result


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
    start_indices : ndarray of shape (n_frames,), dtype int64
        Precomputed start index for each animation frame's window.
    end_indices : ndarray of shape (n_frames,), dtype int64
        Precomputed end index for each animation frame's window.
    label : str
        Label for Y-axis or legend.
    color : str
        Line color (matplotlib color string).
    window_seconds : float
        Time window width in seconds.
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
    use_global_limits : bool
        If True, use global limits for stable Y-axis. Default is True.
    interp : {"linear", "nearest"}
        Interpolation method for cursor value computation.
    update_mode : {"live", "on_pause", "manual"}
        Update mode for dock widget during playback.
    playback_throttle_hz : float
        Maximum update frequency during playback. Default is 10.0.
    scrub_throttle_hz : float
        Maximum update frequency when scrubbing. Default is 20.0.

    Attributes
    ----------
    data : NDArray[np.float64]
        Time series values.
    times : NDArray[np.float64]
        Sample timestamps.
    start_indices : NDArray[np.int64]
        Window start indices per frame.
    end_indices : NDArray[np.int64]
        Window end indices per frame.
    label : str
        Display label.
    color : str
        Line color.
    window_seconds : float
        Window width.
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
    use_global_limits : bool
        Use global limits flag.
    interp : str
        Interpolation method.
    update_mode : str
        Update mode for dock widget.
    playback_throttle_hz : float
        Maximum update frequency during playback.
    scrub_throttle_hz : float
        Maximum update frequency when scrubbing.

    See Also
    --------
    TimeSeriesOverlay : User-facing overlay configuration
    """

    # Full resolution data (no downsampling)
    data: NDArray[np.float64]
    times: NDArray[np.float64]

    # Precomputed window indices per frame
    start_indices: NDArray[np.int64]
    end_indices: NDArray[np.int64]

    # Display settings
    label: str
    color: str
    window_seconds: float
    linewidth: float
    alpha: float
    group: str | None
    normalize: bool
    show_cursor: bool
    cursor_color: str

    # Y-axis limits
    global_vmin: float
    global_vmax: float
    use_global_limits: bool = True

    # Interpolation for cursor value
    interp: Literal["linear", "nearest"] = "linear"

    # Update mode for dock widget during playback
    update_mode: Literal["live", "on_pause", "manual"] = "live"

    # Throttle settings for performance optimization
    playback_throttle_hz: float = 10.0
    scrub_throttle_hz: float = 20.0

    def get_window_slice(self, frame_idx: int) -> tuple[NDArray, NDArray]:
        """O(1) window extraction using precomputed indices.

        Parameters
        ----------
        frame_idx : int
            Animation frame index (0-based).

        Returns
        -------
        tuple[NDArray, NDArray]
            Tuple of (data_slice, times_slice) for the window at this frame.
        """
        start = self.start_indices[frame_idx]
        end = self.end_indices[frame_idx]
        return self.data[start:end], self.times[start:end]

    def get_cursor_value(self, current_time: float) -> float | None:
        """Get interpolated value at cursor time for tooltip display.

        Parameters
        ----------
        current_time : float
            Current animation time in seconds.

        Returns
        -------
        float | None
            Interpolated value at cursor time, or None if current_time
            is outside data range or data is empty.
        """
        if len(self.times) == 0:
            return None
        if current_time < self.times[0] or current_time > self.times[-1]:
            return None

        if self.interp == "nearest":
            idx = np.searchsorted(self.times, current_time)
            # Choose closer of idx-1 or idx
            if idx == 0:
                return float(self.data[0])
            if idx >= len(self.times):
                return float(self.data[-1])
            if current_time - self.times[idx - 1] < self.times[idx] - current_time:
                return float(self.data[idx - 1])
            return float(self.data[idx])
        else:  # linear
            return float(np.interp(current_time, self.times, self.data))


@dataclass
class ObjectVectorData:
    """Internal container for object-vector overlay data aligned to animation frames.

    This is used internally by backends and should not be instantiated by users.
    Created by the conversion pipeline from ObjectVectorOverlay instances.

    Parameters
    ----------
    object_positions : ndarray of shape (n_objects, n_dims), dtype float64
        Static positions of objects in environment coordinates. These positions
        remain fixed throughout the animation.
    animal_positions : ndarray of shape (n_frames, n_dims), dtype float64
        Animal positions aligned to animation frames. NaN values indicate
        frames where position is unavailable.
    firing_rates : ndarray of shape (n_frames,), dtype float64, optional
        Firing rates aligned to animation frames for modulating line appearance.
        None if no modulation was specified.
    color : str
        Line color (matplotlib color string).
    linewidth : float
        Line width in pixels.
    show_objects : bool
        Whether to render object markers.
    object_marker : str
        Marker style for objects.
    object_size : float
        Size of object markers in points.

    Attributes
    ----------
    object_positions : NDArray[np.float64]
        Static object positions.
    animal_positions : NDArray[np.float64]
        Animal positions aligned to frames.
    firing_rates : NDArray[np.float64] | None
        Firing rates aligned to frames.
    color : str
        Line color.
    linewidth : float
        Line width.
    show_objects : bool
        Object marker visibility.
    object_marker : str
        Object marker style.
    object_size : float
        Object marker size.

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
    object_vectors : list[ObjectVectorData], optional
        List of object-vector overlays. Default is empty list.
    regions : dict[int, list[str]] | None, optional
        Region names in normalized format. Key is frame index (0 = all frames),
        value is list of region names. Populated by _convert_overlays_to_data()
        from show_regions parameter. Default is None.
    frame_times : NDArray[np.float64] | None, optional
        Animation frame timestamps. Used for synchronizing time series overlays
        with the animation. Populated by _convert_overlays_to_data() when time
        series overlays are present. Default is None.

    Attributes
    ----------
    positions : list[PositionData]
        List of position overlays.
    bodypart_sets : list[BodypartData]
        List of bodypart overlays.
    head_directions : list[HeadDirectionData]
        List of head direction overlays.
    videos : list[VideoData]
        List of video overlays.
    events : list[EventData]
        List of event overlays.
    timeseries : list[TimeSeriesData]
        List of time series overlays.
    object_vectors : list[ObjectVectorData]
        List of object-vector overlays.
    regions : dict[int, list[str]] | None
        Region names in normalized format (key 0 = all frames).
    frame_times : NDArray[np.float64] | None
        Animation frame timestamps for time series synchronization.

    Notes
    -----
    This container is created automatically by the conversion pipeline when
    `Environment.animate_fields()` is called with overlays. Users should not
    instantiate OverlayData directly.

    All data is aligned to animation frame times and validated for:

    - Shape consistency with environment dimensions
    - Finite values (no NaN/Inf)

    Pickle-ability is validated via ``_validate_pickle_ability()`` during the
    animation pipeline when ``n_workers > 1`` is specified for parallel video
    rendering. Serial rendering (``n_workers=1``) does not require pickling.

    Backends access this object to render overlays without needing to handle
    temporal alignment or validation.
    """

    positions: list[PositionData] = field(default_factory=list)
    bodypart_sets: list[BodypartData] = field(default_factory=list)
    head_directions: list[HeadDirectionData] = field(default_factory=list)
    videos: list[VideoData] = field(default_factory=list)
    events: list[EventData] = field(default_factory=list)
    timeseries: list[TimeSeriesData] = field(default_factory=list)
    object_vectors: list[ObjectVectorData] = field(default_factory=list)
    regions: dict[int, list[str]] | None = None
    frame_times: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Post-initialization hook.

        Notes
        -----
        Currently, overlay pickle-ability is validated in the conversion
        pipeline via ``_validate_pickle_ability`` rather than here. This keeps
        construction of ``OverlayData`` lightweight while still providing
        user-friendly errors when parallel backends are used (n_workers > 1).
        """
        # Intentionally no-op; validation lives in conversion pipeline.
        return None


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
        from neurospatial.animation._utils import _pickling_guidance

        raise ValueError(
            f"WHAT: OverlayData is not pickle-able, preventing parallel video "
            f"rendering.\n"
            f"  Pickling failed with: {type(e).__name__}: {e}\n\n"
            f"WHY: Parallel video rendering (n_workers > 1) requires pickling "
            f"OverlayData to pass it to worker processes. Unpickleable objects "
            f"include lambdas, closures, local functions, and certain class instances.\n\n"
            f"{_pickling_guidance(n_workers=n_workers)}"
        ) from e


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

    HOW: Ensure your environment is 2D (e.g., from_samples with 2D positions)
    and has finite dimension_ranges (no infinite extents).

    Examples
    --------
    >>> env = Environment.from_samples(np.random.rand(100, 2) * 100, bin_size=5)
    >>> _validate_video_env(env)  # Should not raise
    """
    # Check 2D requirement
    if env.n_dims != 2:
        raise ValueError(
            f"VideoOverlay requires a 2D environment.\n\n"
            f"WHAT: Environment has n_dims={env.n_dims}, expected n_dims=2.\n\n"
            f"WHY: Video frames are 2D images that must be transformed into "
            f"environment coordinates. This requires a 2D coordinate system.\n\n"
            f"HOW: Create a 2D environment:\n"
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
            f"HOW: Ensure your environment data has finite coordinates. "
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
    >>> times[0], times[-1]
    (0.0, 1.5)

    Single frame passes validation:

    >>> single = np.array([5.0])
    >>> _validate_frame_times(single, n_frames=1)
    array([5.])
    """
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


def _build_frame_times(
    frame_times: NDArray[np.float64] | None,
    fps: int | None,
    n_frames: int,
) -> NDArray[np.float64]:
    """Build or validate frame times array for animation.

    .. deprecated::
        This function is deprecated. Use _validate_frame_times() instead.
        The fps synthesis functionality is being removed as part of the
        animation API refactoring.

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
    """
    if frame_times is not None:
        return _validate_frame_times(frame_times, n_frames)

    if fps is not None:
        # Synthesize from fps (deprecated functionality)
        return np.arange(n_frames, dtype=np.float64) / fps

    raise ValueError(
        "Either frame_times or fps must be provided. "
        "Got both as None. "
        "HOW: Provide explicit frame_times array or specify fps."
    )


def _align_to_frame_times(
    data: NDArray[np.float64],
    times: NDArray[np.float64] | None,
    frame_times: NDArray[np.float64],
    n_frames: int,
    interp: Literal["linear", "nearest"],
    name: str,
) -> NDArray[np.float64]:
    """Validate timestamps and align data to animation frame times.

    This helper encapsulates the common pattern of validating overlay timestamps
    and performing temporal interpolation to align overlay data with animation
    frames.

    Parameters
    ----------
    data : NDArray[np.float64]
        Source data with shape (n_samples,) or (n_samples, n_dims).
    times : NDArray[np.float64] | None
        Source timestamps with shape (n_samples,). If None, data is assumed
        to already match frame_times (requires len(data) == n_frames).
    frame_times : NDArray[np.float64]
        Animation frame timestamps with shape (n_frames,).
    n_frames : int
        Number of animation frames.
    interp : {"linear", "nearest"}
        Interpolation method to use when aligning.
    name : str
        Overlay name for error messages (e.g., "PositionOverlay").

    Returns
    -------
    NDArray[np.float64]
        Data aligned to frame_times. Shape is (n_frames,) or (n_frames, n_dims).

    Raises
    ------
    ValueError
        If times are not monotonically increasing, or if len(data) != n_frames
        when times is None.
    """
    if times is not None:
        _validate_monotonic_time(times, name=f"{name}.times")
        _validate_temporal_alignment(times, frame_times, name=name)
        interp_fn = _interp_linear if interp == "linear" else _interp_nearest
        return interp_fn(times, data, frame_times)
    else:
        if len(data) != n_frames:
            raise ValueError(
                f"{name} data length ({len(data)}) does not match n_frames "
                f"({n_frames}). When times=None, data must have exactly "
                f"n_frames samples."
            )
        return data


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

    Uses searchsorted for O(n_target log n_source) complexity, avoiding the
    O(n_target * n_source) memory cost of a full distance matrix.

    For frame times exactly at the midpoint between two source times, the
    earlier (left) neighbor is chosen.

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

    # Find nearest indices using helper function
    nearest_indices = _find_nearest_indices(t_src, t_frame)
    valid_mask = nearest_indices >= 0

    if not np.any(valid_mask):
        return result

    # Index into source data
    if x_src.ndim == 1:
        result[valid_mask] = x_src[nearest_indices[valid_mask]]
    else:
        result[valid_mask, :] = x_src[nearest_indices[valid_mask], :]

    return result


# =============================================================================
# Conversion Funnel (Milestone 1.5)
# =============================================================================


def _convert_overlays_to_data(
    overlays: list[OverlayProtocol],
    frame_times: NDArray[np.float64],
    n_frames: int,
    env: Any,
    show_regions: bool | list[str] = False,
) -> OverlayData:
    """Convert overlay configurations to aligned internal data representation.

    This is the main conversion pipeline that:
    1. Calls each overlay's ``convert_to_data()`` method
    2. Dispatches results to appropriate internal data lists
    3. Aggregates all overlays into a single OverlayData container

    Parameters
    ----------
    overlays : list[OverlayProtocol]
        List of overlay configurations to convert. Can be empty or contain
        multiple instances of any overlay type implementing ``OverlayProtocol``.
        Custom overlays must implement the ``convert_to_data()`` method.
    frame_times : NDArray[np.float64]
        Animation frame timestamps with shape (n_frames,). Used as interpolation
        targets for aligning overlay data.
    n_frames : int
        Number of animation frames. Used for validation.
    env : Any
        Environment object with ``n_dims`` and ``dimension_ranges`` attributes.
        Used for validating overlay coordinate dimensions and bounds.
    show_regions : bool or list of str, default=False
        Region names to include in overlay data. If True, all region names
        from env.regions are included. If a list, only those region names
        are included. If False, no regions are included. Normalized to
        dict[int, list[str]] format where key 0 means "all frames".

    Returns
    -------
    OverlayData
        Aggregated container with all overlay data aligned to frame times.
        Contains lists of PositionData, BodypartData, HeadDirectionData,
        and VideoData instances.

    Raises
    ------
    ValueError
        If any validation fails:
        - Non-monotonic timestamps in overlay.times
        - NaN/Inf values in overlay data
        - Shape mismatch between overlay and environment dimensions
        - Skeleton references missing bodypart names
        - No temporal overlap between overlay and frame times
    TypeError
        If overlay does not implement ``OverlayProtocol``, or if
        ``convert_to_data()`` returns an unexpected type (must return
        PositionData, BodypartData, HeadDirectionData, or VideoData).

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

    Custom overlays can be created by implementing ``OverlayProtocol``. The
    ``convert_to_data()`` method must return one of the internal data types:
    ``PositionData``, ``BodypartData``, ``HeadDirectionData``, or ``VideoData``.

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
    video_data_list: list[VideoData] = []
    event_data_list: list[EventData] = []
    timeseries_data_list: list[TimeSeriesData] = []
    object_vector_data_list: list[ObjectVectorData] = []

    # Process each overlay using protocol dispatch
    for overlay in overlays:
        # Validate overlay implements protocol
        if not isinstance(overlay, OverlayProtocol):
            raise TypeError(
                f"Overlay must implement OverlayProtocol, got {type(overlay).__name__}. "
                f"Ensure your overlay has 'times', 'interp' attributes and a "
                f"'convert_to_data()' method."
            )

        # Delegate to overlay's convert_to_data method
        internal_data = overlay.convert_to_data(frame_times, n_frames, env)

        # Dispatch to appropriate list based on return type
        if isinstance(internal_data, PositionData):
            position_data_list.append(internal_data)
        elif isinstance(internal_data, BodypartData):
            bodypart_data_list.append(internal_data)
        elif isinstance(internal_data, HeadDirectionData):
            head_direction_data_list.append(internal_data)
        elif isinstance(internal_data, VideoData):
            video_data_list.append(internal_data)
        elif isinstance(internal_data, EventData):
            event_data_list.append(internal_data)
        elif isinstance(internal_data, TimeSeriesData):
            timeseries_data_list.append(internal_data)
        elif isinstance(internal_data, ObjectVectorData):
            object_vector_data_list.append(internal_data)
        else:
            raise TypeError(
                f"convert_to_data() must return PositionData, BodypartData, "
                f"HeadDirectionData, VideoData, EventData, TimeSeriesData, "
                f"or ObjectVectorData, got {type(internal_data).__name__}."
            )

    # Normalize regions to dict[int, list[str]] format
    # Key 0 means "apply to all frames"
    normalized_regions: dict[int, list[str]] | None = None
    if show_regions:
        if isinstance(show_regions, bool):
            # True → all region names from env.regions
            normalized_regions = {0: list(env.regions.keys())}
        else:
            # list[str] → wrap in dict
            normalized_regions = {0: show_regions}

    # Aggregate all overlay data
    overlay_data = OverlayData(
        positions=position_data_list,
        bodypart_sets=bodypart_data_list,
        head_directions=head_direction_data_list,
        videos=video_data_list,
        events=event_data_list,
        timeseries=timeseries_data_list,
        object_vectors=object_vector_data_list,
        regions=normalized_regions,
        frame_times=frame_times,
    )

    return overlay_data
