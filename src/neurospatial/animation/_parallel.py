"""Parallel frame rendering utilities.

Based on approach from:
https://gist.github.com/edeno/652ee10a76481f00b3eb08906b41c6bf

Key principles:
- Each worker process has its own matplotlib figure (avoid threading issues)
- Frames saved as numbered PNGs for ffmpeg pattern matching
- Workers operate independently on partitioned frame ranges

Overlay Rendering Layer Order (zorder):
- 99: Region boundaries (background)
- 100: Position trails
- 101: Position markers, bodypart skeletons
- 102: Bodypart keypoints
- 103: Head direction arrows
- 104: Event markers (foreground)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Module-level imports for hot path performance (avoid per-frame import overhead)
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.quiver import Quiver
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def _render_event_overlay_matplotlib(ax: Any, event_data: Any, frame_idx: int) -> None:
    """Render event markers for current frame on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on.
    event_data : EventData
        Event overlay data containing positions, frame indices, colors, markers.
    frame_idx : int
        Current frame index.

    Notes
    -----
    Supports three rendering modes based on decay_frames:

    - Cumulative mode (decay_frames=None): All events up to current frame are shown.
      Events accumulate over time and persist permanently.
    - Instant mode (decay_frames=0): Only events on exactly this frame are shown.
    - Decay mode (decay_frames > 0): Events within the decay window are shown
      with alpha decaying based on age (newest = alpha 1.0, oldest = faded).

    Events render at zorder=104, above head direction overlays (103).
    """
    for event_name, positions in event_data.event_positions.items():
        frame_indices = event_data.event_frame_indices[event_name]
        color = event_data.colors[event_name]
        marker = event_data.markers[event_name]

        if len(positions) == 0:
            continue

        # Determine visibility mask based on mode
        decay = event_data.decay_frames
        if decay is None:
            # Cumulative mode: all events up to current frame
            mask = frame_indices <= frame_idx
        elif decay == 0:
            # Instant mode: only show events on their exact frame
            mask = frame_indices == frame_idx
        else:
            # Decay mode: show events within decay window
            min_frame = frame_idx - decay
            mask = (frame_indices >= max(0, min_frame)) & (frame_indices <= frame_idx)

        active_positions = positions[mask]
        active_frames = frame_indices[mask]

        if len(active_positions) == 0:
            continue

        # Compute per-event alpha based on recency
        if decay is not None and decay > 0:
            # Alpha = 1.0 for current frame, decays to lower for oldest
            ages = frame_idx - active_frames  # 0 = newest, decay_frames = oldest
            # +1 prevents alpha from reaching exactly 0 for oldest events
            alphas = 1.0 - (ages / (decay + 1))
        else:
            # Cumulative and instant modes: full opacity
            alphas = np.ones(len(active_positions))

        # Render each event with its computed alpha
        base_rgba = to_rgba(color)

        for pos, alpha in zip(active_positions, alphas, strict=True):
            if np.any(np.isnan(pos)):
                continue
            ax.scatter(
                pos[0],
                pos[1],
                c=[(*base_rgba[:3], alpha)],
                s=event_data.size**2,
                marker=marker,
                zorder=104,
                edgecolors=event_data.border_color,
                linewidths=event_data.border_width,
            )


def _render_position_overlay_matplotlib(ax: Any, pos_data: Any, frame_idx: int) -> None:
    """Render position overlay with trail and marker on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    pos_data : PositionData
        Position overlay data
    frame_idx : int
        Current frame index

    Notes
    -----
    Renders trail using LineCollection with per-segment decaying alpha
    (oldest segments have low alpha, newest have high alpha), and current
    position as a scatter point.
    """
    # Get current position
    current_pos = pos_data.data[frame_idx]

    # Skip if NaN
    if np.any(np.isnan(current_pos)):
        return

    # Render trail if specified
    if pos_data.trail_length is not None and pos_data.trail_length > 0:
        trail_start = max(0, frame_idx - pos_data.trail_length + 1)
        trail_positions = pos_data.data[trail_start : frame_idx + 1]

        # Filter out NaN positions
        valid_mask = ~np.any(np.isnan(trail_positions), axis=1)
        trail_positions = trail_positions[valid_mask]

        if len(trail_positions) > 1:
            # Create line segments with decaying alpha using LineCollection
            segments = [
                trail_positions[i : i + 2] for i in range(len(trail_positions) - 1)
            ]
            # Compute per-segment alpha (oldest=low, newest=high)
            alphas = [
                (i + 1) / len(trail_positions) * 0.7 for i in range(len(segments))
            ]
            # Convert color to RGBA with per-segment alpha
            base_rgba = to_rgba(pos_data.color)
            colors = [(*base_rgba[:3], alpha) for alpha in alphas]

            lc = LineCollection(segments, colors=colors, linewidths=1.5, zorder=100)
            ax.add_collection(lc)

    # Render current position marker
    ax.scatter(
        current_pos[0],
        current_pos[1],
        c=pos_data.color,
        s=pos_data.size**2,  # Matplotlib scatter uses area
        zorder=101,
        edgecolors="white",
        linewidths=0.5,
    )


def _render_bodypart_overlay_matplotlib(
    ax: Any, bodypart_data: Any, frame_idx: int
) -> None:
    """Render bodypart overlay with skeleton on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    bodypart_data : BodypartData
        Bodypart overlay data
    frame_idx : int
        Current frame index

    Notes
    -----
    Uses LineCollection for efficient skeleton rendering (single call per frame).
    """
    # Render bodypart points
    for part_name, positions in bodypart_data.bodyparts.items():
        pos = positions[frame_idx]

        # Skip NaN positions
        if np.any(np.isnan(pos)):
            continue

        # Determine color
        if bodypart_data.colors and part_name in bodypart_data.colors:
            color = bodypart_data.colors[part_name]
        else:
            color = "cyan"

        # Render point
        ax.scatter(
            pos[0],
            pos[1],
            c=color,
            s=25,
            zorder=102,
            edgecolors="white",
            linewidths=0.5,
        )

    # Render skeleton using LineCollection
    skeleton = bodypart_data.skeleton
    if skeleton is not None and len(skeleton.edges) > 0:
        skeleton_segments = []

        for start_part, end_part in skeleton.edges:
            if (
                start_part in bodypart_data.bodyparts
                and end_part in bodypart_data.bodyparts
            ):
                start_pos = bodypart_data.bodyparts[start_part][frame_idx]
                end_pos = bodypart_data.bodyparts[end_part][frame_idx]

                # Skip if either endpoint is NaN
                if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                    continue

                skeleton_segments.append([start_pos, end_pos])

        if skeleton_segments:
            lc = LineCollection(
                skeleton_segments,
                colors=skeleton.edge_color,
                linewidths=skeleton.edge_width,
                zorder=101,
            )
            ax.add_collection(lc)


def _render_head_direction_overlay_matplotlib(
    ax: Any,
    head_dir_data: Any,
    frame_idx: int,
    env: Any,
    position_data: Any | None = None,
) -> None:
    """Render head direction overlay as arrow on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    head_dir_data : HeadDirectionData
        Head direction overlay data
    frame_idx : int
        Current frame index
    env : Environment
        Environment for positioning arrow
    position_data : PositionData | None, optional
        If provided, arrows are anchored at the position coordinates for each frame.
        If None, arrows are anchored at the environment centroid (fixed reference).
        Default is None.

    Notes
    -----
    Renders direction as an arrow using matplotlib quiver.

    When `position_data` is provided, arrows follow the animal's position and
    visualize heading direction at that location. This is the recommended mode
    for tracking moving animals.
    """
    # Determine if data is angles or vectors
    is_angles = head_dir_data.data.ndim == 1

    # Determine arrow origin: prefer position data if available
    if position_data is not None:
        origin = position_data.data[frame_idx]
        # Skip if position is NaN (use centroid fallback)
        if np.any(np.isnan(origin)):
            origin = np.mean(env.bin_centers, axis=0)
    else:
        # Fallback: use centroid of environment
        origin = np.mean(env.bin_centers, axis=0)

    # Compute direction vector
    if is_angles:
        angle = head_dir_data.data[frame_idx]
        # Skip rendering if angle is NaN
        if np.isnan(angle):
            return
        direction = np.array([np.cos(angle), np.sin(angle)]) * head_dir_data.length
    else:
        direction = head_dir_data.data[frame_idx]
        # Skip if direction is NaN
        if np.any(np.isnan(direction)):
            return
        # Normalize and scale
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm * head_dir_data.length

    # Render arrow using quiver
    ax.quiver(
        origin[0],
        origin[1],
        direction[0],
        direction[1],
        color=head_dir_data.color,
        scale=1,
        scale_units="xy",
        angles="xy",
        width=0.006,
        headwidth=4,
        headlength=5,
        zorder=103,
    )


def _render_video_background(
    ax: Any,
    video_data: Any,
    frame_idx: int,
    _env: Any = None,
) -> None:
    """Render video overlay frame as background image on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on.
    video_data : VideoData
        Video overlay data containing frame indices and reader.
    frame_idx : int
        Current animation frame index.
    _env : Environment, optional
        Reserved for future use (e.g., spatial validation). Default is None.

    Notes
    -----
    This function creates a new imshow artist each call (suitable for parallel
    rendering). For sequential rendering with artist reuse, use VideoFrameRenderer.

    Renders video beneath or above the field based on z_order:
    - "below": zorder=-1 (rendered beneath field, default zorder=0)
    - "above": zorder=1 (rendered above field)
    """
    # Get video frame index for this animation frame
    if frame_idx < 0 or frame_idx >= len(video_data.frame_indices):
        return  # Out of bounds

    video_frame_idx = video_data.frame_indices[frame_idx]

    if video_frame_idx < 0:
        return  # -1 indicates no video for this frame

    # Get video frame from reader
    frame_rgb = video_data.get_frame(frame_idx)
    if frame_rgb is None:
        return

    # Calculate extent in environment coordinates
    if video_data.transform_to_env is not None:
        # Use transform to compute extent from video corners
        h, w = frame_rgb.shape[:2]
        corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        corners_cm = video_data.transform_to_env(corners_px)
        extent = [
            corners_cm[:, 0].min(),
            corners_cm[:, 0].max(),
            corners_cm[:, 1].min(),
            corners_cm[:, 1].max(),
        ]
    else:
        # No transform - use env_bounds directly
        xmin, xmax, ymin, ymax = video_data.env_bounds
        extent = [xmin, xmax, ymin, ymax]

    # Determine zorder based on z_order setting
    zorder = -1 if video_data.z_order == "below" else 1

    # Render video frame
    ax.imshow(
        frame_rgb,
        extent=extent,
        aspect="auto",
        origin="lower",  # Use lower origin (Y-flip is in calibration transform)
        alpha=video_data.alpha,
        zorder=zorder,
    )


class VideoFrameRenderer:
    """Manages video artist for efficient sequential frame updates.

    For sequential rendering (e.g., widget backend), this class reuses a single
    matplotlib imshow artist across frames, updating only the data via set_data().
    This is faster than creating new artists each frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to render on.
    video_data : VideoData
        Video overlay data.
    env : Environment
        Environment for spatial context.

    Attributes
    ----------
    video_data : VideoData
        Video overlay data.
    _artist : AxesImage | None
        The matplotlib imshow artist, created on first render.
    _extent : list[float]
        Pre-computed extent for the video.

    Notes
    -----
    For parallel rendering (n_workers > 1), use _render_video_background()
    instead as each worker needs fresh artists.
    """

    def __init__(self, ax: Any, video_data: Any, env: Any) -> None:
        """Initialize the renderer.

        Note: ax is received but not stored because render() receives ax as
        a parameter (same as other overlay renderers in OverlayArtistManager).
        This keeps the API consistent with how matplotlib artists are managed.
        """
        self.video_data = video_data
        self._artist: Any | None = None
        self._extent = self._compute_extent(video_data)

    def _compute_extent(self, video_data: Any) -> list[float]:
        """Compute extent once (assumes constant frame size)."""
        if video_data.transform_to_env is not None:
            # Need a sample frame to get dimensions
            # Try to get first valid frame
            for i, idx in enumerate(video_data.frame_indices):
                if idx >= 0:
                    frame = video_data.get_frame(i)
                    if frame is not None:
                        h, w = frame.shape[:2]
                        corners_px = np.array(
                            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64
                        )
                        corners_cm = video_data.transform_to_env(corners_px)
                        return [
                            corners_cm[:, 0].min(),
                            corners_cm[:, 0].max(),
                            corners_cm[:, 1].min(),
                            corners_cm[:, 1].max(),
                        ]
            # Fallback to env_bounds if no valid frames
            return list(video_data.env_bounds)
        else:
            return list(video_data.env_bounds)

    def render(self, ax: Any, frame_idx: int) -> None:
        """Render video frame, reusing artist if possible.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to render on.
        frame_idx : int
            Animation frame index.
        """
        # Get video frame index
        if frame_idx < 0 or frame_idx >= len(self.video_data.frame_indices):
            if self._artist is not None:
                self._artist.set_visible(False)
            return

        video_frame_idx = self.video_data.frame_indices[frame_idx]

        if video_frame_idx < 0:
            # No video for this frame
            if self._artist is not None:
                self._artist.set_visible(False)
            return

        frame_rgb = self.video_data.get_frame(frame_idx)
        if frame_rgb is None:
            if self._artist is not None:
                self._artist.set_visible(False)
            return

        # Determine zorder based on z_order setting
        zorder = -1 if self.video_data.z_order == "below" else 1

        if self._artist is None:
            # First frame: create artist
            self._artist = ax.imshow(
                frame_rgb,
                extent=self._extent,
                aspect="auto",
                origin="lower",
                alpha=self.video_data.alpha,
                zorder=zorder,
            )
        else:
            # Subsequent frames: reuse artist with set_data
            self._artist.set_data(frame_rgb)
            self._artist.set_visible(True)


def _render_regions_matplotlib(
    ax: Any, env: Any, show_regions: bool | list[str], region_alpha: float
) -> None:
    """Render environment regions as patches on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    env : Environment
        Environment with regions
    show_regions : bool | list[str]
        If True, show all regions. If list, show only specified regions.
    region_alpha : float
        Alpha transparency for regions (0-1)

    Notes
    -----
    Uses PathPatch for polygon regions and circles for point regions.
    """
    if not show_regions or len(env.regions) == 0:
        return

    # Determine which regions to show
    if isinstance(show_regions, bool):
        region_names = list(env.regions.keys())
    else:
        region_names = show_regions

    # Render each region
    for region_name in region_names:
        if region_name not in env.regions:
            continue

        region = env.regions[region_name]

        if region.kind == "point":
            # Point region: render as circle
            coords = region.data
            circle = Circle(
                coords,
                radius=5.0,  # Visual marker size
                facecolor="white",
                edgecolor="white",
                alpha=region_alpha,
                zorder=99,
            )
            ax.add_patch(circle)
        elif region.kind == "polygon":
            # Polygon region: use PathPatch
            # Extract coordinates from Shapely polygon
            exterior_coords = np.array(region.data.exterior.coords)
            path = MplPath(exterior_coords)
            patch = PathPatch(
                path,
                facecolor="white",
                edgecolor="white",
                alpha=region_alpha,
                zorder=99,
            )
            ax.add_patch(patch)


def _clear_overlay_artists(ax: Any) -> None:
    """Clear overlay artists from axes while preserving the primary image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to clear overlay artists from

    Notes
    -----
    Removes collections (LineCollection, PathCollection) added by overlays,
    but keeps the first image artist (the field visualization).
    This enables artist reuse for the field while properly clearing overlays.
    """
    # Remove all collections (trails, skeletons, scatter points)
    # Collections are added by LineCollection and scatter calls
    while len(ax.collections) > 0:
        ax.collections[-1].remove()

    # Remove all patches (regions)
    # Patches are added by PathPatch and Circle calls
    while len(ax.patches) > 0:
        ax.patches[-1].remove()

    # Remove all quiver/arrow artists (head direction)
    # These are stored as separate artists
    for artist in ax.get_children():
        # quiver creates FancyArrow or FancyArrowPatch objects
        if hasattr(artist, "arrow_patch") or type(artist).__name__ == "FancyArrow":
            artist.remove()


# =============================================================================
# Overlay Artist Manager (persistent artist reuse)
# =============================================================================


@dataclass
class OverlayArtistManager:
    """Manages persistent matplotlib artists for efficient overlay rendering.

    Instead of creating and destroying artists each frame, this manager keeps
    persistent references and updates their data. This provides significant
    performance gains for animations with overlays.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to render overlays on.
    env : Environment
        Environment for spatial context (bin_centers for head direction anchor).
    overlay_data : OverlayData
        Overlay data containing positions, bodyparts, head directions.
    show_regions : bool or list of str
        Region display configuration.
    region_alpha : float
        Alpha transparency for regions.

    Attributes
    ----------
    _position_trails : list[LineCollection]
        Trail line collections for each position overlay.
    _position_markers : list[PathCollection]
        Scatter point collections for position markers.
    _bodypart_points : list[PathCollection]
        Scatter collections for bodypart keypoints.
    _bodypart_skeletons : list[LineCollection]
        Line collections for bodypart skeletons.
    _head_direction_quivers : list[Quiver | None]
        Quiver artists for head direction arrows.
    _region_patches : list
        Patches for region rendering (only created once, not updated).

    Notes
    -----
    Artist update methods used:

    - LineCollection: `set_segments()`, `set_colors()`
    - PathCollection: `set_offsets()`, `set_facecolors()`
    - Quiver: Recreated each frame (matplotlib limitation - position updates
      require full recreation)

    The manager is designed to be created once per worker/figure and updated
    each frame via `update_frame()`.
    """

    ax: Any
    env: Any
    overlay_data: Any
    show_regions: bool | list[str]
    region_alpha: float

    # Persistent artist references (initialized on first frame)
    _position_trails: list[LineCollection | None] = field(default_factory=list)
    _position_markers: list[PathCollection] = field(default_factory=list)
    _bodypart_points: list[PathCollection] = field(default_factory=list)
    _bodypart_skeletons: list[LineCollection | None] = field(default_factory=list)
    _head_direction_quivers: list[Quiver | None] = field(default_factory=list)
    _region_patches: list[Any] = field(default_factory=list)
    _initialized: bool = False

    def initialize(self, frame_idx: int = 0) -> None:
        """Initialize all overlay artists for the first frame.

        Creates persistent artists that will be reused for subsequent frames.
        Must be called before `update_frame()`.

        Parameters
        ----------
        frame_idx : int, default=0
            Initial frame index for overlay data.
        """
        if self._initialized:
            return

        # Render regions (static, only created once)
        self._initialize_regions()

        if self.overlay_data is None:
            self._initialized = True
            return

        # Initialize position overlays
        for pos_data in self.overlay_data.positions:
            self._initialize_position_overlay(pos_data, frame_idx)

        # Initialize bodypart overlays
        for bodypart_data in self.overlay_data.bodypart_sets:
            self._initialize_bodypart_overlay(bodypart_data, frame_idx)

        # Initialize head direction overlays
        paired_position = (
            self.overlay_data.positions[0]
            if len(self.overlay_data.positions) == 1
            else None
        )
        for head_dir_data in self.overlay_data.head_directions:
            self._initialize_head_direction_overlay(
                head_dir_data, frame_idx, paired_position
            )

        self._initialized = True

    def _initialize_regions(self) -> None:
        """Create region patches (static, not updated per frame)."""
        if not self.show_regions or len(self.env.regions) == 0:
            return

        region_names: list[str]
        if isinstance(self.show_regions, bool):
            region_names = list(self.env.regions.keys())
        else:
            region_names = self.show_regions

        for region_name in region_names:
            if region_name not in self.env.regions:
                continue

            region = self.env.regions[region_name]

            if region.kind == "point":
                coords = region.data
                circle = Circle(
                    coords,
                    radius=5.0,
                    facecolor="white",
                    edgecolor="white",
                    alpha=self.region_alpha,
                    zorder=99,
                )
                self.ax.add_patch(circle)
                self._region_patches.append(circle)
            elif region.kind == "polygon":
                exterior_coords = np.array(region.data.exterior.coords)
                path = MplPath(exterior_coords)
                patch = PathPatch(
                    path,
                    facecolor="white",
                    edgecolor="white",
                    alpha=self.region_alpha,
                    zorder=99,
                )
                self.ax.add_patch(patch)
                self._region_patches.append(patch)

    def _initialize_position_overlay(self, pos_data: Any, frame_idx: int) -> None:
        """Initialize position overlay artists."""
        current_pos = pos_data.data[frame_idx]

        # Create trail LineCollection (empty initially, will be populated)
        trail_lc: LineCollection | None = None
        if pos_data.trail_length is not None and pos_data.trail_length > 0:
            # Create empty LineCollection that will be updated
            trail_lc = LineCollection([], linewidths=1.5, zorder=100)
            self.ax.add_collection(trail_lc)
            self._position_trails.append(trail_lc)
        else:
            self._position_trails.append(None)

        # Create position marker scatter
        if not np.any(np.isnan(current_pos)):
            marker = self.ax.scatter(
                [current_pos[0]],
                [current_pos[1]],
                c=[pos_data.color],
                s=[pos_data.size**2],
                zorder=101,
                edgecolors="white",
                linewidths=0.5,
            )
        else:
            # Create with dummy data (will be updated)
            marker = self.ax.scatter(
                [0],
                [0],
                c=[pos_data.color],
                s=[pos_data.size**2],
                zorder=101,
                edgecolors="white",
                linewidths=0.5,
            )
            marker.set_visible(False)
        self._position_markers.append(marker)

        # Update trail for initial frame
        self._update_position_trail(len(self._position_trails) - 1, pos_data, frame_idx)

    def _initialize_bodypart_overlay(self, bodypart_data: Any, frame_idx: int) -> None:
        """Initialize bodypart overlay artists."""
        # Collect all bodypart positions for this frame
        all_offsets = []
        all_colors = []

        for part_name, positions in bodypart_data.bodyparts.items():
            pos = positions[frame_idx]
            if not np.any(np.isnan(pos)):
                all_offsets.append(pos)
                if bodypart_data.colors and part_name in bodypart_data.colors:
                    all_colors.append(bodypart_data.colors[part_name])
                else:
                    all_colors.append("cyan")

        # Create scatter for all bodyparts
        if all_offsets:
            offsets_array = np.array(all_offsets)
            points = self.ax.scatter(
                offsets_array[:, 0],
                offsets_array[:, 1],
                c=all_colors,
                s=25,
                zorder=102,
                edgecolors="white",
                linewidths=0.5,
            )
        else:
            # Create with dummy data
            points = self.ax.scatter(
                [0],
                [0],
                c=["cyan"],
                s=25,
                zorder=102,
                edgecolors="white",
                linewidths=0.5,
            )
            points.set_visible(False)
        self._bodypart_points.append(points)

        # Create skeleton LineCollection
        skeleton_lc: LineCollection | None = None
        skeleton = bodypart_data.skeleton
        if skeleton is not None and len(skeleton.edges) > 0:
            skeleton_lc = LineCollection(
                [],
                colors=skeleton.edge_color,
                linewidths=skeleton.edge_width,
                zorder=101,
            )
            self.ax.add_collection(skeleton_lc)
        # Append skeleton BEFORE calling update (update accesses by index)
        self._bodypart_skeletons.append(skeleton_lc)
        # Now update skeleton with initial data
        if skeleton_lc is not None:
            self._update_bodypart_skeleton(
                len(self._bodypart_skeletons) - 1, bodypart_data, frame_idx
            )

    def _initialize_head_direction_overlay(
        self, head_dir_data: Any, frame_idx: int, position_data: Any | None
    ) -> None:
        """Initialize head direction overlay artist."""
        # Head direction requires quiver which is complex to update
        # We'll store a reference and recreate as needed
        quiver = self._create_head_direction_quiver(
            head_dir_data, frame_idx, position_data
        )
        self._head_direction_quivers.append(quiver)

    def _create_head_direction_quiver(
        self, head_dir_data: Any, frame_idx: int, position_data: Any | None
    ) -> Quiver | None:
        """Create quiver artist for head direction."""
        is_angles = head_dir_data.data.ndim == 1

        # Determine arrow origin
        if position_data is not None:
            origin = position_data.data[frame_idx]
            if np.any(np.isnan(origin)):
                origin = np.mean(self.env.bin_centers, axis=0)
        else:
            origin = np.mean(self.env.bin_centers, axis=0)

        # Compute direction vector
        if is_angles:
            angle = head_dir_data.data[frame_idx]
            if np.isnan(angle):
                return None
            direction = np.array([np.cos(angle), np.sin(angle)]) * head_dir_data.length
        else:
            direction = head_dir_data.data[frame_idx]
            if np.any(np.isnan(direction)):
                return None
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * head_dir_data.length

        quiver: Quiver = self.ax.quiver(
            origin[0],
            origin[1],
            direction[0],
            direction[1],
            color=head_dir_data.color,
            scale=1,
            scale_units="xy",
            angles="xy",
            width=0.006,
            headwidth=4,
            headlength=5,
            zorder=103,
        )
        return quiver

    def update_frame(self, frame_idx: int) -> None:
        """Update all overlay artists for a new frame.

        Parameters
        ----------
        frame_idx : int
            Frame index for overlay data.
        """
        if not self._initialized:
            self.initialize(frame_idx)
            return

        if self.overlay_data is None:
            return

        # Update position overlays
        for i, pos_data in enumerate(self.overlay_data.positions):
            self._update_position_marker(i, pos_data, frame_idx)
            self._update_position_trail(i, pos_data, frame_idx)

        # Update bodypart overlays
        for i, bodypart_data in enumerate(self.overlay_data.bodypart_sets):
            self._update_bodypart_points(i, bodypart_data, frame_idx)
            skeleton = bodypart_data.skeleton
            if (
                skeleton is not None
                and len(skeleton.edges) > 0
                and self._bodypart_skeletons[i] is not None
            ):
                self._update_bodypart_skeleton(i, bodypart_data, frame_idx)

        # Update head direction overlays (recreate quiver - matplotlib limitation)
        paired_position = (
            self.overlay_data.positions[0]
            if len(self.overlay_data.positions) == 1
            else None
        )
        for i, head_dir_data in enumerate(self.overlay_data.head_directions):
            self._update_head_direction(i, head_dir_data, frame_idx, paired_position)

    def _update_position_marker(self, idx: int, pos_data: Any, frame_idx: int) -> None:
        """Update position marker scatter point."""
        marker = self._position_markers[idx]
        current_pos = pos_data.data[frame_idx]

        if np.any(np.isnan(current_pos)):
            marker.set_visible(False)
        else:
            marker.set_visible(True)
            marker.set_offsets([current_pos])

    def _update_position_trail(self, idx: int, pos_data: Any, frame_idx: int) -> None:
        """Update position trail line collection."""
        trail_lc = self._position_trails[idx]
        if trail_lc is None or pos_data.trail_length is None:
            return

        trail_start = max(0, frame_idx - pos_data.trail_length + 1)
        trail_positions = pos_data.data[trail_start : frame_idx + 1]

        # Filter out NaN positions
        valid_mask = ~np.any(np.isnan(trail_positions), axis=1)
        trail_positions = trail_positions[valid_mask]

        if len(trail_positions) > 1:
            # Create line segments
            segments = [
                trail_positions[i : i + 2] for i in range(len(trail_positions) - 1)
            ]
            # Compute per-segment alpha
            alphas = [
                (i + 1) / len(trail_positions) * 0.7 for i in range(len(segments))
            ]
            base_rgba = to_rgba(pos_data.color)
            colors = [(*base_rgba[:3], alpha) for alpha in alphas]

            trail_lc.set_segments(segments)
            trail_lc.set_colors(colors)
            trail_lc.set_visible(True)
        else:
            trail_lc.set_segments([])
            trail_lc.set_visible(False)

    def _update_bodypart_points(
        self, idx: int, bodypart_data: Any, frame_idx: int
    ) -> None:
        """Update bodypart scatter points."""
        points = self._bodypart_points[idx]

        all_offsets = []
        all_colors = []

        for part_name, positions in bodypart_data.bodyparts.items():
            pos = positions[frame_idx]
            if not np.any(np.isnan(pos)):
                all_offsets.append(pos)
                if bodypart_data.colors and part_name in bodypart_data.colors:
                    all_colors.append(bodypart_data.colors[part_name])
                else:
                    all_colors.append("cyan")

        if all_offsets:
            offsets_array = np.array(all_offsets)
            points.set_offsets(offsets_array)
            points.set_facecolor(all_colors)
            points.set_visible(True)
        else:
            points.set_visible(False)

    def _update_bodypart_skeleton(
        self, idx: int, bodypart_data: Any, frame_idx: int
    ) -> None:
        """Update bodypart skeleton line collection."""
        skeleton_lc = self._bodypart_skeletons[idx]
        if skeleton_lc is None:
            return

        skeleton = bodypart_data.skeleton
        if skeleton is None:
            return

        skeleton_segments = []
        for start_part, end_part in skeleton.edges:
            if (
                start_part in bodypart_data.bodyparts
                and end_part in bodypart_data.bodyparts
            ):
                start_pos = bodypart_data.bodyparts[start_part][frame_idx]
                end_pos = bodypart_data.bodyparts[end_part][frame_idx]

                if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                    continue

                skeleton_segments.append([start_pos, end_pos])

        if skeleton_segments:
            skeleton_lc.set_segments(skeleton_segments)
            skeleton_lc.set_visible(True)
        else:
            skeleton_lc.set_segments([])
            skeleton_lc.set_visible(False)

    def _update_head_direction(
        self,
        idx: int,
        head_dir_data: Any,
        frame_idx: int,
        position_data: Any | None,
    ) -> None:
        """Update head direction quiver (recreated each frame)."""
        # Remove old quiver if it exists
        old_quiver = self._head_direction_quivers[idx]
        if old_quiver is not None:
            old_quiver.remove()

        # Create new quiver for this frame
        new_quiver = self._create_head_direction_quiver(
            head_dir_data, frame_idx, position_data
        )
        self._head_direction_quivers[idx] = new_quiver

    def clear(self) -> None:
        """Remove all overlay artists from axes.

        Call this when disposing of the manager or resetting the figure.
        """
        # Remove trail collections
        for trail_lc in self._position_trails:
            if trail_lc is not None:
                trail_lc.remove()
        self._position_trails.clear()

        # Remove position markers
        for marker in self._position_markers:
            marker.remove()
        self._position_markers.clear()

        # Remove bodypart points
        for points in self._bodypart_points:
            points.remove()
        self._bodypart_points.clear()

        # Remove bodypart skeletons
        for skeleton_lc in self._bodypart_skeletons:
            if skeleton_lc is not None:
                skeleton_lc.remove()
        self._bodypart_skeletons.clear()

        # Remove head direction quivers
        for quiver in self._head_direction_quivers:
            if quiver is not None:
                quiver.remove()
        self._head_direction_quivers.clear()

        # Remove region patches
        for patch in self._region_patches:
            patch.remove()
        self._region_patches.clear()

        self._initialized = False


def _render_all_overlays(
    ax: Any,
    env: Any,
    frame_idx: int,
    overlay_data: Any | None,
    show_regions: bool | list[str],
    region_alpha: float,
) -> None:
    """Render all overlays for a single frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to render on
    env : Environment
        Environment for spatial context
    frame_idx : int
        Current frame index
    overlay_data : OverlayData | None
        Overlay data containing positions, bodyparts, head directions, videos
    show_regions : bool | list[str]
        Region display configuration
    region_alpha : float
        Alpha transparency for regions

    Notes
    -----
    This function orchestrates rendering of all overlay types for a frame.
    Called before saving each frame in the worker process.

    Rendering order (controlled by zorder):
    1. Video backgrounds (z_order="below", zorder=-1)
    2. Regions (zorder=99)
    3. Position/bodypart overlays (zorder=100-102)
    4. Head direction overlays (zorder=103)
    5. Event overlays (zorder=104)
    6. Video foregrounds (z_order="above", zorder=1)
    """
    # Early return if no overlay data
    if overlay_data is None:
        # Still render regions even without overlay data
        _render_regions_matplotlib(ax, env, show_regions, region_alpha)
        return

    # Render video backgrounds first (z_order="below")
    for video_data in overlay_data.videos:
        if video_data.z_order == "below":
            _render_video_background(ax, video_data, frame_idx)

    # Render regions
    _render_regions_matplotlib(ax, env, show_regions, region_alpha)

    # Check if overlay_data has any actual overlays (excluding videos)
    overlay_data_present = (
        len(overlay_data.positions) > 0
        or len(overlay_data.bodypart_sets) > 0
        or len(overlay_data.head_directions) > 0
        or len(overlay_data.events) > 0
    )

    if overlay_data_present:
        # Render position overlays
        for pos_data in overlay_data.positions:
            _render_position_overlay_matplotlib(ax, pos_data, frame_idx)

        # Render bodypart overlays
        for bodypart_data in overlay_data.bodypart_sets:
            _render_bodypart_overlay_matplotlib(ax, bodypart_data, frame_idx)

        # Render head direction overlays
        # Auto-pair with position overlay when there's exactly one position
        # (consistent with napari backend behavior)
        paired_position = (
            overlay_data.positions[0] if len(overlay_data.positions) == 1 else None
        )
        for head_dir_data in overlay_data.head_directions:
            _render_head_direction_overlay_matplotlib(
                ax, head_dir_data, frame_idx, env, position_data=paired_position
            )

        # Render event overlays (zorder=104, above head direction)
        for event_data in overlay_data.events:
            _render_event_overlay_matplotlib(ax, event_data, frame_idx)

    # Render video foregrounds last (z_order="above")
    for video_data in overlay_data.videos:
        if video_data.z_order == "above":
            _render_video_background(ax, video_data, frame_idx)


def parallel_render_frames(
    env: Environment,
    fields: list[NDArray[np.float64]],
    output_dir: str,
    cmap: str,
    vmin: float,
    vmax: float,
    frame_labels: list[str] | None,
    dpi: int,
    n_workers: int,
    reuse_artists: bool = True,
    overlay_data: Any | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
    frame_times: NDArray[np.float64] | None = None,
) -> str:
    """Render frames in parallel across worker processes.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure. Must be pickle-able
        (will be serialized to workers). Call env.clear_cache() first.
    fields : list of ndarray of shape (n_bins,), dtype float64
        All fields to render. Each array contains field values for one frame.
    output_dir : str
        Directory to save frame PNGs.
    cmap : str
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float
        Minimum value for color scale normalization.
    vmax : float
        Maximum value for color scale normalization.
    frame_labels : list of str or None
        Frame labels for each frame.
    dpi : int
        Resolution for rendering in dots per inch.
    n_workers : int
        Number of parallel workers.
    reuse_artists : bool, default=True
        If True, reuse the same AxesImage artist across frames (fast path).
        Updates image data only, avoiding layout/allocation overhead.
        If False, clear and redraw each frame (original behavior).
    overlay_data : OverlayData or None, optional
        Overlay data to render on top of fields. Default is None.
    show_regions : bool or list of str, default=False
        If True, render all regions. If list, render specified regions only.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque.

    Returns
    -------
    frame_pattern : str
        ffmpeg input pattern (e.g., "/tmp/frame_%05d.png")

    Raises
    ------
    ValueError
        If environment or overlay_data is not pickle-able when n_workers > 1.
        Error message includes WHAT/WHY/HOW format with actionable solutions:
        - For environment: Call env.clear_cache() or use n_workers=1
        - For overlay_data: Remove unpickleable objects or use n_workers=1

    Examples
    --------
    .. code-block:: python

        import tempfile
        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = parallel_render_frames(
                env, fields, tmpdir, "viridis", 0.0, 1.0, None, 100, 2
            )
            print("frame_" in pattern and ".png" in pattern)
            # True
    """
    import pickle

    n_frames = len(fields)

    # Cap workers to available frames
    n_workers = min(n_workers, max(1, n_frames))

    # Validate pickle-ability for parallel rendering (n_workers > 1)
    if n_workers > 1:
        # Validate environment is pickle-able
        try:
            pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            # WHAT: Environment not pickle-able
            # WHY: Parallel rendering requires pickling to send to workers
            # HOW: Call env.clear_cache() or use n_workers=1
            raise ValueError(
                f"WHAT: Environment is not pickle-able for parallel rendering.\n"
                f"WHY: Parallel rendering (n_workers={n_workers}) requires serializing "
                f"the environment to send to worker processes.\n"
                f"HOW: Choose one of these solutions:\n"
                f"  1. Call env.clear_cache() to remove unpickleable cached objects\n"
                f"  2. Use n_workers=1 for serial rendering (no pickling required)\n"
                f"Original error: {e}"
            ) from e

        # Validate overlay_data is pickle-able (if provided)
        if overlay_data is not None:
            try:
                pickle.dumps(overlay_data, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                # WHAT: overlay_data not pickle-able
                # WHY: Parallel rendering requires pickling overlay data
                # HOW: Remove unpickleable objects or use n_workers=1
                raise ValueError(
                    f"WHAT: overlay_data is not pickle-able for parallel rendering.\n"
                    f"WHY: Parallel rendering (n_workers={n_workers}) requires serializing "
                    f"overlay_data to send to worker processes.\n"
                    f"HOW: Choose one of these solutions:\n"
                    f"  1. Remove unpickleable objects (lambdas, closures, local functions)\n"
                    f"  2. Ensure overlay_data uses only standard types (numpy arrays, "
                    f"strings, numbers)\n"
                    f"  3. Use n_workers=1 for serial rendering (no pickling required)\n"
                    f"Original error: {e}"
                ) from e

    # Partition frames across workers
    frames_per_worker = n_frames // n_workers
    worker_tasks = []

    for worker_id in range(n_workers):
        start_idx = worker_id * frames_per_worker
        if worker_id == n_workers - 1:
            # Last worker takes remainder
            end_idx = n_frames
        else:
            end_idx = start_idx + frames_per_worker

        worker_fields = fields[start_idx:end_idx]
        worker_frame_labels = frame_labels[start_idx:end_idx] if frame_labels else None

        worker_tasks.append(
            {
                "env": env,
                "fields": worker_fields,
                "start_frame_idx": start_idx,
                "output_dir": output_dir,
                "cmap": cmap,
                "vmin": vmin,
                "vmax": vmax,
                "frame_labels": worker_frame_labels,
                "dpi": dpi,
                "digits": max(5, len(str(max(0, n_frames - 1)))),  # Pass to workers
                "reuse_artists": reuse_artists,
                "overlay_data": overlay_data,
                "show_regions": show_regions,
                "region_alpha": region_alpha,
                "scale_bar": scale_bar,
                "frame_times": frame_times,  # For time series rendering
            }
        )

    # Render in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(
            tqdm(
                executor.map(_render_worker_frames, worker_tasks),
                total=n_workers,
                desc="Workers",
            )
        )

    # Return ffmpeg pattern (0-indexed for compatibility)
    # ffmpeg expects: frame_00000.png, frame_00001.png, etc.
    digits = max(5, len(str(max(0, n_frames - 1))))
    pattern = str(Path(output_dir) / f"frame_%0{digits}d.png")

    return pattern


def _render_worker_frames(task: dict) -> None:
    """Render frames in a worker process.

    Each worker creates its own matplotlib figure to avoid
    threading issues and memory accumulation.

    Parameters
    ----------
    task : dict
        Worker task specification with keys:
        - env: Environment
        - fields: list of fields to render
        - start_frame_idx: global frame index offset
        - output_dir: where to save PNGs
        - cmap, vmin, vmax: colormap settings
        - frame_labels: optional frame labels
        - dpi: resolution
        - digits: number of digits for frame padding

    Notes
    -----
    This function is called by ProcessPoolExecutor and must be
    pickle-able (i.e., defined at module level, not nested).

    Examples
    --------
    .. code-block:: python

        import tempfile
        from pathlib import Path
        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            task = {
                "env": env,
                "fields": fields,
                "start_frame_idx": 0,
                "output_dir": tmpdir,
                "cmap": "viridis",
                "vmin": 0.0,
                "vmax": 1.0,
                "frame_labels": None,
                "dpi": 50,
                "digits": 5,
            }
            _render_worker_frames(task)
            png_files = list(Path(tmpdir).glob("frame_*.png"))
            len(png_files)
            # 3
    """
    # Set Agg backend BEFORE any pyplot imports
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in (
            "agg",
            "module://matplotlib_inline.backend_inline",
        ):
            matplotlib.use("Agg", force=True)
    except Exception:
        pass

    # Import pyplot only AFTER backend is set (avoids GUI backend binding in workers)
    import matplotlib.pyplot as plt
    from matplotlib.image import AxesImage

    env = task["env"]
    fields = task["fields"]
    start_idx = task["start_frame_idx"]
    output_dir = task["output_dir"]
    cmap = task["cmap"]
    vmin = task["vmin"]
    vmax = task["vmax"]
    frame_labels = task["frame_labels"]
    dpi = task["dpi"]
    # Get digits from task, fallback for backward compatibility with tests
    digits = task.get("digits", 5)
    reuse_flag = task.get("reuse_artists", True)
    # Extract overlay parameters
    overlay_data = task.get("overlay_data")
    show_regions = task.get("show_regions", False)
    region_alpha = task.get("region_alpha", 0.3)
    scale_bar = task.get("scale_bar", False)
    # Time series parameters
    frame_times = task.get("frame_times")

    # Lean rcParams for bulk rasterization
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.xmargin": 0,
            "axes.ymargin": 0,
            "path.simplify": True,
            "path.simplify_threshold": 0.5,
            "agg.path.chunksize": 10000,
        }
    )

    # Check if we have time series data to render
    timeseries_data = overlay_data.timeseries if overlay_data is not None else []
    has_timeseries = len(timeseries_data) > 0 and frame_times is not None

    # Create figure - use GridSpec layout if time series present
    ts_manager: Any = None  # TimeSeriesArtistManager or None
    if has_timeseries and frame_times is not None:
        from neurospatial.animation._timeseries import (
            _setup_video_figure_with_timeseries,
        )

        fig, ax, ts_manager = _setup_video_figure_with_timeseries(
            env=env,
            timeseries_data=timeseries_data,
            frame_times=frame_times,  # Type narrowing: frame_times is not None here
            dpi=dpi,
            figsize=(12, 6),  # Wider to accommodate time series column
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        ax.set_axis_off()

    # Render first frame normally to establish artists and limits
    env.plot_field(
        fields[0],
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        scale_bar=scale_bar,
    )

    # Add label for first frame if provided
    if frame_labels and frame_labels[0]:
        ax.set_title(frame_labels[0], fontsize=14)

    # Try to identify the primary image artist (field) to update.
    # Use ax.images[-1] (last image) because env.plot_field() is called last,
    # ensuring the field artist is the most recently added image.
    # This supports future video overlay rendering before the field (z_order="below").
    primary_im: AxesImage | None = ax.images[-1] if ax.images else None
    reuse_artists = bool(reuse_flag and primary_im is not None)

    # Create overlay artist manager for efficient artist reuse
    # (only when we have overlays and are using artist reuse path)
    overlay_manager: OverlayArtistManager | None = None
    if reuse_artists and (overlay_data is not None or show_regions):
        overlay_manager = OverlayArtistManager(
            ax=ax,
            env=env,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
        )
        overlay_manager.initialize(frame_idx=0)
    else:
        # Render overlays for frame 0 using legacy path
        _render_all_overlays(ax, env, 0, overlay_data, show_regions, region_alpha)

    # Freeze autoscale to avoid changing extents while updating data
    if reuse_artists:
        ax.set_autoscale_on(False)

    # Initialize time series for frame 0
    if ts_manager is not None:
        ts_manager.update(0, timeseries_data)

    try:
        # Save frame 0
        frame_number = start_idx
        filename = f"frame_{frame_number:0{digits}d}.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath)

        if reuse_artists:
            # Fast path: update the image data only
            # Type checker: reuse_artists=True guarantees primary_im is not None
            assert primary_im is not None
            for local_idx in range(1, len(fields)):
                field = fields[local_idx]

                # Matplotlib wants array-like; ensure C-order to avoid copies later
                # If `field` is masked or not C-contiguous, ascontiguousarray avoids hidden copies
                if not isinstance(field, np.ndarray) or not field.flags["C_CONTIGUOUS"]:
                    data = np.ascontiguousarray(np.array(field))
                else:
                    data = field

                primary_im.set_data(data)  # reuse the same artist

                # Update title if labels provided
                if frame_labels and frame_labels[local_idx]:
                    ax.set_title(frame_labels[local_idx], fontsize=14)

                # Update overlay artists efficiently (no clear/recreate)
                if overlay_manager is not None:
                    overlay_manager.update_frame(local_idx)
                else:
                    # Fallback: clear and re-render (no overlays case)
                    _clear_overlay_artists(ax)
                    _render_all_overlays(
                        ax, env, local_idx, overlay_data, show_regions, region_alpha
                    )

                # Update time series artists if present
                if ts_manager is not None:
                    ts_manager.update(local_idx, timeseries_data)

                # No need to clear or re-layout. Draw and save.
                frame_number = start_idx + local_idx
                filename = f"frame_{frame_number:0{digits}d}.png"
                filepath = Path(output_dir) / filename
                fig.savefig(filepath)
        else:
            # Fallback: redraw per frame (original behavior)
            for local_idx in range(1, len(fields)):
                ax.clear()
                ax.set_axis_off()
                env.plot_field(
                    fields[local_idx],
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    scale_bar=scale_bar,
                )

                # Add label if provided
                if frame_labels and frame_labels[local_idx]:
                    ax.set_title(frame_labels[local_idx], fontsize=14)

                # Render overlays for this frame
                _render_all_overlays(
                    ax, env, local_idx, overlay_data, show_regions, region_alpha
                )

                # Update time series artists if present
                if ts_manager is not None:
                    ts_manager.update(local_idx, timeseries_data)

                frame_number = start_idx + local_idx
                filename = f"frame_{frame_number:0{digits}d}.png"
                filepath = Path(output_dir) / filename
                fig.savefig(filepath)
    finally:
        # Clean up figure (prevent memory leaks)
        plt.close(fig)
