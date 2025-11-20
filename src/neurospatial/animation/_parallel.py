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
- 103: Head direction arrows (foreground)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


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
            from matplotlib.collections import LineCollection
            from matplotlib.colors import to_rgba

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
    from matplotlib.collections import LineCollection

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
    if bodypart_data.skeleton:
        skeleton_segments = []

        for start_part, end_part in bodypart_data.skeleton:
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
                colors=bodypart_data.skeleton_color,
                linewidths=bodypart_data.skeleton_width,
                zorder=101,
            )
            ax.add_collection(lc)


def _render_head_direction_overlay_matplotlib(
    ax: Any, head_dir_data: Any, frame_idx: int, env: Any
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

    Notes
    -----
    Renders direction as an arrow using matplotlib quiver.
    """
    # Determine if data is angles or vectors
    is_angles = head_dir_data.data.ndim == 1

    # Get centroid of environment for arrow origin
    centroid = np.mean(env.bin_centers, axis=0)

    # Compute direction vector
    if is_angles:
        angle = head_dir_data.data[frame_idx]
        direction = np.array([np.cos(angle), np.sin(angle)]) * head_dir_data.length
    else:
        direction = head_dir_data.data[frame_idx]
        # Normalize and scale
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm * head_dir_data.length

    # Render arrow using quiver
    ax.quiver(
        centroid[0],
        centroid[1],
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
    from matplotlib.patches import Circle, PathPatch
    from matplotlib.path import Path as MplPath

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
        Overlay data containing positions, bodyparts, head directions
    show_regions : bool | list[str]
        Region display configuration
    region_alpha : float
        Alpha transparency for regions

    Notes
    -----
    This function orchestrates rendering of all overlay types for a frame.
    Called before saving each frame in the worker process.
    """
    # Render regions
    _render_regions_matplotlib(ax, env, show_regions, region_alpha)

    # Early return if no overlay data
    if overlay_data is None:
        return

    # Check if overlay_data has any actual overlays
    overlay_data_present = (
        len(overlay_data.positions) > 0
        or len(overlay_data.bodypart_sets) > 0
        or len(overlay_data.head_directions) > 0
    )

    if not overlay_data_present:
        return

    # Render position overlays
    for pos_data in overlay_data.positions:
        _render_position_overlay_matplotlib(ax, pos_data, frame_idx)

    # Render bodypart overlays
    for bodypart_data in overlay_data.bodypart_sets:
        _render_bodypart_overlay_matplotlib(ax, bodypart_data, frame_idx)

    # Render head direction overlays
    for head_dir_data in overlay_data.head_directions:
        _render_head_direction_overlay_matplotlib(ax, head_dir_data, frame_idx, env)


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
) -> str:
    """Render frames in parallel across worker processes.

    Parameters
    ----------
    env : Environment
        Must be pickle-able (will be serialized to workers)
    fields : list of arrays
        All fields to render
    output_dir : str
        Directory to save frame PNGs
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    frame_labels : list of str or None
        Frame labels for each frame
    dpi : int
        Resolution
    n_workers : int
        Number of parallel workers
    reuse_artists : bool, default=True
        If True, reuse the same AxesImage artist across frames (fast path).
        Updates image data only, avoiding layout/allocation overhead.
        If False, clear and redraw each frame (original behavior).
    overlay_data : OverlayData | None, optional
        Overlay data to render on top of fields. Default is None.
    show_regions : bool | list[str], default=False
        If True, render all regions. If list, render specified regions only.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays (0-1).

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

    # Create figure once for this worker
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
    )

    # Add label for first frame if provided
    if frame_labels and frame_labels[0]:
        ax.set_title(frame_labels[0], fontsize=14)

    # Render overlays for frame 0
    _render_all_overlays(ax, env, 0, overlay_data, show_regions, region_alpha)

    # Try to identify the primary image artist to update
    # This assumes env.plot_field uses imshow/Images; otherwise we fall back
    primary_im: AxesImage | None = ax.images[0] if ax.images else None
    reuse_artists = bool(reuse_flag and primary_im is not None)

    # Freeze autoscale to avoid changing extents while updating data
    if reuse_artists:
        ax.set_autoscale_on(False)

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

                # Clear overlay artists from previous frame (keep primary image)
                _clear_overlay_artists(ax)

                # Render overlays for this frame
                _render_all_overlays(
                    ax, env, local_idx, overlay_data, show_regions, region_alpha
                )

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
                )

                # Add label if provided
                if frame_labels and frame_labels[local_idx]:
                    ax.set_title(frame_labels[local_idx], fontsize=14)

                # Render overlays for this frame
                _render_all_overlays(
                    ax, env, local_idx, overlay_data, show_regions, region_alpha
                )

                frame_number = start_idx + local_idx
                filename = f"frame_{frame_number:0{digits}d}.png"
                filepath = Path(output_dir) / filename
                fig.savefig(filepath)
    finally:
        # Clean up figure (prevent memory leaks)
        plt.close(fig)
