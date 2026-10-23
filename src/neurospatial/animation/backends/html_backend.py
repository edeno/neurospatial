"""Standalone HTML player backend with instant scrubbing.

Overlay rendering
-----------------
Positions and trails, body parts and skeletons, head directions, events
(including decay), and regions are rendered into each image by Matplotlib.
Both embedded and external-frame exports support these overlays; JavaScript
provides playback controls without drawing a second overlay layer. Coordinates
are in environment units.

Video, time series, and object-vector overlays are not implemented by this
backend and are skipped with explicit warnings.

Legacy overlay JSON schema
--------------------------
``_serialize_overlay_data`` converts overlays to a limited JSON schema. No
player draws it: ``render_html`` bakes overlays into its images. The schema is:

.. code-block:: json

    {
        "positions": [
            {
                "data": [[x0, y0], [x1, y1], ...],  // (n_frames, 2) trajectory
                "color": "red",                      // CSS color string
                "size": 10.0,                        // Marker size in pixels
                "trail_length": 5                    // Frames of trail to show
            },
            // ... more position overlays
        ],
        "events": [
            {
                "event_positions": {
                    "spikes": [[x0, y0], [x1, y1], ...]  // Event positions
                },
                "frame_indices": {
                    "spikes": [1, 4, 7, ...]             // Frame indices when events occur
                },
                "colors": {"spikes": "#ff0000"},         // Color per event type
                "size": 8.0,                             // Marker size in pixels
                "border_color": "white",
                "border_width": 0.5
            },
            // ... more event overlays
        ],
        "regions": [
            {
                "name": "goal",
                "kind": "point",                     // "point" or "polygon"
                "coordinates": [x, y]                // Point: [x, y]
            },
            {
                "name": "arena",
                "kind": "polygon",
                "coordinates": [[x0, y0], [x1, y1], ...]  // Polygon vertices
            },
            // ... more regions
        ],
        "dimension_ranges": [[x_min, x_max], [y_min, y_max]],
        "region_alpha": 0.3                          // 0.0-1.0 transparency
    }

Coordinate System
-----------------
All coordinates are in environment units, not pixels. The JavaScript
renderer transforms these to canvas coordinates based on dimension_ranges.

- Position data: (n_frames, 2) array as nested lists, each row is [x, y]
- Event positions: Per-event-type dict of [x, y] coordinates
- Region coordinates: Environment (x, y) coordinates
- dimension_ranges: Bounding box for coordinate scaling

"""

from __future__ import annotations

import base64
import logging
import os
import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from neurospatial.animation._html_player import (
    _generate_html_player,
    _generate_non_embedded_html_player,
)
from neurospatial.animation._region_names import shown_region_names
from neurospatial.animation._utils import _emit_status

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData
    from neurospatial.environment import Environment

logger = logging.getLogger(__name__)


# =============================================================================
# TypedDict definitions for overlay JSON structure
# =============================================================================


class PositionOverlayJSON(TypedDict):
    """JSON structure for a single position overlay.

    Attributes
    ----------
    data : list[list[float]]
        Trajectory data as nested list of [x, y] coordinates per frame.
        Shape: (n_frames, 2) flattened to nested lists.
    color : str
        CSS color string (e.g., "red", "#FF0000", "rgb(255, 0, 0)").
    size : float
        Marker diameter in logical screen pixels.
    trail_length : int
        Number of previous frames to show as decaying trail.
    """

    data: list[list[float]]
    color: str
    size: float
    trail_length: int


class PointRegionJSON(TypedDict):
    """JSON structure for a point region.

    Attributes
    ----------
    name : str
        Region identifier.
    kind : str
        Always "point" for this type.
    coordinates : list[float]
        Point location as [x, y] in environment coordinates.
    """

    name: str
    kind: str  # Literal["point"]
    coordinates: list[float]


class PolygonRegionJSON(TypedDict):
    """JSON structure for a polygon region.

    Attributes
    ----------
    name : str
        Region identifier.
    kind : str
        Always "polygon" for this type.
    coordinates : list[list[float]]
        Polygon vertices as [[x0, y0], [x1, y1], ...] in environment coordinates.
    """

    name: str
    kind: str  # Literal["polygon"]
    coordinates: list[list[float]]


class EventOverlayJSON(TypedDict):
    """JSON structure for a single event overlay.

    Attributes
    ----------
    event_positions : dict[str, list[list[float]]]
        Event positions per event type as nested list of [x, y] coordinates.
        Keys are event type names (e.g., "spikes", "cell_001").
    frame_indices : dict[str, list[int]]
        Frame indices per event type when each event occurs.
    colors : dict[str, str]
        CSS color per event type (e.g., {"spikes": "#ff0000"}).
    size : float
        Marker diameter in logical screen pixels.
    border_color : str
        Border/edge color for markers.
    border_width : float
        Border/edge width in pixels.
    """

    event_positions: dict[str, list[list[float]]]
    frame_indices: dict[str, list[int]]
    colors: dict[str, str]
    size: float
    border_color: str
    border_width: float


class OverlayDataJSON(TypedDict, total=False):
    """Complete JSON structure for overlay data passed to JavaScript.

    This TypedDict documents the structure of the JSON data embedded in HTML
    animations. The JavaScript renderer consumes this structure for drawing
    overlays on the canvas.

    Attributes
    ----------
    positions : list[PositionOverlayJSON]
        List of position overlay objects with trajectory data.
    events : list[EventOverlayJSON]
        List of event overlay objects with event positions and frame indices.
    regions : list[PointRegionJSON | PolygonRegionJSON]
        List of region objects (points or polygons).
    dimension_ranges : list[list[float]]
        Environment bounding box as [[x_min, x_max], [y_min, y_max]].
    region_alpha : float
        Transparency for region rendering (0.0 = transparent, 1.0 = opaque).

    Examples
    --------
    >>> overlay_json: OverlayDataJSON = {
    ...     "positions": [
    ...         {
    ...             "data": [[0, 0], [1, 1]],
    ...             "color": "red",
    ...             "size": 10,
    ...             "trail_length": 5,
    ...         }
    ...     ],
    ...     "events": [
    ...         {
    ...             "event_positions": {"spikes": [[10, 20], [30, 40]]},
    ...             "frame_indices": {"spikes": [1, 4]},
    ...             "colors": {"spikes": "#ff0000"},
    ...             "size": 8,
    ...             "border_color": "white",
    ...             "border_width": 0.5,
    ...         }
    ...     ],
    ...     "regions": [{"name": "goal", "kind": "point", "coordinates": [50, 50]}],
    ...     "dimension_ranges": [[0, 100], [0, 100]],
    ...     "region_alpha": 0.3,
    ... }
    """

    positions: list[PositionOverlayJSON]
    events: list[EventOverlayJSON]
    regions: list[PointRegionJSON | PolygonRegionJSON]
    dimension_ranges: list[list[float]]
    region_alpha: float


def _estimate_overlay_json_size(
    overlay_data: OverlayData | None,
    env: Environment,
    show_regions: bool | list[str],
) -> float:
    """Estimate size of overlay JSON in megabytes.

    Parameters
    ----------
    overlay_data : OverlayData | None
        Overlay data containing positions, bodyparts, and head directions
    env : Environment
        Environment for extracting region data
    show_regions : bool | list[str]
        Whether to include regions in serialization

    Returns
    -------
    estimated_size_mb : float
        Estimated JSON size in megabytes

    Notes
    -----
    Estimation is based on:
    - Position data: ~20 bytes per coordinate (JSON overhead + float representation)
    - Event data: ~25 bytes per event (position + frame index + JSON overhead)
    - Regions: ~30 bytes per vertex (polygon coordinates)
    - Small overhead for metadata (colors, names, etc.)

    The estimate is conservative (slightly overestimates) to ensure warnings
    trigger before actual size becomes problematic.
    """
    if overlay_data is None and not show_regions:
        return 0.0

    total_bytes = 0.0

    # Estimate position overlay size
    if overlay_data is not None:
        for pos_data in overlay_data.positions:
            # Each position: (n_frames, n_dims) floats
            # JSON float: ~15 chars avg, plus brackets/commas
            n_coords = pos_data.data.size  # Total number of coordinates
            bytes_per_coord = 20  # Conservative estimate (includes JSON overhead)
            total_bytes += n_coords * bytes_per_coord

            # Add overhead for color, size, trail_length (small)
            total_bytes += 100  # Metadata per overlay

        # Estimate event overlay size
        for event_data in overlay_data.events:
            # Each event: (n_events, n_dims) floats for positions + frame index
            for _event_type, positions in event_data.event_positions.items():
                n_events = len(positions)
                # Position coordinates (2D) + frame index int
                bytes_per_event = 25  # 2 floats + 1 int + JSON overhead
                total_bytes += n_events * bytes_per_event

            # Add overhead for colors, markers, sizes, borders
            total_bytes += 200  # Metadata per event overlay

    # Estimate region size
    # Prefer normalized format from overlay_data.regions if available
    region_names: list[str] | None = None
    if overlay_data is not None and overlay_data.regions is not None:
        region_names = overlay_data.regions.get(0, [])
    elif show_regions:
        region_names = shown_region_names(env, show_regions)

    if region_names:
        for name in region_names:
            region = env.regions[name]

            if region.kind == "point":
                # Point region: 2 coordinates
                total_bytes += 2 * 20 + 50  # Coords + metadata
            elif region.kind == "polygon":
                # Polygon: estimate ~100 vertices (conservative)
                coords = list(region.data.exterior.coords)  # type: ignore[union-attr]
                n_vertices = len(coords)
                bytes_per_vertex = 30  # Two floats + JSON overhead
                total_bytes += n_vertices * bytes_per_vertex + 100  # Coords + metadata

    # Add overhead for dimension_ranges and other metadata
    total_bytes += 500  # Small fixed overhead

    return total_bytes / 1e6  # Convert to MB


def _serialize_overlay_data(
    overlay_data: OverlayData | None,
    env: Environment,
    show_regions: bool | list[str],
    region_alpha: float,
) -> dict[str, Any]:
    """Serialize overlay data to JSON-compatible format.

    Parameters
    ----------
    overlay_data : OverlayData | None
        Overlay data containing positions, bodyparts, and head directions
    env : Environment
        Environment for extracting region data
    show_regions : bool | list[str]
        Whether to include regions in serialization
    region_alpha : float
        Region transparency value

    Returns
    -------
    overlay_dict : dict
        JSON-compatible dictionary with overlay data

    Notes
    -----
    Positions, events (instant mode), and regions are serialized.
    Bodyparts and head directions are not supported in HTML backend.
    """
    # Get dimension ranges (convert to list for JSON)
    dim_ranges = env.dimension_ranges if env.dimension_ranges is not None else []

    result: dict[str, Any] = {
        "positions": [],
        "events": [],
        "regions": [],
        "dimension_ranges": [list(r) for r in dim_ranges],
        "region_alpha": region_alpha,
    }

    # Serialize position overlays
    if overlay_data is not None:
        for pos_data in overlay_data.positions:
            result["positions"].append(
                {
                    "data": pos_data.data.tolist(),  # Convert to list for JSON
                    "color": pos_data.color,
                    "size": pos_data.size,
                    "trail_length": pos_data.trail_length,
                }
            )

        # Serialize event overlays
        for event_data in overlay_data.events:
            # Convert numpy arrays to lists for JSON serialization
            event_positions_json: dict[str, list[list[float]]] = {}
            frame_indices_json: dict[str, list[int]] = {}

            for event_type, positions in event_data.event_positions.items():
                event_positions_json[event_type] = positions.tolist()

            for event_type, indices in event_data.event_frame_indices.items():
                frame_indices_json[event_type] = indices.tolist()

            result["events"].append(
                {
                    "event_positions": event_positions_json,
                    "frame_indices": frame_indices_json,
                    "colors": event_data.colors,
                    "size": event_data.size,
                    "border_color": event_data.border_color,
                    "border_width": event_data.border_width,
                }
            )

    # Serialize regions
    # Prefer normalized format from overlay_data.regions if available
    region_names: list[str] | None = None
    if overlay_data is not None and overlay_data.regions is not None:
        # Use normalized format (key 0 = all frames)
        region_names = overlay_data.regions.get(0, [])
    elif show_regions:
        # Fall back to show_regions parameter
        region_names = shown_region_names(env, show_regions)

    if region_names:
        for name in region_names:
            region = env.regions[name]

            if region.kind == "point":
                result["regions"].append(
                    {
                        "name": name,
                        "kind": "point",
                        "coordinates": region.data.tolist(),
                    }
                )
            elif region.kind == "polygon":
                # Get exterior coordinates from shapely polygon
                coords = list(region.data.exterior.coords)  # type: ignore[union-attr]
                result["regions"].append(
                    {
                        "name": name,
                        "kind": "polygon",
                        "coordinates": coords,
                    }
                )

    return result


def render_html(
    env: Environment,
    fields: list[NDArray[np.float64]],
    save_path: str | None,
    *,
    overwrite: bool = False,
    fps: int = 30,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: list[str] | None = None,
    dpi: int = 100,
    image_format: str = "png",
    max_html_frames: int = 500,
    title: str = "Spatial Field Animation",
    embed: bool = True,
    frames_dir: str | Path | None = None,
    n_workers: int | None = None,
    overlay_data: OverlayData | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | Any = False,  # bool | ScaleBarConfig
    dry_run: bool = False,
) -> Path | None:
    """Generate standalone HTML player with instant scrubbing.

    Pre-renders all frames as base64-encoded images embedded in HTML (default),
    or writes frames to disk and references them (embed=False for large sequences).
    JavaScript provides play/pause/scrub controls with zero latency.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    fields : list of ndarray of shape (n_bins,), dtype float64
        Fields to animate. Each array contains field values for one frame.
    save_path : str or None
        Output path. If None, defaults to 'animation.html'.
    overwrite : bool, default=False
        Allow replacing existing video/HTML files and nonempty HTML frames
        directories. By default, an existing target raises before rendering.
    fps : int, default=30
        Frames per second for playback.
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float, optional
        Minimum value for color scale normalization. If None, the 1st
        percentile of the frames' minima.
    vmax : float, optional
        Maximum value for color scale normalization. If None, the 99th
        percentile of the frames' maxima, so one extreme frame does not
        wash out the rest; values beyond either limit get its color.
    frame_labels : list of str, optional
        Frame labels for each frame.
    dpi : int, default=100
        Resolution for rendering in dots per inch.
    image_format : {"png", "jpeg"}, default="png"
        Image format for frames. PNG is lossless with sharp edges (ideal for
        scientific visualization). JPEG is lossy but typically 2-10x smaller
        for high-DPI renders (dpi >= 100). External-frame mode always writes PNG.

        **Recommendation**: Use ``image_format='jpeg'`` when file size matters
        and you're using high DPI. For low DPI (<100) or when preserving exact
        values matters, use PNG (default).
    max_html_frames : int, default=500
        Maximum number of frames allowed. Prevents creating huge HTML files
        that crash browsers.
    title : str, default="Spatial Field Animation"
        Animation title displayed in browser.
    embed : bool, default=True
        If True, embed frames as base64 data URLs (self-contained HTML).
        If False, write frames to disk and reference via relative paths
        (smaller HTML). The max_html_frames limit applies to both modes.
    frames_dir : str or Path or None, optional
        Directory to write frames when embed=False. Defaults to a sibling
        directory based on save_path name (e.g., "animation/").
    n_workers : int or None, optional
        Number of parallel workers for frame rendering (embed=False only).
        Defaults to CPU count / 2.
    overlay_data : OverlayData or None, optional
        Aligned positions (with trails), body parts and skeletons, head
        directions, and events (including decay), rendered into the frame
        images in both embed modes. Video, time series, and object-vector
        overlays are skipped with warnings. Default is None.
    show_regions : bool or list of str, default=False
        Whether to show environment regions. If True, show all regions.
        If list, show only regions with names in the list.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque.
    scale_bar : bool or ScaleBarConfig, default=False
        Draw a scale bar in every frame. If True, uses the default config.
    dry_run : bool, default=False
        If True, check the inputs and output paths, report the number of
        frames and the estimated file size, and return None without
        rendering or writing anything.

    Returns
    -------
    save_path : Path or None
        Path to saved HTML file, or None if ``dry_run=True``.

    Raises
    ------
    FileExistsError
        If an output already exists and overwrite is False.
    ValueError
        If number of frames exceeds max_html_frames

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from neurospatial import Environment

        positions = np.random.randn(100, 2) * 50
        env = Environment.from_samples(positions, bin_size=10.0)
        fields = [np.random.rand(env.n_bins) for _ in range(20)]
        frame_times = np.arange(len(fields), dtype=float) / 30.0
        path = env.animate_fields(
            fields, frame_times=frame_times, backend="html", save_path="output.html"
        )
        print(f"HTML player saved to {path}")

    Notes
    -----
    The HTML player includes:
    - Play/pause buttons
    - Frame scrubbing slider
    - Speed control (0.25x to 4x)
    - Keyboard shortcuts (space = play/pause, arrows = step, Home/End = first/last, ,/. = step)
    - Frame counter and labels

    For large datasets, consider:
    - Reducing DPI to decrease file size
    - Using image_format='jpeg' for lossy compression
    - Subsampling frames before export
    - Using video or napari backend instead

    See Also
    --------
    render_video : Export as video file
    render_napari : Interactive GPU viewer
    """
    from neurospatial._validation import check_writable
    from neurospatial.animation.rendering import (
        _validate_frame_labels,
        animation_colormap_range,
        render_field_to_image_bytes,
    )

    output_path = Path("animation.html" if save_path is None else save_path)
    targets = [output_path]
    if not embed:
        frames_dir = (
            Path(frames_dir) if frames_dir is not None else output_path.with_suffix("")
        )
        targets.append(frames_dir)
    check_writable(
        targets,
        overwrite=overwrite,
        what="HTML file or frames directory",
        argument="save_path or frames_dir",
    )

    n_frames = len(fields)

    # Validate frame_labels length if provided
    frame_labels = _validate_frame_labels(
        frame_labels=frame_labels,
        n_frames=n_frames,
        backend_name="html",
    )

    # Use one Matplotlib overlay path in both embed modes. Filter unsupported
    # types without mutating the caller's data or accessing its video readers.
    if overlay_data is not None:
        # Check for unsupported video overlays
        has_video_overlays = len(overlay_data.videos) > 0
        if has_video_overlays:
            warnings.warn(
                "HTML backend does not support video overlays.\n"
                "\n"
                "WHAT: This HTML exporter does not implement video compositing.\n"
                "WHY: Video layers are currently supported by the video and napari backends.\n"
                "\n"
                "Fix:\n"
                "  1. Use video backend for full video overlay support:\n"
                "     env.animate_fields(fields, frame_times=frame_times,\n"
                "                        backend='video', save_path='output.mp4',\n"
                "                        overlays=[video_overlay], ...)\n"
                "\n"
                "  2. Use napari backend for interactive viewing:\n"
                "     env.animate_fields(fields, frame_times=frame_times,\n"
                "                        backend='napari', overlays=[video_overlay])\n"
                "\n"
                "Video overlays will be skipped. Other supported overlays will render.",
                UserWarning,
                stacklevel=2,
            )

        # Check for unsupported time series overlays
        has_timeseries_overlays = len(overlay_data.timeseries) > 0
        if has_timeseries_overlays:
            warnings.warn(
                "HTML backend does not support time series overlays.\n"
                "\n"
                "WHAT: This HTML exporter does not implement time series panels.\n"
                "WHY: The HTML exporter renders a single spatial panel.\n"
                "\n"
                "Fix:\n"
                "  1. Use video backend for time series support:\n"
                "     env.animate_fields(fields, frame_times=frame_times,\n"
                "                        backend='video', save_path='output.mp4',\n"
                "                        overlays=[timeseries_overlay], ...)\n"
                "\n"
                "  2. Use napari backend for interactive viewing:\n"
                "     env.animate_fields(fields, frame_times=frame_times, backend='napari',\n"
                "                        overlays=[timeseries_overlay])\n"
                "\n"
                "Time series overlays will be skipped. Other overlays will render.",
                UserWarning,
                stacklevel=2,
            )

        if overlay_data.object_vectors:
            warnings.warn(
                "HTML backend does not support object-vector overlays. "
                "They will be skipped. Use backend='fastplotlib' to view these "
                "overlays, or Animation.export() to include them in an export.",
                UserWarning,
                stacklevel=2,
            )

        if has_video_overlays or has_timeseries_overlays or overlay_data.object_vectors:
            overlay_data = replace(
                overlay_data, videos=[], timeseries=[], object_vectors=[]
            )
    elif show_regions:
        # The single-frame renderer enters its overlay path only for a data
        # container; a regions-only export still needs that path.
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData()

    # Normalize image_format
    image_format = image_format.lower()
    if image_format not in ("png", "jpeg"):
        raise ValueError(f"image_format must be 'png' or 'jpeg', got '{image_format}'")

    # Estimate file size BEFORE rendering
    # JPEG compression effectiveness depends on image size and content:
    # - Large images (DPI >= 100): 5-10x smaller than PNG
    # - Medium images (DPI 75-100): 2-5x smaller
    # - Small images (DPI < 75): PNG may be smaller due to JPEG overhead
    # Use conservative estimate that works across range
    size_factor = 0.05 if image_format == "jpeg" else 0.1
    estimated_mb = n_frames * size_factor * (dpi / 100) ** 2

    # Hard limit check
    if n_frames > max_html_frames:
        raise ValueError(
            f"HTML backend supports max {max_html_frames} frames (got {n_frames}).\n"
            f"Estimated file size: {estimated_mb:.0f} MB\n"
            f"\n"
            f"Options:\n"
            f"  1. Subsample frames:\n"
            f"     fields_subset = fields[::10]  # Every 10th frame\n"
            f"     subset_times = frame_times[::10]\n"
            f"     env.animate_fields(fields_subset, frame_times=subset_times, backend='html', ...)\n"
            f"\n"
            f"  2. Use video backend:\n"
            f"     env.animate_fields(fields, frame_times=frame_times, backend='video', save_path='output.mp4')\n"
            f"\n"
            f"  3. Use Napari for interactive viewing:\n"
            f"     env.animate_fields(fields, frame_times=frame_times, backend='napari')\n"
            f"\n"
            f"  4. Override limit (NOT RECOMMENDED):\n"
            f"     env.animate_fields(fields, frame_times=frame_times, backend='html', max_html_frames={n_frames})\n"
        )

    # Warn about large files (with JPEG recommendation)
    if estimated_mb > 50:
        jpeg_hint = (
            ""
            if image_format == "jpeg"
            else "  - Use image_format='jpeg' (typically 2-10x smaller, best for DPI >= 100)\n"
        )
        warnings.warn(
            f"\nHTML export will create a large file (~{estimated_mb:.0f} MB with {image_format.upper()}).\n"
            f"Consider:\n"
            f"  - Reduce DPI: dpi=50 (current: {dpi})\n"
            f"  - Subsample frames: fields[::5]\n"
            f"{jpeg_hint}",
            UserWarning,
            stacklevel=2,
        )

    if dry_run:
        report = [
            "=" * 60,
            "HTML Export Dry Run Estimate:",
            "=" * 60,
            f"  Frames:          {n_frames:,}",
            f"  Image format:    {image_format.upper() if embed else 'PNG'}",
            f"  Est. file size:  {estimated_mb:.0f} MB",
            f"  Output path:     {output_path}",
        ]
        if not embed:
            report.append(f"  Frames directory: {frames_dir}")
        report += ["", "To proceed, call again with dry_run=False", "=" * 60]
        _emit_status("\n".join(report))
        return None

    vmin, vmax = animation_colormap_range(fields, vmin, vmax)

    # ---- Non-embedded mode: write frames to disk, lightweight HTML ----
    if not embed:
        from neurospatial.animation._parallel import parallel_render_frames

        assert isinstance(frames_dir, Path)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of workers
        if n_workers is None:
            cpu_count = os.cpu_count() or 2  # Default to 2 if cpu_count() returns None
            n_workers = max(1, cpu_count // 2)
        elif n_workers < 1:
            raise ValueError(f"n_workers must be positive (got {n_workers})")

        # Use parallel renderer for speed & consistency
        logger.info("Rendering %d frames to %s...", n_frames, frames_dir)
        _ = parallel_render_frames(
            env=env,
            fields=fields,
            output_dir=str(frames_dir),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            frame_labels=frame_labels,
            dpi=dpi,
            n_workers=n_workers,
            overlay_data=overlay_data,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
        )

        # Generate frame labels if not provided
        if frame_labels is None:
            frame_labels = [f"Frame {i + 1}" for i in range(n_frames)]

        # Generate lightweight HTML with relative frame references
        html = _generate_non_embedded_html_player(
            frames_dir=frames_dir,
            output_path=output_path,
            n_frames=n_frames,
            frame_labels=frame_labels,
            fps=fps,
            title=title,
            image_format="png",  # parallel_render_frames always uses PNG
        )

        # Write HTML
        output_path.write_text(html, encoding="utf-8")

        logger.info("✓ HTML saved to %s (frames in %s)", output_path, frames_dir)
        return output_path

    # ---- Embedded mode (original behavior) ----
    # Pre-render all frames to base64 with overlays baked in
    logger.info("Rendering %d frames to %s...", n_frames, image_format.upper())
    frames_b64 = []

    for frame_idx, field in enumerate(tqdm(fields, desc="Encoding frames")):
        # Render field with overlays directly in matplotlib (not JavaScript)
        image_bytes = render_field_to_image_bytes(
            env,
            field,
            cmap,
            vmin,
            vmax,
            dpi,
            image_format=image_format,
            overlay_data=overlay_data,
            frame_idx=frame_idx,
            show_regions=show_regions,
            region_alpha=region_alpha,
            scale_bar=scale_bar,
        )

        # Convert to base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        frames_b64.append(b64)

    # Generate frame labels
    if frame_labels is None:
        frame_labels = [f"Frame {i + 1}" for i in range(len(fields))]

    # Create HTML without JavaScript overlay data (overlays are baked into images)
    html = _generate_html_player(
        frames_b64=frames_b64,
        frame_labels=frame_labels,
        fps=fps,
        title=title,
        image_format=image_format,
    )

    # Write to file with UTF-8 encoding (for Unicode button symbols on Windows)
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / 1e6
    logger.info("✓ HTML saved to %s (%.1f MB)", output_path, file_size_mb)

    return output_path
