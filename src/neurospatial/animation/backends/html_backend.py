"""Standalone HTML player backend with instant scrubbing.

Overlay JSON Schema
-------------------
The HTML backend serializes overlay data to JSON for rendering in the browser.
The schema is:

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
All coordinates are in environment space (cm), NOT pixels. The JavaScript
renderer transforms these to canvas coordinates based on dimension_ranges.

- Position data: (n_frames, 2) array as nested lists, each row is [x, y]
- Event positions: Per-event-type dict of [x, y] coordinates
- Region coordinates: Environment (x, y) coordinates
- dimension_ranges: Bounding box for coordinate scaling

Limitations
-----------
- Only PositionOverlay, EventOverlay (instant mode), and regions are supported
- EventOverlay decay_frames > 0 falls back to instant mode (warning emitted)
- BodypartOverlay and HeadDirectionOverlay are silently skipped (warning emitted)
- VideoOverlay is not supported (warning emitted)
- Large overlay datasets (>1 MB JSON) may cause browser rendering delays
"""

from __future__ import annotations

import base64
import html as html_module
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from neurospatial.animation.overlays import OverlayData
    from neurospatial.environment.core import Environment


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
        Marker size in environment units.
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
        Marker size in environment units.
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
        region_names = (
            show_regions if isinstance(show_regions, list) else list(env.regions.keys())
        )

    if region_names:
        for name in region_names:
            if name not in env.regions:
                continue
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
        region_names = (
            show_regions if isinstance(show_regions, list) else list(env.regions.keys())
        )

    if region_names:
        for name in region_names:
            if name not in env.regions:
                continue
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
    **kwargs,  # Accept other parameters gracefully
) -> Path:
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
    fps : int, default=30
        Frames per second for playback.
    cmap : str, default="viridis"
        Matplotlib colormap name (e.g., "viridis", "hot", "plasma").
    vmin : float, optional
        Minimum value for color scale normalization. If None, computed from
        all fields using NaN-robust min.
    vmax : float, optional
        Maximum value for color scale normalization. If None, computed from
        all fields using NaN-robust max.
    frame_labels : list of str, optional
        Frame labels for each frame.
    dpi : int, default=100
        Resolution for rendering in dots per inch.
    image_format : {"png", "jpeg"}, default="png"
        Image format for frames. PNG is lossless with sharp edges (ideal for
        scientific visualization). JPEG is lossy but typically 2-10x smaller
        for high-DPI renders (dpi >= 100).

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
        (smaller HTML, suitable for 500+ frames).
    frames_dir : str or Path or None, optional
        Directory to write frames when embed=False. Defaults to a sibling
        directory based on save_path name (e.g., "animation_frames/").
    n_workers : int or None, optional
        Number of parallel workers for frame rendering (embed=False only).
        Defaults to CPU count / 2.
    overlay_data : OverlayData or None, optional
        Overlay data for rendering positions and regions. HTML backend supports:
        - Position overlays (with trails)
        - Region overlays
        Bodypart and head direction overlays are not supported and will emit warnings.
        Default is None.
    show_regions : bool or list of str, default=False
        Whether to show environment regions. If True, show all regions.
        If list, show only regions with names in the list.
    region_alpha : float, default=0.3
        Alpha transparency for region overlays, range [0.0, 1.0] where 0.0 is
        fully transparent and 1.0 is fully opaque.
    **kwargs
        Other parameters (accepted gracefully for compatibility).

    Returns
    -------
    save_path : Path
        Path to saved HTML file

    Raises
    ------
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
        path = env.animate_fields(fields, backend="html", save_path="output.html")
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
    from neurospatial.animation.rendering import (
        _validate_frame_labels,
        compute_global_colormap_range,
        render_field_to_image_bytes,
    )

    n_frames = len(fields)

    # Validate frame_labels length if provided
    frame_labels = _validate_frame_labels(
        frame_labels=frame_labels,
        n_frames=n_frames,
        backend_name="html",
    )

    # Validate overlay capabilities and emit warnings for unsupported types
    # Note: Most overlay types (positions, bodyparts, head direction, events)
    # are now rendered via matplotlib and are fully supported.
    # Only video overlays require special handling.
    if overlay_data is not None:
        # Check for unsupported video overlays
        has_video_overlays = len(overlay_data.videos) > 0
        if has_video_overlays:
            warnings.warn(
                "HTML backend does not support video overlays.\n"
                "\n"
                "WHAT: Video overlays cannot be rendered in standalone HTML files.\n"
                "WHY: HTML embeds frames as base64 images; video compositing requires\n"
                "     frame-by-frame rendering which is only supported in video/napari.\n"
                "\n"
                "HOW to fix:\n"
                "  1. Use video backend for full video overlay support:\n"
                "     env.animate_fields(fields, backend='video', save_path='output.mp4',\n"
                "                        overlays=[video_overlay], ...)\n"
                "\n"
                "  2. Use napari backend for interactive viewing:\n"
                "     env.animate_fields(fields, backend='napari', overlays=[video_overlay])\n"
                "\n"
                "Video overlays will be skipped. Other overlays (positions, regions) will render.",
                UserWarning,
                stacklevel=2,
            )

        # Check for unsupported time series overlays
        has_timeseries_overlays = len(overlay_data.timeseries) > 0
        if has_timeseries_overlays:
            warnings.warn(
                "HTML backend does not support time series overlays.\n"
                "\n"
                "WHAT: Time series overlays require a separate panel with scrolling\n"
                "      time series plots, which cannot be rendered in HTML.\n"
                "WHY: HTML embeds frames as static base64 images. Time series display\n"
                "     requires dynamic chart updates and multi-panel layout.\n"
                "\n"
                "HOW to fix:\n"
                "  1. Use video backend for time series support:\n"
                "     env.animate_fields(fields, backend='video', save_path='output.mp4',\n"
                "                        overlays=[timeseries_overlay], ...)\n"
                "\n"
                "  2. Use napari backend for interactive viewing:\n"
                "     env.animate_fields(fields, backend='napari',\n"
                "                        overlays=[timeseries_overlay])\n"
                "\n"
                "Time series overlays will be skipped. Other overlays will render.",
                UserWarning,
                stacklevel=2,
            )

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
            f"     env.animate_fields(fields_subset, backend='html', ...)\n"
            f"\n"
            f"  2. Use video backend:\n"
            f"     env.animate_fields(fields, backend='video', save_path='output.mp4')\n"
            f"\n"
            f"  3. Use Napari for interactive viewing:\n"
            f"     env.animate_fields(fields, backend='napari')\n"
            f"\n"
            f"  4. Override limit (NOT RECOMMENDED):\n"
            f"     env.animate_fields(fields, backend='html', max_html_frames={n_frames})\n"
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

    # Estimate overlay JSON size and warn if large (threshold: 1 MB)
    # 1 MB of JSON is substantial for browser rendering - larger datasets
    # should use video or napari backends for better performance
    overlay_size_mb = _estimate_overlay_json_size(overlay_data, env, show_regions)
    if overlay_size_mb > 1.0:
        warnings.warn(
            f"\nHTML overlay data will be large (~{overlay_size_mb:.1f} MB JSON).\n"
            f"Large overlay datasets can slow down browser rendering.\n"
            f"\n"
            f"Consider these alternatives:\n"
            f"  1. Subsample position data:\n"
            f"     # Subsample overlay positions to match reduced frame rate\n"
            f"     positions_subsampled = positions[::5]  # Every 5th position\n"
            f"     overlay = PositionOverlay(data=positions_subsampled, ...)\n"
            f"\n"
            f"  2. Use video backend for full-fidelity overlays:\n"
            f"     env.animate_fields(fields, backend='video', save_path='output.mp4',\n"
            f"                        overlays=[overlay], ...)\n"
            f"\n"
            f"  3. Use Napari for interactive viewing:\n"
            f"     env.animate_fields(fields, backend='napari', overlays=[overlay], ...)\n",
            UserWarning,
            stacklevel=2,
        )

    # Compute global color scale
    vmin_computed, vmax_computed = compute_global_colormap_range(fields, vmin, vmax)
    vmin = vmin if vmin is not None else vmin_computed
    vmax = vmax if vmax is not None else vmax_computed

    # Set default save path if None
    if save_path is None:
        save_path = "animation.html"
    output_path = Path(save_path)

    # ---- Non-embedded mode: write frames to disk, lightweight HTML ----
    if not embed:
        from neurospatial.animation._parallel import parallel_render_frames

        # Default frames directory: sibling folder based on HTML name
        if frames_dir is None:
            frames_dir = output_path.with_suffix(
                ""
            )  # e.g., "animation.html" -> "animation/"
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of workers
        if n_workers is None:
            cpu_count = os.cpu_count() or 2  # Default to 2 if cpu_count() returns None
            n_workers = max(1, cpu_count // 2)
        elif n_workers < 1:
            raise ValueError(f"n_workers must be positive (got {n_workers})")

        # Use parallel renderer for speed & consistency
        print(f"Rendering {n_frames} frames to {frames_dir}...")
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
        )

        # Generate frame labels if not provided
        if frame_labels is None:
            frame_labels = [f"Frame {i + 1}" for i in range(n_frames)]

        # Serialize overlay data
        overlay_json = _serialize_overlay_data(
            overlay_data, env, show_regions, region_alpha
        )

        # Generate lightweight HTML with relative frame references
        html = _generate_non_embedded_html_player(
            frames_dir=frames_dir,
            n_frames=n_frames,
            frame_labels=frame_labels,
            fps=fps,
            title=title,
            image_format="png",  # parallel_render_frames always uses PNG
            overlay_data=overlay_json,
        )

        # Write HTML
        output_path.write_text(html, encoding="utf-8")

        print(f"✓ HTML saved to {output_path} (frames in {frames_dir})")
        return output_path

    # ---- Embedded mode (original behavior) ----
    # Pre-render all frames to base64 with overlays baked in
    print(f"Rendering {n_frames} frames to {image_format.upper()}...")
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
        overlay_data=None,  # Overlays are now rendered in matplotlib, not JavaScript
    )

    # Write to file with UTF-8 encoding (for Unicode button symbols on Windows)
    output_path.write_text(html, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"✓ HTML saved to {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def _generate_html_player(
    frames_b64: list[str],
    frame_labels: list[str],
    fps: int,
    title: str,
    image_format: str,
    overlay_data: dict | None = None,
) -> str:
    """Generate HTML with embedded frames and JavaScript controls.

    Parameters
    ----------
    frames_b64 : list of str
        Base64-encoded frame images
    frame_labels : list of str
        Frame labels for display
    fps : int
        Frames per second for playback
    title : str
        Animation title
    image_format : str
        Image format (png or jpeg)
    overlay_data : dict | None, optional
        JSON-compatible dictionary with overlay data (positions and regions)

    Returns
    -------
    html : str
        Complete HTML document with embedded player

    Notes
    -----
    The generated HTML includes:
    - Responsive CSS layout
    - Play/pause/prev/next buttons
    - Range slider for frame scrubbing
    - Speed control dropdown
    - Frame counter display
    - Keyboard shortcuts (space, arrows)
    - ARIA labels for accessibility

    Security:
    - Title is HTML-escaped to prevent injection attacks
    - Frame labels are JSON-encoded and safely inserted via textContent
    """
    n_frames = len(frames_b64)

    # Sanitize title to prevent HTML/JS injection
    safe_title = html_module.escape(title)

    # JSON-encode data for JavaScript
    frames_json = json.dumps(frames_b64)
    labels_json = json.dumps(frame_labels)
    overlay_json_str = json.dumps(overlay_data) if overlay_data else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 900px;
        }}
        h1 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #333;
        }}
        .frame-container {{
            position: relative;
            width: 100%;
        }}
        #frame {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
        }}
        #overlay-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        #controls {{
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #007bff;
            color: white;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #0056b3;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        #slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }}
        #slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info {{
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 14px;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .speed-control label {{
            font-size: 14px;
            color: #666;
        }}
        .speed-control select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>

        <div class="frame-container">
            <img id="frame" src="" alt="Animation frame" />
            <canvas id="overlay-canvas"></canvas>
        </div>

        <div id="controls">
            <div class="button-row">
                <button id="play" aria-label="Play animation">▶ Play</button>
                <button id="pause" aria-label="Pause animation">⏸ Pause</button>
                <button id="prev" aria-label="Previous frame">⏮ Prev</button>
                <button id="next" aria-label="Next frame">⏭ Next</button>
                <div class="speed-control">
                    <label for="speed">Speed:</label>
                    <select id="speed" aria-label="Playback speed">
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>

            <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" aria-label="Frame slider" />

            <div class="info">
                <span id="label" aria-live="polite"></span>
                <span id="frame-counter" aria-live="polite"></span>
            </div>
        </div>
    </div>

    <script>
        // Frame data (embedded)
        const frames = {frames_json};
        const labels = {labels_json};
        const baseFPS = {fps};
        const overlayData = {overlay_json_str};

        // State
        let currentFrame = 0;
        let playing = false;
        let animationId = null;
        let lastTime = 0;
        let accumulator = 0;
        let speedMultiplier = 1.0;

        // Elements
        const img = document.getElementById('frame');
        const slider = document.getElementById('slider');
        const labelSpan = document.getElementById('label');
        const counterSpan = document.getElementById('frame-counter');
        const playBtn = document.getElementById('play');
        const pauseBtn = document.getElementById('pause');
        const prevBtn = document.getElementById('prev');
        const nextBtn = document.getElementById('next');
        const speedSelect = document.getElementById('speed');
        const canvas = document.getElementById('overlay-canvas');
        const ctx = canvas.getContext('2d');

        // Overlay rendering function
        function renderOverlays(frameIdx) {{
            if (!overlayData) return;

            // Match canvas size to image with HiDPI support (devicePixelRatio)
            const imgEl = document.getElementById('frame');
            const imgWidth = imgEl.naturalWidth || imgEl.width;
            const imgHeight = imgEl.naturalHeight || imgEl.height;
            const dpr = window.devicePixelRatio || 1;

            // Scale canvas by devicePixelRatio for sharp rendering on HiDPI displays
            const targetWidth = imgWidth * dpr;
            const targetHeight = imgHeight * dpr;
            if (canvas.width !== targetWidth || canvas.height !== targetHeight) {{
                canvas.width = targetWidth;
                canvas.height = targetHeight;
                canvas.style.width = imgWidth + 'px';
                canvas.style.height = imgHeight + 'px';
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            }}
            // Clear canvas (separate from resize)
            ctx.clearRect(0, 0, imgWidth, imgHeight);

            // Get dimension ranges for coordinate scaling
            const dimRanges = overlayData.dimension_ranges || [[0, 1], [0, 1]];
            const xRange = dimRanges[0];
            const yRange = dimRanges[1];

            // Scaling functions: data coords -> canvas pixels (CSS pixels, not device pixels)
            function scaleX(x) {{
                const extent = xRange[1] - xRange[0];
                if (extent === 0) return imgWidth / 2;  // Center point for zero extent
                return ((x - xRange[0]) / extent) * imgWidth;
            }}
            function scaleY(y) {{
                // Flip y-axis (image has y=0 at top)
                const extent = yRange[1] - yRange[0];
                if (extent === 0) return imgHeight / 2;  // Center point for zero extent
                return ((yRange[1] - y) / extent) * imgHeight;
            }}

            // Render positions
            if (overlayData.positions && overlayData.positions.length > 0 && frameIdx < overlayData.positions[0].data.length) {{
                overlayData.positions.forEach(posOverlay => {{
                    const positions = posOverlay.data;
                    const trailLength = posOverlay.trail_length || 0;

                    // Draw trail - stroke each segment separately for proper alpha fade
                    if (trailLength > 0) {{
                        ctx.lineWidth = 2;
                        const startIdx = Math.max(0, frameIdx - trailLength);

                        for (let i = startIdx + 1; i <= frameIdx; i++) {{
                            const prevPos = positions[i - 1];
                            const pos = positions[i];

                            // Calculate alpha for this segment (older = more faded)
                            const age = frameIdx - i;
                            const alpha = 0.3 + 0.7 * (1 - age / trailLength);

                            ctx.globalAlpha = alpha;
                            ctx.strokeStyle = posOverlay.color;
                            ctx.beginPath();
                            ctx.moveTo(scaleX(prevPos[0]), scaleY(prevPos[1]));
                            ctx.lineTo(scaleX(pos[0]), scaleY(pos[1]));
                            ctx.stroke();
                        }}
                        ctx.globalAlpha = 1.0;
                    }}

                    // Draw current position marker
                    const currentPos = positions[frameIdx];
                    const x = scaleX(currentPos[0]);
                    const y = scaleY(currentPos[1]);

                    ctx.fillStyle = posOverlay.color;
                    ctx.beginPath();
                    ctx.arc(x, y, posOverlay.size, 0, 2 * Math.PI);
                    ctx.fill();
                }});
            }}

            // Render regions
            if (overlayData.regions) {{
                overlayData.regions.forEach(region => {{
                    ctx.strokeStyle = 'cyan';
                    ctx.lineWidth = 2;
                    ctx.globalAlpha = overlayData.region_alpha || 0.3;

                    if (region.kind === 'point') {{
                        const x = scaleX(region.coordinates[0]);
                        const y = scaleY(region.coordinates[1]);
                        ctx.beginPath();
                        ctx.arc(x, y, 10, 0, 2 * Math.PI);
                        ctx.stroke();
                    }} else if (region.kind === 'polygon') {{
                        ctx.beginPath();
                        region.coordinates.forEach((coord, i) => {{
                            const x = scaleX(coord[0]);
                            const y = scaleY(coord[1]);
                            if (i === 0) {{
                                ctx.moveTo(x, y);
                            }} else {{
                                ctx.lineTo(x, y);
                            }}
                        }});
                        ctx.closePath();
                        ctx.stroke();
                    }}

                    ctx.globalAlpha = 1.0;
                }});
            }}

            // Render events (instant mode - visible only on their exact frame)
            if (overlayData.events && overlayData.events.length > 0) {{
                overlayData.events.forEach(eventOverlay => {{
                    const eventPositions = eventOverlay.event_positions;
                    const frameIndices = eventOverlay.frame_indices;
                    const colors = eventOverlay.colors;
                    const size = eventOverlay.size || 8;
                    const borderColor = eventOverlay.border_color || 'white';
                    const borderWidth = eventOverlay.border_width || 0.5;

                    // Iterate over each event type
                    Object.keys(eventPositions).forEach(eventType => {{
                        const positions = eventPositions[eventType];
                        const indices = frameIndices[eventType];
                        const color = colors[eventType] || '#ff0000';

                        // Find events that occur on this frame
                        for (let i = 0; i < indices.length; i++) {{
                            if (indices[i] === frameIdx) {{
                                const pos = positions[i];
                                if (!pos || pos.length < 2) continue;

                                // Skip NaN positions
                                if (isNaN(pos[0]) || isNaN(pos[1])) continue;

                                const x = scaleX(pos[0]);
                                const y = scaleY(pos[1]);

                                // Draw filled circle with border
                                ctx.fillStyle = color;
                                ctx.strokeStyle = borderColor;
                                ctx.lineWidth = borderWidth;
                                ctx.beginPath();
                                ctx.arc(x, y, size, 0, 2 * Math.PI);
                                ctx.fill();
                                ctx.stroke();
                            }}
                        }}
                    }});
                }});
            }}
        }}

        // Initialize
        function init() {{
            updateFrame(0);
            updateControls();
        }}

        function updateFrame(idx) {{
            if (idx < 0 || idx >= frames.length) return;

            currentFrame = idx;

            // Update image (instant - just changes src)
            // Use dynamic MIME type based on format
            img.src = 'data:image/{image_format};base64,' + frames[idx];

            // Update UI
            slider.value = idx;
            labelSpan.textContent = labels[idx];
            counterSpan.textContent = `${{idx + 1}} / ${{frames.length}}`;

            // Render overlays
            renderOverlays(idx);
        }}

        function updateControls() {{
            playBtn.disabled = playing;
            pauseBtn.disabled = !playing;
        }}

        // Animation loop using requestAnimationFrame
        function animate(timestamp) {{
            if (!playing) return;

            const delta = timestamp - lastTime;
            lastTime = timestamp;
            accumulator += delta;

            const frameDelay = (1000 / baseFPS) / speedMultiplier;

            while (accumulator >= frameDelay) {{
                currentFrame = (currentFrame + 1) % frames.length;
                updateFrame(currentFrame);
                accumulator -= frameDelay;
            }}

            animationId = requestAnimationFrame(animate);
        }}

        function play() {{
            if (playing) return;
            playing = true;
            updateControls();
            lastTime = performance.now();
            accumulator = 0;
            animationId = requestAnimationFrame(animate);
        }}

        function pause() {{
            if (!playing) return;
            playing = false;
            updateControls();
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
        }}

        function stepForward() {{
            pause();
            updateFrame(Math.min(frames.length - 1, currentFrame + 1));
        }}

        function stepBackward() {{
            pause();
            updateFrame(Math.max(0, currentFrame - 1));
        }}

        // Event listeners
        slider.oninput = (e) => {{
            pause();
            updateFrame(parseInt(e.target.value));
        }};

        playBtn.onclick = play;
        pauseBtn.onclick = pause;
        nextBtn.onclick = stepForward;
        prevBtn.onclick = stepBackward;

        speedSelect.onchange = (e) => {{
            speedMultiplier = parseFloat(e.target.value);
            // No need to restart - accumulator handles speed changes gracefully
        }};

        // Keyboard shortcuts
        document.onkeydown = (e) => {{
            if (e.key === ' ') {{
                e.preventDefault();
                playing ? pause() : play();
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                stepForward();
            }} else if (e.key === 'Home') {{
                e.preventDefault();
                pause();
                updateFrame(0);
            }} else if (e.key === 'End') {{
                e.preventDefault();
                pause();
                updateFrame(frames.length - 1);
            }} else if (e.key === ',') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === '.') {{
                e.preventDefault();
                stepForward();
            }}
        }};

        // Start
        init();
    </script>
</body>
</html>"""

    return html


def _generate_non_embedded_html_player(
    frames_dir: Path,
    n_frames: int,
    frame_labels: list[str],
    fps: int,
    title: str,
    image_format: str,
    overlay_data: dict | None = None,
) -> str:
    """Generate HTML player with disk-backed frames (non-embedded).

    Parameters
    ----------
    frames_dir : Path
        Directory containing frame images
    n_frames : int
        Number of frames
    frame_labels : list of str
        Frame labels for display
    fps : int
        Frames per second for playback
    title : str
        Animation title
    image_format : str
        Image format (png or jpeg)
    overlay_data : dict | None, optional
        JSON-compatible dictionary with overlay data (positions and regions)

    Returns
    -------
    html : str
        Complete HTML document referencing external frames

    Notes
    -----
    This generates a lightweight HTML file that references frames via relative
    paths instead of embedding them as base64. Suitable for large sequences
    (500+ frames) where embedded mode would create huge HTML files.

    The frames must exist in frames_dir with zero-padded names:
    frame_00000.png, frame_00001.png, etc.
    """
    # Sanitize title
    safe_title = html_module.escape(title)

    # JSON-encode labels and overlay data
    labels_json = json.dumps(frame_labels)
    overlay_json_str = json.dumps(overlay_data) if overlay_data else "null"

    # Relative directory name for JavaScript
    frames_dir_name = frames_dir.name

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 900px;
        }}
        h1 {{
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #333;
        }}
        .frame-container {{
            position: relative;
            width: 100%;
        }}
        #frame {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
        }}
        #overlay-canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        #controls {{
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #007bff;
            color: white;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #0056b3;
        }}
        button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        #slider {{
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: #ddd;
            border-radius: 3px;
            outline: none;
        }}
        #slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
        }}
        #slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #007bff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info {{
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 14px;
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .speed-control label {{
            font-size: 14px;
            color: #666;
        }}
        .speed-control select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{safe_title}</h1>

        <div class="frame-container">
            <img id="frame" src="" alt="Animation frame" />
            <canvas id="overlay-canvas"></canvas>
        </div>

        <div id="controls">
            <div class="button-row">
                <button id="play" aria-label="Play animation">▶ Play</button>
                <button id="pause" aria-label="Pause animation">⏸ Pause</button>
                <button id="prev" aria-label="Previous frame">⏮ Prev</button>
                <button id="next" aria-label="Next frame">⏭ Next</button>
                <div class="speed-control">
                    <label for="speed">Speed:</label>
                    <select id="speed" aria-label="Playback speed">
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>

            <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0" aria-label="Frame slider" />

            <div class="info">
                <span id="label" aria-live="polite"></span>
                <span id="frame-counter" aria-live="polite"></span>
            </div>
        </div>
    </div>

    <script>
        // Frame data (disk-backed)
        const N = {n_frames};
        const labels = {labels_json};
        const baseFPS = {fps};
        const framesDir = "{frames_dir_name}";
        const overlayData = {overlay_json_str};

        // Zero-pad frame numbers (5 digits)
        function zpad(i, digits=5) {{ return (""+i).padStart(digits,"0"); }}

        // Build frame URLs dynamically
        const frameSrc = Array.from({{length: N}}, (_,i) => `${{framesDir}}/frame_${{zpad(i)}}.{image_format}`);

        // State
        let currentFrame = 0;
        let playing = false;
        let animationId = null;
        let lastTime = 0;
        let accumulator = 0;
        let speedMultiplier = 1.0;

        // Elements
        const img = document.getElementById('frame');
        const slider = document.getElementById('slider');
        const labelSpan = document.getElementById('label');
        const counterSpan = document.getElementById('frame-counter');
        const playBtn = document.getElementById('play');
        const pauseBtn = document.getElementById('pause');
        const prevBtn = document.getElementById('prev');
        const nextBtn = document.getElementById('next');
        const speedSelect = document.getElementById('speed');
        const canvas = document.getElementById('overlay-canvas');
        const ctx = canvas.getContext('2d');

        // Overlay rendering function
        function renderOverlays(frameIdx) {{
            if (!overlayData) return;

            // Match canvas size to image with HiDPI support (devicePixelRatio)
            const imgEl = document.getElementById('frame');
            const imgWidth = imgEl.naturalWidth || imgEl.width;
            const imgHeight = imgEl.naturalHeight || imgEl.height;
            const dpr = window.devicePixelRatio || 1;

            // Scale canvas by devicePixelRatio for sharp rendering on HiDPI displays
            const targetWidth = imgWidth * dpr;
            const targetHeight = imgHeight * dpr;
            if (canvas.width !== targetWidth || canvas.height !== targetHeight) {{
                canvas.width = targetWidth;
                canvas.height = targetHeight;
                canvas.style.width = imgWidth + 'px';
                canvas.style.height = imgHeight + 'px';
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            }}
            // Clear canvas (separate from resize)
            ctx.clearRect(0, 0, imgWidth, imgHeight);

            // Get dimension ranges for coordinate scaling
            const dimRanges = overlayData.dimension_ranges || [[0, 1], [0, 1]];
            const xRange = dimRanges[0];
            const yRange = dimRanges[1];

            // Scaling functions: data coords -> canvas pixels (CSS pixels, not device pixels)
            function scaleX(x) {{
                const extent = xRange[1] - xRange[0];
                if (extent === 0) return imgWidth / 2;  // Center point for zero extent
                return ((x - xRange[0]) / extent) * imgWidth;
            }}
            function scaleY(y) {{
                // Flip y-axis (image has y=0 at top)
                const extent = yRange[1] - yRange[0];
                if (extent === 0) return imgHeight / 2;  // Center point for zero extent
                return ((yRange[1] - y) / extent) * imgHeight;
            }}

            // Render positions
            if (overlayData.positions && overlayData.positions.length > 0 && frameIdx < overlayData.positions[0].data.length) {{
                overlayData.positions.forEach(posOverlay => {{
                    const positions = posOverlay.data;
                    const trailLength = posOverlay.trail_length || 0;

                    // Draw trail - stroke each segment separately for proper alpha fade
                    if (trailLength > 0) {{
                        ctx.lineWidth = 2;
                        const startIdx = Math.max(0, frameIdx - trailLength);

                        for (let i = startIdx + 1; i <= frameIdx; i++) {{
                            const prevPos = positions[i - 1];
                            const pos = positions[i];

                            // Calculate alpha for this segment (older = more faded)
                            const age = frameIdx - i;
                            const alpha = 0.3 + 0.7 * (1 - age / trailLength);

                            ctx.globalAlpha = alpha;
                            ctx.strokeStyle = posOverlay.color;
                            ctx.beginPath();
                            ctx.moveTo(scaleX(prevPos[0]), scaleY(prevPos[1]));
                            ctx.lineTo(scaleX(pos[0]), scaleY(pos[1]));
                            ctx.stroke();
                        }}
                        ctx.globalAlpha = 1.0;
                    }}

                    // Draw current position marker
                    const currentPos = positions[frameIdx];
                    const x = scaleX(currentPos[0]);
                    const y = scaleY(currentPos[1]);

                    ctx.fillStyle = posOverlay.color;
                    ctx.beginPath();
                    ctx.arc(x, y, posOverlay.size, 0, 2 * Math.PI);
                    ctx.fill();
                }});
            }}

            // Render regions
            if (overlayData.regions) {{
                overlayData.regions.forEach(region => {{
                    ctx.strokeStyle = 'cyan';
                    ctx.lineWidth = 2;
                    ctx.globalAlpha = overlayData.region_alpha || 0.3;

                    if (region.kind === 'point') {{
                        const x = scaleX(region.coordinates[0]);
                        const y = scaleY(region.coordinates[1]);
                        ctx.beginPath();
                        ctx.arc(x, y, 10, 0, 2 * Math.PI);
                        ctx.stroke();
                    }} else if (region.kind === 'polygon') {{
                        ctx.beginPath();
                        region.coordinates.forEach((coord, i) => {{
                            const x = scaleX(coord[0]);
                            const y = scaleY(coord[1]);
                            if (i === 0) {{
                                ctx.moveTo(x, y);
                            }} else {{
                                ctx.lineTo(x, y);
                            }}
                        }});
                        ctx.closePath();
                        ctx.stroke();
                    }}

                    ctx.globalAlpha = 1.0;
                }});
            }}

            // Render events (instant mode - visible only on their exact frame)
            if (overlayData.events && overlayData.events.length > 0) {{
                overlayData.events.forEach(eventOverlay => {{
                    const eventPositions = eventOverlay.event_positions;
                    const frameIndices = eventOverlay.frame_indices;
                    const colors = eventOverlay.colors;
                    const size = eventOverlay.size || 8;
                    const borderColor = eventOverlay.border_color || 'white';
                    const borderWidth = eventOverlay.border_width || 0.5;

                    // Iterate over each event type
                    Object.keys(eventPositions).forEach(eventType => {{
                        const positions = eventPositions[eventType];
                        const indices = frameIndices[eventType];
                        const color = colors[eventType] || '#ff0000';

                        // Find events that occur on this frame
                        for (let i = 0; i < indices.length; i++) {{
                            if (indices[i] === frameIdx) {{
                                const pos = positions[i];
                                if (!pos || pos.length < 2) continue;

                                // Skip NaN positions
                                if (isNaN(pos[0]) || isNaN(pos[1])) continue;

                                const x = scaleX(pos[0]);
                                const y = scaleY(pos[1]);

                                // Draw filled circle with border
                                ctx.fillStyle = color;
                                ctx.strokeStyle = borderColor;
                                ctx.lineWidth = borderWidth;
                                ctx.beginPath();
                                ctx.arc(x, y, size, 0, 2 * Math.PI);
                                ctx.fill();
                                ctx.stroke();
                            }}
                        }}
                    }});
                }});
            }}
        }}

        // Initialize
        function init() {{
            updateFrame(0);
            updateControls();
        }}

        function updateFrame(idx) {{
            if (idx < 0 || idx >= N) return;

            currentFrame = idx;

            // Update image and render overlays when loaded
            img.onload = () => renderOverlays(idx);
            img.src = frameSrc[idx];

            // Prefetch neighboring frames for smoother scrubbing
            prefetchFrames(idx);

            // Update UI
            slider.value = idx;
            labelSpan.textContent = labels[idx];
            counterSpan.textContent = `${{idx + 1}} / ${{N}}`;
        }}

        function updateControls() {{
            playBtn.disabled = playing;
            pauseBtn.disabled = !playing;
        }}

        // Animation loop using requestAnimationFrame
        function animate(timestamp) {{
            if (!playing) return;

            const delta = timestamp - lastTime;
            lastTime = timestamp;
            accumulator += delta;

            const frameDelay = (1000 / baseFPS) / speedMultiplier;

            while (accumulator >= frameDelay) {{
                currentFrame = (currentFrame + 1) % N;
                updateFrame(currentFrame);
                accumulator -= frameDelay;
            }}

            animationId = requestAnimationFrame(animate);
        }}

        function play() {{
            if (playing) return;
            playing = true;
            updateControls();
            lastTime = performance.now();
            accumulator = 0;
            animationId = requestAnimationFrame(animate);
        }}

        function pause() {{
            if (!playing) return;
            playing = false;
            updateControls();
            if (animationId) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
        }}

        function stepForward() {{
            pause();
            updateFrame(Math.min(N - 1, currentFrame + 1));
        }}

        function stepBackward() {{
            pause();
            updateFrame(Math.max(0, currentFrame - 1));
        }}

        // Event listeners
        slider.oninput = (e) => {{
            pause();
            updateFrame(parseInt(e.target.value));
        }};

        playBtn.onclick = play;
        pauseBtn.onclick = pause;
        nextBtn.onclick = stepForward;
        prevBtn.onclick = stepBackward;

        speedSelect.onchange = (e) => {{
            speedMultiplier = parseFloat(e.target.value);
            // No need to restart - accumulator handles speed changes gracefully
        }};

        // Keyboard shortcuts
        document.onkeydown = (e) => {{
            if (e.key === ' ') {{
                e.preventDefault();
                playing ? pause() : play();
            }} else if (e.key === 'ArrowLeft') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === 'ArrowRight') {{
                e.preventDefault();
                stepForward();
            }} else if (e.key === 'Home') {{
                e.preventDefault();
                pause();
                updateFrame(0);
            }} else if (e.key === 'End') {{
                e.preventDefault();
                pause();
                updateFrame(N - 1);
            }} else if (e.key === ',') {{
                e.preventDefault();
                stepBackward();
            }} else if (e.key === '.') {{
                e.preventDefault();
                stepForward();
            }}
        }};

        // Prefetch window for non-embedded mode
        const prefetchSize = 5;  // Preload +/-5 frames
        const imageCache = new Map();

        function prefetchFrames(centerIdx) {{
            for (let offset = -prefetchSize; offset <= prefetchSize; offset++) {{
                const idx = centerIdx + offset;
                if (idx >= 0 && idx < N && !imageCache.has(idx)) {{
                    const img = new Image();
                    img.src = frameSrc[idx];
                    imageCache.set(idx, img);
                }}
            }}
            // Limit cache size
            if (imageCache.size > prefetchSize * 4) {{
                // Remove entries far from current frame
                for (const [key] of imageCache) {{
                    if (Math.abs(key - centerIdx) > prefetchSize * 2) {{
                        imageCache.delete(key);
                    }}
                }}
            }}
        }}

        // Start
        init();
    </script>
</body>
</html>"""

    return html
