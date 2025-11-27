"""Main annotation entry point."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from neurospatial.annotation._types import (
    AnnotationConfig,
    MultipleBoundaryStrategy,
    RegionType,
)

# Module logger for debug output
# Enable with: logging.getLogger("neurospatial.annotation.core").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import napari
    from numpy.typing import NDArray
    from shapely.geometry import Polygon

    from neurospatial import Environment
    from neurospatial.annotation._boundary_inference import BoundaryConfig
    from neurospatial.regions import Regions
    from neurospatial.transforms import VideoCalibration


class AnnotationResult(NamedTuple):
    """Result from an annotation session.

    Attributes
    ----------
    environment : Environment or None
        Discretized environment if boundary was annotated, else None.
    regions : Regions
        All annotated regions (excluding environment boundary).

    """

    environment: Environment | None
    regions: Regions


def annotate_video(
    video_path: str | Path,
    *,
    config: AnnotationConfig | None = None,
    initial_regions: Regions | None = None,
    calibration: VideoCalibration | None = None,
    mode: Literal["environment", "regions", "both"] = "both",
    bin_size: float | None = None,
    initial_boundary: Polygon | NDArray[np.float64] | None = None,
    boundary_config: BoundaryConfig | None = None,
    # Individual params (can be overridden by config)
    frame_index: int | None = None,
    simplify_tolerance: float | None = None,
    multiple_boundaries: MultipleBoundaryStrategy | None = None,
    show_positions: bool | None = None,
) -> AnnotationResult:
    """Launch interactive napari annotation on a video frame.

    Opens a napari viewer with the specified video frame. Users can draw
    polygons to define an environment boundary and/or named regions.
    After closing the viewer, annotations are converted to Regions and
    optionally an Environment.

    Parameters
    ----------
    video_path : str or Path
        Path to video file (any format supported by OpenCV).
    config : AnnotationConfig, optional
        Configuration for annotation UI settings. Groups frame_index,
        simplify_tolerance, multiple_boundaries, and show_positions.
        Individual parameters override config values if both provided.
    initial_regions : Regions, optional
        Pre-existing regions to display for editing.
    calibration : VideoCalibration, optional
        Pixel-to-cm transform. If provided, output coordinates are in cm.
        If None, coordinates remain in pixels.
    mode : {"environment", "regions", "both"}, default="both"
        What to annotate:
        - "environment": Only expect environment boundary
        - "regions": Only expect named regions
        - "both": Expect both boundary and regions
    bin_size : float, optional
        Bin size for environment discretization. Required if mode is
        "environment" or "both".
    initial_boundary : Polygon or NDArray, optional
        Pre-drawn boundary for editing. Can be:

        - Shapely Polygon: Used directly as boundary
        - NDArray (n, 2): Position data to infer boundary from

        If None, user draws boundary manually.
    boundary_config : BoundaryConfig, optional
        Configuration for boundary inference when initial_boundary is an array.
        If None, uses BoundaryConfig defaults (convex_hull, 2% buffer, 1% simplify).
    frame_index : int, optional
        Which frame to display for annotation. Overrides config.frame_index.
        Default is 0 (first frame).
    simplify_tolerance : float, optional
        Tolerance for polygon simplification using Douglas-Peucker algorithm.
        Removes vertices that deviate less than this distance from the simplified line.
        Overrides config.simplify_tolerance.

        Units depend on calibration:
        - With calibration: environment units (typically cm)
        - Without calibration: pixels

        Recommended values:
        - For cm: 1.0-2.0 (removes hand-drawn jitter)
        - For pixels: 2.0-5.0
    multiple_boundaries : {"last", "first", "error"}, optional
        How to handle multiple environment boundaries.
        Overrides config.multiple_boundaries. Default is "last".

        - "last": Use the last drawn boundary (default). A warning is emitted.
        - "first": Use the first drawn boundary. A warning is emitted.
        - "error": Raise ValueError if multiple boundaries are drawn.
    show_positions : bool, optional
        If True and initial_boundary is an array, show positions as a
        Points layer for reference while editing.
        Overrides config.show_positions. Default is False.

    Returns
    -------
    AnnotationResult
        Named tuple containing:
        - environment: Environment or None
        - regions: Regions collection

    Raises
    ------
    ValueError
        If bin_size is not provided when mode requires environment creation,
        or if ``multiple_boundaries="error"`` and multiple environment
        boundaries are drawn.
    ImportError
        If napari is not installed.

    Examples
    --------
    >>> from neurospatial.annotation import annotate_video
    >>> # Simple annotation (pixel coordinates)
    >>> result = annotate_video("experiment.mp4", bin_size=10.0)
    >>> print(result.environment)  # Environment from boundary
    >>> print(result.regions)  # Named regions

    >>> # With calibration (cm coordinates)
    >>> from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
    >>> transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
    >>> calib = VideoCalibration(transform, (640, 480))
    >>> result = annotate_video("experiment.mp4", calibration=calib, bin_size=2.0)

    Notes
    -----
    This function blocks until the napari viewer is closed. The viewer runs
    in the same Python process, and the function returns only after the user
    closes it (via the "Save and Close" button, Escape key, or window close).

    If multiple environment boundaries are drawn, only the last one is used
    and a warning is emitted.

    Environments with Holes
    ^^^^^^^^^^^^^^^^^^^^^^^
    Users can draw "hole" polygons inside the environment boundary to create
    excluded areas. Press M to cycle to hole mode (red) after drawing the
    boundary. Holes are subtracted from the boundary using Shapely's
    difference operation before creating the Environment.

    Coordinate Systems
    ^^^^^^^^^^^^^^^^^^
    - Napari shapes: (row, col) with origin at top-left
    - Video pixels: (x, y) with origin at top-left
    - Environment: (x, y) with origin at bottom-left (if calibrated)

    See Also
    --------
    regions_from_labelme : Import from LabelMe JSON
    regions_from_cvat : Import from CVAT XML
    AnnotationConfig : Configuration dataclass for annotation settings.

    """
    # Resolve config with individual parameter overrides
    resolved = _resolve_config_params(
        config,
        frame_index,
        simplify_tolerance,
        multiple_boundaries,
        show_positions,
    )
    logger.debug(
        "Resolved config: frame_index=%d, simplify_tolerance=%s, "
        "multiple_boundaries=%s, show_positions=%s",
        resolved.frame_index,
        resolved.simplify_tolerance,
        resolved.multiple_boundaries,
        resolved.show_positions,
    )

    # Validate parameters early (before expensive imports)
    _validate_annotate_params(mode, bin_size)

    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari is required for interactive annotation. "
            "Install with: pip install napari[all]",
        ) from e

    # Convert to Path and load video frame
    video_path = Path(video_path)
    logger.debug("Loading video frame %d from %s", resolved.frame_index, video_path)
    frame = _load_video_frame(video_path, resolved.frame_index)
    logger.debug("Loaded frame with shape %s", frame.shape)

    # Process initial boundary (from polygon or positions)
    boundary_polygon, positions_for_display = _process_initial_boundary(
        initial_boundary,
        boundary_config,
        resolved.show_positions,
    )

    # Handle conflict: initial_boundary takes precedence over env regions in initial_regions
    if boundary_polygon is not None and initial_regions is not None:
        initial_regions = _filter_environment_regions(initial_regions)

    # Setup napari viewer with all layers and widgets
    _viewer, shapes = _setup_annotation_viewer(
        video_path,
        frame,
        mode,
        boundary_polygon,
        positions_for_display,
        initial_regions,
        calibration,
    )

    # Run napari (blocking until viewer closes)
    napari.run()

    # Convert annotations to result
    return _process_annotation_results(
        shapes,
        mode,
        bin_size,
        calibration,
        resolved.simplify_tolerance,
        resolved.multiple_boundaries,
    )


def _add_initial_regions(
    shapes_layer: napari.layers.Shapes,
    regions: Regions,
    calibration: VideoCalibration | None,
) -> None:
    """Add existing regions to shapes layer for editing.

    Transforms region polygon coordinates back to napari pixel space and adds
    them to the shapes layer with appropriate features (name, role) for editing.

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        The shapes layer to add regions to. Must be an initialized napari
        Shapes layer.
    regions : Regions
        Existing regions to display. Only polygon regions are added; point
        regions are skipped.
    calibration : VideoCalibration or None
        If provided, transforms cm coordinates back to pixels for display.
        The inverse of the calibration transform is used.

    Notes
    -----
    This function modifies the shapes_layer in place. Non-polygon regions
    (points) are silently skipped as they cannot be represented as shapes.

    """
    from shapely import get_coordinates

    from neurospatial.annotation._helpers import (
        rebuild_features,
        sync_face_colors_from_features,
    )

    data = []
    names = []
    roles = []

    for name, region in regions.items():
        if region.kind != "polygon":
            continue

        # Get polygon vertices in world coordinates
        coords = get_coordinates(region.data)

        # Transform to pixels if calibration provided
        if calibration is not None:
            coords = calibration.transform_cm_to_px(coords)

        # Convert to napari (row, col) order
        coords_rc = coords[:, ::-1]
        data.append(coords_rc)
        names.append(name)
        roles.append(region.metadata.get("role", "region"))

    if data:
        # Add shapes to layer
        shapes_layer.add(data, shape_type="polygon")

        # Update features DataFrame using centralized builder
        shapes_layer.features = rebuild_features(roles, names)

        # Sync face colors from features
        sync_face_colors_from_features(shapes_layer)

        # Note: Don't reset feature_defaults here - let the widget control the mode.
        # The widget's initial_mode determines what role/name new shapes will have.


def _add_positions_layer(
    viewer: napari.Viewer,
    positions: NDArray[np.float64],
    calibration: VideoCalibration | None,
) -> None:
    """Add positions as semi-transparent Points layer for reference.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    positions : NDArray[np.float64]
        Position data. NaN/Inf values are automatically filtered.
        Coordinate system depends on calibration:

        - With calibration: environment units (cm), Y-up origin
        - Without calibration: video pixels (x, y), Y-down origin
    calibration : VideoCalibration, optional
        Transform from environment coords (cm) to video pixels.

    Notes
    -----
    Mirrors the pattern in add_initial_boundary_to_shapes for consistency.

    """
    # Filter out NaN/Inf values (common in tracking data for lost frames)
    valid_mask = np.all(np.isfinite(positions), axis=1)
    coords = positions[valid_mask].copy()

    if len(coords) == 0:
        return  # No valid positions to display

    # Transform to pixels if calibration provided
    # NOTE: transform_cm_to_px handles Y-flip internally - don't double-flip!
    if calibration is not None:
        coords = calibration.transform_cm_to_px(coords)

    # Convert to napari (row, col) order
    coords_rc = coords[:, ::-1]

    # Subsample if too many points (for performance)
    if len(coords_rc) > 5000:
        step = len(coords_rc) // 5000
        coords_rc = coords_rc[::step]

    viewer.add_points(
        coords_rc,
        name="Trajectory (reference)",
        size=8,
        face_color="cyan",
        border_color="darkblue",
        border_width=0.5,
        opacity=0.5,
        blending="translucent",
    )


def _filter_environment_regions(regions: Regions) -> Regions:
    """Filter out environment regions and warn if any were found.

    This function is called when both initial_boundary and initial_regions
    are provided. The initial_boundary takes precedence, so environment
    regions in initial_regions are removed.

    Parameters
    ----------
    regions : Regions
        The regions collection to filter.

    Returns
    -------
    Regions
        New Regions collection without environment regions.

    """
    from neurospatial.regions import Regions

    env_regions = [
        name for name, r in regions.items() if r.metadata.get("role") == "environment"
    ]

    if env_regions:
        warnings.warn(
            f"Both initial_boundary and environment regions in initial_regions "
            f"({env_regions}) provided. Using initial_boundary; ignoring "
            f"environment regions from initial_regions.",
            UserWarning,
            stacklevel=3,
        )
        # Filter out environment regions
        return Regions(
            r for r in regions.values() if r.metadata.get("role") != "environment"
        )

    return regions


# ---------------------------------------------------------------------------
# Helper functions for annotate_video() decomposition
# ---------------------------------------------------------------------------


def _resolve_config_params(
    config: AnnotationConfig | None,
    frame_index: int | None,
    simplify_tolerance: float | None,
    multiple_boundaries: MultipleBoundaryStrategy | None,
    show_positions: bool | None,
) -> AnnotationConfig:
    """Merge config with individual parameter overrides.

    Individual parameters take precedence over config values. If no config
    is provided, uses AnnotationConfig defaults.

    Parameters
    ----------
    config : AnnotationConfig or None
        Base configuration object.
    frame_index, simplify_tolerance, multiple_boundaries, show_positions
        Individual parameter overrides (None means use config value).

    Returns
    -------
    AnnotationConfig
        Resolved configuration with all values set.

    """
    # Start with defaults or provided config
    base = config if config is not None else AnnotationConfig()

    # Override with individual params if provided
    return AnnotationConfig(
        frame_index=frame_index if frame_index is not None else base.frame_index,
        simplify_tolerance=(
            simplify_tolerance
            if simplify_tolerance is not None
            else base.simplify_tolerance
        ),
        multiple_boundaries=(
            multiple_boundaries
            if multiple_boundaries is not None
            else base.multiple_boundaries
        ),
        show_positions=(
            show_positions if show_positions is not None else base.show_positions
        ),
    )


def _validate_annotate_params(
    mode: Literal["environment", "regions", "both"],
    bin_size: float | None,
) -> None:
    """Validate annotation parameters before expensive imports.

    Parameters
    ----------
    mode : {"environment", "regions", "both"}
        Annotation mode.
    bin_size : float or None
        Bin size for environment discretization.

    Raises
    ------
    ValueError
        If bin_size is required but not provided.

    """
    if mode in ("environment", "both") and bin_size is None:
        raise ValueError(
            f"bin_size is required when mode={mode!r}. "
            "Provide bin_size for environment discretization.",
        )


def _load_video_frame(video_path: Path, frame_index: int) -> NDArray[np.uint8]:
    """Load a single frame from a video file.

    Parameters
    ----------
    video_path : Path
        Path to video file.
    frame_index : int
        Frame index to load.

    Returns
    -------
    NDArray[np.uint8]
        RGB frame array with shape (H, W, 3).

    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    IndexError
        If frame_index is out of range.

    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    from neurospatial.animation._video_io import VideoReader

    reader = VideoReader(str(video_path))
    try:
        return reader[frame_index]
    except (IndexError, KeyError) as e:
        raise IndexError(
            f"frame_index={frame_index} is out of range. "
            f"Video '{video_path.name}' contains {reader.n_frames} frames "
            f"(valid indices: 0 to {reader.n_frames - 1}).\n\n"
            f"frame_index specifies which video frame to use for annotation "
            "(0 = first frame).",
        ) from e


def _process_initial_boundary(
    initial_boundary: Polygon | NDArray[np.float64] | None,
    boundary_config: BoundaryConfig | None,
    show_positions: bool,
) -> tuple[Polygon | None, NDArray[np.float64] | None]:
    """Process initial boundary into polygon and optional positions for display.

    Parameters
    ----------
    initial_boundary : Polygon, NDArray, or None
        Pre-drawn boundary or position data to infer boundary from.
    boundary_config : BoundaryConfig or None
        Configuration for boundary inference when initial_boundary is an array.
    show_positions : bool
        If True and initial_boundary is an array, return positions for display.

    Returns
    -------
    boundary_polygon : Polygon or None
        The boundary polygon (either provided directly or inferred).
    positions_for_display : NDArray or None
        Position data for display layer (only if show_positions=True and
        initial_boundary was an array).

    """
    if initial_boundary is None:
        return None, None

    if isinstance(initial_boundary, np.ndarray):
        from neurospatial.annotation._boundary_inference import boundary_from_positions

        boundary_polygon = boundary_from_positions(
            initial_boundary,
            config=boundary_config,
        )
        positions_for_display = initial_boundary if show_positions else None
        return boundary_polygon, positions_for_display
    # Assume Shapely Polygon
    return initial_boundary, None


def _setup_annotation_viewer(
    video_path: Path,
    frame: NDArray[np.uint8],
    mode: Literal["environment", "regions", "both"],
    boundary_polygon: Polygon | None,
    positions_for_display: NDArray[np.float64] | None,
    initial_regions: Regions | None,
    calibration: VideoCalibration | None,
) -> tuple[napari.Viewer, napari.layers.Shapes]:
    """Create and configure napari viewer for annotation.

    Parameters
    ----------
    video_path : Path
        Path to video file (for window title).
    frame : NDArray[np.uint8]
        Video frame to display.
    mode : {"environment", "regions", "both"}
        Annotation mode.
    boundary_polygon : Polygon or None
        Initial boundary polygon to display.
    positions_for_display : NDArray or None
        Position data for reference Points layer.
    initial_regions : Regions or None
        Pre-existing regions to display.
    calibration : VideoCalibration or None
        Coordinate transform for conversions.

    Returns
    -------
    viewer : napari.Viewer
        Configured napari viewer.
    shapes : napari.layers.Shapes
        Shapes layer for annotations.

    """
    import napari

    from neurospatial.annotation._napari_widget import (
        add_initial_boundary_to_shapes,
        create_annotation_widget,
        setup_shapes_layer_for_annotation,
    )

    # Create viewer with reasonable default size
    viewer = napari.Viewer(title=f"Annotate: {video_path.name}")
    viewer.window.resize(1400, 900)
    viewer.add_image(frame, name="video_frame", rgb=True)

    # Determine initial mode based on user intent
    initial_annotation_mode: RegionType = (
        "region" if mode == "regions" else "environment"
    )

    # Add positions layer FIRST so it appears below shapes
    if positions_for_display is not None:
        _add_positions_layer(viewer, positions_for_display, calibration)

    # Add shapes layer with annotation settings
    shapes = setup_shapes_layer_for_annotation(
        viewer,
        initial_mode=initial_annotation_mode,
    )

    # Add initial regions (order matters for feature preservation)
    if initial_regions is not None:
        _add_initial_regions(shapes, initial_regions, calibration)

    # Add initial boundary (prepends to front and reorders)
    if boundary_polygon is not None:
        add_initial_boundary_to_shapes(
            shapes,
            boundary_polygon,
            calibration=calibration,
        )
        # Start in vertex editing mode for immediate adjustment
        shapes.mode = "direct"

    # Add annotation control widget
    widget = create_annotation_widget(
        viewer,
        "Annotations",
        initial_mode=initial_annotation_mode,
    )
    viewer.window.add_dock_widget(
        widget,
        name="Annotation Controls",
        area="right",
    )

    # Set initial status bar
    viewer.status = "Annotation mode: M=cycle modes, Escape=save and close"

    return viewer, shapes


def _process_annotation_results(
    shapes: napari.layers.Shapes,
    mode: Literal["environment", "regions", "both"],
    bin_size: float | None,
    calibration: VideoCalibration | None,
    simplify_tolerance: float | None,
    multiple_boundaries: MultipleBoundaryStrategy,
) -> AnnotationResult:
    """Convert napari shapes to AnnotationResult.

    Parameters
    ----------
    shapes : napari.layers.Shapes
        Shapes layer containing annotations.
    mode : {"environment", "regions", "both"}
        Annotation mode.
    bin_size : float or None
        Bin size for environment discretization.
    calibration : VideoCalibration or None
        Coordinate transform.
    simplify_tolerance : float or None
        Polygon simplification tolerance.
    multiple_boundaries : MultipleBoundaryStrategy
        How to handle multiple environment boundaries.

    Returns
    -------
    AnnotationResult
        Result containing environment and regions.

    """
    from neurospatial.annotation._napari_widget import get_annotation_data
    from neurospatial.annotation.converters import (
        env_from_boundary_region,
        shapes_to_regions,
    )
    from neurospatial.regions import Regions

    # Get annotation data from shapes layer
    shapes_data, names, roles = get_annotation_data(shapes)
    logger.debug(
        "Processing %d shapes: names=%s, roles=%s",
        len(shapes_data),
        names,
        roles,
    )

    # Handle empty annotations
    if not shapes_data:
        logger.debug("No annotations drawn, returning empty result")
        warnings.warn(
            "No annotations were drawn. Returning empty result.",
            UserWarning,
            stacklevel=3,  # Points to annotate_video() caller
        )
        return AnnotationResult(environment=None, regions=Regions([]))

    # Convert shapes to regions
    regions, env_boundary, holes = shapes_to_regions(
        shapes_data,
        names,
        roles,
        calibration,
        simplify_tolerance,
        multiple_boundaries=multiple_boundaries,
    )

    # Warn about mode mismatch: environment boundary drawn but mode="regions"
    if mode == "regions" and env_boundary is not None:
        warnings.warn(
            "An environment boundary was drawn but mode='regions' was specified. "
            "The boundary will be ignored. Use mode='both' or mode='environment' "
            "to include the boundary in the result.",
            UserWarning,
            stacklevel=3,
        )

    # Warn if holes exist but no environment boundary
    if holes and env_boundary is None:
        warnings.warn(
            f"{len(holes)} hole(s) were drawn but no environment boundary exists. "
            "Holes are only meaningful when an environment boundary is defined.",
            UserWarning,
            stacklevel=3,
        )

    # Build environment if requested and boundary exists
    environment = None
    if mode in ("environment", "both") and env_boundary is not None:
        assert bin_size is not None  # Validated earlier
        logger.debug("Building environment with bin_size=%s", bin_size)
        environment = env_from_boundary_region(env_boundary, bin_size, holes=holes)
        logger.debug(
            "Created environment with %d bins, attaching %d regions",
            environment.n_bins,
            len(regions),
        )
        # Attach regions to environment
        for name, region in regions.items():
            environment.regions.add(
                name,
                polygon=region.data,
                metadata=dict(region.metadata),
            )

    logger.debug(
        "Annotation complete: environment=%s, regions=%d",
        "created" if environment else "None",
        len(regions),
    )
    return AnnotationResult(environment=environment, regions=regions)
