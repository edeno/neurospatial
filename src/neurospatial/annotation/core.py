"""Main annotation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from neurospatial.annotation._types import MultipleBoundaryStrategy, Role

if TYPE_CHECKING:
    import napari
    from numpy.typing import NDArray
    from shapely.geometry import Polygon

    from neurospatial import Environment
    from neurospatial.annotation._boundary_inference import BoundaryConfig
    from neurospatial.regions import Regions
    from neurospatial.transforms import VideoCalibration


class AnnotationResult(NamedTuple):
    """
    Result from an annotation session.

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
    frame_index: int = 0,
    initial_regions: Regions | None = None,
    calibration: VideoCalibration | None = None,
    mode: Literal["environment", "regions", "both"] = "both",
    bin_size: float | None = None,
    simplify_tolerance: float | None = None,
    multiple_boundaries: MultipleBoundaryStrategy = "last",
    initial_boundary: Polygon | NDArray[np.float64] | None = None,
    boundary_config: BoundaryConfig | None = None,
    show_positions: bool = False,
) -> AnnotationResult:
    """
    Launch interactive napari annotation on a video frame.

    Opens a napari viewer with the specified video frame. Users can draw
    polygons to define an environment boundary and/or named regions.
    After closing the viewer, annotations are converted to Regions and
    optionally an Environment.

    Parameters
    ----------
    video_path : str or Path
        Path to video file (any format supported by OpenCV).
    frame_index : int, default=0
        Which frame to display for annotation.
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
    simplify_tolerance : float, optional
        If provided, simplifies hand-drawn polygons using Douglas-Peucker
        algorithm. Removes jagged edges from freehand drawing. Tolerance
        is in output units (cm if calibration provided, else pixels).
        Recommended: 1.0-2.0 for typical use cases.
    multiple_boundaries : {"last", "first", "error"}, default="last"
        How to handle multiple environment boundaries:

        - "last": Use the last drawn boundary (default). A warning is emitted.
        - "first": Use the first drawn boundary. A warning is emitted.
        - "error": Raise ValueError if multiple boundaries are drawn.
    initial_boundary : Polygon or NDArray, optional
        Pre-drawn boundary for editing. Can be:

        - Shapely Polygon: Used directly as boundary
        - NDArray (n, 2): Position data to infer boundary from

        If None, user draws boundary manually.
    boundary_config : BoundaryConfig, optional
        Configuration for boundary inference when initial_boundary is an array.
        If None, uses BoundaryConfig defaults (convex_hull, 2% buffer, 1% simplify).
    show_positions : bool, default=False
        If True and initial_boundary is an array, show positions as a
        Points layer for reference while editing.

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
    """
    # Validate parameters early (before expensive imports)
    if mode in ("environment", "both") and bin_size is None:
        raise ValueError(
            f"bin_size is required when mode={mode!r}. "
            "Provide bin_size for environment discretization."
        )

    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari is required for interactive annotation. "
            "Install with: pip install napari[all]"
        ) from e

    from neurospatial.animation._video_io import VideoReader
    from neurospatial.annotation._napari_widget import (
        create_annotation_widget,
        get_annotation_data,
        setup_shapes_layer_for_annotation,
    )
    from neurospatial.annotation.converters import (
        env_from_boundary_region,
        shapes_to_regions,
    )
    from neurospatial.regions import Regions

    # Validate video path
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load video frame
    reader = VideoReader(str(video_path))
    try:
        frame = reader[frame_index]  # (H, W, 3) RGB uint8
    except (IndexError, KeyError) as e:
        raise IndexError(
            f"Frame index {frame_index} is out of range for video '{video_path.name}'. "
            f"Video has {reader.n_frames} frames (indices 0-{reader.n_frames - 1})."
        ) from e

    # Process initial boundary
    boundary_polygon = None
    positions_for_display = None

    if initial_boundary is not None:
        if isinstance(initial_boundary, np.ndarray):
            # Infer boundary from positions
            from neurospatial.annotation._boundary_inference import (
                boundary_from_positions,
            )

            boundary_polygon = boundary_from_positions(
                initial_boundary,
                config=boundary_config,
            )
            if show_positions:
                positions_for_display = initial_boundary
        else:
            # Assume Shapely Polygon
            boundary_polygon = initial_boundary

    # Handle conflict: initial_boundary vs environment region in initial_regions
    # initial_boundary takes precedence - warn if both provided
    if boundary_polygon is not None and initial_regions is not None:
        initial_regions = _filter_environment_regions(initial_regions)

    # Create viewer with reasonable default size for annotation work
    viewer = napari.Viewer(title=f"Annotate: {video_path.name}")
    viewer.window.resize(1400, 900)  # Larger window for comfortable annotation
    viewer.add_image(frame, name="video_frame", rgb=True)

    # Determine initial annotation mode based on user's intent
    # "regions" mode → start in region mode (user doesn't want environment boundary)
    # "environment" or "both" → start in environment mode
    initial_annotation_mode: Role = "region" if mode == "regions" else "environment"

    # Add shapes layer with annotation-optimized settings
    # (features-based coloring, text labels, keyboard shortcuts)
    shapes = setup_shapes_layer_for_annotation(
        viewer, initial_mode=initial_annotation_mode
    )

    # IMPORTANT: Order matters for feature preservation
    # 1. First add initial_regions (if any) - these are non-environment regions
    if initial_regions is not None:
        _add_initial_regions(shapes, initial_regions, calibration)

    # 2. Then add initial_boundary - this prepends to front and reorders
    if boundary_polygon is not None:
        from neurospatial.annotation._napari_widget import (
            add_initial_boundary_to_shapes,
        )

        add_initial_boundary_to_shapes(
            shapes,
            boundary_polygon,
            calibration=calibration,
        )

    # 3. Finally add positions layer (separate layer, no conflict)
    if positions_for_display is not None:
        _add_positions_layer(viewer, positions_for_display, calibration)

    # Add annotation control widget (magicgui-based)
    widget = create_annotation_widget(
        viewer, "Annotations", initial_mode=initial_annotation_mode
    )
    viewer.window.add_dock_widget(
        widget,
        name="Annotation Controls",
        area="right",
    )

    # Set initial status bar with shortcut reminder
    viewer.status = "Annotation mode: M=cycle modes, Escape=save and close"

    # Run napari (blocking until viewer closes)
    napari.run()

    # Get annotation data from shapes layer
    shapes_data, names, roles = get_annotation_data(shapes)

    # Handle empty annotations
    if not shapes_data:
        import warnings

        warnings.warn(
            "No annotations were drawn. Returning empty result.",
            UserWarning,
            stacklevel=2,
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

    # Warn if holes exist but no environment boundary
    if holes and env_boundary is None:
        import warnings

        warnings.warn(
            f"{len(holes)} hole(s) were drawn but no environment boundary exists. "
            "Holes are only meaningful when an environment boundary is defined.",
            UserWarning,
            stacklevel=2,
        )

    # Build environment if requested and boundary exists
    environment = None
    if mode in ("environment", "both") and env_boundary is not None:
        # bin_size was validated at function start for these modes
        assert bin_size is not None
        environment = env_from_boundary_region(env_boundary, bin_size, holes=holes)
        # Attach regions to environment
        for name, region in regions.items():
            environment.regions.add(
                name,
                polygon=region.data,
                metadata=dict(region.metadata),
            )

    return AnnotationResult(environment=environment, regions=regions)


def _add_initial_regions(
    shapes_layer: napari.layers.Shapes,
    regions: Regions,
    calibration: VideoCalibration | None,
) -> None:
    """
    Add existing regions to shapes layer for editing.

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
    """
    Add positions as semi-transparent Points layer for reference.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    positions : NDArray[np.float64]
        Position data. Coordinate system depends on calibration:

        - With calibration: environment units (cm), Y-up origin
        - Without calibration: video pixels (x, y), Y-down origin
    calibration : VideoCalibration, optional
        Transform from environment coords (cm) to video pixels.

    Notes
    -----
    Mirrors the pattern in add_initial_boundary_to_shapes for consistency.
    """
    coords = positions.copy()

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
        size=3,
        face_color="cyan",
        opacity=0.3,
        blending="translucent",
    )


def _filter_environment_regions(regions: Regions) -> Regions:
    """
    Filter out environment regions and warn if any were found.

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
        import warnings

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
