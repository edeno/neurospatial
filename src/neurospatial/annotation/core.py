"""Main annotation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    import napari

    from neurospatial import Environment
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

    Returns
    -------
    AnnotationResult
        Named tuple containing:
        - environment: Environment or None
        - regions: Regions collection

    Raises
    ------
    ValueError
        If bin_size is not provided when mode requires environment creation.
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
    The napari viewer runs in blocking mode. The function returns only after
    the user closes the viewer (via the "Save and Close" button or window close).

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

    # Validate parameters
    if mode in ("environment", "both") and bin_size is None:
        raise ValueError(
            f"bin_size is required when mode={mode!r}. "
            "Provide bin_size for environment discretization."
        )

    # Validate video path
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load video frame
    reader = VideoReader(str(video_path))
    frame = reader[frame_index]  # (H, W, 3) RGB uint8

    # Create viewer with reasonable default size for annotation work
    viewer = napari.Viewer(title=f"Annotate: {video_path.name}")
    viewer.window.resize(1400, 900)  # Larger window for comfortable annotation
    viewer.add_image(frame, name="video_frame", rgb=True)

    # Determine initial annotation mode based on user's intent
    # "regions" mode → start in region mode (user doesn't want environment boundary)
    # "environment" or "both" → start in environment mode
    initial_annotation_mode = "region" if mode == "regions" else "environment"

    # Add shapes layer with annotation-optimized settings
    # (features-based coloring, text labels, keyboard shortcuts)
    shapes = setup_shapes_layer_for_annotation(
        viewer, initial_mode=initial_annotation_mode
    )

    # Add existing regions if provided
    if initial_regions is not None:
        _add_initial_regions(shapes, initial_regions, calibration)

    # Add annotation control widget (magicgui-based)
    widget = create_annotation_widget(
        viewer, "Annotations", initial_mode=initial_annotation_mode
    )
    viewer.window.add_dock_widget(
        widget,
        name="Annotation Controls",
        area="right",
    )

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
    regions, env_boundary = shapes_to_regions(
        shapes_data, names, roles, calibration, simplify_tolerance
    )

    # Build environment if requested and boundary exists
    environment = None
    if mode in ("environment", "both") and env_boundary is not None:
        # bin_size was validated at function start for these modes
        assert bin_size is not None
        environment = env_from_boundary_region(env_boundary, bin_size)
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

    from neurospatial.annotation._napari_widget import (
        ROLE_COLORS,
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

        # Restore feature_defaults for new shapes (environment first)
        shapes_layer.feature_defaults["role"] = "environment"
        shapes_layer.feature_defaults["name"] = "arena"
        shapes_layer.current_face_color = ROLE_COLORS["environment"]
