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
    from collections.abc import Callable

    import napari
    from numpy.typing import NDArray
    from shapely.geometry import LineString, Polygon

    from neurospatial.annotation._boundary_inference import BoundaryConfig
    from neurospatial.annotation._napari_widget import _AnnotationWidget
    from neurospatial.annotation.project import AnnotationProject
    from neurospatial.annotation.segmentation import SegmentationProvider
    from neurospatial.environment import Environment
    from neurospatial.landmarks import Landmarks
    from neurospatial.ops.transforms import VideoCalibration
    from neurospatial.regions import Regions


__all__ = ["AnnotationResult", "annotate_project", "annotate_video"]


class AnnotationResult(NamedTuple):
    """Result from an annotation session.

    Attributes
    ----------
    environment : Environment or None
        Discretized environment if boundary was annotated, else None. It
        holds the regions, the walls (clipped to the boundary) and the
        landmarks.
    regions : Regions
        All annotated regions (excluding environment boundary).
    walls : dict of str to LineString
        Each drawn wall by name, as drawn. Pass them to
        ``OpenArena(polygon, walls=walls)`` to build another environment.
    landmarks : Landmarks
        The drawn landmarks; attach them with ``env.with_landmarks``.

    """

    environment: Environment | None
    regions: Regions
    walls: dict[str, LineString]
    landmarks: Landmarks


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
    project_path: str | Path | None = None,
    autosave_interval: float | None = 30.0,
    segmentation_provider: SegmentationProvider | None = None,
) -> AnnotationResult:
    """Launch interactive napari annotation on a video frame.

    Opens a napari viewer with the specified video frame. Users can draw
    polygons to define an environment boundary and/or named regions, paths
    for named walls, and named landmarks: an ellipse marks a point (a reward
    port or a cue), a path a line (a monitor) and a polygon an area (an
    object). Press M to cycle through these. After closing the viewer,
    annotations are converted to Regions, walls and Landmarks, and optionally
    an Environment that holds all of them.

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
        If None, uses BoundaryConfig defaults (alpha_shape, 2% buffer, 1% simplify).
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

    project_path : str or Path, optional
        Save the editable draft, including its reference image, to this file.
        A fresh session requires a new path; use :func:`annotate_project` to
        resume an existing project. Without a path, saving is available in the UI.
    autosave_interval : float or None, default=30.0
        Seconds between draft saves when a project path is set. Must be finite
        and positive; None disables periodic saves.
    segmentation_provider : SegmentationProvider, optional
        Provider for interactive outline proposals. The default CPU region
        grower needs no model download. Proposals require explicit acceptance.

    Returns
    -------
    AnnotationResult
        Named tuple containing:
        - environment: Environment or None
        - regions: Regions collection
        - walls: dict of wall name to LineString
        - landmarks: Landmarks collection

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
    Simple annotation (pixel coordinates):

    >>> from neurospatial.annotation import annotate_video  # doctest: +SKIP
    >>> result = annotate_video("experiment.mp4", bin_size=10.0)  # doctest: +SKIP
    >>> print(result.environment)  # doctest: +SKIP
    >>> print(result.regions)  # doctest: +SKIP

    With calibration (cm coordinates):

    >>> from neurospatial.ops.transforms import (  # doctest: +SKIP
    ...     VideoCalibration,
    ...     calibrate_from_scale_bar,
    ... )
    >>> transform = calibrate_from_scale_bar(
    ...     (0, 0), (200, 0), 100.0, (640, 480)
    ... )  # doctest: +SKIP
    >>> calib = VideoCalibration(transform, (640, 480))  # doctest: +SKIP
    >>> result = annotate_video(
    ...     "experiment.mp4", calibration=calib, bin_size=2.0
    ... )  # doctest: +SKIP

    Notes
    -----
    This function blocks until the napari viewer is closed. The viewer runs
    in the same Python process, and the function returns only after the user
    closes it via the "Done" button or window close. Escape keeps napari's
    normal behavior for canceling the current drawing or selection.

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
    _validate_autosave_interval(autosave_interval)
    if project_path is not None and Path(project_path).exists():
        raise FileExistsError(
            f"Annotation project already exists: {project_path}. "
            "Use annotate_project to resume it, or choose a new project_path."
        )
    if initial_regions is not None:
        _validate_initial_metadata(initial_regions)
    _require_napari()

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

    return _run_annotation_editor(
        frame,
        video_path=video_path,
        frame_index=resolved.frame_index,
        calibration=calibration,
        mode=mode,
        bin_size=bin_size,
        simplify_tolerance=resolved.simplify_tolerance,
        multiple_boundaries=resolved.multiple_boundaries,
        project_path=project_path,
        autosave_interval=autosave_interval,
        segmentation_provider=segmentation_provider,
        boundary_polygon=boundary_polygon,
        positions_for_display=positions_for_display,
        initial_regions=initial_regions,
    )


def annotate_project(
    project: AnnotationProject | str | Path,
    *,
    project_path: str | Path | None = None,
    autosave_interval: float | None = 30.0,
    segmentation_provider: SegmentationProvider | None = None,
) -> AnnotationResult:
    """Resume an editable annotation project without the original video.

    The embedded RGB frame, pixel geometry, calibration, labels and export
    settings are restored before the annotation controls are created. Invalid
    draft polygons remain editable; export succeeds after they are repaired.

    Parameters
    ----------
    project : AnnotationProject or str or Path
        In-memory draft or saved project file. The source video is never opened.
    project_path : str or Path, optional
        Destination for draft saves. When resuming a file, defaults to that file.
        For an in-memory project, choose a destination in the UI or pass one here.
    autosave_interval : float or None, default=30.0
        Seconds between saves to the current project path. Must be positive
        and finite; None disables periodic saves.
    segmentation_provider : SegmentationProvider, optional
        Outline proposal provider; defaults to the CPU region grower.

    Returns
    -------
    AnnotationResult
        Validated regions, walls, landmarks and optional discretized environment
        using the session's current calibration and bin size.

    Raises
    ------
    ValueError
        If the project is corrupt, has an unsupported version, requires a
        missing bin size, or autosave settings are invalid. Invalid geometry
        must be edited before export.
    ImportError
        If napari is not installed.

    Examples
    --------
    >>> from neurospatial.annotation import annotate_project
    >>> result = annotate_project("arena.project")  # doctest: +SKIP

    See Also
    --------
    annotate_video : Start a new annotation from a video frame.
    AnnotationProject : Portable editable draft with its reference frame.
    """
    from neurospatial.annotation._project_layer import _validate_shapes_for_display
    from neurospatial.annotation.project import AnnotationProject

    _validate_autosave_interval(autosave_interval)
    if isinstance(project, (str, Path)):
        source = Path(project)
        draft = AnnotationProject.load(source)
        if project_path is None:
            project_path = source
    elif isinstance(project, AnnotationProject):
        draft = project
    else:
        raise TypeError("project must be an AnnotationProject or a project file path.")
    _validate_annotate_params(draft.mode, draft.bin_size)
    _validate_shapes_for_display(draft.shapes)
    _require_napari()
    return _run_annotation_editor(
        draft.frame,
        video_path=None if draft.video_path is None else Path(draft.video_path),
        frame_index=draft.frame_index,
        calibration=draft.calibration,
        mode=draft.mode,
        bin_size=draft.bin_size,
        simplify_tolerance=draft.simplify_tolerance,
        multiple_boundaries=draft.multiple_boundaries,
        project_path=project_path,
        autosave_interval=autosave_interval,
        segmentation_provider=segmentation_provider,
        draft=draft,
    )


def _require_napari() -> None:
    """Fail before loading frames or creating a viewer without the GUI extra."""
    try:
        import napari  # noqa: F401 - verify the optional GUI dependency
    except ImportError as exc:
        raise ImportError(
            "napari is required for interactive annotation. "
            "Install with: pip install napari[all]"
        ) from exc


def _validate_initial_metadata(regions: Regions) -> None:
    """Reject polygon metadata that an editable project cannot store."""
    from neurospatial.annotation.project import _json_value
    from neurospatial.regions.core import _thaw_metadata

    for name, region in regions.items():
        if region.kind != "polygon":
            continue
        try:
            _json_value(_thaw_metadata(region.metadata))
        except ValueError as exc:
            raise ValueError(
                f"Cannot edit region {name!r}: {exc} Editable annotations store "
                "metadata as JSON objects, whose keys are strings. Convert the "
                "region's metadata keys to strings before annotating."
            ) from exc


def _validate_autosave_interval(autosave_interval: float | None) -> None:
    """Reject invalid timer settings before importing or creating a viewer."""
    if autosave_interval is not None and (
        isinstance(autosave_interval, bool)
        or not isinstance(autosave_interval, (int, float, np.integer, np.floating))
        or not np.isfinite(autosave_interval)
        or autosave_interval <= 0
    ):
        raise ValueError("autosave_interval must be positive and finite, or None.")


def _run_annotation_editor(
    frame: NDArray[np.uint8],
    *,
    video_path: Path | None,
    frame_index: int,
    calibration: VideoCalibration | None,
    mode: Literal["environment", "regions", "both"],
    bin_size: float | None,
    simplify_tolerance: float | None,
    multiple_boundaries: MultipleBoundaryStrategy,
    project_path: str | Path | None,
    autosave_interval: float | None,
    segmentation_provider: SegmentationProvider | None,
    boundary_polygon: Polygon | None = None,
    positions_for_display: NDArray[np.float64] | None = None,
    initial_regions: Regions | None = None,
    draft: AnnotationProject | None = None,
) -> AnnotationResult:
    """Run the common editor and export from its current scientific settings."""
    from neurospatial.annotation._session import run_annotation_session
    from neurospatial.annotation._workflow import AnnotationProjectSession

    viewer, shapes, flush_pending, controls = _setup_annotation_viewer(
        video_path or Path("Annotation project"),
        frame,
        mode,
        boundary_polygon,
        positions_for_display,
        initial_regions,
        calibration,
        draft=draft,
    )
    try:
        session = AnnotationProjectSession(
            viewer,
            shapes,
            controls,
            frame=frame,
            video_path=video_path,
            frame_index=frame_index,
            calibration=calibration,
            mode=mode,
            bin_size=bin_size,
            simplify_tolerance=simplify_tolerance,
            multiple_boundaries=multiple_boundaries,
            project_path=project_path,
            autosave_interval=autosave_interval,
            segmentation_provider=segmentation_provider,
        )
    except Exception:
        viewer.close()
        raise

    def prepare_result() -> AnnotationResult:
        flush_pending()
        _validate_annotate_params(mode, session.bin_size)
        result = _process_annotation_results(
            shapes,
            mode,
            session.bin_size,
            session.calibration,
            simplify_tolerance,
            multiple_boundaries,
        )
        session.save_if_path()
        return result

    try:
        return run_annotation_session(viewer, prepare_result)
    finally:
        session.dispose()


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
    from neurospatial.annotation._draft import add_polygon

    for name, region in regions.items():
        if region.kind == "polygon":
            add_polygon(
                shapes_layer,
                region.data,
                name,
                region.metadata.get("role", "region"),
                calibration,
                region.metadata,
            )


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

    from neurospatial.animation import _VideoReader

    # Use the reader as a context manager so its resources (e.g. the prefetch
    # thread pool) are always released, even when frame_index is out of range.
    with _VideoReader(str(video_path)) as reader:
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
    *,
    draft: AnnotationProject | None = None,
) -> tuple[napari.Viewer, napari.layers.Shapes, Callable[[], None], _AnnotationWidget]:
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
    flush_pending : Callable[[], None]
        Finish pending widget edits before constructing an annotation result.
    controls : _AnnotationWidget
        Annotation controls used by the shared project workflow.

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

    # Restore ring identities, primitive types and labels before widget state is
    # initialized, so reopening cannot rename or reclassify saved shapes.
    if draft is not None:
        from neurospatial.annotation._project_layer import restore_project_shapes

        restore_project_shapes(draft, shapes)
        if draft.shapes:
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
    viewer.status = "Annotation mode: M=cycle roles; Save Project keeps your draft; Done returns results"

    return viewer, shapes, widget.flush_pending, widget


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
        _drafts_to_regions,
        env_from_boundary_region,
        shapes_to_walls_and_landmarks,
    )
    from neurospatial.landmarks import Landmarks
    from neurospatial.regions import Regions

    # Get annotation data from shapes layer
    shapes_data, names, roles, shape_types = get_annotation_data(shapes)
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
        return AnnotationResult(
            environment=None, regions=Regions([]), walls={}, landmarks=Landmarks()
        )

    from neurospatial.annotation._draft import polygon_drafts

    drafts = polygon_drafts(shapes, shapes_data, names, roles)

    from neurospatial.annotation.converters import _validate_drawn_shapes

    _validate_drawn_shapes(shapes_data, names, roles, shape_types, calibration)
    walls, landmarks = shapes_to_walls_and_landmarks(
        shapes_data, names, roles, shape_types, calibration=calibration
    )
    # Convert the area shapes to regions
    regions, env_boundary, holes = _drafts_to_regions(
        [draft for draft in drafts if draft.role not in ("wall", "landmark")],
        calibration=calibration,
        simplify_tolerance=simplify_tolerance,
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
        environment = env_from_boundary_region(
            env_boundary, bin_size, holes=holes, walls=walls
        )
        logger.debug(
            "Created environment with %d bins, attaching %d regions",
            environment.n_bins,
            len(regions),
        )
        # Attach regions and landmarks to environment
        environment = environment.with_regions(regions).with_landmarks(landmarks)
    elif walls:
        warnings.warn(
            f"{len(walls)} wall(s) were drawn but no environment was built, so "
            "result.walls holds them unattached. Draw an environment boundary "
            "with mode='both' or mode='environment' to apply them.",
            UserWarning,
            stacklevel=3,
        )

    logger.debug(
        "Annotation complete: environment=%s, regions=%d, walls=%d, landmarks=%d",
        "created" if environment else "None",
        len(regions),
        len(walls),
        len(landmarks),
    )
    return AnnotationResult(
        environment=environment, regions=regions, walls=walls, landmarks=landmarks
    )
