"""Convert between napari shapes and neurospatial regions, walls and landmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import shapely.geometry as shp
from shapely import Polygon

from neurospatial.annotation._types import MultipleBoundaryStrategy, RegionType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from neurospatial.annotation._draft import PolygonDraft
    from neurospatial.environment import Environment
    from neurospatial.landmarks import Landmark, Landmarks
    from neurospatial.ops.transforms import VideoCalibration
    from neurospatial.regions import Region, Regions

# Napari shape types that are drawn as open lines, and as closed areas.
_LINE_SHAPES = frozenset({"path", "line"})
_AREA_SHAPES = frozenset({"polygon", "rectangle"})


def _to_world(
    vertices_rc: NDArray[np.float64], calibration: VideoCalibration | None
) -> NDArray[np.float64]:
    """Napari (row, col) vertices as (x, y), in cm with a calibration."""
    pts_px = vertices_rc[:, ::-1].astype(np.float64)
    return pts_px if calibration is None else calibration.transform_px_to_cm(pts_px)


def _landmark_from_shape(
    name: str, shape_type: str, points: NDArray[np.float64]
) -> Landmark:
    """Build a landmark whose kind follows from its napari shape type.

    An ellipse marks a point at its center, a path or line marks a line, and
    any other shape marks a polygon.
    """
    from neurospatial.landmarks import Landmark

    if shape_type == "ellipse":
        return Landmark(name, "point", points.mean(axis=0))
    if shape_type in _LINE_SHAPES:
        return Landmark(name, "line", points)
    return Landmark(name, "polygon", points)


def shapes_to_walls_and_landmarks(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    region_types: list[RegionType],
    shape_types: list[str],
    *,
    calibration: VideoCalibration | None = None,
) -> tuple[dict[str, shp.LineString], Landmarks]:
    """Convert napari wall and landmark shapes to walls and landmarks.

    A wall drawn as a path or line is that line; one drawn as a polygon or
    rectangle is its closed outline. A landmark's kind follows from its shape:
    an ellipse marks a point at its center (a reward port or a cue), a path or
    line a line (a monitor), and a polygon or rectangle an area (an object).
    Shapes of other annotation types are ignored.

    Parameters
    ----------
    shapes_data : list of NDArray
        Vertices of each shape, shape (n_vertices, 2), in napari (row, col)
        order.
    names : list of str
        Name for each shape.
    region_types : list of RegionType
        Annotation type for each shape.
    shape_types : list of str
        Napari shape type for each shape, such as "path" or "ellipse".
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm).

    Returns
    -------
    walls : dict of str to LineString
        Each wall by name.
    landmarks : Landmarks
        The landmarks, in drawing order.

    Warns
    -----
    UserWarning
        For a shape that cannot become a wall or landmark (an ellipse drawn as
        a wall, or too few vertices); it is skipped.
    """
    import warnings

    from neurospatial.landmarks import Landmarks

    walls: dict[str, shp.LineString] = {}
    landmarks: list[Landmark] = []
    for vertices_rc, name, region_type, shape_type in zip(
        shapes_data, names, region_types, shape_types, strict=True
    ):
        if region_type not in ("wall", "landmark"):
            continue
        points = _to_world(vertices_rc, calibration)
        if region_type == "wall":
            if shape_type in _AREA_SHAPES:
                points = np.vstack([points, points[:1]])
            if shape_type == "ellipse" or len(points) < 2:
                warnings.warn(
                    f"Skipping wall '{name}': draw a wall as a path, a line or a "
                    f"polygon, not a {shape_type} with {len(points)} vertices.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            walls[str(name)] = shp.LineString(points)
            continue
        try:
            landmark = _landmark_from_shape(str(name), shape_type, points)
        except ValueError as error:
            warnings.warn(
                f"Skipping landmark '{name}': {error}",
                UserWarning,
                stacklevel=2,
            )
            continue
        landmarks.append(landmark)
    return walls, Landmarks(landmarks)


def shapes_to_regions(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    region_types: list[RegionType],
    *,
    calibration: VideoCalibration | None = None,
    simplify_tolerance: float | None = None,
    multiple_boundaries: MultipleBoundaryStrategy = "last",
    validate: bool = True,
    min_area: float = 1e-6,
) -> tuple[Regions, Region | None, list[Region]]:
    """Convert napari polygon shapes to Regions.

    Parameters
    ----------
    shapes_data : list of NDArray
        List of polygon vertices from napari Shapes layer. Each array has
        shape (n_vertices, 2) in napari (row, col) order.
    names : list of str
        Name for each shape.
    region_types : list of RegionType
        Type for each shape: "environment", "hole", or "region".
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm)
        using ``calibration.transform_px_to_cm``.
    simplify_tolerance : float, optional
        If provided, simplifies polygons using Shapely's Douglas-Peucker
        algorithm. Tolerance is in output coordinate units (cm if calibration
        provided, else pixels). Recommended: 1.0 for cleaner boundaries.
    multiple_boundaries : {"last", "first", "error"}, default="last"
        How to handle multiple environment boundaries:

        - "last": Use the last drawn boundary (default). A warning is emitted.
        - "first": Use the first drawn boundary. A warning is emitted.
        - "error": Raise ValueError if multiple boundaries are drawn.
    validate : bool, default=True
        Whether to emit advisory warnings about small areas, overlap and
        regions outside the boundary. Invalid polygon geometry always raises,
        including when this is False; simplification does not repair it.
    min_area : float, default=1e-6
        Minimum polygon area threshold for validation. Polygons with area
        below this trigger a warning (if validate=True).

    Returns
    -------
    regions : Regions
        All regions with role="region".
    env_boundary : Region or None
        The region with role="environment", if any.
    holes : list of Region
        All regions with role="hole" (to be subtracted from environment).

    Raises
    ------
    ValueError
        If polygon geometry is invalid, or if ``multiple_boundaries="error"``
        and multiple environment boundaries are drawn. Invalid geometry is
        rejected even when ``validate=False``; it is never repaired implicitly.

    Notes
    -----
    Napari shapes use (row, col) order. This function converts to (x, y)
    pixel coordinates before applying calibration.

    Holes are only meaningful when an environment boundary exists. They are
    used to create excluded areas within the environment.

    """
    from neurospatial.annotation._draft import PolygonDraft

    drafts = [
        PolygonDraft(vertices, name, role, [], {})
        for vertices, name, role in zip(shapes_data, names, region_types, strict=True)
    ]
    return _drafts_to_regions(
        drafts,
        calibration=calibration,
        simplify_tolerance=simplify_tolerance,
        multiple_boundaries=multiple_boundaries,
        validate=validate,
        min_area=min_area,
    )


def _drafts_to_regions(
    drafts: list[PolygonDraft],
    *,
    calibration: VideoCalibration | None = None,
    simplify_tolerance: float | None = None,
    multiple_boundaries: MultipleBoundaryStrategy = "last",
    validate: bool = True,
    min_area: float = 1e-6,
) -> tuple[Regions, Region | None, list[Region]]:
    """Convert explicitly grouped polygon drafts without losing interior rings."""
    import warnings

    from neurospatial.regions import Region, Regions

    regions_list: list[Region] = []
    holes_list: list[Region] = []
    env_boundaries: list[Region] = []

    coord_system = "pixels" if calibration is None else "cm"
    for draft in drafts:
        poly_rc, name, region_type = draft.vertices, draft.name, draft.role
        pts_world = _to_world(poly_rc, calibration)

        # Skip degenerate polygons (need ≥3 vertices to form an area), but warn
        # so the user knows a drawn shape was dropped rather than annotated.
        if len(pts_world) < 3:
            warnings.warn(
                f"Skipping shape '{name}': a polygon needs at least 3 vertices "
                f"to define an area, but this shape has {len(pts_world)}.",
                UserWarning,
                stacklevel=2,
            )
            continue

        poly = shp.Polygon(
            pts_world, [_to_world(ring, calibration) for ring in draft.interiors]
        )
        if not poly.is_valid:
            from shapely.validation import explain_validity

            raise ValueError(
                f"Polygon '{name}' is invalid: {explain_validity(poly)}. "
                "Edit its vertices or interior rings before saving."
            )

        # Validate polygon geometry before simplification
        if validate:
            from neurospatial.annotation.validation import validate_polygon_geometry

            validate_polygon_geometry(poly, str(name), min_area=min_area)

        # Optional simplification (Douglas-Peucker algorithm)
        if simplify_tolerance is not None:
            poly = poly.simplify(tolerance=simplify_tolerance, preserve_topology=True)
            # Ensure polygon is still valid after simplification
            if not poly.is_valid:
                raise ValueError(
                    f"Simplified polygon '{name}' is invalid; edit its vertices."
                )

        metadata = {
            **draft.metadata,
            "source": "napari_annotation",
            "coord_system": coord_system,
            "role": region_type,  # Keep 'role' key for backward compat in metadata
        }

        region = Region(
            name=str(name),
            kind="polygon",
            data=poly,
            metadata=metadata,
        )

        if region_type == "environment":
            env_boundaries.append(region)
        elif region_type == "hole":
            holes_list.append(region)
        else:
            regions_list.append(region)

    # Handle multiple environment boundaries according to strategy
    env_boundary: Region | None = None
    if len(env_boundaries) > 1:
        if multiple_boundaries == "error":
            names_list = [b.name for b in env_boundaries]
            raise ValueError(
                f"Multiple environment boundaries ({len(env_boundaries)}) were drawn: "
                f"{names_list}. Set multiple_boundaries='last' or 'first' to select one, "
                "or draw only one environment boundary.",
            )
        if multiple_boundaries == "first":
            env_boundary = env_boundaries[0]
            warnings.warn(
                f"Multiple environment boundaries ({len(env_boundaries)}) were drawn. "
                f"Using the first one ('{env_boundary.name}') as specified by "
                "multiple_boundaries='first'.",
                UserWarning,
                stacklevel=2,
            )
        else:  # "last" (default)
            env_boundary = env_boundaries[-1]
            warnings.warn(
                f"Multiple environment boundaries ({len(env_boundaries)}) were drawn. "
                f"Using the last one ('{env_boundary.name}') as specified by "
                "multiple_boundaries='last'.",
                UserWarning,
                stacklevel=2,
            )
    elif len(env_boundaries) == 1:
        env_boundary = env_boundaries[0]

    # Validate region overlap and containment
    if validate:
        from neurospatial.annotation.validation import (
            validate_region_overlap,
            validate_region_within_boundary,
        )

        regions_container = Regions(regions_list)

        # Check for heavy overlap between regions
        validate_region_overlap(regions_container)

        # Check if regions are within boundary
        if env_boundary is not None:
            for region in regions_list:
                validate_region_within_boundary(region, env_boundary)

        return regions_container, env_boundary, holes_list

    return Regions(regions_list), env_boundary, holes_list


def subtract_holes_from_boundary(
    boundary: Region,
    holes: list[Region],
) -> Region:
    """Subtract hole polygons from an environment boundary.

    Uses Shapely's difference operation to create a boundary with holes.
    This is used to create environments with excluded interior areas.

    Parameters
    ----------
    boundary : Region
        The environment boundary polygon.
    holes : list of Region
        Hole polygons to subtract from the boundary.

    Returns
    -------
    Region
        New boundary region with holes subtracted.

    Notes
    -----
    Holes that don't intersect the boundary have no effect.
    The resulting polygon may have interior rings (holes) that
    Environment.from_polygon handles correctly.

    """
    from neurospatial.regions import Region

    if not holes:
        return boundary

    # Start with the boundary polygon
    result_poly = cast("Polygon", boundary.data)

    # Subtract each hole
    for hole in holes:
        hole_poly = cast("Polygon", hole.data)
        result_poly = result_poly.difference(hole_poly)

    # Ensure result is valid
    if not result_poly.is_valid:
        raise ValueError(
            "Subtracting environment holes produced invalid geometry; edit the boundary or holes before saving."
        )
    if result_poly.geom_type != "Polygon":
        raise ValueError(
            f"Subtracting environment holes split the boundary into "
            f"{len(getattr(result_poly, 'geoms', ()))} separate areas. The "
            "environment must be one connected area: shrink or delete the hole "
            "that crosses it."
        )

    # Create new region with updated polygon
    metadata = dict(boundary.metadata)
    metadata["holes_subtracted"] = len(holes)

    return Region(
        name=boundary.name,
        kind="polygon",
        data=result_poly,
        metadata=metadata,
    )


def env_from_boundary_region(
    boundary: Region,
    bin_size: float,
    holes: list[Region] | None = None,
    walls: Mapping[str, shp.LineString] | None = None,
    **from_polygon_kwargs,
) -> Environment:
    """Create Environment from an annotated boundary polygon.

    Parameters
    ----------
    boundary : Region
        Region with kind="polygon" defining the environment boundary.
    bin_size : float
        Bin size for discretization (in same units as boundary coordinates).
    holes : list of Region, optional
        Hole polygons to subtract from the boundary before creating the
        environment. These create excluded areas within the environment.
    walls : mapping of str to LineString, optional
        Named walls inside the boundary. Each is clipped to the boundary, so a
        hand-drawn wall may start on or just past the boundary's edge.
    **from_polygon_kwargs
        Additional arguments passed to ``Environment.from_polygon()``.

    Warns
    -----
    UserWarning
        When clipping removes more than 1% of a wall's length, or all of it.

    Returns
    -------
    Environment
        Environment whose bins cover the boundary polygon.

    Raises
    ------
    ValueError
        If boundary is not a polygon region.

    See Also
    --------
    Environment.from_polygon : Factory method used internally.
    subtract_holes_from_boundary : Used to subtract holes from boundary.

    """
    from neurospatial.environment import Environment

    if boundary.kind != "polygon":
        raise ValueError(f"Boundary must be polygon, got {boundary.kind}")

    # Subtract holes if provided
    if holes:
        boundary = subtract_holes_from_boundary(boundary, holes)

    # Cast is safe because we validated kind=="polygon" above
    polygon = cast("Polygon", boundary.data)
    if walls:
        from neurospatial.arena import OpenArena

        segments, wall_names = _clipped_walls(walls, polygon)
        from_polygon_kwargs["arena"] = OpenArena(
            polygon, walls=segments, wall_names=wall_names
        )
    if boundary.metadata.get("coord_system") in ("cm", "pixels"):
        from_polygon_kwargs.setdefault("units", boundary.metadata["coord_system"])
    return Environment.from_polygon(
        polygon=polygon,
        bin_size=bin_size,
        **from_polygon_kwargs,
    )


def _clipped_walls(
    walls: Mapping[str, shp.LineString], polygon: Polygon
) -> tuple[list[shp.LineString], list[str]]:
    """Each wall's pieces inside ``polygon``, with the wall's name for each piece.

    Every piece passes the arena's containment test (see
    :func:`geometry._arena_fit.clip_wall`).
    """
    import warnings

    from neurospatial.geometry._arena_fit import clip_wall

    segments: list[shp.LineString] = []
    wall_names: list[str] = []
    for name, wall in walls.items():
        pieces = clip_wall(wall, polygon)
        kept = sum(piece.length for piece in pieces)
        if kept < 0.99 * wall.length:
            warnings.warn(
                f"Wall '{name}' extends past the environment boundary; kept the "
                f"{kept / wall.length:.0%} of it inside."
                if pieces
                else f"Skipping wall '{name}': it lies outside the environment "
                "boundary.",
                UserWarning,
                stacklevel=3,
            )
        segments.extend(pieces)
        wall_names.extend([name] * len(pieces))
    return segments, wall_names


def _validate_drawn_shapes(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    region_types: list[RegionType],
    shape_types: list[str],
    calibration: VideoCalibration | None,
) -> None:
    """Require every interactive drawing to survive conversion without repair."""
    for vertices, name, role, shape_type in zip(
        shapes_data, names, region_types, shape_types, strict=True
    ):
        points = _to_world(vertices, calibration)
        try:
            if not np.isfinite(points).all():
                raise ValueError("coordinates must be finite")
            if shape_type in _AREA_SHAPES or role not in ("wall", "landmark"):
                polygon = shp.Polygon(points)
                if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                    raise ValueError(
                        "invalid polygon: draw an area with positive area and no crossing edges"
                    )
            if role == "wall":
                if shape_type == "ellipse" or len(points) < 2:
                    raise ValueError("draw a wall as a path, line, or polygon")
                line = shp.LineString(points)
                if not line.is_valid or line.length <= 0:
                    raise ValueError(
                        "draw a wall with distinct vertices and positive length"
                    )
            elif role == "landmark":
                _landmark_from_shape(str(name), shape_type, points)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot save {role} '{name}': {exc}. Edit or delete this shape and try again."
            ) from exc
