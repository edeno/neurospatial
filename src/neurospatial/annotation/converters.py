"""Convert between napari shapes and neurospatial Regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import shapely.geometry as shp
from shapely import Polygon

from neurospatial.annotation._types import MultipleBoundaryStrategy, RegionType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial import Environment
    from neurospatial.ops.transforms import VideoCalibration
    from neurospatial.regions import Region, Regions


def shapes_to_regions(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    region_types: list[RegionType],
    calibration: VideoCalibration | None = None,
    simplify_tolerance: float | None = None,
    *,
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
        Whether to validate polygon geometry and emit warnings for issues
        like self-intersecting polygons or very small areas.
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
        If ``multiple_boundaries="error"`` and multiple environment boundaries
        are drawn.

    Notes
    -----
    Napari shapes use (row, col) order. This function converts to (x, y)
    pixel coordinates before applying calibration.

    Holes are only meaningful when an environment boundary exists. They are
    used to create excluded areas within the environment.

    """
    import warnings

    from neurospatial.regions import Region, Regions

    regions_list: list[Region] = []
    holes_list: list[Region] = []
    env_boundaries: list[Region] = []

    for poly_rc, name, region_type in zip(
        shapes_data,
        names,
        region_types,
        strict=True,
    ):
        # Convert napari (row, col) to video (x, y) pixels
        pts_px = poly_rc[:, ::-1].astype(np.float64)

        # Apply calibration if available
        if calibration is not None:
            pts_world = calibration.transform_px_to_cm(pts_px)
            coord_system = "cm"
        else:
            pts_world = pts_px
            coord_system = "pixels"

        # Skip invalid polygons
        if len(pts_world) < 3:
            continue

        poly = shp.Polygon(pts_world)

        # Validate polygon geometry before simplification
        if validate:
            from neurospatial.annotation.validation import validate_polygon_geometry

            validate_polygon_geometry(poly, str(name), min_area=min_area)

        # Optional simplification (Douglas-Peucker algorithm)
        if simplify_tolerance is not None:
            poly = poly.simplify(tolerance=simplify_tolerance, preserve_topology=True)
            # Ensure polygon is still valid after simplification
            if not poly.is_valid:
                poly = poly.buffer(0)  # Standard fix for self-intersections

        metadata = {
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
        result_poly = result_poly.buffer(0)

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
    **from_polygon_kwargs
        Additional arguments passed to ``Environment.from_polygon()``.

    Returns
    -------
    Environment
        Discretized environment fitted to the boundary polygon.

    Raises
    ------
    ValueError
        If boundary is not a polygon region.

    See Also
    --------
    Environment.from_polygon : Factory method used internally.
    subtract_holes_from_boundary : Used to subtract holes from boundary.

    """
    from neurospatial import Environment

    if boundary.kind != "polygon":
        raise ValueError(f"Boundary must be polygon, got {boundary.kind}")

    # Subtract holes if provided
    if holes:
        boundary = subtract_holes_from_boundary(boundary, holes)

    # Cast is safe because we validated kind=="polygon" above
    return Environment.from_polygon(
        polygon=cast("Polygon", boundary.data),
        bin_size=bin_size,
        **from_polygon_kwargs,
    )
