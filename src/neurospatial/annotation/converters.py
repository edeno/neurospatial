"""Convert between napari shapes and neurospatial Regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import shapely.geometry as shp
from shapely import Polygon

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from neurospatial import Environment
    from neurospatial.regions import Region, Regions
    from neurospatial.transforms import VideoCalibration


def shapes_to_regions(
    shapes_data: list[NDArray[np.float64]],
    names: list[str],
    roles: list[str],
    calibration: VideoCalibration | None = None,
    simplify_tolerance: float | None = None,
) -> tuple[Regions, Region | None]:
    """
    Convert napari polygon shapes to Regions.

    Parameters
    ----------
    shapes_data : list of NDArray
        List of polygon vertices from napari Shapes layer. Each array has
        shape (n_vertices, 2) in napari (row, col) order.
    names : list of str
        Name for each shape.
    roles : list of str
        Role for each shape: "environment" or "region".
    calibration : VideoCalibration, optional
        If provided, transforms pixel coordinates to world coordinates (cm)
        using ``calibration.transform_px_to_cm``.
    simplify_tolerance : float, optional
        If provided, simplifies polygons using Shapely's Douglas-Peucker
        algorithm. Tolerance is in output coordinate units (cm if calibration
        provided, else pixels). Recommended: 1.0 for cleaner boundaries.

    Returns
    -------
    regions : Regions
        All regions with role="region".
    env_boundary : Region or None
        The region with role="environment", if any.

    Notes
    -----
    Napari shapes use (row, col) order. This function converts to (x, y)
    pixel coordinates before applying calibration.
    """
    import warnings

    from neurospatial.regions import Region, Regions

    regions_list: list[Region] = []
    env_boundary: Region | None = None
    env_boundary_count = 0

    for poly_rc, name, role in zip(shapes_data, names, roles, strict=True):
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

        # Optional simplification (Douglas-Peucker algorithm)
        if simplify_tolerance is not None:
            poly = poly.simplify(tolerance=simplify_tolerance, preserve_topology=True)

        metadata = {
            "source": "napari_annotation",
            "coord_system": coord_system,
            "role": role,
        }

        region = Region(
            name=str(name),
            kind="polygon",
            data=poly,
            metadata=metadata,
        )

        if role == "environment":
            env_boundary_count += 1
            env_boundary = region
        else:
            regions_list.append(region)

    # Warn if multiple environment boundaries were drawn (only last one used)
    if env_boundary_count > 1:
        warnings.warn(
            f"Multiple environment boundaries ({env_boundary_count}) were drawn. "
            f"Only the last one ('{env_boundary.name if env_boundary else 'unknown'}') will be used.",
            UserWarning,
            stacklevel=2,
        )

    return Regions(regions_list), env_boundary


def env_from_boundary_region(
    boundary: Region,
    bin_size: float,
    **from_polygon_kwargs,
) -> Environment:
    """
    Create Environment from an annotated boundary polygon.

    Parameters
    ----------
    boundary : Region
        Region with kind="polygon" defining the environment boundary.
    bin_size : float
        Bin size for discretization (in same units as boundary coordinates).
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
    """
    from neurospatial import Environment

    if boundary.kind != "polygon":
        raise ValueError(f"Boundary must be polygon, got {boundary.kind}")

    # Cast is safe because we validated kind=="polygon" above
    return Environment.from_polygon(
        polygon=cast("Polygon", boundary.data),
        bin_size=bin_size,
        **from_polygon_kwargs,
    )
