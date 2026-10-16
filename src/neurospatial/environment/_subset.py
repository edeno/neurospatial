"""Select owned bins while preserving their existing navigation geometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import shapely

    from neurospatial.environment._protocols import SubsetConsumer
    from neurospatial.environment.core import Environment


def _subset_mask(
    env: SubsetConsumer,
    *,
    bins: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Validate a spatial selection and build its active-bin mask."""
    # --- Input Validation ---

    # Exactly one selection parameter must be provided
    n_params = sum([bins is not None, region_names is not None, polygon is not None])
    if n_params == 0:
        raise ValueError(
            "Exactly one of {bins, region_names, polygon} must be provided."
        )
    if n_params > 1:
        raise ValueError(
            "Exactly one of {bins, region_names, polygon} must be provided. "
            f"Got {n_params} parameters."
        )

    # --- Build Selection Mask ---

    if bins is not None:
        # Validate bins parameter
        bins = np.asarray(bins)

        # Check dtype
        if bins.dtype != bool:
            raise ValueError(
                f"bins must be boolean array (dtype=bool), got dtype={bins.dtype}"
            )

        # Check shape
        if bins.shape != (env.n_bins,):
            raise ValueError(
                f"bins must have shape (n_bins,) = ({env.n_bins},), "
                f"got shape {bins.shape}"
            )

        mask = bins

    elif region_names is not None:
        # Validate region_names parameter
        if not isinstance(region_names, (list, tuple)):
            raise ValueError(
                f"region_names must be a list or tuple, got {type(region_names)}"
            )

        if len(region_names) == 0:
            raise ValueError("region_names cannot be empty")

        # Check all regions exist
        for name in region_names:
            if name not in env.regions:
                available = list(env.regions.keys())
                raise ValueError(
                    f"Region '{name}' not found in environment. "
                    f"Available regions: {available}"
                )

        # Build mask from regions
        mask = np.zeros(env.n_bins, dtype=bool)
        for name in region_names:
            region = env.regions[name]
            if region.kind == "point":
                raise ValueError(
                    f"Region '{name}' is a point-type region. "
                    "subset() only supports polygon-type regions. "
                    "Use a boolean mask (bins parameter) to select bins containing specific points."
                )
            elif region.kind == "polygon":
                # Use vectorized shapely operation for performance
                from shapely import contains_xy

                if env.bin_centers.shape[1] != 2:
                    raise ValueError(
                        f"Polygon regions only work for 2D environments. "
                        f"This environment has {env.bin_centers.shape[1]} dimensions. "
                        "Use bins parameter for N-dimensional selection."
                    )

                # Vectorized containment check (much faster than loop)
                in_region = contains_xy(
                    region.data, env.bin_centers[:, 0], env.bin_centers[:, 1]
                )
                mask |= in_region

    elif polygon is not None:
        # Validate polygon parameter
        try:
            import shapely.geometry.base
            from shapely import contains_xy

            # Type check
            if not isinstance(polygon, shapely.geometry.base.BaseGeometry):
                raise TypeError(
                    f"polygon must be a Shapely geometry object, got {type(polygon)}"
                )

            # Dimension check
            if env.bin_centers.shape[1] != 2:
                raise ValueError(
                    f"Polygon selection only works for 2D environments. "
                    f"This environment has {env.bin_centers.shape[1]} dimensions."
                )

            # Vectorized containment check (150x faster than Python loop)
            mask = contains_xy(polygon, env.bin_centers[:, 0], env.bin_centers[:, 1])

        except (AttributeError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid polygon: {e}") from e

    else:
        # Should never reach here due to earlier validation
        raise RuntimeError("No selection method specified (should be unreachable)")

    # Apply invert if requested
    if invert:
        mask = ~mask

    # Check that selection is not empty
    if not np.any(mask):
        raise ValueError(
            f"No bins selected. Selection resulted in empty mask. (invert={invert})"
        )

    return mask


def _subset(
    env: SubsetConsumer,
    *,
    bins: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> Environment:
    """Derive selected bins and their graph, bounds and preserved metadata."""
    mask = _subset_mask(
        env, bins=bins, region_names=region_names, polygon=polygon, invert=invert
    )
    from neurospatial.geometry._components import HexagonShapes, TriangleShapes
    from neurospatial.geometry._derive import subset_binning

    binning = subset_binning(env._binning, mask)
    selected = np.flatnonzero(mask)
    child_of = {int(old): new for new, old in enumerate(selected)}
    graph = nx.Graph()
    graph.graph.update(env.connectivity.graph)
    # Retain the parent's node iteration order as well as edge order and
    # orientation; edge fields index that order even on loaded graphs.
    for old, data in env.connectivity.nodes(data=True):
        if old in child_of:
            graph.add_node(child_of[old], **data)
    for index, (u, v) in enumerate(binning.bins.pairs):
        old_u, old_v = selected[u], selected[v]
        data = dict(env.connectivity.edges[old_u, old_v])
        if binning.rectilinear is not None:
            data["edge_id"] = index
        graph.add_edge(int(u), int(v), **data)
    params = {"source": "subset", "original_n_bins": env.n_bins}
    tag = "MaskedGrid" if binning.rectilinear is not None else "subset"
    if binning.rectilinear is not None:
        ranges = [
            (float(edge[0]), float(edge[-1])) for edge in binning.rectilinear.edges
        ]
    else:
        shapes = binning.shapes
        vertices = binning.bins.centers
        if binning.grid_frame is not None:
            vertices = binning.grid_frame.corners()
        elif isinstance(shapes, HexagonShapes):
            angles = shapes.orientation + np.pi / 2 + np.arange(6) * np.pi / 3
            offsets = shapes.radius * np.column_stack((np.cos(angles), np.sin(angles)))
            vertices = (shapes.centers[:, None, :] + offsets).reshape(-1, 2)
        elif isinstance(shapes, TriangleShapes):
            vertices = shapes.points[shapes.triangles].reshape(-1, 2)
        else:
            from neurospatial.geometry._polygon import PolygonCells

            if isinstance(shapes, PolygonCells):
                import shapely

                vertices = np.concatenate(
                    [shapely.get_coordinates(cell) for cell in shapes.polygons]
                )
        ranges = []
        for axis in vertices.T:
            low, high = float(axis.min()), float(axis.max())
            margin = 0.05 * (high - low) if shapes is not None else 0.0
            ranges.append((low - margin, high + margin))
    env_cls = cast("type[Environment]", type(env))
    result = env_cls._from_binning(
        binning,
        connectivity=graph,
        dimension_ranges=ranges,
        layout_type_tag=tag,
        layout_type="mask" if binning.rectilinear is not None else "subset",
        build_params=params,
        is_linearized_track=env.is_linearized_track,
        arena=env.arena,
        units=env.units,
        frame=env.frame,
    )
    return result
