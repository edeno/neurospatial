"""Cartesian re-binning and selection algorithms for owned environment data."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    import shapely

    from neurospatial.environment._protocols import RebinConsumer, SubsetConsumer
    from neurospatial.environment.core import Environment


def _rebin(
    env: RebinConsumer,
    factor: int | tuple[int, ...],
) -> Environment:
    """Coarsen a regular grid by an integer factor."""
    from neurospatial.layout.engines.regular_grid import RegularGridLayout

    # --- Input validation ---

    grid = env._binning.rectilinear
    if (
        grid is None
        or any(
            np.max(np.abs(np.diff(edge) - np.diff(edge)[0]))
            > 1e-7 * np.diff(edge)[0] + 4 * np.spacing(np.max(np.abs(edge)))
            for edge in grid.edges
        )
        or not np.allclose(
            env._binning.bins.sizes,
            np.prod([edge[1] - edge[0] for edge in grid.edges]),
            rtol=64 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise NotImplementedError(
            "rebin() requires a uniform rectilinear grid; descriptive layout "
            "names cannot establish cell geometry. Supply uniform grid data "
            "or use subset() for spatial cropping."
        )

    # Parse factor
    # Cast to RegularGridLayout to access grid-specific attributes
    grid_shape = grid.shape
    grid_edges = grid.edges

    # Validate grid attributes (RegularGridLayout always has these)
    if grid_shape is None:
        raise RuntimeError(
            "RegularGridLayout missing grid_shape. "
            "This is an internal error - please report this bug."
        )
    if grid_edges is None:
        raise RuntimeError(
            "RegularGridLayout missing grid_edges. "
            "This is an internal error - please report this bug."
        )

    n_dims = len(grid_shape)

    factor_tuple = (factor,) * n_dims if isinstance(factor, int) else tuple(factor)

    # Validate factor dimensions
    if len(factor_tuple) != n_dims:
        raise ValueError(
            f"factor has {len(factor_tuple)} elements but environment has "
            f"{n_dims} dimensions. factor must be int or tuple matching "
            "environment dimensionality."
        )

    # Validate factor values
    for i, f in enumerate(factor_tuple):
        if not isinstance(f, (int, np.integer)):
            raise ValueError(
                f"factor[{i}] = {f} must be an integer, got {type(f).__name__}"
            )
        if f <= 0:
            raise ValueError(
                f"factor[{i}] = {f} must be positive. "
                "Coarsening factor must be at least 1."
            )
        if f > grid_shape[i]:
            raise ValueError(
                f"factor[{i}] = {f} is too large for grid shape {grid_shape}. "
                f"Dimension {i} has only {grid_shape[i]} bins."
            )

    # Check for non-divisible dimensions
    truncated_shape = tuple(
        s // f * f for s, f in zip(grid_shape, factor_tuple, strict=True)
    )
    if truncated_shape != grid_shape:
        warnings.warn(
            f"Grid shape {grid_shape} is not evenly divisible by factor "
            f"{factor_tuple}. Grid will be truncated to {truncated_shape} "
            f"before coarsening.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    # --- Compute new grid parameters ---

    # Truncate grid edges if needed
    # grid_edges was already asserted non-None above but mypy needs reminder
    assert grid_edges is not None
    truncated_edges = []
    for edges, trunc_size in zip(grid_edges, truncated_shape, strict=True):
        # Keep edges up to truncated_size + 1 (edges define bins)
        truncated_edges.append(edges[: trunc_size + 1])

    # Compute new coarsened edges
    coarse_edges = tuple(
        edges[::f] for edges, f in zip(truncated_edges, factor_tuple, strict=True)
    )

    # New grid shape
    coarse_shape = tuple(len(edges) - 1 for edges in coarse_edges)

    # --- Compute new bin centers from coarsened edges ---

    # For each coarse bin, compute center from the coarse grid edges
    # This avoids issues with active/inactive bins from the original grid

    # Create bin centers from coarse edges using meshgrid
    coarse_grid_centers = []
    for edges in coarse_edges:
        # Bin centers are midpoints between edges
        centers = (edges[:-1] + edges[1:]) / 2
        coarse_grid_centers.append(centers)

    # Create meshgrid of bin centers
    if n_dims == 1:
        center_coords = [coarse_grid_centers[0]]
    else:
        center_grids = np.meshgrid(*coarse_grid_centers, indexing="ij")
        center_coords = [grid.ravel() for grid in center_grids]

    # Stack into (n_bins, n_dims)
    coarse_bin_centers = np.column_stack(center_coords)

    # Finalize active cells before constructing the graph or publishing an
    # environment. The original arena remains the continuous authority.
    keep = np.ones(len(coarse_bin_centers), dtype=bool)
    if env.arena.domain_known:
        domain = env.arena.polygonal_domain()
        if domain is not None:
            from shapely import contains_xy

            keep = contains_xy(
                domain, coarse_bin_centers[:, 0], coarse_bin_centers[:, 1]
            )
        else:
            keep = env.arena.contains(coarse_bin_centers)
        if not np.any(keep):
            raise ValueError(
                "Rebinning retains no coarse bin center inside the original arena. "
                "Use a smaller factor or rebuild a grid with centers in the domain."
            )

    # --- Build new connectivity graph ---

    # A diagonal changes more than one coordinate. Inspect actual edges:
    # small or sparsely active grids need not have any full-degree nodes.
    connect_diagonal = any(
        np.count_nonzero(env.bin_centers[u] != env.bin_centers[v]) > 1
        for u, v in env.connectivity.edges()
    )

    # Create new layout
    from neurospatial.layout.helpers.regular_grid import (
        _create_regular_grid_connectivity_graph,
    )

    # Build connectivity only between eligible coarse cells.
    active_mask_coarse = keep.reshape(coarse_shape)

    coarse_connectivity = _create_regular_grid_connectivity_graph(
        full_grid_bin_centers=coarse_bin_centers,
        active_mask_nd=active_mask_coarse,
        grid_shape=coarse_shape,
        connect_diagonal=connect_diagonal,
    )

    # --- Create new Environment ---

    # Create new layout instance
    new_layout = RegularGridLayout()
    new_layout._grid_spec = env._binning.grid_spec
    new_layout.bin_centers = coarse_bin_centers[keep]
    new_layout.connectivity = coarse_connectivity
    new_layout.dimension_ranges = tuple((edges[0], edges[-1]) for edges in coarse_edges)
    new_layout.grid_edges = coarse_edges
    new_layout.grid_shape = coarse_shape
    new_layout.active_mask = active_mask_coarse
    new_layout._layout_type_tag = "RegularGrid"
    new_layout._build_params_used = {
        "bin_size": tuple(
            (edges[-1] - edges[0]) / (len(edges) - 1) for edges in coarse_edges
        ),
        "dimension_ranges": new_layout.dimension_ranges,
        "connect_diagonal_neighbors": connect_diagonal,
    }

    # Create new environment - cast to Environment class for proper type checking
    from neurospatial.layout.base import LayoutEngine

    env_cls = cast("type[Environment]", env.__class__)
    coarse_env = env_cls._from_built_layout(
        cast("LayoutEngine", new_layout),
        name=f"{env.name}_rebinned" if env.name else "",
        layout_type_used="RegularGrid",
        layout_params_used=new_layout._build_params_used,
        regions=deepcopy(env.regions),
        arena=env.arena,
        units=env.units,
        frame=env.frame,
    )

    return coarse_env


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
    from neurospatial.layout._components import HexagonShapes, TriangleShapes
    from neurospatial.layout._derive import subset_binning

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
            from neurospatial.layout._polygon import PolygonCells

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
