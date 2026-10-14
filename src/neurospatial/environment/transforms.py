"""Transform implementations for Cartesian environments.

Private module functions create environments with modified spatial structure.
The scientist-facing entry points belong to ``CartesianCoordinates``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    import shapely

    from neurospatial.environment._protocols import RebinConsumer, SubsetConsumer
    from neurospatial.environment.core import Environment

from neurospatial.regions import Regions


def _rebin(
    env: RebinConsumer,
    factor: int | tuple[int, ...],
) -> Environment:
    """Coarsen a regular grid by an integer factor."""
    from neurospatial.layout.engines.regular_grid import RegularGridLayout

    # --- Input validation ---

    # Check layout type using _layout_type_tag (Protocol-safe check)
    if env.layout._layout_type_tag != "RegularGrid":
        raise NotImplementedError(
            "rebin() is only supported for RegularGridLayout. "
            f"Current layout type: {env.layout._layout_type_tag}. "
            "For other layout types, consider using smooth() for field smoothing "
            "or subset() for spatial cropping."
        )

    # Parse factor
    # Cast to RegularGridLayout to access grid-specific attributes
    layout = cast("RegularGridLayout", env.layout)
    grid_shape = layout.grid_shape
    grid_edges = layout.grid_edges

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

    # --- Build new connectivity graph ---

    # A diagonal changes more than one coordinate. Inspect actual edges:
    # small or sparsely active grids need not have any full-degree nodes.
    connect_diagonal = any(
        np.count_nonzero(layout.bin_centers[u] != layout.bin_centers[v]) > 1
        for u, v in env.connectivity.edges()
    )

    # Create new layout
    from neurospatial.layout.helpers.regular_grid import (
        _create_regular_grid_connectivity_graph,
    )

    # Build connectivity for the coarsened grid (all active)
    active_mask_coarse = np.ones(coarse_shape, dtype=bool)

    coarse_connectivity = _create_regular_grid_connectivity_graph(
        full_grid_bin_centers=coarse_bin_centers,
        active_mask_nd=active_mask_coarse,
        grid_shape=coarse_shape,
        connect_diagonal=connect_diagonal,
    )

    # --- Create new Environment ---

    # Create new layout instance
    new_layout = RegularGridLayout()
    new_layout.bin_centers = coarse_bin_centers
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
    coarse_env = env_cls(
        layout=cast("LayoutEngine", new_layout),
        layout_type_used="RegularGrid",
        layout_params_used=new_layout._build_params_used,
        name=f"{env.name}_rebinned" if env.name else "",
        regions=Regions(),  # Start with empty regions
    )

    # Preserve metadata
    if hasattr(env, "units") and env.units is not None:
        coarse_env.units = env.units
    if hasattr(env, "frame") and env.frame is not None:
        coarse_env.frame = env.frame

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
