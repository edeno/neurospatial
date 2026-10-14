"""Transform implementations for Cartesian environments.

Private module functions create environments with modified spatial structure.
The scientist-facing entry points belong to ``CartesianCoordinates``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

if TYPE_CHECKING:
    import networkx as nx
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


def _grid_subset(env: SubsetConsumer, mask: NDArray[np.bool_]) -> Environment:
    """Build a serializable grid from selected active bins, keeping their edges.

    The selected cells are rebuilt as a masked grid, so the subset saves and
    loads like any grid. Its graph keeps exactly the parent's edges among the
    selected bins, as saved graphs do on read-back: a parent whose graph is not
    plain mask adjacency (a 4-connected grid read from NWB, a wall in a saved
    graph) keeps that connectivity.
    """
    from neurospatial.environment.factories import _owned_copy

    # Round-trip safe path. env.active_mask is the original
    # grid-shape boolean mask; its True positions in row-major
    # order correspond 1:1 to active bin indices [0, n_bins).
    assert env.active_mask is not None  # for type checker
    assert env.grid_edges is not None
    flat_active_indices = np.flatnonzero(env.active_mask.ravel())
    selected_flat = flat_active_indices[mask]
    new_mask_flat = np.zeros(env.active_mask.size, dtype=bool)
    new_mask_flat[selected_flat] = True
    new_active_mask = new_mask_flat.reshape(env.active_mask.shape)

    # Preserve the parent's diagonal-neighbor setting if it had
    # one; default to True (matches Environment.from_grid_mask).
    parent_params = env._layout_params_used or {}
    connect_diagonal = bool(parent_params.get("connect_diagonal_neighbors", True))

    layout_params = _owned_copy(
        {
            "active_mask": new_active_mask,
            "grid_edges": env.grid_edges,
            "connect_diagonal_neighbors": connect_diagonal,
        }
    )
    env_cls = cast("type[Environment]", env.__class__)
    layout = env_cls._create_layout("MaskedGrid", layout_params)
    layout.connectivity = _restrict_edges(
        env.connectivity, np.flatnonzero(mask), layout.connectivity
    )
    sub_env = env_cls("", layout, "MaskedGrid", layout_params)
    # Preserve metadata (regions are still dropped per the
    # documented contract).
    if env.units is not None:
        sub_env.units = env.units
    if env.frame is not None:
        sub_env.frame = env.frame
    return sub_env


def _restrict_edges(
    parent: nx.Graph, selected: NDArray[np.intp], rebuilt: nx.Graph
) -> nx.Graph:
    """Join the rebuilt grid's nodes by exactly the parent's edges among them.

    Parameters
    ----------
    parent : networkx.Graph
        The parent environment's connectivity, keyed by its active-bin ids.
    selected : NDArray[np.intp], shape (n_selected,)
        Parent bin ids of the subset's bins, ascending; ``selected[i]`` is
        subset bin ``i``.
    rebuilt : networkx.Graph
        The graph the masked-grid layout built for the selected cells, whose
        node attributes index the shared grid.

    Returns
    -------
    networkx.Graph
        The rebuilt nodes, with the parent's edges and edge attributes. Edges
        the rebuilt graph also has keep its order; any others follow, and
        ``edge_id`` numbers them in that order.
    """
    import networkx as nx

    new_id = {int(old): new for new, old in enumerate(selected)}
    kept = {
        frozenset((new_id[u], new_id[v])): data
        for u, v, data in parent.subgraph(new_id).edges(data=True)
    }
    order = [(u, v) for u, v in rebuilt.edges() if frozenset((u, v)) in kept]
    order += [(min(edge), max(edge)) for edge in kept if not rebuilt.has_edge(*edge)]
    graph = nx.Graph()
    graph.add_nodes_from(rebuilt.nodes(data=True))
    for edge_id, (u, v) in enumerate(order):
        graph.add_edge(u, v, **{**kept[frozenset((u, v))], "edge_id": edge_id})
    return graph


class _SubsetLayout:
    """Minimal layout for subsets of environments without a rectangular grid.

    Graph (track), hexagonal, triangular-mesh and NWB-reconstructed
    environments, and subsets of subsets, are subset onto this layout; grids of
    rectangular cells are rebuilt with ``from_grid_mask`` instead.
    """

    def __init__(
        self,
        bin_centers: NDArray[np.float64],
        connectivity,
        dimension_ranges: tuple[tuple[float, float], ...],
        build_params: dict,
        is_linearized_track: bool = False,
    ) -> None:
        self.bin_centers = bin_centers
        self.connectivity = connectivity
        self.dimension_ranges = dimension_ranges
        self._layout_type_tag = "subset"
        self._build_params_used = build_params
        # Preserve the parent layout's is_linearized_track so a subset of a
        # graph (linearized) env stays 1-D in 2-D space rather
        # than masquerading as a fully N-D layout.
        self.is_linearized_track = is_linearized_track
        # Lazy KDTree + nearest-neighbor-distance cache. The
        # KDTree was previously rebuilt on every point_to_bin_index
        # call, which is O(n log n) per query and dominated cost
        # for trajectory binning. A cached subset env now spends
        # ~one KDTree construction per layout lifetime.
        self._kdtree: Any = None
        self._oof_threshold: float | None = None

    def _ensure_kdtree(self) -> None:
        """Build the KDTree (and out-of-env distance threshold) once."""
        if self._kdtree is not None:
            return
        from scipy.spatial import cKDTree

        tree = cKDTree(self.bin_centers)
        self._kdtree = tree
        # Pre-compute the out-of-env threshold from the median
        # nearest-neighbor distance. For a 1-bin layout we leave
        # it as None and short-circuit thresholding below.
        if len(self.bin_centers) > 1:
            nearest_neighbor_dist = tree.query(self.bin_centers, k=2, workers=-1)[0][
                :, 1
            ]
            typical_bin_spacing = float(np.median(nearest_neighbor_dist))
            self._oof_threshold = 10.0 * typical_bin_spacing

    def build(self) -> None:
        """Build the layout (no-op for subset layouts)."""
        pass  # Already built

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Find the nearest bin index for each query point.

        Matches the standard layout contract: accepts a batch of
        points shaped ``(n_points, n_dims)`` (a single 1-D point
        is also tolerated for backwards compatibility) and
        returns an array of bin indices shaped ``(n_points,)``.
        Points farther than ``10 * typical_bin_spacing`` from
        the nearest bin center are returned as ``-1`` so
        trajectory callers (``Environment.bin_at`` ->
        ``Environment.occupancy``, ``Environment.bin_sequence``)
        consistently flag out-of-env samples. The 10× threshold
        matches the heuristic in ``ops.binning.map_points_to_bins``;
        tightening would spuriously reject points near sparse
        regions of the subset graph.

        Note: grids of rectangular cells take the ``from_grid_mask``
        path and never reach this layout. It serves the layouts that
        path cannot rebuild: graph (1-D linearized) envs, whose 2-D
        embedding ``from_grid_mask`` would silently drop, and
        hexagonal and mesh envs, whose cells are not rectangular. A
        registered serializable subset layout is tracked separately.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        self._ensure_kdtree()
        distances, indices = self._kdtree.query(pts, k=1, workers=-1)

        # Apply the same "10x typical spacing" outside heuristic
        # used by ops.binning.map_points_to_bins so subset envs
        # behave like Cartesian envs do for clearly out-of-env
        # samples. The threshold itself is precomputed once in
        # _ensure_kdtree. We deliberately keep it loose
        # (10x) here because _SubsetLayout has no notion of an
        # active mask -- only a graph -- and tightening would
        # spuriously reject points near sparse regions.
        indices = indices.astype(np.intp)
        if self._oof_threshold is not None:
            indices[distances > self._oof_threshold] = -1
        return cast("NDArray[np.intp]", indices)

    def bin_sizes(self) -> NDArray[np.float64]:
        """
        Approximate per-bin sizes from connectivity edge lengths.

        Each bin's size is estimated as the mean ``distance`` of its
        incident edges. For graph (1-D linearized) subsets this has the
        right *dimension*: a length. Hexagonal and mesh subsets also use
        this layout, and for them the estimate is a length, not the cell
        area the parent environment reports.

        The value remains an approximation: it is the mean spacing to
        neighboring bins, not the exact extent of the bin, and it falls
        back to ``1.0`` for isolated bins (no incident edges). Callers
        needing exact bin extents should not rely on this estimate.

        Returns
        -------
        NDArray[np.float64], shape (n_bins,)
            Approximate length of each bin, estimated from the mean
            length of its incident connectivity edges.
        """
        # Estimate from connectivity graph: mean incident edge length.
        # See the docstring for why a length (not area/volume) estimate
        # is the correct dimension here -- this layout is graph-only.
        sizes = np.ones(len(self.bin_centers))
        for node in self.connectivity.nodes():
            neighbors = tuple(self.connectivity.neighbors(node))
            if neighbors:
                distances = [self.connectivity[node][n]["distance"] for n in neighbors]
                sizes[node] = np.mean(distances)
        return sizes

    def plot(self, ax: Any | None = None, **kwargs: Any) -> Any:
        """
        Plot the layout bin centers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        **kwargs : dict
            Additional keyword arguments passed to scatter().

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        # Plot bin centers
        if self.bin_centers.shape[1] == 2:
            ax.scatter(
                self.bin_centers[:, 0],
                self.bin_centers[:, 1],
                **kwargs,
            )
        return ax


def _graph_subset(env: SubsetConsumer, mask: NDArray[np.bool_]) -> Environment:
    """Build the induced graph on selected active bins."""

    # --- Graph-env fallback: extract induced subgraph ---------------

    # Get selected node indices
    selected_nodes = np.where(mask)[0].tolist()

    # Extract induced subgraph
    subgraph = env.connectivity.subgraph(selected_nodes).copy()

    # --- Renumber Nodes ---

    # Create mapping: old_node_id -> new_node_id
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

    # Create new graph with renumbered nodes
    import networkx as nx

    new_graph = nx.Graph()

    # Add nodes with renumbered IDs and preserved attributes
    for old_id in selected_nodes:
        new_id = old_to_new[old_id]
        node_attrs = env.connectivity.nodes[old_id].copy()
        new_graph.add_node(new_id, **node_attrs)

    # Add edges with renumbered node IDs and preserved attributes
    for u, v, edge_data in subgraph.edges(data=True):
        new_u = old_to_new[u]
        new_v = old_to_new[v]
        new_graph.add_edge(new_u, new_v, **edge_data)

    # --- Extract Bin Centers ---

    # Extract bin centers for selected bins (in new order)
    new_bin_centers = env.bin_centers[selected_nodes]

    # --- Create New Environment ---

    # Use from_layout factory method with custom layout
    # We need to create a minimal layout object that provides the required interface

    # Create a custom layout that wraps our subset data
    # We'll use a simple object that implements the LayoutEngine protocol

    # Compute dimension ranges from bin centers
    n_dims = new_bin_centers.shape[1]
    dimension_ranges = tuple(
        (float(new_bin_centers[:, i].min()), float(new_bin_centers[:, i].max()))
        for i in range(n_dims)
    )

    # Create layout. Preserve the parent's is_linearized_track flag so a graph
    # subset (1-D linearized track in 2-D space) stays 1-D.
    layout = _SubsetLayout(
        bin_centers=new_bin_centers,
        connectivity=new_graph,
        dimension_ranges=dimension_ranges,
        build_params={"source": "subset", "original_n_bins": env.n_bins},
        is_linearized_track=env.is_linearized_track,
    )

    # Create new environment - directly instantiate
    # (from_layout is for factory pattern with string kind)
    # Cast to Environment class for proper type checking
    from neurospatial.layout.base import LayoutEngine

    env_cls = cast("type[Environment]", env.__class__)
    sub_env = env_cls(
        name="",
        layout=cast(
            "LayoutEngine", layout
        ),  # _SubsetLayout implements LayoutEngine protocol
        layout_type_used="subset",
        layout_params_used={"source": "subset", "original_n_bins": env.n_bins},
        regions=Regions(),  # Empty regions as documented
    )

    # --- Preserve Metadata ---

    # Copy units and frame if present. The subset preserves the concrete
    # environment type (Environment or EgocentricPolarEnvironment) because
    # ``env_cls`` above is ``env.__class__``.
    if hasattr(env, "units") and env.units is not None:
        sub_env.units = env.units
    if hasattr(env, "frame") and env.frame is not None:
        sub_env.frame = env.frame

    # Note: Regions are intentionally dropped (as documented)

    return sub_env
