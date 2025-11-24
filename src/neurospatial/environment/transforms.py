"""Transform methods for Environment.

This module provides transformation methods that create new Environment
instances with modified structure or data.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv

if TYPE_CHECKING:
    import shapely

    from neurospatial.environment.core import Environment
    from neurospatial.transforms import Affine2D, AffineND

from neurospatial.regions import Regions


class EnvironmentTransforms:
    """Mixin providing environment transformation methods."""

    def rebin(
        self: SelfEnv,
        factor: int | tuple[int, ...],
    ) -> Environment:
        """Coarsen regular grid by integer factor (geometry-only operation).

        Creates a new environment with coarser spatial resolution by reducing
        the number of bins. This method only modifies the grid geometry and
        connectivity; it does not aggregate any field values.

        Only supported for RegularGridLayout environments.

        Parameters
        ----------
        factor : int or tuple of int
            Coarsening factor per dimension. If int, applied uniformly to all
            dimensions. If tuple, must match the number of dimensions.
            Each factor must be a positive integer.

        Returns
        -------
        coarse_env : Environment
            New environment with reduced resolution. The new grid shape is
            ``original_shape // factor`` in each dimension. All bins in the
            coarsened grid are marked as active.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        NotImplementedError
            If environment layout is not RegularGridLayout.
        ValueError
            If factor is not positive or if factor is too large for grid shape.

        See Also
        --------
        smooth : Apply diffusion kernel smoothing to fields.
        subset : Extract spatial subset of environment.
        map_points_to_bins : Map original bin centers to coarsened bins.

        Notes
        -----
        **Geometry only**: This method only coarsens the grid structure. To
        aggregate field values (occupancy, spike counts, etc.) from the original
        grid to the coarsened grid, map the original bin centers and aggregate:

            >>> from neurospatial import map_points_to_bins
            >>> coarse = env.rebin(factor=2)
            >>> coarse_indices = map_points_to_bins(env.bin_centers, coarse)
            >>> coarse_field = np.bincount(
            ...     coarse_indices, weights=field, minlength=coarse.n_bins
            ... )

        **Grid-only operation**: This method only works for environments with
        RegularGridLayout. Other layout types will raise NotImplementedError.

        **Non-divisible dimensions**: If the grid shape is not evenly divisible
        by the factor in any dimension, the grid is truncated to the largest
        multiple of the factor. A warning is issued in this case.

        **Connectivity**: The connectivity graph is rebuilt for the coarsened
        grid with the same connectivity pattern as the original (e.g., if
        original had diagonal connections, coarsened grid will too).

        **Bin centers**: New bin centers are computed from the coarsened grid
        edges as midpoints between edge positions.

        **Active bins**: All bins in the coarsened environment are marked as
        active, even if the original grid had inactive regions.

        **Metadata preservation**: The units and frame attributes are copied
        from the original environment to the coarsened environment.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> # Create 10x10 grid
        >>> data = np.random.rand(1000, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=10.0)
        >>> env.layout.grid_shape
        (10, 10)
        >>>
        >>> # Coarsen by factor 2 → 5x5 grid
        >>> coarse = env.rebin(factor=2)
        >>> coarse.layout.grid_shape
        (5, 5)
        >>>
        >>> # Anisotropic coarsening with tuple
        >>> coarse_aniso = env.rebin(factor=(2, 5))
        >>> coarse_aniso.layout.grid_shape
        (5, 2)
        >>>
        >>> # Aggregate a field to the coarsened grid
        >>> from neurospatial import map_points_to_bins
        >>> occupancy = np.random.rand(env.n_bins) * 100
        >>> coarse_indices = map_points_to_bins(env.bin_centers, coarse)
        >>> coarse_occupancy = np.bincount(
        ...     coarse_indices, weights=occupancy, minlength=coarse.n_bins
        ... )
        >>> # Total time is preserved
        >>> np.isclose(occupancy.sum(), coarse_occupancy.sum())
        True

        """
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        # --- Input validation ---

        # Check layout type using _layout_type_tag (Protocol-safe check)
        if self.layout._layout_type_tag != "RegularGrid":
            raise NotImplementedError(
                "rebin() is only supported for RegularGridLayout. "
                f"Current layout type: {self.layout._layout_type_tag}. "
                "For other layout types, consider using smooth() for field smoothing "
                "or subset() for spatial cropping."
            )

        # Parse factor
        # Cast to RegularGridLayout to access grid-specific attributes
        layout = cast("RegularGridLayout", self.layout)
        grid_shape = layout.grid_shape
        grid_edges = layout.grid_edges

        # Assert non-None (RegularGridLayout always has these)
        assert grid_shape is not None, "RegularGridLayout must have grid_shape"
        assert grid_edges is not None, "RegularGridLayout must have grid_edges"

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
                stacklevel=2,
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

        # Check if original had diagonal connections
        # Sample: check degree of a center node (not on boundary)
        center_node = grid_shape[0] // 2
        if n_dims == 2:
            center_flat_idx = center_node * grid_shape[1] + grid_shape[1] // 2
        else:
            # For higher dims, just check if any node has more than 2*n_dims neighbors
            center_flat_idx = 0

        # Get degree
        if center_flat_idx in self.connectivity:
            degree = self.connectivity.degree(center_flat_idx)
            # 2D: 4-conn has degree 4, 8-conn has degree 8
            # 3D: 6-conn has degree 6, 26-conn has degree 26
            # Heuristic: if degree > 2*n_dims, assume diagonal connections
            connect_diagonal = degree > 2 * n_dims
        else:
            # Default to True (common case)
            connect_diagonal = True

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
        new_layout.dimension_ranges = tuple(
            (edges[0], edges[-1]) for edges in coarse_edges
        )
        new_layout.grid_edges = coarse_edges
        new_layout.grid_shape = coarse_shape
        new_layout.active_mask = active_mask_coarse
        new_layout._layout_type_tag = "RegularGrid"
        new_layout._build_params_used = {
            "bin_size": tuple(
                (edges[-1] - edges[0]) / (len(edges) - 1) for edges in coarse_edges
            ),
            "dimension_ranges": new_layout.dimension_ranges,
            "rebinned_from": f"factor={factor_tuple}",
        }

        # Create new environment - cast to Environment class for proper type checking
        from neurospatial.layout.base import LayoutEngine

        env_cls = cast("type[Environment]", self.__class__)
        coarse_env = env_cls(
            layout=cast("LayoutEngine", new_layout),
            name=f"{self.name}_rebinned" if self.name else "",
            regions=Regions(),  # Start with empty regions
        )
        coarse_env._layout_type_used = "RegularGrid"
        coarse_env._layout_params_used = new_layout._build_params_used
        coarse_env._setup_from_layout()

        # Preserve metadata
        if hasattr(self, "units") and self.units is not None:
            coarse_env.units = self.units
        if hasattr(self, "frame") and self.frame is not None:
            coarse_env.frame = self.frame

        return coarse_env

    def subset(
        self: SelfEnv,
        *,
        bins: NDArray[np.bool_] | None = None,
        region_names: Sequence[str] | None = None,
        polygon: shapely.Polygon | None = None,
        invert: bool = False,
    ) -> Environment:
        """Create new environment containing subset of bins.

        Extracts a subgraph from the environment containing only the selected
        bins. Node indices are renumbered to be contiguous [0, n'-1]. This
        operation drops all regions; users can re-add regions to the subset
        environment if needed.

        Parameters
        ----------
        bins : NDArray[np.bool_], shape (n_bins,), optional
            Boolean mask of bins to keep. True = keep, False = discard.
        region_names : Sequence[str], optional
            Keep bins whose centers lie inside these named regions.
            Regions must exist in self.regions. Only polygon-type regions
            are supported (point-type regions will raise ValueError).
        polygon : shapely.Polygon, optional
            Keep bins whose centers lie inside this polygon. Only works
            for 2D environments.
        invert : bool, default=False
            If True, invert the selection mask (select complement).

        Returns
        -------
        sub_env : Environment
            New environment with selected bins renumbered to [0, n'-1].
            Connectivity is the induced subgraph. All regions are dropped.
            Metadata (units, frame) is preserved.

        Raises
        ------
        ValueError
            If none or multiple selection parameters provided, if mask has
            wrong shape/dtype, if region names don't exist, if selection is empty.

        Notes
        -----
        Exactly one of {bins, region_names, polygon} must be provided.

        The connectivity graph is the induced subgraph: only edges where both
        endpoints are in the selection are kept. This may create disconnected
        components if the selection is not contiguous.

        Node attributes ('pos', 'source_grid_flat_index', 'original_grid_nd_index')
        and edge attributes ('distance', 'vector', 'edge_id', 'angle_2d') are
        preserved from the original graph.

        See Also
        --------
        rebin : Coarsen grid resolution (grid-only).
        components : Find connected components.

        Examples
        --------
        >>> # Extract bins inside 'goal' region
        >>> goal_env = env.subset(region_names=["goal"])
        >>>
        >>> # Crop to polygon
        >>> from shapely.geometry import box
        >>> cropped = env.subset(polygon=box(0, 0, 50, 50))
        >>>
        >>> # Select bins by boolean mask
        >>> mask = env.bin_centers[:, 0] < 50  # Left half
        >>> left_env = env.subset(bins=mask)
        >>>
        >>> # Inverted selection (everything except region)
        >>> outside = env.subset(region_names=["obstacle"], invert=True)

        """
        # --- Input Validation ---

        # Exactly one selection parameter must be provided
        n_params = sum(
            [bins is not None, region_names is not None, polygon is not None]
        )
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
            if bins.shape != (self.n_bins,):
                raise ValueError(
                    f"bins must have shape (n_bins,) = ({self.n_bins},), "
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
                if name not in self.regions:
                    available = list(self.regions.keys())
                    raise ValueError(
                        f"Region '{name}' not found in environment. "
                        f"Available regions: {available}"
                    )

            # Build mask from regions
            mask = np.zeros(self.n_bins, dtype=bool)
            for name in region_names:
                region = self.regions[name]
                if region.kind == "point":
                    raise ValueError(
                        f"Region '{name}' is a point-type region. "
                        "subset() only supports polygon-type regions. "
                        "Use a boolean mask (bins parameter) to select bins containing specific points."
                    )
                elif region.kind == "polygon":
                    # Use vectorized shapely operation for performance
                    from shapely import contains_xy

                    if self.bin_centers.shape[1] != 2:
                        raise ValueError(
                            f"Polygon regions only work for 2D environments. "
                            f"This environment has {self.bin_centers.shape[1]} dimensions. "
                            "Use bins parameter for N-dimensional selection."
                        )

                    # Vectorized containment check (much faster than loop)
                    in_region = contains_xy(
                        region.data, self.bin_centers[:, 0], self.bin_centers[:, 1]
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
                if self.bin_centers.shape[1] != 2:
                    raise ValueError(
                        f"Polygon selection only works for 2D environments. "
                        f"This environment has {self.bin_centers.shape[1]} dimensions."
                    )

                # Vectorized containment check (150x faster than Python loop)
                mask = contains_xy(
                    polygon, self.bin_centers[:, 0], self.bin_centers[:, 1]
                )

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

        # --- Extract Subgraph ---

        # Get selected node indices
        selected_nodes = np.where(mask)[0].tolist()

        # Extract induced subgraph
        subgraph = self.connectivity.subgraph(selected_nodes).copy()

        # --- Renumber Nodes ---

        # Create mapping: old_node_id -> new_node_id
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}

        # Create new graph with renumbered nodes
        import networkx as nx

        new_graph = nx.Graph()

        # Add nodes with renumbered IDs and preserved attributes
        for old_id in selected_nodes:
            new_id = old_to_new[old_id]
            node_attrs = self.connectivity.nodes[old_id].copy()
            new_graph.add_node(new_id, **node_attrs)

        # Add edges with renumbered node IDs and preserved attributes
        for u, v, edge_data in subgraph.edges(data=True):
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            new_graph.add_edge(new_u, new_v, **edge_data)

        # --- Extract Bin Centers ---

        # Extract bin centers for selected bins (in new order)
        new_bin_centers = self.bin_centers[selected_nodes]

        # --- Create New Environment ---

        # Use from_layout factory method with custom layout
        # We need to create a minimal layout object that provides the required interface

        # Create a custom layout that wraps our subset data
        # We'll use a simple object that implements the LayoutEngine protocol
        class SubsetLayout:
            """Minimal layout for subset environment."""

            def __init__(
                self,
                bin_centers: NDArray[np.float64],
                connectivity,
                dimension_ranges: tuple[tuple[float, float], ...],
                build_params: dict,
            ) -> None:
                self.bin_centers = bin_centers
                self.connectivity = connectivity
                self.dimension_ranges = dimension_ranges
                self._layout_type_tag = "subset"
                self._build_params_used = build_params
                self.is_1d = False

            def build(self) -> None:
                pass  # Already built

            def point_to_bin_index(self, point: NDArray[np.float64]) -> int:
                # Use KDTree for nearest neighbor
                from scipy.spatial import cKDTree

                tree = cKDTree(self.bin_centers)
                _, idx = tree.query(point)
                return int(idx)

            def bin_sizes(self) -> NDArray[np.float64]:
                # Estimate from connectivity graph
                # Use edge distances to estimate bin sizes
                sizes = np.ones(len(self.bin_centers))
                for node in self.connectivity.nodes():
                    neighbors = tuple(self.connectivity.neighbors(node))
                    if neighbors:
                        distances = [
                            self.connectivity[node][n]["distance"] for n in neighbors
                        ]
                        sizes[node] = np.mean(distances)
                return sizes

            def plot(self, ax: Any | None = None, **kwargs: Any) -> Any:
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

        # Compute dimension ranges from bin centers
        n_dims = new_bin_centers.shape[1]
        dimension_ranges = tuple(
            (float(new_bin_centers[:, i].min()), float(new_bin_centers[:, i].max()))
            for i in range(n_dims)
        )

        # Create layout
        layout = SubsetLayout(
            bin_centers=new_bin_centers,
            connectivity=new_graph,
            dimension_ranges=dimension_ranges,
            build_params={"source": "subset", "original_n_bins": self.n_bins},
        )

        # Create new environment - directly instantiate
        # (from_layout is for factory pattern with string kind)
        # Cast to Environment class for proper type checking
        from neurospatial.layout.base import LayoutEngine

        env_cls = cast("type[Environment]", self.__class__)
        sub_env = env_cls(
            name="",
            layout=cast(
                "LayoutEngine", layout
            ),  # SubsetLayout implements LayoutEngine protocol
            layout_type_used="subset",
            layout_params_used={"source": "subset", "original_n_bins": self.n_bins},
            regions=Regions(),  # Empty regions as documented
        )

        # --- Preserve Metadata ---

        # Copy units and frame if present
        if hasattr(self, "units") and self.units is not None:
            sub_env.units = self.units
        if hasattr(self, "frame") and self.frame is not None:
            sub_env.frame = self.frame

        # Note: Regions are intentionally dropped (as documented)

        return sub_env

    def apply_transform(
        self: SelfEnv,
        transform: AffineND | Affine2D,
        *,
        name: str | None = None,
    ) -> Environment:
        """Apply affine transformation to environment, returning a new instance.

        Transforms the environment's bin centers, connectivity graph, and regions
        using an affine transformation. Supports both 2D (Affine2D) and N-D
        (AffineND) transforms. The transformation must match the environment's
        dimensionality.

        Parameters
        ----------
        transform : AffineND or Affine2D
            Affine transformation to apply. Must match environment dimensionality:

            - For 2D environments: use Affine2D or AffineND with n_dims=2
            - For 3D environments: use AffineND with n_dims=3
            - For N-D environments: use AffineND with n_dims=N

            Create transforms using factory functions like `translate()`,
            `scale_2d()`, `from_rotation_matrix()`, or compose using `@` operator.
        name : str, optional
            Name for the transformed environment. If None, appends "_transformed"
            to the original name.

        Returns
        -------
        transformed_env : Environment
            New Environment instance with transformed coordinates. The transformation
            is applied to:

            - `bin_centers`: All bin positions are transformed
            - `connectivity`: Node 'pos' attributes updated, edge distances/vectors recomputed
            - `regions`: Point and polygon regions are transformed
            - `metadata`: Units preserved, frame updated with "_transformed" suffix

            All other properties are copied from the source environment.

        Raises
        ------
        RuntimeError
            If environment is not fitted (use factory methods like
            `Environment.from_samples()`).
        ValueError
            If transform dimensionality doesn't match environment dimensionality.

        See Also
        --------
        estimate_transform : Estimate transformation from point correspondences
        neurospatial.transforms.Affine2D : 2D affine transformation class
        neurospatial.transforms.AffineND : N-D affine transformation class
        neurospatial.transforms.translate : Create translation transform
        neurospatial.transforms.scale_2d : Create scaling transform
        neurospatial.transforms.from_rotation_matrix : Create rotation transform

        Notes
        -----
        **Pure function**: This method does not modify the source environment;
        it returns a new Environment instance.

        **Transformation order**: When composing transforms, use the `@` operator.
        `T1 @ T2` applies T2 first, then T1:

            >>> composed = rotation @ translation  # translation first, then rotation
            >>> transformed = env.apply_transform(composed)

        **Edge attributes**: After transformation:

        - `distance`: Recomputed from transformed positions
        - `vector`: Recomputed as displacement between transformed positions
        - `angle_2d`: Recomputed for 2D environments only

        **Dimension ranges**: Bounding box is recomputed by transforming all
        corner points of the original bounding box.

        Examples
        --------
        Translation (2D):

        >>> from neurospatial import Environment
        >>> from neurospatial.transforms import translate
        >>> import numpy as np
        >>> # Create 2D environment
        >>> data = np.random.rand(200, 2) * 100
        >>> env = Environment.from_samples(data, bin_size=5.0, name="session1")
        >>> # Translate by (10, 20) cm
        >>> transform = translate(10, 20)
        >>> env_shifted = env.apply_transform(transform, name="session1_aligned")
        >>> # Bin centers are translated
        >>> np.allclose(env_shifted.bin_centers, env.bin_centers + [10, 20])
        True

        Rotation (2D):

        >>> from neurospatial.transforms import Affine2D
        >>> # 45-degree rotation
        >>> angle = np.pi / 4
        >>> R = np.array(
        ...     [
        ...         [np.cos(angle), -np.sin(angle), 0],
        ...         [np.sin(angle), np.cos(angle), 0],
        ...         [0, 0, 1],
        ...     ]
        ... )
        >>> transform = Affine2D(R)
        >>> env_rotated = env.apply_transform(transform)
        >>> # Distances preserved under rotation
        >>> assert env_rotated.n_bins == env.n_bins

        Composed transforms (scale → rotate → translate):

        >>> from neurospatial.transforms import scale_2d, translate
        >>> # Build transformation pipeline
        >>> T = translate(50, 50) @ Affine2D(R) @ scale_2d(1.2)
        >>> env_aligned = env.apply_transform(T, name="aligned")

        Cross-session alignment using landmarks:

        >>> from neurospatial.transforms import estimate_transform
        >>> # Session 1 landmarks (e.g., arena corners)
        >>> landmarks_s1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> # Session 2 landmarks (same physical locations, different coordinates)
        >>> landmarks_s2 = np.array([[5, 10], [95, 15], [90, 105], [0, 100]])
        >>> # Estimate rigid transform (rotation + translation)
        >>> T = estimate_transform(landmarks_s1, landmarks_s2, kind="rigid")
        >>> # Transform session 1 environment to session 2 coordinates
        >>> env_s1_aligned = env_s1.apply_transform(T, name="session1_in_s2_coords")

        3D transformation:

        >>> from scipy.spatial.transform import Rotation
        >>> from neurospatial.transforms import from_rotation_matrix, translate_3d
        >>> # Create 3D environment
        >>> data_3d = np.random.randn(500, 3) * 20
        >>> env_3d = Environment.from_samples(data_3d, bin_size=3.0)
        >>> # Rotate 45 degrees around z-axis and translate
        >>> R_3d = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        >>> rotation = from_rotation_matrix(R_3d)
        >>> translation = translate_3d(10, 20, 30)
        >>> T_3d = translation @ rotation
        >>> env_3d_transformed = env_3d.apply_transform(T_3d)

        With regions (regions are automatically transformed):

        >>> env.regions.add("goal", point=[80, 90])
        >>> env_transformed = env.apply_transform(translate(10, 10))
        >>> # Region is transformed along with environment
        >>> goal_region = env_transformed.regions["goal"]
        >>> np.allclose(goal_region.data, [90, 100])
        True

        """
        # Import here to avoid circular dependency
        # (transforms.py imports Environment, which imports this mixin)
        # Delegate to free function
        # Cast self to Environment for type checking
        from neurospatial.transforms import (
            apply_transform_to_environment as _apply_transform_impl,
        )

        env = cast("Environment", self)

        return _apply_transform_impl(env, transform, name=name)
