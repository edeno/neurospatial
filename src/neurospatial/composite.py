"""CompositeEnvironment: merges multiple Environment instances into a single unified Environment-like API.
Bridge edges between sub-environments are inferred automatically via mutual-nearest-neighbor (MNN).

This class exposes the same public interface as the base `Environment` class:
  - Properties: n_dims, n_bins, bin_centers, connectivity, is_1d, dimension_ranges,
                grid_edges, grid_shape, active_mask, regions
  - Methods:    bin_at, contains, neighbors, distance_between, bin_center_of,
                bins_in_region, mask_for_region, shortest_path, info,
                save, load, bin_attributes, edge_attributes, plot

(Note: factory methods like from_layout are not included, since CompositeEnvironment
wraps pre-fitted sub-environments. plot_1d is not applicable for composite environments.)
"""

from collections.abc import Sequence
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import KDTree

from neurospatial._constants import KDTREE_COMPOSITE_LEAF_SIZE
from neurospatial._logging import log_composite_build
from neurospatial.environment import Environment
from neurospatial.regions import Region, Regions


class CompositeEnvironment:
    """A composite environment that merges multiple child Environment instances into one.

    It automatically infers "bridge" edges between every pair of sub-environments by finding
    mutually nearest neighbor bin-centers (MNN). It then presents the same interface as
    the base `Environment` class.

    Attributes
    ----------
    environments : List[Environment]
        List of constituent Environment instances that make up the composite.
    name : str
        Name for the composite environment.
    layout : None
        Not applicable for composite environments (set to None).
    bin_centers : NDArray[np.float64]
        Combined bin centers from all sub-environments, shape (n_total_bins, n_dims).
    connectivity : nx.Graph
        Combined connectivity graph with bridge edges between sub-environments.
    bridges : List[Tuple[int, int, Dict[str, Any]]]
        List of bridge edges connecting different sub-environments.
        Each tuple is (source_bin, target_bin, edge_attributes).
    dimension_ranges : Sequence[Tuple[float, float]]
        Combined dimension ranges across all sub-environments.
    grid_edges : Tuple[NDArray[np.float64], ...] | None
        Not applicable for composite environments (set to None).
    grid_shape : Tuple[int, ...] | None
        Not applicable for composite environments (set to None).
    active_mask : NDArray[np.bool_] | None
        Not applicable for composite environments (set to None).
    regions : Regions
        Manages symbolic spatial regions defined within this composite environment.
    is_1d : bool
        True if all sub-environments are 1D, False otherwise.
    _environment_bin_ranges : Dict[str, Tuple[int, int]]
        Mapping of sub-environment names to their bin index ranges in the composite.
    _layout_type_used : str
        Always "Composite" for composite environments.
    _layout_params_used : Dict[str, Any]
        Parameters used to construct the composite.

    """

    is_1d: bool
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    grid_shape: tuple[int, ...] | None
    active_mask: NDArray[np.bool_] | None
    regions: Regions
    _layout_type_used: str
    _layout_params_used: dict[str, Any]
    _n_dims: int

    def __init__(
        self,
        subenvs: list[Environment],
        auto_bridge: bool = True,
        max_mnn_distance: float | None = None,
        use_kdtree_query: bool = True,
    ):
        """Build a CompositeEnvironment from a list of pre-fitted Environment instances.

        Parameters
        ----------
        subenvs : List[Environment]
            A list of fitted Environment objects. All must share the same n_dims.
        auto_bridge : bool, default=True
            If True, automatically infer "bridge edges" between each pair of sub-environments
            using a mutual nearest-neighbor heuristic on their bin_centers.
        max_mnn_distance : Optional[float]
            If provided, any automatically inferred bridge whose Euclidean distance exceeds
            this threshold is discarded. If None, no distance filtering is applied.
        use_kdtree_query : bool, default=True
            If True, use KDTree-based bin_at() for O(M log N) performance. If False,
            use sequential query through each sub-environment (original O(N×M) behavior).

        Raises
        ------
        TypeError
            If subenvs is not a list or tuple, or if any element is not an Environment instance.
        ValueError
            If subenvs is empty, if any environment is not fitted, or if environments
            have different dimensionalities.

        Common Pitfalls
        ---------------
        1. **Dimension mismatch**: All sub-environments must have the same number of
           dimensions (n_dims). Mixing 2D and 3D environments will raise an error.
           Before creating the composite, verify that all environments have the same
           n_dims property (e.g., check env1.n_dims == env2.n_dims). This typically
           occurs when combining data from different recording modalities.

        2. **No bridge edges found**: If auto_bridge=True but the sub-environments
           are very far apart, no bridge edges may be created, leaving the composite
           disconnected. Try increasing max_mnn_distance to allow bridges over longer
           distances, or set auto_bridge=False if you intend to work with disconnected
           components. Use the bridges property to verify that bridge edges were created.

        3. **Overlapping bins**: If sub-environments have bins at the same or very
           similar spatial locations, the composite will have duplicate bins at those
           locations. This can lead to unexpected behavior in spatial queries. Ensure
           that sub-environments represent distinct, non-overlapping spatial regions
           (e.g., different arms of a maze, different rooms). Check bin_centers to
           verify that bin locations are spatially separated.

        """
        # Validate container type
        if not isinstance(subenvs, (list, tuple)):
            raise TypeError(
                f"subenvs must be a list or tuple of Environment instances, "
                f"got {type(subenvs).__name__}. "
                f"Did you pass a single Environment instead of a list? "
                f"Use [env] to wrap it in a list."
            )

        # Validate not empty
        if len(subenvs) == 0:
            raise ValueError(
                "At least one sub-environment is required. Received empty list."
            )

        # Validate each element is Environment instance
        for i, env in enumerate(subenvs):
            if not isinstance(env, Environment):
                raise TypeError(
                    f"subenvs[{i}] must be an Environment instance, "
                    f"got {type(env).__name__}. "
                    f"All elements of subenvs must be Environment objects."
                )

        self._use_kdtree_query = use_kdtree_query

        # Validate that all sub-environments share the same n_dims and are fitted
        self._n_dims = subenvs[0].n_dims
        if not subenvs[0]._is_fitted:
            raise ValueError("Sub-environment 0 is not fitted.")

        for i, e in enumerate(subenvs[1:], 1):
            if not e._is_fitted:
                raise ValueError(f"Sub-environment {i} is not fitted.")
            if e.n_dims != self._n_dims:
                raise ValueError(
                    f"All sub-environments must share the same n_dims. "
                    f"Env 0 has {self._n_dims}, Env {i} has {e.n_dims}.\n"
                    "\n"
                    "Common cause:\n"
                    "  This typically occurs when mixing environments created from data with "
                    "different dimensionalities (e.g., 2D position tracking data and 3D spatial data).\n"
                    "\n"
                    "To fix:\n"
                    "  1. Check that all data_samples arrays used to create environments have the same "
                    "number of columns (n_dims)\n"
                    "  2. Ensure all environments represent the same spatial dimensionality "
                    "(all 2D or all 3D)\n"
                    "  3. Verify each environment's n_dims property before creating the composite"
                )

        # Build index offsets for each sub-environment
        self._subenvs_info = []
        offset = 0
        for env in subenvs:
            n_bins = env.bin_centers.shape[0]
            self._subenvs_info.append(
                {"env": env, "start_idx": offset, "end_idx": offset + n_bins - 1},
            )
            offset += n_bins
        self._total_bins = offset

        # Stack all bin_centers into one array of shape (N_total, n_dims)
        self.bin_centers = np.vstack([env.bin_centers for env in subenvs])

        # Build the composite connectivity graph (nodes only for now)
        self.connectivity = nx.Graph()
        self.connectivity.add_nodes_from(range(self._total_bins))

        # Add each sub-environment’s edges, reindexed by offset
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            for u, v, data in env_i.connectivity.edges(data=True):
                self.connectivity.add_edge(u + base, v + base, **data)

        # Infer MNN-based bridges if requested
        self._bridge_list: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        if auto_bridge:
            self._infer_mnn_bridges(max_mnn_distance)

        # Build KDTree for optimized bin_at() if requested
        self._kdtree: KDTree | None = None
        if self._use_kdtree_query and self.bin_centers.shape[0] > 0:
            self._kdtree = KDTree(
                self.bin_centers, leaf_size=KDTREE_COMPOSITE_LEAF_SIZE
            )

        # Properties to match Environment interface
        self.is_1d = False
        if self.bin_centers.shape[0] > 0:
            min_coords = np.min(self.bin_centers, axis=0)
            max_coords = np.max(self.bin_centers, axis=0)
            self.dimension_ranges = tuple(
                (min_coords[i], max_coords[i]) for i in range(self._n_dims)
            )
        else:
            self.dimension_ranges = (
                tuple(
                    (np.nan, np.nan)
                    for _ in range(self._n_dims)  # Or None, as per Environment
                )
                if self._n_dims > 0
                else None
            )
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        # “all_regions” will hold every Region from every sub‐environment
        all_regions: list[Region] = []
        for child in subenvs:
            # child.regions is itself a Regions (mapping name → Region).
            # We want to pull out each Region object
            for reg in child.regions.values():
                # If you suspect two children might have regions with the same name,
                # you can either rename here (e.g. prefix with child.name) or let
                # Regions(...) raise a KeyError. Below we simply re‐use the original name,
                # assuming no collisions.
                all_regions.append(reg)

        # Now create a single Regions object containing every Region from every child
        self.regions = Regions(all_regions)

        self._layout_type_used = "Composite"
        self._layout_params_used = {
            "num_sub_environments": len(subenvs),
            "auto_bridge": auto_bridge,
            "max_mnn_distance": max_mnn_distance,
            "sub_environment_types": [sub_env.layout_type for sub_env in subenvs],
        }
        self._is_fitted = (
            True  # Composite environment is considered 'fitted' upon construction
        )

        # Log composite environment creation
        n_bridges = len(self.bridges) if hasattr(self, "bridges") else 0
        log_composite_build(
            n_subenvs=len(subenvs),
            total_bins=self._total_bins,
            n_bridges=n_bridges,
        )

    def _add_bridge_edge(
        self,
        i_env: int,
        i_bin: int,
        j_env: int,
        j_bin: int,
        w: float,
    ):
        """Add a bridge edge between bin i_bin of sub-environment i_env and bin j_bin of sub-environment j_env,
        with weight w. Raises ValueError if indices are out-of-range.
        """
        n_sub = len(self._subenvs_info)
        if not (0 <= i_env < n_sub) or not (0 <= j_env < n_sub):
            raise ValueError(f"Invalid sub-environment indices: {i_env}, {j_env}")

        block_i = self._subenvs_info[i_env]
        block_j = self._subenvs_info[j_env]
        max_i = block_i["end_idx"] - block_i["start_idx"]
        max_j = block_j["end_idx"] - block_j["start_idx"]
        if not (0 <= i_bin <= max_i) or not (0 <= j_bin <= max_j):
            raise ValueError(f"Bin index out-of-range for bridge: {i_bin}/{j_bin}")

        source_composite_bin = block_i["start_idx"] + i_bin
        target_composite_bin = block_j["start_idx"] + j_bin
        self.connectivity.add_edge(
            source_composite_bin,
            target_composite_bin,
            distance=w,
            weight=1 / w if w > 0 else np.inf,
        )
        self._bridge_list.append(((i_env, i_bin), (j_env, j_bin), w))

    def _infer_mnn_bridges(self, max_distance: float | None = None):
        """Infer “bridge edges” between every pair of sub-environments using a Mutual Nearest Neighbor (MNN) approach:

        1. For each pair (i, j) with i < j:
           a) Build KDTree_i on env_i.bin_centers
           b) Build KDTree_j on env_j.bin_centers
           c) For each bin center in env_i, find its nearest neighbor in env_j (nn_j_of_i)
           d) For each bin center in env_j, find its nearest neighbor in env_i (nn_i_of_j)
           e) If nn_j_of_i[i_idx] == j_idx and nn_i_of_j[j_idx] == i_idx, they are mutual nearest.
              Record (i_idx, j_idx, distance).
        2. If max_distance is provided, only keep pairs with distance ≤ max_distance.
        3. Add each pair as a bridge edge via `_add_bridge_edge`.
        """
        n_sub = len(self._subenvs_info)
        kdtrees = []
        for block in self._subenvs_info:
            centers = block["env"].bin_centers
            kdtrees.append(KDTree(centers, leaf_size=40))

        for i in range(n_sub):
            block_i = self._subenvs_info[i]
            centers_i = block_i["env"].bin_centers
            tree_i = kdtrees[i]

            for j in range(i + 1, n_sub):
                block_j = self._subenvs_info[j]
                centers_j = block_j["env"].bin_centers
                tree_j = kdtrees[j]

                # For each center in i → nearest in j
                dist_ij, idx_ij = tree_j.query(centers_i, k=1)
                idx_ij = idx_ij[:, 0]
                dist_ij = dist_ij[:, 0]

                # For each center in j → nearest in i
                dist_ji, idx_ji = tree_i.query(centers_j, k=1)
                idx_ji = idx_ji[:, 0]
                dist_ji = dist_ji[:, 0]

                for i_idx, j_idx in enumerate(idx_ij):
                    if idx_ji[j_idx] == i_idx:
                        bridge_distance = dist_ij[i_idx]
                        if (max_distance is not None) and (
                            bridge_distance > max_distance
                        ):
                            continue
                        self._add_bridge_edge(
                            i, i_idx, j, j_idx, float(bridge_distance)
                        )

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (same as each sub-environment).

        Returns
        -------
        int
            Number of spatial dimensions.

        """
        return self._n_dims

    @property
    def n_bins(self) -> int:
        """Total number of active bins in the composite environment.

        Returns
        -------
        int
            Total number of bins across all sub-environments.

        """
        return self._total_bins

    @property
    def layout_type(self) -> str:
        """Returns the layout type, which is 'Composite'."""
        return self._layout_type_used

    @property
    def layout_parameters(self) -> dict[str, Any]:
        """Returns parameters used to construct the CompositeEnvironment."""
        return self._layout_params_used

    def bin_at(self, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map points to composite bin indices.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (M, n_dims)
            Array of M points in n_dims-dimensional space.

        Returns
        -------
        NDArray[np.int_], shape (M,)
            Composite bin indices for each point. Returns -1 for points
            outside all sub-environments.

        Notes
        -----
        If use_kdtree_query=True (default), uses KDTree for O(M log N) performance.
        Otherwise, sequentially queries each sub-environment for O(N×M) performance.

        The KDTree approach finds nearest bin centers globally, then verifies each
        point is actually contained by that bin using the sub-environment's contains()
        method. This is much faster for large numbers of sub-environments.

        """
        if points_nd.ndim != 2 or points_nd.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected points_nd of shape (M, {self.n_dims}), got {points_nd.shape}",
            )

        M = points_nd.shape[0]

        # Use KDTree-based approach if available
        # Note: Still respects sub-environment order (earlier in list wins)
        if self._kdtree is not None:
            out = np.full((M,), -1, dtype=int)

            # Process each sub-environment in order (maintain first-match semantics)
            for block in self._subenvs_info:
                env_i = block["env"]
                base = block["start_idx"]

                # Only query points that haven't been matched yet
                unmapped_mask = out == -1
                if not np.any(unmapped_mask):
                    break  # All points mapped

                unmapped_points = points_nd[unmapped_mask]
                sub_idxs = env_i.bin_at(unmapped_points)

                # Update output for matches
                matched_in_subenv = sub_idxs >= 0
                if np.any(matched_in_subenv):
                    # Map back to full array indices
                    unmapped_indices = np.where(unmapped_mask)[0]
                    matched_indices = unmapped_indices[matched_in_subenv]
                    out[matched_indices] = sub_idxs[matched_in_subenv] + base

            return out

        # Fall back to sequential query (original behavior)
        out = np.full((M,), -1, dtype=int)
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            sub_idxs = env_i.bin_at(points_nd)  # expects shape (M,)
            if sub_idxs.dtype not in (np.int32, np.int64):
                sub_idxs = sub_idxs.astype(int)
            mask = (sub_idxs >= 0) & (out == -1)
            out[mask] = sub_idxs[mask] + base

        return out

    def contains(self, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if points are contained in any bin of the composite environment.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (M, n_dims)
            Array of M points in n_dims-dimensional space.

        Returns
        -------
        NDArray[np.bool_], shape (M,)
            Boolean array where True indicates point is within any bin.
            Equivalent to self.bin_at(points_nd) != -1.

        """
        return np.asarray(self.bin_at(points_nd) != -1, dtype=np.bool_)

    def neighbors(self, bin_index: int) -> list[int]:
        """Get neighboring bins in the merged connectivity graph.

        Parameters
        ----------
        bin_index : int
            Composite bin index to query.

        Returns
        -------
        list[int]
            List of composite bin indices that are neighbors of bin_index.

        """
        if not (0 <= bin_index < self._total_bins):
            raise KeyError(
                f"Bin index {bin_index} is out of range [0..{self._total_bins - 1}]",
            )
        return list(self.connectivity.neighbors(bin_index))

    def distance_between(
        self,
        point1: np.ndarray | list[float] | tuple[float, ...],
        point2: np.ndarray | list[float] | tuple[float, ...],
        edge_weight: str = "distance",
    ) -> float:
        """Compute shortest-path distance between two points.

        Parameters
        ----------
        point1 : np.ndarray or list or tuple
            First point coordinates (length n_dims).
        point2 : np.ndarray or list or tuple
            Second point coordinates (length n_dims).
        edge_weight : str, default="distance"
            Edge attribute to use as weight for path computation.

        Returns
        -------
        float
            Shortest path distance between the two points. Returns np.inf
            if either point is outside all sub-environments.

        Notes
        -----
        Maps each point to a bin index via bin_at, then computes the
        shortest path length in the connectivity graph.

        """

        def _to_array(pt):
            arr = np.asarray(pt, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, self.n_dims)
            if arr.ndim != 2 or arr.shape[1] != self.n_dims:
                raise ValueError(
                    f"Expected a point of length {self.n_dims} or shape (1, {self.n_dims}), got {arr.shape}",
                )
            return arr

        arr1 = _to_array(point1)
        arr2 = _to_array(point2)

        bin1 = self.bin_at(arr1)[0]
        bin2 = self.bin_at(arr2)[0]
        if bin1 < 0 or bin2 < 0:
            return float(np.inf)
        return float(
            nx.shortest_path_length(
                self.connectivity,
                source=bin1,
                target=bin2,
                weight=edge_weight,
            ),
        )

    def bin_center_of(self, bin_indices: int | NDArray[np.int_]) -> NDArray[np.float64]:
        """Get bin center coordinates for specified bin indices.

        Parameters
        ----------
        bin_indices : int or NDArray[np.int_]
            Single composite bin index or 1-D array of bin indices.

        Returns
        -------
        NDArray[np.float64]
            N-D coordinate(s) of the specified bin(s). Shape (n_dims,) for
            a single index, or (M, n_dims) for M indices.

        """
        return np.asarray(self.bin_centers)[bin_indices]

    def bin_attributes(self) -> pd.DataFrame:
        """Get concatenated DataFrame of per-bin attributes from all sub-environments.

        Returns
        -------
        pd.DataFrame
            Concatenated bin attributes with columns 'child_active_bin_id'
            and 'composite_bin_id' added to track mapping from sub-environment
            bins to composite bins.

        """
        dfs = []
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            df = env_i.bin_attributes.copy()
            df["child_active_bin_id"] = df.index
            df["composite_bin_id"] = df.index + base
            dfs.append(df)
        composite_df = pd.concat(dfs, ignore_index=True)
        return composite_df

    def edge_attributes(self) -> pd.DataFrame:
        """Get concatenated DataFrame of per-edge attributes from all sub-environments.

        Returns
        -------
        pd.DataFrame
            Concatenated edge attributes with 'u_idx' and 'v_idx' shifted
            to composite bin indices. Includes MNN-inferred bridge edges
            connecting sub-environments.

        """
        dfs = []
        for block in self._subenvs_info:
            env_i = block["env"]
            base = block["start_idx"]
            df = env_i.edge_attributes.copy()
            df["composite_source_bin"] = df["source_bin"] + base
            df["composite_target_bin"] = df["target_bin"] + base
            dfs.append(df)

        if self._bridge_list:
            bridge_rows = []
            for (i_env, i_bin), (j_env, j_bin), w in self._bridge_list:
                block_i = self._subenvs_info[i_env]
                block_j = self._subenvs_info[j_env]
                source_composite_bin = block_i["start_idx"] + i_bin
                target_composite_bin = block_j["start_idx"] + j_bin
                bridge_rows.append(
                    {
                        "composite_source_bin": source_composite_bin,
                        "composite_target_bin": target_composite_bin,
                        "distance": w,
                        "weight": 1 / w,
                    },
                )
            bridge_df = pd.DataFrame(bridge_rows)
            dfs.append(bridge_df)

        composite_edges_df = pd.concat(dfs, ignore_index=True)
        return composite_edges_df

    def bins_in_region(self, region_name: str) -> NDArray[np.int_]:
        """Get composite bin indices that fall within a specified named region.

        Parameters
        ----------
        region_name : str
            Name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.int_]
            Array of composite bin indices (0 to n_bins - 1) that fall within
            the specified region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region type is unsupported or dimensions mismatch.

        Notes
        -----
        This method queries the region against all bin centers in the composite
        environment. For point regions, returns bins containing that point.
        For polygon regions (requires shapely), returns all bins whose centers
        fall within the polygon.

        Examples
        --------
        >>> comp = CompositeEnvironment([env1, env2])
        >>> comp.regions.add("goal", point=[10.0, 5.0])
        >>> goal_bins = comp.bins_in_region("goal")
        >>> print(f"Goal region contains {len(goal_bins)} bins")

        """
        region = self.regions[region_name]

        if region.kind == "point":
            # Point region - find bin at that point
            point_nd = np.asarray(region.data).reshape(1, -1)
            if point_nd.shape[1] != self.n_dims:
                raise ValueError(
                    f"Region point dimension {point_nd.shape[1]} "
                    f"does not match environment dimension {self.n_dims}.",
                )
            bin_idx = self.bin_at(point_nd)
            return np.asarray(bin_idx[bin_idx != -1], dtype=int)

        if region.kind == "polygon":
            # Polygon region - check which bin centers are inside
            try:
                import shapely
            except ImportError as e:
                raise RuntimeError(
                    "Polygon region queries require 'shapely'. "
                    "Install it with: pip install shapely"
                ) from e

            if self.n_dims != 2:
                raise ValueError(
                    f"Polygon regions are only supported for 2D environments. "
                    f"This composite environment has {self.n_dims} dimensions."
                )

            polygon = region.data
            x_coords = self.bin_centers[:, 0]
            y_coords = self.bin_centers[:, 1]
            contained_mask = shapely.contains_xy(polygon, x_coords, y_coords)
            return np.where(contained_mask)[0].astype(int)

        raise ValueError(
            f"Unsupported region kind: '{region.kind}'. "
            f"Supported kinds: 'point', 'polygon'."
        )

    def mask_for_region(self, region_name: str) -> NDArray[np.bool_]:
        """Get boolean mask for bins in a specified region.

        Parameters
        ----------
        region_name : str
            Name of a defined region in `self.regions`.

        Returns
        -------
        NDArray[np.bool_], shape (n_bins,)
            Boolean mask where True indicates the bin is within the region.

        Raises
        ------
        KeyError
            If `region_name` is not found in `self.regions`.
        ValueError
            If region type is unsupported or dimensions mismatch.

        Notes
        -----
        This is a convenience method that returns a boolean mask instead of
        bin indices. Equivalent to:
            mask = np.zeros(env.n_bins, dtype=bool)
            mask[env.bins_in_region(region_name)] = True

        Examples
        --------
        >>> comp = CompositeEnvironment([env1, env2])
        >>> comp.regions.add("arena", polygon=shapely_polygon)
        >>> arena_mask = comp.mask_for_region("arena")
        >>> occupancy_in_arena = occupancy[arena_mask]

        """
        mask = np.zeros(self.n_bins, dtype=bool)
        bins = self.bins_in_region(region_name)
        mask[bins] = True
        return mask

    def shortest_path(
        self, source_bin: int, target_bin: int, edge_weight: str = "distance"
    ) -> list[int]:
        """Find shortest path between two bin indices in the composite graph.

        Parameters
        ----------
        source_bin : int
            Composite bin index to start from (0 to n_bins - 1).
        target_bin : int
            Composite bin index to reach (0 to n_bins - 1).
        edge_weight : str, default="distance"
            Edge attribute to use as weight for pathfinding.

        Returns
        -------
        list[int]
            List of composite bin indices forming the shortest path from
            source_bin to target_bin, including both endpoints. Returns
            empty list if no path exists.

        Raises
        ------
        nx.NodeNotFound
            If source_bin or target_bin is not in the graph.

        Warnings
        --------
        UserWarning
            If no path exists between the bins (disconnected components).

        Notes
        -----
        Uses NetworkX shortest_path with specified edge weights. The path
        may cross bridge edges connecting different sub-environments.

        Examples
        --------
        >>> comp = CompositeEnvironment([env1, env2], auto_bridge=True)
        >>> path = comp.shortest_path(0, 100)  # Path from bin 0 to bin 100
        >>> print(f"Path length: {len(path)} bins")

        """
        try:
            path: list[int] = nx.shortest_path(
                self.connectivity,
                source=source_bin,
                target=target_bin,
                weight=edge_weight,
            )
            return path
        except nx.NetworkXNoPath:
            import warnings

            warnings.warn(
                f"No path found between bin {source_bin} and bin {target_bin}. "
                f"The bins may be in disconnected components. "
                f"Returning empty path.",
                UserWarning,
                stacklevel=2,
            )
            return []

    def info(self, return_string: bool = False) -> str | None:
        """Print or return diagnostic information about the composite environment.

        Parameters
        ----------
        return_string : bool, default=False
            If True, return the info string instead of printing.

        Returns
        -------
        str or None
            If return_string=True, returns the formatted info string.
            Otherwise prints to stdout and returns None.

        Notes
        -----
        Displays summary information including:
        - Number of sub-environments
        - Total bins and dimensions
        - Number of bridge edges connecting sub-environments
        - Per-sub-environment statistics (type, bins, regions)
        - Bridge edge statistics

        Examples
        --------
        >>> comp = CompositeEnvironment([env1, env2], auto_bridge=True)
        >>> comp.info()
        Composite Environment Information
        ==================================
        ...

        """
        lines = []
        lines.append("Composite Environment Information")
        lines.append("=" * 50)
        lines.append(f"Number of sub-environments: {len(self._subenvs_info)}")
        lines.append(f"Total bins: {self.n_bins}")
        lines.append(f"Dimensions: {self.n_dims}")
        lines.append(f"Bridge edges: {len(self._bridge_list)}")
        lines.append("")

        lines.append("Sub-Environment Details:")
        lines.append("-" * 50)
        for i, block in enumerate(self._subenvs_info):
            env_i = block["env"]
            lines.append(f"  [{i}] {env_i.name or '(unnamed)'}")
            lines.append(f"      Type: {env_i.layout_type}")
            lines.append(
                f"      Bins: {env_i.n_bins} (composite indices: {block['start_idx']}-{block['end_idx']})"
            )
            lines.append(f"      Regions: {len(env_i.regions)}")
            if len(env_i.regions) > 0:
                lines.append(f"               {list(env_i.regions.keys())}")
        lines.append("")

        lines.append("Bridge Statistics:")
        lines.append("-" * 50)
        if self._bridge_list:
            distances = [w for _, _, w in self._bridge_list]
            lines.append(f"  Count: {len(self._bridge_list)}")
            lines.append(f"  Min distance: {min(distances):.4f}")
            lines.append(f"  Max distance: {max(distances):.4f}")
            lines.append(f"  Mean distance: {np.mean(distances):.4f}")
        else:
            lines.append(
                "  No bridges (auto_bridge=False or no mutual nearest neighbors found)"
            )
        lines.append("")

        lines.append("Composite Regions:")
        lines.append("-" * 50)
        if len(self.regions) > 0:
            for name, region in self.regions.items():
                lines.append(f"  - {name}: {region.kind}")
        else:
            lines.append("  (No regions defined)")

        info_str = "\n".join(lines)

        if return_string:
            return info_str
        else:
            print(info_str)
            return None

    def save(self, filepath: str) -> None:
        """Save the CompositeEnvironment to a file using pickle.

        Parameters
        ----------
        filepath : str
            Path where the composite environment will be saved.

        Warnings
        --------
        This method uses pickle serialization. Only load files from trusted
        sources, as pickle can execute arbitrary code.

        Notes
        -----
        The saved file contains:
        - All sub-environments with their complete state
        - Bridge edges and connectivity information
        - Regions from all sub-environments
        - Composite metadata

        The file can be loaded with CompositeEnvironment.load().

        Examples
        --------
        >>> comp = CompositeEnvironment([env1, env2])
        >>> comp.save("my_composite_env.pkl")
        >>> loaded = CompositeEnvironment.load("my_composite_env.pkl")

        """
        import pickle
        from pathlib import Path

        # Package everything needed to reconstruct the composite
        save_dict = {
            "subenvs": [block["env"] for block in self._subenvs_info],
            "auto_bridge": False,  # Don't re-infer bridges on load
            "max_mnn_distance": None,
            "use_kdtree_query": self._use_kdtree_query,
            "bridge_list": self._bridge_list,
            "layout_params": self._layout_params_used,
        }

        Path(filepath).write_bytes(pickle.dumps(save_dict))

    @classmethod
    def load(cls, filepath: str) -> "CompositeEnvironment":
        """Load a CompositeEnvironment from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved composite environment file.

        Returns
        -------
        CompositeEnvironment
            Reconstructed composite environment with all sub-environments
            and bridge edges restored.

        Warnings
        --------
        This method uses pickle deserialization. Only load files from trusted
        sources, as pickle can execute arbitrary code.

        Examples
        --------
        >>> comp = CompositeEnvironment.load("my_composite_env.pkl")
        >>> print(f"Loaded composite with {comp.n_bins} bins")

        """
        import pickle
        from pathlib import Path

        save_dict = pickle.loads(Path(filepath).read_bytes())

        # Reconstruct without auto-bridging
        use_kdtree = save_dict.get(
            "use_kdtree_query", True
        )  # Default to True for backwards compatibility
        comp = cls(
            subenvs=save_dict["subenvs"],
            auto_bridge=False,
            max_mnn_distance=None,
            use_kdtree_query=use_kdtree,
        )

        # Restore the saved bridges
        for (i_env, i_bin), (j_env, j_bin), w in save_dict["bridge_list"]:
            # Bridge already exists, skip if duplicate
            source_composite_bin = comp._subenvs_info[i_env]["start_idx"] + i_bin
            target_composite_bin = comp._subenvs_info[j_env]["start_idx"] + j_bin
            if not comp.connectivity.has_edge(
                source_composite_bin, target_composite_bin
            ):
                comp._add_bridge_edge(i_env, i_bin, j_env, j_bin, w)

        return comp

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        sub_env_plot_kwargs: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        bridge_edge_kwargs: dict[str, Any] | None = None,
        show_sub_env_labels: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the composite environment.

        This method plots each sub-environment and then overlays the bridge edges.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The Matplotlib axes to plot on. If None, a new figure and axes
            are created. Defaults to None.
        sub_env_plot_kwargs : Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]], optional
            Keyword arguments to pass to the `plot()` method of each sub-environment.
            If a single dict, it's applied to all sub-environments.
            If a list, it should have the same length as `subenvs`, and each element
            (a dict or None) is passed to the corresponding sub-environment's plot call.
            Defaults to None (empty dict for each).
        bridge_edge_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments for plotting the bridge edges (passed to `ax.plot`).
            Defaults to {'color': 'red', 'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.7}.
        show_sub_env_labels : bool, default=False
            If True, attempts to label the approximate center of each sub-environment.
        **kwargs : Any
            Additional keyword arguments passed to `plt.subplots()` if `ax` is None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the composite environment was plotted.

        """
        if ax is None:
            fig_kwargs: dict[str, Any] = {"figsize": (10, 10)}  # Default figsize
            fig_kwargs.update(kwargs)
            # Determine if plot should be 3D based on n_dims
            if self.n_dims == 3:
                fig_kwargs["projection"] = "3d"

            is_3d = fig_kwargs.get("projection") == "3d"
            if is_3d:
                figsize_val = fig_kwargs.get("figsize", (10, 10))
                fig = plt.figure(figsize=figsize_val)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(
                    **{k: v for k, v in fig_kwargs.items() if k != "projection"},
                )

        # Plot each sub-environment
        for i, block_info in enumerate(self._subenvs_info):
            env_i = block_info["env"]
            current_env_kwargs: dict[str, Any] = {}
            if isinstance(sub_env_plot_kwargs, list):
                if i < len(sub_env_plot_kwargs):
                    kwargs_i = sub_env_plot_kwargs[i]
                    if kwargs_i is not None:
                        current_env_kwargs = kwargs_i
            elif isinstance(sub_env_plot_kwargs, dict):
                current_env_kwargs = sub_env_plot_kwargs

            env_i.plot(ax=ax, **current_env_kwargs)

            if show_sub_env_labels and env_i.n_bins > 0:
                # Add a label at the mean position of the sub-environment's bin centers
                mean_pos = np.mean(env_i.bin_centers, axis=0)
                label_text = f"Env {i}"
                if env_i.name:
                    label_text += f": {env_i.name}"

                if self.n_dims == 2:
                    ax.text(
                        mean_pos[0],
                        mean_pos[1],
                        label_text,
                        color="blue",
                        ha="center",
                        va="center",
                        bbox={"facecolor": "white", "alpha": 0.5, "pad": 0.1},
                    )
                elif self.n_dims == 3:
                    # matplotlib 3D text() signature differs from 2D stubs
                    from typing import Any as _Any

                    text_func = cast("_Any", ax.text)
                    text_func(
                        mean_pos[0],
                        mean_pos[1],
                        mean_pos[2],
                        label_text,
                        color="blue",
                        ha="center",
                        va="center",
                    )

        # Plot bridge edges
        _bridge_kwargs = {
            "color": "red",
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.7,
            "zorder": 0,
        }
        if bridge_edge_kwargs is not None:
            _bridge_kwargs.update(bridge_edge_kwargs)

        for (
            (i_env_idx, i_bin_sub_idx),
            (j_env_idx, j_bin_sub_idx),
            _,
        ) in self._bridge_list:
            block_i = self._subenvs_info[i_env_idx]
            block_j = self._subenvs_info[j_env_idx]

            # Get original bin centers from sub-environments for plotting bridge start/end
            # This avoids issues if self.bin_centers has a different order or structure
            # than the sub-environment's original bin_centers array.
            # However, self.bin_centers is authoritative for the composite.
            # We need composite indices.

            u_composite = block_i["start_idx"] + i_bin_sub_idx
            v_composite = block_j["start_idx"] + j_bin_sub_idx

            pos_u = self.bin_centers[u_composite]
            pos_v = self.bin_centers[v_composite]

            # matplotlib plot() stubs don't properly handle **kwargs
            from typing import Any as _Any

            plot_func = cast("_Any", ax.plot)

            if self.n_dims == 2:
                plot_func([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], **_bridge_kwargs)
            elif self.n_dims == 3:
                plot_func(
                    [pos_u[0], pos_v[0]],
                    [pos_u[1], pos_v[1]],
                    [pos_u[2], pos_v[2]],
                    **_bridge_kwargs,
                )
            # Add other dimensionalities if needed

        ax.set_title("Composite Environment")
        return ax
