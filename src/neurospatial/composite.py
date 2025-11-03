"""CompositeEnvironment: merges multiple Environment instances into a single unified Environment-like API.
Bridge edges between sub-environments are inferred automatically via mutual-nearest-neighbor (MNN).

This class exposes the same public interface as the base `Environment` class:
  - Properties: n_dims, n_bins, bin_centers, connectivity, is_1d, dimension_ranges,
                grid_edges, grid_shape, active_mask, regions
  - Methods:    bin_at, contains, neighbors, distance_between, bin_center_of,
                bin_attributes, edge_attributes

(Note: serialization methods such as save/load and factory methods like from_layout are not included,
since CompositeEnvironment wraps pre-fitted sub-environments.)
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
    ):
        """Build a CompositeEnvironment from a list of pre-fitted Environment instances.

        Parameters
        ----------
        subenvs : List[Environment]
            A list of fitted Environment objects. All must share the same n_dims.
        auto_bridge : bool, default=True
            If True, automatically infer “bridge edges” between each pair of sub-environments
            using a mutual nearest-neighbor heuristic on their bin_centers.
        max_mnn_distance : Optional[float]
            If provided, any automatically inferred bridge whose Euclidean distance exceeds
            this threshold is discarded. If None, no distance filtering is applied.

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
        if len(subenvs) == 0:
            raise ValueError("At least one sub-environment is required.")

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
        for e in subenvs:
            n_bins = e.bin_centers.shape[0]
            self._subenvs_info.append(
                {"env": e, "start_idx": offset, "end_idx": offset + n_bins - 1},
            )
            offset += n_bins
        self._total_bins = offset

        # Stack all bin_centers into one array of shape (N_total, n_dims)
        self.bin_centers = np.vstack([e.bin_centers for e in subenvs])

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
            "sub_environment_types": [s.layout_type for s in subenvs],
        }
        self._is_fitted = (
            True  # Composite environment is considered 'fitted' upon construction
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

        u = block_i["start_idx"] + i_bin
        v = block_j["start_idx"] + j_bin
        self.connectivity.add_edge(u, v, distance=w, weight=1 / w if w > 0 else np.inf)
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
                        d = dist_ij[i_idx]
                        if (max_distance is not None) and (d > max_distance):
                            continue
                        self._add_bridge_edge(i, i_idx, j, j_idx, float(d))

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
        Calls each subenv.bin_at(points_nd) and uses the first match.
        Composite index = sub_idx + start_idx for the matching sub-environment.

        """
        if points_nd.ndim != 2 or points_nd.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected points_nd of shape (M, {self.n_dims}), got {points_nd.shape}",
            )

        M = points_nd.shape[0]
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
                u = block_i["start_idx"] + i_bin
                v = block_j["start_idx"] + j_bin
                bridge_rows.append(
                    {
                        "composite_source_bin": u,
                        "composite_target_bin": v,
                        "distance": w,
                        "weight": 1 / w,
                    },
                )
            bridge_df = pd.DataFrame(bridge_rows)
            dfs.append(bridge_df)

        composite_edges_df = pd.concat(dfs, ignore_index=True)
        return composite_edges_df

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
                if i < len(sub_env_plot_kwargs) and sub_env_plot_kwargs[i] is not None:
                    current_env_kwargs = sub_env_plot_kwargs[i]  # type: ignore[assignment]
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
