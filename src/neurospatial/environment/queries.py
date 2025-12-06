"""Spatial query methods for Environment class.

This module provides the EnvironmentQueries mixin class containing methods
for querying spatial properties of an environment:

- bin_at: Map continuous points to discrete bin indices
- contains: Check if points fall within any active bin
- bin_center_of: Get center coordinates of bins by index
- neighbors: Find neighboring bins for a given bin
- bin_sizes: Get area/volume of each active bin (cached property)
- distance_between: Calculate geodesic distance between two points
- path_between: Find shortest path between two bins
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import cached_property
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted


class EnvironmentQueries:
    """Mixin class providing spatial query methods for Environment.

    This class is NOT a dataclass. It provides methods that query spatial
    properties of an Environment instance, such as mapping points to bins,
    finding neighbors, and computing distances.

    Methods in this class assume they are mixed into an Environment instance
    that provides: layout, bin_centers, connectivity, is_1d attributes.
    """

    @check_fitted
    def bin_at(self: SelfEnv, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map N-dimensional points to bins using geometric containment.

        This method uses the layout-specific geometric logic to determine which
        bin contains each point. This is different from nearest-neighbor mapping
        and respects the actual geometry of the bins (grid cells, hexagons, etc.).

        For nearest-neighbor mapping with caching and tie-breaking, see
        `neurospatial.map_points_to_bins()`.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to map.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            An array of active bin indices (0 to `n_active_bins - 1`).
            A value of -1 indicates that the corresponding point did not map
            to any active bin (e.g., it's outside the environment).

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        See Also
        --------
        neurospatial.map_points_to_bins : Nearest-neighbor mapping with KDTree caching

        Notes
        -----
        **Geometric Containment vs Nearest-Neighbor:**

        - `bin_at()` (this method): Uses layout-specific geometric containment
          to determine which bin actually contains the point. For grids, this
          checks which grid cell the point falls into. For hexagons, which
          hexagon contains it. This is exact but may be slower for large batches.

        - `map_points_to_bins()`: Uses KDTree to find the bin whose center is
          closest to the point. This is fast (O(log N)) with caching, supports
          tie-breaking on boundaries, and allows distance thresholds. Best for
          large batches, trajectory processing, and when approximate assignment
          is acceptable.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> points = np.array([[5.0, 5.0], [15.0, 15.0]])
        >>> indices = env.bin_at(points)
        >>> print(indices)  # doctest: +SKIP
        [12 -1]  # Second point outside environment

        """
        return self.layout.point_to_bin_index(points_nd)

    @check_fitted
    def contains(self: SelfEnv, points_nd: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Check if N-dimensional continuous points fall within any active bin.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            An array of N-dimensional points to check.

        Returns
        -------
        NDArray[np.bool_], shape (n_points,)
            A boolean array where `True` indicates the corresponding point
            maps to an active bin, and `False` indicates it does not.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        Notes
        -----
        This method is optimized to avoid redundant KDTree queries by reusing
        the bin index computation from `bin_at()` and checking for the -1 sentinel.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> points = np.array([[5.0, 5.0], [15.0, 15.0]])
        >>> mask = env.contains(points)
        >>> print(mask)  # doctest: +SKIP
        [ True False]  # First point in environment, second outside

        """
        # Optimized: compute indices once and check for -1 sentinel
        # This avoids redundant KDTree queries compared to calling bin_at() separately
        indices = self.layout.point_to_bin_index(points_nd)
        return np.asarray(indices != -1, dtype=np.bool_)

    @check_fitted
    def bin_center_of(
        self: SelfEnv,
        bin_indices: int | Sequence[int] | NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Given one or more active-bin indices, return their N-D center coordinates.

        Parameters
        ----------
        bin_indices : int or sequence of int
            Index (or list/array of indices) of active bins (0 <= idx < self.n_bins).

        Returns
        -------
        centers : NDArray[np.float64]
            The center coordinate(s) of the requested bin(s).
            Shape is (len(bin_indices), n_dims) if multiple indices,
            or (n_dims,) if single index.

        Raises
        ------
        RuntimeError
            If the environment is not fitted.
        IndexError
            If any bin index is out of range.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> center = env.bin_center_of(0)
        >>> print(center.shape)
        (2,)
        >>> centers = env.bin_center_of([0, 1, 2])
        >>> print(centers.shape)  # doctest: +SKIP
        (3, 2)

        """
        return np.asarray(
            self.bin_centers[np.asarray(bin_indices, dtype=int)], dtype=np.float64
        )

    @check_fitted
    def neighbors(self: SelfEnv, bin_index: int) -> list[int]:
        """Find indices of neighboring active bins for a given active bin index.

        This method delegates to the `neighbors` method of the
        underlying `LayoutEngine`, which typically uses the `connectivity`.

        Parameters
        ----------
        bin_index : int
            The index (0 to `n_active_bins - 1`) of the active bin for which
            to find neighbors.

        Returns
        -------
        list[int]
            A list of active bin indices that are neighbors to `bin_index`.
            Can be used directly for array indexing.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> neighbors = env.neighbors(0)
        >>> print(len(neighbors))  # doctest: +SKIP
        4  # Number of neighbors varies by layout

        """
        return list(self.connectivity.neighbors(bin_index))

    # Note: Decorator order matters - @cached_property must be on top
    # so that @check_fitted can see the underlying method
    @cached_property
    @check_fitted
    def bin_sizes(self: SelfEnv) -> NDArray[np.float64]:
        """Calculate the area (for 2D) or volume (for 3D+) of each active bin.

        This represent the actual size of each bin in the environment, as
        opposed to the requested `bin_size` which is the nominal size used
        during layout creation.

        For 1D environments, this typically returns the length of each bin.
        This method delegates to the `bin_sizes` method of the
        underlying `LayoutEngine`.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            An array containing the area/volume/length of each active bin.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.

        Notes
        -----
        This property is cached after the first call for efficient repeated access.
        The decorator order (@cached_property above @check_fitted) is intentional:
        @cached_property caches the result, and @check_fitted ensures the environment
        is fitted before computing.

        For environments with >100,000 bins, this may consume significant memory.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> sizes = env.bin_sizes
        >>> print(sizes.shape[0] == env.n_bins)
        True

        """
        return self.layout.bin_sizes()

    def distance_between(
        self: SelfEnv,
        point1: NDArray[np.float64],
        point2: NDArray[np.float64],
        edge_weight: str = "distance",
    ) -> float:
        """Calculate the geodesic distance between two points in the environment.

        Points are first mapped to their nearest active bins using `self.bin_at()`.
        The geodesic distance (distance along the shortest path through the space)
        is then the shortest path length in the `connectivity` graph between these
        bins, using the specified `edge_weight`.

        Parameters
        ----------
        point1 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The first N-dimensional point.
        point2 : NDArray[np.float64], shape (n_dims,) or (1, n_dims)
            The second N-dimensional point.
        edge_weight : str, optional
            The edge attribute to use as weight for path calculation,
            by default "distance". If None, the graph is treated as unweighted.

        Returns
        -------
        float
            The geodesic distance. Returns `np.inf` if points do not map to
            valid active bins, if bins are disconnected, or if the connectivity
            graph is not available.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> p1 = np.array([1.0, 1.0])
        >>> p2 = np.array([9.0, 9.0])
        >>> dist = env.distance_between(p1, p2)
        >>> print(dist > 0)
        True

        """
        source_bin = self.bin_at(np.atleast_2d(point1))[0]
        target_bin = self.bin_at(np.atleast_2d(point2))[0]

        if source_bin == -1 or target_bin == -1:
            # One or both points didn't map to a valid active bin
            return np.inf

        try:
            return float(
                nx.shortest_path_length(
                    self.connectivity,
                    source=source_bin,
                    target=target_bin,
                    weight=edge_weight,
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return np.inf

    @check_fitted
    def path_between(
        self: SelfEnv,
        source_active_bin_idx: int,
        target_active_bin_idx: int,
    ) -> list[int]:
        """Find the shortest path between two active bins.

        The path is a sequence of active bin indices (0 to n_active_bins - 1)
        connecting the source to the target. Path calculation uses the
        'distance' attribute on the edges of the `connectivity`
        as weights.

        Parameters
        ----------
        source_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the start of the path.
        target_active_bin_idx : int
            The active bin index (0 to n_active_bins - 1) for the end of the path.

        Returns
        -------
        list[int]
            A list of active bin indices representing the shortest path from
            source to target. The list includes both the source and target indices.
            Returns an empty list if the source and target are the same, or if
            no path exists, or if nodes are not found.

        Raises
        ------
        RuntimeError
            If called before the environment is fitted.
        nx.NodeNotFound
            If `source_active_bin_idx` or `target_active_bin_idx` is not
            a node in the `connectivity`.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> path = env.path_between(0, 5)
        >>> print(len(path) > 0)  # doctest: +SKIP
        True

        """
        graph = self.connectivity

        if source_active_bin_idx == target_active_bin_idx:
            return [source_active_bin_idx]

        try:
            path = nx.shortest_path(
                graph,
                source=source_active_bin_idx,
                target=target_active_bin_idx,
                weight="distance",
            )
            return list(path)
        except nx.NetworkXNoPath:
            warnings.warn(
                f"No path found between active bin {source_active_bin_idx} "
                f"and {target_active_bin_idx}.",
                UserWarning,
            )
            return []
        except nx.NodeNotFound as e:
            # Re-raise if the user provides an invalid node index for active bins
            raise nx.NodeNotFound(
                f"Node not found in connectivity graph: {e}. "
                "Ensure source/target indices are valid active bin indices.",
            ) from e

    def distance_to(
        self: SelfEnv,
        targets: Sequence[int] | str,
        *,
        metric: Literal["euclidean", "geodesic"] = "geodesic",
    ) -> NDArray[np.float64]:
        """Compute distance from each bin to target set.

        This method computes the distance from every bin in the environment to
        the nearest target bin. Useful for:
        - Navigation and path planning
        - Computing distance-based features
        - Analyzing spatial distributions relative to landmarks
        - Creating distance fields for visualization

        Parameters
        ----------
        targets : Sequence[int] or str
            Target bin indices, or a region name. If a region name is provided,
            all bins inside that region are used as targets (multi-source).
        metric : {'euclidean', 'geodesic'}, default='geodesic'
            Distance metric to use:
            - 'euclidean': Straight-line distance in physical coordinates (same units as bin_centers).
            - 'geodesic': Graph distance respecting connectivity (shortest path, in physical units).

        Returns
        -------
        distances : NDArray[np.float64], shape (n_bins,)
            Distance from each bin to the nearest target, in the same units as
            bin_centers (e.g., cm, meters, pixels). For bins unreachable
            from all targets (disconnected graph components), returns np.inf.

        Raises
        ------
        ValueError
            If targets is empty, or if target bin indices are out of range,
            or if metric is invalid.
        KeyError
            If targets is a string (region name) that doesn't exist in self.regions.
        TypeError
            If targets is neither a sequence of integers nor a string.

        See Also
        --------
        rings : Compute k-hop neighborhoods (BFS layers).
        reachable_from : Find bins reachable from a source within a radius.
        distance_field : Low-level function for computing geodesic distances.

        Notes
        -----
        **Geodesic Distance**:
        Uses Dijkstra's algorithm to compute shortest paths on the connectivity
        graph. Edge weights are the 'distance' attribute (physical distance between
        bin centers). For disconnected graphs, unreachable bins have distance np.inf.

        **Euclidean Distance**:
        Computes straight-line distance in the coordinate space, ignoring graph
        connectivity. This is faster but doesn't respect physical barriers.

        **Multi-Source Distances**:
        When multiple targets are provided (or a region containing multiple bins),
        each bin's distance is the minimum distance to any target.

        **Region-Based Targets**:
        If targets is a string, it must be a valid region name in self.regions.
        All bins inside that region (as determined by region_membership) become
        target bins.

        Examples
        --------
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> # Create 10x10 grid
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> # Distance to goal region (polygon covering multiple bins)
        >>> _ = env.regions.add("goal", polygon=box(8.0, 8.0, 10.0, 10.0))
        >>> dist = env.distance_to("goal", metric="geodesic")
        >>> dist.shape
        (100,)
        >>> bool(np.all(dist >= 0.0))
        True

        >>> # Distance to specific bins (opposite corners)
        >>> targets = [0, env.n_bins - 1]
        >>> dist = env.distance_to(targets, metric="euclidean")
        >>> float(dist[targets[0]])
        0.0
        >>> float(dist[targets[1]])
        0.0

        """
        # Validate metric
        if metric not in ("euclidean", "geodesic"):
            raise ValueError(
                f"metric must be 'euclidean' or 'geodesic', got '{metric}'"
            )

        # Handle region name targets
        if isinstance(targets, str):
            region_name = targets
            if region_name not in self.regions:
                raise KeyError(
                    f"Region '{region_name}' not found in environment regions. "
                    f"Available regions: {list(self.regions.keys())}"
                )

            # Get bins in region via membership
            membership = self.region_membership()
            region_idx = list(self.regions.keys()).index(region_name)
            targets = np.where(membership[:, region_idx])[0].tolist()

            if len(targets) == 0:
                warnings.warn(
                    f"Region '{region_name}' contains no bins. "
                    f"All distances will be inf.",
                    UserWarning,
                    stacklevel=2,
                )

        # Convert to numpy array for validation
        try:
            target_array = np.asarray(targets, dtype=np.int32)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"targets must be a sequence of integers or a string (region name), "
                f"got {type(targets).__name__}"
            ) from e

        # Validate targets not empty
        if len(target_array) == 0:
            raise ValueError(
                "targets cannot be empty. Provide at least one target bin index "
                "or a region name containing bins."
            )

        # Validate target indices in range
        if np.any(target_array < 0) or np.any(target_array >= self.n_bins):
            invalid = target_array[(target_array < 0) | (target_array >= self.n_bins)]
            raise ValueError(
                f"Target bin indices must be in range [0, {self.n_bins}), "
                f"got invalid indices: {invalid.tolist()}"
            )

        # Compute distances based on metric
        if metric == "euclidean":
            # Euclidean distance: straight-line distance to nearest target
            # Vectorized implementation using broadcasting for performance
            target_positions = self.bin_centers[target_array]  # (n_targets, n_dims)

            # Broadcasting: (n_bins, 1, n_dims) - (1, n_targets, n_dims) -> (n_bins, n_targets)
            diffs = (
                self.bin_centers[:, np.newaxis, :] - target_positions[np.newaxis, :, :]
            )
            dists_to_targets = np.linalg.norm(diffs, axis=2)  # (n_bins, n_targets)
            distances_result: NDArray[np.float64] = np.min(
                dists_to_targets, axis=1
            )  # (n_bins,)

        else:  # metric == "geodesic"
            # Geodesic distance: graph-based shortest path
            from neurospatial.ops.distance import distance_field

            distances_result = np.asarray(
                distance_field(self.connectivity, sources=target_array.tolist()),
                dtype=np.float64,
            )

        return distances_result

    def reachable_from(
        self: SelfEnv,
        source_bin: int,
        *,
        radius: int | float | None = None,
        metric: Literal["hops", "geodesic"] = "hops",
    ) -> NDArray[np.bool_]:
        """Find all bins reachable from source within optional radius.

        This method performs graph traversal to find which bins can be reached
        from a starting bin, optionally constrained by a maximum distance.
        Useful for:
        - Computing neighborhoods and local regions
        - Identifying reachable areas from a starting position
        - Building distance-limited queries

        Parameters
        ----------
        source_bin : int
            Starting bin index. Must be in range [0, n_bins).
        radius : int, float, or None, optional
            Maximum distance/hops. If None, find all reachable bins in the
            same connected component.
            - For metric='hops': radius is maximum number of edges.
            - For metric='geodesic': radius is maximum graph distance in
              physical units.
        metric : {'hops', 'geodesic'}, default='hops'
            Distance metric to use:
            - 'hops': Count graph edges (breadth-first search).
            - 'geodesic': Sum edge distances in physical units (Dijkstra).

        Returns
        -------
        reachable : NDArray[np.bool_], shape (n_bins,)
            Boolean mask where True indicates reachable bins.
            The source bin is always reachable (reachable[source_bin] = True).

        Raises
        ------
        ValueError
            If source_bin is not in valid range [0, n_bins).
            If radius is negative.
            If metric is not 'hops' or 'geodesic'.

        See Also
        --------
        components : Find connected components.
        distance_between : Compute distance between two bins.

        Notes
        -----
        **Algorithm details**:
        - metric='hops': Uses breadth-first search (BFS) to specified depth.
        - metric='geodesic': Uses Dijkstra's algorithm with distance cutoff.

        **Performance**:
        - With radius=None: O(V + E) where V=bins, E=edges
        - With radius: Depends on local graph density, typically much faster

        The geodesic metric uses the 'distance' attribute on graph edges,
        which represents the Euclidean distance between bin centers.

        Examples
        --------
        >>> # All bins within 3 edges of bin 10
        >>> mask = env.reachable_from(10, radius=3, metric="hops")
        >>> neighbor_bins = np.where(mask)[0]
        >>> print(f"Found {len(neighbor_bins)} neighbors within 3 hops")
        Found 37 neighbors within 3 hops

        >>> # All bins within 50.0 units geodesic distance from goal region
        >>> goal_bin = env.bins_in_region("goal")[0]
        >>> mask = env.reachable_from(goal_bin, radius=50.0, metric="geodesic")
        >>> print(f"Bins within 50 units: {mask.sum()}")
        Bins within 50 units: 125

        >>> # All bins in same component (no radius limit)
        >>> mask = env.reachable_from(source_bin=0, radius=None)
        >>> print(f"Component size: {mask.sum()} bins")
        Component size: 1000 bins

        """
        # Input validation
        if not isinstance(source_bin, (int, np.integer)):
            raise TypeError(
                f"source_bin must be an integer, got {type(source_bin).__name__}"
            )

        if not 0 <= source_bin < self.n_bins:
            raise ValueError(
                f"source_bin must be in range [0, n_bins) where n_bins={self.n_bins}. "
                f"Got source_bin={source_bin}"
            )

        if radius is not None and radius < 0:
            raise ValueError(
                f"radius must be non-negative or None. Got radius={radius}"
            )

        if metric not in ("hops", "geodesic"):
            raise ValueError(
                f"metric must be 'hops' or 'geodesic'. Got metric='{metric}'"
            )

        # Initialize result mask
        reachable = np.zeros(self.n_bins, dtype=bool)

        # Case 1: No radius limit - find entire connected component
        if radius is None:
            # Use NetworkX to find all nodes in same component
            for component_nodes in nx.connected_components(self.connectivity):
                if source_bin in component_nodes:
                    for node in component_nodes:
                        reachable[node] = True
                    break
            return reachable

        # Case 2: Radius-limited search
        if metric == "hops":
            # Breadth-first search to specified depth
            # Use NetworkX's single_source_shortest_path_length with cutoff
            distances = nx.single_source_shortest_path_length(
                self.connectivity, source_bin, cutoff=int(radius)
            )
            # Mark all nodes within radius as reachable
            for node in distances:
                reachable[node] = True

        else:  # metric == 'geodesic'
            # Dijkstra's algorithm with distance cutoff
            # Use NetworkX's single_source_dijkstra_path_length
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.connectivity, source_bin, cutoff=radius, weight="distance"
                )
                # Mark all nodes within radius as reachable
                for node in distances:
                    reachable[node] = True
            except nx.NetworkXError:
                # If source_bin has no edges, only mark itself as reachable
                reachable[source_bin] = True

        return reachable

    def components(
        self: SelfEnv,
        *,
        largest_only: bool = False,
    ) -> list[NDArray[np.int32]]:
        """Find connected components of the environment graph.

        A connected component is a maximal subset of bins where every pair
        of bins is connected by a path through the graph. This is useful for:
        - Identifying disconnected regions in masked environments
        - Finding traversable subregions
        - Detecting isolated islands in the environment

        Parameters
        ----------
        largest_only : bool, default=False
            If True, return only the largest component.
            If False, return all components sorted by size (largest first).

        Returns
        -------
        components : list[NDArray[np.int32]]
            List of bin index arrays, one per component.
            Components are sorted by size (largest first).
            Each array contains the bin indices in that component.

        See Also
        --------
        reachable_from : Find bins reachable from a source within a radius.

        Notes
        -----
        This method uses NetworkX's connected_components algorithm to identify
        connected subgraphs in the environment's connectivity graph.

        For environments with a single connected region (e.g., most regular grids),
        this will return a single component containing all bins.

        Examples
        --------
        >>> # Find all components in environment
        >>> comps = env.components()
        >>> print(f"Found {len(comps)} components")
        Found 2 components
        >>> print(f"Largest component has {len(comps[0])} bins")
        Largest component has 150 bins

        >>> # Get only the largest component
        >>> largest = env.components(largest_only=True)[0]
        >>> print(f"Largest component: {len(largest)} of {env.n_bins} bins")
        Largest component: 150 of 200 bins

        """
        # Find connected components using NetworkX
        component_sets = nx.connected_components(self.connectivity)

        # Convert sets to arrays and sort by size (largest first)
        components = [
            np.asarray(sorted(comp), dtype=np.int32) for comp in component_sets
        ]
        components.sort(key=len, reverse=True)

        # Return only largest if requested
        if largest_only:
            return components[:1]

        return components

    def rings(
        self: SelfEnv,
        center_bin: int,
        *,
        hops: int,
    ) -> list[NDArray[np.int32]]:
        """Compute k-hop neighborhoods (BFS layers).

        This method performs breadth-first search (BFS) from the center bin,
        organizing bins into "rings" by their hop distance. Ring k contains
        all bins exactly k graph edges away from the center. Useful for:
        - Local neighborhood analysis
        - Distance-based feature extraction
        - Spatial smoothing with varying radii
        - Analyzing connectivity patterns

        Parameters
        ----------
        center_bin : int
            Starting bin index.
        hops : int
            Number of hop layers to compute (non-negative).

        Returns
        -------
        rings : list[NDArray[np.int32]], length hops+1
            List of bin index arrays, one per hop distance.
            rings[k] contains all bins exactly k hops from center.
            rings[0] = [center_bin] (the center itself).
            If fewer than hops layers exist (small or disconnected graph),
            later rings will be empty arrays.

        Raises
        ------
        ValueError
            If center_bin is out of range [0, n_bins), or if hops is negative.
        TypeError
            If center_bin is not an integer type, or if hops is not an integer.

        See Also
        --------
        distance_to : Compute distance from each bin to target set.
        reachable_from : Find bins reachable from source within a radius.
        components : Find connected components of the graph.

        Notes
        -----
        **Hop Distance vs Physical Distance**:
        Rings are based on graph edges (hops), not physical distance. In a
        regular grid, 1 hop = 1 grid edge. In irregular graphs, hop distance
        may not correlate with Euclidean distance.

        **Disconnected Graphs**:
        If the center bin is in a disconnected component, rings will only
        cover bins in the same component. Bins in other components will never
        appear in any ring.

        **Ring Coverage**:
        The union of all rings equals the set of bins reachable from center
        within `hops` edges. Rings are mutually disjoint.

        **Performance**:
        Uses BFS with NetworkX. Complexity is O(E + V) where E = edges,
        V = vertices (bins). Very fast even for large graphs.

        Examples
        --------
        >>> import numpy as np
        >>> # Create 10x10 grid
        >>> data = np.array([[i, j] for i in range(10) for j in range(10)])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> # Get 2-hop neighborhood from center
        >>> rings_result = env.rings(center_bin=50, hops=2)
        >>> len(rings_result)
        3
        >>> len(rings_result[0])  # Center only
        1
        >>> len(rings_result[1]) > 0  # First neighbors
        True

        >>> # All rings are disjoint
        >>> all_bins = np.concatenate(rings_result)
        >>> len(all_bins) == len(np.unique(all_bins))
        True

        """
        # Type validation for center_bin
        if not isinstance(center_bin, (int, np.integer)):
            raise TypeError(
                f"center_bin must be an integer, got {type(center_bin).__name__}"
            )

        # Range validation for center_bin
        if center_bin < 0 or center_bin >= self.n_bins:
            raise ValueError(
                f"center_bin must be in range [0, {self.n_bins}), got {center_bin}"
            )

        # Type validation for hops
        if not isinstance(hops, (int, np.integer)):
            raise TypeError(f"hops must be an integer, got {type(hops).__name__}")

        # Validate hops is non-negative
        if hops < 0:
            raise ValueError(f"hops must be non-negative (>= 0), got {hops}")

        # Perform BFS to get shortest path lengths
        try:
            # nx.single_source_shortest_path_length returns dict: {node: distance}
            distances = nx.single_source_shortest_path_length(
                self.connectivity, center_bin, cutoff=hops
            )
        except nx.NetworkXError:
            # If center_bin has no edges (isolated), only ring 0 exists
            distances = {center_bin: 0}

        # Organize bins into rings by hop distance
        # Note: cutoff parameter already ensures dist <= hops, so all nodes are valid
        rings_lists: list[list[int]] = [[] for _ in range(hops + 1)]
        for node, dist in distances.items():
            rings_lists[dist].append(node)

        # Convert to numpy arrays
        rings_arrays: list[NDArray[np.int32]] = [
            np.array(ring, dtype=np.int32) for ring in rings_lists
        ]

        return rings_arrays
