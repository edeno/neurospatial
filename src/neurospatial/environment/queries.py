"""Spatial query methods for Environment class.

This module provides the EnvironmentQueries mixin class containing methods
for querying spatial properties of an environment:

- bin_at: Map continuous points to discrete bin indices
- contains: Check if points fall within any active bin
- bin_center_of: Get center coordinates of bins by index
- neighbors: Find neighboring bins for a given bin
- bin_sizes: Get area/volume of each active bin (cached property)
- distance_between: Calculate geodesic distance between two points
- shortest_path: Find shortest path between two bins
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.environment.decorators import check_fitted

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


class EnvironmentQueries:
    """Mixin class providing spatial query methods for Environment.

    This class is NOT a dataclass. It provides methods that query spatial
    properties of an Environment instance, such as mapping points to bins,
    finding neighbors, and computing distances.

    Methods in this class assume they are mixed into an Environment instance
    that provides: layout, bin_centers, connectivity, is_1d attributes.
    """

    @check_fitted
    def bin_at(self: Environment, points_nd: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map N-dimensional continuous points to discrete active bin indices.

        This method delegates to the `point_to_bin_index` method of the
        underlying `LayoutEngine`.

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
        return self.layout.point_to_bin_index(points_nd)  # type: ignore[no-any-return]

    @check_fitted
    def contains(
        self: Environment, points_nd: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
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
        self: Environment,
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
    def neighbors(self: Environment, bin_index: int) -> list[int]:
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
    def bin_sizes(self: Environment) -> NDArray[np.float64]:
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
        return self.layout.bin_sizes()  # type: ignore[no-any-return]

    def distance_between(
        self: Environment,
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
    def shortest_path(
        self: Environment,
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
        >>> path = env.shortest_path(0, 5)
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
