"""Environment metrics and spatial properties.

This module provides methods for extracting metrics and properties from
spatial environments, including boundary detection, attribute extraction,
and metadata for 1D linearized environments.

Key Features
------------
- Boundary bin detection for identifying edge bins
- Attribute extraction to DataFrames for analysis
- Linearization metadata for 1D environments (GraphLayout)
- Cached properties for efficient repeated access

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking:

    from typing import TYPE_CHECKING, Protocol
    if TYPE_CHECKING:
        from neurospatial.environment._protocols import EnvironmentProtocol, SelfEnv

Then use string annotations in method signatures: `self: "Environment"`

Examples
--------
This class is not used directly. Instead, it's mixed into Environment:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>> data = np.random.rand(100, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Get boundary bins (from EnvironmentMetrics)
    >>> boundary = env.boundary_bins
    >>> print(f"Found {len(boundary)} boundary bins")
    Found ... boundary bins
    >>>
    >>> # Get bin attributes as DataFrame
    >>> df = env.get_bin_attributes()
    >>> print(df.columns.tolist())
    ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']

"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted, versioned_cached_property
from neurospatial.layout._locators import (
    HexLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout.helpers.utils import find_boundary_nodes


class EnvironmentMetrics:
    """Environment metrics and properties mixin.

    This mixin provides methods for extracting metrics and properties
    from spatial environments:
    - Boundary detection
    - Attribute extraction to DataFrames
    - Linearization properties

    All methods assume the Environment is fitted (has been initialized
    via a factory method).

    See Also
    --------
    Environment : Main class that uses this mixin
    EnvironmentQueries : Spatial query methods
    EnvironmentFields : Spatial field operations

    """

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def boundary_bins(self: SelfEnv) -> NDArray[np.intp]:
        """
        Identify boundary bins (bins on the edge of the active region).

        Boundary bins are active bins that have at least one inactive neighbor
        or are on the edge of the grid. Useful for identifying walls, barriers,
        or the perimeter of an arena.

        Returns
        -------
        NDArray[np.intp], shape (n_boundary_bins,)
            Array of bin IDs that are on the boundary.

        Notes
        -----
        - This is a **cached property** - computed once on first access, then cached
        - For large environments (>100,000 bins), initial computation may take a few seconds
        - Uses graph-based boundary detection (checks node degrees and connectivity)

        See Also
        --------
        EnvironmentQueries.neighbors : Get neighbors of a bin

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [2, 3]])
        >>> env = Environment.from_samples(data, bin_size=1.0)
        >>> boundary = env.boundary_bins
        >>> print(f"Found {len(boundary)} boundary bins")  # doctest: +SKIP
        Found 6 boundary bins

        """
        grid_shape, active_mask = self.grid_shape, self.active_mask
        locator = self._binning.locator
        selections = []
        while isinstance(locator, (SubsetLocator, TransformedLocator)):
            if isinstance(locator, SubsetLocator):
                selections.append(locator.parent_to_child)
            locator = locator.base
        if isinstance(locator, HexLocator):
            grid_shape = locator.centers_shape
            mapping = locator.flat_to_active.copy()
        elif isinstance(locator, RectilinearLocator):
            grid_shape = locator.shape
            mask = locator.active_mask
            mask = np.ones(grid_shape, dtype=bool) if mask is None else mask
            mapping = np.full(mask.size, -1, dtype=np.intp)
            mapping[mask.ravel()] = np.arange(np.count_nonzero(mask))
        else:
            mapping = None
        if mapping is not None:
            # Cell adjacency survives coordinate transforms. Compose subset
            # maps before deriving support, without indexing the -1 sentinel.
            for selection in reversed(selections):
                inside = mapping >= 0
                mapping[inside] = selection[mapping[inside]]
            active_mask = (mapping >= 0).reshape(grid_shape)
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=grid_shape,
            active_mask=active_mask,
            layout_kind=self._layout_type_used,
        ).astype(np.intp, copy=False)

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def linearization_properties(
        self: SelfEnv,
    ) -> dict[str, Any]:
        """
        Get linearization metadata for 1D environments.

        Returns a dictionary with linearization information, including:
        - `is_linearized_track`: Whether this is a 1D linearized environment
        - `linear_bin_edges`: Bin edges in linear coordinates (1D only)
        - `linear_bin_centers_1d`: Bin centers in linear coordinates (1D only)
        - `track_graph`: The underlying track graph (1D only)

        Returns
        -------
        dict[str, Any]
            Dictionary with linearization metadata. For non-1D environments,
            only contains `{'is_linearized_track': False}`.

        Notes
        -----
        - This is a **cached property** - computed once on first access, then cached
        - Only 1D environments (created with `from_graph()`) have full linearization info
        - For N-D environments, returns `{'is_linearized_track': False}`

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        linear_to_nd : Convert 1D linear coordinates to N-D positions
        Environment.from_graph : Create 1D linearized environment

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()  # doctest: +SKIP
        >>> G.add_node(0, pos=(0.0, 0.0))  # doctest: +SKIP
        >>> G.add_node(1, pos=(10.0, 0.0))  # doctest: +SKIP
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)  # doctest: +SKIP
        >>> env = Environment.from_graph(
        ...     G, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=2.0
        ... )  # doctest: +SKIP
        >>>
        >>> props = env.linearization_properties  # doctest: +SKIP
        >>> print(props["is_linearized_track"])  # doctest: +SKIP
        True

        """
        if not self._is_linearized_track_env:
            return {"is_linearized_track": False}

        # For 1D environments, extract linearization metadata from layout
        metadata: dict[str, Any] = {"is_linearized_track": True}
        if hasattr(self.layout, "linear_bin_edges"):
            metadata["linear_bin_edges"] = self.layout.linear_bin_edges
        if hasattr(self.layout, "linear_bin_centers_1d"):
            metadata["linear_bin_centers_1d"] = self.layout.linear_bin_centers_1d
        if hasattr(self.layout, "track_graph"):
            metadata["track_graph"] = self.layout.track_graph

        return metadata

    @check_fitted
    def get_bin_attributes(self: SelfEnv) -> pd.DataFrame:
        """
        Extract bin (node) attributes as a pandas DataFrame.

        Returns a DataFrame where each row corresponds to a bin, with columns
        for bin attributes like positions, grid indices, and any custom metadata.

        Returns
        -------
        pd.DataFrame, shape (n_bins, n_attributes)
            DataFrame with bin attributes. Always includes:
            - `'source_grid_flat_index'`: Flat index in original grid
            - `'original_grid_nd_index'`: N-D grid index tuple
            - `'pos_dim0'`, `'pos_dim1'`, ...: Position coordinates

        Notes
        -----
        - Method (not property) since v0.4: the construction cost
          (one DataFrame allocation per bin) is now visible at the
          call site. Repeated calls within one environment state
          version are O(1) — the result is cached internally via
          :class:`versioned_cached_property` and invalidates whenever
          the environment is rebuilt.
        - For large environments (>100,000 bins), the first call may
          take a few seconds.
        - Memory usage: ~8 bytes per attribute per bin (e.g., 10
          attributes × 10k bins = 800 KB).

        See Also
        --------
        get_edge_attributes : Extract edge attributes as a DataFrame.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.get_bin_attributes()
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(df.shape)  # doctest: +SKIP
        (30, 4)

        """
        return self._bin_attributes_cached  # type: ignore[attr-defined]

    @versioned_cached_property
    def _bin_attributes_cached(self: SelfEnv) -> pd.DataFrame:
        # Extract node attributes from connectivity graph
        node_attrs = []
        for node_id in self.connectivity.nodes():
            attrs = self.connectivity.nodes[node_id].copy()
            attrs["node_id"] = node_id
            node_attrs.append(attrs)

        df = pd.DataFrame(node_attrs)

        # Expand position tuple into separate columns
        if "pos" in df.columns:
            pos_array = np.array(df["pos"].tolist())
            for dim in range(pos_array.shape[1]):
                df[f"pos_dim{dim}"] = pos_array[:, dim]
            df = df.drop(columns=["pos"])

        return df.set_index("node_id")

    @check_fitted
    def get_edge_attributes(self: SelfEnv) -> pd.DataFrame:
        """
        Extract edge attributes as a pandas DataFrame.

        Returns a DataFrame where each row corresponds to an edge (connection
        between bins), with columns for edge attributes like distances, vectors,
        and any custom metadata.

        Returns
        -------
        pd.DataFrame, shape (n_edges, n_attributes)
            DataFrame with edge attributes. Always includes:
            - `'source'`: Source bin ID
            - `'target'`: Target bin ID
            - `'distance'`: Euclidean distance between bin centers
            - `'vector'`: Displacement vector tuple
            - `'edge_id'`: Unique edge ID
            - `'angle_2d'`: Angle for 2D layouts (optional)

        Notes
        -----
        - Method (not property) since v0.4: edge-DataFrame allocation is
          O(n_edges) and visible at the call site. Repeated calls within
          one environment state version are O(1) — the result is cached
          internally via :class:`versioned_cached_property` and
          invalidates whenever the environment is rebuilt.
        - For large environments (>100,000 edges), the first call may
          take a few seconds.
        - Memory usage: ~8 bytes per attribute per edge (e.g., 6
          attributes × 10k edges = 480 KB).

        See Also
        --------
        get_bin_attributes : Extract bin attributes as a DataFrame.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.get_edge_attributes()
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source', 'target', 'distance', 'vector', 'edge_id', 'angle_2d']
        >>> print(df.shape)  # doctest: +SKIP
        (100, 6)

        """
        return self._edge_attributes_cached  # type: ignore[attr-defined]

    @versioned_cached_property
    def _edge_attributes_cached(self: SelfEnv) -> pd.DataFrame:
        # Extract edge attributes from connectivity graph
        edge_attrs = []
        for source, target in self.connectivity.edges():
            attrs = self.connectivity.edges[source, target].copy()
            attrs["source"] = source
            attrs["target"] = target
            edge_attrs.append(attrs)

        df = pd.DataFrame(edge_attrs)

        # Expand vector tuple into separate columns
        if "vector" in df.columns and len(df) > 0:
            vector_array = np.array(df["vector"].tolist())
            for dim in range(vector_array.shape[1]):
                df[f"vector_dim{dim}"] = vector_array[:, dim]
            # Keep original vector column for backward compatibility

        return df
