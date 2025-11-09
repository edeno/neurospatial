"""Environment metrics and spatial properties.

This module provides methods for extracting metrics and properties from
spatial environments, including boundary detection, attribute extraction,
and coordinate transformations for 1D linearized environments.

Key Features
------------
- Boundary bin detection for identifying edge bins
- Attribute extraction to DataFrames for analysis
- Linearization support for 1D environments (GraphLayout)
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
    >>> df = env.bin_attributes
    >>> print(df.columns.tolist())
    ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']

"""

from __future__ import annotations

from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.environment._protocols import SelfEnv
from neurospatial.environment.decorators import check_fitted
from neurospatial.layout.helpers.utils import find_boundary_nodes


class EnvironmentMetrics:
    """Environment metrics and properties mixin.

    This mixin provides methods for extracting metrics and properties
    from spatial environments:
    - Boundary detection
    - Attribute extraction to DataFrames
    - Linearization (for 1D environments)
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
    def boundary_bins(self: SelfEnv) -> NDArray[np.int_]:
        """
        Identify boundary bins (bins on the edge of the active region).

        Boundary bins are active bins that have at least one inactive neighbor
        or are on the edge of the grid. Useful for identifying walls, barriers,
        or the perimeter of an arena.

        Returns
        -------
        NDArray[np.int_], shape (n_boundary_bins,)
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
        return find_boundary_nodes(
            graph=self.connectivity,
            grid_shape=self.grid_shape,
            active_mask=self.active_mask,
            layout_kind=self._layout_type_used,
        )

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def linearization_properties(
        self: SelfEnv,
    ) -> dict[str, Any]:
        """
        Get linearization metadata for 1D environments.

        Returns a dictionary with linearization information, including:
        - `is_1d`: Whether this is a 1D linearized environment
        - `linear_bin_edges`: Bin edges in linear coordinates (1D only)
        - `linear_bin_centers_1d`: Bin centers in linear coordinates (1D only)
        - `track_graph`: The underlying track graph (1D only)

        Returns
        -------
        dict[str, Any]
            Dictionary with linearization metadata. For non-1D environments,
            only contains `{'is_1d': False}`.

        Notes
        -----
        - This is a **cached property** - computed once on first access, then cached
        - Only 1D environments (created with `from_graph()`) have full linearization info
        - For N-D environments, returns `{'is_1d': False}`

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
        >>> G = nx.Graph()
        >>> G.add_node(0, pos=(0.0, 0.0))
        >>> G.add_node(1, pos=(10.0, 0.0))
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)
        >>> env = Environment.from_graph(G, edge_order=[(0, 1)], bin_size=2.0)
        >>>
        >>> props = env.linearization_properties
        >>> print(props["is_1d"])
        True

        """
        if not self._is_1d_env:
            return {"is_1d": False}

        # For 1D environments, extract linearization metadata from layout
        metadata: dict[str, Any] = {"is_1d": True}
        if hasattr(self.layout, "linear_bin_edges"):
            metadata["linear_bin_edges"] = self.layout.linear_bin_edges
        if hasattr(self.layout, "linear_bin_centers_1d"):
            metadata["linear_bin_centers_1d"] = self.layout.linear_bin_centers_1d
        if hasattr(self.layout, "track_graph"):
            metadata["track_graph"] = self.layout.track_graph

        return metadata

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def bin_attributes(self: SelfEnv) -> pd.DataFrame:
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
        - This is a **cached property** - computed once on first access, then cached
        - For large environments (>100,000 bins), initial computation may take a few seconds
        - Memory usage: ~8 bytes per attribute per bin (e.g., 10 attributes × 10k bins = 800 KB)

        See Also
        --------
        edge_attributes : Extract edge attributes as a DataFrame

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.bin_attributes
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(df.shape)  # doctest: +SKIP
        (30, 4)

        """
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

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def edge_attributes(self: SelfEnv) -> pd.DataFrame:
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
        - This is a **cached property** - computed once on first access, then cached
        - For large environments (>100,000 edges), initial computation may take a few seconds
        - Memory usage: ~8 bytes per attribute per edge (e.g., 6 attributes × 10k edges = 480 KB)

        See Also
        --------
        bin_attributes : Extract bin attributes as a DataFrame

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(50, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>>
        >>> df = env.edge_attributes
        >>> print(df.columns.tolist())  # doctest: +SKIP
        ['source', 'target', 'distance', 'vector', 'edge_id', 'angle_2d']
        >>> print(df.shape)  # doctest: +SKIP
        (100, 6)

        """
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

    @check_fitted
    def to_linear(
        self: SelfEnv,
        nd_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert N-D positions to 1D linear coordinates.

        For 1D linearized environments (created with `from_graph()`), this
        converts N-D positions (e.g., 2D x,y coordinates) to their corresponding
        1D linear coordinates along the linearized track.

        Parameters
        ----------
        nd_position : NDArray[np.floating], shape (n_samples, n_dims)
            N-D position array in the original coordinate space.
            For 2D environments, shape is (n_samples, 2).

        Returns
        -------
        NDArray[np.floating], shape (n_samples,)
            1D linear coordinates corresponding to the input positions.

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_1d` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - For N-D environments, use `bin_at()` to map positions to bins instead
        - Linear coordinates start at 0 and increase monotonically along the track

        See Also
        --------
        linear_to_nd : Convert 1D linear coordinates back to N-D positions
        Environment.from_graph : Create 1D linearized environment
        is_1d : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()
        >>> G.add_node(0, pos=(0.0, 0.0))
        >>> G.add_node(1, pos=(10.0, 0.0))
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)
        >>> env = Environment.from_graph(G, edge_order=[(0, 1)], bin_size=2.0)
        >>>
        >>> # Convert 2D position to 1D linear coordinate
        >>> position_2d = np.array([[5.0, 0.0]])  # Halfway along track
        >>> position_1d = env.to_linear(position_2d)
        >>> print(f"Linear position: {position_1d[0]:.1f}")  # doctest: +SKIP
        Linear position: 5.0

        """
        if not self._is_1d_env:
            msg = (
                "to_linear() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_at() to map positions to bins for N-D environments."
            )
            raise AttributeError(msg)

        # Delegate to layout's linearization method
        if not hasattr(self.layout, "to_linear"):
            msg = (
                "Layout does not support linearization. "
                "This should not happen for GraphLayout - please report this bug."
            )
            raise AttributeError(msg)

        # Cast to Any to work around Protocol limitation (GraphLayout has this method)
        return cast(
            "NDArray[np.float64]", cast("Any", self.layout).to_linear(nd_position)
        )

    @check_fitted
    def linear_to_nd(
        self: SelfEnv,
        linear_position: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Convert 1D linear coordinates to N-D positions.

        For 1D linearized environments (created with `from_graph()`), this
        converts 1D linear coordinates back to the original N-D coordinate
        space (e.g., 2D x,y coordinates).

        Parameters
        ----------
        linear_position : NDArray[np.floating], shape (n_samples,)
            1D linear coordinate array.

        Returns
        -------
        NDArray[np.floating], shape (n_samples, n_dims)
            N-D positions corresponding to the input linear coordinates.
            For 2D environments, shape is (n_samples, 2).

        Raises
        ------
        AttributeError
            If this environment is not 1D (does not support linearization).
            Check `env.is_1d` before calling this method.

        Notes
        -----
        - **Only works for 1D environments** created with `Environment.from_graph()`
        - Inverse operation of `to_linear()`
        - Useful for plotting 1D data in the original 2D space

        See Also
        --------
        to_linear : Convert N-D positions to 1D linear coordinates
        Environment.from_graph : Create 1D linearized environment
        is_1d : Check if environment supports linearization

        Examples
        --------
        >>> import numpy as np
        >>> import networkx as nx
        >>> from neurospatial import Environment
        >>>
        >>> # Create a simple 1D track
        >>> G = nx.Graph()
        >>> G.add_node(0, pos=(0.0, 0.0))
        >>> G.add_node(1, pos=(10.0, 0.0))
        >>> G.add_edge(0, 1, edge_id=0, distance=10.0)
        >>> env = Environment.from_graph(G, edge_order=[(0, 1)], bin_size=2.0)
        >>>
        >>> # Convert 1D linear coordinate to 2D position
        >>> linear_pos = np.array([5.0])  # Halfway along track
        >>> position_2d = env.linear_to_nd(linear_pos)
        >>> print(f"2D position: {position_2d[0]}")  # doctest: +SKIP
        2D position: [5. 0.]

        """
        if not self._is_1d_env:
            msg = (
                "linear_to_nd() is only available for 1D environments (GraphLayout). "
                f"This environment is {self.n_dims}D. "
                "Use bin_center_of() to get bin centers for N-D environments."
            )
            raise AttributeError(msg)

        # Delegate to layout's linearization method
        if not hasattr(self.layout, "linear_to_nd"):
            msg = (
                "Layout does not support linearization. "
                "This should not happen for GraphLayout - please report this bug."
            )
            raise AttributeError(msg)

        # Cast to Any to work around Protocol limitation (GraphLayout has this method)
        return cast(
            "NDArray[np.float64]",
            cast("Any", self.layout).linear_to_nd(linear_position),
        )
