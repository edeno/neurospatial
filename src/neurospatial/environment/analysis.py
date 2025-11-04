"""Analysis methods for Environment class.

This module provides methods for analyzing and extracting information from
spatial environments, including boundary detection, attribute extraction,
and coordinate transformations.

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

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from neurospatial.environment.core import Environment

Then use string annotations in method signatures: `self: "Environment"`

Examples
--------
This class is not used directly. Instead, it's mixed into Environment:

    >>> from neurospatial import Environment
    >>> import numpy as np
    >>> data = np.random.rand(100, 2) * 10
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>>
    >>> # Get boundary bins (from EnvironmentAnalysis)
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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurospatial.environment.decorators import check_fitted
from neurospatial.layout.helpers.utils import find_boundary_nodes

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


class EnvironmentAnalysis:
    """Analysis methods mixin for Environment.

    This mixin provides methods for analyzing spatial environments:
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
    EnvironmentVisualization : Plotting methods

    """

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def boundary_bins(self: Environment) -> NDArray[np.int_]:
        """Get the boundary bin indices.

        Returns
        -------
        NDArray[np.int_], shape (n_boundary_bins,)
            An array of indices of the boundary bins in the environment.
            These are the bins that are at the edges of the active area.

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> boundary = env.boundary_bins
        >>> print(f"Number of boundary bins: {len(boundary)}")
        Number of boundary bins: ...

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
        self: Environment,
    ) -> dict[str, Any] | None:
        """If the environment uses a GraphLayout, returns properties needed
        for linearization (converting a 2D/3D track to a 1D line) using the
        `track_linearization` library.

        These properties are typically passed to `track_linearization.get_linearized_position`.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary with keys 'track_graph', 'edge_order', 'edge_spacing'
            if the layout is `GraphLayout` and parameters are available.
            Individual values may be None if not available in layout parameters.
            Returns `None` for non-1D environments.

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # props = env.linearization_properties
        >>> # if props is not None:
        >>> #     print(props.keys())
        >>> #     dict_keys(['track_graph', 'edge_order', 'edge_spacing'])

        Notes
        -----
        This property returns None for non-1D environments or environments
        not created from graph-based layouts.

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if hasattr(self.layout, "to_linear") and hasattr(self.layout, "linear_to_nd"):
            return {
                "track_graph": self._layout_params_used.get("graph_definition"),
                "edge_order": self._layout_params_used.get("edge_order"),
                "edge_spacing": self._layout_params_used.get("edge_spacing"),
            }
        return None

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def bin_attributes(self: Environment) -> pd.DataFrame:
        """Build a DataFrame of attributes for each active bin (node) in the environment's graph.

        Returns
        -------
        df : pandas.DataFrame
            Rows are indexed by `active_bin_id` (int), matching 0..(n_bins-1).
            Columns correspond to node attributes. If a 'pos' attribute exists
            for any node and is non-null, it will be expanded into columns
            'pos_dim0', 'pos_dim1', ..., with numeric coordinates.

        Raises
        ------
        ValueError
            If there are no active bins (graph has zero nodes).

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        For very large environments (>100,000 bins), this method converts
        the entire connectivity graph to a DataFrame which may consume
        significant memory. Consider querying specific attributes from
        self.connectivity directly if you only need a subset.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> df = env.bin_attributes
        >>> print(df.columns.tolist())
        ['source_grid_flat_index', 'original_grid_nd_index', 'pos_dim0', 'pos_dim1']
        >>> print(f"Shape: {df.shape}")
        Shape: (..., 4)

        """
        graph = self.connectivity
        if graph.number_of_nodes() == 0:
            raise ValueError("No active bins in the environment.")

        df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        df.index.name = "active_bin_id"  # Index is 0..N-1

        if "pos" in df.columns and not df["pos"].dropna().empty:
            pos_df = pd.DataFrame(df["pos"].tolist(), index=df.index)
            pos_df.columns = [f"pos_dim{i}" for i in range(pos_df.shape[1])]
            df = pd.concat([df.drop(columns="pos"), pos_df], axis=1)

        return df

    @cached_property
    @check_fitted  # Check only on first access, then value is cached
    def edge_attributes(self: Environment) -> pd.DataFrame:
        """Return a Pandas DataFrame where each row corresponds to one directed edge
        (u → v) in the connectivity graph, and columns include all stored edge
        attributes (e.g. 'distance', 'vector', 'weight', 'angle_2d', etc.).

        The DataFrame will have a MultiIndex of (source_bin, target_bin). If you
        prefer flat columns, you can reset the index.

        Returns
        -------
        pd.DataFrame
            A DataFrame whose index is a MultiIndex (source_bin, target_bin),
            and whose columns are the union of all attribute-keys stored on each edge.

        Raises
        ------
        ValueError
            If there are no edges in the connectivity graph.
        RuntimeError
            If called before the environment is fitted.

        Notes
        -----
        This property is cached after first access. The fitted check only
        runs on the first computation, not on subsequent cached accesses.

        For very large environments (>100,000 edges), this method converts
        the entire connectivity graph to a DataFrame which may consume
        significant memory. Consider querying specific attributes from
        self.connectivity directly if you only need a subset.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> data = np.random.rand(100, 2) * 10
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> df = env.edge_attributes
        >>> print(df.columns.tolist())
        ['distance', 'vector', 'edge_id', 'angle_2d']
        >>> print(f"Number of edges: {len(df)}")
        Number of edges: ...

        """
        graph = self.connectivity
        if graph.number_of_edges() == 0:
            raise ValueError("No edges in the connectivity graph.")

        # Build a dict of edge_attr_dicts keyed by (u, v)
        # networkx's graph.edges(data=True) yields (u, v, attr_dict)
        edge_dict: dict[tuple[int, int], dict] = {
            (u, v): data.copy() for u, v, data in graph.edges(data=True)
        }

        # Convert that to a DataFrame, using the (u, v) tuples as a MultiIndex
        df = pd.DataFrame.from_dict(edge_dict, orient="index")
        # The index is now a MultiIndex of (u, v)
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=["source_bin", "target_bin"],
        )

        return df

    @check_fitted
    def to_linear(
        self: Environment, points_nd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert N-dimensional points to 1D linearized coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `to_linear` method.

        Parameters
        ----------
        points_nd : NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates corresponding to the input points.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # import numpy as np
        >>> # points = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> # linear_pos = env.to_linear(points)
        >>> # print(linear_pos)
        >>> # [0.5 1.2]  # Example output

        Notes
        -----
        This method requires that the environment was created using a GraphLayout
        (typically via `Environment.from_graph()`). For N-D grid environments,
        this method will raise a TypeError.

        See Also
        --------
        linear_to_nd : Convert linearized coordinates back to N-D
        is_1d : Property indicating if environment is 1D
        linearization_properties : Properties needed for linearization

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "to_linear"):
            raise TypeError(
                "Linearization is only available for 1D environments (GraphLayout). "
                f"This environment has is_1d={self.is_1d}. "
                "Use Environment.from_graph() to create a 1D environment."
            )
        result = self.layout.to_linear(points_nd)
        return np.asarray(result, dtype=np.float64)

    @check_fitted
    def linear_to_nd(
        self: Environment,
        linear_coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert 1D linearized coordinates back to N-dimensional coordinates.

        This method is only applicable if the environment uses a `GraphLayout`
        and `is_1d` is True. It delegates to the layout's
        `linear_to_nd` method.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map to N-D space.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-dimensional coordinates corresponding to the input linear coordinates.

        Raises
        ------
        TypeError
            If the environment is not 1D or not based on a `GraphLayout`.
        RuntimeError
            If called before the environment is fitted.

        Examples
        --------
        >>> from neurospatial import Environment
        >>> # Create a 1D environment (requires GraphLayout)
        >>> # env = Environment.from_graph(...)  # Requires track-linearization
        >>> # import numpy as np
        >>> # linear_pos = np.array([0.5, 1.2])
        >>> # points = env.linear_to_nd(linear_pos)
        >>> # print(points)
        >>> # [[1.0 2.0]
        >>> #  [3.0 4.0]]  # Example output

        Notes
        -----
        This method requires that the environment was created using a GraphLayout
        (typically via `Environment.from_graph()`). For N-D grid environments,
        this method will raise a TypeError.

        See Also
        --------
        to_linear : Convert N-D points to linearized coordinates
        is_1d : Property indicating if environment is 1D
        linearization_properties : Properties needed for linearization

        """
        # Use hasattr instead of isinstance to avoid Protocol/concrete class conflict
        if not self.is_1d or not hasattr(self.layout, "linear_to_nd"):
            raise TypeError(
                "Linearization is only available for 1D environments (GraphLayout). "
                f"This environment has is_1d={self.is_1d}. "
                "Use Environment.from_graph() to create a 1D environment."
            )
        result = self.layout.linear_to_nd(linear_coordinates)
        return np.asarray(result, dtype=np.float64)
