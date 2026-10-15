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

import copy
from collections.abc import Mapping
from typing import Any, cast

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


def _copy_export(value: Any) -> Any:
    """Copy nested export values, including immutable mapping snapshots."""
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _copy_export(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_export(item) for item in value)
    if isinstance(value, list):
        return [_copy_export(item) for item in value]
    return copy.deepcopy(value)


def _copy_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach frame buffers, labels and nested object-column values."""
    result = frame.copy(deep=True)
    result.index = frame.index.copy(deep=True)
    result.columns = frame.columns.copy(deep=True)
    for column in frame.select_dtypes(include="object"):
        result[column] = pd.Series(
            [_copy_export(value) for value in frame[column]],
            index=result.index,
            dtype=object,
        )
    return result


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

    @property
    @check_fitted
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
        - Detection is cached; each access returns an independent array.
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
        cached = cast("NDArray[np.intp]", self._boundary_bins_cached)  # type: ignore[attr-defined]
        return cast("NDArray[np.intp]", cached.copy())

    @versioned_cached_property
    def _boundary_bins_cached(self: SelfEnv) -> NDArray[np.intp]:
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

    @property
    @check_fitted
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
        - Metadata is cached; each access returns an independent dictionary.
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
        return cast(
            "dict[str, Any]",
            _copy_export(self._linearization_properties_cached),  # type: ignore[attr-defined]
        )

    @versioned_cached_property
    def _linearization_properties_cached(self: SelfEnv) -> dict[str, Any]:
        if not self._is_linearized_track_env:
            return {"is_linearized_track": False}

        # The owned public value never exposed mutable builder-only fields.
        return {"is_linearized_track": True}

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
          call site. Attribute extraction is cached internally via
          :class:`versioned_cached_property`; each call returns an independent
          DataFrame, including nested object values.
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
        return _copy_attributes(self._bin_attributes_cached)  # type: ignore[attr-defined]

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
            - `'distance'`: Stored navigation distance between bins
            - `'vector'`: Displacement vector tuple
            - `'edge_id'`: Unique edge ID
            - `'angle_2d'`: Angle for 2D layouts (optional)

        Notes
        -----
        - Method (not property) since v0.4: edge-DataFrame allocation is
          O(n_edges) and visible at the call site. Attribute extraction is
          cached internally via :class:`versioned_cached_property`; each call
          returns an independent DataFrame, including nested object values.
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
        return _copy_attributes(self._edge_attributes_cached)  # type: ignore[attr-defined]

    @versioned_cached_property
    def _edge_attributes_cached(self: SelfEnv) -> pd.DataFrame:
        # Extract edge attributes from connectivity graph
        edge_attrs = []
        centers = self.bin_centers
        graph_orientation = {
            frozenset((source, target)): (source, target)
            for source, target in self.connectivity.edges()
        }
        for source, target in self._binning.bins.pairs:
            attrs = self.connectivity.edges[source, target].copy()
            if graph_orientation[frozenset((source, target))] != (source, target):
                # Undirected graph insertion order does not establish the
                # orientation of retained attributes. Derive it from the pair
                # instead of blindly negating a possibly already aligned vector.
                vector = centers[target] - centers[source]
                vector[vector == 0.0] = 0.0
                attrs["vector"] = tuple(vector)
                if "angle_2d" in attrs:
                    attrs["angle_2d"] = float(np.arctan2(vector[1], vector[0]))
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
