from collections.abc import Sequence
from typing import Any, ClassVar

import matplotlib
import matplotlib.axes
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry._binning import Binning, TrackLinearization
from neurospatial.geometry._face_builders import track_geometry, track_junctions
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._plotting import plot_track, plot_track_linear
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.graph import (
    _create_graph_layout_connectivity_graph,
    _find_bin_for_linear_position,
    _get_graph_bins,
    _project_1d_to_2d,
)
from neurospatial.layout.mixins import _KDTreeMixin
from neurospatial.layout.validation import validate_connectivity_graph


class GraphLayout(_KDTreeMixin):
    """Layout defined by a user-provided graph, typically for 1D tracks.

    The graph's nodes (with 'pos' attributes) and a specified edge order
    are used to create a linearized representation of the space, which is
    then binned. Connectivity is derived from this binned structure.
    Uses `_KDTreeMixin` for point mapping and neighbor finding on the
    N-D embeddings of the linearized bin centers.
    """

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    # Build arguments that are input data, not choices; they are left out
    # of the environment's recorded layout parameters.
    _data_parameters: ClassVar[frozenset[str]] = frozenset(
        {"graph_definition", "edge_order"}
    )
    _layout_type_tag: str
    _build_params_used: dict[str, Any]

    # Layout Specific
    linear_bin_centers_: NDArray[np.float64] | None = None

    def __init__(self) -> None:
        """Initialize a GraphLayout engine."""
        self._layout_type_tag = "Graph"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.linear_bin_centers_ = None
        self._linearization: TrackLinearization | None = None

    @property
    def layout_type(self) -> str:
        """Return standardized category for this layout type."""
        return "graph"

    @property
    def is_grid_compatible(self) -> bool:
        """Return False - graph layouts cannot be rendered as 2D images."""
        return False

    @capture_build_params
    def build(
        self,
        *,
        graph_definition: nx.Graph,  # Original user-provided graph
        edge_order: list[tuple[Any, Any]],
        edge_spacing: float | Sequence[float],
        bin_size: float,  # Linearized bin size
    ) -> None:
        """Build the graph-based (linearized track) layout.

        Parameters
        ----------
        graph_definition : nx.Graph
            The original NetworkX graph. Nodes must have a 'pos' attribute
            (e.g., `(x, y)` coordinates) and edges should ideally have a
            'distance' attribute if not relying on Euclidean distance calculation.
            ``edge_id`` is numbered by ``graph.edges()`` order on an internal
            copy; any ``edge_id`` on the input graph is ignored.
        edge_order : List[Tuple[Any, Any]]
            An ordered sequence of edge tuples (node_id_1, node_id_2) from
            `graph_definition` that defines the ordering of edges in the
            linear space.
        edge_spacing : Union[float, Sequence[float]]
            Spacing (gap) to insert between consecutive edges in `edge_order`
            during linearization. If float, same gap for all. If sequence,
            specifies each gap; length must be `len(edge_order) - 1`.
        bin_size : float
            The desired length of each bin along the linearized space.

        Raises
        ------
        TypeError
            If `graph_definition` is not a NetworkX graph.
        ValueError
            If `edge_order` is empty or `bin_size` is not positive.

        """
        if not isinstance(graph_definition, nx.Graph):
            raise TypeError("graph_definition must be a NetworkX graph.")
        if not edge_order:  # Empty edge_order means no path to linearize
            raise ValueError("edge_order must not be empty.")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive.")

        # track_linearization finds a point's nearest segment by its position in
        # ``graph.edges()`` but looks that segment up by its ``edge_id`` attribute,
        # so the two must coincide. Linearize a copy numbered by enumeration; the
        # caller's graph is never mutated.
        track_graph = graph_definition.copy()
        for enumeration_index, (u, v) in enumerate(track_graph.edges()):
            track_graph.edges[u, v]["edge_id"] = enumeration_index
        self._build_params_used["graph_definition"] = track_graph

        (linear_bin_centers, self.grid_edges, self.active_mask, edge_ids) = (
            _get_graph_bins(
                graph=track_graph,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )

        self.linear_bin_centers_ = linear_bin_centers[self.active_mask]
        self.bin_centers = _project_1d_to_2d(
            self.linear_bin_centers_,
            track_graph,
            edge_order,
            edge_spacing,
        )
        self.grid_shape = (len(self.grid_edges[0]) - 1,)
        self.connectivity = _create_graph_layout_connectivity_graph(
            graph=track_graph,
            bin_centers_nd=self.bin_centers,
            linear_bin_centers=self.linear_bin_centers_,
            original_edge_ids=edge_ids,
            edge_order=edge_order,
        )

        # Compute dimension_ranges dynamically based on actual dimensionality
        n_dims = self.bin_centers.shape[1]

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(self.connectivity, n_dims=n_dims)
        self.dimension_ranges = tuple(
            (
                np.min(self.bin_centers[:, dim]),
                np.max(self.bin_centers[:, dim]),
            )
            for dim in range(n_dims)
        )

        # --- Build KDTree ---
        self._build_kdtree(points_for_tree=self.bin_centers)
        self._linearization = TrackLinearization(
            track_graph, edge_order, edge_spacing, self.grid_edges, self.active_mask
        )

    @property
    def is_linearized_track(self) -> bool:
        """Graph layouts are treated as 1-dimensional due to linearization.

        Returns
        -------
        bool
            Always True, indicating a 1D linearized layout.

        """
        return True

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the N-D embedding of the graph-based layout.

        Displays the original graph used for definition, the N-D positions of
        the binned track segments (active bin centers), and their connectivity.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `figsize` (tuple): Figure size if `ax` is None.
            - `node_kwargs` (dict): Kwargs for plotting original graph nodes.
            - `edge_kwargs` (dict): Kwargs for plotting original graph edges.
            - `bin_node_kwargs` (dict): Kwargs for plotting active bin center nodes.
            - `bin_edge_kwargs` (dict): Kwargs for plotting connectivity graph edges.
            - `show_bin_edges` (bool): Whether to project and plot 1D bin edges in N-D.


        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.

        """
        return plot_track(
            ax,
            title=self._layout_type_tag,
            linearization=self.linearization(),
            bin_centers=self.bin_centers,
            connectivity=self.connectivity,
            grid_edges=self.grid_edges,
            **kwargs,
        )

    def plot_linear_layout(
        self,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the 1D linearized representation of the graph layout.

        Uses `track_linearization.plot_graph_as_1D` to display the track
        segments and nodes in their 1D linearized positions. Overlays the
        1D bin edges from `self.grid_edges`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments passed to
            `track_linearization.plot_graph_as_1D` and for customizing
            the appearance of bin edge lines.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the 1D layout is plotted.

        """
        return plot_track_linear(
            ax,
            title=self._layout_type_tag,
            linearization=self.linearization(),
            grid_edges=self.grid_edges,
            **kwargs,
        )

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N-D points to 1D linearized coordinates along the track.

        Uses `track_linearization.get_linearized_position`.

        Parameters
        ----------
        data_points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to linearize.

        Returns
        -------
        NDArray[np.float64], shape (n_points,)
            1D linearized coordinates. NaNs may be returned for points
            far from the track.

        """
        return self.linearization().to_linear(data_points)

    def linear_to_nd(
        self,
        linear_coordinates: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map 1D linearized coordinates back to N-D coordinates on the track graph.

        Parameters
        ----------
        linear_coordinates : NDArray[np.float64], shape (n_points,)
            1D linearized coordinates to map.

        Returns
        -------
        NDArray[np.float64], shape (n_points, n_dims)
            N-D coordinates corresponding to the input linear positions.

        """
        return self.linearization().linear_to_nd(linear_coordinates)

    def linearization(self) -> TrackLinearization:
        """Return the owned track graph and linear coordinate system.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        if self._linearization is None:
            raise LayoutNotBuiltError(type(self).__name__, "linearization")
        return self._linearization

    def binning(self) -> Binning:
        """Return owned track bins, lookup, linearization and lazy track faces.

        Faces have unit cross-section. A connection between bins on different
        track segments uses the along-track distance through their shared
        junction node as its flux distance.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        if self.connectivity is None:
            raise LayoutNotBuiltError(type(self).__name__, "connectivity")
        track = self.linearization()
        return assemble_binning(
            self.connectivity,
            kind=self._layout_type_tag,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=1,
            locator=self.locator(),
            faces="track",
            face_arrays=(
                track_junctions(
                    self.connectivity, track.graph, self.bin_centers.shape[1]
                ),
            ),
            face_normals=False,
            linearization=track,
            dimension_ranges=self.dimension_ranges,
        )

    def linear_point_to_bin_ind(
        self,
        data_points: NDArray[np.float64],
    ) -> NDArray[np.int_]:
        """Map 1D linearized positions to active 1D bin indices.

        Parameters
        ----------
        data_points : NDArray[np.float64], shape (n_points,)
            1D linearized positions.

        Returns
        -------
        NDArray[np.int_], shape (n_points,)
            Indices of the active 1D bins corresponding to each linear position.
            Returns -1 for positions outside active bins or in gaps.
            Note: These are indices relative to the set of *active* 1D bins,
            not indices into the full `linear_bin_centers_all` array.

        """
        if self.grid_edges is None or self.active_mask is None:
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in ("grid_edges", "active_mask")
                    if getattr(self, name) is None
                ),
            )

        full_grid_ind = _find_bin_for_linear_position(
            data_points,
            bin_edges=self.grid_edges[0],
            active_mask=self.active_mask,
        )
        full_grid_ind = np.atleast_1d(np.asarray(full_grid_ind, dtype=int))

        # _find_bin_for_linear_position returns indices into the FULL (gap-
        # inclusive) bin list. Remap to active-bin indices: active bins are
        # numbered 0..n_active-1 in mask order; inactive/gap bins -> -1.
        n_total_bins = self.active_mask.size
        full_to_active = np.full(n_total_bins, -1, dtype=int)
        full_to_active[self.active_mask] = np.arange(int(self.active_mask.sum()))

        active_ind = np.full(full_grid_ind.shape, -1, dtype=int)
        in_range = full_grid_ind >= 0
        active_ind[in_range] = full_to_active[full_grid_ind[in_range]]
        return active_ind

    def bin_sizes(self) -> NDArray[np.float64]:
        """Return the length of each active 1D bin along the linearized track.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the length of each active linearized bin.

        Raises
        ------
        RuntimeError
            If `grid_edges` or `active_mask` is not populated.

        """
        if self.grid_edges is None or self.active_mask is None:  # pragma: no cover
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in ("grid_edges", "active_mask")
                    if getattr(self, name) is None
                ),
            )
        if not self.grid_edges or self.grid_edges[0].size <= 1:  # pragma: no cover
            raise ValueError(
                "grid_edges (1D) are not properly defined for length calculation.",
            )

        all_1d_bin_lengths = np.diff(self.grid_edges[0])
        return all_1d_bin_lengths[self.active_mask]

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Linear-track face measure: unit cross-section ``A = 1`` on every edge,
        with junction (inter-segment) edge distances corrected to the true
        along-track length.

        An intra-segment edge already carries the correct along-track spacing (its
        ``"distance"`` is the straight-line length along that segment). An
        inter-segment edge currently carries the Euclidean CHORD between the two
        junction bins, which understates the along-track distance around a bend and
        oversmooths. Its ``"distance"`` is replaced by the along-track length
        ``||bin_u - J|| + ||bin_v - J||`` through the shared junction node ``J`` (the
        geodesic over the straight-segment substrate). A straight track has no
        inter-segment edges, so this is a no-op there.
        """
        if self.connectivity is None:
            raise LayoutNotBuiltError(type(self).__name__, "connectivity")
        junctions = track_junctions(
            self.connectivity, self.linearization().graph, self.bin_centers.shape[1]
        )
        return track_geometry(
            self.connectivity, self.bin_centers, self.bin_sizes(), junctions
        )


def _shared_track_node(
    track_edges: list[tuple[Any, Any]], seg_a: int, seg_b: int
) -> Any | None:
    """Track-graph node shared by two track segments, or None if not exactly one."""
    if not (0 <= seg_a < len(track_edges) and 0 <= seg_b < len(track_edges)):
        return None
    shared = set(track_edges[seg_a]) & set(track_edges[seg_b])
    if len(shared) != 1:
        return None
    return next(iter(shared))
