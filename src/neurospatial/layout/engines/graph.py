from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry import (
    Binning,
    TrackLinearization,
    component_extent,
    project_1d_to_2d,
    track_junctions,
)
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._labels import LayoutType
from neurospatial.layout._plotting import _plot_track_linear
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.graph import (
    _find_bin_for_linear_position,
    _get_graph_bins,
    set_straight_distances,
    track_pairs,
)
from neurospatial.layout.mixins import _KDTreeMixin, _NeighborPairsMixin
from neurospatial.layout.validation import validate_cell_size

if TYPE_CHECKING:
    import matplotlib.axes


class GraphLayout(_KDTreeMixin, _NeighborPairsMixin):
    """Layout defined by a user-provided graph, typically for 1D tracks.

    The graph's nodes (with 'pos' attributes) and a specified edge order
    are used to create a linearized representation of the space, which is
    then binned. Neighbor pairs are derived from this binned structure.
    Uses `_KDTreeMixin` for point mapping and neighbor finding on the
    N-D embeddings of the linearized bin centers.
    """

    bin_centers: NDArray[np.float64]
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    layout_type: str = LayoutType.GRAPH.value
    size_parameter: ClassVar[str | None] = "bin_size"
    _build_params_used: dict[str, Any]

    # Layout Specific
    linear_bin_centers_: NDArray[np.float64] | None = None

    def __init__(self) -> None:
        """Initialize a GraphLayout engine."""
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 0), dtype=np.float64)
        self.dimension_ranges = None
        self.grid_edges = None
        self.grid_shape = None
        self.active_mask = None
        self.linear_bin_centers_ = None
        self._linearization: TrackLinearization | None = None
        self._segment_ids: NDArray[np.int_] | None = None

    @capture_build_params(data=("graph_definition", "edge_order"))
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
            (e.g., `(x, y)` coordinates). An edge is the straight segment
            between its nodes; its 'distance', if given, must be that length
            (it is computed when missing).
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
            If `edge_order` is empty, `bin_size` is not positive, or an
            edge's 'distance' is not the distance between its nodes.

        """
        self._begin_build()
        if not isinstance(graph_definition, nx.Graph):
            raise TypeError("graph_definition must be a NetworkX graph.")
        if not edge_order:  # Empty edge_order means no path to linearize
            raise ValueError("edge_order must not be empty.")
        validate_cell_size(bin_size, scalar_only=True)

        # track_linearization finds a point's nearest segment by its position in
        # ``graph.edges()`` but looks that segment up by its ``edge_id`` attribute,
        # so the two must coincide. Linearize a copy numbered by enumeration; the
        # caller's graph is never mutated.
        track_graph = graph_definition.copy()
        for enumeration_index, (u, v) in enumerate(track_graph.edges()):
            track_graph.edges[u, v]["edge_id"] = enumeration_index

        (linear_bin_centers, self.grid_edges, self.active_mask, edge_ids) = (
            _get_graph_bins(
                graph=track_graph,
                edge_order=edge_order,
                edge_spacing=edge_spacing,
                bin_size=bin_size,
            )
        )

        # The bins follow each edge's straight segment; everything measured
        # along the track uses "distance", which must be that length.
        set_straight_distances(track_graph)
        self.linear_bin_centers_ = linear_bin_centers[self.active_mask]
        self.bin_centers = project_1d_to_2d(
            self.linear_bin_centers_,
            track_graph,
            edge_order,
            edge_spacing,
        )
        self.grid_shape = (len(self.grid_edges[0]) - 1,)
        self._segment_ids = edge_ids
        self._set_neighbor_pairs(
            *track_pairs(
                graph=track_graph,
                bin_centers_nd=self.bin_centers,
                original_edge_ids=edge_ids,
                edge_order=edge_order,
            )
        )

        self._own_arrays(
            "bin_centers", "linear_bin_centers_", "active_mask", "grid_edges"
        )
        self._build_kdtree(points_for_tree=self.bin_centers)
        self._linearization = TrackLinearization(
            track_graph, edge_order, edge_spacing, self.grid_edges, self.active_mask
        )
        # The extent of the track's cells: the centerline the bins cover.
        self.dimension_ranges = component_extent(linearization=self._linearization)

    @property
    def is_linearized_track(self) -> bool:
        """Graph layouts are treated as 1-dimensional due to linearization.

        Returns
        -------
        bool
            Always True, indicating a 1D linearized layout.

        """
        return True

    def plot_linear_layout(
        self,
        ax: "matplotlib.axes.Axes | None" = None,
        **kwargs,
    ) -> "matplotlib.axes.Axes":
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
        return _plot_track_linear(
            ax,
            title=self.layout_type,
            linearization=self.linearization(),
            grid_edges=self.grid_edges,
            **kwargs,
        )

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert N-D points to 1D linearized coordinates along the track.

        A point equally near several segments takes the one first in
        ``edge_order``.

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
            N-D coordinates corresponding to the input linear positions. A
            position on no segment (below 0, past the end, inside a gap, or
            not finite) is a row of NaN.

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
        pairs, distances = self._neighbor_pairs()
        assert self._segment_ids is not None
        track = self.linearization()
        return assemble_binning(
            pairs,
            distances,
            kind=self.layout_type,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=1,
            locator=self.locator(),
            faces="track",
            face_arrays=(
                track_junctions(
                    pairs, self._segment_ids, track.graph, self.bin_centers.shape[1]
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
            Bin index (as in ``bin_centers``) of each linear position: bins
            are half-open, and the end of the track belongs to the last bin.
            -1 for a position before the start, past the end, inside a gap,
            or NaN.

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
