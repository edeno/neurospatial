from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError, _format_error
from neurospatial.geometry import (
    Binning,
    GridBinningSpec,
    NearestBinLocator,
    RectilinearGrid,
    RectilinearLocator,
    graph_from_binning,
)
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._plot_cells import _plot_binning
from neurospatial.layout.helpers.neighbor_pairs import euclidean_pair_distances
from neurospatial.layout.helpers.regular_grid import regular_grid_pairs

if TYPE_CHECKING:
    import matplotlib.axes


def _read_only(array: NDArray[np.generic]) -> NDArray[np.generic]:
    """A read-only copy of ``array``."""
    owned = np.array(array, copy=True)
    owned.setflags(write=False)
    return owned


def _no_active_bins_message(engine: str, cause: str, fix: str) -> str:
    """Return the refusal for a build that leaves no active bins."""
    return _format_error(
        f"{engine} has no active bins: {cause}.",
        why=(
            "Why: an environment needs at least one bin; with none, every "
            "position falls outside it and occupancy and rate maps are empty."
        ),
        fix=fix,
    )


# ---------------------------------------------------------------------------
# Neighbor pairs and the graph derived from them
# ---------------------------------------------------------------------------
class _NeighborPairsMixin:
    """Neighbor pairs computed by ``build()`` and a graph derived on request.

    An engine's ``build()`` stores its neighbor pairs and their distances with
    ``_set_neighbor_pairs``, and its ``binning()`` passes them on. The
    ``connectivity`` graph is a convenience built from that binning on first
    access; rebuilding the engine discards it.
    """

    layout_type: str
    _pairs: NDArray[np.intp] | None = None
    _pair_distances: NDArray[np.float64] | None = None
    _connectivity: nx.Graph | None = None

    def _begin_build(self) -> None:
        """Mark the engine unbuilt until ``build()`` sets its pairs again.

        A build that fails partway then leaves ``binning()`` raising
        :class:`LayoutNotBuiltError`, not mixing old and new state.
        """
        self._pairs = self._pair_distances = None
        self._connectivity = None

    def _own_arrays(self, *names: str) -> None:
        """Replace the named arrays, or tuples of arrays, with read-only copies.

        A built engine then shares no memory with the caller's inputs, and
        its arrays cannot drift from the pairs and lookup built from them.
        """
        for name in names:
            value = getattr(self, name)
            if isinstance(value, tuple):
                setattr(self, name, tuple(_read_only(array) for array in value))
            elif value is not None:
                setattr(self, name, _read_only(value))

    def binning(self) -> Binning:  # pragma: no cover - every engine overrides it
        raise NotImplementedError

    def _set_neighbor_pairs(
        self, pairs: NDArray[np.intp], distances: NDArray[np.float64]
    ) -> None:
        """Store the built neighbor pairs and discard any derived graph."""
        self._pairs = pairs
        self._pair_distances = distances
        self._connectivity = None

    def _neighbor_pairs(self) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return the built neighbor pairs and distances, or raise if unbuilt."""
        if self._pairs is None or self._pair_distances is None:
            raise LayoutNotBuiltError(type(self).__name__, "binning")
        return self._pairs, self._pair_distances

    @property
    def connectivity(self) -> nx.Graph:
        """Navigation graph of the built bins, derived from ``binning()``.

        Nodes are bin IDs ``0..n_bins - 1`` and edges are the binning's
        neighbor pairs, with the node and edge attributes of
        :func:`neurospatial.geometry.graph_from_binning` (the same as
        ``env.connectivity``). The graph is built on first access and cached
        until the engine is built again. It is an ordinary graph: editing it
        does not change the engine's bins or its ``binning()``.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        if self._connectivity is None:
            self._connectivity = graph_from_binning(self.binning())
        return self._connectivity

    def plot(
        self, ax: "matplotlib.axes.Axes | None" = None, **kwargs: Any
    ) -> "matplotlib.axes.Axes":
        """Draw the bins' cells and connectivity graph, as ``Environment.plot`` does.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Options of the drawing for this layout's cells. Grids:
            ``figsize``, ``cmap``, ``alpha``, ``show_connectivity``,
            ``node_size``, ``node_color``. Hexagons: ``figsize``,
            ``node_kwargs``, ``edge_kwargs``. Meshes: ``show_triangles``,
            ``show_centroids``, ``show_connectivity``, ``show_boundary`` and
            their ``*_kwargs``. Tracks: ``figsize``.

        Returns
        -------
        matplotlib.axes.Axes

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        return _plot_binning(
            self.binning(), self.connectivity, self.layout_type, ax, **kwargs
        )


# ---------------------------------------------------------------------------
# KD-tree mixin (nearest-bin lookup)
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """Mixin providing nearest-bin lookup through an owned data locator.

    Layouts call ``_build_kdtree`` after finalizing their bin centers. The
    locator builds its KD-tree lazily and keeps the original bin-index mapping.
    """

    _locator: NearestBinLocator | None = None

    def _build_kdtree(self, points_for_tree: NDArray[np.float64]) -> None:
        """Prepare nearest-bin lookup over the bin centers.

        Parameters
        ----------
        points_for_tree : ndarray of float64, shape (n_bins, n_dims)
            Bin centers, in bin order.

        Raises
        ------
        ValueError
            If the centers are not a 2-D array.
        """
        if points_for_tree.ndim != 2:
            raise ValueError(
                "points_for_tree must be a 2D array with shape (n_points, n_dims).",
            )
        self._locator = NearestBinLocator(points_for_tree)

    def locator(self) -> NearestBinLocator:
        """Return the built layout's owned nearest-bin locator."""
        if self._locator is None:
            centers = getattr(self, "bin_centers", None)
            if centers is None or len(centers) == 0:
                raise LayoutNotBuiltError(type(self).__name__, "bin_centers")
            self._build_kdtree(centers)
        assert self._locator is not None
        return self._locator


class _GridMixin(_NeighborPairsMixin):
    """Mixin for grid-based layout engines (e.g., RegularGrid, ShapelyPolygon).

    Provides common functionality for layouts that are based on an underlying
    N-dimensional grid, such as point lookup from grid definitions,
    default plotting, and `bin_size` for uniform grids.

    Assumes the inheriting class defines grid-specific attributes like
    `grid_edges`, `grid_shape`, `active_mask` and `bin_centers`.
    """

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None
    bin_centers: NDArray[np.float64] | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None
    layout_type: str
    _locator: RectilinearLocator | None = None
    _grid_spec: GridBinningSpec | None = None

    def _build_locator(self, *, connect_diagonal: bool | None = None) -> None:
        """Snapshot cells and explicit native construction choices together.

        ``connect_diagonal`` is supplied by builders, independently of whether
        the current mask happens to contain any diagonal pair. Legacy locator
        restoration omits it and does not infer a historical preference.
        """
        if self.grid_edges is None or self.grid_shape is None:
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in ("grid_edges", "grid_shape")
                    if getattr(self, name) is None
                ),
            )
        self._locator = RectilinearLocator(
            self.grid_edges,
            self.grid_shape,
            np.ones(self.grid_shape, dtype=bool)
            if self.active_mask is None
            else self.active_mask,
        )

        if connect_diagonal is not None:
            self._grid_spec = GridBinningSpec(
                np.array([edge[0] for edge in self.grid_edges]),
                np.eye(len(self.grid_edges) + 1),
                bool(connect_diagonal),
            )

    def locator(self) -> RectilinearLocator:
        """Return the owned locator, building it on first use."""
        if self._locator is None:
            self._build_locator()
        assert self._locator is not None
        return self._locator

    def _assign_cells(
        self, full_grid_bin_centers: NDArray[np.float64], *, connect_diagonal: bool
    ) -> None:
        """Keep the active cells, pair neighbors and snapshot lookup and grid choices.

        Every Cartesian grid engine finishes ``build()`` here once it has set
        ``grid_edges``, ``grid_shape`` and ``active_mask``.

        Parameters
        ----------
        full_grid_bin_centers : ndarray of float64, shape (n_cells, n_dims)
            Centers of every grid cell, active or not, in flat grid order.
        connect_diagonal : bool
            Whether cells that share only a corner are neighbors. Recorded in
            the grid's binning specification even when no such pair is active.
        """
        assert self.active_mask is not None and self.grid_shape is not None
        if not self.active_mask.any():
            raise ValueError(
                _no_active_bins_message(
                    type(self).__name__,
                    f"none of the {self.active_mask.size} cells of its "
                    f"{self.grid_shape} grid is active",
                    "Use a smaller bin_size so cell centers fall inside the "
                    "area, or check that the polygon, mask or positions cover "
                    "the arena.",
                )
            )
        self.bin_centers = full_grid_bin_centers[self.active_mask.ravel()]
        pairs = regular_grid_pairs(self.active_mask, connect_diagonal=connect_diagonal)
        self._set_neighbor_pairs(
            pairs, euclidean_pair_distances(self.bin_centers, pairs)
        )
        self._own_arrays("bin_centers", "active_mask", "grid_edges")
        self._build_locator(connect_diagonal=connect_diagonal)

    def binning(self) -> Binning:
        """Return owned bins, lookup, lazy grid faces and grid components.

        Faces use one width per axis. Face directions come from the grid
        cells, and the binning specification records the grid's anchor and
        diagonal-neighbor choice for later re-binning.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        pairs, distances = self._neighbor_pairs()
        assert self.bin_centers is not None
        return assemble_binning(
            pairs,
            distances,
            kind=self.layout_type,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=self.bin_centers.shape[1],
            locator=self.locator(),
            faces="grid",
            face_arrays=(self._axis_widths(),),
            rectilinear=self.rectilinear_grid(),
            grid_spec=self._grid_spec,
            dimension_ranges=self.dimension_ranges,
        )

    @property
    def is_linearized_track(self) -> bool:
        """Indicate if the grid layout is 1-dimensional.

        Standard grid layouts (RegularGrid, etc.) are generally N-D (N>=1).
        This mixin's default assumes not strictly 1D in the sense of a
        linearized track (which GraphLayout would handle).

        Returns
        -------
        bool
            False, as this mixin is for general N-D grids.

        """
        return False

    def bin_sizes(self) -> NDArray[np.float64]:
        """Calculate area/volume for each active bin, assuming a uniform grid.

        Computes the product of bin side lengths for each dimension from
        `grid_edges`. Assumes all bins in the grid have the same dimensions.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area/volume for each active bin.

        Raises
        ------
        RuntimeError
            If `grid_edges` or `bin_centers` is not populated.

        """
        if self.grid_edges is None or self.bin_centers is None:  # pragma: no cover
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in ("grid_edges", "bin_centers")
                    if getattr(self, name) is None
                ),
            )
        if not self.grid_edges or not all(
            len(e) > 1 for e in self.grid_edges
        ):  # pragma: no cover
            raise ValueError(
                "grid_edges are not properly defined for area/volume calculation.",
            )

        single_bin_measure = np.prod(self._axis_widths())

        return np.full(self.bin_centers.shape[0], single_bin_measure)

    def _axis_widths(self) -> NDArray[np.float64]:
        """Per-axis bin width from the first spacing of each axis's grid edges.

        Shared by the cell measure and the face measure so the two stay
        consistent; ``MaskedGridLayout.build`` rejects nonuniform edges.
        """
        assert self.grid_edges is not None
        return np.array([float(np.diff(edges)[0]) for edges in self.grid_edges])

    def rectilinear_grid(self) -> RectilinearGrid:
        """Axis-aligned cells of this grid: per-axis edges, shape and active mask."""
        if (
            self.grid_edges is None
            or self.grid_shape is None
            or self.active_mask is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "rectilinear_grid")
        return RectilinearGrid(self.grid_edges, self.grid_shape, self.active_mask)
