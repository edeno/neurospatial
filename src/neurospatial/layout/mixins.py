from collections.abc import Sequence

import matplotlib
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.layout._capabilities import RectilinearGrid
from neurospatial.layout._locators import NearestBinLocator, RectilinearLocator
from neurospatial.layout._plotting import plot_grid
from neurospatial.layout._rebin_spec import GridBinningSpec


# ---------------------------------------------------------------------------
# KD-tree mixin (for point_to_bin_index)
# ---------------------------------------------------------------------------
class _KDTreeMixin:
    """Mixin providing nearest-bin lookup through an owned data locator.

    Layouts call ``_build_kdtree`` after finalizing their bin centers. The
    locator builds its KD-tree lazily and keeps the original bin-index mapping.
    """

    _locator: NearestBinLocator | None = None

    def _build_kdtree(
        self,
        points_for_tree: NDArray[np.float64],
        mask_for_points_in_tree: NDArray[np.bool_] | None = None,
    ) -> None:
        """Prepare nearest-bin geometry, preserving masked source indices.

        Parameters
        ----------
        points_for_tree : ndarray of float64, shape (n_points, n_dims)
            Bin coordinates from which the locator is built.
        mask_for_points_in_tree : ndarray of bool, shape (n_points,), optional
            Select centers while keeping their original bin indices.

        Raises
        ------
        ValueError
            If the centers or optional mask have an incompatible shape.
        """
        if points_for_tree.ndim != 2:
            raise ValueError(
                "points_for_tree must be a 2D array with shape (n_points, n_dims).",
            )
        if mask_for_points_in_tree is not None:
            if (
                mask_for_points_in_tree.ndim != 1
                or mask_for_points_in_tree.shape[0] != points_for_tree.shape[0]
            ):
                raise ValueError(
                    "mask_for_points_in_tree must be 1D and match the "
                    "number of rows in points_for_tree.",
                )
            centers = points_for_tree[mask_for_points_in_tree]
            bin_indices = np.flatnonzero(mask_for_points_in_tree).astype(np.intp)
        else:
            centers = points_for_tree
            bin_indices = None
        self._locator = NearestBinLocator(centers, bin_indices=bin_indices)

    def locator(self) -> NearestBinLocator:
        """Return the built layout's owned nearest-bin locator."""
        if self._locator is None:
            # Older pickles contain the tree and source mapping rather than a
            # locator; their data is sufficient to construct the new record.
            legacy_tree = getattr(self, "_kdtree", None)
            if legacy_tree is not None:
                self._locator = NearestBinLocator(
                    legacy_tree.data,
                    bin_indices=getattr(self, "_kdtree_nodes_to_bin_indices_map", None),
                )
            else:
                centers = getattr(self, "bin_centers", None)
                if centers is None or len(centers) == 0:
                    raise LayoutNotBuiltError(type(self).__name__, "bin_centers")
                self._build_kdtree(centers)
        assert self._locator is not None
        return self._locator

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map points to their nearest bin, with -1 for non-finite rows.

        An unbuilt or empty layout returns -1 for every query point. Built
        layouts reject a coordinate width that differs from their bin centers.
        """
        if self._locator is None and getattr(self, "_kdtree", None) is None:
            centers = getattr(self, "bin_centers", None)
            if centers is None or len(centers) == 0:
                return np.full(np.atleast_2d(points).shape[0], -1, dtype=np.intp)
        return self.locator().locate(points)


class _GridMixin:
    """Mixin for grid-based layout engines (e.g., RegularGrid, ShapelyPolygon).

    Provides common functionality for layouts that are based on an underlying
    N-dimensional grid, such as `point_to_bin_index` using grid definitions,
    default plotting, and `bin_size` for uniform grids.

    Assumes the inheriting class defines grid-specific attributes like
    `grid_edges`, `grid_shape`, `active_mask`, `bin_centers`, and
    `connectivity`.
    """

    grid_edges: tuple[NDArray[np.float64], ...] | None = None
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None
    bin_centers: NDArray[np.float64] | None = None
    connectivity: nx.Graph | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None
    _layout_type_tag: str = "_Grid_Layout"
    _active_mask_inverse_map: NDArray[np.intp] | None = None
    _locator: RectilinearLocator | None = None
    _grid_spec: GridBinningSpec | None = None

    @property
    def active_mask_inverse_map(self) -> NDArray[np.intp] | None:
        """Cached inverse mapping from flat grid index to active bin index.

        Returns None if no active_mask is set. Otherwise returns an array where
        inverse_map[flat_idx] = active_bin_idx (or -1 if not active).

        This is computed once and cached for efficient repeated lookups.
        """
        if self.active_mask is None:
            return None

        # Compute and cache on first access
        if self._active_mask_inverse_map is None:
            active_mask_flat = self.active_mask.ravel()
            inverse_map = np.full(active_mask_flat.size, -1, dtype=np.intp)
            active_indices = np.flatnonzero(active_mask_flat)
            inverse_map[active_indices] = np.arange(len(active_indices), dtype=np.intp)
            self._active_mask_inverse_map = inverse_map

        return self._active_mask_inverse_map

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
        self._active_mask_inverse_map = None
        self._locator = RectilinearLocator(
            self.grid_edges,
            self.grid_shape,
            self.active_mask,
            self.active_mask_inverse_map,
        )

        if connect_diagonal is not None:
            self._grid_spec = GridBinningSpec(
                np.array([edge[0] for edge in self.grid_edges]),
                np.eye(len(self.grid_edges) + 1),
                bool(connect_diagonal),
            )

    def locator(self) -> RectilinearLocator:
        """Return the owned locator, rebuilding it for older saved layouts."""
        if self._locator is None:
            self._build_locator()
        assert self._locator is not None
        return self._locator

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map N-D points to active bin indices based on grid structure.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, n_dims)
            N-D points to map.

        Returns
        -------
        NDArray[np.intp], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points outside active areas.

        Raises
        ------
        RuntimeError
            If grid attributes (`grid_edges`, `grid_shape`) are not set.

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

        return self.locator().locate(points)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        figsize=(7, 7),
        cmap: str = "bone_r",
        alpha: float = 0.7,
        show_connectivity: bool = True,
        node_size: float = 20,
        node_color: str = "blue",
    ) -> matplotlib.axes.Axes:
        """Plot the grid-based layout.

        For 2D grids, displays the `active_mask` using `pcolormesh` and
        optionally overlays the `connectivity`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        figsize : Tuple[float, float], default=(7, 7)
            Size of the figure if a new one is created.
        cmap : str, default="bone_r"
            Colormap for the `active_mask` plot.
        alpha : float, default=0.7
            Transparency for the `active_mask` plot.
        show_connectivity : bool, default=True
            If True, draw the connectivity graph nodes and edges.
        node_size : float, default=20
            Size of nodes if `show_connectivity` is True.
        node_color : str, default="blue"
            Color of nodes if `show_connectivity` is True.
        **kwargs : Any
            Additional keyword arguments passed to `ax.pcolormesh` or
            NetworkX drawing functions.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout was plotted.

        Raises
        ------
        RuntimeError
            If layout attributes are not built.
        NotImplementedError
            If attempting to plot a non-2D grid layout with this method.

        """
        if (
            self.bin_centers is None
            or self.grid_edges is None
            or self.active_mask is None
            or self.grid_shape is None
            or self.connectivity is None
        ):
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in (
                        "bin_centers",
                        "grid_edges",
                        "active_mask",
                        "grid_shape",
                        "connectivity",
                    )
                    if getattr(self, name) is None
                ),
            )

        return plot_grid(
            ax,
            title=self._layout_type_tag,
            bin_centers=self.bin_centers,
            connectivity=self.connectivity,
            dimension_ranges=self.dimension_ranges,
            rectilinear=self.rectilinear_grid(),
            figsize=figsize,
            cmap=cmap,
            alpha=alpha,
            show_connectivity=show_connectivity,
            node_size=node_size,
            node_color=node_color,
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

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Cartesian face measure: product of the OTHER axes' widths for an
        axis-aligned neighbor; corner-only (diagonal) neighbors get ``A = 0``.

        On a uniform grid this is ``A = h**(n_dims - 1)`` for face-adjacent edges
        (and ``A = 1`` in 1D). Masks/holes only drop nodes; the geometry of the
        remaining edges is unchanged.
        """
        if self.connectivity is None:
            raise LayoutNotBuiltError(type(self).__name__, "connectivity")
        g = self.connectivity.copy()
        assert self.bin_centers is not None
        centers = self.bin_centers
        n_dims = centers.shape[1]
        widths = self._axis_widths()
        for u, v, data in g.edges(data=True):
            offset = centers[u] - centers[v]
            # Native grid centers share exactly stored coordinates along every
            # unchanged axis. An absolute threshold would depend on units.
            moved = offset != 0.0
            if moved.sum() != 1:  # diagonal / Moore edge -> shares only a corner
                data["A"] = 0.0
                continue
            axis = int(np.flatnonzero(moved)[0])
            data["A"] = float(np.prod([widths[d] for d in range(n_dims) if d != axis]))
        return g, self.bin_sizes()
