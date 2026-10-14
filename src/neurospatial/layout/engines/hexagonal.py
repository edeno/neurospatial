import warnings
from collections.abc import Sequence
from typing import Any

import matplotlib
import matplotlib.axes
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.layout._components import HexagonShapes
from neurospatial.layout._locators import HexLocator
from neurospatial.layout._plotting import plot_hexagons
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.hexagonal import (
    _create_hex_connectivity_graph,
    _create_hex_grid,
    _infer_active_bins_from_hex_grid,
)
from neurospatial.layout.helpers.utils import (
    check_grid_size_safety,
)
from neurospatial.layout.validation import validate_connectivity_graph


class HexagonalLayout:
    """2D layout that tiles a rectangular area with a hexagonal lattice.

    Bin centers are the centers of the hexagons. Hexagons are connected to their
    immediate neighbors. Active hexagons can be inferred from data sample
    occupancy.
    """

    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph | None = None
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = ()
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    _layout_type_tag: str
    _build_params_used: dict[str, Any]

    # Layout Specific
    hexagon_width: float | None = None
    hex_radius_: float | None = None
    hex_orientation_: float | None = None
    grid_offset_x_: float | None = None
    grid_offset_y_: float | None = None
    _source_flat_to_active_id_map: dict[int, int] | None = None
    _locator: HexLocator | None = None

    def __init__(self) -> None:
        """Initialize a HexagonalLayout engine."""
        self._layout_type_tag = "Hexagonal"
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2))
        self.connectivity = None
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
        self.hexagon_width = None
        self.hex_radius_ = None
        self.hex_orientation_ = None
        self.grid_offset_x_ = None
        self.grid_offset_y_ = None
        self._source_flat_to_active_id_map = None
        self._locator = None

    @property
    def layout_type(self) -> str:
        """Return standardized category for this layout type."""
        return "hexagonal"

    @property
    def is_grid_compatible(self) -> bool:
        """Return False - hexagonal layouts cannot be rendered as 2D images."""
        return False

    @capture_build_params
    def build(
        self,
        *,
        hexagon_width: float,
        dimension_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
        positions: NDArray[np.float64] | None = None,
        infer_active_bins: bool = True,
        bin_count_threshold: int = 0,
    ) -> None:
        """Build the hexagonal grid layout.

        Parameters
        ----------
        hexagon_width : float
            The width of the hexagons (distance between parallel sides).
        dimension_ranges : Optional[Tuple[Tuple[float,float], Tuple[float,float]]], optional
            Explicit `[(min_x, max_x), (min_y, max_y)]` for the area to tile.
            If None (default), range is inferred from `positions`.
        positions : Optional[NDArray[np.float64]], shape (n_samples, 2), optional
            2D data used to infer `dimension_ranges` (if not provided) and/or
            to infer active hexagons (if `infer_active_bins` is True).
            Defaults to None.
        infer_active_bins : bool, default=True
            If True and `positions` are provided, infers active hexagons
            based on occupancy. If False, all hexagons within the defined
            area are considered active.
        bin_count_threshold : int, default=0
            If `infer_active_bins` is True, the minimum number of samples a
            hexagon must contain to be considered active.

        Raises
        ------
        ValueError
            If `dimension_ranges` and `positions` are both None, or if
            `hexagon_width` is not positive.

        """
        self.hexagon_width = hexagon_width
        (
            full_grid_bin_centers,
            self.grid_shape,
            self.hex_radius_,
            self.hex_orientation_,
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.dimension_ranges,
        ) = _create_hex_grid(
            positions=positions,
            dimension_range=dimension_ranges,
            hexagon_width=self.hexagon_width,
        )
        if (
            self.hex_radius_ is None
            or self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.grid_shape is None
        ):
            raise RuntimeError(
                "Hexagonal grid generation failed; missing geometry parameters."
            )

        # Safety check: warn or error if grid is very large
        n_dims = 2  # Hexagonal is always 2D
        check_grid_size_safety(self.grid_shape, n_dims)

        if infer_active_bins and positions is not None:
            assert (
                self.hex_radius_ is not None
                and self.grid_offset_x_ is not None
                and self.grid_offset_y_ is not None
            )
            active_bin_original_flat_indices = _infer_active_bins_from_hex_grid(
                positions=positions,
                centers_shape=self.grid_shape,
                hex_radius=self.hex_radius_,
                min_x=self.grid_offset_x_,
                min_y=self.grid_offset_y_,
                bin_count_threshold=bin_count_threshold,
            )
        else:
            active_bin_original_flat_indices = np.arange(len(full_grid_bin_centers))

        nd_active_mask = np.zeros(self.grid_shape, dtype=bool).ravel()
        nd_active_mask[active_bin_original_flat_indices] = True
        self.active_mask = nd_active_mask.reshape(self.grid_shape)

        self.bin_centers = full_grid_bin_centers[active_bin_original_flat_indices]

        self.connectivity = _create_hex_connectivity_graph(
            active_original_flat_indices=active_bin_original_flat_indices,
            full_grid_bin_centers=full_grid_bin_centers,
            centers_shape=self.grid_shape,
        )

        # Validate connectivity graph has required attributes
        validate_connectivity_graph(self.connectivity, n_dims=2)

        self._source_flat_to_active_id_map = {
            data["source_grid_flat_index"]: node_id
            for node_id, data in self.connectivity.nodes(data=True)
        }
        self._build_locator()

    def _build_locator(self) -> None:
        """Snapshot hexagonal geometry and the active-bin mapping."""
        if (
            self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.hex_radius_ is None
            or self.grid_shape is None
            or self._source_flat_to_active_id_map is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "hexagonal geometry")
        assert len(self.grid_shape) == 2
        flat_to_active = np.full(np.prod(self.grid_shape), -1, dtype=np.intp)
        for source, active in self._source_flat_to_active_id_map.items():
            flat_to_active[source] = active
        self._locator = HexLocator(
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.hex_radius_,
            (self.grid_shape[0], self.grid_shape[1]),
            flat_to_active,
        )

    def locator(self) -> HexLocator:
        """Return the owned locator, rebuilding it for older saved layouts."""
        if self._locator is None:
            self._build_locator()
        assert self._locator is not None
        return self._locator

    @property
    def is_linearized_track(self) -> bool:
        """Hexagonal layouts are 2-dimensional.

        Returns
        -------
        bool
            Always False, indicating a 2D layout.

        """
        return False

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the hexagonal layout.

        Displays active hexagons and their connectivity graph.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            Axes to plot on. If None, new figure and axes are created.
        **kwargs : Any
            Additional keyword arguments:
            - `show_hexagons` (bool, default=True): Whether to draw hexagon cells.
            - `hexagon_kwargs` (dict): Kwargs for `matplotlib.patches.RegularPolygon`.
            - Other kwargs are passed to `_generic_graph_plot` for the graph.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the layout is plotted.

        """
        return plot_hexagons(
            ax,
            title=self._layout_type_tag,
            shapes=(
                self.bin_shapes()
                if self.hex_radius_ is not None and self.bin_centers is not None
                else None
            ),
            connectivity=self.connectivity,
            dimension_ranges=self.dimension_ranges,
            **kwargs,
        )

    def bin_shapes(self) -> HexagonShapes:
        """Return the active hexagons in bin order with owned center coordinates."""
        if self.hex_radius_ is None or self.bin_centers is None:
            raise LayoutNotBuiltError(type(self).__name__, "hex_radius_")
        return HexagonShapes(
            self.bin_centers,
            self.hex_radius_,
            float(self.hex_orientation_) if self.hex_orientation_ is not None else 0.0,
        )

    def point_to_bin_index(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map 2D points to active hexagonal bin indices.

        Uses specialized logic to determine which hexagon each point falls into,
        then maps this to an active bin index.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, 2)
            2D points to map.

        Returns
        -------
        NDArray[np.intp], shape (n_points,)
            Active bin indices (0 to N-1). -1 for points not in an active hexagon.

        """
        if (
            self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.hex_radius_ is None
            or self.grid_shape is None
            or self._source_flat_to_active_id_map is None
        ):
            # This can happen if build() failed or was incomplete (e.g. no active bins)
            warnings.warn(
                "HexagonalLayout is not fully initialized or has no active bins. "
                "Cannot map points to bin indices.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.full(points.shape[0], -1, dtype=np.intp)

        return self.locator().locate(points)

    def bin_sizes(self) -> NDArray[np.float64]:
        """Calculate the area of each hexagonal bin.

        All active hexagons are assumed to have the same area.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area for each active hexagonal bin.

        Raises
        ------
        RuntimeError
            If `hex_radius_` or `bin_centers` is not populated.

        """
        if self.hex_radius_ is None or self.bin_centers is None:  # pragma: no cover
            raise LayoutNotBuiltError(
                type(self).__name__,
                next(
                    name
                    for name in ("hex_radius_", "bin_centers")
                    if getattr(self, name) is None
                ),
            )

        # Area of a regular hexagon: (3 * sqrt(3) / 2) * side_length^2
        # For pointy-top hexagons, side_length (s) is equal to hex_radius_ (R, center to vertex).
        single_hex_area = 3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0
        return np.full(self.bin_centers.shape[0], single_hex_area)

    def finite_volume_geometry(self) -> tuple[nx.Graph, NDArray[np.float64]]:
        """Hex face measure: the shared hexagon side length ``s`` on every edge.

        For a pointy-top regular hexagonal lattice adjacent centers are ``sqrt(3)*s``
        apart, so ``s`` is derived per edge as ``distance / sqrt(3)`` from the
        edge's center-to-center ``"distance"`` (equivalently the layout's stored
        side length). Hex lattices are K-orthogonal, so no hand-tuned constant is
        needed.
        """
        if self.connectivity is None:
            raise LayoutNotBuiltError(type(self).__name__, "connectivity")
        g = self.connectivity.copy()
        for _u, _v, data in g.edges(data=True):
            data["A"] = float(data["distance"]) / np.sqrt(3.0)
        return g, self.bin_sizes()
