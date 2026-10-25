from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import LayoutNotBuiltError
from neurospatial.geometry import (
    Binning,
    HexagonShapes,
    HexLocator,
    component_extent,
    mask_active_map,
)
from neurospatial.layout._assembly import assemble_binning
from neurospatial.layout._labels import LayoutType
from neurospatial.layout.base import capture_build_params
from neurospatial.layout.helpers.hexagonal import (
    _create_hex_grid,
    _infer_active_bins_from_hex_grid,
    hex_grid_pairs,
)
from neurospatial.layout.helpers.neighbor_pairs import euclidean_pair_distances
from neurospatial.layout.helpers.utils import (
    check_grid_size_safety,
)
from neurospatial.layout.mixins import _NeighborPairsMixin, _no_active_bins_message


class HexagonalLayout(_NeighborPairsMixin):
    """2D layout that tiles a rectangular area with a hexagonal lattice.

    Bin centers are the centers of the hexagons. Hexagons are connected to their
    immediate neighbors. Active hexagons can be inferred from data sample
    occupancy.
    """

    bin_centers: NDArray[np.float64]
    dimension_ranges: Sequence[tuple[float, float]] | None = None

    grid_edges: tuple[NDArray[np.float64], ...] | None = ()
    grid_shape: tuple[int, ...] | None = None
    active_mask: NDArray[np.bool_] | None = None

    layout_type: ClassVar[str] = LayoutType.HEXAGONAL.value
    size_parameter: ClassVar[str | None] = "hexagon_width"
    _build_params_used: dict[str, Any]

    # Layout Specific
    hexagon_width: float | None = None
    hex_radius_: float | None = None
    hex_orientation_: float | None = None
    grid_offset_x_: float | None = None
    grid_offset_y_: float | None = None
    _locator: HexLocator | None = None

    def __init__(self) -> None:
        """Initialize a HexagonalLayout engine."""
        self._build_params_used = {}
        self.bin_centers = np.empty((0, 2))
        self.dimension_ranges = None
        self.grid_edges = ()
        self.grid_shape = None
        self.active_mask = None
        self.hexagon_width = None
        self.hex_radius_ = None
        self.hex_orientation_ = None
        self.grid_offset_x_ = None
        self.grid_offset_y_ = None
        self._locator = None

    @capture_build_params(data=("positions", "dimension_ranges"))
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
        self._begin_build()
        self.hexagon_width = hexagon_width
        (
            full_grid_bin_centers,
            self.grid_shape,
            self.hex_radius_,
            self.grid_offset_x_,
            self.grid_offset_y_,
        ) = _create_hex_grid(
            positions=positions,
            dimension_range=dimension_ranges,
            hexagon_width=self.hexagon_width,
        )
        self.hex_orientation_ = 0.0  # point-up

        # Safety check: warn or error if grid is very large
        check_grid_size_safety(self.grid_shape)

        if infer_active_bins and positions is not None:
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
        if len(active_bin_original_flat_indices) == 0:
            raise ValueError(
                _no_active_bins_message(
                    type(self).__name__,
                    f"no hexagon holds at least bin_count_threshold="
                    f"{bin_count_threshold} samples",
                    "Lower bin_count_threshold, use a larger hexagon_width so "
                    "each hexagon collects more samples, or check that the "
                    "positions are finite and cover the arena.",
                )
            )

        nd_active_mask = np.zeros(self.grid_shape, dtype=bool).ravel()
        nd_active_mask[active_bin_original_flat_indices] = True
        self.active_mask = nd_active_mask.reshape(self.grid_shape)

        self.bin_centers = full_grid_bin_centers[active_bin_original_flat_indices]

        pairs = hex_grid_pairs(self.active_mask)
        self._set_neighbor_pairs(
            pairs, euclidean_pair_distances(self.bin_centers, pairs)
        )
        # The extent of the active hexagons, corners included.
        self.dimension_ranges = component_extent(shapes=self.bin_shapes())
        self._own_arrays("bin_centers", "active_mask")
        self._build_locator()

    def _build_locator(self) -> None:
        """Snapshot hexagonal geometry and the active-bin mapping."""
        if (
            self.grid_offset_x_ is None
            or self.grid_offset_y_ is None
            or self.hex_radius_ is None
            or self.grid_shape is None
            or self.active_mask is None
        ):
            raise LayoutNotBuiltError(type(self).__name__, "locator")
        assert len(self.grid_shape) == 2
        self._locator = HexLocator(
            self.grid_offset_x_,
            self.grid_offset_y_,
            self.hex_radius_,
            (self.grid_shape[0], self.grid_shape[1]),
            mask_active_map(self.active_mask),
        )

    def locator(self) -> HexLocator:
        """Return the owned locator, building it on first use."""
        if self._locator is None:
            self._build_locator()
        assert self._locator is not None
        return self._locator

    def binning(self) -> Binning:
        """Return owned hexagons, lookup and lazy hexagon-side faces.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.
        """
        pairs, distances = self._neighbor_pairs()
        return assemble_binning(
            pairs,
            distances,
            kind=self.layout_type,
            centers=self.bin_centers,
            sizes=self.bin_sizes(),
            measure_dim=self.bin_centers.shape[1],
            locator=self.locator(),
            faces="hex",
            shapes=self.bin_shapes(),
            dimension_ranges=self.dimension_ranges,
        )

    @property
    def is_linearized_track(self) -> bool:
        """Hexagonal layouts are 2-dimensional.

        Returns
        -------
        bool
            Always False, indicating a 2D layout.

        """
        return False

    def bin_shapes(self) -> HexagonShapes:
        """Return the active hexagons in bin order with owned center coordinates."""
        if self.hex_radius_ is None or self.bin_centers is None:
            raise LayoutNotBuiltError(type(self).__name__, "bin_shapes")
        return HexagonShapes(
            self.bin_centers,
            self.hex_radius_,
            float(self.hex_orientation_) if self.hex_orientation_ is not None else 0.0,
        )

    def bin_sizes(self) -> NDArray[np.float64]:
        """Calculate the area of each hexagonal bin.

        All active hexagons are assumed to have the same area.

        Returns
        -------
        NDArray[np.float64], shape (n_active_bins,)
            Array containing the constant area for each active hexagonal bin.

        Raises
        ------
        LayoutNotBuiltError
            If the layout has not been built.

        """
        if self.hex_radius_ is None or self.bin_centers is None:
            raise LayoutNotBuiltError(type(self).__name__, "bin_sizes")

        # Area of a regular hexagon: (3 * sqrt(3) / 2) * side_length^2
        # For pointy-top hexagons, side_length (s) is equal to hex_radius_ (R, center to vertex).
        single_hex_area = 3.0 * np.sqrt(3.0) / 2.0 * self.hex_radius_**2.0
        return np.full(self.bin_centers.shape[0], single_hex_area)
