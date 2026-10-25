"""Owned binning has no layout-interface requirement."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import networkx as nx
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.geometry._binning import Binning
    from neurospatial.layout import (
        HexagonalLayout,
        LayoutType,
        RegularGridLayout,
        create_layout,
    )

    def _check(env: Environment, points: NDArray[np.float64]) -> None:
        assert_type(env._binning, Binning)
        assert_type(env._binning.locator.locate(points), NDArray[np.intp])

    def _check_create_layout() -> None:
        # The documented members of every built-in engine resolve on the
        # engine create_layout returns.
        layout = create_layout(
            LayoutType.REGULAR_GRID,
            bin_size=2.0,
            dimension_ranges=[(0, 100), (0, 100)],
        )
        assert_type(layout.layout_type, str)
        assert_type(layout.connectivity, nx.Graph)
        assert_type(layout.bin_sizes(), NDArray[np.float64])
        assert_type(layout.plot(), matplotlib.axes.Axes)
        assert_type(layout.binning(), Binning)
        _ = layout.bin_centers
        _ = create_layout("RegularGrid", bin_size=2.0).bin_centers

    def _check_layout_type_is_a_class_label(
        grid: RegularGridLayout, hexagons: HexagonalLayout
    ) -> None:
        # The label names the engine class; an instance cannot relabel itself.
        grid.layout_type = "Hexagonal"  # type: ignore[misc]
        hexagons.layout_type = "RegularGrid"  # type: ignore[misc]
        create_layout("Graph").layout_type = "x"  # type: ignore[misc]
