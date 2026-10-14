"""Static check that the saved-data facade satisfies the layout interface.

Run with ``uv run mypy tests/typing/binning_layout.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial.layout._binning import Binning
    from neurospatial.layout._binning_layout import BinningLayout
    from neurospatial.layout.base import LayoutEngine

    def _check(layout: BinningLayout, points: NDArray[np.float64]) -> None:
        engine: LayoutEngine = layout
        assert_type(engine.point_to_bin_index(points), NDArray[np.intp])
        assert_type(layout.binning(), Binning)
        assert_type(layout.grid_edges, tuple[NDArray[np.float64], ...])
        layout.grid_edges = ()  # type: ignore[misc]
        layout.grid_shape = (1,)  # type: ignore[misc]
        layout.active_mask = np.ones(1, dtype=bool)  # type: ignore[misc]
        _ = layout.finite_volume_geometry  # type: ignore[attr-defined]
        _ = layout.rectilinear_grid  # type: ignore[attr-defined]
        _ = layout.to_linear  # type: ignore[attr-defined]
