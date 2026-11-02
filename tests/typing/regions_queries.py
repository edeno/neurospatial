"""Static contract: region queries and plots return precise types.

``regions_containing_points`` returns a DataFrame; ``plot_regions`` returns
its Axes. Run with ``uv run mypy tests/typing/regions_queries.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Regions
    from neurospatial.regions import (
        plot_regions,
        points_in_any_region,
        regions_containing_points,
    )

    def _queries(positions: NDArray[np.float64], regions: Regions) -> None:
        assert_type(points_in_any_region(positions, regions), NDArray[np.bool_])
        assert_type(regions_containing_points(positions, regions), pd.DataFrame)
        assert_type(plot_regions(regions), matplotlib.axes.Axes)
