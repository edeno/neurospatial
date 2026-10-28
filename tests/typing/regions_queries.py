"""Static contract: region queries and plots return precise types.

``regions_containing_points`` returns a DataFrame by default and a list of
names with ``return_dataframe=False``; ``plot_regions`` returns its Axes. Run
with ``uv run mypy tests/typing/regions_queries.py``.
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

    def _queries(positions: NDArray[np.float64], regions: Regions, flag: bool) -> None:
        assert_type(points_in_any_region(positions, regions), NDArray[np.bool_])
        assert_type(regions_containing_points(positions, regions), pd.DataFrame)
        assert_type(
            regions_containing_points(positions, regions, return_dataframe=True),
            pd.DataFrame,
        )
        assert_type(
            regions_containing_points(positions, regions, return_dataframe=False),
            list[list[str]],
        )
        assert_type(
            regions_containing_points(positions, regions, return_dataframe=flag),
            pd.DataFrame | list[list[str]],
        )
        assert_type(plot_regions(regions), matplotlib.axes.Axes)
