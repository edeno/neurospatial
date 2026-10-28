"""Static contract: region queries return precise types.

``regions_containing_points`` returns a DataFrame by default and a list of
names with ``return_dataframe=False``. Run
with ``uv run mypy tests/typing/regions_queries.py``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Regions
    from neurospatial.regions import (
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
