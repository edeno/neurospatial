"""Owned binning has no layout-interface requirement."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.layout._binning import Binning

    def _check(env: Environment, points: NDArray[np.float64]) -> None:
        assert_type(env._binning, Binning)
        assert_type(env._binning.locator.locate(points), NDArray[np.intp])
