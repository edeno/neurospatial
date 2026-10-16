"""Forwarded warnings should identify the scientist's call site."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurospatial import Environment


@pytest.mark.parametrize("method", ["plot", "plot_field"])
def test_scale_bar_warning_points_to_user_call(method):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), grid_edges=(np.arange(3.0) * 1e-10,) * 2
    )
    figure, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if method == "plot":
                env.plot(ax=ax, scale_bar=True)
            else:
                env.plot_field(
                    np.arange(env.n_bins, dtype=float), ax=ax, scale_bar=True
                )
        assert captured
        assert all(Path(item.filename) == Path(__file__) for item in captured)
    finally:
        plt.close(figure)
