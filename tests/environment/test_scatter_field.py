"""Transformed spatial fields keep readable, unit-independent scatter markers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba

from neurospatial import Environment
from neurospatial.ops.transforms import Affine2D, scale_2d


def _rotated_grid(n_cells=4):
    edges = np.arange(n_cells + 1, dtype=float)
    env = Environment.from_grid_mask(np.ones((n_cells, n_cells), bool), (edges, edges))
    angle = np.pi / 4
    rotation = Affine2D(
        np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    return env.apply_transform(rotation)


def _scatter(ax):
    return next(c for c in ax.collections if isinstance(c, PathCollection))


def test_default_scatter_marker_area_does_not_depend_on_coordinate_units():
    centimeters = _rotated_grid(n_cells=20)
    centimeters = centimeters.with_metadata(units="cm")
    meters = centimeters.apply_transform(scale_2d(0.01))
    meters = meters.with_metadata(units="m")
    figures = []
    sizes = []
    try:
        for env in (centimeters, meters):
            fig, ax = plt.subplots(figsize=(2, 2))
            figures.append(fig)
            env.plot_field(np.arange(env.n_bins, dtype=float), ax=ax, colorbar=False)
            fig.canvas.draw()
            sizes.append(_scatter(ax).get_sizes())

        np.testing.assert_allclose(sizes[0], sizes[1], rtol=1e-10)
        assert np.all((sizes[0] > 0) & (sizes[0] <= 100))
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("per_bin", [False, True])
@pytest.mark.parametrize("skip_nan", [False, True])
def test_scatter_honors_marker_area_and_aligns_per_bin_sizes(per_bin, skip_nan):
    env = _rotated_grid()
    field = np.arange(env.n_bins, dtype=float)
    field[::3] = np.nan
    sizes = np.linspace(10.0, 40.0, env.n_bins) if per_bin else 17.0
    fig, ax = plt.subplots()
    try:
        result = env.plot_field(
            field,
            ax=ax,
            colorbar=False,
            s=sizes,
            edgecolors="black",
            nan_color=None if skip_nan else "gray",
        )
        expected = sizes[np.isfinite(field)] if per_bin and skip_nan else sizes
        np.testing.assert_allclose(_scatter(ax).get_sizes(), expected)
        assert result is ax
        expected_count = np.isfinite(field).sum() if skip_nan else env.n_bins
        assert len(_scatter(ax).get_offsets()) == expected_count
    finally:
        plt.close(fig)


def test_scatter_draws_missing_support_with_requested_nan_color():
    env = _rotated_grid()
    field = np.arange(env.n_bins, dtype=float)
    field[0] = np.nan
    cmap = plt.get_cmap("viridis")
    original_bad_color = cmap.get_bad().copy()
    fig, ax = plt.subplots()
    try:
        env.plot_field(field, ax=ax, colorbar=False, cmap=cmap, nan_color="magenta")
        fig.canvas.draw()

        np.testing.assert_allclose(_scatter(ax).get_facecolors()[0], to_rgba("magenta"))
        assert not np.ma.getmaskarray(_scatter(ax).get_offsets()[0]).any()
        np.testing.assert_array_equal(cmap.get_bad(), original_bad_color)
    finally:
        plt.close(fig)
