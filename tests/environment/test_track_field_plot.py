"""Track field plots preserve physical display coordinates and missing support."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from neurospatial import Environment


def test_track_field_keeps_display_gaps_and_internal_missing_bins():
    env = Environment.linear_track(
        node_positions=[(0.0, 0.0), (20.0, 0.0), (20.0, 15.0)],
        bin_size=5.0,
        edge_spacing=10.0,
        units="cm",
    )
    field = np.arange(env.n_bins, dtype=float) + 1
    field[1] = np.nan
    original = field.copy()
    fig, ax = plt.subplots()
    try:
        assert env.plot_field(field, ax=ax, linewidth=1.0) is ax
        line = ax.lines[0]
        x, y = line.get_data()
        active = env.active_mask.ravel()
        np.testing.assert_allclose(x[active], env.to_linear(env.bin_centers))
        np.testing.assert_array_equal(y[active], field)
        assert np.any(~active), "The example must include an artificial track gap."
        assert np.isnan(y[~active]).all()
        assert "cm" in ax.get_xlabel()
        assert "linearized" in ax.get_xlabel().lower()
        starts = sum(code == Path.MOVETO for _, code in line.get_path().iter_segments())
        assert starts >= 3  # Two track segments and the internal missing bin.
        # No filled polygon may bridge a location absent from the line.
        missing_x = x[np.isnan(y)]
        for collection in ax.collections:
            for polygon in collection.get_paths():
                xmin, xmax = (
                    np.nanmin(polygon.vertices[:, 0]),
                    np.nanmax(polygon.vertices[:, 0]),
                )
                assert not np.any((missing_x > xmin) & (missing_x < xmax))
        np.testing.assert_array_equal(field, original)
    finally:
        plt.close(fig)


def test_all_missing_track_field_is_labeled_without_fabricating_values():
    env = Environment.linear_track(endpoints=[(0, 0), (20, 0)], bin_size=5, units="cm")
    fig, ax = plt.subplots()
    try:
        env.plot_field(np.full(env.n_bins, np.nan), ax=ax)
        assert np.isnan(ax.lines[0].get_ydata()).all()
        assert "cm" in ax.get_xlabel()
        assert all(not collection.get_paths() for collection in ax.collections)
    finally:
        plt.close(fig)
