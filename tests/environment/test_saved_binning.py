"""Loaded data components retain scientific plots and their renderer choices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from .test_plot_from_binning import render_case


@pytest.mark.slow
@pytest.mark.parametrize("operation", ["plot", "plot_field"])
def test_loaded_environment_plots_like_original(environment_round_trip, operation):
    original, restore = environment_round_trip
    loaded = restore()
    before, expected = render_case(original, operation)
    after, actual = render_case(loaded, operation)
    try:
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    finally:
        plt.close(before)
        plt.close(after)


def test_loaded_environment_renders_by_component(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    expected_figure, expected_axes = plt.subplots()
    actual_figure, actual_axes = plt.subplots()
    try:
        field = np.arange(original.n_bins, dtype=float)
        original.plot_field(field, ax=expected_axes, colorbar=False)
        loaded.plot_field(field, ax=actual_axes, colorbar=False)
        expected_classes = [
            type(artist).__name__ for artist in expected_axes.collections
        ]
        actual_classes = [type(artist).__name__ for artist in actual_axes.collections]
        assert actual_classes == expected_classes
        assert actual_classes
        if original.is_linearized_track:
            assert actual_axes.lines
    finally:
        plt.close(expected_figure)
        plt.close(actual_figure)


@pytest.mark.parametrize("built_environment", ["track"], indirect=True)
def test_loaded_track_plot_1d(environment_round_trip):
    original, restore = environment_round_trip
    loaded = restore()
    before, expected = render_case(original, "plot_1d")
    after, actual = render_case(loaded, "plot_1d")
    try:
        assert actual["warnings"].size == 0
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    finally:
        plt.close(before)
        plt.close(after)
