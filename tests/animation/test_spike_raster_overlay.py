"""SpikeRasterOverlay and EpochOverlay: their rows, checks and drawn panels."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import SpikeRasterOverlay
from neurospatial.animation.overlays import (
    OverlayProtocol,
    SpikeRasterData,
    _convert_overlays_to_data,
)

SPIKES = [np.array([0.1, 0.5]), np.array([0.3]), np.array([], dtype=float)]


@pytest.fixture(scope="module")
def env():
    return Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (200, 2)), bin_size=2.0
    )


def test_rows_follow_the_order_and_keep_each_units_label(env):
    raster = SpikeRasterOverlay(SPIKES, order=[2, 0, 1], unit_ids=["a", "b", "c"])
    assert isinstance(raster, OverlayProtocol)
    (data,) = _convert_overlays_to_data([raster], np.arange(3.0), 3, env).spike_rasters
    assert isinstance(data, SpikeRasterData)
    assert data.unit_ids.tolist() == ["c", "a", "b"]  # bottom row first
    for train, unit in zip(data.spike_times, [2, 0, 1], strict=True):
        np.testing.assert_array_equal(train, SPIKES[unit])


def test_labels_default_to_the_units_positions():
    (_, ids) = SpikeRasterOverlay(SPIKES)._rows
    assert ids.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"order": [0, 0, 1]}, "each of the 3 units once"),
        ({"order": [0, 1]}, "each of the 3 units once"),
        ({"unit_ids": [7, 7, 8]}, "repeat|duplicate"),
    ],
)
def test_a_raster_that_cannot_be_drawn_is_refused(options, message):
    with pytest.raises(ValueError, match=message):
        SpikeRasterOverlay(SPIKES, **options)


def test_spikes_must_be_finite_and_there_must_be_a_unit():
    with pytest.raises(ValueError, match="Unit 1 has spike times that are NaN"):
        SpikeRasterOverlay([np.array([0.1]), np.array([np.nan])])
    with pytest.raises(ValueError, match="at least one unit"):
        SpikeRasterOverlay([])


def test_scene_frames_draw_each_spike_in_its_units_row(env):
    from neurospatial.animation import Animation, EpochOverlay

    times = np.arange(6) * 0.1
    raster = SpikeRasterOverlay(SPIKES, order=[2, 0, 1], unit_ids=["a", "b", "c"])
    with Animation(
        env,
        np.zeros((6, env.n_bins)),
        frame_times=times,
        overlays=[raster, EpochOverlay([[0.05, 0.25]], label="Run")],
        window_seconds=2.0,
    ) as scene:
        scene.render_frame(0.3, dpi=30)
        axes = {ax.get_ylabel(): ax for ax in scene._renderer.figure.axes}
        rows = axes["Unit"]
        assert [tick.get_text() for tick in rows.get_yticklabels()] == ["c", "a", "b"]
        spikes = sorted(
            (round(float(segment[0, 0]), 6), round(float(segment[:, 1].mean())))
            for collection in rows.collections
            for segment in collection.get_segments()
        )
        # Unit "a" (row 1) spiked at 0.1 and 0.5 s, unit "b" (row 2) at 0.3 s.
        assert spikes == [(0.1, 1), (0.3, 2), (0.5, 1)]
        epochs = next(
            ax
            for ax in scene._renderer.figure.axes
            if [t.get_text() for t in ax.get_yticklabels()] == ["Run"]
        )
        assert len(epochs.collections) == 1


def test_rate_maps_follow_the_rows_and_must_cover_the_bins(env):
    rates = np.arange(3 * env.n_bins, dtype=float).reshape(3, -1)
    raster = SpikeRasterOverlay(SPIKES, order=[2, 0, 1], rates=rates)
    (data,) = _convert_overlays_to_data([raster], np.arange(3.0), 3, env).spike_rasters
    np.testing.assert_array_equal(data.rates, rates[[2, 0, 1]])
    wrong = SpikeRasterOverlay(SPIKES, rates=rates[:, :-1])
    with pytest.raises(ValueError, match="one rate map per unit"):
        _convert_overlays_to_data([wrong], np.arange(3.0), 3, env)


@pytest.mark.parametrize(
    ("make_source", "difference"),
    [
        (lambda env: env.with_metadata(units="m"), "units or coordinate frames"),
        (
            lambda env: Environment.from_samples(
                np.random.default_rng(0).uniform(0, 10, (200, 2)) + 0.5,
                bin_size=2.0,
            ),
            "bin",
        ),
    ],
)
def test_rate_results_must_share_the_animated_bins(env, make_source, difference):
    from neurospatial import IncompatibleEnvironmentError
    from neurospatial.encoding.spatial import SpatialRatesResult

    source = make_source(env)
    rates = SpatialRatesResult(
        np.ones((3, source.n_bins)), np.ones(source.n_bins), source, "binned", 2.0
    )
    with pytest.raises(IncompatibleEnvironmentError, match=difference):
        SpikeRasterOverlay(SPIKES, rates=rates).convert_to_data(np.arange(3.0), 3, env)
    same = SpatialRatesResult(
        np.ones((3, env.n_bins)), np.ones(env.n_bins), env, "binned", 2.0
    )
    data = SpikeRasterOverlay(SPIKES, rates=same).convert_to_data(np.arange(3.0), 3, env)
    assert data.rates.shape == (3, env.n_bins)


def test_epochs_are_sorted_merged_and_checked():
    from neurospatial.animation import EpochOverlay
    from neurospatial.animation.overlays import EpochData

    data = EpochOverlay([[3.0, 4.0], [1.0, 2.0], [1.5, 2.5]]).convert_to_data(
        np.arange(3.0), 3, None
    )
    assert isinstance(data, EpochData)
    np.testing.assert_array_equal(data.intervals, [[1.0, 2.5], [3.0, 4.0]])
    with pytest.raises(ValueError, match="epochs"):
        EpochOverlay([[2.0, 1.0]])
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        EpochOverlay([[0.0, 1.0]], alpha=2.0)
