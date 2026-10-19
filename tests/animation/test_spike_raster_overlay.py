"""SpikeRasterOverlay: its rows, its checks, and backends that skip it."""

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
        ({"window_seconds": 0.0}, "window_seconds must be positive"),
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


def test_other_backends_skip_the_raster_with_a_warning(env):
    fields = np.zeros((3, env.n_bins))
    with pytest.warns(UserWarning, match="html backend does not draw SpikeRaster"):
        env.animate_fields(
            fields,
            frame_times=np.arange(3.0),
            backend="html",
            dry_run=True,
            overlays=[SpikeRasterOverlay(SPIKES)],
        )


def test_rate_maps_follow_the_rows_and_must_cover_the_bins(env):
    rates = np.arange(3 * env.n_bins, dtype=float).reshape(3, -1)
    raster = SpikeRasterOverlay(SPIKES, order=[2, 0, 1], rates=rates)
    (data,) = _convert_overlays_to_data([raster], np.arange(3.0), 3, env).spike_rasters
    np.testing.assert_array_equal(data.rates, rates[[2, 0, 1]])
    wrong = SpikeRasterOverlay(SPIKES, rates=rates[:, :-1])
    with pytest.raises(ValueError, match="one rate map per unit"):
        _convert_overlays_to_data([wrong], np.arange(3.0), 3, env)
