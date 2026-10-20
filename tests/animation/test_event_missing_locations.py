"""Events survive missing tracking while every renderer hides unknown locations."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.animation.overlays import EventOverlay


@pytest.fixture
def located_events():
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        (np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 10.0])),
    )
    times = np.array([0.0, 0.2, 0.6, 1.0])
    positions = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
    overlay = EventOverlay.from_trajectory(
        times,
        positions,
        np.array([0.0, 0.2, 1.0]),
    )
    data = overlay.convert_to_data(times, len(times), env)
    assert np.isnan(data.event_positions["event"][2]).all()
    return env, times, data


@pytest.mark.parametrize("decay", [None, 0, 2])
@pytest.mark.parametrize("all_missing", [False, True])
def test_native_events_submit_only_located_points(located_events, decay, all_missing):
    from neurospatial.animation.backends.fastplotlib_backend import _Events

    _, times, data = located_events
    data.decay_frames = decay
    if all_missing:
        data.event_positions["event"][:] = np.nan
    source_positions = data.event_positions["event"].copy()
    graphics = []

    class Plot:
        def add_scatter(self, positions, **kwargs):
            assert np.isfinite(positions).all()
            graphic = SimpleNamespace(data=positions, **kwargs)
            graphics.append(graphic)
            return graphic

    drawn = _Events(Plot(), data, lambda time: int(np.searchsorted(times, time)), times)
    for time in times:
        drawn.update(time)
    if all_missing:
        assert graphics == []
    else:
        assert len(graphics) == 1
        assert_array_equal(graphics[0].data[:, :2], source_positions[[0, 1, 3]])
        assert len(drawn.kinds[0][0]) == 3
    assert_array_equal(data.event_positions["event"], source_positions)
    assert_array_equal(data.event_times["event"], times)


@pytest.mark.parametrize("decay", [None, 0, 2])
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.napari
def test_legacy_napari_events_keep_finite_layer_bounds(
    located_events, decay, all_missing, monkeypatch
):
    from napari.layers import Points

    from neurospatial.animation.backends import napari_backend as backend

    env, times, data = located_events
    data.decay_frames = decay
    if all_missing:
        data.event_positions["event"][:] = np.nan
    source_positions = data.event_positions["event"].copy()
    layers = []

    def add_points(positions, **kwargs):
        assert np.isfinite(positions).all()
        layer = Points(positions, **kwargs)
        layers.append(layer)
        return layer

    monkeypatch.setattr(
        backend, "_register_event_visibility_callback", lambda *args: None
    )
    result = backend._render_event_overlay(
        SimpleNamespace(add_points=add_points), data, env
    )
    assert result == layers
    if all_missing:
        assert layers == []
    else:
        assert len(layers) == 1
        assert len(layers[0].data) == 3
        assert np.isfinite(layers[0].extent.data).all()
        if decay != 0:
            assert_array_equal(layers[0].metadata["event_frame_indices"], [0, 1, 3])
    assert_array_equal(data.event_positions["event"], source_positions)
    assert_array_equal(data.event_times["event"], times)


@pytest.mark.napari
@pytest.mark.parametrize("decay", [None, 0, 2])
def test_legacy_napari_gap_events_in_viewer(located_events, decay):
    import napari

    from neurospatial.animation.backends.napari_backend import _render_event_overlay

    env, times, data = located_events
    data.decay_frames = decay
    viewer = napari.Viewer(show=False)
    try:
        viewer.add_image(np.zeros((len(times), 10, 10)))
        (layer,) = _render_event_overlay(viewer, data, env)
        assert len(layer.data) == 3
        assert np.isfinite(layer.extent.data).all()
        for frame in (3, 0, 2, 1):
            viewer.dims.current_step = (frame, 0, 0)
        assert_array_equal(data.event_times["event"], times)
        assert np.isnan(data.event_positions["event"][2]).all()
    finally:
        viewer.close()
