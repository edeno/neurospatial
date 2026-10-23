"""Events survive missing tracking while every renderer hides unknown locations.

Scene frames are checked in test_scene_frame_pixels.py.
"""

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
def test_napari_scene_events_show_only_located_points(located_events, decay):
    pytest.importorskip("napari")
    from neurospatial.animation import Animation
    from neurospatial.animation._prepared import PreparedScene
    from neurospatial.animation.backends._napari_scene import NapariScene

    env, times, _ = located_events
    overlay = EventOverlay.from_trajectory(
        times,
        np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]),
        np.array([0.0, 0.2, 1.0]),
        decay_frames=decay,
    )
    with Animation(
        env, np.zeros((len(times), env.n_bins)), frame_times=times, overlays=[overlay]
    ) as scene:
        # The native layer model, without a Qt viewer.
        handle = NapariScene.__new__(NapariScene)
        handle.scene, handle._helper, handle._updates = scene, PreparedScene(scene), []
        handle._panel_stacks = {None: []}
        layers = []

        def add(kind, values, **options):
            layers.append(SimpleNamespace(data=values, **options))
            return layers[-1]

        handle._add = add
        handle._build_overlays()
        (layer,) = layers
        for update in handle._updates:
            update(times[-1], None)
        # Napari draws (y, x); the event at 0.6 s has no tracked location.
        shown = layer.data[:, ::-1]
        assert np.isfinite(shown).all()
        expected = {None: [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]], 0: [[4.0, 5.0]]}
        assert_array_equal(shown, expected.get(decay, [[2.0, 3.0], [4.0, 5.0]]))
