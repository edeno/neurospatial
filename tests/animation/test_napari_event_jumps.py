"""Napari event visibility stays exact when the view jumps.

Moving the slider, typing a time or playback that skips frames can jump
further than the decay window, in either direction; the events shown must
still be exactly those the time calls for.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import EventOverlay

pytestmark = [pytest.mark.napari, pytest.mark.xdist_group(name="napari_gui")]


@pytest.mark.parametrize("decay", [None, 0, 5])
def test_a_napari_layer_follows_jumps_back_and_forth(decay):
    pytest.importorskip("napari")
    env = Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), (np.arange(11.0), np.arange(11.0))
    )
    n = 200
    frame_times = np.arange(n) * 0.1
    # One event per frame, each at its own x so the shown set is identifiable.
    x = np.linspace(0.5, 9.5, n)
    events = EventOverlay.from_fixed_positions(
        frame_times, np.column_stack([x, np.full(n, 5.0)]), decay_frames=decay
    )
    view = env.animate_fields(
        np.ones((n, env.n_bins)),
        frame_times=frame_times,
        overlays=[events],
        backend="napari",
        show=False,
    )
    try:
        layer = next(
            layer for layer in view.native.layers if layer.name.startswith("Events")
        )
        frames = np.arange(n)
        for frame in (100, 1, 10, 199, 10, 0, 150, 149):
            view.seek(frame_times[frame])
            if decay is None:
                expected = frames <= frame
            else:
                expected = (frames >= frame - decay) & (frames <= frame)
            # napari stores (y, x); x identifies the event.
            np.testing.assert_allclose(np.sort(layer.data[:, 1]), x[expected])
            np.testing.assert_array_equal(layer.data[:, 0], 5.0)
    finally:
        view.close()
