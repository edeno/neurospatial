"""Napari event visibility stays exact when the frame jumps.

The napari backend updates which events are shown incrementally. Moving the
slider, typing a frame or playback that skips frames can jump further than the
decay window, in either direction; the events shown must still be exactly
those the frame calls for.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.animation.backends.napari_backend import _show_run, _visible_run


def _expected(frames, frame, decay):
    """The events a frame shows, as every backend defines it."""
    if decay is None:
        return frames <= frame
    return (frames >= frame - decay) & (frames <= frame)


@pytest.mark.parametrize("decay", [None, 0, 1, 5, 40])
def test_any_sequence_of_frames_shows_exactly_the_right_events(decay):
    rng = np.random.default_rng(0)
    # Uneven events: several on some frames, none on others.
    frames = np.sort(rng.integers(0, 300, size=500))
    shown = np.zeros(len(frames), dtype=bool)
    run = (0, 0)
    steps = np.concatenate(
        [
            np.arange(0, 20),  # playback, one frame at a time
            rng.integers(-10, 320, size=200),  # jumps, past both ends too
            [299, 0, 299, 150, 150, 149],  # end to end, repeats, one back
        ]
    )
    for frame in steps:
        new = _visible_run(frames, int(frame), decay)
        _show_run(shown, run, new)
        run = new
        np.testing.assert_array_equal(shown, _expected(frames, frame, decay))


def test_only_a_change_is_reported():
    shown = np.zeros(10, dtype=bool)
    assert _show_run(shown, (0, 0), (2, 5))
    assert not _show_run(shown, (2, 5), (2, 5))


@pytest.mark.napari
def test_a_napari_layer_follows_jumps_back_and_forth():
    napari = pytest.importorskip("napari")
    from neurospatial import Environment
    from neurospatial.animation.backends.napari_backend import _render_event_overlay
    from neurospatial.animation.overlays import EventData

    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 10, (500, 2)), bin_size=1.0
    )
    n = 200
    frames = np.arange(n)
    data = EventData(
        event_positions={"spike": np.c_[np.linspace(1, 9, n), np.full(n, 5.0)]},
        event_frame_indices={"spike": frames},
        colors={"spike": "red"},
        markers={"spike": "o"},
        size=5.0,
        decay_frames=5,
        border_color="white",
        border_width=0.1,
        opacity=1.0,
    )
    viewer = napari.Viewer(show=False)
    try:
        viewer.add_image(np.zeros((n, 10, 10)))
        (layer,) = _render_event_overlay(viewer, data, env)
        # napari opens at the middle frame; then far back, far forward, back.
        for frame in (1, 10, 100, 10, 199, 0):
            viewer.dims.current_step = (frame, 0, 0)
            np.testing.assert_array_equal(layer.shown, _expected(frames, frame, 5))
    finally:
        viewer.close()
