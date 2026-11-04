"""The scene renderer draws a track's overlays where it draws the track's field.

``Environment.plot_field`` draws a linearized track's field against distance
along the track, so a position on that panel is its distance along the
track, at zero field value, not its raw ``(x, y)``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection

from neurospatial import Environment
from neurospatial.animation import (
    Animation,
    EventOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
)

TIMES = np.linspace(0.0, 1.0, 11)


def _l_track() -> Environment:
    """An L: east 40 cm along y = 20, then north 30 cm; linear 0 is (10, 20)."""
    graph = nx.Graph()
    graph.add_node("start", pos=(10.0, 20.0))
    graph.add_node("corner", pos=(50.0, 20.0))
    graph.add_node("end", pos=(50.0, 50.0))
    graph.add_edge("start", "corner", distance=40.0)
    graph.add_edge("corner", "end", distance=30.0)
    return Environment.from_graph(
        graph,
        [("start", "corner"), ("corner", "end")],
        bin_size=2.0,
        edge_spacing=10.0,
        units="cm",
    )


def _stem() -> Environment:
    """A 1-D stem from 5 to 45 cm: its linear 0 is the native 5 cm."""
    graph = nx.Graph()
    graph.add_node("a", pos=(5.0,))
    graph.add_node("b", pos=(45.0,))
    graph.add_edge("a", "b", distance=40.0)
    return Environment.from_graph(graph, [("a", "b")], bin_size=2.0, units="cm")


def _marker_points(scene: Animation, time: float) -> np.ndarray:
    scene.render_frame(time, dpi=30)
    ax = scene._renderer.figure.axes[0]
    points = [
        artist.get_offsets()
        for artist in ax.collections
        if isinstance(artist, PathCollection) and artist.get_visible()
    ]
    return np.concatenate(points) if points else np.empty((0, 2))


def _scene(env: Environment, positions: np.ndarray, **overlay) -> Animation:
    return Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=TIMES[[0, -1]],
        overlays=[PositionOverlay(positions, TIMES, **overlay)],
    )


@pytest.mark.parametrize("make_env", [_l_track, _stem], ids=["2-D L", "1-D stem"])
def test_track_position_marker_is_drawn_at_its_distance_along_the_track(make_env):
    env = make_env()
    if env.n_dims == 2:
        # Out along the east arm, then up the north arm: y is not zero.
        positions = np.column_stack(
            [
                np.r_[np.linspace(12, 50, 6), np.full(5, 50.0)],
                np.r_[np.full(6, 20.0), np.linspace(26, 48, 5)],
            ]
        )
    else:
        positions = np.linspace(6.0, 44.0, TIMES.size)[:, None]
    scene = _scene(env, positions)
    try:
        for index in (2, 8):
            expected = env.to_linear(positions[index][None])[0]
            # The native coordinate differs from the track's: the test can
            # tell a raw position from a linearized one.
            assert not np.isclose(expected, positions[index][0])
            (point,) = _marker_points(scene, TIMES[index])
            np.testing.assert_allclose(point, [expected, 0.0])
            ax = scene._renderer.figure.axes[0]
            low, high = ax.get_xlim()
            assert low <= expected <= high
    finally:
        scene.close()


def test_arena_position_marker_stays_at_its_raw_position():
    # Negative control: off a track the marker is the position itself.
    env = Environment.from_grid_mask(
        np.ones((4, 4), dtype=bool), (np.arange(5.0) * 10, np.arange(5.0) * 10)
    )
    positions = np.column_stack([np.linspace(5, 35, 11), np.linspace(35, 5, 11)])
    scene = _scene(env, positions)
    try:
        (point,) = _marker_points(scene, TIMES[3])
        np.testing.assert_allclose(point, positions[3])
    finally:
        scene.close()


def test_track_trail_and_events_are_along_the_track_and_never_join_segments():
    env = _l_track()
    positions = np.column_stack(
        [
            np.r_[np.linspace(12, 50, 6), np.full(5, 50.0)],
            np.r_[np.full(6, 20.0), np.linspace(26, 48, 5)],
        ]
    )
    along = env.to_linear(positions)
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=TIMES[[0, -1]],
        overlays=[
            PositionOverlay(positions, TIMES, trail_length=20, color="cyan"),
            EventOverlay(
                event_times={"lap": np.array([TIMES[7]])},
                positions=positions,
                position_times=TIMES,
            ),
        ],
    )
    try:
        points = _marker_points(scene, TIMES[-1])
        # The event's marker is at the event's distance along the track.
        assert any(np.allclose(p, [along[7], 0.0]) for p in points)
        ax = scene._renderer.figure.axes[0]
        trails = [
            c
            for c in ax.collections
            if isinstance(c, LineCollection) and len(c.get_segments())
        ]
        segments = np.concatenate([np.asarray(c.get_segments()) for c in trails])
        assert np.all(segments[:, :, 1] == 0)
        # Between the arms the track skips edge_spacing; no step crosses it.
        gap = (along[5], along[6])
        assert not any(
            min(s[:, 0]) <= gap[0] < gap[1] <= max(s[:, 0]) for s in segments
        )
    finally:
        scene.close()


def test_track_scene_warns_that_it_leaves_out_heading_arrows():
    env = _l_track()
    positions = np.column_stack([np.linspace(12, 48, 11), np.full(11, 20.0)])
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=TIMES[[0, -1]],
        overlays=[
            PositionOverlay(positions, TIMES),
            HeadDirectionOverlay(np.zeros(11), TIMES, length=5.0),
        ],
    )
    try:
        with pytest.warns(UserWarning, match="leaves out HeadDirectionOverlay"):
            scene.render_frame(0.5, dpi=30)
    finally:
        scene.close()
