"""Movie clocks preserve native observations without widening scientific support."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationView, PositionOverlay
from neurospatial.animation._export import prepare_export
from neurospatial.animation._prepared import (
    PreparedScene,
    iter_observation_clocks,
)
from neurospatial.animation._scene_rendering import SceneRenderer
from neurospatial.animation.overlays import PositionData


@pytest.mark.parametrize("render", ["scene", "view"])
@pytest.mark.parametrize("adapter", [False, True], ids=["position", "custom-adapter"])
def test_export_retains_isolated_overlay_observations_and_excludes_stop(
    render, adapter, monkeypatch, tmp_path
):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )
    observed = np.array([100.1, 100.2, 100.3])
    positions = np.array([[0.2, 0.2], [1.0, 1.0], [1.8, 1.8]])

    class Adapter:
        times = None
        interp = "linear"

        def convert_to_data(self, frame_times, n_frames, env):
            return PositionData(
                positions,
                "red",
                5,
                None,
                times=observed,
                interval_valid=np.array([False, False]),
            )

    overlay = (
        Adapter()
        if adapter
        else PositionOverlay(
            positions, observed, break_before=np.array([False, True, True])
        )
    )
    with Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=[100.0, 100.5],
        overlays=[overlay],
    ) as scene:
        prepared = PreparedScene(scene)
        item = scene._overlay_data.positions[0]
        captures = []

        def capture(time):
            captures.append((time, prepared.sample(item, item.data, time)))
            return np.zeros((2, 2, 3), dtype=np.uint8)

        # The same arbitrary-time scientific query remains unsupported.
        assert np.isnan(
            prepared.sample(item, item.data, np.nextafter(100.2, -np.inf))
        ).all()
        if render == "scene":
            monkeypatch.setattr(
                SceneRenderer, "render", lambda self, time: capture(time)
            )
            scene.export(
                tmp_path / "scene.gif", time_range=(100.1, 100.3), fps=10, dpi=25
            )
        else:

            @contextmanager
            def capture_session():
                yield capture

            with AnimationView._from_callbacks(
                scene,
                SimpleNamespace(),
                seek=lambda time: None,
                close=lambda: None,
                capture_session=capture_session,
            ) as view:
                view.select_time(100.1, 100.3).export(tmp_path / "view.gif", fps=10)

        np.testing.assert_array_equal([time for time, _ in captures], observed[:2])
        np.testing.assert_array_equal([row for _, row in captures], positions[:2])
        assert all(time < observed[-1] for time, _ in captures)


def test_export_clock_enumerates_prepared_and_inherited_overlay_clocks():
    # Adapter-prepared clocks are authoritative, rather than adapter input
    # attributes. The inherited scene clock is used for explicitly aligned rows.
    scene_clock = np.array([100.0, 100.5])
    native = np.array([100.1, 100.2, 100.3])
    field_clock = np.array([100.0, 100.25, 100.5])
    inherited = SimpleNamespace(times=None)
    sampled = [SimpleNamespace(times=native.copy()) for _ in range(6)]
    data = SimpleNamespace(
        positions=[inherited, sampled[0]],
        bodypart_sets=[sampled[1]],
        head_directions=[sampled[2]],
        object_vectors=[sampled[3]],
        videos=[sampled[4]],
        timeseries=[sampled[5]],
        events=[
            SimpleNamespace(event_times={"empty": np.array([]), "lick": native[::-1]})
        ],
        spike_rasters=[SimpleNamespace(spike_times=[np.array([]), native])],
        epochs=[SimpleNamespace(intervals=np.array([[100.1, 100.2], [100.2, 100.3]]))],
    )
    scene = SimpleNamespace(
        frame_times=scene_clock,
        fields={"field": SimpleNamespace(times=field_clock)},
        _overlay_data=data,
    )
    clocks = list(iter_observation_clocks(scene))
    assert clocks[0] is field_clock
    assert clocks[1] is scene_clock
    assert all(
        clock is item.times for clock, item in zip(clocks[2:8], sampled, strict=True)
    )
    np.testing.assert_array_equal(clocks[8], native)
    np.testing.assert_array_equal(clocks[9], native)
    np.testing.assert_array_equal(clocks[10], [100.1, 100.2])
    np.testing.assert_array_equal(clocks[11], [100.2, 100.3])


def test_point_clock_enumeration_uses_bounded_chunks_and_keeps_sorted_views():
    from neurospatial.animation._prepared import _ordered_clock_chunks

    native = np.arange(200_000, dtype=float)
    chunks = list(_ordered_clock_chunks(native))
    assert max(map(len, chunks)) <= 65_536
    assert sum(map(len, chunks)) == len(native)
    assert all(np.shares_memory(chunk, native) for chunk in chunks)
    reversed_chunks = list(_ordered_clock_chunks(native[::-1]))
    assert max(map(len, reversed_chunks)) <= 65_536
    assert all(np.all(chunk[1:] >= chunk[:-1]) for chunk in reversed_chunks)


def test_empty_overlay_clocks_and_conflicting_source_floats(tmp_path):
    source = np.array([100.1, 100.3])
    query = 100.1 + 0.1
    observation = np.array([100.2])
    plan = prepare_export(
        tmp_path / "empty.gif",
        source,
        fps=10,
        sample_times=iter([np.array([]), observation]),
    )
    assert plan.times[1] == observation[0]
    # An exact source query wins over a distinct, nearby overlay float.
    source = np.array([100.1, query, 100.3])
    plan = prepare_export(
        tmp_path / "conflict.gif", source, fps=10, sample_times=[observation]
    )
    assert plan.times[1] == query
    assert plan.times[1] != observation[0]
