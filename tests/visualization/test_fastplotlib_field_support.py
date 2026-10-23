"""Native track panels retain each field's observation clock and support."""

from __future__ import annotations

import os

os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")

import numpy as np
import pytest

pytest.importorskip("fastplotlib")
wgpu = pytest.importorskip("wgpu")
try:
    _ADAPTER = wgpu.gpu.request_adapter_sync()
except Exception:
    _ADAPTER = None
if _ADAPTER is None:
    pytest.skip("no GPU adapter for fastplotlib", allow_module_level=True)

from neurospatial import Environment  # noqa: E402
from neurospatial.animation import Animation, AnimationField  # noqa: E402


class LazyRows:
    """Record bounded source reads; reject accidental eager materialization."""

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.values[index]

    def __array__(self, *args, **kwargs):
        raise AssertionError("A field source must remain lazy")


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 1), dtype=bool),
        grid_edges=(np.array([0.0, 10, 20]), np.array([0.0, 10])),
    )


def test_native_track_respects_breaks_with_coarse_scene_clock(env):
    track = Environment.linear_track(endpoints=[[0.0, 5], [20.0, 5]], bin_size=10)
    field = AnimationField(
        np.ones((4, env.n_bins)),
        [0.0, 0.1, 0.2, 0.3],
        break_before=[False, False, True, False],
        vmin=0,
        vmax=1,
    )
    scene = Animation(env, {"posterior": field}, frame_times=[0.0, 0.3], track=track)
    view = scene.show(backend="fastplotlib")
    try:
        native = view.native
        native.capture_frame(0.15)
        panel = native._tracks[0]
        np.testing.assert_allclose(panel.clock, field.times)
        np.testing.assert_allclose(panel.ends, [0.1, 0.1, 0.3, 0.3])
        assert np.isnan(scene.sample_fields(0.15)["posterior"]).all()
        starts = panel.corners[:, 0, 0, 0]
        stops = panel.corners[:, 0, 1, 0]
        assert not np.any((starts <= 0.15) & (stops > 0.15))
        # Exactly observed samples remain available without painting the break.
        np.testing.assert_array_equal(scene.sample_fields(0.1)["posterior"], 1.0)
        assert native.time == pytest.approx(0.0)  # capture restores viewer time
    finally:
        scene.close()


def test_track_windows_use_independent_sources_and_reuse_only_overlap(env):
    track = Environment.linear_track(endpoints=[[0.0, 5], [20.0, 5]], bin_size=10)
    sources = [
        LazyRows(np.arange(2000, dtype=float).reshape(1000, 2)),
        LazyRows(np.arange(800, dtype=float).reshape(400, 2)),
    ]
    fields = [
        AnimationField(sources[0], 100 + np.arange(1000) * 0.01, max_gap=None),
        AnimationField(sources[1], 100 + np.arange(400) * 0.025, max_gap=None),
    ]
    scene = Animation(
        env,
        {"first": fields[0], "second": fields[1]},
        frame_times=[100.0, 109.9],
        track=track,
        window_seconds=0.2,
    )
    native = scene.show(backend="fastplotlib").native
    try:
        assert len(native._tracks[0].clock) == 1000
        assert len(native._tracks[1].clock) == 400
        for time in (0.3, 0.32, 0.6, 0.31, 8.0):
            data = native._read(time)
            for k, panel in enumerate(native._tracks):
                window = panel.frames_at(time)
                assert data.first[k] == window.start
                np.testing.assert_array_equal(data.rows[k], sources[k].values[window])
        for source in sources:
            source.reads.clear()
        native._read(8.01)
        for source in sources:
            slices = [index for index in source.reads if isinstance(index, slice)]
            assert sum(index.stop - index.start for index in slices) <= 2
        assert native._last_windows[0][1].shape[0] <= 22
        assert native._last_windows[1][1].shape[0] <= 10
    finally:
        native.close()
        native.close()
        assert native.figure.canvas.get_closed()


def test_isolated_field_samples_have_no_positive_duration_cells(env):
    track = Environment.linear_track(endpoints=[[0.0, 5], [20.0, 5]], bin_size=10)
    field = AnimationField(np.ones((3, 2)), [0.0, 0.1, 0.2], max_gap=0.05)
    native = Animation(env, field, track=track).show(backend="fastplotlib").native
    try:
        panel = native._tracks[0]
        np.testing.assert_array_equal(panel.ends, panel.clock)
        assert np.isnan(native._read(0.05).values[0]).all()
        assert np.all(panel.corners[:, :, 0, 0] == panel.corners[:, :, 1, 0])
    finally:
        native.close()
        native.close()
        assert native.figure.canvas.get_closed()


def test_native_epoch_masks_and_projected_scales_use_supported_source_rows(env):
    from neurospatial.animation._prepared import PreparedScene

    track = Environment.linear_track(endpoints=[[0.0, 5], [20.0, 5]], bin_size=10)
    field = AnimationField(
        np.repeat(
            np.array([999.0, 10.0, 20.0, 999.0, 999.0])[:, None], env.n_bins, axis=1
        ),
        [0.0, 0.05, 0.1, 0.15, 0.2],
        epochs=[(0.05, 0.15)],
        max_gap=None,
    )
    scene = Animation(env, {"posterior": field}, frame_times=[0.0, 0.2], track=track)
    view = scene.show(backend="fastplotlib")
    try:
        native = view.native
        prepared = PreparedScene(scene)
        scale = prepared.track_scales["posterior"]
        expected = (scale.vmin, scale.vmax)
        assert expected[0] > 1  # Neither coarse scene frame observes a valid row.
        assert (native._track_scales[0].vmin, native._track_scales[0].vmax) == expected
        data = native._read(0.1)
        np.testing.assert_array_equal(data.values[0], 20)
        np.testing.assert_array_equal(data.rows[0][1:3], [[10, 10], [20, 20]])
        assert np.isnan(data.rows[0][[0, 3, 4]]).all()
        assert np.isnan(native._last_windows[0][1][[0, 3, 4]]).all()
        # The excluded stop is missing exactly, while its preceding held cell
        # remains supported throughout [0.1, 0.15).
        assert np.isnan(native._read(0.15).values[0]).all()
        np.testing.assert_array_equal(native._read(0.125).values[0], 20)
        panel = native._tracks[0]
        assert panel.ends[2] == pytest.approx(0.15)
        native.capture_frame(0.125)
    finally:
        scene.close()
