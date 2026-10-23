"""Prepared streams share support and panel data across complete-scene consumers."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import (
    Animation,
    AnimationField,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
)
from neurospatial.animation._capabilities import require_scene_view
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._sampling import sample_at_times
from neurospatial.animation._scene_rendering import SceneRenderer
from neurospatial.animation._tracks import track_extents
from neurospatial.animation.overlays import VideoData


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


class MappedVideo:
    """A custom source whose native samples reorder and omit reader frames."""

    times = np.array([0.0, 0.1, 0.2])
    interp = "nearest"

    def convert_to_data(self, frame_times, n_frames, env):
        frames = np.stack(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in [20, 100, 200]]
        )
        return VideoData(
            np.array([2, -1, 0]),
            frames,
            None,
            (0, 2, 0, 2),
            1.0,
            "above",
            times=self.times,
        )


def test_scene_video_uses_prepared_native_mapping_for_every_read(env):
    scene = Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0.0, 0.2], overlays=[MappedVideo()]
    )
    prepared = PreparedScene(scene)
    video = scene._overlay_data.videos[0]
    np.testing.assert_array_equal(video.frame_indices, [2, -1, 0])
    assert prepared.video_frame(video, -0.001) is None
    assert prepared.video_frame(video, 0.201) is None
    try:
        for time, expected in [(0.0, 200), (0.05, 200), (0.1, None), (0.2, 20)]:
            scene.render_frame(time, dpi=25)
            images = [
                image
                for image in scene._renderer.figure.axes[0].images
                if image.get_visible()
            ]
            if expected is None:
                assert not images
            else:
                assert images[0].get_array()[0, 0, 0] == expected
    finally:
        scene.close()


def test_1d_headings_are_refused_when_the_scene_is_built():
    # No view of a 1D scene draws them, and render_frame and export share the
    # widget's renderer: the scene refuses them before any frame or file.
    env = Environment.from_grid_mask(np.ones(3, dtype=bool), (np.arange(4.0),))
    times = np.array([0.0, 0.1])
    with pytest.raises(
        ValueError, match=r"1D.*HeadDirectionOverlay.*TimeSeriesOverlay"
    ):
        Animation(
            env,
            np.ones((2, env.n_bins)),
            frame_times=times,
            overlays=[HeadDirectionOverlay(np.array([0.0, 1.0]), times=times)],
        )


def test_1d_positions_and_object_vectors_have_defined_spatial_baseline():
    env = Environment.from_grid_mask(np.ones(3, dtype=bool), (np.arange(4.0),))
    times = np.array([0.0, 0.1])
    scene = Animation(
        env,
        np.ones((2, env.n_bins)),
        frame_times=times,
        overlays=[
            PositionOverlay(
                np.array([[0.0], [1.0]]), times=times, color="cyan", trail_length=2
            ),
            ObjectVectorOverlay(
                np.array([[2.0]]),
                np.array([[0.0], [1.0]]),
                times=times,
                color="magenta",
            ),
        ],
    )
    require_scene_view(scene, "widget", {})
    try:
        scene.render_frame(0.05, dpi=25)
        ax = scene._renderer.figure.axes[0]
        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_rgba

        (segment,) = next(
            lines.get_segments()
            for lines in ax.collections
            if isinstance(lines, LineCollection)
            and np.allclose(lines.get_edgecolor(), to_rgba("magenta"))
        )
        np.testing.assert_allclose(segment[:, 0], [0.5, 2.0])
        np.testing.assert_array_equal(segment[:, 1], [0, 0])
        points = np.concatenate([artist.get_offsets() for artist in ax.collections])
        assert any(np.allclose(point, [2.0, 0.0]) for point in points)
        assert any(np.allclose(point, [0.5, 0.0]) for point in points)
    finally:
        scene.close()


def test_prepared_panels_keep_native_support_and_half_open_field_endpoints():
    env = Environment.from_grid_mask(np.ones(2, dtype=bool), (np.arange(3.0),))
    field = AnimationField(
        np.array([[1, np.nan], [2, 3], [100, 100], [4, 5], [6, 7]], dtype=float),
        [0.0, 0.05, 0.1, 0.2, 0.25],
        epochs=[(0, 0.1), (0.2, 0.3)],
        max_gap=None,
    )
    scene = Animation(
        env,
        {"posterior": field},
        frame_times=[0.0, 0.25],
        window_seconds=0.5,
        overlays=[
            TimeSeriesOverlay(
                np.array([0.0, 2.0, 1.0]), np.array([0.0, 0.1, 0.2]), normalize=True
            ),
            SpikeRasterOverlay([np.array([0.01, 0.2])], unit_ids=["unit"]),
        ],
    )
    prepared = PreparedScene(scene)
    assert not hasattr(prepared, "figure")
    assert prepared.scene.fields["posterior"].values is field.values
    window = prepared.track_window(scene.fields["posterior"], 0.1)
    # The unsupported (0.1, 0.2) interval and the row at 0.1 hold no cells.
    left, right = window.time_edges[:-1], window.time_edges[1:]
    assert np.isnan(window.values[:, (left < 0.2) & (right > 0.1)]).all()
    assert not np.any(window.values == 100)
    np.testing.assert_array_equal(window.values[:, left == 0.0], [[1], [np.nan]])
    np.testing.assert_array_equal(scene.sample_fields(0.075)["posterior"], [2, 3])
    assert np.isnan(scene.sample_fields(0.1)["posterior"]).all()
    signal = scene._overlay_data.timeseries[0]
    times, values = prepared.signal_window(signal, 0.1)
    np.testing.assert_array_equal(times, signal.times)
    np.testing.assert_array_equal(values, [0, 1, 0.5])
    try:
        renderer = SceneRenderer(scene, dpi=25)
        renderer.render(0.1)
        mesh = renderer.figure.axes[1].images[0]
        np.testing.assert_array_equal(mesh.get_array().filled(np.nan), window.values)
        signal_ax = renderer.figure.axes[2]
        np.testing.assert_array_equal(signal_ax.lines[0].get_ydata(), values)
        raster_ax = renderer.figure.axes[3]
        actual = np.array(
            [segment[0, 0] for segment in raster_ax.collections[0].get_segments()]
        )
        np.testing.assert_array_equal(
            actual, prepared.raster_window(scene._overlay_data.spike_rasters[0], 0, 0.1)
        )
        renderer.close()
    finally:
        scene.close()


def test_sampling_trusts_prepared_support_and_reads_requested_rows(monkeypatch):
    times = np.arange(100_000, dtype=float)
    values = np.ones((len(times), 2))
    isfinite = np.isfinite
    sizes = []

    def bounded_isfinite(array, *args, **kwargs):
        sizes.append(np.asarray(array).size)
        return isfinite(array, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", bounded_isfinite)
    sample_valid = np.ones(len(times), dtype=bool)
    sample_valid[11] = False
    result = sample_at_times(
        times,
        values,
        [10.5, 12.5],
        interp="linear",
        sample_valid=sample_valid,
        interval_valid=sample_valid[:-1] & sample_valid[1:],
    )
    np.testing.assert_array_equal(result, [[np.nan, np.nan], [1.0, 1.0]])
    assert max(sizes, default=0) <= 4


def test_fresh_napari_preparation_imports_no_canvas_or_renderer():
    script = textwrap.dedent("""
        import sys
        import numpy as np
        from neurospatial import Environment
        from neurospatial.animation import Animation, AnimationField
        from neurospatial.animation.backends import _napari_scene
        env = Environment.from_grid_mask(np.ones((2, 2), dtype=bool), (np.arange(3.), np.arange(3.)))
        track = Environment.linear_track(endpoints=[[0, .5], [2, .5]], bin_size=1)
        scene = Animation(env, AnimationField(np.ones((2, env.n_bins)), [0, .1]), track=track)
        prepared = _napari_scene.PreparedScene(scene)
        assert not hasattr(prepared, 'figure')
        assert not hasattr(_napari_scene, 'SceneRenderer')
        assert 'matplotlib.backends.backend_agg' not in sys.modules
        assert 'matplotlib.pyplot' not in sys.modules
        assert prepared.track_window(scene.fields['field'], .05).values.shape == (2, 1)
        scene.close()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_overlapping_track_windows_reuse_rows_with_bounded_cache():
    class Rows:
        shape = (500, 2)

        def __init__(self):
            self.reads = []

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, index):
            self.reads.append(index)
            return np.array([index, index + 1.0])

    env = Environment.from_grid_mask(np.ones(2, dtype=bool), (np.arange(3.0),))
    rows = Rows()
    scene = Animation(
        env,
        AnimationField(rows, np.arange(500.0) / 100, vmin=0, vmax=500),
        window_seconds=2.0,
    )
    helper = PreparedScene(scene)
    rows.reads.clear()
    field = scene.fields["field"]
    helper.track_window(field, 1.0)
    first = set(rows.reads)
    rows.reads.clear()
    helper.track_window(field, 1.01)
    assert first.isdisjoint(rows.reads)
    assert len(rows.reads) <= 2
    assert len(first) > 128
    helper._row_budget = 2048
    for time in np.arange(1.0, 4.5, 0.01):
        helper.track_window(field, time)
    assert len(helper._rows) <= 4096
    assert helper._row_bytes <= helper._row_budget
    scene.close()


def test_cached_field_rows_own_snapshots_of_reused_source_buffers(env):
    class ScratchRows:
        shape = (3, 4)

        def __init__(self):
            self.buffer = np.empty(4)

        def __len__(self):
            return 3

        def __getitem__(self, index):
            self.buffer.fill(index)
            return self.buffer

    scene = Animation(
        env, AnimationField(ScratchRows(), [0.0, 0.1, 0.2], vmin=0, vmax=2)
    )
    helper = PreparedScene(scene)
    zero = helper.field_samples(0.0)["field"]
    helper.field_samples(0.1)
    assert np.all(zero == 0)
    assert helper.field_samples(0.0)["field"] is zero
    np.testing.assert_array_equal(zero, np.zeros(4))
    scene.close()


def _below_video(times):
    return VideoData(
        np.zeros(len(times), dtype=int),
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        None,
        (0.0, 2.0, 0.0, 2.0),
        1.0,
        "below",
        times=np.asarray(times, dtype=float),
    )


class _Video:
    """A custom overlay returning prepared video data."""

    times = None
    interp = "nearest"

    def __init__(self, data):
        self.data = data

    def convert_to_data(self, frame_times, n_frames, env):
        return self.data


def test_one_see_through_rule_fades_from_the_color_floor_in_every_view(env):
    """Opacity rises from vmin to the typical peak, the same in every view."""
    from neurospatial.environment._fastplotlib import ColorScale, perceived_alpha

    times = [0.0, 0.1, 0.2]
    # Negative values: fading from zero instead of vmin would hide them.
    values = np.array([[-2.0, 0.0, 1.0, 2.0], [-2.0, 1.0, 2.0, 2.0], [0.0] * 4])
    scene = Animation(
        env,
        AnimationField(values, times, vmin=-2, vmax=2),
        overlays=[_Video(_below_video(times))],
    )
    scale = PreparedScene(scene).field_scales["field"]
    assert scale.opaque_from == 2.0  # median of the frames' maxima
    alpha = scale.colors()[:, 3]
    np.testing.assert_allclose(alpha, np.linspace(0, 1, 256))
    # fastplotlib blends in linear light: the same rule, gamma-corrected.
    lut = ColorScale(
        scale.cmap, scale.vmin, scale.vmax, "lightgray", opaque_from=scale.opaque_from
    ).lut[1:, 3]
    np.testing.assert_allclose(
        lut, perceived_alpha(np.linspace(0, 1, len(lut))), atol=1e-6
    )
    scene.close()


def test_colorbars_mark_the_ends_that_sampled_values_pass(env):
    track = Environment.linear_track(endpoints=[[0.0, 1.0], [2.0, 1.0]], bin_size=1.0)
    values = np.array([[-1.0, 0.5, 0.5, 3.0], [0.2, 0.4, 0.6, 0.8]])
    scene = Animation(
        env,
        {
            "clipped": AnimationField(values, [0.0, 0.1], vmin=0, vmax=1),
            "inside": AnimationField(values.clip(0, 1), [0.0, 0.1], vmin=0, vmax=1),
        },
        track=track,
    )
    prepared = PreparedScene(scene)
    assert prepared.field_scales["clipped"].extend == (True, True)
    assert prepared.field_scales["inside"].extend == (False, False)
    # The projection's limits are its own; its sums pass neither end here.
    projected = prepared.track_scales["clipped"]
    assert projected.label == "Projected clipped (sum per track bin)"
    scene.render_frame(0.0, dpi=30)
    colorbars = {
        ax.get_ylabel(): ax._colorbar.extend
        for ax in scene._renderer.figure.axes
        if hasattr(ax, "_colorbar")
    }
    assert colorbars["clipped"] == "both"
    assert colorbars["inside"] == "neither"
    assert colorbars[projected.label] == projected.extend_name
    scene.close()


class _CountedRows:
    """A lazy source that counts its row reads."""

    def __init__(self, n_bins):
        self.n_bins, self.reads = n_bins, 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.reads += 1
        return np.full(self.n_bins, float(index))


def test_projected_scale_statistics_are_read_once_and_travel_with_the_field(env):
    import pickle

    track = Environment.linear_track(endpoints=[[0.0, 1.0], [2.0, 1.0]], bin_size=1.0)
    rows = _CountedRows(env.n_bins)
    scene = Animation(env, AnimationField(rows, [0.0, 0.1, 0.2]), track=track)
    first = PreparedScene(scene).track_scales["field"]
    reads = rows.reads
    assert PreparedScene(scene).track_scales["field"] == first
    assert rows.reads == reads
    # A worker process rebuilds the scene from pickled fields.
    field = pickle.loads(pickle.dumps(scene.fields["field"]))
    rebuilt = Animation(env, {"field": field}, track=track)
    assert PreparedScene(rebuilt).track_scales["field"] == first
    assert field.values.reads == reads
    scene.close()
    rebuilt.close()


def test_heading_arrows_start_at_the_one_position_else_the_arena_center(env):
    times = np.array([0.0, 0.1, 0.2])
    headings = np.array([0.0, np.pi / 2, np.nan])
    tracked = Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=times,
        overlays=[
            PositionOverlay(np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]), times),
            HeadDirectionOverlay(headings, times, length=2.0),
        ],
    )
    compass = Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=times,
        overlays=[HeadDirectionOverlay(headings, times, length=2.0)],
    )
    for scene, origin in ((tracked, [1.0, 1.0]), (compass, [1.0, 1.0])):
        prepared = PreparedScene(scene)
        heading = scene._overlay_data.head_directions[0]
        start, displacement = prepared.heading_arrow(heading, 0.1)
        np.testing.assert_allclose(start, origin)
        np.testing.assert_allclose(displacement, [0.0, 2.0], atol=1e-12)
        # A missing heading draws no arrow.
        assert prepared.heading_arrow(heading, 0.2) is None
        scene.close()
    prepared = PreparedScene(tracked)
    heading = tracked._overlay_data.head_directions[0]
    np.testing.assert_allclose(prepared.heading_arrow(heading, 0.0)[0], [0.5, 0.5])


@pytest.mark.parametrize("decay", [None, 0, 1, 3])
def test_shown_events_follow_the_frame_rule_for_unordered_records(env, decay):
    """The shown slice equals the per-event rule, for records in any order."""
    from neurospatial.animation import EventOverlay
    from neurospatial.animation._style import _event_visibility

    rng = np.random.default_rng(4)
    frame_times = np.sort(rng.uniform(0, 2, 30))
    event_times = rng.uniform(frame_times[0], frame_times[-1], 60)
    positions = rng.uniform(0, 2, (60, 2))
    positions[::5] = np.nan  # events without a tracked location
    overlay = EventOverlay.from_fixed_positions(event_times, positions)
    overlay.decay_frames = decay
    scene = Animation(
        env, np.ones((30, env.n_bins)), frame_times=frame_times, overlays=[overlay]
    )
    prepared = PreparedScene(scene)
    events = scene._overlay_data.events[0]
    (name,) = events.event_positions
    times, located = events.event_times[name], events.event_positions[name]
    frames = np.maximum(np.searchsorted(frame_times, times, "right") - 1, 0)
    keys = set()
    for time in np.linspace(frame_times[0], frame_times[-1], 97):
        frame = int(np.searchsorted(frame_times, time, "right")) - 1
        visible, alphas = _event_visibility(frames, frame, decay)
        visible &= (times <= time) & np.isfinite(located).all(axis=1)
        order = np.argsort(times[visible], kind="stable")
        shown = prepared.events(events, time)[name]
        np.testing.assert_array_equal(shown.positions, located[visible][order])
        # A view drawing every located event once selects the same ones.
        np.testing.assert_array_equal(
            prepared.event_positions(events)[name][shown.rows], shown.positions
        )
        np.testing.assert_array_equal(
            shown.opacity, (alphas[visible] * events.opacity)[order]
        )
        keys.add(shown.key)
    assert len(keys) > 1
    scene.close()


def test_signal_panels_share_one_limit_rule(env):
    times = np.linspace(0, 1, 11)
    scene = Animation(
        env,
        np.ones((11, env.n_bins)),
        frame_times=times,
        overlays=[
            TimeSeriesOverlay(np.full(11, 3.0), times, label="constant"),
            # A speed's tiny floor reads as zero.
            TimeSeriesOverlay(2e-6 + times * 5, times, label="speed"),
            TimeSeriesOverlay(times * 4, times, group="g", normalize=True),
            TimeSeriesOverlay(times - 2, times, group="g"),
        ],
    )
    prepared = PreparedScene(scene)
    limits = [
        prepared.signal_limits(group) for group in prepared.signal_groups.values()
    ]
    assert limits == [(2.5, 3.5), (0.0, 5.000002), (-2.0, 1.0)]
    scene.render_frame(0.5, dpi=30)
    panels = [ax for ax in scene._renderer.figure.axes if ax.get_xlabel() == "Time (s)"]
    assert [ax.get_ylim() for ax in panels] == limits
    scene.close()


def test_each_overlay_stream_is_sampled_once_per_frame_for_every_panel(
    env, monkeypatch
):
    from neurospatial.animation import _prepared

    times = np.linspace(0, 1, 11)
    positions = np.column_stack([times * 2, times * 2])
    scene = Animation(
        env,
        {name: AnimationField(np.ones((11, env.n_bins)), times) for name in "abc"},
        overlays=[
            PositionOverlay(positions, times, trail_length=3),
            HeadDirectionOverlay(times * 3, times),
        ],
    )
    scene.render_frame(0.0, dpi=30)
    calls = []
    sample = _prepared.sample_observation

    def counted(item, values, time, **kwargs):
        calls.append(type(item).__name__)
        return sample(item, values, time, **kwargs)

    monkeypatch.setattr(_prepared, "sample_observation", counted)
    scene.render_frame(0.45, dpi=30)
    # Three panels draw the marker, trail and arrow from one sample each.
    assert sorted(calls) == ["HeadDirectionData", "PositionData"]
    scene.close()


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_parts_sampled_together_equal_each_part_sampled_alone(interp):
    from neurospatial.animation._sampling import resolve_support, sample_parts

    rng = np.random.default_rng(5)
    times = np.sort(rng.uniform(0, 1, 40))
    parts, valid, intervals = [], [], []
    for _ in range(4):
        rows = rng.normal(size=(40, 2))
        rows[rng.random(40) < 0.2] = np.nan
        rows, sample_valid, interval_valid = resolve_support(
            rows, rng.random(40) > 0.1, rng.random(39) > 0.1
        )
        parts.append(rows)
        valid.append(sample_valid)
        intervals.append(interval_valid)
    queries = np.r_[times, rng.uniform(-0.1, 1.1, 200)]
    for query in queries:
        together = sample_parts(
            times,
            parts,
            np.column_stack(valid),
            np.column_stack(intervals),
            query,
            interp=interp,
        )
        alone = [
            sample_at_times(
                times,
                rows,
                np.array([query]),
                interp=interp,
                sample_valid=sample_valid,
                interval_valid=interval_valid,
            )[0]
            for rows, sample_valid, interval_valid in zip(
                parts, valid, intervals, strict=True
            )
        ]
        np.testing.assert_array_equal(together, alone)


def test_track_window_without_a_shown_interval_is_empty_and_defined():
    track = Environment.linear_track(
        node_positions=[[0.0, 0.0], [2.0, 0.0]], bin_size=1.0
    )
    values = np.ones((3, track.n_bins))
    # The 4.9 s interval is longer than max_gap, so it holds no sample.
    field = AnimationField(values, [0.0, 0.1, 5.0], max_gap=0.5)
    scene = Animation(track, field, window_seconds=1.0)
    window = PreparedScene(scene).track_window(scene.fields["field"], 2.5)
    np.testing.assert_array_equal(window.time_edges, [2.0])  # the window's start
    assert window.observed.shape == (0,)
    assert window.values.shape == (len(window.track_edges) - 1, 0)
    scene.render_frame(2.5, dpi=20)
    scene.close()


def test_track_mesh_rows_follow_the_track_and_leave_segment_spacing_empty():
    track = Environment.linear_track(
        node_positions=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
        bin_size=1.0,
        edge_spacing=1.5,
    )
    values = np.arange(3 * track.n_bins, dtype=float).reshape(3, -1)
    scene = Animation(
        track, AnimationField(values, [0.0, 0.1, 0.2]), window_seconds=1.0
    )
    window = PreparedScene(scene).track_window(scene.fields["field"], 0.15)
    starts, stops = track_extents(track)
    # Four bins and the spacing between the two segments, bottom to top.
    np.testing.assert_allclose(window.track_edges, [0, 1, 2, 3.5, 4.5, 5.5])
    np.testing.assert_allclose(window.time_edges, [0.0, 0.1, 0.2])
    np.testing.assert_array_equal(window.values[2], [np.nan, np.nan])
    np.testing.assert_array_equal(window.values[[0, 1, 3, 4]], values[:2].T)
    np.testing.assert_allclose(starts, window.track_edges[[0, 1, 3, 4]])
    np.testing.assert_allclose(stops, window.track_edges[[1, 2, 4, 5]])
    edges, bin_rows = PreparedScene(scene).track_rows
    np.testing.assert_array_equal(bin_rows, [0, 1, 3, 4])
    np.testing.assert_allclose(edges, window.track_edges)
    scene.close()


def test_track_columns_say_which_hold_a_sample_and_follow_the_window():
    track = Environment.linear_track(endpoints=[[0.0, 0.5], [2.0, 0.5]], bin_size=1.0)
    times = [0.0, 0.1, 0.2, 0.3]
    field = AnimationField(
        np.ones((4, track.n_bins)), times, break_before=[False, False, True, False]
    )
    scene = Animation(track, field, window_seconds=1.0)
    prepared = PreparedScene(scene)
    window = prepared.track_window(scene.fields["field"], 0.15)
    # The break before 0.2 leaves an empty column between two held samples.
    np.testing.assert_allclose(window.time_edges, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(window.observed, [True, False, True])
    assert np.isnan(window.values[:, ~window.observed]).all()
    # A view may narrow the window every time panel shows.
    prepared.window_seconds = 0.1
    assert prepared.bounds(0.05) == pytest.approx((0.0, 0.1))
    narrow = prepared.track_window(scene.fields["field"], 0.05)
    np.testing.assert_allclose(narrow.time_edges, [0.0, 0.1])
    scene.close()


def test_track_paths_break_where_the_position_changes_segment():
    # Two segments 1.5 apart along the track: a line from one to the other
    # would cross positions along the track that were never visited.
    track = Environment.linear_track(
        node_positions=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
        bin_size=1.0,
        edge_spacing=1.5,
    )
    times = np.arange(4) * 0.1
    along = np.array([0.5, 1.5, 4.0, 5.0])  # 0-1, 0-1, then 1-2, 1-2
    position = PositionOverlay(track.linear_to_nd(along), times=times)
    with Animation(
        track,
        AnimationField(np.zeros((4, track.n_bins)), times),
        overlays=[position],
        window_seconds=2.0,
    ) as scene:
        paths = PreparedScene(scene).track_window(scene.fields["field"], 0.15).paths
    np.testing.assert_allclose(paths[:, :, 1], [[0.5, 1.5], [4.0, 5.0]], atol=1e-9)
