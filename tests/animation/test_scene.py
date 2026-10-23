"""Physical-clock scenes retain support, identity and complete export content."""

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import (
    BodypartOverlay,
    EpochOverlay,
    EventOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
    VideoOverlay,
)
from neurospatial.animation.scene import Animation, AnimationField


@pytest.fixture
def scene_env():
    points = np.random.default_rng(42).uniform(0, 10, (100, 2))
    return Environment.from_samples(points, bin_size=2)


def test_lazy_source_is_not_materialized_and_metadata_is_owned(scene_env):
    class LazyRows:
        shape = (1000, scene_env.n_bins)

        def __init__(self):
            self.reads = []

        def __len__(self):
            return 1000

        def __array__(self, *args):
            raise AssertionError("Must not materialize the source")

        def __getitem__(self, index):
            self.reads.append(index)
            return np.full(scene_env.n_bins, index)

    values = LazyRows()
    times = np.arange(1000) * 0.1
    breaks = np.zeros(1000, dtype=bool)
    breaks[3] = True
    field = AnimationField(values, times, break_before=breaks)
    scene = Animation(scene_env, {"posterior": field})
    assert values.reads == []
    times[0] = -9
    breaks[3] = False
    assert field.times[0] == 0
    assert field.break_before[3]
    assert not field.times.flags.writeable
    assert not field.interval_valid.flags.writeable
    np.testing.assert_array_equal(scene.sample_fields(0.15)["posterior"], 1)
    assert values.reads == [1]
    assert np.isnan(scene.sample_fields(0.25)["posterior"]).all()
    assert values.reads == [1]


def test_stream_support_and_independent_clocks(scene_env):
    n = scene_env.n_bins
    first = AnimationField(
        np.array([np.zeros(n), np.ones(n), np.full(n, 2)]), [0, 0.2, 2]
    )
    second = AnimationField(np.array([np.full(n, 10), np.full(n, 11)]), [0.1, 0.3])
    scene = Animation(scene_env, {"first": first, "second": second})
    np.testing.assert_array_equal(scene.frame_times, [0, 0.1, 0.2, 0.3, 2])
    np.testing.assert_array_equal(scene.sample_fields(0.15)["first"], 0)
    np.testing.assert_array_equal(scene.sample_fields(0.15)["second"], 10)
    assert np.isnan(scene.sample_fields(1)["first"]).all()
    assert np.isnan(scene.sample_fields(-0.1)["first"]).all()
    np.testing.assert_array_equal(scene.sample_fields(2)["first"], 2)
    assert np.isnan(scene.sample_fields(2.01)["first"]).all()


def test_memmap_stays_caller_owned(scene_env, tmp_path):
    source = np.memmap(
        tmp_path / "values.dat", mode="w+", shape=(3, scene_env.n_bins), dtype=float
    )
    source[:] = 1
    field = AnimationField(source, [0, 0.1, 0.2])
    with Animation(scene_env, {"rate": field}):
        assert field.values is source
    assert not source._mmap.closed
    source._mmap.close()


def test_selected_export_matches_same_complete_scene_frame(
    scene_env, tmp_path, monkeypatch
):
    times = np.arange(5) * 0.1
    positions = np.column_stack((np.arange(5) + 1, np.arange(5) + 1))
    overlays = [
        PositionOverlay(positions, times, trail_length=3),
        BodypartOverlay({"nose": positions + 0.2}, times),
        HeadDirectionOverlay(np.zeros(5), times, length=1, width=0.05),
        ObjectVectorOverlay(np.array([[8.0, 8.0]]), positions, times, color="magenta"),
        EventOverlay.from_fixed_positions(
            np.array([0.1]), np.array([[4.0, 6.0]]), decay_frames=3
        ),
        TimeSeriesOverlay(np.arange(5.0), times, label="Speed", color="cyan"),
        SpikeRasterOverlay([np.array([0.05, 0.25]), np.array([0.15])], unit_ids=[7, 3]),
        EpochOverlay([(0.05, 0.25)], label="Run"),
        VideoOverlay(np.zeros((5, 3, 3, 3), dtype=np.uint8), times=times),
    ]
    with pytest.warns(UserWarning, match="no calibration"):
        scene = Animation(
            scene_env,
            {
                "rate": AnimationField(
                    np.arange(5 * scene_env.n_bins).reshape(5, -1), times
                ),
                "posterior": AnimationField(
                    np.ones((5, scene_env.n_bins)), times, cmap="magma"
                ),
            },
            overlays=overlays,
        )
    expected = scene.render_frame(0.2, dpi=30)
    captured = []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def append_data(self, frame):
            captured.append(frame)

    import imageio.v2

    monkeypatch.setattr(imageio.v2, "get_writer", lambda *args, **kwargs: Writer())
    scene.export(
        tmp_path / "selected.mp4", time_range=(0.2, 0.4), fps=10, speed=1, dpi=30
    )
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], expected)
    assert not np.array_equal(captured[0], captured[-1])
    assert (
        len(scene._renderer.figure.axes) == 7
    )  # two fields, two colorbars, three time panels
    scene.close()


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.nan], []])
def test_invalid_times_are_actionable(times):
    with pytest.raises(ValueError, match=r"times|timestamp|samples"):
        AnimationField(np.ones((len(times), 3)), times)


def test_plain_arrays_require_physical_times(scene_env):
    with pytest.raises(ValueError, match="frame_times"):
        Animation(scene_env, np.ones((3, scene_env.n_bins)))


def test_export_refuses_overwrite_and_invalid_ranges(scene_env, tmp_path):
    scene = Animation(
        scene_env, np.ones((3, scene_env.n_bins)), frame_times=[0, 0.1, 0.2]
    )
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        scene.export(existing)
    assert existing.read_bytes() == b"original"
    with pytest.raises(ValueError, match="inside"):
        scene.export(tmp_path / "bad.mp4", time_range=(-1, 0.1))


def test_track_cells_keep_gap_and_break_support():
    env = Environment.from_samples(np.arange(10.0)[:, None], bin_size=1)
    field = AnimationField(
        np.ones((4, env.n_bins)),
        [0, 0.1, 0.2, 0.3],
        break_before=np.array([False, False, True, False]),
    )
    scene = Animation(env, {"posterior": field})
    assert scene.render_frame(0.15, dpi=30).shape[-1] == 3
    track_ax = scene._renderer.figure.axes[-1]
    collection = track_ax.collections[0]
    paths = collection.get_paths()
    assert len(paths) == 2 * env.n_bins
    # No heatmap polygon extends into the declared recording break.
    for path in paths:
        left, right = np.min(path.vertices[:, 0]), np.max(path.vertices[:, 0])
        assert right <= 0.1 or left >= 0.2
    scene.close()
    with pytest.raises(RuntimeError, match="closed"):
        scene.sample_fields(0.1)


def test_decoding_uses_result_clock_and_probability_label(scene_env):
    from neurospatial.decoding import DecodingResult

    times = np.array([2.0, 2.1, 2.2])
    result = DecodingResult(
        np.full((3, scene_env.n_bins), 1 / scene_env.n_bins), scene_env, times
    )
    scene = Animation.from_decoding(result, cmap="magma")
    np.testing.assert_array_equal(scene.frame_times, times)
    assert scene.fields["posterior"].values is result.posterior
    assert scene.fields["posterior"].label == "Probability"
    assert scene.fields["posterior"].cmap == "magma"


def test_overlay_metadata_is_a_scene_snapshot(scene_env):
    overlay = SpikeRasterOverlay(
        [np.array([0.1]), np.array([0.2])], unit_ids=np.array([8, 3])
    )
    scene = Animation(
        scene_env,
        np.ones((3, scene_env.n_bins)),
        frame_times=[0, 0.1, 0.2],
        overlays=[overlay],
    )
    overlay._rows[1][0] = 99
    np.testing.assert_array_equal(
        scene.overlay_data().spike_rasters[0].unit_ids, [8, 3]
    )


def test_events_start_at_actual_time_and_keep_cumulative_and_fading_modes(scene_env):
    times = np.arange(5) * 0.1
    event = EventOverlay.from_fixed_positions(np.array([0.15]), np.array([[4.0, 4.0]]))
    scene = Animation(
        scene_env, np.ones((5, scene_env.n_bins)), frame_times=times, overlays=[event]
    )
    scene.render_frame(0.14, dpi=30)
    assert len(scene._renderer.figure.axes[0].collections) == 1
    scene.render_frame(0.15, dpi=30)
    assert len(scene._renderer.figure.axes[0].collections) == 2
    scene.render_frame(0.4, dpi=30)
    assert len(scene._renderer.figure.axes[0].collections) == 2
    scene.close()
    event.decay_frames = 2
    fading = Animation(
        scene_env, np.ones((5, scene_env.n_bins)), frame_times=times, overlays=[event]
    )
    fading.render_frame(0.15, dpi=30)
    newest = fading._renderer.figure.axes[0].collections[-1].get_alpha()
    fading.render_frame(0.25, dpi=30)
    older = fading._renderer.figure.axes[0].collections[-1].get_alpha()
    assert 0 < older < newest
    fading.close()


def test_video_below_field_is_visible_through_zero_values(scene_env):
    frames = np.full((3, 4, 4, 3), [255, 0, 0], dtype=np.uint8)
    overlay = VideoOverlay(
        frames, times=np.array([0.0, 0.1, 0.2]), z_order="below", alpha=1
    )
    values = np.zeros((3, scene_env.n_bins))
    with pytest.warns(UserWarning, match="no calibration"):
        scene = Animation(
            scene_env, values, frame_times=[0.0, 0.1, 0.2], overlays=[overlay]
        )
    rgb = scene.render_frame(0.1, dpi=30)
    assert (
        np.count_nonzero((rgb[..., 0] > 240) & (rgb[..., 1] < 10) & (rgb[..., 2] < 10))
        > 20
    )
    scene.close()


def test_export_does_not_replace_file_created_during_rendering(
    scene_env, tmp_path, monkeypatch
):
    import imageio.v2

    scene = Animation(
        scene_env, np.ones((3, scene_env.n_bins)), frame_times=[0.0, 0.1, 0.2]
    )
    path = tmp_path / "raced.mp4"

    class Writer:
        def __enter__(self):
            path.write_bytes(b"created by another writer")
            return self

        def __exit__(self, *exc):
            return False

        def append_data(self, frame):
            pass

    monkeypatch.setattr(imageio.v2, "get_writer", lambda *args, **kwargs: Writer())
    with pytest.raises(FileExistsError):
        scene.export(path, fps=10, dpi=30)
    assert path.read_bytes() == b"created by another writer"



@pytest.mark.parametrize("raced", [False, True])
def test_export_without_hard_links_still_refuses_to_replace(
    scene_env, tmp_path, monkeypatch, raced
):
    # exFAT, FAT32 and some network mounts cannot hard-link the finished file.
    import errno

    import imageio.v2

    import neurospatial.animation._export as export_module

    scene = Animation(
        scene_env, np.ones((3, scene_env.n_bins)), frame_times=[0.0, 0.1, 0.2]
    )
    path = tmp_path / "drive.mp4"

    class Writer:
        def __init__(self, temporary):
            self.temporary = temporary

        def __enter__(self):
            if raced:
                path.write_bytes(b"created by another writer")
            return self

        def __exit__(self, *exc):
            with open(self.temporary, "wb") as file:
                file.write(b"encoded")
            return False

        def append_data(self, frame):
            pass

    def no_links(source, destination):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(
        imageio.v2, "get_writer", lambda temporary, **kwargs: Writer(temporary)
    )
    monkeypatch.setattr(export_module.os, "link", no_links)
    if raced:
        with pytest.raises(FileExistsError):
            scene.export(path, fps=10, dpi=30)
        assert path.read_bytes() == b"created by another writer"
    else:
        assert scene.export(path, fps=10, dpi=30) == path
        assert path.read_bytes() == b"encoded"
    assert [item.name for item in tmp_path.iterdir()] == ["drive.mp4"]
    scene.close()


def test_projected_track_uses_its_own_stable_scale(scene_env):
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 5.0))
    graph.add_node(1, pos=(10.0, 5.0))
    graph.add_edge(0, 1, distance=10.0)
    track = Environment.from_graph(graph, [(0, 1)], bin_size=5, edge_spacing=0)
    scene = Animation(
        scene_env,
        {
            "posterior": AnimationField(
                np.ones((3, scene_env.n_bins)),
                [0.0, 0.1, 0.2],
                vmin=0,
                vmax=1,
                label="Probability",
            )
        },
        track=track,
    )
    scene.render_frame(0.1, dpi=30)
    projected_ax = next(
        ax
        for ax in scene._renderer.figure.axes
        if ax.get_title() == "posterior along track"
    )
    mesh = projected_ax.collections[0]
    assert np.max(mesh.get_array()) > 1
    assert mesh.norm.vmax >= np.max(mesh.get_array())
    assert scene._renderer.limits["posterior"] == (0, 1)
    assert any(
        "sum per track bin" in ax.get_ylabel() for ax in scene._renderer.figure.axes
    )
    scene.close()
