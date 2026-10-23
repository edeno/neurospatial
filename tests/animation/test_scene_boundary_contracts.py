"""Scenes own immutable descriptors and readers, and validate native requests."""

import copy
import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.animation import Animation, AnimationField
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    VideoData,
)
from neurospatial.regions import Regions


class LazyRows:
    def __init__(self, n_bins):
        self.shape = (3, n_bins)
        self.reads = []

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        self.reads.append(index)
        return np.full(self.shape[1], index, dtype=float)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Do not materialize caller-owned field sources")

    def __deepcopy__(self, memo):
        raise AssertionError("Do not copy caller-owned field sources")


def test_independent_field_samples_own_reused_reader_buffers():
    class ScratchRows:
        shape = (2, 4)

        def __init__(self):
            self.buffer = np.empty(4)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            self.buffer.fill(index)
            return self.buffer

    source = ScratchRows()
    field = AnimationField(source, [0, 0.1])
    first = field.sample(0.0)
    second = field.sample(0.1)
    np.testing.assert_array_equal(first, np.zeros(4))
    np.testing.assert_array_equal(second, np.ones(4))
    assert not np.shares_memory(first, source.buffer)
    from neurospatial.animation._field_stats import frame_extremes

    lows, highs = frame_extremes(source)
    np.testing.assert_array_equal(lows, [0, 1])
    np.testing.assert_array_equal(highs, [0, 1])


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


def test_field_descriptor_is_frozen_and_scene_owns_metadata_without_reading(env):
    source = LazyRows(env.n_bins)
    field = AnimationField(
        source, [100, 100.1, 100.2], break_before=[False, False, True]
    )
    with Animation(env, {"rate": field}) as scene:
        owned = scene.fields["rate"]
        assert owned is not field
        assert owned.values is source
        assert source.reads == []
        for descriptor in (field, owned):
            with pytest.raises(FrozenInstanceError):
                descriptor.times = np.array([0, 1, 2])
            with pytest.raises(FrozenInstanceError):
                descriptor.max_gap = 20
            for name in ("times", "break_before", "sample_valid", "interval_valid"):
                array = getattr(descriptor, name)
                with pytest.raises(ValueError):
                    array.flags.writeable = True
        np.testing.assert_array_equal(scene.frame_times, [100, 100.1, 100.2])
        np.testing.assert_array_equal(scene.sample_fields(100.05)["rate"], 0)
        assert np.isnan(scene.sample_fields(100.15)["rate"]).all()
        assert owned.color_limits() == field.color_limits()


@pytest.mark.parametrize(
    "operation", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_field_copy_and_pickle_restore_readonly_metadata_without_materializing(
    operation,
):
    source = LazyRows(2)
    field = AnimationField(
        source, [100, 100.1, 100.2], break_before=[False, False, True]
    )
    restored = operation(field)
    assert restored.values.reads == []
    for name in ("times", "break_before", "sample_valid", "interval_valid"):
        assert not getattr(restored, name).flags.writeable
        np.testing.assert_array_equal(getattr(restored, name), getattr(field, name))
    if operation in (copy.copy, copy.deepcopy):
        assert restored.values is source
    assert np.isnan(restored.sample(100.15)).all()


def test_scene_snapshot_accepts_mixed_object_unit_ids(env):
    identifiers = np.empty(2, dtype=object)
    identifiers[:] = ["cell-a", 23]
    overlay = SpikeRasterOverlay(
        [np.array([100.05]), np.array([100.15])], unit_ids=identifiers
    )
    with Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=[100, 100.1, 100.2],
        overlays=[overlay],
    ) as scene:
        identifiers[0] = "changed"
        assert scene._overlay_data.spike_rasters[0].unit_ids.tolist() == ["cell-a", 23]
        assert not scene.overlays[0]._rows[1].flags.writeable


def test_overlay_realign_borrows_single_reader_and_cannot_mutate_authority(env):
    reader = SimpleNamespace(close=Mock())
    acquisitions = []

    class Video:
        def convert_to_data(self, frame_times, n_frames, env):
            acquisitions.append(True)
            return VideoData(
                np.arange(n_frames),
                reader,
                None,
                (0, 2, 0, 2),
                0.5,
                "above",
                times=frame_times,
            )

    scene = Animation(
        env,
        np.ones((3, env.n_bins)),
        frame_times=[100, 100.1, 100.2],
        overlays=[Video()],
    )
    assert len(acquisitions) == 1
    data = scene.overlay_data([100.05, 100.15])
    assert data is not scene._overlay_data
    assert data.videos[0].reader is reader
    # Realigned indices are new arrays; shared prepared arrays are read-only.
    data.videos[0].frame_indices[0] = -100
    assert scene._overlay_data.videos[0].frame_indices[0] == 0
    with pytest.raises(ValueError, match="read-only"):
        scene._overlay_data.videos[0].frame_indices[0] = -100
    assert len(acquisitions) == 1
    with pytest.raises(ValueError, match="within the scene recording"):
        scene.overlay_data([99, 100])
    scene.close()
    scene.close()
    # A custom adapter's reader is borrowed: the scene never closes it.
    reader.close.assert_not_called()


@pytest.mark.parametrize("category", ["pose", "headings", "vectors", "videos"])
def test_fast_1d_rejects_omitted_overlay_before_backend_allocation(
    category, monkeypatch
):
    from neurospatial.animation.backends import fastplotlib_backend

    env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1)
    times = np.array([0.0, 0.1, 0.2])
    positions = np.array([[0.0], [1.0], [2.0]])
    choices = {
        "pose": BodypartOverlay({"nose": positions}, times),
        "headings": HeadDirectionOverlay(np.zeros(3), times),
        "vectors": ObjectVectorOverlay(np.array([[2.0]]), positions, times),
    }
    if category == "videos":
        # Custom protocol overlays may supply data unsupported by a renderer.
        class Video:
            def convert_to_data(self, frame_times, n_frames, env):
                return VideoData(
                    np.arange(n_frames),
                    np.zeros((3, 2, 2, 3), dtype=np.uint8),
                    None,
                    (0, 2, 0, 2),
                    0.5,
                    "above",
                    times=frame_times,
                )

        overlay = Video()
    else:
        overlay = choices[category]
    render = Mock()
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    with (
        Animation(
            env, np.ones((3, env.n_bins)), frame_times=times, overlays=[overlay]
        ) as scene,
        pytest.raises(ValueError, match=r"cannot draw.*Overlay"),
    ):
        scene.show(backend="fastplotlib")
    render.assert_not_called()


@pytest.mark.parametrize("option", [{"layout": "grid"}, {"layout": "horizontal"}])
def test_napari_layout_accepts_overlays_and_regions(env, option, monkeypatch):
    from neurospatial.animation.backends import _napari_scene

    regions = Regions()
    regions.add("reward", point=(1, 1))
    env = env.with_regions(regions)
    build = Mock()
    monkeypatch.setattr(_napari_scene, "NapariScene", build)
    times = np.array([0, 0.1])
    with Animation(
        env,
        {"a": np.ones((2, env.n_bins)), "b": np.zeros((2, env.n_bins))},
        frame_times=times,
        overlays=[PositionOverlay(np.ones((2, 2)), times)],
        show_regions=True,
    ) as scene:
        scene.show(backend="napari", **option)
        build.assert_called_once_with(scene, **option)


def test_napari_holes_rejected_before_viewer_allocation(env, monkeypatch):
    from neurospatial.animation.backends import _napari_scene

    regions = Regions()
    regions.add(
        "arena",
        polygon=Polygon(
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
        ),
    )
    env = env.with_regions(regions)
    build = Mock()
    monkeypatch.setattr(_napari_scene, "NapariScene", build)
    with (
        Animation(
            env, np.ones((2, env.n_bins)), frame_times=[0, 0.1], show_regions=True
        ) as scene,
        pytest.raises(ValueError, match="polygons with holes"),
    ):
        scene.show(backend="napari")
    build.assert_not_called()


@pytest.mark.parametrize("backend", ["widget", "napari", "fastplotlib"])
def test_scene_unknown_options_rejected_before_native_allocation(env, backend):
    with (
        Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene,
        pytest.raises(TypeError, match=r"Unknown.*view options.*window_secs"),
    ):
        scene.show(backend=backend, window_secs=2)


def test_fast_scale_bar_rejected_before_native_allocation(env, monkeypatch):
    from neurospatial.animation.backends import fastplotlib_backend

    render = Mock()
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    with (
        Animation(
            env, np.ones((2, env.n_bins)), frame_times=[0, 0.1], scale_bar=True
        ) as scene,
        pytest.raises(ValueError, match="cannot draw scale_bar"),
    ):
        scene.show(backend="fastplotlib")
    render.assert_not_called()


def test_fast_track_override_validated_before_native_allocation(env, monkeypatch):
    from neurospatial.animation.backends import fastplotlib_backend

    track = Environment.linear_track(
        endpoints=[[0.0, 0.5], [2.0, 0.5]], bin_size=1, units="cm"
    )
    render = Mock()
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    with (
        Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene,
        pytest.raises(ValueError, match="track units"),
    ):
        scene.show(backend="fastplotlib", track=track)
    render.assert_not_called()


def test_fast_region_override_validated_before_native_allocation(env, monkeypatch):
    from neurospatial.animation.backends import fastplotlib_backend

    render = Mock()
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    with (
        Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene,
        pytest.raises(ValueError, match="missing"),
    ):
        scene.show(backend="fastplotlib", show_regions=["missing"])
    render.assert_not_called()


def test_napari_1d_rejected_before_native_allocation(monkeypatch):
    from neurospatial.animation.backends import _napari_scene

    env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1)
    build = Mock()
    monkeypatch.setattr(_napari_scene, "NapariScene", build)
    with (
        Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene,
        pytest.raises(ValueError, match="cannot draw 1D"),
    ):
        scene.show(backend="napari")
    build.assert_not_called()


def test_1d_regions_rejected_before_any_field_reads_or_readers():
    regions = Regions()
    regions.add("reward", point=(1,))
    env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1).with_regions(
        regions
    )
    values = LazyRows(env.n_bins)
    conversions = []

    class Video:
        times = None
        interp = "nearest"

        def convert_to_data(self, *args):
            conversions.append(True)
            raise AssertionError("Readers must not be acquired for an invalid scene")

    with pytest.raises(ValueError, match="cannot draw regions in a 1D environment"):
        Animation(
            env,
            AnimationField(values, [0, 0.1, 0.2]),
            overlays=[Video()],
            show_regions=True,
        )
    assert not values.reads
    assert not conversions


def test_1d_scene_without_regions_to_show_renders():
    # The scene refuses 1-D regions once, when it is built; with no regions
    # there is nothing to refuse, in the scene or its views.
    env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1)
    with Animation(
        env, np.ones((2, env.n_bins)), frame_times=[0, 0.1], show_regions=True
    ) as scene:
        assert scene.render_frame(0.0, dpi=20).ndim == 3


def test_napari_invalid_ndisplay_rejected_before_native_allocation(env, monkeypatch):
    from neurospatial.animation.backends import _napari_scene

    build = Mock()
    monkeypatch.setattr(_napari_scene, "NapariScene", build)
    with (
        Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene,
        pytest.raises(ValueError, match="ndisplay must be 2 or 3"),
    ):
        scene.show(backend="napari", ndisplay=1)
    build.assert_not_called()


@pytest.mark.napari
def test_supplied_napari_viewer_honors_explicit_title_ndisplay_and_visibility(env):
    import napari

    viewer = napari.Viewer(title="Existing viewer", ndisplay=2, show=False)
    with Animation(env, np.ones((2, env.n_bins)), frame_times=[0, 0.1]) as scene:
        try:
            with scene.show(
                backend="napari",
                viewer=viewer,
                title="Recording review",
                ndisplay=3,
                show=False,
            ) as view:
                assert view.native is viewer
                assert viewer.title == "Recording review"
                assert viewer.dims.ndisplay == 3
                assert not viewer.window._qt_window.isVisible()
                view.seek(0.1)
                assert view.time == 0.1
            assert not viewer.layers
        finally:
            viewer.close()
