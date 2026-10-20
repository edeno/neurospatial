"""Identified events retain their scientific context through viewing and export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment
from neurospatial.animation import (
    Animation,
    AnimationField,
    AnimationView,
    EpochOverlay,
    EventInspector,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    TimeSeriesOverlay,
)


@pytest.fixture
def recording():
    env = Environment.from_samples(
        np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        bin_size=0.5,
    )
    times = np.linspace(10, 11, 11)
    fields = np.linspace(0, 1, len(times) * env.n_bins).reshape(len(times), -1)
    animation = Animation(
        env,
        fields,
        frame_times=times,
        overlays=[
            PositionOverlay(np.column_stack([times - 10, times - 10]), times=times)
        ],
    )
    yield animation
    animation.close()


@pytest.fixture
def records():
    return pd.DataFrame(
        {
            "event_id": ["ripple-17", "ripple-3", "ripple-42"],
            "start_time": [10.1, 10.2, 10.7],
            "stop_time": [10.4, 10.5, 10.9],
            "score": [0.9, 0.8, 0.2],
            "metadata": [{"trial": [1]}, {"trial": [1]}, {"trial": [2]}],
        }
    )


def test_event_tables_preserve_schema_and_own_nested_metadata(recording, records):
    records["trial_type"] = pd.Categorical(
        ["run", "rest", "run"], categories=["rest", "run", "sleep"], ordered=True
    )
    records["trial_number"] = pd.Series([1, None, 3], dtype="Int64")
    records["label"] = pd.Series(["one", None, "three"], dtype="string")
    records.attrs = {"recording": {"units": ["seconds"]}}
    records.index = pd.Index([4, 7, 9], name="source_row")
    expected = records.copy(deep=True).reset_index(drop=True)
    inspector = EventInspector(recording, records)
    pd.testing.assert_frame_equal(inspector.events_table(), expected)

    records.loc[4, "trial_type"] = "rest"
    records.loc[4, "trial_number"] = 99
    records.loc[4, "metadata"]["trial"].append(99)
    records.attrs["recording"]["units"].append("changed")
    result = inspector.events_table()
    assert result.loc[0, "trial_type"] == "run"
    assert result.loc[0, "trial_number"] == 1
    assert result.loc[0, "metadata"] == {"trial": [1]}
    assert result.attrs == {"recording": {"units": ["seconds"]}}

    result.loc[0, "trial_type"] = "rest"
    result.loc[0, "trial_number"] = 88
    result.loc[0, "metadata"]["trial"].append(88)
    result.attrs["recording"]["units"].append("also changed")
    inspector.filter_events([False, True, True])
    visible = inspector.events_table(visible_only=True)
    assert visible.index.tolist() == [0, 1]
    assert visible["trial_type"].dtype == expected["trial_type"].dtype
    assert visible["trial_number"].dtype == expected["trial_number"].dtype
    visible.loc[0, "metadata"]["trial"].append(77)
    visible.attrs["recording"]["units"].append("visible changed")
    owned = inspector.events_table()
    assert owned.loc[0, "trial_type"] == "run"
    assert owned.loc[0, "trial_number"] == 1
    assert owned["metadata"].tolist() == [
        {"trial": [1]},
        {"trial": [1]},
        {"trial": [2]},
    ]
    assert owned.attrs == {"recording": {"units": ["seconds"]}}


@pytest.mark.parametrize("as_records", [False, True])
def test_mixed_large_event_ids_preserve_identity(recording, as_records):
    ids = [9007199254740993, 1.5]
    records = pd.DataFrame(
        {
            "event_id": pd.Series(ids, dtype=object),
            "start_time": [10.1, 10.4],
            "stop_time": [10.3, 10.6],
        }
    )
    inspector = EventInspector(
        recording, records.to_dict("records") if as_records else records
    )
    result = inspector.events_table()
    assert result["event_id"].dtype == object
    assert result.event_id.tolist() == ids
    assert isinstance(result.event_id.iloc[0], int)
    inspector.select(ids[0])
    assert inspector.selected_id == ids[0]
    assert inspector.time_range() == pytest.approx((10.1, 10.3))
    inspector.select(ids[1])
    assert inspector.time_range() == pytest.approx((10.4, 10.6))
    result.at[0, "event_id"] = "changed"
    assert inspector.events_table().event_id.tolist() == ids


def test_mapping_events_with_missing_ids_raise_actionable_error(recording):
    records = [
        {"event_id": "one", "start_time": 10.1, "stop_time": 10.3},
        {"start_time": 10.4, "stop_time": 10.6},
    ]
    with pytest.raises(ValueError, match="nonmissing"):
        EventInspector(recording, records)


@pytest.mark.parametrize(
    "ids",
    [
        [1, 2, 3],
        [9007199254740993, 9007199254740995, 9007199254740997],
        ["a", "b", "c"],
        [1.25, 2.5, 3.75],
        [np.int64(1), np.int32(2), np.uint64(3)],
        [np.str_("a"), np.str_("b"), np.str_("c")],
        [np.float64(1.25), np.float32(2.5), np.float64(3.75)],
    ],
)
def test_json_export_preserves_scalar_event_identity(recording, records, tmp_path, ids):
    records["event_id"] = pd.Series(ids, dtype=object)
    inspector = EventInspector(recording, records)
    path = inspector.export_events(tmp_path / "review.json")
    restored = pd.read_json(path, orient="table")
    reloaded = EventInspector(recording, restored)
    assert reloaded.events_table().event_id.tolist() == ids
    reloaded.select(ids[1])
    assert reloaded.time_range() == pytest.approx((10.2, 10.5))


def test_json_export_accepts_boolean_ids(recording, records, tmp_path):
    records = records.iloc[:2].copy()
    records["event_id"] = pd.Series([np.bool_(True), np.bool_(False)], dtype=object)
    inspector = EventInspector(recording, records)
    path = inspector.export_events(tmp_path / "booleans.json")
    assert pd.read_json(path, orient="table").event_id.tolist() == [True, False]


@pytest.mark.parametrize(
    "suffix, serializer", [(".csv", "to_csv"), (".json", "to_json")]
)
def test_event_serialization_failure_does_not_truncate_previous_review(
    recording, records, tmp_path, monkeypatch, suffix, serializer
):
    inspector = EventInspector(recording, records)
    path = tmp_path / f"review{suffix}"
    path.write_text("previous review", encoding="utf-8")

    def fail_serialization(*args, **kwargs):
        raise ValueError("Cannot serialize review metadata")

    monkeypatch.setattr(pd.DataFrame, serializer, fail_serialization)
    message = "Event IDs cannot round-trip" if suffix == ".json" else "Cannot serialize"
    with pytest.raises(ValueError, match=message):
        inspector.export_events(path, overwrite=True)
    assert path.read_text(encoding="utf-8") == "previous review"


@pytest.mark.parametrize(
    "ids",
    [
        [("a", 1), ("a", 2), ("b", 1)],
        [9007199254740993, 1.5, 2.5],
        [1, 1.5, 2.5],
        [np.inf, 1.5, 2.5],
        [-np.inf, 1.5, 2.5],
        [1 / 3, 1.5, 2.5],
        [np.longdouble("1.25"), np.longdouble("2.5"), np.longdouble("3.75")],
    ],
)
def test_json_rejects_lossy_ids_before_opening_file(recording, records, tmp_path, ids):
    records["event_id"] = pd.Series(ids, dtype=object)
    inspector = EventInspector(recording, records)
    path = tmp_path / "review.json"
    with pytest.raises(ValueError, match="Event IDs cannot round-trip"):
        inspector.export_events(path)
    assert not path.exists()
    path.write_text("previous review", encoding="utf-8")
    with pytest.raises(ValueError, match="Event IDs cannot round-trip"):
        inspector.export_events(path, overwrite=True)
    assert path.read_text(encoding="utf-8") == "previous review"


def test_empty_event_table_preserves_schema_and_exports_json(
    recording, records, tmp_path
):
    records["trial_type"] = pd.Categorical(
        ["run", "rest", "run"], categories=["rest", "run", "sleep"], ordered=True
    )
    records["trial_number"] = pd.Series([1, None, 3], dtype="Int64")
    records = records.iloc[:0]
    records.attrs = {"recording": {"units": ["seconds"]}}
    inspector = EventInspector(recording, records)
    pd.testing.assert_frame_equal(inspector.events_table(), records)
    path = inspector.export_events(tmp_path / "empty.json")
    restored = pd.read_json(path, orient="table")
    pd.testing.assert_frame_equal(restored, records)


def test_filter_and_annotation_preserve_overlap_identity(recording, records, tmp_path):
    inspector = recording.inspect_events(records, padding=0.05)
    assert len(inspector.events_table()) == 3  # overlapping rows stay distinct
    records.at[0, "metadata"]["trial"].append(99)
    assert inspector.events_table().at[0, "metadata"] == {"trial": [1]}
    inspector.annotate("ripple-3", review="accepted", tags=["forward"])
    inspector.select("ripple-3")
    inspector.filter_events(inspector.events_table()["score"] >= 0.8)
    assert inspector.selected_id == "ripple-3"
    assert inspector.time_range() == pytest.approx((10.15, 10.55))
    exported = inspector.events_table()
    exported.at[1, "annotation_tags"].append("changed")
    assert inspector.events_table().at[1, "annotation_tags"] == ["forward"]

    path = inspector.export_events(tmp_path / "review.json", selected_only=True)
    restored = pd.read_json(path, orient="table")
    assert restored["event_id"].tolist() == ["ripple-3"]
    assert restored["annotation_review"].tolist() == ["accepted"]
    assert restored["score"].tolist() == [0.8]
    assert restored["metadata"].iloc[0] == {"trial": [1]}
    with pytest.raises(FileExistsError):
        inspector.export_events(path)

    inspector.filter_events(np.array([False, False, True]))
    assert inspector.selected_id == "ripple-42"
    inspector.reset_filter().select("ripple-3")
    assert inspector.events_table().iloc[1]["annotation_review"] == "accepted"


@pytest.mark.parametrize(
    "changed, message",
    [
        ({"event_id": ["a", "a", "c"]}, "unique"),
        ({"event_id": ["a", None, "c"]}, "nonmissing"),
        ({"start_time": [10.4, 10.2, 10.7]}, "start_time < stop_time"),
        ({"stop_time": [np.inf, 10.5, 10.9]}, "finite"),
        ({"start_time": [0, 10.2, 10.7], "stop_time": [1, 10.5, 10.9]}, "overlap"),
    ],
)
def test_bad_events_fail_before_view(recording, records, changed, message):
    with pytest.raises(ValueError, match=message):
        EventInspector(recording, records.assign(**changed))


def test_empty_filter_never_exports_stale_selection(recording, records, tmp_path):
    inspector = EventInspector(recording, records)
    inspector.filter_events(np.zeros(3, dtype=bool))
    assert inspector.selected_id is None
    with pytest.raises(ValueError, match="No event"):
        inspector.export(tmp_path / "empty.gif")
    assert not (tmp_path / "empty.gif").exists()
    with pytest.raises(ValueError, match="Boolean"):
        inspector.filter_events([1, 0, 1])


def test_common_view_uses_recording_seconds_and_closes_once(recording, monkeypatch):
    moved, closed, exports = [], [], []
    native = object()
    view = AnimationView(
        recording, native, seek=moved.append, close=lambda: closed.append(1)
    )
    assert view.native is native
    view.seek(10.25)
    view.select_time(10.2, 10.6)
    monkeypatch.setattr(
        recording, "export", lambda path, **kw: exports.append((path, kw))
    )
    view.export("ripple.mp4", fps=20)
    assert moved == [10.25]
    assert exports == [("ripple.mp4", {"fps": 20, "time_range": (10.2, 10.6)})]
    with pytest.raises(ValueError, match="within"):
        view.seek(0.2)
    with pytest.raises(ValueError, match="nonempty"):
        view.select_time(10.5, 10.5)
    view.close()
    view.close()
    assert closed == [1]
    with pytest.raises(RuntimeError, match="closed"):
        view.seek(10.4)


def test_single_frame_view_exports_without_an_empty_explicit_range(recording, tmp_path):
    scene = Animation(
        recording.env,
        np.ones((1, recording.env.n_bins)),
        frame_times=[10.0],
    )
    view = AnimationView(scene, object(), seek=lambda time: None, close=lambda: None)
    try:
        assert view.export(tmp_path / "still.gif", dpi=25).exists()
    finally:
        view.close()
        scene.close()


def test_widget_selection_and_selected_event_export(recording, records, tmp_path):
    pytest.importorskip("ipywidgets")
    from PIL import Image

    inspector = EventInspector(recording, records)
    container = inspector.show(dpi=35, figsize=(4, 3))
    assert inspector.view is not None
    assert inspector.view.native in container.children
    inspector.select("ripple-3")
    assert inspector.view.time == pytest.approx(10.2)
    assert inspector.view.time_range == pytest.approx((10.2, 10.5))
    # The widget's exact canvas is the shared renderer's output.
    import io

    frame = np.asarray(Image.open(io.BytesIO(inspector.view.native.children[0].value)))
    np.testing.assert_array_equal(
        frame, recording.render_frame(10.2, dpi=35, figsize=(4, 3))
    )
    path = inspector.export(tmp_path / "event.gif", fps=10, dpi=35, figsize=(4, 3))
    with Image.open(path) as movie:
        assert movie.n_frames == 3
        movie.seek(0)
        # GIF palette quantization changes colors, but not panel placement.
        error = np.mean(np.abs(np.asarray(movie.convert("RGB"), dtype=float) - frame))
        assert error < 5
    inspector.close()
    # Closing controls leaves the parent scene usable.
    assert recording.render_frame(10.4, dpi=35, figsize=(4, 3)).shape[-1] == 3


def test_complete_scene_exports_multiple_fields_and_observations(recording, tmp_path):
    env = recording.env
    times = recording.frame_times
    positions = np.column_stack([times - 10, times - 10])
    scene = Animation(
        env,
        {
            "posterior": AnimationField(
                np.ones((len(times), env.n_bins)) / env.n_bins,
                times,
                label="Probability",
            ),
            "likelihood": AnimationField(
                np.arange(len(times) * env.n_bins).reshape(len(times), -1),
                times,
                cmap="magma",
                label="Likelihood",
            ),
        },
        overlays=[
            PositionOverlay(positions, times=times),
            ObjectVectorOverlay(
                animal_positions=positions,
                object_positions=np.array([[1.0, 0.0]]),
                times=times,
            ),
            TimeSeriesOverlay(times=times, data=np.sin(times), label="LFP"),
            SpikeRasterOverlay([np.array([10.2, 10.7])], unit_ids=["cell-A"]),
            EpochOverlay(np.array([[10.1, 10.4]]), label="Ripple"),
        ],
    )
    with scene:
        image = scene.render_frame(10.2, dpi=35)
        axes = scene._renderer.figure.axes
        assert {"posterior", "likelihood"} <= {ax.get_title() for ax in axes}
        assert any(
            [text.get_text() for text in ax.get_yticklabels()] == ["cell-A"]
            for ax in axes
        )
        assert any("LFP" in ax.get_legend_handles_labels()[1] for ax in axes)
        assert image.ndim == 3
        path = scene.export(
            tmp_path / "complete.gif", time_range=(10.2, 10.4), fps=10, dpi=35
        )
        assert path.stat().st_size > 100


def test_native_view_time_follows_the_users_slider_and_borrows_scene_readers(
    recording, monkeypatch
):
    from neurospatial.animation.backends import fastplotlib_backend
    from neurospatial.animation.overlays import OverlayData, VideoData

    video = VideoData(
        frame_indices=np.array([0]),
        reader=Mock(),
        transform_to_env=None,
        env_bounds=(0, 1, 0, 1),
        alpha=1.0,
        z_order="below",
        _owns_reader=True,
    )
    data = OverlayData(videos=[video], frame_times=recording.frame_times)
    monkeypatch.setattr(recording, "_overlay_data", data)

    class Native(SimpleNamespace):
        @property
        def time(self):
            return recording.frame_times[0] + self.widget.indices["time"]

    native = Native(
        widget=SimpleNamespace(indices={"time": 0.0}),
        show=Mock(),
        seek=Mock(),
        close=Mock(),
        capture_session=Mock(),
    )
    render = Mock(return_value=native)
    monkeypatch.setattr(fastplotlib_backend, "render_fastplotlib", render)
    view = recording.show(backend="fastplotlib")
    try:
        native.widget.indices["time"] = 0.35
        assert view.time == pytest.approx(10.35)
        view.seek(10.7)
        native.seek.assert_called_once_with(10.7)
        # The native clock is authoritative even before a requested frame is drawn.
        assert view.time == pytest.approx(10.35)
        arguments = render.call_args.kwargs
        assert not arguments["overlay_data"].videos[0]._owns_reader
        assert video._owns_reader  # The scene keeps ownership for later exports.
        callback = arguments["_field_at_time"]
        np.testing.assert_allclose(
            callback(10.35)[0], next(iter(recording.sample_fields(10.35).values()))
        )
    finally:
        view.close()
        view.close()
    native.close.assert_called_once()
    assert view.time == pytest.approx(10.35)


def test_napari_scene_view_uses_physical_clock_adapter(recording, monkeypatch):
    from neurospatial.animation.backends import _napari_scene

    handle = SimpleNamespace(
        viewer=object(), time=10.2, seek=Mock(), close=Mock(), capture_session=Mock()
    )
    build = Mock(return_value=handle)
    monkeypatch.setattr(_napari_scene, "build_napari_scene", build)
    view = recording.show(backend="napari", show=False)
    try:
        build.assert_called_once_with(recording, show=False)
        assert view.native is handle.viewer
        assert view.time == 10.2
        view.seek(10.35)
        handle.seek.assert_called_once_with(10.35)
    finally:
        view.close()
    handle.close.assert_called_once()


def test_widget_playback_advances_physical_seconds_on_an_irregular_clock(recording):
    pytest.importorskip("ipywidgets")
    times = np.array([10.0, 10.1, 11.0])
    with Animation(
        recording.env, np.zeros((3, recording.env.n_bins)), frame_times=times
    ) as scene:
        view = scene.show(speed=2.0, max_playback_fps=10, dpi=20)
        play, slider = view.native.children[1].children
        try:
            assert play.interval == 100  # One 100 ms wall-clock tick.
            play.value = 1
            assert view.time == pytest.approx(10.2)  # At 2x speed, advance 200 ms.
            play.value = play.max
            assert view.time == pytest.approx(11.0)
            slider.value = 10.35
            assert view.time == pytest.approx(10.35)
            view.native.close()
            with pytest.raises(RuntimeError, match="closed"):
                view.seek(10.5)
        finally:
            view.close()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"speed": 0}, "speed"),
        ({"speed": np.nan}, "speed"),
        ({"max_playback_fps": 0}, "max_playback_fps"),
        ({"max_playback_fps": 1.5}, "max_playback_fps"),
    ],
)
def test_notebook_playback_rejects_invalid_physical_timing(recording, options, message):
    pytest.importorskip("ipywidgets")
    with pytest.raises(ValueError, match=message):
        recording.show(**options)


def test_tuple_event_ids_remain_atomic_during_selection_and_export(
    recording, records, tmp_path
):
    ids = [("session-A", 17), ("session-A", 3), ("session-B", 42)]
    records["event_id"] = ids
    inspector = EventInspector(recording, records)
    inspector.select(ids[1])
    assert inspector.selected_id == ids[1]
    assert inspector.time_range() == pytest.approx((10.2, 10.5))
    inspector.filter_events(np.array([False, True, False]))
    path = inspector.export_events(tmp_path / "tuple-ids.csv", selected_only=True)
    assert len(pd.read_csv(path)) == 1
    assert inspector.events_table().event_id.tolist() == ids


def test_event_controls_preserve_filter_annotation_and_close_their_view(
    recording, records
):
    pytest.importorskip("ipywidgets")
    inspector = EventInspector(recording, records)
    inspector.select("ripple-3").filter_events(np.array([True, True, False]))
    container = inspector.show(dpi=20)
    selector = container.children[0].children[0]
    decision, note, _ = container.children[0].children[1].children
    first_view = inspector.view
    try:
        assert selector.value == "ripple-3"
        assert [value for _, value in selector.options] == ["ripple-17", "ripple-3"]
        decision.value = "accepted"
        note.value = "forward replay"
        selector.value = "ripple-17"
        selector.value = "ripple-3"
        assert decision.value == "accepted"
        assert note.value == "forward replay"
        rows = inspector.events_table()
        assert rows.loc[1, "annotation_review"] == "accepted"
        assert rows.loc[1, "annotation_note"] == "forward replay"
        container.close()
        assert inspector.view is None
        assert not inspector._observers
        with pytest.raises(RuntimeError, match="closed"):
            first_view.seek(10.4)
        # Reopening retains the filter and selection. Closing an old container
        # must not close the new view or retain observers that can edit it.
        second = inspector.show(dpi=20)
        second_view = inspector.view
        container.close()
        assert inspector.view is second_view
        assert inspector.selected_id == "ripple-3"
        assert inspector.events_table(visible_only=True).event_id.tolist() == [
            "ripple-17",
            "ripple-3",
        ]
        second.close()
    finally:
        inspector.close()


def test_preexisting_numeric_annotation_column_accepts_review_metadata(
    recording, records
):
    records["annotation_review"] = [0, 1, 2]
    inspector = EventInspector(recording, records)
    inspector.annotate("ripple-3", review={"decision": "accepted"})
    assert inspector.events_table().loc[1, "annotation_review"] == {
        "decision": "accepted"
    }


def test_notebook_final_partial_tick_stops_without_rewinding_play_control(recording):
    pytest.importorskip("ipywidgets")
    view = recording.show(dpi=20, max_playback_fps=30)
    play = view.native.children[1].children[0]
    try:
        # One second does not divide evenly into the rounded 33 ms UI ticks.
        assert play.max == 31
        play.playing = True
        play.value = play.max
        assert play.value == play.max
        assert not play.playing
        assert view.time == pytest.approx(11.0)
    finally:
        view.close()
