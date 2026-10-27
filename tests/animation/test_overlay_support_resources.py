"""Scientific event support and transactional decoder ownership regressions."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial.animation._sampling import sample_at_times
from neurospatial.animation.overlays import (
    EventOverlay,
    PositionOverlay,
    VideoData,
    VideoOverlay,
    _convert_overlays_to_data,
    _realign_prepared_overlays,
)


@pytest.fixture
def env():
    return SimpleNamespace(
        n_dims=2,
        dimension_ranges=((0.0, 10.0), (0.0, 10.0)),
        coordinate_system=SimpleNamespace(kind="cartesian"),
    )


@pytest.mark.parametrize("interp", ["linear", "nearest"])
@pytest.mark.parametrize("gap", ["duration", "break", "missing"])
def test_event_locations_respect_tracking_support(env, interp, gap):
    times = np.array([0.0, 0.2, 1.0, 1.2])
    positions = np.column_stack((times, times))
    breaks = None
    max_gap = None if gap == "break" else 0.5
    if gap == "break":
        breaks = np.array([False, False, True, False])
    elif gap == "missing":
        positions[2] = np.nan
    events = np.array([0.0, 0.1, 0.2, 0.6, 1.0, 1.2])
    overlay = EventOverlay.from_trajectory(
        events,
        positions,
        times,
        interp=interp,
        max_gap=max_gap,
        break_before=breaks,
    )
    data = overlay.convert_to_data(events, len(events), env)
    assert_array_equal(data.event_times["event"], events)
    assert np.isnan(data.event_positions["event"][3]).all()
    assert_allclose(data.event_positions["event"][[0, 2, 5]], positions[[0, 1, 3]])
    if gap == "missing":
        assert np.isnan(data.event_positions["event"][4]).all()
    else:
        assert_allclose(data.event_positions["event"][4], positions[2])
    if interp == "nearest":
        assert_allclose(data.event_positions["event"][1], positions[0])


def test_events_outside_tracking_keep_identity_and_missing_location(env):
    events = np.array([0.0, 0.5, 1.0])
    overlay = EventOverlay.from_trajectory(
        events,
        np.array([[1.0, 2.0]]),
        np.array([0.5]),
    )
    data = overlay.convert_to_data(events, 3, env)
    assert_array_equal(data.event_times["event"], events)
    assert np.isnan(data.event_positions["event"][[0, 2]]).all()
    assert_allclose(data.event_positions["event"][1], [1.0, 2.0])


def test_fixed_events_are_not_a_sampled_trajectory(env):
    events = np.array([0.0, 3.0, 8.0])
    data = EventOverlay.from_fixed_positions(
        events, np.array([[2.0, 4.0]])
    ).convert_to_data(events, 3, env)
    assert_allclose(data.event_positions["event"], [[2.0, 4.0]] * 3)


@pytest.mark.parametrize("explicit", [False, True])
def test_event_coordinate_dimensions_checked_at_environment(env, explicit):
    kwargs = (
        {"event_positions": np.ones((2, 3))}
        if explicit
        else {"positions": np.ones((2, 3)), "position_times": np.array([0.0, 1.0])}
    )
    overlay = EventOverlay(np.array([0.0, 1.0]), **kwargs)
    with pytest.raises(ValueError, match="dimension"):
        overlay.convert_to_data(np.array([0.0, 1.0]), 2, env)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_times": np.array([0.0, 0.0])},
        {"position_times": np.array([0.0])},
        {"position_times": np.array([0.0, np.inf])},
        {"break_before": np.array([True])},
        {"max_gap": -1.0},
        {"positions": np.array([[0.0, np.inf], [1.0, 1.0]])},
    ],
)
def test_event_trajectory_validates_observation_contract(env, kwargs):
    params = {"positions": np.zeros((2, 2)), "position_times": np.array([0.0, 1.0])}
    params.update(kwargs)
    overlay = EventOverlay(np.array([0.0, 1.0]), **params)
    with pytest.raises(ValueError):
        overlay.convert_to_data(np.array([0.0, 1.0]), 2, env)


@pytest.fixture
def readers(monkeypatch):
    opened = []

    class Reader:
        n_frames = 3
        original_size_px = (4, 4)
        frame_size_px = (4, 4)

        def __init__(self, *args, **kwargs):
            self.close_count = 0
            opened.append(self)

        def close(self):
            self.close_count += 1

    monkeypatch.setattr("neurospatial._video_io._VideoReader", Reader)
    return opened


@pytest.mark.parametrize(
    "failure", ["times", "length", "calibration", "crop", "frame_count"]
)
def test_video_conversion_closes_acquired_reader_on_validation_failure(
    env, readers, failure
):
    overlay = VideoOverlay("test.mp4", times=np.arange(3.0))
    if failure == "times":
        overlay.times = np.array([0.0, 0.0, 1.0])
    elif failure == "length":
        overlay.times = np.array([0.0, 1.0])
    elif failure == "calibration":
        overlay.calibration = object()
    elif failure == "crop":
        overlay.crop = (0, 0, 10, 10)
    else:
        overlay.times = None
    frames = np.arange(2.0) if failure == "frame_count" else np.arange(3.0)
    with (
        pytest.raises((ValueError, AttributeError)),
        pytest.warns(UserWarning) if overlay.calibration is None else nullcontext(),
    ):
        overlay.convert_to_data(frames, len(frames), env)
    assert len(readers) == 1
    assert readers[0].close_count == 1


def test_pipeline_closes_all_owned_readers_on_later_failure(env, readers):
    overlays = [
        VideoOverlay("first.mp4", times=np.arange(3.0)),
        VideoOverlay("second.mp4", times=np.arange(3.0)),
        object(),
    ]
    with pytest.raises(TypeError), pytest.warns(UserWarning):
        _convert_overlays_to_data(overlays, np.arange(3.0), 3, env)
    assert len(readers) == 2
    assert [r.close_count for r in readers] == [1, 1]


def test_conversion_opens_one_reader_and_realign_borrows_it(env, readers):
    with pytest.warns(UserWarning):
        prepared, owned = _convert_overlays_to_data(
            [VideoOverlay("test.mp4", times=np.arange(3.0))],
            np.array([0.0, 2.0]),
            2,
            env,
        )
    for clock in (np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5])):
        result = _realign_prepared_overlays(prepared, clock)
        assert result.videos[0].reader is prepared.videos[0].reader
        result.videos[0].frame_indices[0] = -1
        assert prepared.videos[0].frame_indices[0] == 0
    assert len(readers) == 1
    assert readers[0].close_count == 0
    owned.close()
    assert readers[0].close_count == 1


def test_single_conversion_failure_closes_previously_opened_reader(env, readers):
    class Fails:
        calls = 0

        def convert_to_data(self, frame_times, n_frames, env):
            self.calls += 1
            raise ValueError("adapter conversion failed")

    failing = Fails()
    with pytest.raises(ValueError, match="adapter"), pytest.warns(UserWarning):
        _convert_overlays_to_data(
            [VideoOverlay("test.mp4", times=np.arange(3.0)), failing],
            np.arange(3.0),
            3,
            env,
        )
    assert failing.calls == 1
    assert len(readers) == 1
    assert readers[0].close_count == 1


def test_borrowed_reader_is_not_closed_on_pipeline_failure(env):
    reader = SimpleNamespace(close_count=0)

    def close():
        reader.close_count += 1

    reader.close = close

    class BorrowedVideo:
        def convert_to_data(self, frame_times, n_frames, env):
            return VideoData(
                np.arange(n_frames),
                reader,
                None,
                (0, 1, 0, 1),
                1.0,
                "above",
                times=frame_times,
            )

    with pytest.raises(TypeError):
        _convert_overlays_to_data([BorrowedVideo(), object()], np.arange(3.0), 3, env)
    assert reader.close_count == 0


def test_prepared_realign_preserves_break_when_clock_skips_gap(env):
    overlay = PositionOverlay(
        np.ones((4, 2)),
        np.array([0.0, 0.2, 1.0, 1.2]),
    )
    prepared, _ = _convert_overlays_to_data([overlay], np.array([0.0, 1.2]), 2, env)
    result = _realign_prepared_overlays(prepared, np.array([0.0, 1.2]))
    assert_array_equal(result.positions[0].sample_valid, [True, True])
    assert_array_equal(result.positions[0].interval_valid, [False])


@pytest.mark.parametrize("vector", [False, True])
def test_prepared_heading_realign_matches_original_sampling(env, vector):
    from neurospatial.animation.overlays import HeadDirectionOverlay

    headings = np.array([3.0, -3.0])
    if vector:
        headings = np.column_stack((np.cos(headings), np.sin(headings)))
    overlay = HeadDirectionOverlay(headings, times=np.array([0.0, 0.2]))
    frames = np.array([0.0, 0.1, 0.2])
    prepared, _ = _convert_overlays_to_data([overlay], frames, len(frames), env)
    result = _realign_prepared_overlays(prepared, frames)
    expected = sample_at_times(
        overlay.times,
        headings,
        frames,
        interp="linear",
        sample_valid=np.ones(2, dtype=bool),
        interval_valid=np.ones(1, dtype=bool),
        circular=not vector,
    )
    assert_allclose(result.head_directions[0].data, expected)
    if not vector:
        # The short way across +/-pi.
        assert abs(result.head_directions[0].data[1]) > 3.0
