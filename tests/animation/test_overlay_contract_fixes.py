"""Raster identity, canonical calibration, and custom adapter clock contracts."""

import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neurospatial import Environment
from neurospatial.animation.calibration import calibrate_video
from neurospatial.animation.overlays import (
    SpikeRasterOverlay,
    VideoData,
    VideoOverlay,
    _convert_overlays_to_data,
    _realign_prepared_overlays,
)
from neurospatial.encoding.spatial import SpatialRatesResult
from neurospatial.ops.transforms import Affine2D, VideoCalibration


@pytest.fixture
def env():
    return Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool), (np.arange(3.0), np.arange(3.0))
    )


@pytest.mark.parametrize("order", [None, [1, 0]])
def test_raster_mixed_identity_selects_correct_rate_rows(env, order):
    rates = SpatialRatesResult(
        np.vstack([np.full(env.n_bins, value) for value in [10.0, 99.0, 20.0]]),
        np.ones(env.n_bins),
        env,
        "binned",
        2.0,
        unit_ids=[1, "1", "b"],
    )
    raster = SpikeRasterOverlay(
        [np.array([0.0]), np.array([0.1])],
        unit_ids=[1, "b"],
        rates=rates,
        order=order,
    )
    data = raster.convert_to_data(np.array([0.0, 0.1]), 2, env)
    expected_ids = [1, "b"] if order is None else ["b", 1]
    assert data.unit_ids.tolist() == expected_ids
    assert_array_equal(
        data.rates[:, 0], [10.0, 20.0] if order is None else [20.0, 10.0]
    )


@pytest.mark.parametrize(
    "units, scale",
    [(None, 1.0), ("cm", 1.0), ("m", 0.01), ("millimeters", 10.0), ("um", 10000.0)],
)
def test_canonical_video_calibration_converts_to_environment_units(units, scale):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        (np.arange(3.0), np.arange(3.0)),
        units=units,
    )
    calibration = VideoCalibration(
        Affine2D(np.array([[10.0, 0, 20.0], [0, 10.0, 30.0], [0, 0, 1.0]])), (2, 2)
    )
    data = VideoOverlay(
        np.zeros((2, 2, 2, 3), dtype=np.uint8),
        calibration,
    ).convert_to_data(np.array([0.0, 0.1]), 2, env)
    assert_allclose(
        data.transform_to_env(np.array([[1.0, 1.0]])), [[30 * scale, 40 * scale]]
    )
    assert_allclose(calibration.transform_px_to_cm(np.array([[1.0, 1.0]])), [[30, 40]])


@pytest.mark.parametrize("units", ["pixels", "px", "custom"])
def test_calibrated_video_refuses_units_without_cm_conversion(env, units):
    env = env.with_metadata(units=units)
    source = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    calibration = VideoCalibration(Affine2D(np.eye(3)), (2, 2))
    with pytest.raises(ValueError, match="conversion from centimeters"):
        VideoOverlay(source, calibration).convert_to_data(np.array([0.0, 0.1]), 2, env)
    with pytest.warns(UserWarning, match="no calibration"):
        VideoOverlay(source).convert_to_data(np.array([0.0, 0.1]), 2, env)


@pytest.mark.parametrize(
    "units, scale", [(None, 1.0), ("cm", 1.0), ("m", 0.01), ("mm", 10.0)]
)
@pytest.mark.parametrize("method", ["landmarks", "scale_bar", "cm_per_px"])
def test_calibrate_video_wrapper_keeps_canonical_cm_and_fits_env_units(
    units, scale, method, tmp_path, monkeypatch
):
    env = Environment.from_grid_mask(
        np.ones((2, 2), dtype=bool),
        (np.array([0.0, 50.0, 100.0]) * scale, np.array([0.0, 40.0, 80.0]) * scale),
        units=units,
    )
    path = tmp_path / "video.mp4"
    path.touch()
    monkeypatch.setattr(
        "neurospatial.animation.calibration._get_video_frame_size", lambda _: (10, 8)
    )
    landmarks_px = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0]])
    landmarks_cm = np.column_stack(
        (landmarks_px[:, 0] * 10, (8 - landmarks_px[:, 1]) * 10)
    )
    kwargs = {"cm_per_px": 10.0}
    if method == "scale_bar":
        kwargs = {"scale_bar": ((0.0, 0.0), (10.0, 0.0), 100.0)}
    elif method == "landmarks":
        kwargs = {
            "landmarks_px": landmarks_px,
            "landmarks_env": landmarks_cm * scale,
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibration = calibrate_video(env, path, **kwargs)
    assert not [w for w in caught if "coverage" in str(w.message)]
    assert_allclose(calibration.cm_per_px, 10.0)
    assert_allclose(
        calibration.transform_px_to_cm(landmarks_px), landmarks_cm, atol=1e-12
    )
    data = VideoOverlay(
        np.zeros((1, 8, 10, 3), dtype=np.uint8),
        calibration,
    ).convert_to_data(np.array([0.0]), 1, env)
    assert_allclose(
        data.transform_to_env(landmarks_px), landmarks_cm * scale, atol=1e-12
    )


@dataclass(frozen=True)
class ReorderedVideo:
    times: np.ndarray
    reader: np.ndarray
    calls: list
    interp: str = "nearest"

    def convert_to_data(self, frame_times, n_frames, env):
        self.calls.append(frame_times.copy())
        return VideoData(
            np.array([2, 0]), self.reader, None, (0, 2, 0, 2), 1.0, "above"
        )


def test_native_conversion_preserves_custom_mapping_and_invokes_adapter_once(env):
    times = np.array([0.0, 0.2])
    reader = np.stack(
        [np.full((2, 2, 3), color, dtype=np.uint8) for color in [10, 20, 30]]
    )
    adapter = ReorderedVideo(times, reader, [])
    prepared = _convert_overlays_to_data(
        [adapter],
        np.array([0.0, 0.1, 0.2]),
        3,
        env,
        own_samples=True,
    )
    assert len(adapter.calls) == 1
    assert_array_equal(adapter.calls[0], times)
    assert_array_equal(prepared.videos[0].frame_indices, [2, 0])
    assert prepared.videos[0].reader is reader
    assert prepared.videos[0].get_frame(0)[0, 0, 0] == 30
    times[:] = 99
    assert_array_equal(prepared.videos[0].times, [0.0, 0.2])
    aligned = _realign_prepared_overlays(prepared, np.array([0.0, 0.1, 0.2]))
    assert_array_equal(aligned.videos[0].frame_indices, [2, 2, 0])


@pytest.mark.parametrize(
    "indices",
    [np.array([0.0, 1.0]), np.array([0, -2]), np.array([0, 3]), np.array([0])],
)
def test_conversion_validates_custom_video_mapping(env, indices):
    class Adapter:
        times = np.array([0.0, 0.1])
        interp = "nearest"

        def convert_to_data(self, frame_times, n_frames, env):
            return VideoData(
                indices,
                np.zeros((3, 2, 2, 3), dtype=np.uint8),
                None,
                (0, 2, 0, 2),
                1.0,
                "above",
            )

    with pytest.raises(ValueError, match="frame_indices"):
        _convert_overlays_to_data(
            [Adapter()], np.array([0.0, 0.1]), 2, env, own_samples=True
        )


def test_adapter_output_clock_and_missing_video_rows_are_authoritative(env):
    output_times = np.array([0.0, 0.05, 0.1])
    indices = np.array([2, -1, 0])
    reader = np.stack(
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in [10, 20, 30]]
    )

    class Adapter:
        times = np.array([0.0, 0.1])
        interp = "nearest"
        calls = 0

        def convert_to_data(self, frame_times, n_frames, env):
            self.calls += 1
            return VideoData(
                indices, reader, None, (0, 2, 0, 2), 1.0, "above", times=output_times
            )

    adapter = Adapter()
    prepared = _convert_overlays_to_data(
        [adapter], output_times, 3, env, own_samples=True
    )
    item = prepared.videos[0]
    assert adapter.calls == 1
    assert_array_equal(item.times, output_times)
    assert_array_equal(item.frame_indices, [2, -1, 0])
    assert item.get_frame(1) is None
    assert item.get_frame(0)[0, 0, 0] == 30
    output_times[:] = 100
    indices[:] = -1
    assert_array_equal(item.times, [0.0, 0.05, 0.1])
    assert_array_equal(item.frame_indices, [2, -1, 0])


def test_raster_does_not_merge_integer_and_string_labels(env):
    raster = SpikeRasterOverlay([np.array([0.0]), np.array([0.1])], unit_ids=[1, "1"])
    data = raster.convert_to_data(np.array([0.0, 0.1]), 2, env)
    assert data.unit_ids.tolist() == [1, "1"]
    with pytest.raises(ValueError, match=r"Unit 'b'.*NaN or infinite"):
        SpikeRasterOverlay([np.array([0.0]), np.array([np.inf])], unit_ids=[1, "b"])


def test_returned_video_clock_without_adapter_times_validates_scene_overlap(env):
    class Reader:
        n_frames = 3
        close_count = 0

        def close(self):
            self.close_count += 1

    reader = Reader()

    class Adapter:
        times = None
        interp = "nearest"

        def convert_to_data(self, frame_times, n_frames, env):
            return VideoData(
                np.array([2, 0]),
                reader,
                None,
                (0, 2, 0, 2),
                1.0,
                "above",
                times=np.array([10.0, 10.1]),
                _owns_reader=True,
            )

    with pytest.raises(ValueError, match="No temporal overlap"):
        _convert_overlays_to_data(
            [Adapter()], np.array([0.0, 0.1]), 2, env, own_samples=True
        )
    assert reader.close_count == 1


def test_returned_position_clock_warns_on_partial_scene_overlap(env):
    from neurospatial.animation.overlays import PositionData

    class Adapter:
        times = None
        interp = "linear"

        def convert_to_data(self, frame_times, n_frames, env):
            return PositionData(
                np.ones((2, 2)), "red", 5, None, times=np.array([0.0, 0.1])
            )

    with pytest.warns(UserWarning, match="Partial temporal overlap"):
        data = _convert_overlays_to_data(
            [Adapter()], np.array([0.0, 0.2, 1.0]), 3, env, own_samples=True
        )
    assert_array_equal(data.positions[0].times, [0.0, 0.1])
