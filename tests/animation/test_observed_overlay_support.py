"""Tracking observations keep the same scientific support through animation."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._sampling import sample_at_times
from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    ObjectVectorOverlay,
    PositionOverlay,
    SpikeRasterOverlay,
    VideoOverlay,
    _convert_overlays_to_data,
    _validate_frame_times,
)
from neurospatial.decoding._result import _align_true_positions
from neurospatial.encoding.spatial import SpatialRatesResult
from neurospatial.ops.transforms import Affine2D, VideoCalibration


@pytest.fixture
def env():
    return Environment.from_samples(
        np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]]),
        bin_size=2.0,
        units="cm",
        frame="arena",
    )


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_missing_samples_gaps_breaks_and_exact_observations(env, interp):
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 2.0, 2.2])
    positions = np.column_stack([times, times])
    positions[2] = np.nan
    breaks = np.array([False, False, False, False, True, False, False])
    queries = np.unique(np.r_[times, 0.1, 0.3, 0.5, 0.7, 1.0, 2.1])
    overlay = PositionOverlay(positions, times, interp=interp, break_before=breaks)
    data = overlay.convert_to_data(queries, len(queries), env)
    decoded, _ = _align_true_positions(
        queries,
        times,
        positions,
        n_dims=2,
        max_gap=0.5,
        break_before=breaks,
        call="test",
    )
    np.testing.assert_array_equal(np.isfinite(data.data), np.isfinite(decoded))
    if interp == "linear":
        np.testing.assert_allclose(data.data, decoded, equal_nan=True)
    np.testing.assert_array_equal(
        data.data[np.isin(queries, times[[1, 3, 4, 5]])], positions[[1, 3, 4, 5]]
    )


def test_native_sampling_retains_support_and_method(env):
    times = np.array([0.0, 0.2, 2.0, 2.2])
    positions = np.column_stack([times, times])
    overlay = PositionOverlay(positions, times, interp="nearest")
    data = _convert_overlays_to_data(
        [overlay], np.array([0.0, 2.2]), 2, env, own_samples=True
    ).positions[0]
    assert data.interp == "nearest"
    np.testing.assert_array_equal(data.interval_valid, [True, False, True])
    np.testing.assert_array_equal(data.sample_valid, np.ones(4, dtype=bool))
    actual = sample_at_times(
        data.times,
        data.data,
        np.array([0.1, 1.0, 2.0, 2.3]),
        interp=data.interp,
        sample_valid=data.sample_valid,
        interval_valid=data.interval_valid,
    )
    np.testing.assert_allclose(
        actual,
        [[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0], [np.nan, np.nan]],
        equal_nan=True,
    )


def test_pose_and_heading_missingness_are_independent(env):
    times = np.array([0.0, 0.2, 0.4])
    nose = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0]])
    body = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    query = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    data = BodypartOverlay({"nose": nose, "body": body}, times).convert_to_data(
        query, 5, env
    )
    assert np.isnan(data.bodyparts["nose"][1:4]).all()
    assert np.isfinite(data.bodyparts["body"]).all()
    headings = HeadDirectionOverlay(
        np.array([3.0, np.nan, -3.0]), times
    ).convert_to_data(query, 5, env)
    assert np.isnan(headings.data[1:4]).all()
    exact = HeadDirectionOverlay(np.array([3 * np.pi, -3.0]), np.array([0.0, 0.2]))
    aligned = exact.convert_to_data(np.array([0.0, 0.1, 0.2]), 3, env).data
    assert aligned[0] == 3 * np.pi
    assert abs(aligned[1]) > 3.0


def test_object_vectors_follow_missing_position_and_declared_break(env):
    overlay = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]),
        np.array([[1.0, 1.0], [2.0, 2.0], [np.nan, np.nan]]),
        np.array([0.0, 0.2, 0.4]),
        break_before=np.array([False, True, False]),
    )
    positions = overlay.convert_to_data(
        np.array([0.0, 0.1, 0.2, 0.3, 0.4]), 5, env
    ).animal_positions
    np.testing.assert_array_equal(
        np.isfinite(positions).all(axis=1), [True, False, True, False, False]
    )


@pytest.mark.parametrize(
    "bad", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 1)), np.array(1.0)]
)
def test_frame_clock_is_finite_and_one_dimensional(bad):
    with pytest.raises(ValueError, match="finite 1-D"):
        _validate_frame_times(bad, bad.size)


def test_position_units_are_checked_without_rescaling(env):
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = PositionOverlay(
        positions, units="centimeters", frame="arena"
    ).convert_to_data(np.array([0.0, 0.1]), 2, env)
    np.testing.assert_array_equal(result.data, positions)
    for kwargs in ({"units": "m"}, {"frame": "camera"}):
        with pytest.raises(ValueError, match="explicitly convert"):
            PositionOverlay(positions, **kwargs).convert_to_data(
                np.array([0.0, 0.1]), 2, env
            )


def test_rates_align_unit_identity_before_row_order_and_validate_geometry(env):
    rates = np.vstack([np.full(env.n_bins, 20.0), np.full(env.n_bins, 10.0)])
    result = SpatialRatesResult(
        rates, np.ones(env.n_bins), env, "binned", 2.0, unit_ids=np.array(["b", "a"])
    )
    raster = SpikeRasterOverlay(
        [np.array([0.1]), np.array([0.2])],
        unit_ids=["a", "b"],
        order=[1, 0],
        rates=result,
    )
    data = raster.convert_to_data(np.array([0.0, 0.2]), 2, env)
    np.testing.assert_array_equal(data.unit_ids, ["b", "a"])
    np.testing.assert_array_equal(data.rates, rates)
    moved = Environment.from_samples(
        np.array([[100.0, 100.0], [110.0, 110.0], [100.0, 110.0], [110.0, 100.0]]),
        bin_size=2.0,
        units="cm",
        frame="arena",
    )
    assert moved.n_bins == env.n_bins
    with pytest.raises(ValueError, match="geometry"):
        raster.convert_to_data(np.array([0.0, 0.2]), 2, moved)
    with pytest.raises(ValueError, match="missing raster unit IDs"):
        SpikeRasterOverlay(
            [np.array([0.1])], unit_ids=["absent"], rates=result
        ).convert_to_data(np.array([0.0, 0.2]), 2, env)


def test_crop_resize_maps_pixel_centers_back_to_original_calibration(env):
    pytest.importorskip("cv2")
    original = np.eye(3)
    original[0, 0], original[1, 1] = 2.0, -2.0
    calibration = VideoCalibration(Affine2D(original), (11, 9))
    frames = np.arange(2 * 9 * 11 * 3, dtype=np.uint8).reshape(2, 9, 11, 3)
    overlay = VideoOverlay(frames, calibration, crop=(2, 1, 7, 7), downsample=2)
    data = overlay.convert_to_data(np.array([0.0, 0.1]), 2, env)
    assert data.reader.shape == (2, 3, 3, 3)
    pixels = np.array([[0.0, 0.0], [2.0, 2.0]])
    source_pixels = (pixels + 0.5) * (7 / 3) - 0.5 + np.array([2.0, 1.0])
    np.testing.assert_allclose(
        data.transform_to_env(pixels), calibration.transform_px_to_cm(source_pixels)
    )
    np.testing.assert_array_equal(calibration.transform_px_to_cm.A, original)
    assert calibration.frame_size_px == (11, 9)


@pytest.mark.parametrize(
    "overlay",
    [
        PositionOverlay(np.array([[np.inf, 0.0]])),
        HeadDirectionOverlay(np.array([np.inf])),
        BodypartOverlay({"nose": np.array([[np.inf, 0.0]])}),
    ],
)
def test_infinity_is_not_missing_tracking(env, overlay):
    with pytest.raises(ValueError, match="Inf"):
        overlay.convert_to_data(np.array([0.0]), 1, env)


def test_file_video_processing_uses_actual_resized_dimensions(env, monkeypatch):
    from neurospatial import _video_io

    class Reader:
        n_frames = 2
        original_size_px = (11, 9)
        frame_size_px = (3, 3)

        def __init__(self, source, **kwargs):
            assert kwargs["crop"] == (2, 1, 7, 7)
            assert kwargs["downsample"] == 2

        def close(self):
            pass

    monkeypatch.setattr(_video_io, "_VideoReader", Reader)
    calibration = VideoCalibration(Affine2D(np.eye(3)), (11, 9))
    data = VideoOverlay(
        "example.mp4", calibration, crop=(2, 1, 7, 7), downsample=2
    ).convert_to_data(np.array([0.0, 0.1]), 2, env)
    np.testing.assert_allclose(
        data.transform_to_env(np.array([[0.0, 0.0]])), [[2.0 + 2 / 3, 1.0 + 2 / 3]]
    )
    assert data._owns_reader


def test_nwb_adapter_keeps_declared_position_units(monkeypatch):
    from types import SimpleNamespace

    from neurospatial.io.nwb import _behavior
    from neurospatial.io.nwb._overlays import position_overlay_from_nwb

    sample = SimpleNamespace(
        positions=np.array([[1.0, 2.0]]), times=np.array([0.0]), units="m"
    )
    monkeypatch.setattr(_behavior, "read_position", lambda *args, **kwargs: sample)
    overlay = position_overlay_from_nwb(object())
    assert overlay.units == "m"
    np.testing.assert_array_equal(overlay.positions, sample.positions)


def test_coarse_frame_trails_retain_source_breaks(env):
    times = np.array([0.0, 0.1, 0.2, 0.3])
    positions = np.column_stack([times, times])
    breaks = np.array([False, False, True, False])
    data = PositionOverlay(positions, times, break_before=breaks).convert_to_data(
        np.array([0.0, 0.3]), 2, env
    )
    assert np.isfinite(data.data).all()  # Both endpoints remain observations.
    np.testing.assert_array_equal(data.sample_valid, [True, True])
    np.testing.assert_array_equal(data.interval_valid, [False])


def test_supplied_masks_cannot_make_missing_values_observed():
    result = sample_at_times(
        np.array([0.0, 0.2]),
        np.array([1.0, np.nan]),
        np.array([0.0, 0.05, 0.2]),
        interp="nearest",
        sample_valid=np.array([True, True]),
        interval_valid=np.array([True]),
    )
    np.testing.assert_allclose(result, [1.0, np.nan, np.nan], equal_nan=True)


def test_single_sample_object_rate_keeps_its_time_axis(env):
    data = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]), np.array([[1.0, 1.0]]), firing_rates=np.array([2.0])
    ).convert_to_data(np.array([0.0]), 1, env)
    assert data.firing_rates.shape == (1,)
    np.testing.assert_array_equal(data.firing_rates, [2.0])
