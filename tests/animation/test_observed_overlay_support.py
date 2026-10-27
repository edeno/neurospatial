"""Tracking observations keep the same scientific support through animation."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation._sampling import resolve_support, sample_at_times
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
from neurospatial.ops.transforms import AffineND, VideoCalibration


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
    data = overlay.convert_to_data(queries, len(queries), env).on_clock(queries)
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
    data = _convert_overlays_to_data([overlay], np.array([0.0, 2.2]), 2, env)[
        0
    ].positions[0]
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
    data = (
        BodypartOverlay({"nose": nose, "body": body}, times, max_gap=0.3)
        .convert_to_data(query, 5, env)
        .on_clock(query)
    )
    assert np.isnan(data.bodyparts["nose"][1:4]).all()
    assert np.isfinite(data.bodyparts["body"]).all()
    headings = (
        HeadDirectionOverlay(np.array([3.0, np.nan, -3.0]), times, max_gap=0.3)
        .convert_to_data(query, 5, env)
        .on_clock(query)
    )
    assert np.isnan(headings.data[1:4]).all()
    exact = HeadDirectionOverlay(np.array([3 * np.pi, -3.0]), np.array([0.0, 0.2]))
    clock = np.array([0.0, 0.1, 0.2])
    aligned = exact.convert_to_data(clock, 3, env).on_clock(clock).data
    assert aligned[0] == 3 * np.pi
    assert abs(aligned[1]) > 3.0


def test_object_vectors_follow_missing_position_and_declared_break(env):
    overlay = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]),
        np.array([[1.0, 1.0], [2.0, 2.0], [np.nan, np.nan]]),
        np.array([0.0, 0.2, 0.4]),
        break_before=np.array([False, True, False]),
    )
    clock = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    positions = overlay.convert_to_data(clock, 5, env).on_clock(clock).animal_positions
    np.testing.assert_array_equal(
        np.isfinite(positions).all(axis=1), [True, False, True, False, False]
    )


TIMES = np.arange(40) / 10
MAX_GAP = 0.35
# Row 5 alone and rows 11-12 are bridged (0.2 s and 0.3 s gaps); rows 20-26
# leave a 0.8 s tracking loss; missing row 32 carries a declared break.
LOST = np.isin(np.arange(40), [5, 11, 12, *range(20, 27), 32])
BREAKS = np.arange(40) == 32


def _deleted(lost):
    """The kept rows, with each lost row's break moved to the next kept row."""
    kept = np.flatnonzero(~lost)
    crossed = np.cumsum(BREAKS)[kept]
    return kept, np.r_[False, crossed[1:] > crossed[:-1]]


def _missing_and_deleted(make, rows, lost):
    """``make(rows, times, break_before)`` with NaN rows, and with them deleted."""
    kept, breaks = _deleted(lost)
    with_nan = np.array(rows, dtype=float)
    with_nan[lost] = np.nan
    return make(with_nan, TIMES, BREAKS), make(rows[kept], TIMES[kept], breaks)


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(139)
    angles = np.angle(np.exp(1j * np.cumsum(rng.normal(0, 0.8, 40))))
    return {
        "positions": rng.uniform(1, 9, (40, 2)),
        "tail": rng.uniform(1, 9, (40, 2)),
        "angles": angles,
        "vectors": np.column_stack([np.cos(angles), np.sin(angles)]),
        "rates": rng.uniform(0, 20, 40),
        # Inside the recording, at missing rows' times, and between rows.
        "queries": np.unique(np.r_[TIMES, np.arange(-5, 400) * 0.01 + 0.004]),
    }


@pytest.mark.parametrize("interp", ["linear", "nearest"])
@pytest.mark.parametrize(
    ("kind", "attribute"),
    [
        ("position", "data"),
        ("angles", "data"),
        ("vectors", "data"),
        ("object", "animal_positions"),
    ],
)
def test_missing_rows_are_sampled_like_deleted_rows(
    env, recording, interp, kind, attribute
):
    """Every view samples the valid observations around a missing row."""
    from neurospatial.animation import Animation
    from neurospatial.animation._prepared import PreparedScene

    options = {"interp": interp, "max_gap": MAX_GAP}
    makers = {
        "position": lambda rows, times, breaks: PositionOverlay(
            rows, times, trail_length=5, break_before=breaks, **options
        ),
        "angles": lambda rows, times, breaks: HeadDirectionOverlay(
            rows, times, break_before=breaks, **options
        ),
        "vectors": lambda rows, times, breaks: HeadDirectionOverlay(
            rows, times, break_before=breaks, **options
        ),
        "object": lambda rows, times, breaks: ObjectVectorOverlay(
            np.array([[5.0, 5.0]]), rows, times, break_before=breaks, **options
        ),
    }
    rows = recording["positions" if kind in ("position", "object") else kind]
    queries = recording["queries"]
    frames = TIMES
    sampled = []
    for overlay in _missing_and_deleted(makers[kind], rows, LOST):
        data = overlay.convert_to_data(frames, len(frames), env)
        on_clock = getattr(data.on_clock(queries), attribute)
        with Animation(
            env,
            np.zeros((len(frames), env.n_bins)),
            frame_times=frames,
            overlays=[overlay],
        ) as scene:
            item = next(scene._overlay_data.sampled())
            prepared = PreparedScene(scene)
            values = getattr(item, attribute)
            circular = kind == "angles"
            views = np.stack(
                [prepared.sample(item, values, t, circular=circular) for t in queries]
            )
            trails = (
                [prepared.trail(item, t) for t in queries] if kind == "position" else []
            )
        sampled.append((on_clock, views, trails))
    (
        (missing, missing_views, missing_trails),
        (deleted, deleted_views, deleted_trails),
    ) = sampled
    np.testing.assert_array_equal(missing, deleted)
    np.testing.assert_array_equal(missing_views, deleted_views)
    np.testing.assert_array_equal(missing, missing_views)
    for (segments, alphas), (expected, expected_alphas) in zip(
        missing_trails, deleted_trails, strict=True
    ):
        np.testing.assert_array_equal(segments, expected)
        np.testing.assert_array_equal(alphas, expected_alphas)
    if kind == "position":
        # Trails join the observations on either side of a bridged gap.
        across = rows[[4, 6]]
        assert any(
            (segments == across).all(axis=(1, 2)).any()
            for segments, _ in missing_trails
        )
    # The 0.8 s tracking loss and the declared break stay unsampled.
    unsupported = ((queries > 1.9) & (queries < 2.7)) | (
        (queries > 3.1) & (queries < 3.3)
    )
    assert np.isnan(missing[unsupported]).all()
    assert np.isfinite(missing[(queries >= 0) & (queries <= 3.9) & ~unsupported]).all()


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_missing_pose_and_rate_rows_are_sampled_like_deleted_rows(
    env, recording, interp
):
    """Each body part and an object-vector rate skip only their own missing rows."""
    from neurospatial.animation import Animation, Skeleton
    from neurospatial.animation._prepared import PreparedScene

    queries = recording["queries"]
    options = {"interp": interp, "max_gap": MAX_GAP}
    tail_lost = np.isin(np.arange(40), [3, 4, 15, 32])
    skeleton = Skeleton(name="rat", nodes=("nose", "tail"), edges=(("nose", "tail"),))
    nose, tail = recording["positions"].copy(), recording["tail"].copy()
    nose[LOST], tail[tail_lost] = np.nan, np.nan
    body = BodypartOverlay(
        {"nose": nose, "tail": tail},
        TIMES,
        skeleton=skeleton,
        break_before=BREAKS,
        **options,
    )
    with Animation(
        env, np.zeros((40, env.n_bins)), frame_times=TIMES, overlays=[body]
    ) as scene:
        prepared = PreparedScene(scene)
        poses = [
            prepared.pose(scene._overlay_data.bodypart_sets[0], t) for t in queries
        ]
    on_clock = body.convert_to_data(TIMES, 40, env).on_clock(queries).bodyparts
    for part, lost in (("nose", LOST), ("tail", tail_lost)):
        kept, breaks = _deleted(lost)
        expected = (
            BodypartOverlay(
                {part: recording["positions" if part == "nose" else "tail"][kept]},
                TIMES[kept],
                break_before=breaks,
                **options,
            )
            .convert_to_data(TIMES, 40, env)
            .on_clock(queries)
            .bodyparts[part]
        )
        index = 0 if part == "nose" else 1
        np.testing.assert_array_equal(on_clock[part], expected)
        np.testing.assert_array_equal(
            np.stack([pose.points[index] for pose in poses]), expected
        )

    # A rate is usable where its own row and the position row are observed.
    rates = recording["rates"].copy()
    rates[tail_lost] = np.nan
    vectors = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]),
        recording["positions"],
        TIMES,
        firing_rates=rates,
        break_before=BREAKS,
        **options,
    )
    kept, breaks = _deleted(tail_lost)
    deleted = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]),
        recording["positions"][kept],
        TIMES[kept],
        firing_rates=recording["rates"][kept],
        break_before=breaks,
        **options,
    )
    actual, expected = (
        overlay.convert_to_data(TIMES, 40, env).on_clock(queries).firing_rates
        for overlay in (vectors, deleted)
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("interp", ["linear", "nearest"])
def test_events_on_a_path_with_missing_rows_match_deleted_rows(env, recording, interp):
    from neurospatial.animation.overlays import EventOverlay

    queries = recording["queries"]
    events = queries[(queries >= 0) & (queries <= 3.9)][::3]
    located = [
        EventOverlay(
            event_times=events,
            positions=rows,
            position_times=times,
            interp=interp,
            max_gap=MAX_GAP,
            break_before=breaks,
        )
        .convert_to_data(TIMES, 40, env)
        .event_positions
        for rows, times, breaks in _missing_and_deleted(
            lambda *args: args, recording["positions"], LOST
        )
    ]
    np.testing.assert_array_equal(located[0]["event"], located[1]["event"])


def test_nearest_position_across_a_missing_row_is_the_nearer_observation(env):
    """Between observations at 0 and 0.8 s, 0.5 s is nearer the later one."""
    times = np.array([0.0, 0.4, 0.8])
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 10.0]])
    data = PositionOverlay(positions, times, interp="nearest").convert_to_data(
        times, 3, env
    )
    np.testing.assert_array_equal(
        data.on_clock(np.array([0.3, 0.4, 0.5])).data,
        [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]],
    )


@pytest.mark.parametrize(
    "bad", [np.array([np.nan]), np.array([np.inf]), np.ones((2, 1)), np.array(1.0)]
)
def test_frame_clock_is_finite_and_one_dimensional(bad):
    with pytest.raises(ValueError, match="finite 1-D"):
        _validate_frame_times(bad, bad.size)


@pytest.mark.parametrize("overlay_type", [PositionOverlay, BodypartOverlay])
def test_position_units_are_checked_without_rescaling(env, overlay_type):
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = {"nose": positions} if overlay_type is BodypartOverlay else positions
    result = overlay_type(values, units="centimeters", frame="arena").convert_to_data(
        np.array([0.0, 0.1]), 2, env
    )
    coordinates = (
        result.bodyparts["nose"] if overlay_type is BodypartOverlay else result.data
    )
    np.testing.assert_array_equal(coordinates, positions)
    for kwargs in ({"units": "m"}, {"frame": "camera"}):
        with pytest.raises(ValueError, match="explicitly convert"):
            overlay_type(values, **kwargs).convert_to_data(np.array([0.0, 0.1]), 2, env)


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
    with pytest.raises(ValueError, match="differs from the animated environment"):
        raster.convert_to_data(np.array([0.0, 0.2]), 2, moved)
    with pytest.raises(ValueError, match="missing raster unit IDs"):
        SpikeRasterOverlay(
            [np.array([0.1])], unit_ids=["absent"], rates=result
        ).convert_to_data(np.array([0.0, 0.2]), 2, env)


def test_crop_resize_maps_pixel_centers_back_to_original_calibration(env):
    pytest.importorskip("cv2")
    original = np.eye(3)
    original[0, 0], original[1, 1] = 2.0, -2.0
    calibration = VideoCalibration(AffineND(original), (11, 9))
    frames = np.arange(2 * 9 * 11 * 3, dtype=np.uint8).reshape(2, 9, 11, 3)
    overlay = VideoOverlay(frames, calibration, crop=(2, 1, 7, 7), downsample=2)
    data = overlay.convert_to_data(np.array([0.0, 0.1]), 2, env)
    assert (data.reader.n_frames, data.reader.frame_size_px) == (2, (3, 3))
    assert data.get_frame(1).shape == (3, 3, 3)
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
    with pytest.raises(ValueError, match="infinite"):
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
    calibration = VideoCalibration(AffineND(np.eye(3)), (11, 9))
    data = VideoOverlay(
        "example.mp4", calibration, crop=(2, 1, 7, 7), downsample=2
    ).convert_to_data(np.array([0.0, 0.1]), 2, env)
    np.testing.assert_allclose(
        data.transform_to_env(np.array([[0.0, 0.0]])), [[2.0 + 2 / 3, 1.0 + 2 / 3]]
    )
    assert isinstance(data.reader, Reader)


@pytest.mark.parametrize("frame", [None, "camera"])
def test_nwb_adapter_keeps_declared_position_units(monkeypatch, frame):
    from neurospatial.io.nwb import NWBPosition, _behavior
    from neurospatial.io.nwb._overlays import position_overlay_from_nwb

    sample = NWBPosition(np.array([0.0]), np.array([[1.0, 2.0]]), "m", frame=frame)
    monkeypatch.setattr(_behavior, "read_position", lambda *args, **kwargs: sample)
    overlay = position_overlay_from_nwb(object())
    assert overlay.units == "m"
    assert overlay.frame == frame
    np.testing.assert_array_equal(overlay.positions, sample.positions)


def test_coarse_frame_trails_retain_source_breaks(env):
    times = np.array([0.0, 0.1, 0.2, 0.3])
    positions = np.column_stack([times, times])
    breaks = np.array([False, False, True, False])
    clock = np.array([0.0, 0.3])
    data = (
        PositionOverlay(positions, times, break_before=breaks)
        .convert_to_data(clock, 2, env)
        .on_clock(clock)
    )
    assert np.isfinite(data.data).all()  # Both endpoints remain observations.
    np.testing.assert_array_equal(data.sample_valid, [True, True])
    np.testing.assert_array_equal(data.interval_valid, [False])


def test_supplied_masks_cannot_make_missing_values_observed():
    values, sample_valid, interval_valid = resolve_support(
        np.array([1.0, np.nan]), np.array([True, True]), np.array([True])
    )
    np.testing.assert_array_equal(sample_valid, [True, False])
    np.testing.assert_array_equal(interval_valid, [False])
    result = sample_at_times(
        np.array([0.0, 0.2]),
        values,
        np.array([0.0, 0.05, 0.2]),
        interp="nearest",
        sample_valid=sample_valid,
        interval_valid=interval_valid,
    )
    np.testing.assert_allclose(result, [1.0, np.nan, np.nan], equal_nan=True)


def test_single_sample_object_rate_keeps_its_time_axis(env):
    data = ObjectVectorOverlay(
        np.array([[5.0, 5.0]]), np.array([[1.0, 1.0]]), firing_rates=np.array([2.0])
    ).convert_to_data(np.array([0.0]), 1, env)
    assert data.firing_rates.shape == (1,)
    np.testing.assert_array_equal(data.firing_rates, [2.0])
