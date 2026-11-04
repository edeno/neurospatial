"""A session's video: frame times, calibration, refusals and the drawn scene."""

from __future__ import annotations

import dataclasses
import hashlib

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.image as mpimg
import numpy as np
import pytest

from neurospatial._exceptions import InputError
from neurospatial.datasets._overview import VIDEO_FIELD_OPACITY
from neurospatial.datasets._video import (
    CALIBRATION_SOURCES,
    Landmarks,
    LedFit,
    VideoSource,
    fit_pixel_affine,
    led_fit_watch_out,
    mirror_watch_out,
    video_calibration,
    video_frame_times,
)

#: The synthetic camera: pixels per cm, and the border around the box, px.
SCALE, MARGIN = 2.0, 10.0
FPS = 30.0
#: Seconds of video, and the frames the overlay is read back at.
DURATION = 4.0
CHECKED = (1.0, 2.0, 3.0)
DPI = 200


def _to_px(positions, bounds):
    """cm to video pixels (column, row): x to the right, y up the frame."""
    (x0, _), (_, y1) = bounds
    return np.column_stack(
        [
            MARGIN + SCALE * (positions[:, 0] - x0),
            MARGIN + SCALE * (y1 - positions[:, 1]),
        ]
    )


def _frame_size(bounds):
    (x0, x1), (y0, y1) = bounds
    return (
        int(np.ceil(SCALE * (x1 - x0) + 2 * MARGIN)),
        int(np.ceil(SCALE * (y1 - y0) + 2 * MARGIN)),
    )


def _write_video(path, dots, size):
    """An AVI of black frames, each with a white disk of radius 8 px at its dot."""
    cv2 = pytest.importorskip("cv2")
    width, height = size
    rows, columns = np.mgrid[0:height, 0:width]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height)
    )
    for column, row in dots:
        frame = np.zeros((height, width, 3), np.uint8)
        frame[(columns - column) ** 2 + (rows - row) ** 2 <= 8**2] = 255
        writer.write(frame)
    writer.release()
    return path


def _source(path, cache, frame_times, calibration, basis, tried=()):
    times, times_basis, _ = video_frame_times(frame_times, frame_times)
    return VideoSource(
        dataset="synthetic_2026",
        external_file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
        frame_times=times,
        frame_times_basis=times_basis,
        calibration=calibration,
        calibration_basis=basis,
        calibration_tried=tried,
        url="https://example.org/never-fetched.avi",
        cache=cache,
    )


@pytest.fixture
def camera(square_recording, tmp_path):
    """The box session without headings, filmed from above by a known camera.

    The video draws a white dot at the session's position at each frame,
    through ``_to_px``. The calibration is fitted from pixel tracking (the
    same mapping at every sample) to the positions, as a file storing both
    would give.
    """
    rec = dataclasses.replace(square_recording, headings=None, heading_source="")
    bounds = rec.env.dimension_ranges
    size = _frame_size(bounds)
    frame_times = np.arange(round(DURATION * FPS)) / FPS
    at_frames = np.column_stack(
        [np.interp(frame_times, rec.times, rec.positions[:, i]) for i in range(2)]
    )
    folder = tmp_path / "synthetic_2026"
    dot = _write_video(folder / "dot.avi", _to_px(at_frames, bounds), size)
    dark = _write_video(
        folder / "dark.avi", np.full((len(frame_times), 2), -99.0), size
    )
    calibration, basis, tried = video_calibration(
        bin_size=2.0,
        tracking=(_to_px(rec.positions, bounds), rec.positions, size, "simulated"),
    )
    assert tried == (f"{CALIBRATION_SOURCES[0]}: none given",)
    return {
        "rec": rec,
        "positions": at_frames,
        "frame_times": frame_times,
        "size": size,
        "dot": _source(dot, tmp_path, frame_times, calibration, basis, tried),
        "dark": _source(dark, tmp_path, frame_times, calibration, basis, tried),
    }


def _exported(rec, source, tmp_path, name):
    """The scene with ``source`` drawn below, exported at ``CHECKED``; RGB frames."""
    scene = dataclasses.replace(rec, video=source).animation(
        time_range=(0.0, DURATION), video=True
    )
    frames = tmp_path / name
    scene.export(
        tmp_path / f"{name}.html",
        time_range=(CHECKED[0], CHECKED[-1] + 0.5),
        fps=1.0,
        dpi=DPI,
        embed=False,
        frames_dir=frames,
    )
    paths = sorted(frames.glob("frame_*.png"))
    assert len(paths) == len(CHECKED)
    return [(mpimg.imread(path)[..., :3] * 255).astype(int) for path in paths]


def _marker_centre(image):
    """The centre of the red, round PositionOverlay marker, (column, row)."""
    from scipy import ndimage

    red = (image[..., 0] > 200) & (image[..., 1] < 80) & (image[..., 2] < 80)
    labels, n_blobs = ndimage.label(red)
    round_blobs = [
        index + 1
        for index, (rows, cols) in enumerate(ndimage.find_objects(labels))
        # The time cursor of the raster panel is red too, but a line.
        if max(rows.stop - rows.start, cols.stop - cols.start)
        <= 2 * min(rows.stop - rows.start, cols.stop - cols.start)
    ]
    assert len(round_blobs) == 1, f"{len(round_blobs)} round red blobs of {n_blobs}"
    rows, cols = np.nonzero(labels == round_blobs[0])
    return np.array([cols.mean(), rows.mean()])


def _dot_centre(image, dark):
    """The centre of what the dot adds to the frame, (column, row).

    The difference from the same scene over a dark video is the video's
    contribution alone, wherever the see-through field or the marker leaves
    the dot visible; a ring around the marker when the two coincide.
    """
    added = np.abs(image - dark).sum(axis=2) > 150
    rows, cols = np.nonzero(added)
    assert rows.size, "the dot is not visible in the frame"
    return np.array([cols.mean(), rows.mean()])


def _checked_positions(camera):
    """The session's positions at the frames read back, cm."""
    return camera["positions"][np.searchsorted(camera["frame_times"], CHECKED)]


def _distances(camera, tmp_path, calibration, name):
    """Dot-to-marker distance at each checked frame, in camera pixels.

    The exported frame's scale, rendered pixels per cm, is measured from the
    markers themselves: their spacing against the positions' spacing.
    """
    dot = dataclasses.replace(camera["dot"], calibration=calibration)
    dark = dataclasses.replace(camera["dark"], calibration=calibration)
    lit = _exported(camera["rec"], dot, tmp_path, f"{name}-dot")
    unlit = _exported(camera["rec"], dark, tmp_path, f"{name}-dark")
    markers = np.array([_marker_centre(base) for base in unlit])
    dots = np.array(
        [_dot_centre(image, base) for image, base in zip(lit, unlit, strict=True)]
    )
    positions = _checked_positions(camera)
    rendered_per_cm = np.linalg.norm(markers[-1] - markers[0]) / np.linalg.norm(
        positions[-1] - positions[0]
    )
    return np.linalg.norm(dots - markers, axis=1) / (rendered_per_cm / SCALE)


def test_overlay_lands_on_tracked_point(camera, tmp_path):
    """The video's dot lies within a camera pixel of the marker; a wrong calibration misses.

    The tolerance is one camera pixel, the calibration's own unit:
    Matplotlib places an upsampled image and a marker each to about half a
    rendered pixel.
    """
    from neurospatial.ops.transforms import AffineND, VideoCalibration

    true = camera["dot"].calibration
    assert np.allclose(
        true.transform_px_to_cm(
            _to_px(camera["positions"], camera["rec"].env.dimension_ranges)
        ),
        camera["positions"],
    )
    assert np.all(_distances(camera, tmp_path, true, "true") <= 1.0)

    width, height = camera["size"]
    matrix = np.asarray(true.transform_px_to_cm.A)
    # The negative controls are informative only away from the flip's axis
    # (the frame's middle row) and the scaling's centre (pixel 0, 0).
    checked = _checked_positions(camera)
    middle = true.transform_px_to_cm(np.array([[width / 2, (height - 1) / 2]]))[0]
    assert np.all(np.abs(checked[:, 1] - middle[1]) > 10.0)
    flipped = VideoCalibration(
        AffineND(matrix @ np.array([[1, 0, 0], [0, -1, height - 1], [0, 0, 1]])),
        frame_size_px=(width, height),
    )
    scaled = VideoCalibration(
        AffineND(np.diag([1.2, 1.2, 1.0]) @ matrix), frame_size_px=(width, height)
    )
    for name, wrong in (("flipped", flipped), ("scaled", scaled)):
        assert np.all(_distances(camera, tmp_path, wrong, name) > 3.0), name


def test_frame_times_follow_position_clock(camera):
    """Same frames on a clock 22.6 s early: the position times, inferred, with a watch-out."""
    position_times = np.arange(300) / FPS + 7696.886
    video_times = position_times - 22.6
    times, basis, watch_out = video_frame_times(video_times, position_times)
    np.testing.assert_array_equal(times, position_times)
    assert basis.basis == "inferred"
    assert "+22.60 s" in basis.evidence
    assert "300 frames" in basis.evidence
    assert watch_out is not None
    assert "+22.60 s" in watch_out

    # The recording carries both, as a loader records them.
    rec = dataclasses.replace(
        camera["rec"],
        video=dataclasses.replace(
            camera["dot"], frame_times=times, frame_times_basis=basis
        ),
        conversions=(*camera["rec"].conversions, basis),
        watch_outs=(*camera["rec"].watch_outs, watch_out),
    )
    assert rec.video.frame_times_basis in rec.conversions
    assert any("+22.60 s" in line for line in rec.watch_outs)


def test_frame_times_drift_and_other_counts():
    """A clock offset with a slow drift is still the same frames; another count is not."""
    video = np.arange(67131) / FPS + 7674.337
    drifting = 7696.886 + (video - video[0]) * 1.000257
    times, basis, watch_out = video_frame_times(video, drifting)
    np.testing.assert_array_equal(times, drifting)
    assert basis.basis == "inferred"
    assert "+22.55 to +23.12 s" in watch_out

    times, basis, watch_out = video_frame_times(video, drifting[:-100])
    np.testing.assert_array_equal(times, video)
    assert basis.basis == "stated"
    assert "another count" in basis.evidence
    assert "start +22.55 s and end" not in watch_out
    assert "start -22.55 s" in watch_out
    # Another count on the same span: the video's own clock, nothing to flag.
    times, basis, watch_out = video_frame_times(video, video[:-1] + 1e-3)
    assert (basis.basis, watch_out) == ("stated", None)

    # Equal counts, but no offset and rate relates the clocks: not the same frames.
    jittered = drifting + np.random.default_rng(0).uniform(-0.1, 0.1, video.size)
    times, basis, _ = video_frame_times(video, np.sort(jittered))
    np.testing.assert_array_equal(times, video)
    assert basis.basis == "stated"

    # The same clock: the video's own times, stated, no watch-out.
    times, basis, watch_out = video_frame_times(video, video)
    assert (basis.basis, watch_out) == ("stated", None)


def test_calibration_sources_in_order(camera):
    """A stated transform wins; a fit wider than a bin falls through to landmarks."""
    from neurospatial.ops.transforms import AffineND

    rec = camera["rec"]
    bounds = rec.env.dimension_ranges
    size = camera["size"]
    pixels = _to_px(rec.positions, bounds)
    transform = AffineND(np.asarray(camera["dot"].calibration.transform_px_to_cm.A))
    calibration, basis, tried = video_calibration(
        bin_size=2.0,
        stated=(transform, size, "the file's video series"),
        tracking=(pixels, rec.positions, size, "simulated"),
    )
    assert basis.basis == "stated"
    assert tried == ()
    assert calibration.transform_px_to_cm == transform

    noisy = pixels + np.random.default_rng(1).normal(0, 10.0, pixels.shape)
    (x0, x1), (y0, y1) = bounds
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    landmarks = Landmarks(
        px=_to_px(corners, bounds),
        env=corners,
        frame_size_px=size,
        frame=0,
        placed_by="a test, 2026-11-03",
    )
    calibration, basis, tried = video_calibration(
        bin_size=2.0,
        tracking=(noisy, rec.positions, size, "simulated"),
        landmarks=landmarks,
    )
    assert "landmarks" in basis.text
    assert "residual" in basis.evidence
    assert len(tried) == 2
    assert "more than one 2 cm bin" in tried[1]
    np.testing.assert_allclose(
        calibration.transform_px_to_cm(landmarks.px), corners, atol=1e-9
    )


def test_refuses_without_calibration(camera):
    """No calibration source: InputError naming each one tried. No video: says so."""
    absent = (
        "the file's video series states no pixel size",
        "the file stores no pixel tracking",
        "the registry holds no landmarks for this video",
        "the registry holds no LED fit for this video",
    )
    calibration, basis, tried = video_calibration(bin_size=2.0, absent=absent)
    assert calibration is None
    assert basis is None
    uncalibrated = dataclasses.replace(
        camera["dot"], calibration=None, calibration_basis=None, calibration_tried=tried
    )
    rec = dataclasses.replace(camera["rec"], video=uncalibrated)
    summary = " ".join(repr(rec).split())
    assert "video: 120 frames" in summary
    assert "no calibration, so it is not drawn" in summary
    with pytest.raises(InputError, match="no calibration") as refused:
        rec.animation(video=True)
    message = str(refused.value)
    for source, reason in zip(CALIBRATION_SOURCES, absent, strict=True):
        assert f"{source}: {reason}" in message
    # Nothing was fetched or drawn: the frames are never stretched over the box.
    assert rec.animation().overlays[0].__class__.__name__ == "PositionOverlay"

    with pytest.raises(InputError, match="has no video"):
        dataclasses.replace(camera["rec"], video=None).animation(video=True)


def test_fetch_verifies_the_checksum(camera, tmp_path):
    """A cached video is used only when its SHA-256 matches the pinned one."""
    source = camera["dot"]
    assert source.fetch() == tmp_path / "synthetic_2026" / "dot.avi"
    tampered = dataclasses.replace(source, sha256="0" * 64)
    with pytest.raises((ValueError, FileExistsError)):
        tampered.fetch()


def test_video_source_refuses_bad_fields(camera):
    """Two locations, a bad checksum and a calibration without its basis each fail."""
    with pytest.raises(InputError) as refused:
        dataclasses.replace(
            camera["dot"], asset_id="abc", sha256="xyz", calibration_basis=None
        )
    message = str(refused.value)
    assert "exactly one of asset_id and url" in message
    assert "'xyz' is not 64 hexadecimal digits" in message
    assert "calibration and calibration_basis together" in message
    assert not camera["dot"].frame_times.flags.writeable


#: A camera: 0.5 cm per pixel, turned 10 degrees, rows running down (y up).
TRUE_AFFINE = np.array(
    [
        [0.5 * np.cos(0.17), 0.5 * np.sin(0.17), -5.0],
        [0.5 * np.sin(0.17), -0.5 * np.cos(0.17), 105.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_led_fit_recovers_a_known_camera():
    """A known affine plus 1 px of noise per axis is recovered by the fit.

    Tolerances, set before the first run: the fitted map puts every corner
    of a 220 x 220 px frame within 0.2 cm of the true one (the expected
    error at 3000 samples is about 0.02 cm), and the median residual is
    within 10% of 0.589 cm, the median distance for 0.5 cm of Gaussian
    noise per axis (0.5 sqrt(2 ln 2)).
    """
    from neurospatial.ops.transforms import AffineND

    rng = np.random.default_rng(3)
    pixels = rng.uniform(0, 220, (3000, 2))
    positions = AffineND(TRUE_AFFINE)(pixels)
    noisy = pixels + rng.normal(0.0, 1.0, pixels.shape)
    noisy[::50] = np.nan  # an undetected LED is left out
    transform, in_cm, in_px = fit_pixel_affine(noisy, positions)
    corners = np.array([[0, 0], [220, 0], [0, 220], [220, 220]], dtype=float)
    error = np.linalg.norm(transform(corners) - AffineND(TRUE_AFFINE)(corners), axis=1)
    assert np.all(error < 0.2)
    assert in_cm.size == 3000 - 60
    assert np.median(in_cm) == pytest.approx(0.5 * np.sqrt(2 * np.log(2)), rel=0.1)
    assert np.median(in_px) == pytest.approx(np.sqrt(2 * np.log(2)), rel=0.1)


def test_robust_fit_ignores_gross_outliers():
    """20% of detections thrown anywhere in the frame do not move the robust fit.

    Tolerance, set before the first run: every frame corner within 0.1 cm
    of the true map (1 px of noise is 0.5 cm; the expected error with 2,400
    good samples is about 0.03 cm). Plain least squares misses by far more.
    """
    from neurospatial.ops.transforms import AffineND

    rng = np.random.default_rng(5)
    pixels = rng.uniform(0, 220, (3000, 2))
    positions = AffineND(TRUE_AFFINE)(pixels)
    detected = pixels + rng.normal(0.0, 1.0, pixels.shape)
    gross = rng.random(3000) < 0.2
    detected[gross] = rng.uniform(0, 220, (int(gross.sum()), 2))
    corners = np.array([[0, 0], [220, 0], [0, 220], [220, 220]], dtype=float)
    truth = AffineND(TRUE_AFFINE)(corners)
    robust, in_cm, _ = fit_pixel_affine(
        detected, positions, huber_cm=2.0, outlier_cm=5.0
    )
    assert np.all(np.linalg.norm(robust(corners) - truth, axis=1) < 0.1)
    assert np.mean(in_cm > 5.0) == pytest.approx(gross.mean(), abs=0.01)
    plain, _, _ = fit_pixel_affine(detected, positions)
    assert np.max(np.linalg.norm(plain(corners) - truth, axis=1)) > 1.0


def _led_fit(median_cm):
    return LedFit(
        matrix=TRUE_AFFINE,
        frame_size_px=(220, 220),
        method="the brightest point after a blur",
        frames="every 30th frame from 0 (100 frames)",
        n_frames=100,
        median_cm=median_cm,
        p90_cm=4.0 * median_cm,
        median_px=2.0 * median_cm,
        p90_px=8.0 * median_cm,
        outlier_cm=5.0,
        outlier_share=0.17,
        lag_scan=((-1, 1.1), (0, median_cm), (1, 0.9)),
        fitted_by="a test, 2026-11-04",
    )


def test_led_fit_is_the_fourth_source():
    """Used after the other three, with its method and residuals; refused above a bin."""
    calibration, basis, tried = video_calibration(bin_size=2.5, led_fit=_led_fit(1.5))
    assert len(tried) == 3
    assert basis.basis == "inferred"
    assert "the brightest point after a blur" in basis.evidence
    assert "median 1.5 cm (3 px), 90th percentile 6 cm (12 px)" in basis.evidence
    assert "the median is the statistic compared with the 2.5 cm bin" in basis.evidence
    np.testing.assert_allclose(calibration.transform_px_to_cm.A, TRUE_AFFINE)
    assert "90th-percentile residual 6 cm (12 px)" in led_fit_watch_out(_led_fit(1.5))
    assert "aligned by construction" in led_fit_watch_out(_led_fit(1.5))
    assert "17.0% of LED detections are more than 5 cm off" in led_fit_watch_out(
        _led_fit(1.5)
    )
    assert "17.0% of detections are more than 5 cm off" in basis.evidence
    assert "-1: 1.1, +0: 1.5, +1: 0.9" in basis.evidence

    calibration, basis, tried = video_calibration(bin_size=2.5, led_fit=_led_fit(3.0))
    assert calibration is None
    assert basis is None
    assert len(tried) == 4
    assert tried[3].startswith(CALIBRATION_SOURCES[3])
    assert "the median is more than one 2.5 cm bin" in tried[3]


def test_led_detection_finds_the_brightest_spot():
    """The fit script's detector finds a lit disk on a dark frame, to a pixel."""
    pytest.importorskip("cv2")
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "fit_video_led_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("fit_video_led_calibration", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    rows, columns = np.mgrid[0:120, 0:80]
    frame = np.zeros((120, 80, 3), np.uint8)
    frame[(columns - 23) ** 2 + (rows - 91) ** 2 <= 9] = (90, 200, 255)
    frame[5, 70] = 255  # one hot pixel is blurred away
    column, row = script.detect_led(frame)
    assert abs(column - 23) <= 1
    assert abs(row - 91) <= 1


def test_track_session_video_is_drawn_in_the_plane(track_recording, tmp_path):
    """A track's bins lie along it, so its video scene is a grid over its footprint.

    The track stays the scene's along-track panel, and the occupancy is
    counted in the grid's bins.
    """
    from neurospatial.ops.transforms import AffineND

    rec = track_recording
    (x0, y0), (x1, y1) = rec.env.arena.domain_bounds()
    bounds = ((x0, x1), (y0, y1))
    size = _frame_size(bounds)
    frame_times = np.arange(round(DURATION * FPS)) / FPS
    path = _write_video(
        tmp_path / "synthetic_2026" / "track.avi",
        np.full((len(frame_times), 2), -99.0),
        size,
    )
    to_cm = AffineND(
        np.array(
            [
                [1 / SCALE, 0.0, x0 - MARGIN / SCALE],
                [0.0, -1 / SCALE, y1 + MARGIN / SCALE],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    calibration, basis, tried = video_calibration(
        bin_size=2.0, stated=(to_cm, size, "a test camera")
    )
    source = _source(path, tmp_path, frame_times, calibration, basis, tried)
    scene = dataclasses.replace(rec, video=source).animation(
        time_range=(0.0, DURATION), video=True
    )
    assert scene.track is rec.env
    assert not scene.env.is_linearized_track
    assert scene.env.n_dims == 2
    (low_x, high_x), (low_y, high_y) = scene.env.dimension_ranges
    assert low_x <= x0 + 2.0
    assert high_x >= x1 - 2.0
    assert low_y <= y0 + 2.0
    assert high_y >= y1 - 2.0
    occupancy = scene.fields["occupancy"][0]
    expected = scene.env.occupancy(rec.times, rec.positions)
    np.testing.assert_allclose(occupancy, np.where(expected > 0, expected, np.nan))
    assert type(scene.overlays[0]).__name__ == "VideoOverlay"
    assert scene.fields["occupancy"].max_opacity == VIDEO_FIELD_OPACITY < 1
    # Without the video, the track is drawn along itself, opaque, as before.
    plain = rec.animation(time_range=(0.0, DURATION))
    assert plain.env is rec.env
    assert plain.fields["occupancy"].max_opacity == 1.0


def test_mirror_watch_out_follows_the_handedness():
    """A calibration keeping the image's handedness shows the video mirrored."""
    from neurospatial.ops.transforms import AffineND, VideoCalibration

    flipped = VideoCalibration(AffineND(TRUE_AFFINE), frame_size_px=(220, 220))
    assert mirror_watch_out(flipped) is None  # y up the rows, as a player shows it
    kept = np.diag([-1.0, 1.0, 1.0]) @ TRUE_AFFINE
    mirrored = VideoCalibration(AffineND(kept), frame_size_px=(220, 220))
    assert "mirrored relative to the raw video file" in mirror_watch_out(mirrored)
