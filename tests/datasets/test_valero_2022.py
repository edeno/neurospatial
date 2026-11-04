"""Valero et al. (2022), DANDI 000568: synthetic NWB files and the real session."""

from __future__ import annotations

import dataclasses
import datetime
import itertools
from pathlib import Path

import numpy as np
import pytest

from neurospatial._exceptions import InputError
from neurospatial.datasets import load
from neurospatial.datasets._catalog import VideoAsset, sessions
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._video import CALIBRATION_SOURCES

DEFAULT = "sub-fCamk2_ses-fCamk2-201014-sess3"
#: The video clock's lead on the position clock in the synthetic file, s.
LEAD = 22.6
#: Seconds of the track epoch before the first position sample.
LATE = 20.0
#: The real file's device sets the 'manufacturer' field pynwb now deprecates.
OLD_DEVICE = pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)


def _entry(session=DEFAULT):
    return next(entry for entry in sessions("valero_2022") if entry.session == session)


def _write_session(
    path, *, video_files=("cam/uuid_external_file_0.avi",), second_run=False
):
    """An NWB file laid out as DANDI 000568's: a 1-D run along y in cm, 30 Hz.

    Five epochs, the track third; position starts ``LATE`` s into it, and the
    video's timestamps run ``LEAD`` s ahead of the positions'.
    """
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.epoch import TimeIntervals
    from pynwb.file import Subject
    from pynwb.image import ImageSeries

    nwb = pynwb.NWBFile(
        "pulsed optogenetics in CA1",
        "synthetic",
        datetime.datetime(2016, 3, 15, tzinfo=datetime.timezone.utc),
        subject=Subject(subject_id="fCamk2", species="Mus musculus"),
    )
    bounds = [0.0, 100.0, 200.0, 800.0, 900.0, 1000.0]
    nwb.add_epoch_column("epoch_name", "The name of the epoch")
    for index, (start, stop) in enumerate(itertools.pairwise(bounds)):
        nwb.add_epoch(
            start_time=start, stop_time=stop - 1e-3, epoch_name=f"fCamk2_{index}"
        )
    times = np.arange(200.0 + LATE, 790.0, 1 / 30)
    if second_run:  # the track run again in the fifth epoch, after sleep
        times = np.concatenate([times, np.arange(905.0, 990.0, 1 / 30)])
    phase = 2 * np.pi * (times - times[0]) / 20.0
    xy = np.column_stack([12.0 + 2.0 * np.sin(5 * phase), 58.0 - 55.0 * np.cos(phase)])
    position = Position(name="LinearMazePositionTracking")
    position.add_spatial_series(
        SpatialSeries(
            name="SpatialSeriesRaw",
            description="(x,y) coordinates tracking subject movement from above "
            "with camera on a PVC linear track (110 cm long, 6.35 cm wide)",
            data=xy,
            timestamps=times,
            reference_frame="Arbitrary, camera",
            unit="cm",
        )
    )
    behavior = nwb.create_processing_module("behavior", "tracking")
    behavior.add(position)
    states = TimeIntervals(name="SleepStates", description="sleep states")
    states.add_column("label", "state")
    for start, stop, label in (
        (0, 99, "NREMstate"),
        (100, 199, "WAKEstate"),
        (800, 999, "NREMstate"),
    ):
        states.add_interval(float(start), float(stop), label=label)
    behavior.add(states)
    ripples = TimeIntervals(name="RippleTimeIntervals", description="ripples")
    ripples.add_interval(50.0, 50.1)
    ripples.add_interval(850.0, 850.08)
    nwb.create_processing_module("ecephys", "lfp").add(ripples)
    device = nwb.create_device("probe")
    group = nwb.create_electrode_group("shank0", "shank", "CA1", device)
    for _ in range(4):
        nwb.add_electrode(group=group, location="CA1")
    rng = np.random.default_rng(0)
    for _ in range(3):
        nwb.add_unit(spike_times=np.sort(rng.uniform(1.0, 999.0, 400)))
    nwb.add_acquisition(
        ImageSeries(
            name="ImageSeriesTrackingVideo1",
            description="Video recorded with Basler camera.",
            external_file=list(video_files),
            starting_frame=[0, len(times) // 2][: len(video_files)],
            format="external",
            timestamps=times - LEAD,
            unit="Frames",
        )
    )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return times


@pytest.fixture
def synthetic(tmp_path):
    """A synthetic session in the cache, its entry naming its one video."""
    folder = tmp_path / "valero_2022"
    folder.mkdir()
    path = folder / "sub-x_ses-x-no-raw-data_behavior+ecephys+image.nwb"
    times = _write_session(path)
    entry = dataclasses.replace(
        _entry(),
        asset_path=f"sub-x/{path.name}",
        videos=(
            VideoAsset(
                asset_path="sub-x/cam/uuid_external_file_0.avi",
                asset_id="video-asset",
                sha256="a" * 64,
                size_bytes=1000,
            ),
        ),
    )
    return entry, path, times


def test_loads_a_session_with_its_video(synthetic):
    """Positions in cm on the position clock; the video timed by it, not calibrated."""
    from neurospatial.datasets import valero_2022

    entry, path, times = synthetic
    rec = run_checks(valero_2022.load_session(entry, path), stated=entry.info.stated)
    np.testing.assert_allclose(rec.times, times)
    assert rec.position_units == "cm"
    assert rec.env.is_linearized_track
    assert set(rec.epochs) >= {"track", "track_before_tracking", "wake", "nrem"}
    np.testing.assert_allclose(rec.epochs["track_before_tracking"], [[200.0, times[0]]])
    assert rec.events["ripples"].shape == (2, 2)
    assert set(rec.units["brain_region"]) == {"CA1"}

    video = rec.video
    assert video.external_file == "cam/uuid_external_file_0.avi"
    assert video.asset_id == "video-asset"
    assert video.cache == path.parents[1]
    np.testing.assert_array_equal(video.frame_times, rec.times)
    assert video.frame_times_basis.basis == "inferred"
    assert video.frame_times_basis in rec.conversions
    assert any(f"+{LEAD:.2f} s" in line for line in rec.watch_outs)
    assert video.calibration is None
    assert [line.split(":")[0] for line in video.calibration_tried] == list(
        CALIBRATION_SOURCES
    )
    with pytest.raises(InputError, match="no calibration"):
        rec.animation(video=True)


def test_session_without_one_video_file_has_none(synthetic, tmp_path):
    """A video split across two files, or none, gives rec.video None and says why."""
    from neurospatial.datasets import valero_2022

    entry, path, _ = synthetic
    split = tmp_path / "split.nwb"
    _write_session(split, video_files=("cam/a.avi", "cam/b.avi"))
    rec = valero_2022.load_session(entry, split)
    assert rec.video is None
    assert any("split across 2 files" in line for line in rec.watch_outs)
    with pytest.raises(InputError, match="has no video"):
        rec.animation(video=True)

    unlisted = dataclasses.replace(entry, videos=())
    with pytest.raises(ValueError, match="does not list beside it"):
        valero_2022.load_session(unlisted, path)


@pytest.mark.realdata
@OLD_DEVICE
def test_loads_default_session():
    """The survey's session: 35 CA1 units, 30 Hz, the 110 cm track along y."""
    rec = load("valero_2022")
    assert rec.session == DEFAULT
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == _entry().n_units == 35
    assert rec.sampling_rate == pytest.approx(30.0, rel=1e-6)
    assert rec.times.size == 67_131
    low, high = np.nanmin(rec.positions, axis=0), np.nanmax(rec.positions, axis=0)
    np.testing.assert_allclose(low, [4.1666, 0.5458], atol=1e-3)
    np.testing.assert_allclose(high, [20.6797, 116.4804], atol=1e-3)
    assert 100.0 <= np.ptp(rec.positions[:, 1]) <= 125.0
    assert rec.events["ripples"].shape == (49, 2)
    assert rec.provenance.sha256 == _entry().sha256
    assert rec.video.frame_times.size == 67_131
    np.testing.assert_array_equal(rec.video.frame_times, rec.times)


@pytest.mark.realdata
@OLD_DEVICE
def test_video_scene(tmp_path):
    """The default session's video fetches with its checksum; a 2 s export over it.

    The video is calibrated by the stored LED fit, and the watch-out says
    what that alignment cannot show. A frame sheet is written for the person
    who verifies the video.
    """
    cv2 = pytest.importorskip("cv2")
    rec = load("valero_2022")
    path = rec.video.fetch()
    assert path.name == Path(rec.video.external_file).name
    assert path.stat().st_size == rec.video.size_bytes
    from neurospatial._video_io import _VideoReader

    with _VideoReader(path) as reader:
        assert reader.n_frames == rec.video.frame_times.size
    assert rec.video.calibration is not None
    assert "detected in the video" in rec.video.calibration_basis.text
    assert rec.video.calibration_basis in rec.conversions
    assert any("aligned by construction" in line for line in rec.watch_outs)
    assert any("of LED detections are more than 5 cm off" in w for w in rec.watch_outs)
    assert any("mirrored relative to the raw video" in w for w in rec.watch_outs)
    # A run: the 2 s the head moves farthest in, among those without a
    # tracking jump (a step over 2.5 cm, 75 cm/s) or a held sample (a step
    # of 0), so the frames spread along the track and every one is tracked.
    window = round(2.0 * rec.sampling_rate)
    step = np.linalg.norm(np.diff(rec.positions, axis=0), axis=1)
    steps = np.lib.stride_tricks.sliding_window_view(step, window)
    moved = np.linalg.norm(rec.positions[window:] - rec.positions[:-window], axis=1)
    moved[(steps.max(axis=1) > 2.5) | (steps.min(axis=1) == 0.0)] = 0.0
    start = float(rec.times[int(np.argmax(moved))])
    scene = rec.animation(time_range=(start, start + 2.0), video=True)
    frames = tmp_path / "frames"
    scene.export(
        tmp_path / "scene.html", fps=10, dpi=100, embed=False, frames_dir=frames
    )
    exported = sorted(frames.glob("frame_*.png"))
    assert len(exported) == 20
    sheet = np.hstack([cv2.imread(str(p)) for p in exported[::5]])
    path = tmp_path / "frame_sheet.png"
    assert cv2.imwrite(str(path), sheet)  # for the person who verifies the video


def test_session_running_the_track_twice(synthetic, tmp_path):
    """Tracking in two epochs: both are 'track' rows, each with its untracked stretches."""
    from neurospatial.datasets import valero_2022

    entry, _, _ = synthetic
    path = tmp_path / "twice.nwb"
    times = _write_session(path, second_run=True)
    rec = run_checks(valero_2022.load_session(entry, path), stated=entry.info.stated)
    np.testing.assert_allclose(
        rec.epochs["track"], [[200.0, 800.0 - 1e-3], [900.0, 1000.0 - 1e-3]]
    )
    np.testing.assert_allclose(
        rec.epochs["track_before_tracking"], [[200.0, times[0]], [900.0, 905.0]]
    )
    assert rec.epochs["track_after_tracking"].shape == (2, 2)
    assert set(rec.epochs) >= {"epoch_1", "epoch_2", "epoch_4"}
    assert "epoch_5" not in rec.epochs  # it is the second track row
    epochs_line = next(c for c in rec.conversions if c.text.startswith("epochs:"))
    assert "epoch_1 = fCamk2_0" in epochs_line.text
