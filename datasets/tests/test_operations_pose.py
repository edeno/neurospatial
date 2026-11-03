from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest
from _record import Recording
from operations import read_pose_operation

pynwb = pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")

N_SAMPLES = 1000
NAN_SHARE = 0.3


@pytest.fixture
def pose_recording(tmp_path) -> Recording:
    """A small NWB file with two keypoints in cm; ``tail`` is NaN on 30 % of samples."""
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton

    rng = np.random.default_rng(0)
    times = np.arange(N_SAMPLES) / 30.0
    nose = 50.0 + np.cumsum(rng.normal(0, 0.5, (N_SAMPLES, 2)), axis=0)
    tail = nose - 5.0
    tail[rng.permutation(N_SAMPLES)[: int(NAN_SHARE * N_SAMPLES)]] = np.nan
    nwbfile = pynwb.NWBFile(
        session_description="pose test",
        identifier="pose-test",
        session_start_time=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
    )
    skeleton = Skeleton(
        name="skeleton",
        nodes=["nose", "tail"],
        edges=np.array([[0, 1]], dtype=np.uint8),
    )
    series = [
        PoseEstimationSeries(
            name=name,
            description=name,
            data=data,
            confidence=np.ones(N_SAMPLES),
            timestamps=times,
            reference_frame="arena corner",
            unit="cm",
        )
        for name, data in (("nose", nose), ("tail", tail))
    ]
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    behavior.add(
        PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=series,
            description="test",
            skeleton=skeleton,
        )
    )
    nwbfile.create_processing_module("Skeletons", "skeletons").add(skeleton)
    path = tmp_path / "pose.nwb"
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    return Recording(
        dataset="pose_test",
        session="0",
        source=str(path),
        license="n/a",
        times=times,
        positions=nose,
        position_units="cm",
        params={"bin_size": 2.0, "nwb_source": str(path), "pose_keypoint": "tail"},
    )


def test_read_pose_operation_reports_nan_share(pose_recording):
    outcome = read_pose_operation(pose_recording, {})
    assert outcome.status == "ok"
    assert outcome.summary["keypoints"] == "nose,tail"
    assert outcome.summary["nan_share_tail"] == pytest.approx(NAN_SHARE, abs=0.01)
    assert outcome.summary["nan_share_nose"] == 0.0
    # The NaN-heavy keypoint still feeds the occupancy API (no spikes here).
    assert outcome.summary["fed"] == "occupancy"
    assert outcome.summary["keypoint"] == "tail"


def test_read_pose_operation_feeds_rate_maps_with_spikes(pose_recording):
    rng = np.random.default_rng(1)
    duration = (N_SAMPLES - 1) / 30.0
    spikes = tuple(np.sort(rng.uniform(0.0, duration, 200)) for _ in range(2))
    rec = replace(pose_recording, spike_times=spikes, unit_ids=np.arange(2))
    outcome = read_pose_operation(rec, {})
    assert outcome.status == "ok"
    assert outcome.summary["fed"] == "compute_spatial_rates"
    assert outcome.summary["nan_share_tail"] == pytest.approx(NAN_SHARE, abs=0.01)


def test_read_pose_operation_is_na_without_pose(tmp_path):
    path = tmp_path / "empty.nwb"
    nwbfile = pynwb.NWBFile(
        session_description="no pose",
        identifier="no-pose",
        session_start_time=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
    )
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    rec = Recording(
        dataset="no_pose",
        session="0",
        source=str(path),
        license="n/a",
        params={"nwb_source": str(path)},
    )
    outcome = read_pose_operation(rec, {})
    assert outcome.status == "n/a"
