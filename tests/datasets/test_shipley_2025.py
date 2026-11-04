"""Shipley et al. (2026), UCL RDR 30902972: synthetic members in the record's layout, and the real day."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._checks import run_checks

DEFAULT = "m4005_2020-09-26"
CAMERA_HZ = 30.05
PPM = 274.0
CENTRE = np.array([105.0, 95.0])
#: Arm directions, 45 degrees apart from -150 degrees.
ANGLES = np.radians(-150.0 + 45.0 * np.arange(8))


def _star_path(n_per_arm=400, seed=0, skip_arm=None):
    """Runs out and back along each arm of the stated maze, cm, with 1 cm noise."""
    rng = np.random.default_rng(seed)
    reach = 24.7 / 2 + 80.0
    runs = []
    for k, angle in enumerate(ANGLES):
        if k == skip_arm:
            continue
        out = np.linspace(0.0, reach - 2.0, n_per_arm // 2)
        along = np.r_[out, out[::-1]]
        runs.append(CENTRE + along[:, None] * [np.cos(angle), np.sin(angle)])
    path = np.vstack(runs)
    return path + rng.normal(0.0, 1.0, path.shape)


def _write_members(folder, *, trial_names=("0", "1", "S2"), spikes_in=None, gap=0.05):
    """The four members for a day of three trials at 30.05 Hz.

    Trials follow each other on the concatenated clock ``gap`` s apart. Each
    spike is mapped to its nearest frame, as the authors' nearest-neighbour
    search does. ``spikes_in`` limits the spikes to those trials.
    """
    from scipy.io import savemat

    rng = np.random.default_rng(1)
    path = _star_path()
    n_trial = len(path)
    xy, times, ends, start = [], [], [], 0.0
    for _ in trial_names:
        xy.append(path)
        t = start + np.arange(n_trial) / CAMERA_HZ
        times.append(t)
        ends.append(n_trial * (len(ends) + 1))
        start = t[-1] + gap
    xy, times = np.vstack(xy), np.concatenate(times)
    clusters = np.array([2, 4, 7])  # 4 and 7 are good
    spike_time = np.sort(rng.uniform(times[0], times[-1], 6000))
    labels = rng.choice(clusters, spike_time.size)
    if spikes_in is not None:
        keep = np.zeros(spike_time.size, dtype=bool)
        for k in spikes_in:
            first = 0 if k == 0 else ends[k - 1]
            keep |= (spike_time >= times[first]) & (spike_time <= times[ends[k] - 1])
        spike_time, labels = spike_time[keep], labels[keep]
    samples = np.round(spike_time * 30_000.0).astype(np.uint64)
    np.save(folder / "spike_times.npy", samples[:, None])
    np.save(folder / "spike_clusters.npy", labels.astype(np.int32))
    (folder / "cluster_info.tsv").write_text(
        "id\tAmplitude\tContamPct\tKSLabel\tamp\tch\tdepth\tfr\tgroup\tn_spikes\tsh\n"
        "2\t500.0\t80.0\tmua\t30.0\t3\t1.0\t9.0\tnoise\t10\t0\n"
        "4\t1300.0\t\tgood\t44.0\t4\t2.0\t1.3\tgood\t10\t0\n"
        "7\t1700.0\t15.2\tmua\t45.0\t5\t2.0\t0.5\tgood \t10\t0\n",
        encoding="utf-8",
    )
    good = np.isin(labels, [4, 7])
    good_times = samples[good].astype(float) / 30_000.0
    frame = np.searchsorted(times, good_times)
    frame = np.clip(frame, 1, times.size - 1)
    nearer = np.abs(times[frame - 1] - good_times) < np.abs(times[frame] - good_times)
    frame = frame - nearer
    tetrode = np.empty((1, 2), dtype=object)
    for k in range(2):
        mine = np.arange(good_times.size) % 2 == k
        tetrode[0, k] = {
            "timestamp": good_times[mine],
            "pos_sample": (frame[mine] + 1).astype(np.int32),
            "cut": labels[good][mine].astype(np.int32),
            "id": k + 1,
        }
    header = np.array(
        [["sample_rate", "30.0 hz"], ["pixels_per_metre", "274"]], dtype=object
    )
    pixels = xy * PPM / 100.0
    savemat(
        folder / "trialData_newMUA.mat",
        {
            "trialData": {
                "tetrode": tetrode,
                "pos": {
                    "xy": pixels,
                    "dir": np.full(len(xy), 270.0),
                    "dir_disp": np.full(len(xy), 270.0),
                    "n_leds": 1,
                    "header": header,
                    "trialFinalInd": np.array(ends, dtype=np.int32),
                    "trialNames": np.array(list(trial_names), dtype=object),
                },
            }
        },
    )
    return times, xy, spike_time, labels


@pytest.fixture
def members(tmp_path):
    """A synthetic day: three trials ('0', '1', and rest 'S2')."""
    return tmp_path, *_write_members(tmp_path)


def test_loads_a_day(members, session_entry):
    from neurospatial.datasets import shipley_2025

    folder, times, xy, spike_time, labels = members
    entry = session_entry("shipley_2025")
    rec = run_checks(shipley_2025.load_session(entry, folder), stated=entry.info.stated)
    # Spikes: the good clusters only, samples / 30 kHz.
    assert list(rec.units.index) == [4, 7]
    for unit, train in zip((4, 7), rec.spike_times, strict=True):
        expected = np.round(spike_time[labels == unit] * 30_000.0) / 30_000.0
        np.testing.assert_allclose(train, np.sort(expected))
    # An empty Phy cell stays missing.
    assert np.isnan(rec.units.loc[4, "ContamPct"])
    assert rec.units.loc[7, "ContamPct"] == pytest.approx(15.2)
    # Positions: pixels / 274 pixels per metre, in cm.
    np.testing.assert_allclose(rec.positions, xy, atol=1e-9)
    # Times: each trial's line recovers the 30.05 Hz camera to within a frame's
    # rounding of the nearest-frame map.
    np.testing.assert_allclose(rec.times, times, atol=2e-3)
    assert rec.sampling_rate == pytest.approx(CAMERA_HZ, rel=1e-3)
    clock = next(c for c in rec.conversions if c.text.startswith("times:"))
    assert clock.basis == "inferred"
    rates = [
        float(r)
        for r in clock.evidence.split("fitted rates ")[1].split(" Hz")[0].split(" to ")
    ]
    assert rates == pytest.approx([CAMERA_HZ, CAMERA_HZ], abs=0.01)
    # Epochs and trials.
    assert set(rec.epochs) == {"exploration", "trials", "maze", "rest"}
    assert rec.epochs["maze"].shape == (2, 2)
    assert rec.epochs["rest"].shape == (1, 2)
    assert list(rec.trials.intervals_table()["trial"]) == ["0", "1"]
    np.testing.assert_allclose(rec.params["analysis_epochs"], rec.epochs["maze"])
    assert rec.headings is None
    text = " ".join(rec.watch_outs)
    assert "Position times are rebuilt per trial" in text
    assert "pos.dir equals dir_disp in all" in text
    assert "Genotype: m4005 is App NL-G-F/ChAT-cre" in text


def test_star_is_placed_on_the_arms(members, session_entry):
    """The star's centre and arm directions come back from the samples."""
    from neurospatial.datasets.shipley_2025 import star_graph

    graph, evidence = star_graph(_star_path())
    np.testing.assert_allclose(graph.nodes["centre"]["pos"], CENTRE, atol=0.5)
    for k, angle in enumerate(ANGLES):
        end = np.asarray(graph.nodes[f"arm_{k}"]["pos"])
        direction = (end - CENTRE) / np.linalg.norm(end - CENTRE)
        assert abs(np.angle(np.exp(1j * (np.arctan2(*direction[::-1]) - angle)))) < 0.02
        assert graph.edges["centre", f"platform_{k}"]["distance"] == pytest.approx(
            12.35
        )
        assert graph.edges[f"platform_{k}", f"arm_{k}"]["distance"] == 80.0
    assert "8 of them the principal axes" in evidence


def test_an_unvisited_arm_keeps_the_octagons_direction():
    from neurospatial.datasets.shipley_2025 import star_graph

    graph, evidence = star_graph(_star_path(skip_arm=3))
    assert "arms [3] have fewer than 100 samples" in evidence
    end = np.asarray(graph.nodes["arm_3"]["pos"]) - graph.nodes["centre"]["pos"]
    assert abs(np.angle(np.exp(1j * (np.arctan2(end[1], end[0]) - ANGLES[3])))) < 0.05


def test_good_clusters_must_match_the_curated_spikes(members, session_entry):
    from neurospatial.datasets import shipley_2025

    folder = members[0]
    text = (folder / "cluster_info.tsv").read_text(encoding="utf-8")
    (folder / "cluster_info.tsv").write_text(
        text.replace("\tnoise\t", "\tgood\t"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differ from the"):
        shipley_2025.load_session(session_entry("shipley_2025"), folder)


def test_a_trial_without_spikes_cannot_be_timed(tmp_path, session_entry):
    from neurospatial.datasets import shipley_2025

    _write_members(tmp_path, spikes_in=(1, 2))
    with pytest.raises(
        ValueError,
        match=r"trial '0' \(position samples 1 to 3200\) has 0 spikes mapped",
    ):
        shipley_2025.load_session(session_entry("shipley_2025"), tmp_path)


def test_trials_that_overlap_on_the_clock_are_refused(tmp_path, session_entry):
    """Trial 1 starting 0.5 s before trial 0 ends: no one clock holds both."""
    from neurospatial.datasets import shipley_2025

    _write_members(tmp_path, gap=-0.5)
    with pytest.raises(ValueError, match="do not increase") as raised:
        shipley_2025.load_session(session_entry("shipley_2025"), tmp_path)
    overlap = float(str(raised.value).split("starts ")[1].split(" ms")[0])
    assert overlap == pytest.approx(500.0, abs=2.0)
    assert "where trial '1' starts" in str(raised.value)
    assert "before trial '0' ends" in str(raised.value)


def test_position_times_report_the_header_rate_when_the_map_used_it():
    from neurospatial.datasets.shipley_2025 import position_times

    index = np.arange(1, 3001)
    times, clock = position_times(
        np.repeat(index, 3), np.repeat((index - 1) / 30.0, 3), np.array([3000]), ["0"]
    )
    np.testing.assert_allclose(times, np.arange(3000) / 30.0, atol=1e-9)
    assert "Every fitted rate is the header's 30 Hz" in clock.watch_out


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """m4005 on 2020-09-26, as the survey found it."""
    rec = load("shipley_2025")
    assert_default_session(rec, session_entry("shipley_2025"))
    assert sum(t.size for t in rec.spike_times) == 1_201_058
    assert rec.times.size == 146_101
    assert rec.epochs["maze"].shape == (11, 2)
    assert rec.epochs["rest"].shape == (1, 2)
    assert len(rec.trials) == 11
    assert rec.sampling_rate == pytest.approx(30.05, abs=0.01)
    np.testing.assert_allclose(np.ptp(rec.positions, axis=0), [175.0, 174.1], atol=0.1)
    clock = next(c for c in rec.conversions if c.text.startswith("times:"))
    assert "fitted rates 30.047 to 30.064 Hz" in clock.evidence
    assert "largest residual is 41 ms" in clock.evidence
    assert rec.env.is_linearized_track
    assert rec.env.units == "cm"


@pytest.mark.realdata
def test_a_day_whose_trials_overlap_is_refused():
    """m4005 on 2020-09-28: trial 8's line starts 0.53 ms before trial 7's ends."""
    with pytest.raises(ValueError, match=r"trial '8' starts 0\.53 ms before trial '7'"):
        load("shipley_2025", "m4005_2020-09-28")
