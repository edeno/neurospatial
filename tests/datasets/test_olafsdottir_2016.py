"""Ólafsdóttir et al. (2016), Zenodo 5566548: a synthetic day in Axona files, and the real one."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from neurospatial.datasets import olafsdottir_2016
from neurospatial.datasets._axona import DATA_END, microvolts_per_unit
from neurospatial.datasets._catalog import Member, sessions
from neurospatial.datasets._checks import run_checks
from neurospatial.datasets._lfp import LFPFile

SESSION = "R2336-2015-11-01"
TRACK, REST = "20151101_R2336_track1", "20151101_R2336_sleepPOST"
PPM = 300.0
#: The synthetic Z's ends and corners in the camera's frame, cm: rotated 20 degrees.
ANGLE = np.radians(20.0)


def _z_nodes() -> np.ndarray:
    rise = np.sqrt(220.0**2 - 190.0**2)
    local = np.array([[0.0, rise], [190.0, rise], [0.0, 0.0], [190.0, 0.0]])
    rotation = np.array(
        [[np.cos(ANGLE), -np.sin(ANGLE)], [np.sin(ANGLE), np.cos(ANGLE)]]
    )
    return local @ rotation.T + np.array([60.0, 30.0])


def _file(*lines: str, data: bytes) -> bytes:
    head = "".join(f"{line}\r\n" for line in lines).encode("latin-1")
    return head + b"data_start" + data + DATA_END


def _common(stem: str, start: str, duration: int) -> tuple[str, ...]:
    return (
        "trial_date Sunday, 1 Nov 2015",
        f"trial_time {start}",
        f"comments {stem}",
        f"duration {duration}",
    )


def _write_set(path, start: str, duration: int, gains: dict[int, int]) -> None:
    lines = [*_common(path.stem, start, duration), "ADC_fullscale_mv 1500"]
    lines += [f"gain_ch_{ch} {gains.get(ch, 10000)}" for ch in range(64)]
    lines += ["lightBearing_1 0", "EEG_ch_1 33", "saveEEG_ch_1 1", "EEG_ch_2 0"]
    path.write_text("\r\n".join(lines) + "\r\n", encoding="latin-1")


def _write_pos(path, start: str, duration: int, pixels: np.ndarray) -> None:
    """``pixels``: (n, 4) big x, y, little x, y, 1023 where not found."""
    words = np.zeros((len(pixels), 8), dtype=np.int64)
    words[:, :4] = pixels
    data = b"".join(
        int(k).to_bytes(4, "big") + row.astype(">u2").tobytes()
        for k, row in enumerate(words)
    )
    path.write_bytes(
        _file(
            *_common(path.stem, start, duration),
            "timebase 50 hz",
            "bytes_per_timestamp 4",
            "sample_rate 50.0 hz",
            "pos_format t,x1,y1,x2,y2,numpix1,numpix2",
            "bytes_per_coord 2",
            f"pixels_per_metre {PPM:g}",
            f"num_pos_samples {len(pixels)}",
            data=data,
        )
    )


def _write_tetrode(path, start: str, duration: int, stamps, waves) -> None:
    data = b"".join(
        b"".join(
            int(stamp).to_bytes(4, "big", signed=True) + wave[k].astype("i1").tobytes()
            for k in range(4)
        )
        for stamp, wave in zip(stamps, waves, strict=True)
    )
    path.write_bytes(
        _file(
            *_common(path.stem, start, duration),
            "num_chans 4",
            "timebase 96000 hz",
            "bytes_per_timestamp 4",
            "samples_per_spike 50",
            "sample_rate 48000 hz",
            "bytes_per_sample 1",
            "spike_format t,ch1,t,ch2,t,ch3,t,ch4",
            f"num_spikes {len(stamps)}",
            data=data,
        )
    )


def _write_cut(path, trial: str, clusters) -> None:
    body = "  ".join(str(c) for c in clusters)
    path.write_text(
        f"n_clusters: 3\r\nn_channels: 4\r\n\r\nExact_cut_for: {trial} spikes: "
        f"{len(clusters)}\r\n  {body}",
        encoding="latin-1",
    )


def _write_egf(path, start: str, duration: int, values: np.ndarray) -> None:
    path.write_bytes(
        _file(
            *_common(path.stem, start, duration),
            "num_chans 1",
            "sample_rate 4800 Hz",
            "bytes_per_sample 2",
            f"num_EGF_samples {values.size}",
            data=values.astype("<i2").tobytes(),
        )
    )


def _laps(times: np.ndarray, nodes: np.ndarray, period: float) -> np.ndarray:
    """Positions shuttling end to end along the Z, one way per ``period`` s."""
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    total = lengths.sum()
    phase = (times % (2 * period)) / period
    along = np.where(phase < 1, phase, 2 - phase) * total
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    segment = np.clip(np.searchsorted(edges, along, side="right") - 1, 0, 2)
    t = (along - edges[segment]) / lengths[segment]
    return nodes[segment] + t[:, None] * (nodes[segment + 1] - nodes[segment])


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    """A track trial of 60 s and a rest of 30 s two minutes after its start."""
    folder = tmp_path_factory.mktemp("olafsdottir_2016") / SESSION
    folder.mkdir()
    rng = np.random.default_rng(7)
    gains = {0: 12000, 1: 12000, 2: 6000, 3: 6000, 32: 9000, 33: 9000}
    for stem, start, duration in ((TRACK, "10:00:00", 60), (REST, "10:02:00", 30)):
        _write_set(folder / f"{stem}.set", start, duration, gains)
    times = np.arange(1, 3001) / 50.0
    cm = _laps(times, _z_nodes(), period=6.0)
    big = np.round(cm * PPM / 100.0)
    little = big - np.array([6.0, 0.0])
    pixels = np.column_stack([big, little])
    pixels[10] = [1023, 1023, 1023, 1023]
    pixels[20, 2:] = 1023
    _write_pos(folder / f"{TRACK}.pos", "10:00:00", 60, pixels)
    rest = np.full((1500, 4), 1023)
    rest[:, :2] = [40, 50]
    _write_pos(folder / f"{REST}.pos", "10:02:00", 30, rest)
    spikes = {}
    for stem, duration in ((TRACK, 60), (REST, 30)):
        for tetrode in (1, 9):
            stamps = np.sort(rng.integers(0, duration * 96000, size=40))
            waves = rng.integers(-60, 100, size=(40, 4, 50))
            clusters = rng.integers(0, 3, size=40)
            _write_tetrode(folder / f"{stem}.{tetrode}", "", duration, stamps, waves)
            if not (stem == REST and tetrode == 1):
                _write_cut(folder / f"{stem}_{tetrode}.cut", stem, clusters)
            spikes[stem, tetrode] = (stamps, waves, clusters)
    egf = {TRACK: np.arange(60 * 4800) % 2000 - 1000, REST: np.full(30 * 4800, 7)}
    for stem, values in egf.items():
        _write_egf(folder / f"{stem}.egf", "", len(values) // 4800, values)
    default = next(e for e in sessions("olafsdottir_2016") if e.default)
    members = tuple(
        Member(path=f"r2336/2015-11-01/{p.name}", sha256="0" * 64, size_bytes=1)
        for p in sorted(folder.iterdir())
        if not p.name.endswith(".egf")
    )
    lfp = tuple(
        LFPFile(
            path=f"{SESSION}/{stem}.egf",
            sha256=hashlib.sha256((folder / f"{stem}.egf").read_bytes()).hexdigest(),
            size_bytes=(folder / f"{stem}.egf").stat().st_size,
            url=default.url,
            member=f"r2336/2015-11-01/{stem}.egf",
        )
        for stem in (TRACK, REST)
    )
    entry = dataclasses.replace(
        default, session=SESSION, verified=False, members=members, lfp_files=lfp
    )
    rec = olafsdottir_2016.load_session(entry, folder)
    return rec, entry, folder, spikes, gains, cm, egf


def test_positions_are_the_big_led_in_cm_and_1023_is_nan(day):
    rec, _, _, _, _, cm, _ = day
    assert rec.position_units == "cm"
    np.testing.assert_array_equal(rec.times[:3], [0.02, 0.04, 0.06])
    expected = np.round(cm * PPM / 100.0) * 100.0 / PPM
    expected[10] = np.nan
    np.testing.assert_allclose(rec.positions, expected)


def test_no_headings_and_the_little_led_is_kept(day):
    rec, _, _, _, _, cm, _ = day
    assert rec.headings is None
    little = np.round(cm * PPM / 100.0) - np.array([6.0, 0.0])
    expected = little * 100.0 / PPM
    expected[[10, 20]] = np.nan
    np.testing.assert_allclose(rec.params["little_led_positions"], expected)
    # Where both LEDs are found, little-to-big points along +x, which the
    # synthetic path follows only on part of the Z: a watch-out gives the offset.
    assert any(text.startswith("No headings") for text in rec.watch_outs)


def test_spike_times_are_on_the_days_clock_and_units_join_the_trials(day):
    rec, _, _, spikes, _, _, _ = day
    stamps_track, _, ids_track = spikes[TRACK, 9]
    stamps_rest, _, ids_rest = spikes[REST, 9]
    train = rec.spike_times[list(rec.units.index).index("T9_c2")]
    expected = np.concatenate(
        [
            stamps_track[ids_track == 2] / 96000,
            120.0 + stamps_rest[ids_rest == 2] / 96000,
        ]
    )
    np.testing.assert_allclose(train, expected)
    row = rec.units.loc["T9_c2"]
    assert row["brain_region"] == "CA1"
    assert (row["n_spikes_track"], row["n_spikes_rest"]) == (
        np.sum(ids_track == 2),
        np.sum(ids_rest == 2),
    )
    assert rec.units.loc["T1_c1", "brain_region"] == "MEC"
    # Cluster 0 is Tint's unassigned spikes, never a unit.
    assert not any(name.endswith("_c0") for name in rec.units.index)
    # Tetrode 1 is unsorted in the rest: none of its units fire there.
    assert (rec.units.loc[["T1_c1", "T1_c2"], "n_spikes_rest"] == 0).all()


def test_epochs_place_the_rest_by_its_start_time(day):
    rec = day[0]
    np.testing.assert_array_equal(rec.epochs["track"], [[0.0, 60.0]])
    np.testing.assert_array_equal(rec.epochs["rest"], [[120.0, 150.0]])
    np.testing.assert_allclose(rec.params["rest_times"][[0, -1]], [120.02, 150.0])
    np.testing.assert_allclose(rec.params["rest_positions"][0], [40 / 3, 50 / 3])


def test_marks_hold_every_spike_with_its_peaks_in_microvolts(day):
    rec, _, _, spikes, _, _, _ = day
    k = rec.marks.electrodes.index("T1")
    stamps, waves, ids = spikes[TRACK, 1]
    _, waves_rest, _ = spikes[REST, 1]
    scale = 1000 * 1500 / (np.array([12000, 12000, 6000, 6000]) * 128)
    np.testing.assert_allclose(
        rec.marks.amplitudes[k][:40], waves.max(axis=2) * scale, rtol=1e-6
    )
    np.testing.assert_allclose(rec.marks.times[k][:40], stamps / 96000)
    # The rest's tetrode 1 has no cut: its spikes are marked unsorted.
    np.testing.assert_array_equal(rec.marks.clusters[k], np.r_[ids, np.full(40, -1)])
    assert rec.marks.n_spikes == 160
    full = rec.marks.waveforms("T1")
    np.testing.assert_allclose(full[40:], waves_rest * scale[None, :, None], rtol=1e-6)
    assert any("tetrodes [1]" in text for text in rec.watch_outs)


def test_marks_select_keeps_the_spikes_of_the_intervals(day):
    rec = day[0]
    times, amplitudes = rec.marks.select("T9", rec.epochs["rest"])
    assert times.size == 40
    assert amplitudes.shape == (40, 4)
    assert times.min() >= 120.0


def test_the_fitted_z_is_the_stated_one_from_the_start_end(day):
    rec = day[0]
    nodes = rec.params["track_nodes"]
    truth = _z_nodes()
    # Corners within a cm (pixels are 1/3 cm); ends where 99.5% of a runway's samples
    # stop, so up to 0.5% of a 190 cm runway, about 1 cm, short, and a pixel.
    np.testing.assert_allclose(nodes[1:3], truth[1:3], atol=1.0)
    np.testing.assert_allclose(nodes[[0, 3]], truth[[0, 3]], atol=2.5)
    assert rec.params["track_length"] == pytest.approx(600.0, abs=8.0)
    assert rec.env.is_linearized_track
    assert rec.env_kind == "inferred"


def test_runs_go_end_to_end_with_their_direction(day):
    rec = day[0]
    # 60 s of 6 s one-way runs: 10 runs, the first leaving the start end.
    assert len(rec.trials) == 10
    assert rec.trials.direction.tolist()[:2] == ["outbound", "inbound"]
    assert len(rec.epochs["outbound"]) == len(rec.epochs["inbound"]) == 5


def test_checks_pass_on_the_synthetic_day(day):
    rec, entry, *_ = day
    checked = run_checks(rec, stated=entry.info.stated)
    assert checked.failed_checks == ()


def test_lfp_reader_gives_microvolts_on_the_days_clock(day):
    rec, _, folder, _, gains, _, egf = day
    assert set(rec.lfp.channels) == {"T9.1"}
    sources = tuple(folder / f"{stem}.egf" for stem in (TRACK, REST))
    samples, times = rec.lfp.reader(sources, ("T9.1",))
    scale = 1000 * 1500 / (gains[32] * 32768)
    np.testing.assert_allclose(samples[: 4800 * 60, 0], egf[TRACK] * scale)
    np.testing.assert_allclose(samples[4800 * 60 :, 0], 7 * scale)
    assert times[0] == pytest.approx(1 / 4800)
    assert times[4800 * 60] == pytest.approx(120.0 + 1 / 4800)
    assert (
        microvolts_per_unit({"ADC_fullscale_mv": "1500", "gain_ch_32": "9000"}, 32, 2)
        == scale
    )


def test_run_intervals_skip_a_return_to_the_same_end():
    times = np.arange(8.0)
    linear = np.array([1.0, 50.0, 4.0, 60.0, 99.0, 50.0, 97.0, np.nan])
    rows, direction = olafsdottir_2016.run_intervals(times, linear, 100.0)
    np.testing.assert_array_equal(rows, [[2.0, 4.0]])
    assert direction.tolist() == ["outbound"]


def test_session_members_keep_what_a_load_reads():
    names = [
        "r2217/2014-12-13/20141213_R2217_track1.4",
        "r2217/2014-12-13/20141213_R2217_track1_4.cut",
        "r2217/2014-12-13/20141213_R2217_track1.clu.4",
        "r2217/2014-12-13/20141213_R2217_track1.pos",
        "r2217/2014-12-13/20141213_R2217_track1.egf3",
        "r2217/2014-12-13/20141213_R2217_track1.eeg",
        "r2217/2014-12-13/20141213_R2117_screening.set",
        "r2217/2014-12-13/20141218_R2117_screening_cell_info.mat",
        "r2142/2014-08-06/20140806_R2142_sleepPOST.cut",
        "r2337/2015-11-27/20151127_R2337_screening_6.ut",
        "README.txt",
    ]
    days = olafsdottir_2016.session_members(names)
    assert list(days) == ["R2142-2014-08-06", "R2217-2014-12-13"]
    load, lfp = days["R2217-2014-12-13"]
    assert [n.split("/")[-1] for n in load] == [
        "20141213_R2117_screening.set",
        "20141213_R2217_track1.4",
        "20141213_R2217_track1.pos",
        "20141213_R2217_track1_4.cut",
    ]
    assert [n.split("/")[-1] for n in lfp] == ["20141213_R2217_track1.egf3"]
    assert days["R2142-2014-08-06"] == ([], [])


# --------------------------------------------------------------------------
# The real default session
# --------------------------------------------------------------------------

#: The worked result's criterion, fixed in writing before any analysis: the
#: share of candidates judged replay lies within 10 points of the paper's
#: pooled 1,826 of 4,382.
PAPER_SHARE = 1826 / 4382
SHARE_BAND = (PAPER_SHARE - 0.10, PAPER_SHARE + 0.10)
SEED = 2026
#: The paper's line fit, as its Supplementary Software runs it: a band of 15
#: bins (30 cm at 2 cm) either side of the line.
CODE_BAND_BINS = 15
#: Events of the code-against-preset comparison: the first N_COMPARED, in time
#: order, of N_DRAWN drawn with SEED (the docs page gives all N_DRAWN).
N_DRAWN, N_COMPARED = 150, 30
#: On those events, per variant of the code: replay by both, by the code alone,
#: by the preset alone. On all 150: code 91, preset 93, both 88.
CODE_COMPARISON = {
    "code": (20, 1, 1),
    "speed_200": (20, 1, 1),
    "every_line": (20, 1, 1),
    "median_off_track": (21, 1, 0),
    "mean_over_bins": (20, 1, 1),
}


@pytest.fixture(scope="module")
def rec():
    from neurospatial.datasets import load

    return load("olafsdottir_2016")


def _rate_map_positions(rec):
    """Positions with the paper's excluded ends and corners NaN."""
    params = rec.params
    nodes = params["track_nodes"]
    linear, _ = olafsdottir_2016._linear(rec.positions, nodes)
    corners = np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))[:-1]
    keep = (linear > params["end_exclusion"]) & (
        linear < params["track_length"] - params["end_exclusion"]
    )
    for corner in corners:
        keep &= np.abs(linear - corner) > params["corner_exclusion"]
    positions = rec.positions.copy()
    positions[~keep] = np.nan
    return positions


def _rate_maps(rec, unit_ids):
    """Outbound and inbound rate maps of ``unit_ids``, as the paper builds them."""
    from neurospatial import compute_spatial_rates

    names = list(rec.units.index)
    trains = [rec.spike_times[names.index(unit)] for unit in unit_ids]
    positions = _rate_map_positions(rec)
    return {
        direction: compute_spatial_rates(
            rec.env,
            trains,
            rec.times,
            positions,
            method="binned",
            bandwidth=rec.params["bandwidth"],
            min_speed=rec.params["min_speed"],
            epochs=rec.epochs[direction],
            unit_ids=np.asarray(unit_ids),
        )
        for direction in ("outbound", "inbound")
    }


def _longest_run(above):
    best = run = 0
    for flag in above:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _place_cells(rec):
    """CA1 units with a field of 20 cm or more above their mean rate, peak above 1 Hz.

    The paper's rule; the field as bins above the cell's mean rate on the track
    is the 2017 paper's. Our choice, fixed before the run: a mean track rate
    above 5 Hz is an interneuron (the paper: "narrow waveforms and high firing
    rates", no numbers).
    """
    ca1 = [u for u in rec.units.index if rec.units.loc[u, "brain_region"] == "CA1"]
    track = rec.epochs["track"][0]
    mean = rec.units.loc[ca1, "n_spikes_track"].to_numpy() / (track[1] - track[0])
    maps = _rate_maps(rec, ca1)
    order = np.argsort(rec.env.to_linear(rec.env.bin_centers))
    min_bins = int(np.ceil(20.0 / rec.params["bin_size"]))
    place = []
    for k, unit in enumerate(ca1):
        if mean[k] > 5.0:
            continue
        for result in maps.values():
            rate = np.asarray(result.firing_rates[k])[order]
            above = np.nan_to_num(rate, nan=-np.inf) > mean[k]
            if np.nanmax(rate) > 1.0 and _longest_run(above) >= min_bins:
                place.append(unit)
                break
    return ca1, place


@pytest.fixture(scope="module")
def place(rec):
    return _place_cells(rec)


@pytest.fixture(scope="module")
def candidates(rec, place):
    """``literature_methods.olafsdottir_2016`` on the rest, at 1 kHz."""
    pytest.importorskip("ripple_detection.literature_methods")
    from ripple_detection.literature_methods import Recording, run_method

    from neurospatial.datasets._ripple import spike_counts

    ca1, cells = place
    names = list(rec.units.index)
    rest = rec.epochs["rest"][0]
    grid = rest[0] + np.arange(int((rest[1] - rest[0]) * 1000)) / 1000.0
    counts = spike_counts(
        [rec.spike_times[names.index(u)] for u in ca1], grid, sampling_rate=1000.0
    )
    signals = Recording.from_arrays(
        grid, 1000.0, multiunit=counts, place_cells=np.isin(ca1, cells)
    )
    table = run_method("olafsdottir_2016", signals)
    return table[["start_time", "end_time"]].to_numpy()


def _event_inputs(rec, cells, start, stop):
    from neurospatial.datasets._ripple import event_rows

    names = list(rec.units.index)
    trains = [rec.spike_times[names.index(u)] for u in cells]
    dt = rec.params["replay_dt"]
    return event_rows(trains, start, stop, time_bin=dt, step=dt)


@pytest.fixture(scope="module")
def judged(rec, place, candidates):
    """Each candidate's OLAFSDOTTIR_2016 verdict, p-value and template."""
    from neurospatial.decoding import decode_position
    from neurospatial.replay import detect_replay
    from neurospatial.replay.methods import OLAFSDOTTIR_2016

    _, cells = place
    maps = _rate_maps(rec, cells)
    dt = rec.params["replay_dt"]
    rows = []
    for k, (start, stop) in enumerate(candidates):
        counts, centres = _event_inputs(rec, cells, start, stop)
        posteriors = {
            name: decode_position(
                rec.env, counts, model, dt=dt, times=centres
            ).posterior
            for name, model in maps.items()
        }
        result = detect_replay(
            rec.env,
            posteriors,
            centres,
            replay_method=OLAFSDOTTIR_2016,
            spike_counts=counts,
            encoding_models=maps,
            dt=dt,
            rng=SEED + k,
        )
        rows.append((result.is_replay, result.p_value, result.template))
    return rows


@pytest.mark.realdata
def test_default_session(rec, assert_default_session):
    entry = next(e for e in sessions("olafsdottir_2016") if e.default)
    assert_default_session(rec, entry)
    assert rec.n_units == 59
    assert int(np.sum(rec.units["brain_region"] == "CA1")) == 56
    assert rec.sampling_rate == pytest.approx(50.0)
    np.testing.assert_array_equal(rec.epochs["track"], [[0.0, 1663.0]])
    np.testing.assert_array_equal(rec.epochs["rest"], [[1822.0, 8306.0]])
    assert (len(rec.epochs["outbound"]), len(rec.epochs["inbound"])) == (34, 34)
    assert rec.params["track_length"] == pytest.approx(674.3, abs=0.1)
    assert rec.marks.n_spikes == 2_585_815
    assert sum(int(np.sum(c == -1)) for c in rec.marks.clusters) == 531_375
    assert set(rec.lfp.channels) == {"T7.2", "T9.1", "T14.1"}


@pytest.mark.realdata
def test_a_units_spikes_are_its_clusters_marks(rec):
    k = rec.marks.electrodes.index("T10")
    for cluster in (1, 2, 4):
        train = rec.spike_times[list(rec.units.index).index(f"T10_c{cluster}")]
        np.testing.assert_array_equal(
            train, rec.marks.times[k][rec.marks.clusters[k] == cluster]
        )


@pytest.mark.realdata
def test_waveforms_peak_where_the_marks_say(rec):
    k = rec.marks.electrodes.index("T10")
    waves = rec.marks.waveforms("T10")
    assert waves.shape == (rec.marks.times[k].size, 4, 50)
    np.testing.assert_allclose(waves.max(axis=2), rec.marks.amplitudes[k], rtol=1e-6)


@pytest.mark.realdata
def test_ripple_power_rises_in_the_candidates(rec, candidates):
    """One clock for spikes and LFP: CA1 ripple power is 10.6 times higher in events."""
    from scipy.signal import butter, sosfiltfilt

    samples, times = rec.lfp.fetch(("T9.1",))
    power = (
        sosfiltfilt(butter(4, [150, 250], "band", fs=4800, output="sos"), samples[:, 0])
        ** 2
    )
    inside = np.zeros(times.size, dtype=bool)
    for start, stop in candidates:
        first, last = np.searchsorted(times, [start, stop])
        inside[first:last] = True
    rest = (times >= rec.epochs["rest"][0, 0]) & (times < rec.epochs["rest"][0, 1])
    ratio = power[inside].mean() / power[rest & ~inside].mean()
    assert ratio == pytest.approx(10.58, abs=0.01)


@pytest.mark.realdata
def test_place_cells_and_candidates(place, candidates):
    ca1, cells = place
    assert (len(ca1), len(cells)) == (56, 52)
    assert len(candidates) == 1118


@pytest.mark.realdata
def test_worked_result_share_of_candidates_judged_replay(judged):
    """The criterion fixed before the run is not met: 60.9% against 31.7 to 51.7%."""
    replay = np.array([row[0] for row in judged])
    share = replay.mean()
    assert (int(replay.sum()), len(replay)) == (681, 1118)
    assert share == pytest.approx(
        olafsdottir_2016.WORKED_RESULT["attempted"]["value"], abs=5e-4
    )
    assert not SHARE_BAND[0] <= share <= SHARE_BAND[1]


# --------------------------------------------------------------------------
# The paper's Supplementary Software, ported for comparison only
# --------------------------------------------------------------------------

_EPS8 = np.finfo(float).eps ** 8


def _code_posterior(rates, counts, dt):
    """decode_calcPosterior.m: Poisson, rates + eps^8, empty time bins 0; (n_pos, n_time)."""
    lam = (np.nan_to_num(rates, nan=0.0) + _EPS8) * dt
    log = counts @ np.log(lam) - lam.sum(axis=0)[None, :]
    log -= log.max(axis=1, keepdims=True)
    post = np.exp(log)
    post /= post.sum(axis=1, keepdims=True)
    post[counts.sum(axis=1) == 0] = 0.0
    return post.T


def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _code_line_score(
    p, dt, bin_size, *, min_speed=100.0, third=True, off_track="zero", empty="zero"
):
    """lineTraj_decode.m's best line, and the variants it differs from neurospatial by.

    The code: speeds of 100 to 5000 cm/s in 50 cm/s steps, a band of 15 bins
    either side of round(x * slope) + c, zero beyond the track, lines on the
    posterior for under a third of its columns dropped, the best band mass
    over the summed probability. ``min_speed=200`` is the paper's text,
    ``third=False`` keeps every line, ``off_track="median"`` gives a column off
    the track the median band mass over the track (Davidson et al.), and
    ``empty="mean"`` divides by every time bin.
    """
    n_pos, n_time = p.shape
    speeds = np.arange(-5000.0, 5001.0, 50.0)
    speeds = speeds[np.abs(speeds) >= min_speed]
    cum = np.vstack([np.zeros((1, n_time)), np.cumsum(p, axis=0)])
    x = np.arange(n_time)
    band = CODE_BAND_BINS
    if off_track == "median":
        centres = np.arange(n_pos)
        median = np.median(
            cum[np.clip(centres + band + 1, 0, n_pos)]
            - cum[np.clip(centres - band, 0, n_pos)],
            axis=0,
        )
    best = -np.inf
    for speed in speeds:
        rows = _matlab_round(x * speed / bin_size * dt).astype(int)
        offsets = np.arange(-rows.max() - band - 1, n_pos - rows.min() + band + 1)
        line = offsets[:, None] + rows[None, :]
        on = (line >= 0) & (line < n_pos)
        mass = (
            cum[np.clip(line + band + 1, 0, n_pos), x[None, :]]
            - cum[np.clip(line - band, 0, n_pos), x[None, :]]
        )
        if off_track == "median":
            mass = np.where(on, mass, median[None, :])
        score = mass.sum(axis=1)
        kept = on.sum(axis=1) >= (n_time / 3 if third else 1)
        if kept.any():
            best = max(best, score[kept].max())
    total = p.sum() if empty == "zero" else n_time
    return best / total if np.isfinite(best) and total > 0 else np.nan


def _code_p(rates, counts, dt, bin_size, rng, **variant):
    """The score's p against 100 maps each rotated by 1 to n - 1 bins (the Methods)."""
    observed = _code_line_score(
        _code_posterior(rates, counts, dt), dt, bin_size, **variant
    )
    null = np.array(
        [
            _code_line_score(
                _code_posterior(
                    np.stack(
                        [
                            np.roll(r, s)
                            for r, s in zip(
                                rates,
                                rng.integers(1, rates.shape[1], len(rates)),
                                strict=True,
                            )
                        ]
                    ),
                    counts,
                    dt,
                ),
                dt,
                bin_size,
                **variant,
            )
            for _ in range(100)
        ]
    )
    return (np.sum(null >= observed) + 1) / 101


CODE_VARIANTS = {
    "code": {},
    "speed_200": {"min_speed": 200.0},
    "every_line": {"third": False},
    "median_off_track": {"off_track": "median"},
    "mean_over_bins": {"empty": "mean"},
}


@pytest.fixture(scope="module")
def compared(rec, place, candidates, judged):
    """On N_COMPARED events drawn with SEED: the preset's verdict and each code variant's."""
    _, cells = place
    maps = _rate_maps(rec, cells)
    order = np.argsort(rec.env.to_linear(rec.env.bin_centers))
    arrays = {name: np.asarray(m.firing_rates)[:, order] for name, m in maps.items()}
    dt, bin_size = rec.params["replay_dt"], rec.params["bin_size"]
    drawn = np.random.default_rng(SEED).choice(len(candidates), N_DRAWN, replace=False)
    pick = np.sort(drawn)[:N_COMPARED]
    rows = []
    for k in pick:
        counts, _ = _event_inputs(rec, cells, *candidates[k])
        row = {"ours": judged[k][0]}
        for name, variant in CODE_VARIANTS.items():
            p = min(
                _code_p(
                    arrays[d],
                    counts,
                    dt,
                    bin_size,
                    np.random.default_rng([SEED, int(k), i]),
                    **variant,
                )
                for i, d in enumerate(arrays)
            )
            row[name] = bool(p < 0.2)
        rows.append(row)
    return rows


@pytest.mark.realdata
def test_the_papers_code_against_the_preset(compared):
    """Verdicts both ways, and what each of the code's choices changes."""
    ours = np.array([row["ours"] for row in compared])
    counts = {}
    for name in CODE_VARIANTS:
        theirs = np.array([row[name] for row in compared])
        counts[name] = (
            int(np.sum(theirs & ours)),
            int(np.sum(theirs & ~ours)),
            int(np.sum(~theirs & ours)),
        )
    assert counts == CODE_COMPARISON


@pytest.mark.realdata
@pytest.mark.parametrize(
    ("session", "n_ca1", "n_place", "paper_session"),
    [
        ("R2336-2015-11-01", 56, 52, 2),
        ("R2336-2015-11-04", 71, 59, 1),
        ("R2337-2015-11-27", 72, 49, 1),
        ("R2337-2015-12-01", 46, 43, 2),
    ],
)
def test_days_matched_to_the_papers_sessions(session, n_ca1, n_place, paper_session):
    """CA1 units bound the paper's place cells; place-cell order agrees within a rat."""
    from neurospatial.datasets import load

    day = load("olafsdottir_2016", session)
    ca1, cells = _place_cells(day)
    assert (len(ca1), len(cells)) == (n_ca1, n_place)
    assert day.params["paper_session"] == (session.split("-")[0], paper_session)
    place_cells, *_ = olafsdottir_2016.PAPER_TABLE[session.split("-")[0]][
        paper_session - 1
    ]
    assert n_ca1 >= place_cells
