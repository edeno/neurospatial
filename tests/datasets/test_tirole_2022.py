"""The tirole_2022 loader: synthetic files in the deposit's format, then the real ones."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from neurospatial.datasets import load, tirole_2022

DT = 0.04
SCALE = 0.43


def _track(t, start, stop, mask, cm_per_pixel):
    """One element of ``position.linear``: linear 0-200 cm while on the track."""
    on = (t >= start) & (t <= stop)
    linear = np.full(t.size, np.nan)
    linear[on] = np.linspace(0.0, 200.0, on.sum())
    return {
        "linear": linear,
        "timestamps": t[on],
        "track_mask_x": mask[:, 0],
        "track_mask_y": mask[:, 1],
        "cm_per_pixel": cm_per_pixel,
        "length": 2,
        "branch_type": "mono",
    }


def _position(n_tracks=2):
    """1000 s at 25 Hz: rest pot, tracks of 100 s each, rest pot between and after."""
    t = 5000.0 + np.arange(25_000) * DT
    tracks = [(5200.0, 5300.0), (5400.0, 5500.0), (5700.0, 5800.0), (5850.0, 5900.0)]
    masks = [
        np.column_stack([np.linspace(100, 565, 466), np.full(466, 100.0)]),
        np.column_stack([np.full(454, 300.0), np.linspace(150, 603, 454)]),
    ]
    linear = [
        _track(t, *window, masks[k % 2], 200 / (465.0 if k % 2 == 0 else 453.0))
        for k, window in enumerate(tracks[:n_tracks])
    ]
    x = np.full(t.size, 40.0)
    y = np.full(t.size, 400.0)
    for track, mask in zip(linear, masks * 2, strict=False):
        on = np.isfinite(track["linear"])
        index = np.round(track["linear"][on] / 200 * (len(mask) - 1)).astype(int)
        x[on], y[on] = mask[index, 0], mask[index, 1]
    in_tracks = np.any([np.isfinite(track["linear"]) for track in linear], axis=0)
    sleepbox = np.where(in_tracks, np.nan, t)
    sleepbox[100:105] = np.nan  # a tracking gap in the pot
    return {
        "t": t,
        "x": x.astype(np.uint16),
        "y": y.astype(np.uint16),
        "cm_per_pixel": SCALE,
        "sleepbox": sleepbox,
        "linear": linear,
        "v_cm": np.zeros(t.size),
    }


def _clusters():
    """Three units with ids 1-3 on tetrodes 2, 2 and 11, spikes in any order."""
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(5000.0, 5999.0, 300))
    ids = np.tile([3, 1, 2], 100)
    return {
        "spike_id": ids.astype(np.uint8),
        "spike_times": times,
        "id_conversion": np.array([[1, 2001], [2, 2004], [3, 11001]], dtype=np.uint16),
    }


def _write(folder, info, *, n_tracks=2):
    folder.mkdir(parents=True, exist_ok=True)
    savemat(
        folder / f"{info.spikes_file}_extracted_position.mat",
        {"position": _position(n_tracks)},
    )
    savemat(
        folder / f"{info.spikes_file}_extracted_clusters.mat", {"clusters": _clusters()}
    )
    return folder


def _load(tmp_path, session_entry, session="rat3_session2", **kwargs):
    info = tirole_2022.SESSIONS[session]
    folder = _write(tmp_path / "tirole_2022" / session, info, **kwargs)
    return tirole_2022.load_session(session_entry("tirole_2022", session), folder)


def test_positions_are_pixels_times_the_files_average_scale(tmp_path, session_entry):
    rec = _load(tmp_path, session_entry)
    stored = _position()
    np.testing.assert_allclose(rec.positions[:, 0], stored["x"] * SCALE)
    np.testing.assert_allclose(rec.positions[:, 1], stored["y"] * SCALE)
    assert rec.position_units == "cm"
    bases = {c.text.split(":")[0]: c.basis for c in rec.conversions}
    assert bases["positions"] == "stated"
    assert rec.env_kind == "declared"
    assert rec.env.is_linearized_track
    # Each track's own scale against the average is reported, not applied.
    assert any("track_1 0.4301 (+0.0%)" in text for text in rec.watch_outs)
    assert any("track_2 0.4415 (+2.7%)" in text for text in rec.watch_outs)


def test_linearized_joins_the_tracks_and_the_epochs_follow_them(
    tmp_path, session_entry
):
    rec = _load(tmp_path, session_entry)
    stored = _position()
    on_1 = np.isfinite(stored["linear"][0]["linear"])
    on_2 = np.isfinite(stored["linear"][1]["linear"])
    np.testing.assert_array_equal(
        rec.linearized[on_1], stored["linear"][0]["linear"][on_1]
    )
    np.testing.assert_array_equal(
        rec.linearized[on_2], stored["linear"][1]["linear"][on_2]
    )
    assert np.all(np.isnan(rec.linearized[~(on_1 | on_2)]))
    t = stored["t"]
    np.testing.assert_allclose(rec.epochs["pre"], [[t[0], t[on_1][0]]])
    np.testing.assert_allclose(rec.epochs["track_1"], [[t[on_1][0], t[on_1][-1]]])
    np.testing.assert_allclose(
        rec.epochs["between_tracks"], [[t[on_1][-1], t[on_2][0]]]
    )
    np.testing.assert_allclose(rec.epochs["post"], [[t[on_2][-1], t[-1]]])
    # The rest pot's runs, as stored: broken by the gap, each up to the next sample.
    pot = rec.epochs["rest_pot"]
    np.testing.assert_allclose(pot[0], [t[0], t[100]])
    np.testing.assert_allclose(pot[1, 0], t[105])
    assert rec.params["track_epochs"] == ("track_1", "track_2")
    assert set(rec.params["tracks"]) == {"track_1", "track_2"}
    assert rec.params["tracks"]["track_1"].n_bins == 20


def test_a_reexposure_ends_post_and_gets_its_own_epochs(tmp_path, session_entry):
    rec = _load(tmp_path, session_entry, session="rat2_session1", n_tracks=4)
    np.testing.assert_allclose(
        rec.epochs["post"][0, 1], rec.epochs["track_1_reexposure"][0, 0]
    )
    assert rec.epochs["post_reexposure"][0, 0] == rec.epochs["track_2_reexposure"][0, 1]
    assert rec.params["track_epochs"][2:] == (
        "track_1_reexposure",
        "track_2_reexposure",
    )
    # The environment holds the first exposure's two tracks only.
    assert rec.env.n_bins == _load(tmp_path, session_entry).env.n_bins
    assert any("ran both tracks again" in text for text in rec.watch_outs)


def test_units_keep_the_authors_ids_tetrodes_and_clusters(tmp_path, session_entry):
    rec = _load(tmp_path, session_entry)
    stored = _clusters()
    assert rec.units.index.tolist() == [1, 2, 3]
    assert rec.units["tetrode"].tolist() == [2, 2, 11]
    assert rec.units["cluster"].tolist() == [1, 4, 1]
    for unit, train in zip((1, 2, 3), rec.spike_times, strict=True):
        np.testing.assert_array_equal(
            train, stored["spike_times"][stored["spike_id"] == unit]
        )


def test_crossed_labels_and_a_removed_first_track_are_reported(tmp_path, session_entry):
    rec = _load(tmp_path, session_entry, session="rat4_session3")
    assert any(
        "Spikes and position come from RAT1_SESS1, the LFP from RAT4_SESS3" in w
        for w in rec.watch_outs
    )
    assert not any("third track first" in w for w in rec.watch_outs)
    rec = _load(tmp_path, session_entry, session="rat1_session1")
    assert any("from RAT1_SESS2, the LFP from RAT1_SESS1" in w for w in rec.watch_outs)
    assert any("third track first" in w for w in rec.watch_outs)
    rec = _load(tmp_path, session_entry, session="rat3_session2")
    assert not any("labels are crossed" in w for w in rec.watch_outs)


def test_rest_against_table_1_is_reported_when_it_differs():
    info = tirole_2022.SESSIONS["rat5_session2"]  # Table 1: 72 and 86 min
    epochs = {"pre": np.array([[0.0, 86.0 * 60]]), "post": np.array([[0.0, 72.0 * 60]])}
    (text,) = tirole_2022._rest_watch_out(epochs, info)
    assert text.endswith("Table 1 gives them the other way round.")
    epochs = {"pre": np.array([[0.0, 72.4 * 60]]), "post": np.array([[0.0, 85.5 * 60]])}
    assert tirole_2022._rest_watch_out(epochs, info) == ()


def _write_v73_csc(path, labels, raw, times):
    """A CSC struct array as MATLAB v7.3 stores it: HDF5 references, arrays transposed."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as file:
        refs = file.create_group("#refs#")
        group = file.create_group("CSC")
        fields = {
            "channel_label": [
                np.array([[ord(c)] for c in label], dtype=np.uint16) for label in labels
            ],
            "CSCraw": [column[None, :] for column in raw.T],
            "CSCtime": [times[None, :]] * len(labels),
            "ripple_zscore": [np.zeros(2, dtype=np.uint64)] * len(labels),
        }
        for name, values in fields.items():
            references = np.empty((len(labels), 1), dtype=h5py.ref_dtype)
            for i, value in enumerate(values):
                data = refs.create_dataset(f"{name}{i}", data=value)
                if name == "channel_label":
                    data.attrs["MATLAB_class"] = np.bytes_("char")
                elif name == "ripple_zscore":
                    data.attrs["MATLAB_empty"] = np.uint8(1)
                references[i, 0] = data.ref
            group.create_dataset(name, data=references)


def test_read_lfp_takes_channels_by_label_from_v73_and_v7_files(tmp_path):
    labels = ["best_theta", "best_ripple", "best_delta", "best_theta_low_ripple"]
    times = 100.0 + np.arange(50) / 1000.0
    raw = np.arange(200, dtype=float).reshape(50, 4) * 10.0
    v73 = tmp_path / "v73.mat"
    _write_v73_csc(v73, labels, raw, times)
    samples, clock = tirole_2022.read_lfp((v73,), ("best_ripple", "best_theta"))
    np.testing.assert_array_equal(samples, raw[:, [1, 0]])
    np.testing.assert_array_equal(clock, times)

    v7 = tmp_path / "v7.mat"
    struct = np.zeros(
        (1, 4),
        dtype=[("channel_label", object), ("CSCraw", object), ("CSCtime", object)],
    )
    for i, label in enumerate(labels):
        struct[0, i] = (label, raw[:, i : i + 1], times[:, None])
    savemat(v7, {"CSC": struct})
    samples, clock = tirole_2022.read_lfp((v7,), ("best_ripple",))
    np.testing.assert_array_equal(samples[:, 0], raw[:, 1])
    np.testing.assert_array_equal(clock, times)
    with pytest.raises(ValueError, match="holds no channel labelled"):
        tirole_2022.read_lfp((v7,), ("ripple",))


def test_registry_offers_nine_sessions_from_the_paired_files(session_entry):
    from neurospatial.datasets._catalog import sessions

    entries = {entry.session: entry for entry in sessions("tirole_2022")}
    assert set(entries) == set(tirole_2022.SESSIONS)
    assert "rat4_session1" not in entries
    for name, info in tirole_2022.SESSIONS.items():
        entry = entries[name]
        assert [m.path for m in entry.members] == [
            f"{info.spikes_file}_extracted_clusters.mat",
            f"{info.spikes_file}_extracted_position.mat",
        ]
        assert [f.path for f in entry.lfp_files] == [
            f"{info.lfp_file}_extracted_CSC.mat"
        ]


def test_track_environment_is_the_authors_200_cm_in_10_cm_bins():
    env = tirole_2022.track_environment()
    assert env.n_bins == 20
    np.testing.assert_allclose(env.bin_centers[:, 0], np.arange(5.0, 200.0, 10.0))


# ---------------------------------------------------------------- real data


@pytest.mark.realdata
def test_default_session(session_entry, assert_default_session):
    """Rat 3 session 2: 142 units, two tracks, four LFP channels on request."""
    rec = load("tirole_2022")
    assert_default_session(rec, session_entry("tirole_2022"))
    assert rec.n_units == 142
    assert 24.9 <= rec.sampling_rate <= 25.1
    assert rec.lfp.ripple_channels == ("best_ripple",)
    assert rec.params["track_epochs"] == ("track_1", "track_2")
    np.testing.assert_allclose(np.nanmin(rec.linearized), 0.0)
    np.testing.assert_allclose(np.nanmax(rec.linearized), 200.0)


@pytest.mark.realdata
@pytest.mark.parametrize("session", sorted(tirole_2022.SESSIONS))
def test_rest_durations_are_table_1s(session):
    """PRE and POST within a minute of Table 1's: the evidence for each pairing.

    Rat 5 session 2's are Table 1's the other way round, which the session's
    watch-outs say.
    """
    rec = load("tirole_2022", session)
    pre, post = (np.diff(rec.epochs[name])[0, 0] / 60 for name in ("pre", "post"))
    stated = tirole_2022.SESSIONS[session].rest_minutes
    if session == "rat5_session2":
        stated = stated[::-1]
        assert any("the other way round" in text for text in rec.watch_outs)
    assert abs(pre - stated[0]) <= 1
    assert abs(post - stated[1]) <= 1


@pytest.mark.realdata
def test_lfp_is_on_the_clock_of_the_spikes():
    """Population firing peaks at the authors' ripple channel's power peaks."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    rec = load("tirole_2022")
    samples, times = rec.lfp.fetch()
    band = sosfiltfilt(
        butter(3, [150, 250], "bandpass", fs=1000.0, output="sos"), samples[:, 0]
    )
    power = np.convolve(band**2, np.ones(20) / 20, mode="same")
    z = (power - power.mean()) / power.std()
    peaks, _ = find_peaks(z, height=5, distance=500)
    spikes = np.sort(np.concatenate(rec.spike_times))
    lags = np.arange(-0.2, 0.2001, 0.01)
    counts = [
        np.sum(
            np.searchsorted(spikes, times[peaks] + lag + 0.005)
            - np.searchsorted(spikes, times[peaks] + lag - 0.005)
        )
        for lag in lags
    ]
    assert abs(lags[int(np.argmax(counts))]) <= 0.01
    assert max(counts) > 3 * np.mean(counts[:5])
