"""The Axona and Tint readers, on files written from their documented layouts."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets._axona import (
    DATA_END,
    UNTRACKED,
    microvolts_per_unit,
    read_clu,
    read_continuous,
    read_cut,
    read_header,
    read_position,
    read_set,
    read_tetrode,
    tetrode_channels,
)


def _file(*lines: str, data: bytes = b"", end: bytes = DATA_END) -> bytes:
    """An Axona data file: CRLF header lines, data_start, the data, data_end."""
    head = "".join(f"{line}\r\n" for line in lines).encode("latin-1")
    return head + b"data_start" + data + end


_TRIAL = ("trial_date Sunday, 1 Nov 2015", "trial_time 09:30:49", "duration 2")


def _tetrode_lines(n_spikes: int, n_samples: int = 50) -> tuple[str, ...]:
    return (
        *_TRIAL,
        "num_chans 4",
        "timebase 96000 hz",
        "bytes_per_timestamp 4",
        f"samples_per_spike {n_samples}",
        "sample_rate 48000 hz",
        "bytes_per_sample 1",
        "spike_format t,ch1,t,ch2,t,ch3,t,ch4",
        f"num_spikes {n_spikes}     ",
    )


def _spike(stamp: int, samples: np.ndarray, stamps: tuple[int, ...] = ()) -> bytes:
    """One spike: per channel a big-endian timestamp then its signed bytes."""
    stamps = stamps or (stamp,) * 4
    return b"".join(
        int(s).to_bytes(4, "big", signed=True) + samples[k].astype("i1").tobytes()
        for k, s in enumerate(stamps)
    )


@pytest.fixture
def tetrode(tmp_path):
    """Three spikes of 50 samples, the third with each channel's extremes."""
    rng = np.random.default_rng(3)
    waves = rng.integers(-128, 128, size=(3, 4, 50))
    waves[2, :, 0] = [-128, 127, 0, -1]
    stamps = [96_000, 144_000, 959_999]  # 1 s, 1.5 s and just under 10 s
    data = b"".join(_spike(s, w) for s, w in zip(stamps, waves, strict=True))
    path = tmp_path / "trial.10"
    path.write_bytes(_file(*_tetrode_lines(3), data=data))
    return path, waves, stamps


def test_tetrode_times_are_timestamps_over_the_timebase(tetrode):
    path, _, stamps = tetrode
    spikes = read_tetrode(path)
    np.testing.assert_array_equal(spikes.times, np.array(stamps) / 96_000)
    assert spikes.times[1] == 1.5
    assert spikes.sample_rate == 48_000.0
    assert spikes.header["num_spikes"] == "3"


def test_tetrode_waveforms_are_signed_bytes_by_spike_channel_sample(tetrode):
    path, waves, _ = tetrode
    spikes = read_tetrode(path)
    assert spikes.waveforms.dtype == np.int8
    assert spikes.waveforms.shape == (3, 4, 50)
    np.testing.assert_array_equal(spikes.waveforms, waves)
    assert spikes.waveforms[2, :, 0].tolist() == [-128, 127, 0, -1]


def test_tetrode_with_unequal_channel_timestamps_is_refused(tmp_path):
    waves = np.zeros((4, 50))
    path = tmp_path / "trial.1"
    path.write_bytes(
        _file(*_tetrode_lines(1), data=_spike(0, waves, stamps=(10, 10, 11, 10)))
    )
    with pytest.raises(ValueError, match="different timestamps per channel"):
        read_tetrode(path)


def test_tetrode_cut_short_is_refused(tmp_path):
    waves = np.zeros((4, 50))
    path = tmp_path / "trial.1"
    path.write_bytes(_file(*_tetrode_lines(2), data=_spike(0, waves)))
    with pytest.raises(ValueError, match="the header's counts give 432 data bytes"):
        read_tetrode(path)


def test_tetrode_without_data_end_is_refused(tmp_path):
    path = tmp_path / "trial.1"
    path.write_bytes(
        _file(*_tetrode_lines(1), data=_spike(0, np.zeros((4, 50))), end=b"")
    )
    with pytest.raises(ValueError, match="before its end marker"):
        read_tetrode(path)


def test_tetrode_in_another_layout_is_refused(tmp_path):
    lines = [
        line.replace("bytes_per_sample 1", "bytes_per_sample 2")
        for line in _tetrode_lines(0)
    ]
    path = tmp_path / "trial.1"
    path.write_bytes(_file(*lines))
    with pytest.raises(ValueError, match=r"\['bytes_per_sample'\]"):
        read_tetrode(path)


def test_file_without_data_start_is_refused(tmp_path):
    path = tmp_path / "trial.1"
    path.write_bytes(b"num_chans 4\r\n")
    with pytest.raises(ValueError, match="no 'data_start'"):
        read_header(path)


def test_header_offset_is_just_past_data_start(tetrode):
    path, _, _ = tetrode
    header, offset = read_header(path)
    raw = path.read_bytes()
    assert raw[offset - 10 : offset] == b"data_start"
    assert header["spike_format"] == "t,ch1,t,ch2,t,ch3,t,ch4"
    assert header["trial_date"] == "Sunday, 1 Nov 2015"


def _position_file(path, rows: np.ndarray, counter: np.ndarray) -> None:
    lines = (
        *_TRIAL,
        "num_colours 4",
        "timebase 50 hz",
        "bytes_per_timestamp 4",
        "sample_rate 50.0 hz",
        "pos_format t,x1,y1,x2,y2,numpix1,numpix2",
        "bytes_per_coord 2",
        "pixels_per_metre 300",
        f"num_pos_samples {len(rows)}     ",
    )
    data = b"".join(
        int(c).to_bytes(4, "big") + row.astype(">u2").tobytes()
        for c, row in zip(counter, rows, strict=True)
    )
    path.write_bytes(_file(*lines, data=data))


def test_position_words_frame_times_and_scale(tmp_path):
    rows = np.array(
        [
            [300, 150, 310, 160, 40, 12, 52, 0],
            [UNTRACKED, UNTRACKED, 320, 170, 0, 11, 11, 0],
            [600, 559, UNTRACKED, UNTRACKED, 38, 0, 38, 0],
        ]
    )
    path = tmp_path / "trial.pos"
    _position_file(path, rows, np.array([7, 8, 8]))
    track = read_position(path)
    np.testing.assert_array_equal(track.words, rows)
    assert track.words.dtype == np.uint16
    # Frame k, counted from 1, is at k / 50 s; the counter is not a clock.
    np.testing.assert_array_equal(track.times, [0.02, 0.04, 0.06])
    np.testing.assert_array_equal(track.frame_counter, [7, 8, 8])
    assert track.sample_rate == 50.0
    assert track.pixels_per_metre == 300.0


def test_position_count_beyond_the_data_is_refused(tmp_path):
    path = tmp_path / "trial.pos"
    _position_file(path, np.zeros((2, 8)), np.arange(2))
    raw = path.read_bytes().replace(b"num_pos_samples 2", b"num_pos_samples 3")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="the header's counts give 60 data bytes"):
        read_position(path)


def test_eeg_is_signed_bytes(tmp_path):
    path = tmp_path / "trial.eeg"
    values = np.array([-128, -1, 0, 1, 127], dtype="i1")
    path.write_bytes(
        _file(
            *_TRIAL,
            "num_chans 1",
            "sample_rate 250.0 hz",
            "bytes_per_sample 1",
            "num_EEG_samples 5",
            data=values.tobytes(),
        )
    )
    samples, rate, _ = read_continuous(path)
    assert samples.tolist() == [-128, -1, 0, 1, 127]
    assert rate == 250.0


def test_egf_is_little_endian_int16(tmp_path):
    path = tmp_path / "trial.egf2"
    values = np.array([-32768, -2, 256, 32767], dtype="<i2")
    path.write_bytes(
        _file(
            *_TRIAL,
            "num_chans 1",
            "sample_rate 4800 Hz",
            "bytes_per_sample 2",
            "num_EGF_samples 4",
            data=values.tobytes(),
        )
    )
    samples, rate, header = read_continuous(path)
    # 256 big-endian would read as 1: the order is the file's.
    assert samples.tolist() == [-32768, -2, 256, 32767]
    assert rate == 4800.0
    assert header["num_EGF_samples"] == "4"


def _cut(trial: str, ids: list[int], stated: int | None = None) -> str:
    rows = [ids[k : k + 25] for k in range(0, len(ids), 25)]
    body = "\r\n".join("".join(f"  {v}" for v in row) for row in rows)
    return (
        "n_clusters: 3\r\nn_channels: 4\r\nn_params: 2\r\n"
        "times_used_in_Vt:   13   11   13   10\r\n"
        " cluster: 0 center:    0    0    0    0    0    0    0    0\r\n"
        "               min:    0    0    0    0    0    0    0    0\r\n"
        "               max:  256  256  256  256  256  256  256  256\r\n\r\n"
        f"Exact_cut_for: {trial} spikes: {stated if stated is not None else len(ids)}\r\n"
        + body
    )


def test_cut_gives_one_cluster_per_spike_across_rows(tmp_path):
    ids = [0, 2, 1] * 10  # 30 ids over two rows of 25 and 5
    path = tmp_path / "trial_10.cut"
    path.write_text(_cut("20151101_R2336_track1", ids), encoding="latin-1")
    clusters, trial = read_cut(path)
    assert trial == "20151101_R2336_track1"
    np.testing.assert_array_equal(clusters, ids)


def test_cut_with_a_wrong_count_is_refused(tmp_path):
    path = tmp_path / "trial_10.cut"
    path.write_text(_cut("t", [1, 2, 3], stated=4), encoding="latin-1")
    with pytest.raises(ValueError, match="states 4 spikes but holds 3"):
        read_cut(path)


def test_file_without_exact_cut_for_is_refused(tmp_path):
    path = tmp_path / "trial_10.cut"
    path.write_text("n_clusters: 3\r\n", encoding="latin-1")
    with pytest.raises(ValueError, match="not a Tint cut file"):
        read_cut(path)


def test_clu_drops_its_cluster_count(tmp_path):
    path = tmp_path / "trial.clu.3"
    path.write_text("4\n1\n3\n4\n2\n", encoding="latin-1")
    np.testing.assert_array_equal(read_clu(path), [1, 3, 4, 2])


def test_clu_with_an_id_past_its_count_is_refused(tmp_path):
    path = tmp_path / "trial.clu.3"
    path.write_text("2\n1\n3\n", encoding="latin-1")
    with pytest.raises(ValueError, match="past its 2"):
        read_clu(path)


def test_set_is_read_as_key_value_text(tmp_path):
    path = tmp_path / "trial.set"
    path.write_text(
        "trial_time 09:30:49\r\ncomments R2336\r\ngain_ch_36 11000\r\nempty_key\r\n",
        encoding="latin-1",
    )
    settings = read_set(path)
    assert settings == {
        "trial_time": "09:30:49",
        "comments": "R2336",
        "gain_ch_36": "11000",
        "empty_key": "",
    }


def test_tetrode_channels_are_four_per_tetrode_from_zero():
    assert tetrode_channels(1) == (0, 1, 2, 3)
    assert tetrode_channels(16) == (60, 61, 62, 63)


@pytest.mark.parametrize(
    ("gain", "width", "expected"),
    [
        # 1500 mV full scale over a gain of 10,000 is 150 uV over 128 steps.
        (10_000, 1, 150.0 / 128),
        # A 2-byte .egf at gain 1000: 1500 uV over 32,768 steps.
        (1000, 2, 1500.0 / 32768),
    ],
)
def test_microvolts_per_unit_is_full_scale_over_gain_and_steps(gain, width, expected):
    settings = {"ADC_fullscale_mv": "1500", "gain_ch_5": str(gain)}
    assert microvolts_per_unit(settings, 5, width) == pytest.approx(expected, rel=1e-12)


def test_microvolts_per_unit_without_the_gain_is_refused():
    with pytest.raises(ValueError, match="gain_ch_7"):
        microvolts_per_unit({"ADC_fullscale_mv": "1500"}, 7, 1)
