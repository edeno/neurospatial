"""Grieves et al. (2020), UCL RDR 9926477: synthetic session files in the archive's layout, and the real session."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._checks import run_checks

DEFAULT = "926_06042018"
#: The default session's 2-D frame: x3 = x2 + A, y3 = B - y2 (mm).
A, B = -165.6, 1039.6
#: lxrm = lx / 50 + C per axis, and the grid's shape (x, y, z).
C = (5.2, 15.8, 5.75)
SHAPE = (35, 39, 33)
SIDE_MM = 970.0
#: The parts' spans on the session clock, s.
PARTS = {
    "square1": [(0.0, 200.0)],
    "lattice": [(200.0, 800.0)],
    "square2": [(800.0, 1000.0)],
}
NAMES = ("200101a_square", "200101b_lattice", "200101c_square")


def _bars_mm():
    """The aligned lattice's 147 bars from (0, 0, 0) to 970 mm, as (start, stop, NaN) rows."""
    ticks = np.linspace(0.0, SIDE_MM, 7)
    rows = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for u in ticks:
            for v in ticks:
                start, stop = np.zeros(3), np.zeros(3)
                start[others], stop[others] = (u, v), (u, v)
                stop[axis] = SIDE_MM
                rows += [start, stop, np.full(3, np.nan)]
    return np.array(rows)


def _path_mm(times):
    """A smooth closed path through the cube, 60 to 910 mm on each axis, in mm."""
    phase = 2 * np.pi * (times - times[0])
    return np.column_stack(
        [
            485.0 + 425.0 * np.sin(phase / 31.0),
            485.0 + 425.0 * np.sin(phase / 23.0 + 1.0),
            485.0 + 425.0 * np.sin(phase / 47.0 + 2.0),
        ]
    )


def mirror_camera(xyz):
    """The default session's 2-D positions: the 3-D x, y mirrored in y and shifted."""
    return np.column_stack([xyz[:, 0] - A, B - xyz[:, 1]])


def pixel_camera(xyz):
    """2-D positions of another frame: x3 = 2.2 * R x2 + t, R a 90 deg turn then a mirror, plus noise."""
    rotation = np.array([[0.0, 1.0], [1.0, 0.0]])  # a reflection across y = x
    noise = np.random.default_rng(4).normal(0.0, 0.02, (len(xyz), 2))
    return (xyz[:, :2] - [30.0, -40.0]) @ rotation / 2.2 + noise


def _write_char(group, name, text):
    data = np.array([[ord(c)] for c in text], dtype=np.uint16)
    group.create_dataset(name, data=data).attrs["MATLAB_class"] = np.bytes_("char")


def _row(group, name, values, dtype=np.float32):
    group.create_dataset(name, data=np.asarray(values, dtype=dtype).reshape(1, -1))


def write_session(
    path,
    *,
    parts=PARTS,
    names=NAMES,
    camera=mirror_camera,
    velocity_scale=0.1,
    n_bars=147,
    grid_slope=1 / 50,
    heading_offset=0.0,
    spikes_until=None,
    times_dtype=np.float32,
):
    """A session file laid out as the archive's: an ``sdata`` struct in HDF5.

    The lattice part is 50 Hz of a path in mm; ``lattice.poh`` is the
    direction of movement in the 2-D positions' frame, in degrees, plus
    ``heading_offset`` (rad), and ``lattice.pov`` the 2-D speed. One tetrode: cluster 0 (unsorted) and
    clusters 1 and 2, spiking from 0 s to the end of the last part, or to
    ``spikes_until``. Both ``pot`` and the spike times are stored as
    ``times_dtype``, float32 as in the archive.
    """
    h5py = pytest.importorskip("h5py")
    times = np.concatenate(
        [np.arange(start + 0.02, stop, 0.02) for start, stop in parts["lattice"]]
    )
    xyz = _path_mm(times)
    step = np.diff(xyz, axis=0) / np.diff(times)[:, None]
    speed = np.r_[np.nan, np.linalg.norm(step, axis=1)]
    flat = camera(xyz)
    move = np.diff(flat, axis=0)
    flat_speed = np.r_[np.nan, np.linalg.norm(move, axis=1) / np.diff(times)]
    azimuth = np.r_[np.arctan2(move[:, 1], move[:, 0]), 0.0] + heading_offset
    azimuth[-1] = azimuth[-2]
    bars = _bars_mm()[: 3 * n_bars]
    end = max(stop for rows in parts.values() for _, stop in rows)
    if spikes_until is not None:
        end = spikes_until
    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as mat:
        s = mat.create_group("sdata")
        _write_char(s, "rat_num", "999")
        refs = mat.create_group("#refs#")
        for name in names:
            _write_char(refs, name, name)
        s.create_dataset(
            "session_names",
            data=np.array([[refs[name].ref for name in names]], dtype=h5py.ref_dtype),
        )
        config = s.create_group("config3")
        _write_char(config, "lattice_type", "normal")
        config.create_dataset("lat_size", data=np.full((1, 3), SIDE_MM))
        s.create_dataset("position_srate_hz", data=np.array([[50.0]]))
        part_config = s.create_group("part_config")
        for name, rows in parts.items():
            # MATLAB (n, 2) [start stop] rows, stored transposed.
            part_config.create_group(name).create_dataset(
                "times", data=np.array(rows, dtype=float).T
            )
        lattice3 = s.create_group("lattice_3d")
        _row(lattice3, "pot", times, times_dtype)
        for key, column in zip(("pox", "poy", "poz"), xyz.T, strict=True):
            _row(lattice3, key, column)
        _row(lattice3, "pov", speed * velocity_scale)
        for axis, column in enumerate("xyz"):
            _row(lattice3, f"l{column}", bars[:, axis], np.float64)
            _row(
                lattice3,
                f"l{column}rm",
                bars[:, axis] * grid_slope + C[axis],
                np.float64,
            )
        lattice3.create_dataset(
            "dwellmap", data=np.zeros((SHAPE[2], SHAPE[0], SHAPE[1]), dtype=np.float32)
        )
        lattice2 = s.create_group("lattice")
        _row(lattice2, "pot", times, times_dtype)
        _row(lattice2, "pox", flat[:, 0])
        _row(lattice2, "poy", flat[:, 1])
        _row(lattice2, "poh", np.degrees(np.mod(azimuth, 2 * np.pi)))
        _row(lattice2, "pov", flat_speed * velocity_scale)
        trains = {
            "r999_01012020_t3_c0": np.sort(rng.uniform(0.0, end, 5000)),
            "r999_01012020_t3_c1": np.sort(rng.uniform(0.0, end, 300)),
            "r999_01012020_t3_c2": np.sort(rng.uniform(250.0, end - 50.0, 100)),
        }
        for name, spikes in trains.items():
            cell = s.create_group(name)
            _row(cell, "spike_times", spikes, times_dtype)
            if not name.endswith("_c0"):
                _write_char(cell, "automatic_cell_name", "Place cell")
    return times, xyz, trains


def _folder(tmp_path, **change):
    """A synthetic session in the member's directory."""
    folder = tmp_path / DEFAULT
    folder.mkdir()
    written = write_session(folder / f"{DEFAULT}.mat", **change)
    return folder, written


def _edit(folder, field, edit):
    """Change one stored row of the session file in place."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(folder / f"{DEFAULT}.mat", "r+") as mat:
        values = mat[field][()]
        edit(values[0])
        mat[field][...] = values


def _loaded(folder, session_entry):
    from neurospatial.datasets import grieves_2020

    entry = session_entry("grieves_2020")
    return run_checks(
        grieves_2020.load_session(entry, folder), stated=entry.info.stated
    )


def test_loads_the_lattice_part_in_cm(tmp_path, session_entry):
    folder, (times, xyz, trains) = _folder(tmp_path)
    rec = _loaded(folder, session_entry)
    assert rec.failed_checks == ()
    np.testing.assert_allclose(rec.times, times.astype(np.float32))
    np.testing.assert_allclose(rec.positions, xyz.astype(np.float32) / 10.0)
    assert rec.position_units == "cm"
    positions = next(c for c in rec.conversions if c.text.startswith("positions"))
    assert positions.basis == "inferred"
    assert "speed is 10.00 times the stored velocity pov" in positions.evidence
    assert "config3.lat_size is 970 x 970 x 970" in positions.evidence
    # Cluster 0 is left out; the rest are named t<tetrode>c<cluster>.
    assert list(rec.units.index) == ["t3c1", "t3c2"]
    assert list(rec.units["label"]) == ["Place cell", "Place cell"]
    assert list(rec.units["brain_region"]) == ["CA1", "CA1"]
    np.testing.assert_allclose(
        rec.spike_times[1], trains["r999_01012020_t3_c2"].astype(np.float32)
    )
    spikes = next(c for c in rec.conversions if c.text.startswith("spike_times"))
    assert "cluster 0 of each tetrode (5000 spikes) left out" in spikes.text
    # The parts are epochs; only the lattice is analysed and tracked.
    assert {name: rows.tolist() for name, rows in rec.epochs.items()} == {
        name: [list(row) for row in rows] for name, rows in PARTS.items()
    }
    np.testing.assert_array_equal(rec.params["analysis_epochs"], [[200.0, 800.0]])
    np.testing.assert_array_equal(rec.params["train_epochs"][0], [200.0, 260.0])
    np.testing.assert_array_equal(rec.params["test_epochs"][0], [260.0, 320.0])
    assert rec.params["min_speed"] is None
    assert rec.params["bin_size"] == 5.0
    assert rec.animal == "rat 999"
    assert "the aligned lattice (config3.lattice_type 'normal')" in rec.apparatus
    assert "200101a_square, 200101b_lattice, 200101c_square" in rec.clock


def test_lattice_bars_are_kept_in_cm(tmp_path, session_entry):
    folder, _ = _folder(tmp_path)
    bars = _loaded(folder, session_entry).params["lattice_bars"]
    assert bars.shape == (147, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(bars[:, 1] - bars[:, 0], axis=1), 97.0)
    np.testing.assert_allclose(bars.reshape(-1, 3).max(axis=0), [97.0] * 3)


def test_a_separator_row_with_only_some_nan_is_a_separator(tmp_path, session_entry):
    """As in 926_05042018, whose separators hold a NaN only in y."""
    folder, _ = _folder(tmp_path)

    def fill_x(values):
        values[2::3] = 12.5

    _edit(folder, "sdata/lattice_3d/lx", fill_x)
    assert _loaded(folder, session_entry).params["lattice_bars"].shape == (147, 2, 3)


def test_environment_is_the_authors_grid(tmp_path, session_entry):
    """Edges at (k - c) x 5 cm for k = 0 .. n: voxel i spans [i - 1, i] in lxrm's units."""
    folder, _ = _folder(tmp_path)
    rec = _loaded(folder, session_entry)
    assert rec.env_kind == "inferred"
    assert rec.env.n_bins == np.prod(SHAPE)
    assert rec.env.units == "cm"
    for axis, (offset, n_voxels) in enumerate(zip(C, SHAPE, strict=True)):
        np.testing.assert_allclose(
            rec.env.dimension_ranges[axis], [-offset * 5.0, (n_voxels - offset) * 5.0]
        )
    # x = 0.1 cm, in lxrm's units 0.1 / 5 + 5.2 = 5.22: voxel 6 (1-based),
    # spanning [5, 6], centred at (5.5 - 5.2) x 5 = 1.5 cm.
    (bin_index,) = rec.env.bin_at(np.array([[0.1, 0.1, 0.1]]))
    np.testing.assert_allclose(rec.env.bin_centers[bin_index, 0], 1.5)
    assert "holding 100.00% of the tracked samples" in rec.env_source


def test_mirrored_frames_give_minus_poh_as_an_inferred_conversion(
    tmp_path, session_entry
):
    """The exact mirror is the evidence; poh's degrees and frame stay inferences."""
    folder, (_, xyz, _) = _folder(tmp_path)
    rec = _loaded(folder, session_entry)
    move = np.diff(xyz[:, :2], axis=0)
    expected = np.arctan2(move[:, 1], move[:, 0])
    difference = np.angle(np.exp(1j * (rec.headings[:-1] - expected)))
    assert np.max(np.abs(difference)) < 1e-3
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert heading.basis == "inferred"
    assert "(here -poh)" in heading.text
    assert "no source read gives poh's unit or frame" in heading.evidence
    assert "under a mirror in y (a reflection across the x axis)" in heading.evidence
    assert "plus a shift of (-165.6, 1039.6) mm" in heading.evidence
    assert not any("not verified" in text for text in rec.watch_outs)
    assert "not verified" not in rec.heading_source


def test_another_2d_frame_is_fitted_and_its_turn_applied(tmp_path, session_entry):
    """Scaled, reflected and noisy 2-D positions: the fitted turn makes poh a heading."""
    folder, (_, xyz, _) = _folder(tmp_path, camera=pixel_camera)
    rec = _loaded(folder, session_entry)
    move = np.diff(xyz[:, :2], axis=0)
    expected = np.arctan2(move[:, 1], move[:, 0])
    difference = np.angle(np.exp(1j * (rec.headings[:-1] - expected)))
    assert np.median(np.abs(difference)) < 0.05
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert heading.basis == "inferred"
    assert "a reflection across the line at 45.0 deg to the x axis" in heading.evidence
    assert "a scale of 2.2 mm per 2-D unit" in heading.evidence
    assert "not verified" not in rec.heading_source


def test_heading_off_movement_is_kept_and_flagged(tmp_path, session_entry):
    """An azimuth a quarter turn off the movement is kept, turned, and not verified."""
    folder, _ = _folder(tmp_path, heading_offset=np.pi / 2)
    rec = _loaded(folder, session_entry)
    (finding,) = [t for t in rec.watch_outs if "not verified by movement" in t]
    assert "circular mean -90.0 deg" in finding
    assert "fit the 3-D path with R^2 1.000" in finding
    assert "not verified" in rec.heading_source


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"velocity_scale": 1.0}, r"speed is 1\.00 times pov"),
        ({"n_bars": 146}, r"expected 147 \(start, stop, NaN\) bars of 97 cm, got 146"),
        ({"grid_slope": 1 / 40}, r"lxrm is not lx / 50 \+ c"),
    ],
)
def test_a_file_that_breaks_a_conversion_is_refused(
    tmp_path, session_entry, change, match
):
    from neurospatial.datasets import grieves_2020

    folder, _ = _folder(tmp_path, **change)
    with pytest.raises(ValueError, match=match):
        grieves_2020.load_session(session_entry("grieves_2020"), folder)


def test_without_pov_the_unit_rests_on_the_lattice_size(tmp_path, session_entry):
    folder, _ = _folder(tmp_path)
    _edit(folder, "sdata/lattice_3d/pov", lambda values: values.fill(np.nan))
    rec = _loaded(folder, session_entry)
    positions = rec.conversions[0]
    assert "pov holds no value here, so it cannot be compared" in positions.evidence
    assert "config3.lat_size is 970 x 970 x 970" in positions.evidence


def test_a_lattice_of_two_recordings_and_no_second_square(tmp_path, session_entry):
    parts = {"square1": [(0.0, 200.0)], "lattice": [(200.0, 500.0), (500.0, 800.0)]}
    names = ("200101a_square", "200101b_lattice", "200102c_lattice")
    folder, _ = _folder(tmp_path, parts=parts, names=names)
    rec = _loaded(folder, session_entry)
    assert rec.failed_checks == ()
    assert list(rec.epochs) == ["square1", "lattice"]
    np.testing.assert_array_equal(
        rec.params["analysis_epochs"], [[200.0, 500.0], [500.0, 800.0]]
    )
    # Blocks alternate within each recording.
    np.testing.assert_array_equal(
        rec.params["train_epochs"][:4, 0], [200, 320, 440, 500]
    )
    (days,) = [t for t in rec.watch_outs if "were made on 2 days" in t]
    assert "200102c_lattice" in days


def test_a_square_part_ending_after_the_last_spike_is_clipped(tmp_path, session_entry):
    """square2 runs 100 s past the last spike: clipped to it, not refused."""
    folder, _ = _folder(tmp_path, spikes_until=900.0)
    rec = _loaded(folder, session_entry)
    assert rec.failed_checks == ()
    stop = max(float(t.max()) for t in rec.spike_times)
    assert stop <= 900.0
    np.testing.assert_allclose(rec.epochs["square2"], [[800.0, stop]])
    np.testing.assert_array_equal(rec.epochs["square1"], [[0.0, 200.0]])
    assert any(
        text.startswith("1 epoch row reaches past the recording")
        and "epochs['square2'] [800, 1000] s ends" in text
        for text in rec.watch_outs
    )
    clip = next(c for c in rec.conversions if c.text.startswith("epochs: rows"))
    assert "the file's part_config lists them so" in clip.evidence
    (part,) = [t for t in rec.watch_outs if t.startswith("part_config lists")]
    assert part.startswith("part_config lists square2 as [800, 1000] s, longer")
    assert f"stop firing (last spike {stop:.6g} s) before it ends" in part
    assert "covers the data, not the listed part" in part


def test_a_lattice_part_past_the_data_gives_clipped_analysis_epochs(
    tmp_path, session_entry
):
    """analysis_epochs and the decoding blocks follow the clipped lattice row."""
    h5py = pytest.importorskip("h5py")

    parts = {"square1": [(0.0, 200.0)], "lattice": [(200.0, 800.0)]}
    names = NAMES[:2]
    folder, _ = _folder(tmp_path, parts=parts, names=names, spikes_until=790.0)
    with h5py.File(folder / f"{DEFAULT}.mat", "r+") as mat:
        mat["sdata/part_config/lattice/times"][...] = [[200.0], [900.0]]
    rec = _loaded(folder, session_entry)
    assert rec.failed_checks == ()
    (stop,) = rec.epochs["lattice"][:, 1]
    assert stop == pytest.approx(rec.times[-1])
    np.testing.assert_array_equal(rec.params["analysis_epochs"], rec.epochs["lattice"])
    blocks = np.concatenate([rec.params["train_epochs"], rec.params["test_epochs"]])
    assert blocks.max() == pytest.approx(stop)


def test_held_runs_are_found_from_eleven_identical_samples():
    from neurospatial.datasets.grieves_2020 import held_runs

    values = np.r_[0.0, np.full(11, 1.0), np.full(10, 2.0), np.nan, np.nan, 3.0]
    np.testing.assert_array_equal(held_runs(values), [[1, 11]])


def test_held_azimuths_are_reported_and_kept(tmp_path, session_entry):
    from neurospatial.datasets import grieves_2020

    folder, _ = _folder(tmp_path)

    def hold(values):
        values[1000:1100] = values[1000]
        values[2000:2300] = values[2000]

    _edit(folder, "sdata/lattice/poh", hold)
    rec = grieves_2020.load_session(session_entry("grieves_2020"), folder)
    (held,) = [t for t in rec.watch_outs if "repeat the sample before" in t]
    assert "2 runs of 11 or more identical headings (398 repeated samples" in held
    assert np.all(rec.headings[1000:1100] == rec.headings[1000])
    assert not any("exactly 0" in t for t in rec.watch_outs)


def test_an_azimuth_and_speed_both_zero_is_nan_and_a_zero_alone_is_kept(
    tmp_path, session_entry
):
    """poh == 0 with pov == 0 marks no azimuth; poh == 0 with pov != 0 stays 0."""
    from neurospatial.datasets import grieves_2020

    folder, _ = _folder(tmp_path)

    def zero(values):
        values[2000:2300] = 0.0
        values[3000:3005] = 0.0

    def still(values):
        values[2000:2300] = 0.0

    _edit(folder, "sdata/lattice/poh", zero)
    _edit(folder, "sdata/lattice/pov", still)
    rec = grieves_2020.load_session(session_entry("grieves_2020"), folder)
    assert np.isnan(rec.headings[2000:2300]).all()
    # Under the mirror, 0 deg stays 0: the samples are kept where they were.
    assert np.all(rec.headings[3000:3005] == 0.0)
    assert np.isfinite(np.delete(rec.headings, np.s_[2000:2300])).all()
    lost = next(c for c in rec.conversions if c.text.startswith("headings: NaN"))
    assert lost.basis == "inferred"
    assert lost.text.startswith("headings: NaN at the 300 samples whose lattice.poh")
    n = rec.times.size
    assert f"exactly 0 in 305 of {n} samples" in lost.evidence
    assert "300 of them (98.4%) with lattice.pov" in lost.evidence
    assert "pov is never exactly 0 where poh is not" in lost.evidence
    assert "100% of them lie in runs of 11 or more, the longest 6 s" in lost.evidence
    (finding,) = [t for t in rec.watch_outs if "both exactly 0" in t]
    assert finding.startswith(f"{300 / n:.1%} of heading samples (300) have")
    assert "they are NaN in rec.headings" in finding
    assert f"{5 / (n - 300):.2%} of the other samples (5) are exactly 0 deg" in finding
    assert "kept as stored, at 0.0 rad in rec.headings" in finding
    assert "NaN where poh and pov are both 0" in rec.heading_source


def test_samples_outside_the_cube_are_counted(tmp_path, session_entry):
    from neurospatial.datasets import grieves_2020

    folder, _ = _folder(tmp_path)

    def lift(values):
        values[:300] = SIDE_MM + 100.0  # 10 cm above the top face

    _edit(folder, "sdata/lattice_3d/poz", lift)
    rec = grieves_2020.load_session(session_entry("grieves_2020"), folder)
    (finding,) = [t for t in rec.watch_outs if "outside the lattice's cube" in t]
    assert finding.startswith(f"{300 / rec.times.size:.1%} of tracked samples lie")
    assert "up to 10 cm beyond a face" in finding
    # The grid's top is (33 - 5.75) x 5 = 136.25 cm, above the lifted 107 cm.
    assert "the authors' grid, holds 100.0% of them (100.00% of all" in finding


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, "Times (lattice_3d.pot) and spike times are stored in single"),
        (np.float64, None),
    ],
)
def test_single_precision_is_reported_only_for_float32_times(
    tmp_path, session_entry, dtype, expected
):
    folder, _ = _folder(tmp_path, times_dtype=dtype)
    rec = _loaded(folder, session_entry)
    found = [t for t in rec.watch_outs if "single precision" in t]
    if expected is None:
        assert found == []
    else:
        (text,) = found
        assert text.startswith(expected)
        assert "one float32 step is 0.061 ms" in text


def test_similarity_map_recovers_a_reflection_scale_and_shift():
    from neurospatial.datasets.grieves_2020 import similarity_map

    rng = np.random.default_rng(5)
    source = rng.uniform(0.0, 100.0, (50, 2))
    angle = 0.7
    reflection = np.array(
        [[np.cos(angle), np.sin(angle)], [np.sin(angle), -np.cos(angle)]]
    )
    target = 3.5 * source @ reflection.T + [10.0, -4.0]
    rotation, scale, shift = similarity_map(source, target)
    np.testing.assert_allclose(rotation, reflection, atol=1e-12)
    assert scale == pytest.approx(3.5)
    np.testing.assert_allclose(shift, [10.0, -4.0], atol=1e-9)


def test_registry_offers_every_session_file_of_the_archive():
    from neurospatial.datasets._catalog import sessions

    entries = sessions("grieves_2020")
    assert len(entries) == 57
    for entry in entries:
        (member,) = entry.members
        assert member.path == f"{entry.session}.mat"
        assert entry.file_name == "Summary data.zip"
    (default,) = [e for e in entries if e.default]
    assert default.session == DEFAULT
    assert [e.session for e in entries if e.verified] == [DEFAULT]
    # Supplementary Table 1: 1064 cells over every rat's sessions.
    assert sum(entry.n_units for entry in entries) == 1064


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """Rat 926's tilted lattice, as the survey found it: 24 units, 50 Hz, 3-D."""
    rec = load("grieves_2020")
    assert_default_session(rec, session_entry("grieves_2020"))
    assert rec.n_units == 24
    assert rec.n_dims == 3
    assert rec.times.size == 182_099
    assert rec.sampling_rate == pytest.approx(50.0, rel=1e-3)
    assert "tilted lattice (config3.lattice_type 'diagonal')" in rec.apparatus
    np.testing.assert_array_equal(rec.epochs["lattice"], [[1816.0, 5458.0]])
    assert list(rec.epochs) == ["square1", "lattice", "square2"]
    assert rec.env.n_bins == 35 * 39 * 33
    assert rec.params["lattice_bars"].shape == (147, 2, 3)
    # The survey's evidence: position speed / pov 10.03; heading minus
    # movement in the positions' frame, without the 46,411 samples whose poh
    # and pov are both 0, mean 0.9 deg, spread 33 deg.
    assert "is 10.03 times the stored velocity pov" in rec.conversions[0].evidence
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert heading.basis == "inferred"
    assert "(here -poh)" in heading.text
    assert (
        "circular mean 0.9 deg and median absolute deviation 32.8" in heading.evidence
    )
    assert np.isnan(rec.headings).sum() == 46_411
    lost = next(c for c in rec.conversions if c.text.startswith("headings: NaN"))
    assert "exactly 0 in 48585 of 182099 samples (26.7%)" in lost.evidence
    assert "46411 of them (95.5%) with lattice.pov" in lost.evidence
    assert "not verified" not in rec.heading_source
    span = np.nanmax(rec.positions, axis=0) - np.nanmin(rec.positions, axis=0)
    np.testing.assert_allclose(span, [154.5, 170.8, 147.8], atol=0.1)
    # Every tracked sample lies in the authors' grid.
    assert np.all(rec.env.bin_at(rec.positions) >= 0)


@pytest.mark.realdata
def test_default_session_spikes_in_the_squares_are_kept():
    """The units' spikes cover all three parts; the lattice part is analysed."""
    rec = load("grieves_2020")
    spikes = np.concatenate(rec.spike_times)
    lattice = rec.params["analysis_epochs"]
    in_lattice = (spikes >= lattice[0, 0]) & (spikes < lattice[0, 1])
    assert 0.3 < in_lattice.mean() < 0.8
    assert rec.units["fires_in_epochs"].all()


@pytest.mark.realdata
@pytest.mark.parametrize(
    ("session", "listed_stop", "overhang"),
    [("894_21122017", 6651.0, 17.5), ("926_16042018", 6615.0, 94.6)],
)
def test_a_square_part_past_the_last_spike_loads_clipped(
    session, listed_stop, overhang
):
    """square2 ends after the sorted units stop firing: clipped, not refused."""
    rec = load("grieves_2020", session)
    assert rec.failed_checks == ()
    stop = max(float(t.max()) for t in rec.spike_times if t.size)
    assert rec.epochs["square2"][-1, 1] == stop
    assert listed_stop - stop == pytest.approx(overhang, abs=0.05)
    assert any(
        "epochs['square2']" in text and f", {listed_stop:.6g}] s ends" in text
        for text in rec.watch_outs
    )
