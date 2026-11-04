"""Vollan et al. (2025), EBRAINS 10.25493/R5FR-EDG: synthetic Dsession files and the real ones."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial.datasets import DatasetCheckError, load, vollan_2025
from neurospatial.datasets._checks import run_checks

DEFAULT = "of_26018_2"
DT = 0.01


def _walk_m(n: int, seed: int, half: float = 0.72) -> np.ndarray:
    """A reflected random walk filling a square of side ``2 * half`` m centred at 0."""
    rng = np.random.default_rng(seed)
    side = 2 * half
    xy = np.mod(rng.normal(0, 0.01, (n, 2)).cumsum(axis=0), 2 * side)
    return np.where(xy > side, 2 * side - xy, xy) - half


def _speed_filtered(n: int, seed: int) -> np.ndarray:
    """Sample times every 0.01 s from 100 s, with 40 gaps of 0.3 to 3 s removed."""
    rng = np.random.default_rng(seed)
    keep = np.ones(n, dtype=bool)
    for start in rng.choice(np.arange(100, n - 400), 40, replace=False):
        keep[start : start + rng.integers(30, 300)] = False
    return 100.0 + np.flatnonzero(keep) * DT


def _units(n_units: int, times: np.ndarray, seed: int, *, modules=None, hc=False):
    """A ``units.<area>`` struct array; without ``modules`` the module fields are left out."""
    rng = np.random.default_rng(seed)
    fields = [("id", object), ("ks2Label", object), ("spikeTimes", object)]
    if not hc:
        fields += [("cellType", object), ("acorrCluId", object), ("isGrid", object)]
    table = np.zeros((n_units,), dtype=fields)
    for k in range(n_units):
        spikes = np.sort(rng.choice(times, 200, replace=False) + 0.001)
        row = [f"1_{k:04d}", "good" if k % 2 else "mua", spikes]
        if not hc:
            module = 0 if modules is None else modules[k]
            row += ["bursty" if module else "nongrid", float(module), float(module > 0)]
        table[k] = tuple(row)
    return table


def _write_session(
    path,
    times,
    xy_m,
    *,
    kind="open_field",
    hd=None,
    listed=None,
    modules=(0, 1, 1, 2),
    sizes=(2, 1),
    hc_only=False,
    seed=0,
):
    """A ``Dsession`` file laid out as mat_files.md describes, through scipy."""
    import scipy.io as sio

    if hd is None:
        step = np.diff(xy_m, axis=0, append=2 * xy_m[-1:] - xy_m[-2:-1])
        hd = np.arctan2(step[:, 1], step[:, 0])
    if listed is None:
        listed = [[times[0] - 0.4, times[-1] + 100.0]]
    sessions = np.zeros(
        (1,),
        dtype=[
            ("type", object),
            ("name", object),
            ("startTime", object),
            ("endTime", object),
            ("validTimeRanges", object),
        ],
    )
    ranges = np.asarray(listed, dtype=float)
    sessions[0] = (kind, f"{kind}_1", ranges[0, 0], ranges[-1, 1], ranges)
    if hc_only:
        units = {"mec": np.zeros((0, 0)), "hc": _units(3, times, seed + 1, hc=True)}
        clusters = np.zeros((0, 0))
    else:
        unit_table = _units(len(modules), times, seed + 1, modules=modules)
        units = {"mec": unit_table, "hc": np.zeros((0, 0))}
        grid = np.zeros(
            (len(sizes),),
            dtype=[
                ("nUnits", object),
                ("unitIds", object),
                ("medianGridScore", object),
                ("gridScore", object),
                ("gridSpacingMeters", object),
            ],
        )
        for k, size in enumerate(sizes, start=1):
            ids = np.array(
                [f"1_{i:04d}" for i, m in enumerate(modules) if m == k], dtype=object
            )
            grid[k - 1] = (float(size), ids, 1.0 / k, 1.1 / k, 0.4 * k)
        clusters = {"grid": grid, "sessionType": "of"}
    session = {
        "t": times,
        "x": xy_m[:, 0],
        "y": xy_m[:, 1],
        "hd": hd,
        "dt": DT,
        "units": units,
        "sessions": sessions,
        "unitAcorrClus": clusters,
    }
    sio.savemat(path, {"Dsession": session}, do_compression=True)
    return sessions


@pytest.fixture(scope="module")
def box_session(tmp_path_factory):
    times = _speed_filtered(60_000, seed=3)
    xy = _walk_m(times.size, seed=4)
    path = tmp_path_factory.mktemp("vollan") / "of_26018_2.mat"
    _write_session(path, times, xy, kind="open_field")
    return path, times, xy


def _loaded(path, session_entry, session=DEFAULT):
    entry = session_entry("vollan_2025", session)
    return run_checks(vollan_2025.load_session(entry, path), stated=entry.info.stated)


def test_converts_metres_to_cm_and_keeps_the_gaps(box_session, session_entry):
    path, times, xy = box_session
    rec = _loaded(path, session_entry)
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_allclose(rec.positions, xy * 100.0)
    assert rec.position_units == "cm"
    assert rec.conversions[0].basis == "stated"
    assert rec.failed_checks == ()
    # Gaps counted without the loader's code: steps longer than two samples.
    steps = np.diff(times)
    n_gaps = int(np.sum(steps > 2 * DT))
    assert any(
        text.startswith(f"times has {n_gaps} gaps longer than 0.02 s")
        for text in rec.watch_outs
    )
    assert rec.params["max_gap"] == pytest.approx(0.015)
    assert rec.params["min_speed"] is None
    assert rec.params["bin_size"] == 2.5


def test_units_carry_the_authors_modules(box_session, session_entry):
    path, *_ = box_session
    rec = _loaded(path, session_entry)
    assert list(rec.units.index) == ["1_0000", "1_0001", "1_0002", "1_0003"]
    assert list(rec.units["module"]) == [0, 1, 1, 2]
    assert list(rec.units["is_grid"]) == [False, True, True, True]
    assert list(rec.units["label"]) == ["nongrid", "bursty", "bursty", "bursty"]
    assert list(rec.units["brain_region"]) == ["MEC-parasubiculum"] * 4
    assert list(rec.units["ks2_label"]) == ["mua", "good", "mua", "good"]
    np.testing.assert_array_equal(rec.params["module"], [0, 1, 1, 2])
    np.testing.assert_array_equal(rec.params["module_ids"], [1, 2])
    np.testing.assert_allclose(rec.params["module_median_grid_score"], [1.0, 0.5])
    np.testing.assert_allclose(rec.params["module_spacing_cm"], [40.0, 80.0])


def test_listed_session_is_clipped_to_the_recording(box_session, session_entry):
    path, times, _ = box_session
    rec = _loaded(path, session_entry)
    spikes = np.concatenate(rec.spike_times)
    stop = max(times[-1], spikes.max())
    # The start, 0.4 s before the first sample, is within the clock check's
    # slack and kept; the end, 100 s after the data, is clipped.
    np.testing.assert_allclose(rec.epochs["open_field_1"], [[times[0] - 0.4, stop]])
    clip = next(c for c in rec.conversions if c.text.startswith("epochs: rows"))
    assert clip.basis == "inferred"
    assert any(
        text.startswith("1 epoch row reaches past the recording")
        and "s ends 100 s after it" in text
        and "starts" not in text
        for text in rec.watch_outs
    )
    held = times.size * DT
    assert any(
        text.startswith("The samples end 100 s before the listed end")
        and f"{held:g} s at dt" in text
        for text in rec.watch_outs
    )


def test_open_field_is_the_stated_square_on_the_samples(box_session, session_entry):
    path, *_ = box_session
    rec = _loaded(path, session_entry)
    assert rec.env_kind == "declared"
    assert rec.apparatus == "150 x 150 cm open-field box"
    placement = rec.conversions[-1]
    assert placement.basis == "inferred"
    assert "the stated 150 x 150 cm square" in placement.text
    (low_x, high_x), (low_y, high_y) = rec.env.dimension_ranges
    assert high_x - low_x == pytest.approx(150.0, abs=3.0)
    assert abs((low_x + high_x) / 2) < 3.0
    assert abs((low_y + high_y) / 2) < 3.0
    assert not any("from the file's (0, 0)" in text for text in rec.watch_outs)


def test_lit_and_dark_sessions_are_the_circular_arena(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=5)
    rng = np.random.default_rng(6)
    radius = 0.7 * np.sqrt(rng.uniform(0, 1, times.size))
    angle = rng.uniform(-np.pi, np.pi, times.size)
    xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    path = tmp_path / "of.mat"
    _write_session(path, times, xy, kind="open_field_darkness")
    rec = vollan_2025.load_session(session_entry("vollan_2025"), path)
    assert rec.apparatus == "circular open arena, 150 cm in diameter"
    placement = rec.conversions[-1]
    assert placement.text.startswith("environment: a circle 150 cm in diameter")
    assert placement.basis == "inferred"
    (low_x, high_x), _ = rec.env.dimension_ranges
    assert high_x - low_x == pytest.approx(150.0, abs=3.0)


def test_linear_track_is_a_200_cm_track_on_the_principal_axis(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=7)
    rng = np.random.default_rng(8)
    along = 1.0 * np.sin(np.linspace(0, 40 * np.pi, times.size))  # +-1 m
    direction = np.array([np.cos(0.7), np.sin(0.7)])
    xy = np.outer(along, direction) + rng.normal(0, 0.02, (times.size, 2))
    path = tmp_path / "lt.mat"
    _write_session(path, times, xy, kind="linear_track")
    rec = vollan_2025.load_session(session_entry("vollan_2025", "lt_27764_1"), path)
    assert rec.env.is_linearized_track
    assert rec.apparatus == "200 cm linear track"
    assert rec.track_width_quantile == pytest.approx(0.99)
    track = rec.conversions[-2]
    assert "a straight 200 cm track" in track.text
    assert f"at {np.degrees(0.7):.1f} deg" in track.text
    # The modules' scores are the open field's, said so on the track.
    assert any("defined on the rat's open-field session" in t for t in rec.watch_outs)


def test_heading_off_the_path_is_kept_and_reported(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=9)
    xy = _walk_m(times.size, seed=10)
    hd = np.random.default_rng(11).uniform(-np.pi, np.pi, times.size)
    path = tmp_path / "of.mat"
    _write_session(path, times, xy, hd=hd)
    rec = vollan_2025.load_session(session_entry("vollan_2025"), path)
    np.testing.assert_allclose(rec.headings, hd)
    assert any(
        t.startswith("The heading frame is not verified by movement")
        for t in rec.watch_outs
    )


def test_hippocampus_only_file_has_no_modules(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=12)
    path = tmp_path / "of.mat"
    _write_session(path, times, _walk_m(times.size, seed=13), hc_only=True)
    rec = vollan_2025.load_session(session_entry("vollan_2025"), path)
    assert list(rec.units["brain_region"]) == ["hippocampus"] * 3
    assert rec.units["label"].isna().all()
    assert list(rec.units["module"]) == [0, 0, 0]
    assert rec.params["module_ids"].size == 0
    assert any(t.startswith("The file holds no grid modules") for t in rec.watch_outs)


def test_a_session_table_just_before_the_samples_gives_no_epoch(
    tmp_path, session_entry
):
    """The listed session ends 10 s before the first sample, within a recording length."""
    times = _speed_filtered(30_000, seed=14)
    path = tmp_path / "of.mat"
    _write_session(
        path, times, _walk_m(times.size, seed=15), listed=[[0.0, times[0] - 10.0]]
    )
    entry = session_entry("vollan_2025")
    rec = run_checks(vollan_2025.load_session(entry, path), stated=entry.info.stated)
    assert dict(rec.epochs) == {}
    assert rec.failed_checks == ()
    text = " ".join(rec.watch_outs)
    assert "100.0% of the samples lie outside every listed session" in text
    assert "s lies wholly outside it, dropped" in text


def test_a_session_table_a_recording_length_before_the_samples_is_refused(
    tmp_path, session_entry
):
    """As in of_25843_5 and of_26820_2: the listed session ends where the samples begin."""
    times = _speed_filtered(30_000, seed=14)
    length = times[-1] - times[0]
    listed = [[times[0] - 2.0 * length, times[0] - 0.01]]
    path = tmp_path / "of.mat"
    _write_session(path, times, _walk_m(times.size, seed=15), listed=listed)
    entry = session_entry("vollan_2025")
    rec = vollan_2025.load_session(entry, path)
    np.testing.assert_allclose(rec.epochs["open_field_1"], listed)
    with pytest.raises(DatasetCheckError, match="no epoch holds a tracking sample"):
        run_checks(rec, stated=entry.info.stated)
    accepted = run_checks(
        rec, stated=entry.info.stated, accept_failed_checks=("clock",)
    )
    assert accepted.failed_checks == ("clock",)
    np.testing.assert_allclose(accepted.epochs["open_field_1"], listed)


def test_an_arena_off_the_files_centre_is_reported(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=16)
    xy = _walk_m(times.size, seed=17) + np.array([0.0, 0.1])
    path = tmp_path / "of.mat"
    _write_session(path, times, xy)
    rec = vollan_2025.load_session(session_entry("vollan_2025"), path)
    assert any("from the file's (0, 0)" in text for text in rec.watch_outs)
    np.testing.assert_allclose(rec.positions, xy * 100.0)


def test_more_module_units_than_the_module_holds_is_refused(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=18)
    path = tmp_path / "of.mat"
    _write_session(path, times, _walk_m(times.size, seed=19), sizes=(1, 1))
    with pytest.raises(ValueError, match="exceed the module sizes"):
        vollan_2025.load_session(session_entry("vollan_2025"), path)


def test_an_unknown_session_type_is_refused(tmp_path, session_entry):
    times = _speed_filtered(30_000, seed=20)
    path = tmp_path / "of.mat"
    _write_session(path, times, _walk_m(times.size, seed=21), kind="m_maze")
    with pytest.raises(ValueError, match="name no arena the loader knows"):
        vollan_2025.load_session(session_entry("vollan_2025"), path)


@pytest.fixture(scope="module")
def default_session():
    return load("vollan_2025")


@pytest.mark.realdata
def test_loads_default_session(default_session, session_entry, assert_default_session):
    """Rat 26018's lit open field: 644 MEC units, three grid modules, 100 Hz."""
    rec = default_session
    assert_default_session(rec, session_entry("vollan_2025"))
    assert rec.times.size == 60_000
    assert rec.sampling_rate == pytest.approx(100.0, rel=1e-6)
    np.testing.assert_allclose([rec.times[0], rec.times[-1]], [3449.02, 5047.69])
    assert np.bincount(rec.params["module"]).tolist() == [411, 48, 90, 95]
    np.testing.assert_allclose(
        rec.params["module_median_grid_score"], [1.114, 0.986, 0.685], atol=5e-4
    )
    np.testing.assert_allclose(np.ptp(rec.positions, axis=0), [141.9, 140.2], atol=0.1)
    assert rec.apparatus == "circular open arena, 150 cm in diameter"
    # The listed end (5400 s) is clipped to the last spike, just after the
    # last sample (5047.69 s); the listed start is kept.
    spikes = np.concatenate(rec.spike_times)
    np.testing.assert_array_equal(
        rec.epochs["open_field_light_1"], [[3449.0, spikes.max()]]
    )
    assert spikes.max() == pytest.approx(5047.69497, abs=1e-5)
    heading = next(c for c in rec.conversions if c.text.startswith("headings"))
    assert "circular mean of +0.2 deg" in heading.evidence


@pytest.mark.realdata
def test_loads_the_linear_track_session(session_entry):
    """Rat 27764's 200 cm track: 389 units, 84 in the open field's one grid module."""
    rec = load("vollan_2025", "lt_27764_1")
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == session_entry("vollan_2025", "lt_27764_1").n_units == 389
    assert int(np.sum(rec.params["module"] == 1)) == 84
    np.testing.assert_allclose([rec.times[0], rec.times[-1]], [7227.56, 8412.35])
    assert rec.env.is_linearized_track
    assert rec.apparatus == "200 cm linear track"


@pytest.mark.realdata
def test_speed_filtered_times_warn_unless_max_gap_excludes_them(default_session):
    """The authors' 5 cm/s filter looks boolean-masked to compute_spatial_rates."""
    from neurospatial import compute_spatial_rates

    rec = default_session
    trains = rec.spike_times[:3]
    with pytest.warns(UserWarning, match="boolean-masked"):
        compute_spatial_rates(rec.env, trains, rec.times, rec.positions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_spatial_rates(
            rec.env, trains, rec.times, rec.positions, max_gap=rec.params["max_gap"]
        )


@pytest.mark.realdata
def test_module_grid_scores_against_the_authors(default_session):
    """The attempted worked result: modules 1 and 2 agree, module 3 does not."""
    from neurospatial import compute_spatial_rates
    from neurospatial.encoding import grid_score, spatial_autocorrelation

    rec = default_session
    module = rec.params["module"]
    units = np.flatnonzero(module > 0)
    rates = compute_spatial_rates(
        rec.env,
        [rec.spike_times[i] for i in units],
        rec.times,
        rec.positions,
        method="binned",
        bandwidth=5.0,
        max_gap=rec.params["max_gap"],
    )
    scores = np.array(
        [
            grid_score(spatial_autocorrelation(rec.env, rate))
            for rate in np.asarray(rates.firing_rates)
        ]
    )
    medians = {
        int(k): float(np.nanmedian(scores[module[units] == k]))
        for k in rec.params["module_ids"]
    }
    authors = dict(
        zip(
            rec.params["module_ids"].tolist(),
            rec.params["module_median_grid_score"].tolist(),
            strict=True,
        )
    )
    record = vollan_2025.WORKED_RESULT
    assert record["paper"] == round(authors[3], 3)
    assert record["attempted"]["value"] == round(medians[3], 3)
    for k in (1, 2):
        assert abs(medians[k] - authors[k]) <= 0.15
        assert (
            f"{medians[k]:.3f} against {authors[k]:.3f}"
            in (record["attempted"]["criterion"])
        )
    # The criterion fails on module 3, the largest spacing.
    assert abs(medians[3] - authors[3]) > 0.15
