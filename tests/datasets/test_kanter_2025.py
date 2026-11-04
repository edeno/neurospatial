"""Kanter et al. (2025), EBRAINS 10.25493/GTVB-FQR: synthetic NWB files and the real session."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._checks import run_checks

DEFAULT = "29629_2024-Oct-29_15-05-13"
#: A foraging-only session (two probes, LEC and MEC) of the registry.
FORAGING = "27284_2021-Jun-18_11-01-41"
RATE = 120.0


def _path(n_samples=72_000, side=100.0, low=(-40.0, -70.0)):
    """A smooth path filling a ``side`` cm square from ``low``, at 120 Hz, 10-50 cm/s."""
    times = 40.0 + np.arange(n_samples) / RATE
    phase = 2 * np.pi * times
    unit = np.column_stack(
        [0.5 + 0.48 * np.sin(phase / 7.0), 0.5 + 0.48 * np.sin(phase / 11.0)]
    )
    return times, np.asarray(low) + side * unit


def _write_session(
    path,
    times,
    xy_cm,
    headings,
    *,
    epochs,
    tasks,
    side=100.0,
    frame="Bottom left of the arena",
):
    """An NWB file laid out as the archive's: cm stored as metres x 0.01, two probes."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import CompassDirection, Position, SpatialSeries
    from pynwb.file import Subject

    start = datetime.datetime(2024, 10, 29, 15, 5, tzinfo=datetime.timezone.utc)
    nwb = pynwb.NWBFile(
        "OF, OBJ",
        "synthetic",
        start,
        subject=Subject(subject_id="29629_Karri", species="Rattus norvegicus"),
    )
    behavior = nwb.create_processing_module("behavior", "position, direction")
    xyz = np.column_stack([xy_cm, np.full(len(times), 25.0)]).astype(np.float32)
    position = Position(name="position")
    position.add_spatial_series(
        SpatialSeries(
            name="animal_position",
            data=xyz,
            timestamps=times,
            reference_frame=frame,
            unit="meters",
            conversion=0.01,
        )
    )
    behavior.add(position)
    direction = CompassDirection(name="direction")
    direction.add_spatial_series(
        SpatialSeries(
            name="head_direction",
            data=np.mod(headings, 2 * np.pi)[:, None],
            timestamps=times,
            reference_frame="(x=1, y=0) unit vector as defined by recording system",
            unit="radians",
        )
    )
    behavior.add(direction)
    for name, kind in (
        ("task", "task"),
        ("arena", "arena"),
        ("arena_geometry", "shape"),
        ("arena_dimensions", "cm"),
    ):
        nwb.add_epoch_column(name, kind)
    for (a, b), task in zip(epochs, tasks, strict=True):
        nwb.add_epoch(
            start_time=a,
            stop_time=b,
            task=task,
            arena=f"{side:g}_square",
            arena_geometry="square",
            arena_dimensions=[side, side, 50.0],
        )
    device = nwb.create_device("probe")
    groups = [
        nwb.create_electrode_group(name, "", location, device)
        for name, location in (("probe_0", "LEC"), ("probe_1", "HP - CA1"))
    ]
    nwb.add_unit_column("cluster_quality", "quality")
    rng = np.random.default_rng(1)
    # Ids 0-2 on the LEC probe and 0-1 on the CA1 probe: 0 and 1 repeat.
    for group, ids in zip(groups, ((0, 1, 2), (0, 1)), strict=True):
        for unit in ids:
            spikes = np.sort(rng.uniform(times[0], times[-1], 300))
            nwb.add_unit(
                id=unit,
                spike_times=spikes,
                electrode_group=group,
                cluster_quality="good" if unit == 0 else "mua",
            )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def _moving_headings(times, positions, offset=0.0):
    """The direction of movement plus ``offset``, radians."""
    step = np.diff(
        positions, axis=0, append=positions[-1:] + positions[-1:] - positions[-2:-1]
    )
    return np.arctan2(step[:, 1], step[:, 0]) + offset


@pytest.fixture
def object_session(tmp_path):
    """A novel-object session: four epochs, headings 0.6 rad off the movement."""
    times, xy = _path()
    path = tmp_path / "session.nwb"
    edges = np.linspace(times[0], times[-1], 5)
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy, 0.6),
        epochs=np.column_stack([edges[:-1], edges[1:]]),
        tasks=["Novel object task"] * 4,
    )
    return path, times, xy


def test_loads_a_session_in_cm_with_area_unit_ids(object_session, session_entry):
    from neurospatial.datasets import kanter_2025

    path, times, xy = object_session
    entry = session_entry("kanter_2025")
    rec = run_checks(kanter_2025.load_session(entry, path), stated=entry.info.stated)
    np.testing.assert_allclose(rec.times, times)
    # Stored cm as metres x 0.01: back in cm, x and y only (float32 storage).
    np.testing.assert_allclose(rec.positions, xy, atol=1e-4)
    assert rec.positions.shape == (times.size, 2)
    assert rec.position_units == "cm"
    assert list(rec.units.index) == ["LEC-0", "LEC-1", "LEC-2", "CA1-0", "CA1-1"]
    assert list(rec.units["brain_region"]) == ["LEC"] * 3 + ["HP - CA1"] * 2
    assert list(rec.units["file_unit_id"]) == [0, 1, 2, 0, 1]
    assert rec.units["label"].isna().all()
    assert rec.brain_regions == ("LEC", "HP - CA1")
    assert np.all((rec.headings >= -np.pi) & (rec.headings <= np.pi))
    assert list(rec.epochs) == ["empty", "object_1", "object_2", "object_3"]
    assert all(rows.shape == (1, 2) for rows in rec.epochs.values())
    text = " ".join(rec.watch_outs)
    assert "repeats unit ids across its probes (5 rows, 3 distinct ids)" in text
    assert "No object positions" in text
    # The stated corner origin is contradicted by negative coordinates.
    assert "'Bottom left of the arena', but" in text
    assert rec.conversions[0].basis == "stated"
    assert "'meters', conversion 0.01" in rec.conversions[0].evidence


def test_a_heading_off_the_movement_is_kept_and_not_verified(
    object_session, session_entry
):
    from neurospatial.datasets import kanter_2025

    path, *_ = object_session
    rec = kanter_2025.load_session(session_entry("kanter_2025"), path)
    frame = next(c for c in rec.conversions if c.text.startswith("headings: allo"))
    assert frame.basis == "inferred"
    assert "circular mean +0.60 rad" in frame.evidence
    assert any(
        text.startswith("The heading frame is not verified by movement")
        for text in rec.watch_outs
    )


def test_a_heading_along_the_movement_is_verified(tmp_path, session_entry):
    from neurospatial.datasets import kanter_2025

    times, xy = _path(low=(0.0, 0.0))
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy),
        epochs=[[times[0], times[-1]]],
        tasks=["Foraging task"],
    )
    rec = kanter_2025.load_session(session_entry("kanter_2025", FORAGING), path)
    assert not any("heading frame is not verified" in t for t in rec.watch_outs)
    assert list(rec.epochs) == ["foraging"]
    # Positions from (0, 0): the stated corner origin holds, no watch-out,
    # and the square is the stated [0, 100]^2.
    assert not any("Bottom left" in t for t in rec.watch_outs)
    assert not any("No object positions" in t for t in rec.watch_outs)
    placement = next(c for c in rec.conversions if c.text.startswith("environment"))
    assert placement.basis == "stated"
    assert "[0, 100] x [0, 100] cm, its bottom left at the file's origin" in (
        placement.text
    )
    np.testing.assert_allclose(rec.env.dimension_ranges, [[0.0, 100.0]] * 2)


def test_samples_past_the_far_walls_contradict_the_stated_origin(
    tmp_path, session_entry
):
    """No negative coordinate, but y and x reach 108 cm in a stated 100 cm square."""
    from neurospatial.datasets import kanter_2025

    times, xy = _path(low=(10.0, 10.0))
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy),
        epochs=[[times[0], times[-1]]],
        tasks=["Foraging task"],
    )
    rec = kanter_2025.load_session(session_entry("kanter_2025", FORAGING), path)
    (claim,) = [t for t in rec.watch_outs if "'Bottom left of the arena', but" in t]
    assert "have a coordinate below 0 or above the stated 100 cm" in claim
    placement = next(c for c in rec.conversions if c.text.startswith("environment"))
    assert placement.basis == "inferred"
    assert "of the tracked samples contradict" in placement.evidence
    np.testing.assert_allclose(
        np.mean(rec.env.dimension_ranges, axis=1), xy.min(0) / 2 + xy.max(0) / 2, atol=1
    )


@pytest.mark.parametrize(
    ("headings", "reason"),
    [
        (
            # The offset drifts from -1 to +1 rad, its mean near 0.
            lambda times, xy: _moving_headings(times, xy)
            + np.linspace(-1.0, 1.0, times.size),
            "the window means' circular SD is beyond 11.5 degrees",
        ),
        (
            # A constant 0 offset with noise of +-100 degrees: a spread of 50.
            lambda times, xy: _moving_headings(times, xy)
            + np.radians(np.random.default_rng(2).uniform(-100, 100, times.size)),
            "the spread is 45 degrees or more",
        ),
    ],
    ids=["drift", "spread"],
)
def test_a_drifting_or_spread_heading_is_not_verified(
    tmp_path, session_entry, headings, reason
):
    """The shared rule: |mean| <= 0.2 rad, spread < 45 degrees, and no drift."""
    from neurospatial.datasets import kanter_2025

    times, xy = _path(low=(0.0, 0.0))
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        xy,
        headings(times, xy),
        epochs=[[times[0], times[-1]]],
        tasks=["Foraging task"],
    )
    rec = kanter_2025.load_session(session_entry("kanter_2025", FORAGING), path)
    frame = next(c for c in rec.conversions if c.text.startswith("headings: allo"))
    assert f"{reason}" in frame.evidence
    assert "so the frame is not verified" in frame.evidence
    (finding,) = [t for t in rec.watch_outs if "heading frame is not verified" in t]
    if reason.startswith("the window"):
        assert "smeared by the drift" in finding
        assert "per epoch or per window" in finding
        assert "do not depend on the frame" not in finding
    else:
        assert "smeared" not in finding
        assert "with a constant offset, tuning-curve shapes" in finding


def test_an_epoch_past_the_recording_is_clipped_and_reported(tmp_path, session_entry):
    """An epoch ending 20 s after the last sample and spike is clipped to them."""
    from neurospatial.datasets import kanter_2025
    from neurospatial.datasets._checks import run_checks

    times, xy = _path(low=(0.0, 0.0))
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy),
        epochs=[[times[0], times[-1] + 20.0]],
        tasks=["Foraging task"],
    )
    entry = session_entry("kanter_2025", FORAGING)
    rec = run_checks(kanter_2025.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    stop = max(times[-1], max(train.max() for train in rec.spike_times))
    np.testing.assert_allclose(rec.epochs["foraging"], [[times[0], stop]])
    assert any(
        text.startswith("1 epoch row reaches past the recording")
        and "epochs['foraging']" in text
        for text in rec.watch_outs
    )
    clip = next(c for c in rec.conversions if c.text.startswith("epochs: rows"))
    assert clip.basis == "inferred"
    assert "the file's epochs table lists them so" in clip.evidence


def test_place_square_holds_the_most_samples():
    """A 100 cm square over a 100 cm box's samples, not their 150 cm range."""
    from neurospatial.datasets.kanter_2025 import place_square

    rng = np.random.default_rng(0)
    box = rng.uniform(0.0, 100.0, (20_000, 2))
    # 5% of samples on a ledge above the box (a rat on top of the wall).
    ledge = np.column_stack(
        [rng.uniform(0.0, 100.0, 1_000), rng.uniform(100.0, 150.0, 1_000)]
    )
    centre, n_inside = place_square(np.vstack([box, ledge]), 100.0)
    np.testing.assert_allclose(centre, [50.0, 50.0], atol=1.0)
    assert n_inside >= 20_000
    # Samples narrower than the square: the square is centred on their range.
    centre, n_inside = place_square(rng.uniform(10.0, 70.0, (5_000, 2)), 100.0)
    np.testing.assert_allclose(centre, [40.0, 40.0], atol=0.5)
    assert n_inside == 5_000


def test_a_novel_object_session_without_four_epochs_is_refused(tmp_path, session_entry):
    from neurospatial.datasets import kanter_2025

    times, xy = _path()
    path = tmp_path / "session.nwb"
    edges = np.linspace(times[0], times[-1], 4)
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy),
        epochs=np.column_stack([edges[:-1], edges[1:]]),
        tasks=["Novel object task"] * 3,
    )
    with pytest.raises(ValueError, match="novel-object session with 3 epochs"):
        kanter_2025.load_session(session_entry("kanter_2025"), path)


def test_a_session_outside_its_stated_arena_is_described(tmp_path, session_entry):
    """Samples spanning 100 cm in a stated 150 cm arena: a watch-out, values kept."""
    from neurospatial.datasets import kanter_2025

    times, xy = _path(low=(0.0, 0.0))
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        xy,
        _moving_headings(times, xy),
        epochs=[[times[0], times[-1]]],
        tasks=["Foraging task"],
        side=150.0,
    )
    rec = kanter_2025.load_session(session_entry("kanter_2025", FORAGING), path)
    assert any("does not match the 150 cm square arena" in t for t in rec.watch_outs)
    np.testing.assert_allclose(rec.positions, xy, atol=1e-4)


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """Rat 29629's novel-object session, as the survey found it."""
    rec = load("kanter_2025")
    assert_default_session(rec, session_entry("kanter_2025"))
    assert rec.units["brain_region"].value_counts().to_dict() == {
        "HP - CA1": 224,
        "LEC": 156,
    }
    assert (rec.units["cluster_quality"] == "good").sum() == 125
    assert rec.units.index.is_unique
    assert rec.sampling_rate == pytest.approx(120.0, rel=1e-3)
    durations = {k: float(np.diff(v[0])[0]) for k, v in rec.epochs.items()}
    assert durations == pytest.approx(
        {"empty": 605.0, "object_1": 450.0, "object_2": 523.0, "object_3": 452.0}
    )
    np.testing.assert_allclose(np.ptp(rec.positions, axis=0), [98.4, 97.7], atol=0.1)
    assert rec.env.units == "cm"
    # Every sample lies in the placed 100 cm square.
    assert not any("outside the declared environment" in t for t in rec.watch_outs)
    frame = next(c for c in rec.conversions if c.text.startswith("headings: allo"))
    assert "circular mean -0.33 rad, median spread 17 degrees" in frame.evidence
    assert "spread of 81 degrees" in frame.evidence
    assert "17 windows" in frame.evidence
    # The offset drifts (window means -0.62 to +2.33 rad, circular SD 31 deg).
    (finding,) = [t for t in rec.watch_outs if "heading frame is not verified" in t]
    assert "smeared by the drift" in finding


@pytest.mark.realdata
@pytest.mark.parametrize(
    ("session", "name", "listed_stop"),
    [
        ("27284_2021-Jun-16_14-33-11", "foraging", 1528.0),
        ("27285_2021-Jun-30_13-59-11", "foraging", 1639.0),
        ("27963_2022-Feb-08_21-14-27", "foraging", 1258.0),
        ("27963_2022-Feb-10_11-06-56", "object_3", 1912.0),
    ],
)
def test_an_epoch_ending_after_the_data_loads_clipped(session, name, listed_stop):
    """Four sessions whose epoch ends 14.7 to 31.9 s after the data: clipped, not refused."""
    rec = load("kanter_2025", session)
    assert rec.failed_checks == ()
    stop = max(rec.times[-1], max(t.max() for t in rec.spike_times if t.size))
    assert rec.epochs[name][-1, 1] == stop
    assert 14.5 <= listed_stop - stop <= 32.0
    assert any(
        f"epochs['{name}'] [" in text and f", {listed_stop:.6g}] s ends" in text
        for text in rec.watch_outs
    )


@pytest.mark.realdata
def test_a_session_whose_tracking_stopped_is_refused_unless_accepted():
    """26965 Dec-08 11:29: tracking ends at 709 s, spikes and the epoch run to 906 s."""
    from neurospatial.datasets import DatasetCheckError

    session = "26965_2020-Dec-08_11-29-48"
    with pytest.raises(DatasetCheckError, match="outside the tracked span"):
        load("kanter_2025", session)
    rec = load("kanter_2025", session, accept_failed_checks=("clock",))
    assert rec.failed_checks == ("clock",)


def test_a_session_too_slow_to_compare_heading_says_so(tmp_path, session_entry):
    """A path under 20 cm/s throughout: the frame is not compared, and not verified."""
    from neurospatial.datasets import kanter_2025

    times, xy = _path()
    slow = xy[0] + (xy - xy[0]) / 10.0  # a tenth of the speed: under 5 cm/s
    path = tmp_path / "session.nwb"
    _write_session(
        path,
        times,
        slow,
        _moving_headings(times, slow),
        epochs=[[times[0], times[-1]]],
        tasks=["Foraging task"],
    )
    rec = kanter_2025.load_session(session_entry("kanter_2025", FORAGING), path)
    assert any(
        t.startswith("The heading frame is not verified by movement: the rat ran")
        for t in rec.watch_outs
    )
