"""Grosmark & Buzsáki (2016): synthetic files in the OSF and DANDI 000044 layouts, and the real ones."""

from __future__ import annotations

import datetime
import re
from dataclasses import replace

import numpy as np
import pytest

from neurospatial.datasets import DatasetCheckError, load
from neurospatial.datasets._catalog import sessions
from neurospatial.datasets._checks import run_checks

DEFAULT = "Achilles_10252013"
DANDI_COPY = "sub-Achilles_ses-Achilles-10252013"
DT = 0.0256
#: DANDI's epochs of the default session, [start, stop) in s.
PRE, MAZE, POST = (0.0, 18079.5), (18079.5, 20147.0), (20147.0, 34861.1032)


def _entry(session=DEFAULT):
    return next(
        entry for entry in sessions("grosmark_2016") if entry.session == session
    )


def _nwbfile(pynwb, identifier):
    return pynwb.NWBFile(
        "synthetic",
        identifier,
        datetime.datetime(2013, 10, 25, tzinfo=datetime.timezone.utc),
    )


def _add_units(nwb, rng, *, cell_type, span):
    nwb.add_unit_column("location", "the anatomical location of this unit")
    nwb.add_unit_column("cell_type", "the cell type of the unit")
    for location, kind in (("lCA1", cell_type[0]), ("rCA1", cell_type[1])):
        spikes = np.sort(rng.uniform(*span, 3000))
        nwb.add_unit(spike_times=spikes, location=location, cell_type=kind)


def _shuttle(n, low, high):
    """Back-and-forth runs between ``low`` and ``high``, a triangle wave."""
    phase = np.mod(np.arange(n) / 300.0, 2.0)
    return low + (high - low) * np.minimum(phase, 2.0 - phase)


def _write_osf(path, *, late=111.0):
    """A file in the NeMoS tutorial layout: a 1-D position TimeSeries in cm.

    Tracking starts ``late`` seconds into the maze epoch; spikes run from
    the start of PRE to the end of POST.
    """
    pynwb = pytest.importorskip("pynwb")
    nwb = _nwbfile(pynwb, DEFAULT)
    times = np.arange(MAZE[0] + late, MAZE[1] - 5.0, DT)
    values = _shuttle(times.size, 5.4, 165.3)
    nwb.add_acquisition(
        pynwb.TimeSeries(
            name="position",
            data=values,
            timestamps=times,
            unit="cm",
            description="The linearized position of the animal.",
        )
    )
    starts = times[[10, 400]]
    for start, stop in zip(starts, times[[200, 600]], strict=True):
        nwb.add_trial(start_time=start, stop_time=stop)
    rng = np.random.default_rng(0)
    _add_units(nwb, rng, cell_type=("pE", "pI"), span=(1.0, POST[1] - 1.0))
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return times, values


def test_loads_the_osf_layout(tmp_path):
    """1-D positions as stored, DANDI's epochs, and a clock check that passes.

    Spikes fall in the 111 s of the maze epoch before the first position
    sample; the clock check passes because that stretch is an untracked
    epoch, ``maze_before_track``.
    """
    from neurospatial.datasets import grosmark_2016

    path = tmp_path / "Achilles_10252013.nwb"
    times, values = _write_osf(path)
    entry = _entry()
    loaded = grosmark_2016.load_session(entry, path)
    rec = run_checks(loaded, stated=entry.info.stated)
    assert rec.failed_checks == ()
    # Without that epoch, the spikes before tracking fail the clock check.
    epochs = {k: v for k, v in loaded.epochs.items() if k != "maze_before_track"}
    with pytest.raises(DatasetCheckError, match="clock"):
        run_checks(replace(loaded, epochs=epochs), stated=entry.info.stated)
    np.testing.assert_array_equal(rec.times, times)
    np.testing.assert_array_equal(rec.positions, values[:, None])
    assert rec.position_units == "cm"
    np.testing.assert_array_equal(rec.epochs["pre"], [PRE])
    np.testing.assert_array_equal(rec.epochs["maze"], [MAZE])
    np.testing.assert_array_equal(rec.epochs["post"], [POST])
    np.testing.assert_array_equal(
        rec.epochs["maze_before_track"], [[MAZE[0], times[0]]]
    )
    first = np.concatenate(rec.spike_times)
    assert np.any((first > MAZE[0]) & (first < times[0] - 10.0))
    # The stated 160 cm stem as a 1-D track, centred on the samples' range.
    assert rec.env.n_dims == 1
    assert rec.env.n_bins == 80
    assert np.all(np.asarray(rec.env.bin_at(rec.positions)) >= 0)
    np.testing.assert_allclose(
        rec.env.to_linear(np.array([[5.35], [165.35]])), [0.0, 160.0], atol=1e-9
    )
    assert rec.trials is not None
    np.testing.assert_array_equal(rec.trials.start, times[[10, 400]])
    assert list(rec.units["brain_region"]) == ["lCA1", "rCA1"]
    assert list(rec.units["label"]) == ["pE", "pI"]
    epochs = next(c for c in rec.conversions if "PREEpoch" in c.text)
    assert epochs.basis == "inferred"
    assert "spike times of 137 units are identical" in epochs.evidence
    assert rec.license == "not stated"
    assert rec.provenance.archive == "OSF hu5ma"
    assert rec.headings is None


def test_osf_layout_without_a_known_dandi_copy_is_refused(tmp_path):
    from neurospatial.datasets import grosmark_2016

    path = tmp_path / "other.nwb"
    _write_osf(path)
    entry = replace(_entry(), session="Buddy_06272013")
    with pytest.raises(ValueError, match="no DANDI 000044 copy is known"):
        grosmark_2016.load_session(entry, path)


def _write_dandi(path, *, track, positions_cm, linearized_cm, start=MAZE[0]):
    """A file in the DANDI 000044 layout: metres at a fixed rate, from ``start``."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.epoch import TimeIntervals

    nwb = _nwbfile(pynwb, "dandi")
    nwb.subject = pynwb.file.Subject(subject_id="Achilles", species="Rattus norvegicus")
    module = nwb.create_processing_module("behavior", "behavior")
    for kind, data in (
        ("Position", positions_cm / 100.0),
        ("LinearizedPosition", linearized_cm[:, None] / 100.0),
    ):
        container = Position(name=f"{track}{kind}")
        container.add_spatial_series(
            SpatialSeries(
                name=f"{track}{kind.removesuffix('Position')}SpatialSeries"
                if kind == "Position"
                else f"{track}LinearizedTimeSeries",
                data=data,
                rate=1 / DT,
                starting_time=start,
                unit="meters",
                reference_frame="unknown",
            )
        )
        module.add(container)
    stop = start + len(positions_cm) * DT
    states = TimeIntervals(name="states", description="sleep states")
    states.add_column(name="label", description="the state")
    for row in (
        (0.0, 5000.0, "Non-REM"),
        (5000.0, 6000.0, "REM"),
        (6000.0, stop + 10.0, "Awake"),
        (stop + 10.0, stop + 4000.0, "Non-REM"),
    ):
        states.add_row(start_time=row[0], stop_time=row[1], label=row[2])
    module.add(states)
    nwb.add_epoch_column("label", "the epoch")
    for row in ((0.0, start, "PREEpoch"), (start, stop, "MazeEpoch")):
        nwb.add_epoch(start_time=row[0], stop_time=row[1], label=row[2])
    nwb.add_epoch(start_time=stop, stop_time=stop + 4000.0, label="POSTEpoch")
    rng = np.random.default_rng(1)
    _add_units(
        nwb, rng, cell_type=("excitatory", "inhibitory"), span=(1.0, stop + 3999.0)
    )
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


#: The planted linear track: its axis at 30 degrees, the stem's middle at
#: (80, 60) cm, the file's linearized position 0-160 cm over 154 cm of it.
AXIS = np.array([np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
MIDDLE = np.array([80.0, 60.0])
COVERED = 154.0


def _linear_session():
    """30 s off the track, then runs along the stem and into the reward areas."""
    rng = np.random.default_rng(2)
    off = MIDDLE + np.array([0.0, 150.0]) + rng.normal(0.0, 5.0, (int(30 / DT), 2))
    along = _shuttle(30000, -92.0, 92.0)
    lateral = rng.normal(0.0, 2.0, along.size)
    normal = np.array([-AXIS[1], AXIS[0]])
    on = MIDDLE + np.outer(along, AXIS) + np.outer(lateral, normal)
    positions = np.vstack([off, on])
    linear = np.full(len(positions), np.nan)
    inside = np.abs(along) <= COVERED / 2
    linear[len(off) :][inside] = (along[inside] + COVERED / 2) * 160.0 / COVERED
    return positions, linear, along


def test_loads_a_dandi_linear_session(tmp_path):
    """cm from metres, epochs and sleep states, and the planted track recovered."""
    from neurospatial.datasets import grosmark_2016

    positions, linear, along = _linear_session()
    path = tmp_path / "linear.nwb"
    _write_dandi(
        path, track="1.6mLinearMaze", positions_cm=positions, linearized_cm=linear
    )
    entry = _entry("sub-Buddy_ses-Buddy-06272013")
    rec = run_checks(grosmark_2016.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    np.testing.assert_allclose(rec.positions, positions, rtol=1e-12)
    np.testing.assert_allclose(rec.linearized, linear, rtol=1e-12)
    np.testing.assert_allclose(rec.times[0], MAZE[0])
    assert np.median(np.diff(rec.times)) == pytest.approx(DT)
    assert set(rec.epochs) == {"pre", "maze", "post", "awake", "non_rem", "rem"}
    np.testing.assert_array_equal(rec.epochs["pre"], [[0.0, MAZE[0]]])
    assert len(rec.epochs["non_rem"]) == 2
    assert rec.apparatus.startswith("1.9 m wooden linear maze")
    assert list(rec.units["label"]) == ["excitatory", "inhibitory"]
    # The stated maze along the planted axis, the stem centred on the middle:
    # the reward areas' and the stem's ends at 0, 15, 175 and 190 cm.
    ends = np.outer([-95.0, -80.0, 80.0, 95.0], AXIS) + MIDDLE
    np.testing.assert_allclose(
        rec.env.to_linear(ends), [0.0, 15.0, 175.0, 190.0], atol=0.2
    )
    # The shuttle spends the same share beyond each end of the stem.
    text = " ".join(rec.watch_outs)
    shares = re.search(
        r"([\d.]+)% of the samples placed on the track lie beyond the stem's "
        r"start and ([\d.]+)% beyond its end",
        text,
    )
    assert shares is not None
    beyond = 100 * np.mean(along < -80.0)
    np.testing.assert_allclose([float(v) for v in shares.groups()], beyond, atol=0.3)
    assert "For the first 30 s of tracking the rat is not on the track" in text
    assert "0 to 160 cm over 154.0 cm of the 2-D track" in text
    placement = next(c for c in rec.conversions if "least-squares" in c.text)
    assert placement.basis == "inferred"


#: The planted circle: centre and the stated 50 cm radius.
CENTRE = np.array([47.3, 50.5])


def _circle_session():
    """Counter-clockwise laps; the linearized position NaN over a 30 degree sector."""
    rng = np.random.default_rng(3)
    angle = np.mod(np.arange(20000) * 0.02, 2 * np.pi)
    radius = 50.0 + rng.normal(0.0, 1.5, angle.size)
    positions = CENTRE + radius[:, None] * np.column_stack(
        [np.cos(angle), np.sin(angle)]
    )
    linear = np.where(angle < np.radians(330.0), angle * 50.0, np.nan)
    return positions, linear, angle


def test_loads_a_dandi_circular_session(tmp_path):
    from neurospatial.datasets import grosmark_2016

    positions, linear, angle = _circle_session()
    path = tmp_path / "circle.nwb"
    _write_dandi(
        path, track="CircularMaze", positions_cm=positions, linearized_cm=linear
    )
    entry = _entry("sub-Achilles_ses-Achilles-11012013")
    rec = run_checks(grosmark_2016.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    assert rec.apparatus == "1 m diameter wooden circular maze"
    centre = next(c for c in rec.conversions if "centred at" in c.text)
    assert f"({CENTRE[0]:.1f}, {CENTRE[1]:.1f}) cm" in centre.text
    # The loop is the whole circle, in bins of the stated 2 cm, and the
    # tolerance leaves out the 5% of samples farthest from it, by construction.
    assert rec.env.n_bins == 158
    np.testing.assert_allclose(rec.env.bin_sizes, np.pi * 100.0 / 158, rtol=1e-3)
    outside = np.asarray(rec.env.bin_at(rec.positions)) < 0
    assert outside.mean() == pytest.approx(0.05, abs=0.002)
    # env.to_linear runs counter-clockwise from east of the centre.
    north = rec.env.to_linear(np.array([[CENTRE[0], CENTRE[1] + 50.0]]))[0]
    assert north == pytest.approx(rec.env.bin_sizes.sum() / 4, abs=0.1)
    text = " ".join(rec.watch_outs)
    uncovered = np.mean(angle >= np.radians(330.0))
    assert (
        f"{uncovered:.1%} of tracked samples lie within 20 cm of the circle but "
        "have none" in text
    )
    assert (
        "the rat runs counter-clockwise in 100% of running steps (above 5 cm/s), "
        "and rec.linearized increases counter-clockwise in 100.0% of its steps, "
        "so it increases along a run" in text
    )


def test_a_maze_the_paper_does_not_describe_is_refused(tmp_path):
    from neurospatial.datasets import grosmark_2016

    positions, linear, _ = _linear_session()
    path = tmp_path / "other.nwb"
    _write_dandi(
        path, track="3mLinearMaze", positions_cm=positions, linearized_cm=linear
    )
    with pytest.raises(ValueError, match=r"maze '3mLinearMaze' is none of the mazes"):
        grosmark_2016.load_session(_entry("sub-Buddy_ses-Buddy-06272013"), path)


@pytest.mark.realdata
# The OSF file's probe carries the 'manufacturer' field pynwb now deprecates.
@pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
def test_loads_default_session():
    """NeMoS's Achilles_10252013: 137 units, 1-D positions spanning the stem."""
    rec = load("grosmark_2016")
    assert rec.session == DEFAULT
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == _entry().n_units == 137
    assert rec.units["label"].value_counts().to_dict() == {"pE": 120, "pI": 17}
    assert rec.sampling_rate == pytest.approx(39.0626, rel=1e-4)
    assert rec.positions.shape == (12847, 1)
    # The survey's expected extent for this session's linear track, 150-175 cm.
    assert 150.0 <= np.ptp(rec.positions) <= 175.0
    assert {"pre", "maze", "post"} <= set(rec.epochs)
    assert len(rec.trials.start) == 84
    assert np.all(np.asarray(rec.env.bin_at(rec.positions)) >= 0)
    assert rec.provenance.sha256 == _entry().sha256


@pytest.mark.realdata
# The OSF file's probe carries the 'manufacturer' field pynwb now deprecates.
@pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
def test_osf_matches_dandi():
    """The OSF file and DANDI 000044's copy of Achilles 10252013 are one recording.

    Tolerances, set before comparing:

    - units: the same ids in the same order, the same brain regions, the
      authors' labels pE/pI as DANDI's excitatory/inhibitory, and the same
      spike counts and spike times, exactly;
    - times: every OSF position time is a DANDI sample time within 1e-7 s
      (float rounding of a 39 Hz grid ten thousand seconds from zero);
    - positions, after resampling DANDI's 2-D positions to the OSF times
      (the nearest sample, which the time check makes exact): the OSF value
      is a linear function a + b x + c y of them with sqrt(b² + c²) within
      1e-3 of 1 (one cm along the track per cm) and every residual under
      0.1 cm, a twentieth of a 2 cm bin; where DANDI has no 2-D position, the
      OSF value lies on the straight line between its neighbours within
      1e-6 cm (interpolated);
    - epochs: the loader's PRE/MAZE/POST for the OSF file equal DANDI's
      exactly.

    The numbers the loader's watch-outs and conversions quote for this
    session are recomputed here and must appear in their text.
    """
    from neurospatial.datasets.grosmark_2016 import DANDI_EPOCHS

    osf = load("grosmark_2016")
    dandi = load("grosmark_2016", DANDI_COPY)

    np.testing.assert_array_equal(osf.units.index, dandi.units.index)
    assert osf.n_units == dandi.n_units == 137
    np.testing.assert_array_equal(
        osf.units["brain_region"], dandi.units["brain_region"]
    )
    labels = {"pE": "excitatory", "pI": "inhibitory"}
    assert [labels[x] for x in osf.units["label"]] == list(dandi.units["label"])
    np.testing.assert_array_equal(osf.units["n_spikes"], dandi.units["n_spikes"])
    for mine, theirs in zip(osf.spike_times, dandi.spike_times, strict=True):
        np.testing.assert_array_equal(mine, theirs)

    for name, row in DANDI_EPOCHS[DEFAULT].items():
        np.testing.assert_array_equal(dandi.epochs[name], [row])
        np.testing.assert_array_equal(osf.epochs[name], [row])

    index = np.searchsorted(dandi.times, osf.times - 1e-7)
    assert np.max(np.abs(dandi.times[index] - osf.times)) < 1e-7
    xy = dandi.positions[index]
    value = osf.positions[:, 0]
    tracked = np.all(np.isfinite(xy), axis=1)
    design = np.column_stack([np.ones(tracked.sum()), xy[tracked]])
    (a, b, c), *_ = np.linalg.lstsq(design, value[tracked], rcond=None)
    residual = value[tracked] - design @ (a, b, c)
    assert abs(np.hypot(b, c) - 1.0) < 1e-3
    assert np.max(np.abs(residual)) < 0.1

    missing = np.flatnonzero(~tracked)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    for run in runs:
        before, after = run[0] - 1, run[-1] + 1
        share = (osf.times[run] - osf.times[before]) / (
            osf.times[after] - osf.times[before]
        )
        line = value[before] + share * (value[after] - value[before])
        np.testing.assert_allclose(value[run], line, atol=1e-6)

    linear = dandi.linearized[index]
    both = np.isfinite(linear)
    (offset, scale), *_ = np.linalg.lstsq(
        np.column_stack([np.ones(both.sum()), linear[both]]), value[both], rcond=None
    )
    spread = np.std(value[both] - offset - scale * linear[both])
    low, high = value[both].min(), value[both].max()
    extra = ~both

    normal = np.array([-c, b]) / np.hypot(b, c)
    lateral = dandi.positions @ normal - np.median(xy[tracked] @ normal)
    start = (dandi.times < osf.times[0] - 1e-7) & np.all(
        np.isfinite(dandi.positions), axis=1
    )
    away = np.abs(lateral[start])

    text = " ".join([*osf.watch_outs, *(c.evidence for c in osf.conversions)])
    quoted = [
        f"{a:.2f} + {b:.5f} x - {-c:.5f} y",
        f"residual SD {np.std(residual):.2f} cm, at most {np.max(np.abs(residual)):.2f} cm, "
        f"over {tracked.sum():,} samples",
        f"Where both are defined ({both.sum():,} samples) it is {offset:.2f} + "
        f"{scale:.4f} times DANDI's (residual SD {spread:.2f} cm)",
        f"DANDI's 0-160 cm covers {high - low:.1f} cm",
        f"differ by up to {np.max(np.abs(value[both] - linear[both])):.1f} cm",
        f"keeps {extra.sum()} samples where DANDI's is NaN: {(extra & (value < low)).sum()} "
        f"below and {(extra & (value > high)).sum()} above",
        f"{value[extra].min():.1f}-{low:.1f} cm and {high:.1f}-{value[extra].max():.1f} cm",
        f"and {(extra & (value >= low) & (value <= high)).sum()} between them",
        f"{missing.size} position samples, in {len(runs)} runs of "
        f"{min(map(len, runs))} to {max(map(len, runs))} samples (the longest from "
        f"{osf.times[max(runs, key=len)[0]]:,.2f} s)",
        f"{int((away > 20.0).sum())} of the {start.sum()} tracked samples there lie "
        f"more than 20 cm from the track's axis (median {np.median(away):.0f} cm)",
    ]
    for phrase in quoted:
        assert phrase in text, phrase


@pytest.mark.realdata
@pytest.mark.filterwarnings(
    "ignore:The 'manufacturer' field is deprecated:DeprecationWarning"
)
def test_streams_a_dandi_session():
    """Cicero 09172014, the 2 m stem, read by byte range: nothing is cached."""
    from neurospatial.datasets._fetch import cache_root

    session = "sub-Cicero_ses-Cicero-09172014"
    rec = load("grosmark_2016", session, stream=True)
    assert rec.failed_checks == ()
    assert rec.n_units == _entry(session).n_units == 72
    assert rec.apparatus.startswith("2.3 m metal linear maze")
    assert rec.sampling_rate == pytest.approx(39.0626, rel=1e-4)
    assert rec.provenance.url.startswith("https://api.dandiarchive.org/")
    assert not (cache_root() / "grosmark_2016" / _entry(session).file_name).exists()
