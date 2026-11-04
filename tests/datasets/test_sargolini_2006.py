"""Sargolini et al. (2006), DANDI 000582: synthetic NWB files and the real sessions."""

from __future__ import annotations

import dataclasses
import datetime

import numpy as np
import pytest

from neurospatial.datasets import load
from neurospatial.datasets._checks import run_checks

DEFAULT = "sub-10073_ses-17010302"


def _write_session(path, xy_cm, *, second_led=False, histology="MEC LII"):
    """An NWB file laid out as DANDI 000582's: LED positions in cm, as metres x 0.01."""
    pynwb = pytest.importorskip("pynwb")
    from pynwb.behavior import Position, SpatialSeries
    from pynwb.file import Subject

    start = datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc)
    nwb = pynwb.NWBFile(
        "a rat running in a 1 x 1 meter enclosure",
        "synthetic",
        start,
        subject=Subject(subject_id="10073", species="Rattus norvegicus"),
    )
    times = np.arange(len(xy_cm)) / 50.0
    position = Position(name="Position")
    for name in ("SpatialSeriesLED1", "SpatialSeriesLED2")[: 1 + second_led]:
        position.add_spatial_series(
            SpatialSeries(
                name=name,
                data=xy_cm,
                timestamps=times,
                reference_frame="(0,0) is not known.",
                unit="meters",
                conversion=0.01,
            )
        )
    nwb.create_processing_module("behavior", "position").add(position)
    nwb.add_unit_column("unit_name", "tetrode and cell")
    nwb.add_unit_column("histology", "layer")
    rng = np.random.default_rng(0)
    for unit in range(3):
        spikes = np.sort(rng.uniform(times[0], times[-1], 200))
        nwb.add_unit(spike_times=spikes, unit_name=f"t1c{unit}", histology=histology)
    with pynwb.NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return times


@pytest.fixture(scope="module")
def box_path_cm():
    """600 s of a random walk filling the box, centred and scaled as the archive's."""
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 1.0, (30_000, 2)).cumsum(axis=0)
    xy = np.mod(steps, 200.0)
    xy = np.where(xy > 100.0, 200.0 - xy, xy)  # reflect inside [0, 100]
    xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2
    return xy * (100.0 / np.ptp(xy, axis=0).max())


def test_loads_a_session_in_cm(tmp_path, box_path_cm, session_entry):
    path = tmp_path / "session.nwb"
    times = _write_session(path, box_path_cm)
    from neurospatial.datasets import sargolini_2006

    entry = session_entry("sargolini_2006")
    rec = run_checks(sargolini_2006.load_session(entry, path), stated=entry.info.stated)
    np.testing.assert_allclose(rec.times, times)
    # The file stores cm stated as metres x 0.01; read_position gives metres.
    np.testing.assert_allclose(rec.positions, box_path_cm)
    assert rec.position_units == "cm"
    assert rec.conversions[0].basis == "stated"
    assert "'meters', conversion 0.01" in rec.conversions[0].evidence
    assert rec.headings is None
    assert list(rec.units["brain_region"]) == ["MEC LII"] * 3
    assert list(rec.units["unit_name"]) == ["t1c0", "t1c1", "t1c2"]
    assert rec.units["label"].isna().all()
    assert rec.brain_regions == ("MEC LII",)
    assert rec.animal == "rat 10073"
    assert rec.failed_checks == ()
    # The declared square: 100 cm, at the samples' midrange, 2.5 cm bins.
    lows, highs = rec.env.dimension_ranges[0], rec.env.dimension_ranges[1]
    np.testing.assert_allclose([lows, highs], [[-50, 50], [-50, 50]])
    assert rec.env.n_bins == 40 * 40
    assert rec.env.units == "cm"
    assert rec.env_kind == "declared"
    assert rec.conversions[1].basis == "inferred"
    assert "centred at (0, 0) cm" in rec.conversions[1].text
    spans = np.ptp(box_path_cm, axis=0)
    assert f"x {spans[0]:.4g} cm, +0.0%, y {spans[1]:.4g} cm" in rec.conversions[1].text
    # The session's own and the dataset's watch-outs, then nothing else.
    assert (
        rec.watch_outs[: len(sargolini_2006.WATCH_OUTS[DEFAULT])]
        == (sargolini_2006.WATCH_OUTS[DEFAULT])
    )
    text = " ".join(rec.watch_outs)
    assert "stores only the first LED" in text
    assert "moving mean filter" in text
    assert "centred and scaled to the box" in text
    np.testing.assert_allclose(rec.params["train_epochs"][0], [0.0, 60.0])
    assert rec.params["bin_size"] == 2.5


def test_second_led_is_named_and_not_used(tmp_path, box_path_cm, session_entry):
    path = tmp_path / "session.nwb"
    _write_session(path, box_path_cm, second_led=True)
    from neurospatial.datasets import sargolini_2006

    rec = sargolini_2006.load_session(
        session_entry("sargolini_2006", "sub-11138_ses-06040501"), path
    )
    text = " ".join(rec.watch_outs)
    assert "also stores the second LED (SpatialSeriesLED2)" in text
    assert "stores only the first LED" not in text
    assert rec.headings is None
    assert not rec.verified


def test_coordinates_not_fitted_to_the_box_give_no_fit_watch_out(
    tmp_path, box_path_cm, session_entry
):
    """Positions off centre spanning 90 cm: the box goes there, with no fitted note."""
    path = tmp_path / "session.nwb"
    shifted = box_path_cm * 0.9 + [10.0, -5.0]
    _write_session(path, shifted)
    from neurospatial.datasets import sargolini_2006

    rec = sargolini_2006.load_session(session_entry("sargolini_2006"), path)
    assert not any("centred and scaled" in text for text in rec.watch_outs)
    assert "centred at (10, -5) cm" in rec.conversions[1].text
    np.testing.assert_allclose(rec.env.dimension_ranges, [[-40, 60], [-55, 45]])


def test_a_110_cm_span_gives_its_residual_and_the_samples_outside(
    tmp_path, box_path_cm, session_entry
):
    """Samples past the stated side: the residual is recorded and the outside counted."""
    from neurospatial.datasets import sargolini_2006

    path = tmp_path / "session.nwb"
    wide = box_path_cm * 1.1
    _write_session(path, wide)
    entry = session_entry("sargolini_2006")
    rec = run_checks(sargolini_2006.load_session(entry, path), stated=entry.info.stated)
    assert rec.failed_checks == ()
    placement = rec.conversions[1].text
    assert "the samples' range against the stated side: x 110 cm, +10.0%" in placement
    # Outside the square, counted without the environment's own lookup.
    outside = np.any(np.abs(wide) > 50.0, axis=1)
    assert 0 < outside.sum() < len(wide)
    expected = (
        f"{int(outside.sum())} tracked position samples of 30000 "
        f"({outside.mean():.2%}) lie outside the declared environment"
    )
    assert any(text.startswith(expected) for text in rec.watch_outs)
    np.testing.assert_allclose(rec.positions, wide)
    assert not any("centred and scaled" in text for text in rec.watch_outs)


def test_a_two_metre_session_fails_the_units_check(
    tmp_path, box_path_cm, session_entry
):
    """A session spanning 2 m, as some joined sessions do, is not the 1 m box."""
    from neurospatial.datasets import DatasetCheckError, sargolini_2006

    path = tmp_path / "session.nwb"
    _write_session(path, box_path_cm * 2.0)
    entry = session_entry("sargolini_2006")
    rec = sargolini_2006.load_session(entry, path)
    with pytest.raises(DatasetCheckError, match="largest position span is 200 cm"):
        run_checks(rec, stated=entry.info.stated)


@pytest.mark.realdata
def test_loads_default_session(session_entry, assert_default_session):
    """The pynapple DANDI tutorial's session: 8 units, 50 Hz, the 1 m box."""
    rec = load("sargolini_2006")
    assert_default_session(rec, session_entry("sargolini_2006"))
    assert rec.n_units == 8
    # Spike counts from the rates pynapple's tutorial prints for units 0 and 1.
    assert list(rec.units["n_spikes"][:2]) == [1759, 901]
    assert rec.sampling_rate == pytest.approx(50.0, rel=1e-6)
    assert rec.times.size == 30_000
    np.testing.assert_allclose(
        np.nanmin(rec.positions, axis=0), [-50.0, -49.3588], atol=1e-4
    )
    np.testing.assert_allclose(
        np.nanmax(rec.positions, axis=0), [50.0, 49.3588], atol=1e-4
    )
    assert rec.env.n_bins == 1600


@pytest.mark.realdata
def test_loads_the_survey_session():
    """sub-10884, as the survey's loader expected it: 6 units, 50 Hz, within ±55 cm."""
    rec = load("sargolini_2006", "sub-10884_ses-03080402")
    assert rec.verified
    assert rec.failed_checks == ()
    assert rec.n_units == 6
    assert 49.0 <= rec.sampling_rate <= 51.0
    low, high = np.nanmin(rec.positions, axis=0), np.nanmax(rec.positions, axis=0)
    assert np.all((low >= -55.0) & (low <= 0.0))
    assert np.all((high >= 0.0) & (high <= 55.0))
    assert np.isnan(rec.positions[0]).all()


@pytest.mark.realdata
def test_streamed_session_matches_the_download():
    cached = load("sargolini_2006")
    streamed = load("sargolini_2006", stream=True)
    np.testing.assert_array_equal(streamed.positions, cached.positions)
    np.testing.assert_array_equal(streamed.times, cached.times)
    for a, b in zip(streamed.spike_times, cached.spike_times, strict=True):
        np.testing.assert_array_equal(a, b)
    assert dataclasses.replace(streamed.provenance, fetched_at="") == (
        dataclasses.replace(cached.provenance, fetched_at="")
    )
