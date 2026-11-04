"""The Recording data model: validation, read-only arrays and headline numbers."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from neurospatial import InputError
from neurospatial.datasets import Conversion, DatasetCheckError, Provenance


def _swap_two_times(rec):
    times = np.array(rec.times)
    times[[10, 11]] = times[[11, 10]]
    return {"times": times}


def _repeat_a_unit_id(rec):
    units = rec.units.drop(columns=["n_spikes", "fires_in_epochs"])
    units.index = units.index.where(units.index != 1, 0)
    return {"units": units}


BAD_FIELDS = {
    "times": _swap_two_times,
    "positions": lambda rec: {"positions": rec.positions[:-1]},
    "units": _repeat_a_unit_id,
    "headings": lambda rec: {"headings": np.degrees(rec.headings)},
    "epochs": lambda rec: {"epochs": {"run": np.array([[10.0, 5.0]])}},
    "position_units": lambda rec: {"position_units": "inches"},
    "env_kind": lambda rec: {"env_kind": None},
    "failed_checks": lambda rec: {"failed_checks": ("speed",)},
}


@pytest.mark.parametrize("field", sorted(BAD_FIELDS))
def test_validation_refuses_bad_shapes(square_recording, field):
    with pytest.raises(InputError, match=rf"(?m)^- {field}(\[.*\])?: "):
        dataclasses.replace(square_recording, **BAD_FIELDS[field](square_recording))


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_positions_are_refused(square_recording, value):
    """NaN marks lost tracking; an infinite position is refused, not kept."""
    positions = np.array(square_recording.positions)
    positions[5, 1] = value
    with pytest.raises(InputError, match="positions: 1 value is infinite"):
        dataclasses.replace(square_recording, positions=positions)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_headings_are_refused(square_recording, value):
    """NaN marks a missing heading; an infinite one is refused, not kept."""
    headings = np.array(square_recording.headings)
    headings[5] = value
    with pytest.raises(InputError, match="headings: 1 value is infinite"):
        dataclasses.replace(square_recording, headings=headings)


@pytest.mark.parametrize(
    ("basis", "evidence", "named"),
    [("inferred", "", "evidence"), ("guessed", "it looked right", "basis")],
)
def test_conversion_needs_basis_and_evidence(basis, evidence, named):
    with pytest.raises(InputError, match=named):
        Conversion("positions: pixels * 0.43 cm/pixel", basis, evidence)


def test_stated_conversion_needs_no_evidence():
    assert Conversion("spike times: µs / 1e6", "stated").evidence == ""


def test_provenance_needs_a_checksum():
    with pytest.raises(InputError, match="sha256"):
        Provenance("Zenodo 1", "https://x", "abc", 10, "0.0", "2026-11-03")


def test_validation_lists_every_bad_field(square_recording):
    with pytest.raises(InputError) as caught:
        dataclasses.replace(
            square_recording,
            positions=square_recording.positions[:-1],
            position_units="inches",
        )
    message = str(caught.value)
    assert "- positions: " in message
    assert "- position_units: " in message
    assert message.splitlines()[-1].startswith("Fix: ")


@pytest.mark.parametrize(
    "name", ["square_recording", "track_recording", "heading_recording"]
)
def test_summary_keys_read_back(name, request):
    rec = request.getfixturevalue(name)
    summary = rec.summary()
    assert list(summary) == [
        "n_units",
        "duration",
        "n_dims",
        "sampling_rate",
        "tracked_fraction",
        "longest_gap",
        "median_speed",
        "speed_p95",
        "verified",
    ]
    for key, value in summary.items():
        assert isinstance(value, bool | int | float)
        read = getattr(rec, key)
        assert read == value or (np.isnan(read) and np.isnan(value)), key


def test_summary_units_declared(square_recording, heading_recording):
    assert square_recording._repr_unit("duration") == "s"
    assert square_recording._repr_unit("sampling_rate") == "Hz"
    assert square_recording._repr_unit("longest_gap") == "s"
    assert square_recording._repr_unit("median_speed") == "cm/s"
    assert square_recording._repr_unit("speed_p95") == "cm/s"
    assert square_recording._repr_unit("n_units") == ""
    assert heading_recording._repr_unit("median_speed") == ""


def test_summary_numbers_match_the_arrays(square_recording):
    """Tracked share, longest gap and speed agree with hand-made arrays."""
    times = square_recording.times
    # A straight run at 20 cm/s, with samples 100 to 149 lost.
    positions = np.column_stack([20.0 * times, np.full(times.size, 50.0)])
    positions[100:150] = np.nan
    rec = dataclasses.replace(square_recording, positions=positions, headings=None)
    assert rec.n_dims == 2
    assert rec.sampling_rate == pytest.approx(100.0)
    assert rec.duration == pytest.approx(times[-1] - times[0])
    assert rec.tracked_fraction == pytest.approx(1 - 50 / times.size)
    assert rec.longest_gap == pytest.approx(times[150] - times[99])
    assert rec.median_speed == pytest.approx(20.0)
    assert rec.speed_p95 == pytest.approx(20.0)


def test_two_runs_around_a_sleep(square_recording):
    """Duration spans the sleep; the longest gap is within the tracked runs."""
    run = square_recording.times
    times = np.r_[run, run + 3600.0]
    positions = np.r_[square_recording.positions, square_recording.positions]
    positions[1000:1050] = np.nan  # a 0.51 s gap inside the first run
    rec = dataclasses.replace(
        square_recording,
        times=times,
        positions=positions,
        headings=None,
        epochs={
            "run": np.array([[0.0, 60.0], [3600.0, 3660.0]]),
            "sleep": np.array([[60.0, 3600.0]]),
        },
    )
    assert rec.duration == pytest.approx(times[-1] - times[0])
    assert rec.longest_gap == pytest.approx(times[1050] - times[999])
    text = " ".join(repr(rec).split())
    assert "3660 s from the first to the last sample" in text
    assert "longest gap 0.51 s" in text


def test_conversions_must_be_conversions(square_recording):
    with pytest.raises(InputError, match=r"- conversions\[0\]: a tuple"):
        dataclasses.replace(square_recording, conversions=(("x", "inferred", ""),))


def test_untracked_session_has_no_speed(square_recording):
    """With no two tracked samples, speed and gap are NaN and the text prints."""
    positions = np.full_like(square_recording.positions, np.nan)
    positions[0] = 1.0
    rec = dataclasses.replace(square_recording, positions=positions)
    assert np.isnan(rec.median_speed)
    assert np.isnan(rec.speed_p95)
    assert np.isnan(rec.longest_gap)
    assert "speed median nan; 95th percentile nan" in repr(rec)
    assert "longest gap nan\n" in repr(rec)


def test_heading_only_summary(heading_recording):
    assert heading_recording.n_dims == 0
    assert np.isnan(heading_recording.median_speed)
    assert heading_recording.tracked_fraction == pytest.approx(
        np.isfinite(heading_recording.headings).mean()
    )


def test_arrays_read_only(square_recording, open_field):
    rec = square_recording
    arrays = [
        rec.times,
        rec.positions,
        rec.headings,
        rec.spike_times[0],
        rec.epochs["run"],
    ]
    assert all(not array.flags.writeable for array in arrays)
    assert not rec.positions.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.positions = None
    # Arrays are copies: editing the input leaves the recording unchanged.
    positions = np.array(open_field.positions)
    owned = dataclasses.replace(rec, positions=positions)
    positions[0] = -1.0
    assert np.array_equal(owned.positions, open_field.positions)


def test_unit_table_reads_are_copies(square_recording):
    table = square_recording.units
    table.drop(index=table.index[0], inplace=True)
    assert len(square_recording.units) == square_recording.n_units


def test_unit_table_counts_spikes(square_recording):
    rec = square_recording
    assert rec.units.index.name == "unit_id"
    assert rec.units["n_spikes"].tolist() == [len(t) for t in rec.spike_times]
    assert rec.units["fires_in_epochs"].all()
    # A one-microsecond epoch just after the first unit's first spike holds none.
    early = rec.spike_times[0][0]
    later = dataclasses.replace(
        rec, params={"analysis_epochs": np.array([[early + 1e-6, early + 2e-6]])}
    )
    assert not later.units["fires_in_epochs"].iloc[0]


def test_pickle_round_trip(square_recording):
    restored = pickle.loads(pickle.dumps(square_recording))
    assert np.array_equal(restored.positions, square_recording.positions)
    assert not restored.positions.flags.writeable
    assert dict(restored.epochs).keys() == {"run"}
    assert repr(restored) == repr(square_recording)


def test_nwb_unit_table_refuses_a_missing_column():
    import types

    import pandas as pd

    from neurospatial.datasets._recording import nwb_unit_table

    units = types.SimpleNamespace(
        unit_table=pd.DataFrame({"location": ["CA1", "CA3"]}, index=[3, 7])
    )
    table, regions = nwb_unit_table(units, region_column="location")
    assert list(table["brain_region"]) == ["CA1", "CA3"]
    assert regions == ("CA1", "CA3")
    with pytest.raises(ValueError, match=r"no column \['cell_type'\]"):
        nwb_unit_table(units, region_column="location", label_column="cell_type")


class TestClipEpochsToRecording:
    """Epoch rows past the recording are clipped to it, and reported once."""

    TIMES = np.arange(10.0, 20.0, 0.1)  # samples 10.0 to 19.9 s

    def _clip(self, epochs, spikes):
        from neurospatial.datasets._recording import clip_epochs_to_recording

        return clip_epochs_to_recording(
            epochs, self.TIMES, spikes, source="the file's table"
        )

    def test_rows_inside_are_unchanged_and_nothing_is_reported(self):
        rows = np.array([[10.0, 12.0], [15.0, 19.9]])
        clipped = self._clip({"run": rows}, (np.array([10.5, 19.0]),))
        np.testing.assert_array_equal(clipped.epochs["run"], rows)
        assert clipped.conversion is None
        assert clipped.watch_out is None

    def test_an_edge_within_the_clock_checks_slack_is_kept_as_stored(self):
        """The check allows 1% of the recording (here 0.099 s) at its edges."""
        rows = np.array([[9.95, 19.95]])
        clipped = self._clip({"run": rows}, (np.array([12.0]),))
        np.testing.assert_array_equal(clipped.epochs["run"], rows)
        assert clipped.watch_out is None

    def test_the_recording_ends_at_the_last_spike_past_the_samples(self):
        """An epoch with spikes but no samples survives, clipped to the last spike."""
        clipped = self._clip(
            {"square": np.array([[25.0, 40.0]]), "run": np.array([[5.0, 19.9]])},
            (np.array([10.2, 26.0, 31.5]),),
        )
        np.testing.assert_array_equal(clipped.epochs["square"], [[25.0, 31.5]])
        # The first sample (10.0) starts the recording: run starts 5 s early.
        np.testing.assert_array_equal(clipped.epochs["run"], [[10.0, 19.9]])
        assert clipped.watch_out == (
            "2 epoch rows reach past the recording, 10 to 31.5 s (its first and "
            "last sample or spike), and are clipped to it: epochs['square'] "
            "[25, 40] s ends 8.5 s after it, now [25, 31.5] s; epochs['run'] "
            "[5, 19.9] s starts 5 s before it, now [10, 19.9] s. Positions and "
            "spikes are as stored."
        )
        assert clipped.conversion.basis == "inferred"
        assert clipped.conversion.evidence.startswith("the file's table lists them so")

    def test_a_row_wholly_outside_is_dropped_with_its_name(self):
        clipped = self._clip(
            {"before": np.array([[5.0, 9.0]]), "run": np.array([[11.0, 25.0]])},
            (np.array([12.0]),),
        )
        assert list(clipped.epochs) == ["run"]
        np.testing.assert_allclose(clipped.epochs["run"], [[11.0, 19.9]])
        assert "epochs['before'] [5, 9] s lies wholly outside it, dropped" in (
            clipped.watch_out
        )
        assert clipped.watch_out.startswith("2 epoch rows reach past the recording")

    @pytest.mark.parametrize(
        "row",
        [
            [10_000.0, 19_900.0],  # milliseconds read as seconds: far outside
            [12.0, 12_000.0],  # starts inside, ends 1000 times too late
            [-40.0, 15.0],  # starts more than the recording's length early
        ],
    )
    def test_a_row_overhanging_by_more_than_the_recording_is_left_for_the_clock_check(
        self, row, square_recording
    ):
        """No truncation overhangs by more than the recording's own 9.9 s."""
        from neurospatial.datasets._checks import Stated, run_checks

        clipped = self._clip({"run": np.array([row])}, (np.array([12.0]),))
        np.testing.assert_array_equal(clipped.epochs["run"], [row])
        assert clipped.conversion is None
        assert clipped.watch_out is None
        # The same row on a 60 s recording, which the clock check refuses.
        rec = dataclasses.replace(square_recording, epochs=clipped.epochs)
        with pytest.raises(DatasetCheckError, match="clock"):
            run_checks(
                rec, stated=Stated(sampling_rate=100.0, position_span=(80.0, 120.0))
            )
