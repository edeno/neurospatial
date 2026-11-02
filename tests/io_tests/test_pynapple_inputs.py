"""pynapple objects passed straight to analysis functions.

A ``TsGroup`` keeps its unit labels and metadata, a ``TsdFrame`` or ``Tsd``
passed as ``positions`` or ``headings`` keeps its time support and column
order, and ``IntervalSet`` intervals keep their inclusive ends. Each test
compares the pynapple route with the plain-array route it must equal.
"""

from __future__ import annotations

import importlib.util
import re

import numpy as np
import pandas as pd
import pytest

from neurospatial import Environment

pytestmark = [
    pytest.mark.pynapple,
    pytest.mark.skipif(
        importlib.util.find_spec("pynapple") is None,
        reason="requires the pynapple extra",
    ),
]


@pytest.fixture
def nap():
    import pynapple

    return pynapple


@pytest.fixture
def session():
    """A 40 s open-field walk at 100 Hz with three units."""
    rng = np.random.default_rng(0)
    times = np.round(np.arange(4000) * 0.01, 10)
    positions = np.column_stack(
        [
            50.0 + 40.0 * np.sin(times / 3.0),
            20.0 + 15.0 * np.cos(times / 2.0),
        ]
    )
    env = Environment.from_samples(positions, bin_size=10.0)
    trains = [np.sort(rng.uniform(0.0, 39.99, n)) for n in (80, 60, 70)]
    return env, times, positions, trains


@pytest.fixture
def group(nap, session):
    """The three units labelled 5, 6, 7 with a brain-region column."""
    _, _, _, trains = session
    return nap.TsGroup(
        {5: nap.Ts(trains[0]), 6: nap.Ts(trains[1]), 7: nap.Ts(trains[2])},
        metadata={"region": ["CA1", "CA3", "CA1"], "depth": [1.0, 2.0, 3.0]},
    )


# ---------------------------------------------------------------------------
# TsGroup metadata becomes the unit table
# ---------------------------------------------------------------------------


def _expected_table(pd):
    return pd.DataFrame(
        {"region": ["CA1", "CA3", "CA1"], "depth": [1.0, 2.0, 3.0]},
        index=pd.Index([5, 6, 7], name="unit_id"),
    )


def test_spike_trains_accepts_a_tsgroup_with_labels_and_metadata(group, session):
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    st = SpikeTrains(group)

    assert st.unit_ids.tolist() == [5, 6, 7]
    for got, want in zip(st.trains, session[3], strict=True):
        np.testing.assert_array_equal(got, want)
    # pynapple's computed ``rate`` column is not unit metadata.
    pd.testing.assert_frame_equal(
        st.unit_table, _expected_table(pd), check_index_type=False
    )
    assert st.filter("region == 'CA1'").unit_ids.tolist() == [5, 7]


def test_spike_trains_accepts_a_dict_of_trains() -> None:
    from neurospatial.encoding import SpikeTrains

    st = SpikeTrains({"a": [0.1, 0.2], "b": [0.3]})

    assert st.unit_ids.tolist() == ["a", "b"]
    np.testing.assert_array_equal(st.sel(unit_id="b"), [0.3])


def test_spike_trains_from_tsgroup_refuses_other_unit_ids(group):
    from neurospatial.encoding import SpikeTrains

    with pytest.raises(ValueError, match=r"already labelled \[5, 6, 7\]"):
        SpikeTrains(group, unit_ids=[0, 1, 2])


def test_spike_trains_from_tsgroup_with_explicit_table(group):
    import pandas as pd

    from neurospatial.encoding import SpikeTrains

    table = pd.DataFrame({"quality": [0.9, 0.5, 0.7]}, index=pd.Index([5, 6, 7]))
    st = SpikeTrains(group, unit_table=table)

    assert st.unit_table.columns.tolist() == ["quality"]


def test_compute_spatial_rates_keeps_tsgroup_metadata(group, session):
    import pandas as pd

    from neurospatial import compute_spatial_rates

    env, times, positions, trains = session
    from_group = compute_spatial_rates(env, group, times, positions, method="binned")
    from_arrays = compute_spatial_rates(env, trains, times, positions, method="binned")

    np.testing.assert_array_equal(from_group.firing_rates, from_arrays.firing_rates)
    pd.testing.assert_frame_equal(
        from_group.unit_table, _expected_table(pd), check_index_type=False
    )
    assert from_group.sel(unit_id=6).unit_id == 6


def test_population_encoders_keep_tsgroup_metadata(group, session):
    from neurospatial.encoding import (
        compute_directional_rates,
        compute_object_vector_rates,
    )

    env, times, positions, _ = session
    headings = np.arctan2(np.gradient(positions[:, 1]), np.gradient(positions[:, 0]))
    results = [
        compute_directional_rates(group, times, headings),
        compute_object_vector_rates(
            env,
            group,
            times,
            positions,
            np.array([[50.0, 20.0]]),
            distance_bin_size=5.0,
        ),
    ]
    for result in results:
        assert result.unit_table["region"].tolist() == ["CA1", "CA3", "CA1"]
        assert result.unit_table.index.tolist() == [5, 6, 7]


def test_population_peri_event_histogram_keeps_unit_tables(group):
    from neurospatial.encoding import SpikeTrains
    from neurospatial.events import population_peri_event_histogram

    events = np.array([10.0, 20.0, 30.0])
    for spikes in (group, SpikeTrains(group)):
        result = population_peri_event_histogram(
            spikes, events, (-0.5, 0.5), bin_size=0.1
        )
        assert result.unit_ids.tolist() == [5, 6, 7]
        assert result.unit_table["region"].tolist() == ["CA1", "CA3", "CA1"]


def test_tsgroup_with_only_rate_metadata_gives_no_unit_table(nap, session):
    from neurospatial import compute_spatial_rates

    env, times, positions, trains = session
    bare = nap.TsGroup({5: nap.Ts(trains[0]), 6: nap.Ts(trains[1])})

    result = compute_spatial_rates(env, bare, times, positions, method="binned")

    assert result.unit_table is None
    assert result.unit_ids.tolist() == [5, 6]


# ---------------------------------------------------------------------------
# restrict_spike_trains keeps unit identity
# ---------------------------------------------------------------------------


def test_restrict_spike_trains_keeps_tsgroup_labels_and_metadata(group, session):
    from neurospatial.behavior import restrict_spike_trains

    _, _, _, trains = session
    window = (10.0, 20.0)

    kept = restrict_spike_trains(group, window)

    assert kept.unit_ids.tolist() == [5, 6, 7]
    assert kept.unit_table["region"].tolist() == ["CA1", "CA3", "CA1"]
    for uid, train in zip([5, 6, 7], trains, strict=True):
        expected = train[(train >= 10.0) & (train < 20.0)]
        np.testing.assert_array_equal(kept.sel(unit_id=uid), expected)


# ---------------------------------------------------------------------------
# A pynapple TsdFrame / Tsd passed as positions or headings
# ---------------------------------------------------------------------------

# A 0.5 s tracking gap, shorter than the default max_gap, between two epochs.
GAPPED = np.array([[0.0, 20.0], [20.5, 39.99]])


@pytest.fixture
def gapped(nap, session):
    """The session's positions as a TsdFrame recorded in two epochs."""
    _, times, positions, _ = session
    support = nap.IntervalSet(start=GAPPED[:, 0], end=GAPPED[:, 1])
    return nap.TsdFrame(t=times, d=positions, columns=["x", "y"], time_support=support)


def _half_open(rows):
    rows = np.array(rows, dtype=np.float64)
    rows[:, 1] = np.nextafter(rows[:, 1], np.inf)
    return rows


def test_tsdframe_positions_keep_their_time_support_in_occupancy(gapped, session):
    env = session[0]
    times, positions = gapped.t, gapped.to_numpy()

    direct = env.occupancy(times, gapped)
    with_epochs = env.occupancy(times, positions, epochs=gapped.time_support)
    arrays_only = env.occupancy(times, positions)

    # The 0.5 s gap is shorter than max_gap, so arrays alone count it.
    assert arrays_only.sum() == pytest.approx(39.99)
    assert with_epochs.sum() == pytest.approx(39.49)
    np.testing.assert_array_equal(direct, with_epochs)


def test_tsdframe_positions_record_the_adopted_epochs(gapped, session):
    from neurospatial import compute_spatial_rates

    env, _, _, trains = session
    direct = compute_spatial_rates(env, trains, gapped.t, gapped, method="binned")
    explicit = compute_spatial_rates(
        env,
        trains,
        gapped.t,
        gapped.values,
        method="binned",
        epochs=gapped.time_support,
    )

    np.testing.assert_array_equal(direct.occupancy, explicit.occupancy)
    np.testing.assert_array_equal(direct.firing_rates, explicit.firing_rates)
    assert np.nansum(direct.occupancy) == pytest.approx(39.49)
    np.testing.assert_array_equal(
        direct.analysis_settings["epochs"], _half_open(GAPPED)
    )


def test_explicit_epochs_intersect_the_time_support(gapped, session):
    """Only time inside both the epochs and the sample's support is used."""
    env = session[0]
    times, positions = gapped.t, gapped.to_numpy()

    narrowed = env.occupancy(times, gapped, epochs=(0.0, 10.0))
    across_gap = env.occupancy(times, gapped, epochs=(10.0, 30.0))
    # By hand: [10, 20] and [20.5, 30) of the support, without the 0.5 s gap.
    by_hand = env.occupancy(times, positions, epochs=[(10.0, 20.0), (20.5, 30.0)])

    assert narrowed.sum() == pytest.approx(10.0)
    assert across_gap.sum() == pytest.approx(19.5, abs=0.02)
    np.testing.assert_allclose(across_gap, by_hand, atol=0.011)
    # Epochs outside the support leave nothing, as for the arrays route.
    with pytest.warns(UserWarning, match="occupancy is zero"):
        nothing = env.occupancy(times, gapped, epochs=(20.1, 20.4))
    assert nothing.sum() == 0.0


def test_epochs_on_another_clock_are_reported_as_for_arrays(gapped, session):
    """Epochs missing the support are a clock mismatch, not an empty condition."""
    from neurospatial.events import peri_event_distance_histogram

    times = gapped.t
    events = np.array([5.0, 15.0, 25.0])
    in_ms = [(5000.0, 25000.0)]
    for positions in (gapped.to_numpy(), gapped):
        with (
            pytest.warns(UserWarning, match="check that epochs overlap"),
            pytest.raises(ValueError, match="none of the 3 events falls"),
        ):
            peri_event_distance_histogram(
                np.array([1.0]),
                events,
                times,
                positions,
                (-2.0, 2.0),
                bin_size=1.0,
                epochs=in_ms,
            )


def test_decode_session_uses_the_positions_time_support(gapped, session):
    from neurospatial.decoding import decode_session

    env, _, _, trains = session
    direct = decode_session(env, trains, gapped.t, gapped, dt=0.5)
    explicit = decode_session(
        env, trains, gapped.t, gapped.values, dt=0.5, epochs=gapped.time_support
    )

    np.testing.assert_array_equal(direct.times, explicit.times)
    np.testing.assert_array_equal(direct.posterior, explicit.posterior)


def test_compute_speed_uses_the_positions_time_support(gapped):
    from neurospatial.behavior import compute_speed

    direct = compute_speed(gapped.t, gapped)
    explicit = compute_speed(gapped.t, gapped.values, epochs=gapped.time_support)

    np.testing.assert_array_equal(direct, explicit)


def test_tsd_headings_use_their_time_support(nap, gapped, session):
    from neurospatial.encoding import compute_directional_rates

    _, _, _, trains = session
    values = gapped.to_numpy()
    angles = np.arctan2(np.gradient(values[:, 1]), np.gradient(values[:, 0]))
    headings = nap.Tsd(t=gapped.t, d=angles, time_support=gapped.time_support)

    direct = compute_directional_rates(trains, headings.t, headings)
    explicit = compute_directional_rates(
        trains, headings.t, angles, epochs=headings.time_support
    )

    np.testing.assert_array_equal(direct.occupancy, explicit.occupancy)
    assert np.nansum(direct.occupancy) == pytest.approx(39.49)


def test_positions_and_headings_with_different_supports_are_refused(
    nap, gapped, session
):
    from neurospatial.encoding import compute_view_rate

    other = nap.Tsd(
        t=gapped.t,
        d=np.zeros(len(gapped)),
        time_support=nap.IntervalSet(start=0.0, end=39.99),
    )
    with pytest.raises(ValueError, match=r"different pynapple time supports"):
        compute_view_rate(session[0], session[3][0], gapped.t, gapped, other)


@pytest.mark.parametrize("columns", [["y", "x"], ["Y", "X"]])
def test_permuted_tsdframe_positions_are_refused(nap, session, columns):
    from neurospatial import compute_spatial_rates

    env, times, positions, trains = session
    frame = nap.TsdFrame(t=times, d=positions[:, ::-1], columns=columns)

    with pytest.raises(ValueError, match=r"not in x, y, z order") as excinfo:
        compute_spatial_rates(env, trains, times, frame)
    assert f"positions=tsdframe.loc[{sorted(columns, key=str.lower)}]" in str(
        excinfo.value
    )
    with pytest.raises(ValueError, match=r"not in x, y, z order"):
        env.occupancy(times, frame)


def test_ordered_tsdframe_positions_match_plain_arrays(nap, session):
    env, times, positions, _ = session
    frame = nap.TsdFrame(t=times, d=positions, columns=["x", "y"])

    np.testing.assert_array_equal(
        env.occupancy(times, frame), env.occupancy(times, positions)
    )


# ---------------------------------------------------------------------------
# Intervals: IntervalSet accepted where an interval table is, Tsd refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", ["tsd", "tsdframe", "ts"])
def test_a_time_series_as_epochs_names_its_time_support(nap, make):
    from neurospatial._intervals import as_intervals

    t = np.array([1.0, 2.0, 3.0])
    series = {
        "tsd": lambda: nap.Tsd(t=t, d=t),
        "tsdframe": lambda: nap.TsdFrame(t=t, d=np.c_[t, t], columns=["x", "y"]),
        "ts": lambda: nap.Ts(t=t),
    }[make]()
    name = type(series).__name__

    with pytest.raises(ValueError, match=rf"epochs is a pynapple {name}") as excinfo:
        as_intervals(series, name="epochs")
    assert f"epochs={name.lower()}.time_support" in str(excinfo.value)


@pytest.fixture
def trials(nap):
    """Two trials whose second one ends exactly on an event."""
    return nap.IntervalSet(
        start=[5.0, 50.0], end=[20.0, 80.0], metadata={"kind": ["a", "b"]}
    )


def test_filter_by_intervals_accepts_an_intervalset(nap, trials):
    import pandas as pd

    from neurospatial.events import filter_by_intervals

    times = np.array([3.0, 10.0, 20.0, 30.0, 80.0, 81.0])
    events = pd.DataFrame({"timestamp": times})

    kept = filter_by_intervals(events, trials)

    # pynapple intervals include their end, so 20.0 and 80.0 are inside, as
    # in pynapple's own restrict.
    assert kept["timestamp"].tolist() == [10.0, 20.0, 80.0]
    np.testing.assert_array_equal(kept["timestamp"], nap.Ts(times).restrict(trials).t)


def test_intervals_to_events_accepts_an_intervalset(trials):
    from neurospatial.events import intervals_to_events

    starts = intervals_to_events(trials, preserve_fields=["kind"])
    both = intervals_to_events(trials, which="both")

    assert starts["timestamp"].tolist() == [5.0, 50.0]
    assert starts["kind"].tolist() == ["a", "b"]
    # Each stop is the half-open stop holding the inclusive end.
    stops = both.loc[both["boundary"] == "stop", "timestamp"].to_numpy()
    np.testing.assert_array_equal(stops, np.nextafter([20.0, 80.0], np.inf))


def test_intervals_to_events_reads_interval_ids_from_the_metadata(nap):
    from neurospatial.events import intervals_to_events

    iset = nap.IntervalSet(
        start=[5.0, 50.0], end=[20.0, 80.0], metadata={"interval_id": [8, 3]}
    )

    events = intervals_to_events(iset)

    assert events["interval_id"].tolist() == [8, 3]


def test_from_pynapple_reads_an_intervalset_as_intervals(nap, trials):
    """Metadata become fields; a sample at pynapple's inclusive end is inside."""
    from neurospatial import Intervals
    from neurospatial.io import from_pynapple

    intervals = from_pynapple(trials)

    assert isinstance(intervals, Intervals)
    assert intervals.fields == ("kind",)
    assert intervals.kind.tolist() == ["a", "b"]
    assert intervals.interval_ids.tolist() == [0, 1]
    np.testing.assert_array_equal(intervals.start, [5.0, 50.0])
    times = np.array([3.0, 10.0, 20.0, 30.0, 80.0, 81.0])
    ids = intervals.sample_interval_ids(times)
    assert ids.tolist() == [-1, 0, 0, -1, 1, -1]
    np.testing.assert_array_equal(times[ids >= 0], nap.Ts(times).restrict(trials).t)


def test_from_pynapple_reads_missing_text_as_none(nap):
    """An IntervalSet loaded from NWB holds "" where a value is missing."""
    from neurospatial.io import from_pynapple

    iset = nap.IntervalSet(
        start=[0.0, 5.0], end=[2.0, 7.0], metadata={"goal": ["left", ""]}
    )

    assert from_pynapple(iset).goal.tolist() == ["left", None]


def test_to_pynapple_then_from_pynapple_keeps_ids_and_fields():
    from neurospatial import Intervals
    from neurospatial.io import from_pynapple, to_pynapple

    trials = Intervals.from_arrays(
        [4.0, 0.0, 10.0],
        [6.0, 2.5, 12.0],
        interval_ids=[7, 3, 9],
        outcome=["timeout", "reached", None],
        n_licks=[2, 5, 0],
    )

    back = from_pynapple(to_pynapple(trials))

    pd.testing.assert_frame_equal(back.intervals_table(), trials.intervals_table())
    np.testing.assert_array_equal(back.windows, trials.windows)


def test_pairwise_bias_matrix_accepts_a_one_row_intervalset(nap, group):
    from neurospatial.replay import pairwise_bias_matrix

    event = nap.IntervalSet(start=1.0, end=30.0)

    from_set = pairwise_bias_matrix(group, event)
    from_pair = pairwise_bias_matrix(group, (1.0, np.nextafter(30.0, np.inf)))

    np.testing.assert_array_equal(from_set.bias, from_pair.bias)
    np.testing.assert_array_equal(from_set.n_spikes, from_pair.n_spikes)
    # The group's labels name the matrix rows.
    assert from_set.unit_ids.tolist() == [5, 6, 7]


def test_one_event_interval_refuses_several_rows(nap, group):
    from neurospatial.replay import pairwise_bias_matrix

    with pytest.raises(ValueError, match=r"has 2 intervals"):
        pairwise_bias_matrix(group, nap.IntervalSet(start=[1.0, 5.0], end=[2.0, 6.0]))


def test_mean_pairwise_bias_accepts_an_intervalset(nap):
    from neurospatial.replay import mean_pairwise_bias

    # Unit 0 fires at 1.0 and exactly at the cycle's inclusive end, 9.0.
    spikes = [np.array([1.0, 9.0]), np.array([2.0])]
    cycles = nap.IntervalSet(start=[0.0], end=[9.0])

    # Unit 0 fires once before and once after unit 1: no bias. Read as a
    # half-open [0, 9) the spike at 9.0 would be lost and the bias would be 1.
    np.testing.assert_array_equal(
        mean_pairwise_bias(spikes, cycles), [[0.0, 0.0], [0.0, 0.0]]
    )


# ---------------------------------------------------------------------------
# NWB trials read by neurospatial and by pynapple
# ---------------------------------------------------------------------------


def test_trial_stop_closedness_routes_and_how_to_match_them(nap, tmp_path):
    """The documented recipes give one selection on both routes."""
    pytest.importorskip("pynwb")
    from datetime import datetime, timezone

    from pynwb import NWBHDF5IO, NWBFile

    from neurospatial import Intervals
    from neurospatial.behavior import in_epochs
    from neurospatial.io.nwb import read_trials, write_trials

    times = np.round(np.arange(1000) * 0.01, 10)
    nwbfile = NWBFile(
        session_description="trials",
        identifier="closedness",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    # Each trial stops exactly on a sample.
    write_trials(nwbfile, Intervals.from_arrays(times[[100, 400]], times[[300, 700]]))
    path = tmp_path / "trials.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(str(path), "r") as io:
        trials = read_trials(io.read())
    rows = np.column_stack([trials.start, trials.stop])
    iset = nap.load_file(str(path))["trials"]

    half_open = in_epochs(times, trials)
    inclusive = in_epochs(times, iset)
    assert half_open.sum() == 500
    assert inclusive.sum() == len(nap.Ts(times).restrict(iset)) == 502

    # Half-open meaning from pynapple's table; inclusive meaning from rows.
    np.testing.assert_array_equal(
        in_epochs(times, np.column_stack([iset.start, iset.end])), half_open
    )
    np.testing.assert_array_equal(in_epochs(times, rows, closed="both"), inclusive)


# ---------------------------------------------------------------------------
# Positions-only entry points and pynapple timestamps
# ---------------------------------------------------------------------------


def test_from_samples_refuses_permuted_columns_and_reads_ordered_frames(nap, session):
    _, times, positions, _ = session
    swapped = nap.TsdFrame(t=times, d=positions[:, ::-1], columns=["y", "x"])
    ordered = nap.TsdFrame(t=times, d=positions, columns=["x", "y"])

    with pytest.raises(ValueError, match=r"not in x, y, z order"):
        Environment.from_samples(swapped, bin_size=10.0)
    from_frame = Environment.from_samples(ordered, bin_size=10.0)
    from_array = Environment.from_samples(positions, bin_size=10.0)
    np.testing.assert_array_equal(from_frame.bin_centers, from_array.bin_centers)


def test_bin_at_refuses_permuted_columns_and_reads_ordered_frames(nap, session):
    env, times, positions, _ = session
    swapped = nap.TsdFrame(t=times, d=positions[:, ::-1], columns=["y", "x"])
    ordered = nap.TsdFrame(t=times, d=positions, columns=["x", "y"])

    fix = re.escape("positions=tsdframe.loc[['x', 'y']]")
    with pytest.raises(ValueError, match=fix):
        env.bin_at(swapped)
    np.testing.assert_array_equal(env.bin_at(ordered), env.bin_at(positions))


def test_pynapple_times_are_read_as_timestamps_not_values(nap, session):
    """``np.asarray(tsd)`` is the values; the timestamps must be used."""
    from neurospatial import compute_spatial_rates
    from neurospatial.behavior import compute_speed

    env, times, positions, trains = session
    frame = nap.TsdFrame(t=times, d=positions, columns=["x", "y"])
    # A speed trace: its values are not timestamps.
    speed = nap.Tsd(t=times, d=compute_speed(times, positions))

    plain = compute_spatial_rates(env, trains, times, positions, method="binned")
    for clock in (frame, speed, nap.Ts(t=times)):
        rates = compute_spatial_rates(env, trains, clock, positions, method="binned")
        np.testing.assert_array_equal(rates.occupancy, plain.occupancy)
        np.testing.assert_array_equal(rates.firing_rates, plain.firing_rates)
    np.testing.assert_array_equal(
        env.occupancy(frame, frame), env.occupancy(times, positions)
    )
    np.testing.assert_array_equal(
        compute_speed(speed, positions), compute_speed(times, positions)
    )


# ---------------------------------------------------------------------------
# One unit's pynapple train as spike_times, and other time/position arguments
# ---------------------------------------------------------------------------


def test_one_units_tsd_spike_times_are_read_as_timestamps(nap, session):
    """A ``Tsd`` of spike amplitudes must not put every spike at its value."""
    from neurospatial import compute_spatial_rate
    from neurospatial.encoding import compute_directional_rate
    from neurospatial.events import peri_event_histogram

    env, times, positions, trains = session
    spikes = trains[0]
    headings = np.arctan2(np.gradient(positions[:, 1]), np.gradient(positions[:, 0]))
    for train in (nap.Tsd(t=spikes, d=np.full(spikes.size, 3.0)), nap.Ts(spikes)):
        np.testing.assert_array_equal(
            compute_spatial_rate(env, train, times, positions).firing_rate,
            compute_spatial_rate(env, spikes, times, positions).firing_rate,
        )
        np.testing.assert_array_equal(
            compute_directional_rate(train, times, headings).firing_rate,
            compute_directional_rate(spikes, times, headings).firing_rate,
        )
        events = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(
            peri_event_histogram(train, events, (-1.0, 1.0), bin_size=0.1).histogram,
            peri_event_histogram(spikes, events, (-1.0, 1.0), bin_size=0.1).histogram,
        )


def test_decoding_metrics_refuse_permuted_position_frames(nap, session):
    from neurospatial.decoding import decoding_error

    env, times, positions, _ = session
    actual = positions[:100]
    decoded = positions[1:101]
    ordered = nap.TsdFrame(t=times[:100], d=decoded, columns=["x", "y"])
    swapped = nap.TsdFrame(t=times[:100], d=decoded[:, ::-1], columns=["y", "x"])

    np.testing.assert_array_equal(
        decoding_error(env, ordered, actual, metric="euclidean"),
        decoding_error(env, decoded, actual, metric="euclidean"),
    )
    with pytest.raises(ValueError, match=r"decoded_positions=tsdframe"):
        decoding_error(env, swapped, actual, metric="euclidean")


def test_interval_set_with_list_metadata_reads_and_filters(nap):
    """Array-valued metadata (NWB tags loaded by pynapple) becomes tuples."""
    from neurospatial.events import filter_by_intervals
    from neurospatial.io import from_pynapple

    iset = nap.IntervalSet(start=[0.0, 10.0], end=[5.0, 15.0])
    tags = np.empty(2, dtype=object)  # one array per row, as pynapple loads tags
    tags[0], tags[1] = np.array(["run", "maze"]), np.array(["sleep"])
    iset.set_info(tags=tags)
    events = pd.DataFrame({"timestamp": [1.0, 5.0, 7.0, 12.0]})
    assert filter_by_intervals(events, iset)["timestamp"].tolist() == [1.0, 5.0, 12.0]
    intervals = from_pynapple(iset)
    assert intervals.tags.tolist() == [("run", "maze"), ("sleep",)]


def test_interval_set_metadata_that_cannot_be_a_field_is_left_out(nap):
    from neurospatial.io import from_pynapple

    iset = nap.IntervalSet(start=[0.0, 10.0], end=[5.0, 15.0])
    with pytest.warns(UserWarning, match="metadata"):
        iset.set_info(**{"trial type": ["a", "b"]})
    iset.set_info(goal=["left", "right"])
    with pytest.warns(UserWarning, match=r"\['trial type'\].*left out"):
        intervals = from_pynapple(iset)
    assert intervals.fields == ("goal",)


def test_from_pynapple_reads_integral_float_interval_ids(nap):
    """pynapple metadata can store integer ids as floats; they stay the ids."""
    from neurospatial.io import from_pynapple

    iset = nap.IntervalSet(
        start=[0.0, 5.0, 9.0],
        end=[2.0, 7.0, 11.0],
        metadata={"interval_id": [7.0, 12.0, 9.0]},
    )

    intervals = from_pynapple(iset)

    assert intervals.interval_ids.tolist() == [7, 12, 9]
    assert intervals.interval_ids.dtype == np.int64
    assert intervals.fields == ()


@pytest.mark.parametrize(
    ("ids", "problem"),
    [([7.5, 9.0, 12.0], "not whole numbers"), ([7, 7, 9], "repeat")],
)
def test_from_pynapple_refuses_ids_it_cannot_keep(nap, ids, problem):
    """Bad ids raise instead of being silently renumbered 0 to n - 1."""
    from neurospatial.io import from_pynapple

    iset = nap.IntervalSet(
        start=[0.0, 5.0, 9.0], end=[2.0, 7.0, 11.0], metadata={"interval_id": ids}
    )

    with pytest.raises(ValueError, match=problem) as exc:
        from_pynapple(iset)
    assert "Fix:" in str(exc.value)
