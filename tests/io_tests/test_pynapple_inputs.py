"""pynapple objects passed straight to analysis functions.

A ``TsGroup`` keeps its unit labels and metadata, a ``TsdFrame`` or ``Tsd``
passed as ``positions`` or ``headings`` keeps its time support and column
order, and ``IntervalSet`` intervals keep their inclusive ends. Each test
compares the pynapple route with the plain-array route it must equal.
"""

from __future__ import annotations

import importlib.util

import numpy as np
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
    times, positions = gapped.t, gapped.values

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


def test_explicit_epochs_win_over_the_time_support(gapped, session):
    env = session[0]

    narrowed = env.occupancy(gapped.t, gapped, epochs=(0.0, 10.0))

    assert narrowed.sum() == pytest.approx(10.0)


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
    values = gapped.values
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

    starts = intervals_to_events(trials, preserve_columns=["kind"])
    both = intervals_to_events(trials, which="both")

    assert starts["timestamp"].tolist() == [5.0, 50.0]
    assert starts["kind"].tolist() == ["a", "b"]
    # Each stop is the half-open stop holding the inclusive end.
    stops = both.loc[both["boundary"] == "stop", "timestamp"].to_numpy()
    np.testing.assert_array_equal(stops, np.nextafter([20.0, 80.0], np.inf))


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

    from neurospatial.behavior import in_epochs
    from neurospatial.io.nwb import read_trials, write_trials

    times = np.round(np.arange(1000) * 0.01, 10)
    nwbfile = NWBFile(
        session_description="trials",
        identifier="closedness",
        session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    # Each trial stops exactly on a sample.
    write_trials(nwbfile, start_times=times[[100, 400]], stop_times=times[[300, 700]])
    path = tmp_path / "trials.nwb"
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(str(path), "r") as io:
        rows = read_trials(io.read())[["start_time", "stop_time"]].to_numpy()
    iset = nap.load_file(str(path))["trials"]

    half_open = in_epochs(times, rows)
    inclusive = in_epochs(times, iset)
    assert half_open.sum() == 500
    assert inclusive.sum() == len(nap.Ts(times).restrict(iset)) == 502

    # Half-open meaning from pynapple's table; inclusive meaning from rows.
    np.testing.assert_array_equal(
        in_epochs(times, np.column_stack([iset.start, iset.end])), half_open
    )
    np.testing.assert_array_equal(in_epochs(times, rows, closed="both"), inclusive)
