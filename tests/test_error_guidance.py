"""Errors at the public boundary name the call to make.

Each test passes one wrong input of a kind a scientist commonly passes (a
population where one unit is expected, a result object where an array is
expected, a typo, times in milliseconds) and checks that the error names the
function or argument that fixes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import neurospatial
from neurospatial import (
    InputError,
    NeurospatialError,
    compute_spatial_rate,
    compute_spatial_rates,
    decode_position,
)
from neurospatial.decoding import BayesianDecoder, bin_spikes_in_time, decode_session
from neurospatial.encoding import (
    is_place_cell,
    population_coverage,
    spatial_information,
)
from neurospatial.simulation import (
    linear_track_session,
    open_field_session,
    simulate_session,
)


@pytest.fixture(scope="module")
def session():
    return open_field_session(duration=60.0, n_place_cells=3, rng=0)


@pytest.fixture(scope="module")
def rates(session):
    return compute_spatial_rates(
        session.env, session.spike_times, session.times, session.positions
    )


@pytest.fixture(scope="module")
def counts(session):
    spike_counts, _ = bin_spikes_in_time(
        session.spike_times,
        dt=0.025,
        t_start=session.times[0],
        t_stop=session.times[-1],
    )
    return spike_counts


def test_input_error_is_value_error_and_neurospatial_error(session):
    assert issubclass(InputError, ValueError)
    assert issubclass(InputError, NeurospatialError)
    assert neurospatial.InputError is InputError
    # The shared trajectory validator raises it.
    with pytest.raises(InputError, match="times"):
        compute_spatial_rate(
            session.env,
            session.spike_times[0],
            session.times[:-5],
            session.positions,
        )


def test_list_of_trains_to_single_unit_function(session, rates):
    env, times, positions = session.env, session.times, session.positions
    trains = list(session.spike_times[:2])
    with pytest.raises(TypeError, match=r"got 2 trains") as caught:
        compute_spatial_rate(env, trains, times, positions)
    assert "compute_spatial_rates(" in str(caught.value)
    assert "spike_times[i]" in str(caught.value)
    with pytest.raises(TypeError, match=r"got 2 trains") as caught:
        is_place_cell(env, trains, times, positions, criterion="threshold")
    assert "is_place_cell()" in str(caught.value)
    assert "compute_spatial_rates(" in str(caught.value)
    # decode_position takes a count matrix; the fix bins the trains first.
    with pytest.raises(TypeError, match=r"got 2 spike trains") as caught:
        decode_position(env, trains, rates, dt=0.025)
    assert "bin_spikes_in_time(spike_times, dt=" in str(caught.value)
    # Equal-length trains stacked as a 2-D array get the same message.
    stacked = np.array([session.spike_times[0][:5], session.spike_times[1][:5]])
    with pytest.raises(TypeError, match=r"got 2 trains") as caught:
        compute_spatial_rate(env, stacked, times, positions)
    assert "compute_spatial_rates(" in str(caught.value)


def test_result_object_where_array_expected(session, rates, counts):
    single = rates[0]
    with pytest.raises(TypeError) as caught:
        spatial_information(single, occupancy=single.occupancy)
    assert "r.spatial_information()" in str(caught.value)
    assert "SpatialRateResult" in str(caught.value)
    with pytest.raises(TypeError) as caught:
        decode_position(session.env, counts, single, dt=0.025)
    assert "compute_spatial_rates" in str(caught.value)
    assert "population" in str(caught.value)
    with pytest.raises(TypeError) as caught:
        population_coverage(session.env, rates, bandwidth=5.0)
    assert ".firing_rates" in str(caught.value)


def test_fit_count_matrix_hint(session, counts):
    decoder = BayesianDecoder(session.env)
    assert np.issubdtype(np.asarray(counts).dtype, np.integer)
    with pytest.raises(TypeError) as caught:
        decoder.fit(counts, session.times, session.positions)
    message = str(caught.value)
    assert "decode_position" in message
    assert "count matrix of shape (n_time_bins, n_neurons)" in message
    assert "MATLAB" not in message
    # A 0/1 raster gets the more precise raster refusal.
    raster = (np.asarray(counts) > 0).astype(np.int64)
    with pytest.raises(ValueError, match="spike raster"):
        decoder.fit(raster, session.times, session.positions)


def test_all_intervals_excluded_raises(session):
    with pytest.raises(InputError, match="probably not in seconds") as caught:
        compute_spatial_rate(
            session.env,
            session.spike_times[0] * 1000,
            session.times * 1000,
            session.positions,
        )
    # The positions are fine, so the message does not blame them.
    assert "positions" not in str(caught.value)
    assert "out of bounds" not in str(caught.value)
    # warn_on_drop silences warnings, not a refusal of unusable input.
    with pytest.raises(InputError, match="probably not in seconds"):
        compute_spatial_rate(
            session.env,
            session.spike_times[0] * 1000,
            session.times * 1000,
            session.positions,
            warn_on_drop=False,
        )


def test_all_intervals_excluded_raises_for_datetime_index(session):
    pd = pytest.importorskip("pandas")
    times = pd.Timestamp("2026-01-01") + pd.to_timedelta(session.times, unit="s")
    assert isinstance(times, pd.DatetimeIndex)
    with pytest.raises(InputError, match="probably not in seconds") as caught:
        compute_spatial_rate(
            session.env, session.spike_times[0], times, session.positions
        )
    assert "positions" not in str(caught.value)


def test_simulator_kwarg_typo_names_caller(session):
    with pytest.raises(TypeError) as caught:
        open_field_session(n_plce_cells=3, rng=0)
    message = str(caught.value)
    assert "open_field_session()" in message
    assert "'n_plce_cells'" in message
    assert "Did you mean 'n_place_cells'?" in message
    assert "simulate_trajectory_ou" not in message
    with pytest.raises(TypeError) as caught:
        linear_track_session(n_plce_cells=3, rng=0)
    assert "linear_track_session()" in str(caught.value)
    assert "Did you mean 'n_place_cells'?" in str(caught.value)
    with pytest.raises(TypeError) as caught:
        simulate_session(session.env, duration=5.0, speed_maen=5.0, rng=0)
    assert "simulate_session()" in str(caught.value)
    assert "Did you mean 'speed_mean'?" in str(caught.value)


def test_decode_session_kwarg_typo_did_you_mean(session):
    with pytest.raises(TypeError, match=r"Did you mean 'bandwidth'\?") as caught:
        decode_session(
            session.env,
            session.spike_times,
            session.times,
            session.positions,
            bandwdth=5.0,
        )
    assert "decode_session()" in str(caught.value)


def test_invalid_method_did_you_mean(session):
    with pytest.raises(ValueError, match=r"Did you mean 'gaussian_kde'\?") as caught:
        compute_spatial_rate(
            session.env,
            session.spike_times[0],
            session.times,
            session.positions,
            method="gaussian",
        )
    assert "compute_spatial_rate()" in str(caught.value)


def test_no_uppercase_why_labels():
    src = Path(neurospatial.__file__).parent
    offenders = [
        f"{path.relative_to(src)}:{number}"
        for path in sorted(src.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "WHY:" in line or "WHAT:" in line
    ]
    assert offenders == []


def test_other_population_forms_to_single_unit_function(session):
    env, times, positions = session.env, session.times, session.positions
    trains = [np.asarray(t) for t in session.spike_times[:2]]
    ragged = np.empty(2, dtype=object)
    ragged[:] = trains
    with pytest.raises(TypeError, match=r"got 2 trains"):
        compute_spatial_rate(env, ragged, times, positions)
    nap = pytest.importorskip("pynapple")
    with pytest.raises(TypeError, match=r"got 2 trains"):
        compute_spatial_rate(env, [nap.Ts(t) for t in trains], times, positions)


def test_spike_table_is_a_shape_problem_not_trains(session):
    """An (n_spikes, 2) table of spike time and unit is not n trains."""
    trains = session.spike_times[:2]
    table = np.column_stack(
        [np.concatenate(trains), np.repeat([0.0, 1.0], [len(t) for t in trains])]
    )
    with pytest.raises(ValueError, match=r"must be 1-D") as caught:
        compute_spatial_rate(session.env, table, session.times, session.positions)
    assert "table[:, 1] == u" in str(caught.value)


def test_shape_edge_cases_keep_their_precise_messages(session, rates, counts):
    env, times, positions = session.env, session.times, session.positions
    train = np.asarray(session.spike_times[0])
    # A (1, n) row vector, as from scipy.io.loadmat, is one unit on a row.
    with pytest.raises(ValueError, match=r"must be 1-D") as caught:
        compute_spatial_rate(env, train[None, :], times, positions)
    assert "np.ravel(spike_times)" in str(caught.value)
    # Fractional counts as a list of lists are counts, not trains.
    fractional = (np.asarray(counts) * 0.5).tolist()
    with pytest.raises(ValueError, match="fractional entr"):
        decode_position(env, fractional, rates, dt=0.025)
    # Zero-padded trains get the padding fix, not the count-matrix one.
    padded = np.array([[1.0, 5.0, 9.0, 20.0], [2.0, 3.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="padded with zeros"):
        BayesianDecoder(env).fit(padded, times, positions)


def test_rectangular_list_of_counts_is_a_count_matrix(session):
    """A nested list of whole-number counts decodes like the same ndarray.

    Every row here increases, as spike trains do, but the rows have one
    length, hold non-negative whole numbers and are two neurons long: two
    neurons' counts rise in every time bin by chance one time in four.
    """
    two_units = compute_spatial_rates(
        session.env, session.spike_times[:2], session.times, session.positions
    )
    nested = [[0, 1], [1, 3], [0, 2]]
    from_list = decode_position(session.env, nested, two_units, dt=0.1)
    from_array = decode_position(session.env, np.array(nested), two_units, dt=0.1)
    np.testing.assert_array_equal(from_list.posterior, from_array.posterior)


def test_three_neuron_counts_that_do_not_all_increase_decode(session, rates):
    """With three neurons, a count list decodes unless every row increases."""
    nested = [[0, 1, 3], [2, 1, 0], [0, 0, 1]]
    from_list = decode_position(session.env, nested, rates, dt=0.1)
    from_array = decode_position(session.env, np.array(nested), rates, dt=0.1)
    np.testing.assert_array_equal(from_list.posterior, from_array.posterior)


def test_one_time_bin_of_rising_whole_counts_decodes(session, rates):
    """One row is one time bin of counts; three neurons may rise by chance."""
    one_bin = [[0, 1, 3]]
    from_list = decode_position(session.env, one_bin, rates, dt=0.1)
    from_array = decode_position(session.env, np.array(one_bin), rates, dt=0.1)
    np.testing.assert_array_equal(from_list.posterior, from_array.posterior)


def test_equal_length_spike_trains_are_still_refused_as_counts(session, rates):
    # Equal-length trains in seconds are fractional: still spike trains.
    trains = [np.asarray(train[:5]) for train in session.spike_times]
    assert len({train.size for train in trains}) == 1
    with pytest.raises(TypeError, match="got 3 spike trains"):
        decode_position(session.env, trains, rates, dt=0.1)
    one_spike_each = [[0.5], [1.2], [3.3]]
    with pytest.raises(TypeError, match="got 3 spike trains"):
        decode_position(session.env, one_spike_each, rates, dt=0.1)


def test_equal_length_whole_number_trains_are_refused_as_counts(session, rates):
    """Spike trains as sample indices: whole numbers, one length, all rising.

    Three or more neurons whose counts rise in every time bin are implausible
    (one time in six per bin for three neurons), so these rows are trains.
    """
    sample_indices = [[100, 250, 400], [120, 300, 500], [10, 20, 30]]
    with pytest.raises(TypeError, match="got 3 spike trains") as error:
        decode_position(session.env, sample_indices, rates, dt=0.1)
    # A count matrix that does rise in every bin decodes as an ndarray.
    assert "np.asarray" in str(error.value)
    decoded = decode_position(session.env, np.array(sample_indices), rates, dt=0.1)
    assert decoded.posterior.shape == (3, session.env.n_bins)


def test_list_of_pynapple_trains_passed_as_counts(session, rates):
    nap = pytest.importorskip("pynapple")
    trains = [nap.Ts(np.asarray(t)) for t in session.spike_times]
    with pytest.raises(TypeError, match="got 3 spike trains"):
        decode_position(session.env, trains, rates, dt=0.1)


def test_bin_spike_trains_accepts_one_epoch_as_an_array(session):
    from neurospatial.encoding._spatial_counts import bin_spike_trains

    counts, _ = bin_spike_trains(
        session.env,
        [np.asarray(session.spike_times[0])],
        session.times,
        session.positions,
        epochs=np.array([0.0, 30.0]),
    )
    assert counts.shape == (1, session.env.n_bins)


@pytest.mark.parametrize(
    "call",
    [
        lambda s, r, c: compute_spatial_rate(
            s.spike_times[0], s.times, s.positions, s.env
        ),
        lambda s, r, c: decode_position(c, s.env, r, dt=0.025),
        lambda s, r, c: is_place_cell(
            s.spike_times[0], s.env, s.times, s.positions, criterion="threshold"
        ),
    ],
    ids=["env-last", "decode-env-second", "place-cell-env-second"],
)
def test_misplaced_environment_says_move_it_first(session, rates, counts, call):
    """Each call fills every positional slot, so it binds; the Environment in
    a later slot is named, not a raw NumPy error from the data in its place."""
    with pytest.raises(TypeError, match="not the first positional argument") as caught:
        call(session, rates, counts)
    assert "with env first" in str(caught.value)
    assert "Environment.from_samples" not in str(caught.value)


def test_missing_environment_advice_names_units(session):
    """Building an environment without units= would trigger the units-free
    bandwidth warning, so the advice to build one includes it."""
    data = np.asarray(session.positions)
    with pytest.raises(TypeError, match="expects an Environment") as caught:
        compute_spatial_rate(data, session.spike_times[0], session.times, data)
    assert 'Environment.from_samples(positions, bin_size=2.0, units="cm")' in str(
        caught.value
    )
