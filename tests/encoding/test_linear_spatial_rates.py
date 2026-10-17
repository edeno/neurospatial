"""Spatial encoding shares exact linear support between counts and dwell time."""

import copy
import pickle

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment, compute_spatial_rate, compute_spatial_rates
from neurospatial.decoding import BayesianDecoder


def holed_cell():
    parent = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([0.0, 6.0]), np.array([0.0, 2.0])),
    )
    return parent.clip(
        Polygon(
            box(0, 0, 6, 2).exterior.coords, holes=[box(2, 0.5, 4, 1.5).exterior.coords]
        )
    )


def test_linear_rates_match_hand_counted_entry_hole_reentry_and_decode():
    env = holed_cell()
    times = np.array([0.0, 8.0])
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    spikes = np.arange(0.5, 8, 1.0)
    options = {
        "method": "binned",
        "bandwidth": 0,
        "time_allocation": "linear",
        "max_gap": None,
        "warn_on_drop": False,
    }
    single = compute_spatial_rate(env, spikes, times, positions, **options)
    rates = compute_spatial_rates(
        env,
        [spikes, np.array([1.5, 5.5]), np.array([])],
        times,
        positions,
        **options,
    )
    np.testing.assert_allclose(single.occupancy, [4.0])
    np.testing.assert_allclose(single.firing_rate, [1.0])
    np.testing.assert_allclose(rates.occupancy, [4.0])
    np.testing.assert_allclose(rates.firing_rates, [[1.0], [0.5], [0.0]])
    np.testing.assert_allclose(
        rates.occupancy,
        env.occupancy(times, positions, max_gap=None, time_allocation="linear"),
    )
    assert rates.analysis_settings["time_allocation"] == "linear"
    np.testing.assert_allclose(
        rates.analysis_settings["observed_intervals"], [[1, 3], [5, 7]]
    )
    decoded = BayesianDecoder.from_rates(rates, dt=1).predict(
        [spikes, np.array([1.5, 5.5]), np.array([])],
        np.arange(0.0, 8.25, 0.25),
    )
    np.testing.assert_allclose(decoded.posterior, 1.0)


@pytest.mark.parametrize("function", [compute_spatial_rate, compute_spatial_rates])
def test_spatial_allocation_option_rejects_unknown_value(function):
    env = holed_cell()
    spikes = np.array([0.5]) if function is compute_spatial_rate else [np.array([0.5])]
    with pytest.raises(ValueError, match="time_allocation"):
        function(
            env,
            spikes,
            np.array([0.0, 1.0]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            time_allocation="other",
        )


def grid():
    return Environment.from_grid_mask(
        np.ones((2, 1), bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 1.0])),
        connect_diagonal_neighbors=False,
    )


@pytest.mark.parametrize("method", ["binned", "gaussian_kde", "diffusion_kde", "glm"])
def test_linear_single_batch_agree_for_each_rate_method(method):
    env = grid()
    times = np.array([0.0, 2.0])
    positions = np.array([[0.25, 0.5], [1.75, 0.5]])
    trains = [np.array([0.25, 0.75, 1.25, 1.75]), np.array([])]
    options = {
        "method": method,
        "time_allocation": "linear",
        "max_gap": None,
        "warn_on_drop": False,
    }
    options.update(
        {"penalty": 1.0, "rank": 2} if method == "glm" else {"bandwidth": 0.4}
    )
    batch = compute_spatial_rates(env, trains, times, positions, **options)
    np.testing.assert_allclose(batch.occupancy, [1.0, 1.0])
    for index, spikes in enumerate(trains):
        single = compute_spatial_rate(env, spikes, times, positions, **options)
        np.testing.assert_allclose(single.occupancy, batch.occupancy)
        np.testing.assert_allclose(
            single.firing_rate, batch.firing_rates[index], atol=1e-12
        )
    assert batch.analysis_settings["time_allocation"] == "linear"
    np.testing.assert_array_equal(
        batch.analysis_settings["observed_intervals"], [[0, 2]]
    )


@pytest.mark.parametrize("method", ["binned", "glm"])
def test_linear_zero_neurons_keeps_shared_occupancy(method):
    env = grid()
    options = {
        "method": method,
        "max_gap": None,
        "time_allocation": "linear",
        "warn_on_drop": False,
    }
    if method == "glm":
        options.update(penalty=1.0, rank=2)
    else:
        options["bandwidth"] = 0
    result = compute_spatial_rates(
        env,
        [],
        np.array([0.0, 2.0]),
        np.array([[0.25, 0.5], [1.75, 0.5]]),
        **options,
    )
    assert result.firing_rates.shape == (0, 2)
    np.testing.assert_allclose(result.occupancy, [1.0, 1.0])
    assert result.analysis_settings["time_allocation"] == "linear"


def test_linear_entry_included_exit_excluded_and_internal_boundary_owned_by_locator():
    env = holed_cell()
    result = compute_spatial_rate(
        env,
        np.array([1.0, 3.0, 5.0, 7.0, 8.0]),
        np.array([0.0, 8.0]),
        np.array([[-1.0, 1.0], [7.0, 1.0]]),
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [4.0])
    np.testing.assert_allclose(result.firing_rate, [0.5])
    boundary = compute_spatial_rate(
        grid(),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0]),
        np.array([[0.0, 0.5], [2.0, 0.5]]),
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(boundary.occupancy, [1.0, 1.0])
    np.testing.assert_allclose(boundary.firing_rate, [1.0, 1.0])


@pytest.mark.parametrize("reverse", [False, True])
def test_linear_half_open_span_entries_follow_time_direction(reverse):
    parent = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([-2.0, 8.0]), np.array([0.0, 2.0])),
    )
    env = parent.clip(
        Polygon(
            box(0, 0, 6, 2).exterior.coords,
            holes=[box(2, 0.5, 4, 1.5).exterior.coords],
        )
    )
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    if reverse:
        positions = positions[::-1]
    result = compute_spatial_rate(
        env,
        np.array([1.0, 3.0, 5.0, 7.0, 8.0]),
        np.array([0.0, 8.0]),
        positions,
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [4.0])
    np.testing.assert_allclose(result.firing_rate, [0.5])
    np.testing.assert_array_equal(
        result.analysis_settings["observed_intervals"], [[1, 3], [5, 7]]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_linear_internal_boundary_owner_does_not_depend_on_motion_direction(reverse):
    positions = np.array([[0.0, 0.5], [2.0, 0.5]])
    if reverse:
        positions = positions[::-1]
    result = compute_spatial_rate(
        grid(),
        np.array([1.0]),
        np.array([0.0, 2.0]),
        positions,
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [1.0, 1.0])
    np.testing.assert_allclose(result.firing_rate, [0.0, 1.0])


def test_linear_native_exclusive_upper_edge_keeps_its_inherited_owner():
    result = compute_spatial_rate(
        grid(),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 4.0]),
        np.array([[3.0, 0.5], [-1.0, 0.5]]),
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [1.0, 1.0])
    np.testing.assert_allclose(result.firing_rate, [0.0, 1.0])


def test_linear_speed_windows_and_provenance_snapshot_share_retained_support():
    env = holed_cell()
    times = np.array([0.0, 4.0, 8.0])
    positions = np.array([[-1.0, 1.0], [3.0, 1.0], [7.0, 1.0]])
    epochs = np.array([[0.0, 8.0]])
    spike_window = np.array([[0.0, 4.0]])
    result = compute_spatial_rates(
        env,
        [np.array([1.0, 2.0, 3.0, 5.0, 6.0])],
        times,
        positions,
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        speed=np.array([2.0, 0.0, 0.0]),
        min_speed=1.0,
        epochs=epochs,
        spike_window=spike_window,
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [2.0])
    np.testing.assert_allclose(result.firing_rates, [[1.0]])
    expected = copy.deepcopy(dict(result.analysis_settings))
    epochs[:] = -100
    spike_window[:] = -100
    np.testing.assert_array_equal(
        result.analysis_settings["observed_intervals"], [[1, 3]]
    )
    assert dict(result.analysis_settings) == expected
    for restored in [result, copy.deepcopy(result), pickle.loads(pickle.dumps(result))]:
        assert not restored.occupancy.flags.writeable
        assert restored.analysis_settings["time_allocation"] == "linear"
        exported = restored.analysis_settings
        exported["time_allocation"] = "start"
        exported["observed_intervals"][0][0] = -100
        assert restored.analysis_settings["time_allocation"] == "linear"
        assert restored.analysis_settings["observed_intervals"] == [[1, 3]]


@pytest.mark.parametrize("method", ["binned", "glm"])
@pytest.mark.parametrize("n_units", [0, 2])
def test_linear_interval_preparation_happens_once_per_public_batch(
    monkeypatch, method, n_units
):
    from neurospatial.encoding import _spatial_counts

    original = _spatial_counts.prepare_linear_spatial_intervals
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(_spatial_counts, "prepare_linear_spatial_intervals", record)
    options = {
        "method": method,
        "time_allocation": "linear",
        "max_gap": None,
        "warn_on_drop": False,
    }
    if method == "glm":
        options.update(penalty=1.0, rank=2)
    else:
        options["bandwidth"] = 0
    compute_spatial_rates(
        grid(),
        [np.array([0.5, 1.5])] * n_units,
        np.array([0.0, 2.0]),
        np.array([[0.25, 0.5], [1.75, 0.5]]),
        **options,
    )
    assert len(calls) == 1


@pytest.mark.parametrize("function", [compute_spatial_rate, compute_spatial_rates])
def test_default_allocation_preserves_explicit_start_results(function):
    env = grid()
    spikes = np.array([0.25, 0.75, 1.25, 1.75])
    if function is compute_spatial_rates:
        spikes = [spikes]
    args = (env, spikes, np.array([0.0, 2.0]), np.array([[0.25, 0.5], [1.75, 0.5]]))
    options = {
        "method": "binned",
        "bandwidth": 0,
        "max_gap": None,
        "warn_on_drop": False,
    }
    default = function(*args, **options)
    explicit = function(*args, time_allocation="start", **options)
    np.testing.assert_array_equal(default.occupancy, explicit.occupancy)
    np.testing.assert_array_equal(default._get_rates(), explicit._get_rates())
    assert default.analysis_settings == explicit.analysis_settings
    np.testing.assert_array_equal(default.occupancy, [2.0, 0.0])
    np.testing.assert_array_equal(default._get_rates().reshape(-1), [1.0, np.nan])
    assert default.analysis_settings["observed_intervals"] == [[0.0, 2.0]]


def test_linear_spike_trains_preserve_labels_and_batch_selection():
    from neurospatial.encoding import SpikeTrains

    spikes = SpikeTrains(
        [np.array([0.5, 1.5]), np.array([])], unit_ids=["active", "silent"]
    )
    result = compute_spatial_rates(
        grid(),
        spikes,
        np.array([0.0, 2.0]),
        np.array([[0.0, 0.5], [2.0, 0.5]]),
        method="binned",
        bandwidth=0,
        max_gap=None,
        time_allocation="linear",
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(result.unit_ids, ["active", "silent"])
    np.testing.assert_array_equal(result.sel(unit_id="active").firing_rate, [1.0, 1.0])
    np.testing.assert_array_equal(result.sel(unit_id="silent").firing_rate, [0.0, 0.0])


def test_linear_parallel_batch_counts_match_owned_boundary_queries():
    result = compute_spatial_rates(
        grid(),
        [np.array([0.0, 0.5, 1.5, 2.0]), np.array([1.0])],
        np.array([0.0, 2.0]),
        np.array([[0.0, 0.5], [2.0, 0.5]]),
        method="binned",
        bandwidth=0,
        max_gap=None,
        n_jobs=2,
        time_allocation="linear",
        warn_on_drop=False,
    )
    np.testing.assert_allclose(result.occupancy, [1.0, 1.0])
    np.testing.assert_allclose(result.firing_rates, [[2.0, 1.0], [0.0, 1.0]])


def test_linear_missing_endpoints_and_sampling_gap_gate_complete_intervals():
    times = np.arange(4.0)
    positions = np.array([[0.25, 0.5], [1.75, 0.5], [np.nan, np.nan], [0.25, 0.5]])
    spikes = np.array([0.25, 0.75, 1.5, 2.5])
    options = {
        "method": "binned",
        "bandwidth": 0,
        "time_allocation": "linear",
        "warn_on_drop": False,
    }
    result = compute_spatial_rate(
        grid(), spikes, times, positions, max_gap=1.0, **options
    )
    np.testing.assert_allclose(result.occupancy, [0.5, 0.5])
    np.testing.assert_allclose(result.firing_rate, [2.0, 2.0])
    assert result.analysis_settings["observed_intervals"] == [[0.0, 1.0]]
    excluded = compute_spatial_rate(
        grid(), spikes, times, positions, max_gap=0.5, **options
    )
    np.testing.assert_array_equal(excluded.occupancy, [0.0, 0.0])
    assert excluded.analysis_settings["observed_intervals"] == []


def test_linear_recording_window_gate_preserves_complete_observation_rule():
    result = compute_spatial_rate(
        grid(),
        np.array([0.5, 1.5]),
        np.array([0.0, 2.0]),
        np.array([[0.0, 0.5], [2.0, 0.5]]),
        method="binned",
        bandwidth=0,
        time_allocation="linear",
        max_gap=None,
        epochs=(0.25, 1.75),
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(result.occupancy, [0.0, 0.0])
    assert result.analysis_settings["observed_intervals"] == []


@pytest.mark.parametrize("function", [compute_spatial_rate, compute_spatial_rates])
def test_linear_spatial_encoding_refuses_unsupported_dimensions(function):
    env = Environment.from_grid_mask(
        np.ones((1, 1, 1), bool),
        grid_edges=(np.array([0.0, 1.0]),) * 3,
    )
    spikes = np.array([0.5]) if function is compute_spatial_rate else [np.array([0.5])]
    with pytest.raises(NotImplementedError, match=r"2.D|planar|dimension"):
        function(
            env,
            spikes,
            np.array([0.0, 1.0]),
            np.full((2, 3), 0.5),
            method="binned",
            bandwidth=0,
            time_allocation="linear",
            warn_on_drop=False,
        )


@pytest.mark.parametrize(
    ("function", "method", "n_jobs"),
    [
        (compute_spatial_rate, "binned", 1),
        (compute_spatial_rate, "glm", 1),
        (compute_spatial_rates, "binned", 1),
        (compute_spatial_rates, "glm", 1),
        (compute_spatial_rates, "binned", 2),
        (compute_spatial_rates, "glm", 2),
    ],
)
def test_linear_boundary_owned_event_without_dwell_refuses_rate(
    function, method, n_jobs
):
    from joblib import parallel_backend

    env = Environment.from_grid_mask(
        np.ones((2, 2), bool), grid_edges=(np.arange(3.0),) * 2
    )
    times = np.array([0.0, 2.0])
    positions = np.array([[0.5, 1.5], [1.5, 0.5]])
    np.testing.assert_array_equal(env.bin_at(np.array([[1.0, 1.0]])), [3])
    np.testing.assert_array_equal(
        env.occupancy(times, positions, max_gap=None, time_allocation="linear"),
        [0, 1, 1, 0],
    )
    spikes = np.array([1.0])
    if function is compute_spatial_rates:
        spikes = [spikes]
    options = {"bandwidth": 0} if method == "binned" else {"penalty": 1, "rank": 2}
    if function is compute_spatial_rates:
        options["n_jobs"] = n_jobs
    # Expected worker exceptions need no process teardown in this assertion.
    with (
        parallel_backend("threading"),
        pytest.raises(ValueError, match=r"boundary ownership.*zero dwell"),
    ):
        function(
            env,
            spikes,
            times,
            positions,
            method=method,
            max_gap=None,
            time_allocation="linear",
            warn_on_drop=False,
            **options,
        )
