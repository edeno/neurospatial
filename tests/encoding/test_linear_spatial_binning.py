"""Exact fractional support shared by spatial counts and occupancy."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment
from neurospatial.encoding._spatial_counts import (
    _resolve_spatial_time_support,
    bin_spike_trains,
    compute_occupancy,
    prepare_linear_spatial_intervals,
)


def holed_cell():
    base = Environment.from_grid_mask(
        np.ones((1, 1), bool),
        grid_edges=(np.array([0.0, 6.0]), np.array([0.0, 2.0])),
    )
    return base.clip(
        Polygon(
            box(0, 0, 6, 2).exterior.coords, holes=[box(2, 0.5, 4, 1.5).exterior.coords]
        )
    )


def test_linear_context_retains_fractional_runs_and_owned_arrays():
    env = holed_cell()
    times = np.array([0.0, 8.0])
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    record = prepare_linear_spatial_intervals(env, times, positions, max_gap=None)
    np.testing.assert_array_equal(record.support.mask, [True])
    np.testing.assert_allclose(record.support.runs, [[1, 3], [5, 7]])
    np.testing.assert_allclose(record.occupancy, [4])
    times[:] = -99
    positions[:] = -99
    np.testing.assert_allclose(record.support.runs, [[1, 3], [5, 7]])
    for array in (record.support.mask, record.support.runs, record.occupancy):
        assert not array.flags.writeable


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_counts_and_occupancy_share_half_open_retained_support(n_jobs):
    env = holed_cell()
    times = np.array([0.0, 8.0])
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    spikes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], float)
    record = prepare_linear_spatial_intervals(env, times, positions, max_gap=None)
    counts, occupancy = bin_spike_trains(
        env,
        [spikes, spikes[:0]],
        times,
        positions,
        time_allocation="linear",
        linear_intervals=record,
        n_jobs=n_jobs,
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(counts, [[4], [0]])
    np.testing.assert_array_equal(occupancy, [4])
    np.testing.assert_array_equal(
        bin_spike_trains(
            env,
            [spikes],
            times,
            positions,
            time_allocation="linear",
            linear_intervals=record,
            warn_on_drop=False,
        )[0][0],
        [4],
    )


@pytest.mark.parametrize("gate", ["gap", "speed", "epochs", "spike_window", "missing"])
def test_linear_observation_gates_exclude_complete_intervals(gate):
    env = holed_cell()
    times = np.array([0.0, 8.0])
    positions = np.array([[-1.0, 1.0], [7.0, 1.0]])
    options = {"max_gap": None}
    if gate == "gap":
        options["max_gap"] = 1
    elif gate == "speed":
        options.update(speed=np.array([0.0, 0.0]), min_speed=1)
    elif gate == "missing":
        positions[-1] = np.nan
    else:
        options[gate] = (1, 7)
    record = prepare_linear_spatial_intervals(env, times, positions, **options)
    assert record.support.runs.shape == (0, 2)
    np.testing.assert_array_equal(record.occupancy, [0])
    np.testing.assert_array_equal(
        bin_spike_trains(
            env,
            [np.array([2.0, 6.0])],
            times,
            positions,
            time_allocation="linear",
            linear_intervals=record,
            warn_on_drop=False,
        )[0][0],
        [0],
    )


@pytest.mark.parametrize("function", [compute_occupancy, bin_spike_trains])
def test_binning_rejects_unknown_allocation(function):
    env = holed_cell()
    args = [env]
    if function is not compute_occupancy:
        args.append([])
    args.extend([np.array([0.0, 1.0]), np.array([[1.0, 1.0], [1.0, 1.0]])])
    with pytest.raises(ValueError, match="time_allocation"):
        function(*args, time_allocation="other")


def test_linear_warns_when_observed_path_never_enters_spatial_support():
    env = holed_cell()
    with pytest.warns(UserWarning, match="no spatial support"):
        support = _resolve_spatial_time_support(
            env,
            np.array([0.0, 1.0]),
            np.array([[-2.0, 1.0], [-1.0, 1.0]]),
            speed=None,
            min_speed=None,
            max_gap=None,
            epochs=None,
            spike_window=None,
            time_allocation="linear",
            warn_on_drop=True,
        )
    np.testing.assert_array_equal(support.mask, [True])
    assert support.runs.shape == (0, 2)


def test_native_grid_entry_and_exit_use_original_interval_fractions():
    env = Environment.from_grid_mask(
        np.ones((2, 1), bool),
        grid_edges=(np.arange(3.0), np.array([0.0, 1.0])),
    )
    times = np.array([0.0, 4.0])
    positions = np.array([[-1.0, 0.5], [3.0, 0.5]])
    counts, dwell = bin_spike_trains(
        env,
        [np.arange(0.5, 4, 1.0)],
        times,
        positions,
        time_allocation="linear",
        max_gap=None,
        warn_on_drop=False,
    )
    np.testing.assert_array_equal(counts, [[1, 1]])
    np.testing.assert_array_equal(dwell, [1, 1])
    np.testing.assert_array_equal(
        env.occupancy(times, positions, max_gap=None, time_allocation="linear"), dwell
    )


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_boundary_owned_spike_without_dwell_refuses_undefined_exposure(n_jobs):
    from joblib import parallel_backend

    env = Environment.from_grid_mask(
        np.ones((2, 2), bool), grid_edges=(np.arange(3.0),) * 2
    )
    times = np.array([0.0, 2.0])
    positions = np.array([[0.5, 1.5], [1.5, 0.5]])
    record = prepare_linear_spatial_intervals(env, times, positions, max_gap=None)
    np.testing.assert_array_equal(record.occupancy, [0, 1, 1, 0])
    np.testing.assert_array_equal(env.bin_at(np.array([[1.0, 1.0]])), [3])
    with pytest.raises(ValueError, match=r"boundary ownership.*zero dwell"):
        if n_jobs is None:
            bin_spike_trains(
                env,
                [np.array([1.0])],
                times,
                positions,
                time_allocation="linear",
                linear_intervals=record,
                warn_on_drop=False,
            )[0][0]
        else:
            with parallel_backend("threading"):
                bin_spike_trains(
                    env,
                    [np.array([1.0])],
                    times,
                    positions,
                    time_allocation="linear",
                    linear_intervals=record,
                    n_jobs=n_jobs,
                    warn_on_drop=False,
                )


def test_linear_fractional_support_refuses_unresolvable_absolute_clock():
    parent = Environment.from_grid_mask(
        np.ones((1, 1), bool), grid_edges=(np.array([0.0, 1.0]),) * 2
    )
    env = parent.clip(box(0.5, 0, 0.50001, 1))
    positions = np.array([[0.0, 0.5], [1.0, 0.5]])
    with pytest.raises(ValueError, match="elapsed timestamps"):
        prepare_linear_spatial_intervals(
            env, np.array([1e12, 1e12 + 1]), positions, max_gap=None
        )
    elapsed = np.array([0.0, 1.0])
    record = prepare_linear_spatial_intervals(env, elapsed, positions, max_gap=None)
    np.testing.assert_allclose(record.occupancy, [1e-5])
    np.testing.assert_array_equal(record.support.runs, [[0.5, 0.50001]])
    np.testing.assert_array_equal(
        bin_spike_trains(
            env,
            [np.array([0.5])],
            elapsed,
            positions,
            time_allocation="linear",
            linear_intervals=record,
            warn_on_drop=False,
        )[0][0],
        [1],
    )


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_encoding_dwell_matches_environment_linear_occupancy(layout):
    # Rate maps divide by the dwell time Environment.occupancy reports, so the
    # two must allocate each straight interval identically.
    rng = np.random.default_rng(3)
    env = Environment.from_samples(
        rng.uniform(0, 30, (2000, 2)), bin_size=3.0, layout=layout
    )
    times = np.cumsum(rng.uniform(0.01, 0.05, 600))
    positions = np.clip(15 + np.cumsum(rng.normal(0, 0.8, (600, 2)), axis=0), -2, 32)
    positions[[50, 51, 300]] = np.nan
    speed = np.abs(rng.normal(5, 3, 600))
    record = prepare_linear_spatial_intervals(
        env, times, positions, speed=speed, min_speed=4.0
    )
    occupancy = env.occupancy(
        times, positions, speed=speed, min_speed=4.0, time_allocation="linear"
    )
    assert occupancy.sum() > 0
    np.testing.assert_array_equal(record.occupancy, occupancy)
