"""Rate results keep the spatial context their rates were estimated on.

A rate map is only meaningful with the bins it was estimated on: each value
belongs to one bin's place, measure, units and frame. ``rates.env`` is that
context. Environments are immutable, so relabelling an environment later
returns a new one and cannot change an existing result or its exports, and a
result refuses an environment whose bins do not match its arrays.
"""

from __future__ import annotations

import pickle

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates
from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult
from neurospatial.environment import _environment_differences


@pytest.fixture(scope="module")
def env() -> Environment:
    rng = np.random.default_rng(0)
    return Environment.from_samples(
        rng.uniform(0, 50, (500, 2)), bin_size=5.0, units="cm"
    )


@pytest.fixture(scope="module")
def rates(env: Environment) -> SpatialRatesResult:
    rng = np.random.default_rng(1)
    times = np.linspace(0.0, 60.0, 601)
    positions = rng.uniform(0, 50, (times.size, 2))
    spike_times = [np.sort(rng.uniform(0.0, 60.0, n)) for n in (40, 80)]
    return compute_spatial_rates(env, spike_times, times, positions, method="binned")


def test_relabelling_the_environment_later_cannot_relabel_the_result(
    env: Environment, rates: SpatialRatesResult
) -> None:
    env_mm = env.with_metadata(units="mm", frame="camera")
    assert env_mm.units == "mm"
    assert rates.env.units == "cm"
    assert rates.env.frame is None
    assert rates.to_dataframe().attrs["units"]["bin_center_x"] == "cm"
    assert rates[0].env is env


def test_the_estimation_environment_cannot_be_mutated(env: Environment) -> None:
    with pytest.raises(AttributeError):
        env.units = "mm"  # type: ignore[misc]
    with pytest.raises(nx.NetworkXError):
        env.connectivity.add_edge(0, 1)


def test_pickle_keeps_the_estimation_context(
    env: Environment, rates: SpatialRatesResult
) -> None:
    restored = pickle.loads(pickle.dumps(rates))
    assert restored.env.units == "cm"
    assert _environment_differences(restored.env, env) == (None, None)


def test_xarray_export_keeps_the_estimation_units(
    env: Environment, rates: SpatialRatesResult
) -> None:
    pytest.importorskip("xarray")
    env.with_metadata(units="mm", frame="camera")
    assert rates.to_xarray().attrs["units"] == "cm"


_MISMATCHES = pytest.mark.parametrize(
    ("rates_shape", "occupancy_shape"),
    [
        (lambda n: (2, 3), lambda n: (n,)),
        (lambda n: (2, n), lambda n: (4,)),
        (lambda n: (2, n), lambda n: (1, n)),
    ],
    ids=["rates-bins", "occupancy-bins", "occupancy-not-1d"],
)


@_MISMATCHES
def test_population_result_refuses_rates_that_do_not_match_env_bins(
    env: Environment, rates_shape, occupancy_shape
) -> None:
    n_bins = env.n_bins
    with pytest.raises(ValueError, match="bins") as excinfo:
        SpatialRatesResult(
            firing_rates=np.ones(rates_shape(n_bins)),
            occupancy=np.ones(occupancy_shape(n_bins)),
            env=env,
            method="binned",
            bandwidth=5.0,
        )
    assert "Fix:" in str(excinfo.value)


@_MISMATCHES
def test_single_unit_result_refuses_rates_that_do_not_match_env_bins(
    env: Environment, rates_shape, occupancy_shape
) -> None:
    n_bins = env.n_bins
    with pytest.raises(ValueError, match="bins"):
        SpatialRateResult(
            firing_rate=np.ones(rates_shape(n_bins)[1:]),
            occupancy=np.ones(occupancy_shape(n_bins)),
            env=env,
            method="binned",
            bandwidth=5.0,
        )


@pytest.mark.parametrize(
    ("result_class", "argument", "shape"),
    [
        (SpatialRatesResult, "firing_rates", lambda n: (n,)),
        (SpatialRateResult, "firing_rate", lambda n: (1, n)),
    ],
    ids=["population-1d", "single-2d"],
)
def test_rates_with_the_wrong_number_of_axes_are_refused(
    env: Environment, result_class, argument: str, shape
) -> None:
    with pytest.raises(ValueError, match="bins"):
        result_class(
            **{argument: np.ones(shape(env.n_bins))},
            occupancy=np.ones(env.n_bins),
            env=env,
            method="binned",
            bandwidth=5.0,
        )


def test_rates_built_on_their_own_env_are_accepted(env: Environment) -> None:
    result = SpatialRatesResult(
        firing_rates=np.ones((2, env.n_bins)),
        occupancy=np.ones(env.n_bins),
        env=env,
        method="binned",
        bandwidth=5.0,
    )
    assert result[1].env is env
