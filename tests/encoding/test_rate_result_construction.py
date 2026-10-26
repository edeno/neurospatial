"""Construction contract shared by every rate result family.

Spatial, directional, view, object-vector and boundary-vector rate results,
singular and population, check their arrays and settings when built, own
their per-unit metadata and carry unit-label provenance the same way.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import (
    compute_boundary_vector_rates,
    compute_directional_rates,
    compute_egocentric_rates,
    compute_spatial_rates,
    compute_view_rates,
)

N_SAMPLES = 1000
DURATION = 100.0


@pytest.fixture(scope="module")
def recording() -> dict[str, Any]:
    """A seeded open-field session with three units and an object."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, DURATION, N_SAMPLES)
    positions = rng.uniform(10, 90, (N_SAMPLES, 2))
    headings = rng.uniform(-np.pi, np.pi, N_SAMPLES)
    spike_rng = np.random.default_rng(1)
    spike_times = [np.sort(spike_rng.uniform(0, DURATION, k)) for k in (100, 150, 50)]
    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=10.0, units="cm")
    return {
        "env": env,
        "times": times,
        "positions": positions,
        "headings": headings,
        "spike_times": spike_times,
        "object_positions": np.array([[50.0, 50.0]]),
    }


def _spatial(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_spatial_rates(
        rec["env"], rec["spike_times"], rec["times"], rec["positions"], **kwargs
    )


def _directional(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_directional_rates(
        rec["spike_times"], rec["times"], rec["headings"], **kwargs
    )


def _view(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_view_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        **kwargs,
    )


def _object_vector(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_egocentric_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        rec["object_positions"],
        **kwargs,
    )


def _boundary_vector(rec: dict[str, Any], **kwargs: Any) -> Any:
    return compute_boundary_vector_rates(
        rec["env"],
        rec["spike_times"],
        rec["times"],
        rec["positions"],
        rec["headings"],
        direction_frame="egocentric",
        **kwargs,
    )


FAMILIES: dict[str, Callable[..., Any]] = {
    "spatial": _spatial,
    "directional": _directional,
    "view": _view,
    "object_vector": _object_vector,
    "boundary_vector": _boundary_vector,
}


@pytest.fixture(scope="module", params=list(FAMILIES))
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def population(family: str, recording: dict[str, Any]) -> Any:
    """Each family's population result with generated unit labels."""
    return FAMILIES[family](recording)


@pytest.fixture(scope="module")
def labelled_population(family: str, recording: dict[str, Any]) -> Any:
    """Each family's population result with caller-supplied unit labels."""
    return FAMILIES[family](recording, unit_ids=["a", "b", "c"])


# ---------------------------------------------------------------------------
# Unit-label provenance
# ---------------------------------------------------------------------------


def test_generated_labels_are_marked_on_every_unit(population: Any) -> None:
    """Indexing without unit_ids marks each unit's label as generated."""
    assert population._unit_ids_generated is True
    for i, unit in enumerate(population):
        assert unit.unit_id == i
        assert unit._unit_ids_generated is True
    assert population.sel(unit_id=1)._unit_ids_generated is True
    assert dataclasses.replace(population)._unit_ids_generated is True


def test_supplied_labels_are_not_marked_generated(labelled_population: Any) -> None:
    assert labelled_population._unit_ids_generated is False
    assert [unit._unit_ids_generated for unit in labelled_population] == [False] * 3
    assert labelled_population[1].unit_id == "b"


# ---------------------------------------------------------------------------
# Estimator metadata
# ---------------------------------------------------------------------------


def test_every_family_reports_its_estimator(population: Any) -> None:
    """method and bandwidth are fields on every family and reach every export."""
    method, bandwidth = population.method, population.bandwidth
    assert method == population.analysis_settings["method"]
    assert bandwidth == population.analysis_settings["bandwidth"]
    assert (population[0].method, population[0].bandwidth) == (method, bandwidth)
    assert population.summary()["method"] == method
    assert population[0].summary()["method"] == method
    table = population.summary_table()
    assert "method" not in table.columns
    assert (table.attrs["method"], table.attrs["bandwidth"]) == (method, bandwidth)
    assert (population.to_dataframe()["method"] == method).all()


@pytest.mark.parametrize("family_name", ["object_vector", "boundary_vector"])
def test_polar_kernel_estimate_records_its_bandwidth(
    family_name: str, recording: dict[str, Any]
) -> None:
    """A smoothed polar map carries the bandwidth it was smoothed with."""
    binned = FAMILIES[family_name](recording)
    smoothed = FAMILIES[family_name](recording, method="gaussian_kde", bandwidth=8.0)
    assert (binned.method, binned.bandwidth) == ("binned", None)
    assert (smoothed.method, smoothed.bandwidth) == ("gaussian_kde", 8.0)
    assert smoothed[2].bandwidth == 8.0
