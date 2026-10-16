"""Analyses sharing the max_gap gate warn alike when times are not in seconds.

Timestamps in milliseconds make every 33 ms frame look like a 33 s recording
gap. The occupancy family and spatial rate maps must not return an empty or
mostly-empty result silently; they emit one warning that names the likely
cause, from the user's own call site.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate

_TIMES = np.arange(600) / 30.0
_POSITIONS = np.c_[50 + 40 * np.sin(_TIMES), 50 + 40 * np.cos(_TIMES / 1.3)]
_SPIKES = np.sort(np.random.default_rng(0).uniform(0, _TIMES[-1], 100))


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment.from_samples(_POSITIONS, bin_size=5.0)


CALLS = {
    "occupancy": lambda env, t, **kw: env.occupancy(t, _POSITIONS, **kw),
    "bin_sequence": lambda env, t, **kw: env.bin_sequence(t, _POSITIONS, **kw),
    "bin_sequence_with_runs": lambda env, t, **kw: env.bin_sequence_with_runs(
        t, _POSITIONS, **kw
    ),
    "transitions": lambda env, t, **kw: env.transitions(
        times=t, positions=_POSITIONS, **kw
    ),
    "compute_spatial_rate": lambda env, t, **kw: compute_spatial_rate(
        env, _SPIKES * (t[1] / _TIMES[1]), t, _POSITIONS, **kw
    ),
}


def _clock_warnings(record) -> list[warnings.WarningMessage]:
    return [w for w in record if "probably not in seconds" in str(w.message)]


def _run(call, env, times, **kwargs) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call(env, times, **kwargs)
    return list(record)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_milliseconds_warn_that_every_interval_was_excluded(env, name):
    record = _run(CALLS[name], env, _TIMES * 1000.0)
    caught = _clock_warnings(record)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert caught[0].category is UserWarning
    assert caught[0].filename == __file__
    assert "excluded ALL trajectory intervals" in message
    assert "max_gap=0.5" in message
    assert "33.33 s" in message
    assert "max_gap=None" in message


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_median_interval_above_max_gap_warns_even_when_some_survive(env, name):
    # Two of every three intervals are 0.9 s long, one is 0.1 s.
    times = np.cumsum(np.where(np.arange(600) % 3 == 0, 0.1, 0.9))
    caught = _clock_warnings(_run(CALLS[name], env, times))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert caught[0].filename == __file__
    assert "400 of 599 sampling intervals are longer than max_gap=0.5 s" in message
    assert "excluded ALL" not in message


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("kwargs", [{}, {"max_gap": None}])
def test_seconds_or_no_gap_gate_are_silent(env, name, kwargs):
    times = _TIMES if not kwargs else _TIMES * 1000.0
    record = _run(CALLS[name], env, times, **kwargs)
    assert [w for w in record if "max_gap" in str(w.message)] == []


def test_the_warning_reads_the_same_for_occupancy_and_rate_maps(env):
    times = np.cumsum(np.where(np.arange(600) % 3 == 0, 0.1, 0.9))
    occupancy = str(_clock_warnings(_run(CALLS["occupancy"], env, times))[0].message)
    rate = str(
        _clock_warnings(_run(CALLS["compute_spatial_rate"], env, times))[0].message
    )
    assert rate == f"{occupancy} Set warn_on_drop=False to suppress this warning."
