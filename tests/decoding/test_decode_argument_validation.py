"""Every decoding entry point checks dt, dtype and time_chunk the same way.

A bad value raises one ``ValueError`` naming the argument, instead of a raw
NumPy ``TypeError`` (a numeric string, an unparseable dtype), a misleading
downstream message (``dt <= 0``, ``dt=nan``) or silent acceptance
(``dt=True`` or ``time_chunk=True`` used as 1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import (
    BayesianDecoder,
    bin_spikes_in_time,
    decode_position,
    decode_position_summary,
    decode_session,
    decode_session_summary,
    log_poisson_likelihood,
    normalize_to_posterior,
)
from neurospatial.decoding._time_bins import validate_dt


@pytest.fixture(scope="module")
def session() -> dict[str, Any]:
    """A short 1-D recording with two cells, rate maps and counts."""
    env = Environment.from_samples(np.linspace(0.0, 50.0, 100)[:, None], bin_size=10.0)
    times = np.arange(0.0, 10.0, 0.05)
    positions = (25.0 + 20.0 * np.sin(times))[:, None]
    spike_rng = np.random.default_rng(1)
    spikes = [np.sort(spike_rng.uniform(0.0, 10.0, n)) for n in (40, 60)]
    models = np.random.default_rng(2).uniform(1.0, 10.0, (2, env.n_bins))
    counts = np.random.default_rng(3).poisson(1.0, (12, 2))
    decoder = BayesianDecoder.from_models(env, models, dt=0.1)
    return {
        "env": env,
        "times": times,
        "positions": positions,
        "spikes": spikes,
        "models": models,
        "counts": counts,
        "decoder": decoder,
        "log_likelihood": np.log(np.random.default_rng(4).uniform(0.1, 1.0, (5, 3))),
    }


def _dt_calls() -> dict[str, Callable[[dict[str, Any], Any], Any]]:
    return {
        "validate_dt": lambda s, dt: validate_dt(dt),
        "bin_spikes_in_time": lambda s, dt: bin_spikes_in_time(s["spikes"], dt=dt),
        "log_poisson_likelihood": lambda s, dt: log_poisson_likelihood(
            s["counts"], s["models"], dt=dt
        ),
        "decode_position": lambda s, dt: decode_position(
            s["env"], s["counts"], s["models"], dt=dt
        ),
        "decode_position_summary": lambda s, dt: decode_position_summary(
            s["env"], s["counts"], s["models"], dt=dt
        ),
        "decode_session": lambda s, dt: decode_session(
            s["env"], s["spikes"], s["times"], s["positions"], dt=dt
        ),
        "decode_session_summary": lambda s, dt: decode_session_summary(
            s["env"], s["spikes"], s["times"], s["positions"], dt=dt
        ),
        "BayesianDecoder": lambda s, dt: BayesianDecoder(s["env"], dt=dt),
    }


@pytest.mark.parametrize("entry", list(_dt_calls()))
@pytest.mark.parametrize("bad", [0, -0.025, np.nan, np.inf, "0.1", "abc", True, False])
def test_bad_dt_raises(session, entry, bad):
    with pytest.raises(ValueError, match=r"dt must be finite and positive; got"):
        _dt_calls()[entry](session, bad)


def test_validate_dt_returns_a_plain_float():
    assert type(validate_dt(np.float32(0.025))) is float
    assert validate_dt(1) == 1.0


def _dtype_calls() -> dict[str, Callable[[dict[str, Any], Any], Any]]:
    return {
        "normalize_to_posterior": lambda s, dtype: normalize_to_posterior(
            s["log_likelihood"], dtype=dtype
        ),
        "decode_position": lambda s, dtype: decode_position(
            s["env"], s["counts"], s["models"], dt=0.1, dtype=dtype
        ),
        "decode_position_summary": lambda s, dtype: decode_position_summary(
            s["env"], s["counts"], s["models"], dt=0.1, dtype=dtype
        ),
        "decode_session": lambda s, dtype: decode_session(
            s["env"], s["spikes"], s["times"], s["positions"], dt=0.1, dtype=dtype
        ),
        "decode_session_summary": lambda s, dtype: decode_session_summary(
            s["env"], s["spikes"], s["times"], s["positions"], dt=0.1, dtype=dtype
        ),
        "BayesianDecoder": lambda s, dtype: BayesianDecoder(s["env"], dtype=dtype),
    }


@pytest.mark.parametrize("entry", list(_dtype_calls()))
@pytest.mark.parametrize("bad", [np.float16, np.int32, "bogus"])
def test_bad_dtype_raises(session, entry, bad):
    with pytest.raises(ValueError, match=r"dtype must be np\.float32 or np\.float64"):
        _dtype_calls()[entry](session, bad)


def _time_chunk_calls() -> dict[str, Callable[[dict[str, Any], Any], Any]]:
    return {
        "normalize_to_posterior": lambda s, chunk: normalize_to_posterior(
            s["log_likelihood"], time_chunk=chunk
        ),
        "decode_position": lambda s, chunk: decode_position(
            s["env"], s["counts"], s["models"], dt=0.1, time_chunk=chunk
        ),
        "decode_position_summary": lambda s, chunk: decode_position_summary(
            s["env"], s["counts"], s["models"], dt=0.1, time_chunk=chunk
        ),
        "decode_session_summary": lambda s, chunk: decode_session_summary(
            s["env"], s["spikes"], s["times"], s["positions"], dt=0.1, time_chunk=chunk
        ),
        "predict_summary": lambda s, chunk: s["decoder"].predict_summary(
            s["spikes"], s["times"], time_chunk=chunk
        ),
    }


@pytest.mark.parametrize("entry", list(_time_chunk_calls()))
@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True, False])
def test_bad_time_chunk_raises(session, entry, bad):
    with pytest.raises(ValueError, match="time_chunk must be a positive integer"):
        _time_chunk_calls()[entry](session, bad)


@pytest.mark.parametrize(
    "entry", ["decode_position_summary", "decode_session_summary", "predict_summary"]
)
def test_streamed_decoders_refuse_time_chunk_none(session, entry):
    """None would hold the whole posterior, defeating a streamed decode."""
    with pytest.raises(ValueError, match="time_chunk=None is not allowed"):
        _time_chunk_calls()[entry](session, None)


@pytest.mark.parametrize("entry", ["normalize_to_posterior", "decode_position"])
@pytest.mark.parametrize("good", [None, 1, np.int64(3)])
def test_full_posterior_paths_accept_none_and_integers(session, entry, good):
    _time_chunk_calls()[entry](session, good)
